"""Voice server: FastAPI on :8300.

GET /        -> voice/static/index.html
GET /health  -> model/agent/session status
WS  /ws?k=   -> SPEC.md "Voice WebSocket" protocol

Run:  python -m voice.server [--agent auto|fake|real] [--port 8300] [--no-warm]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

from agent.memory import Conversation, SessionStore, session_key
from voice.agent_bridge import AgentRunner
from voice.tts import SAMPLE_RATE as TTS_RATE, split_sentences

log = logging.getLogger("voice.server")
STATIC = Path(__file__).parent / "static"
MAX_AUDIO_BYTES = 16000 * 2 * 30          # 30 s of 16 kHz PCM16
HISTORY_TURNS = 3
def memory_dir_default() -> str | None:
    """$VOICE_MEMORY_DIR (empty = memory off), else $VOICE_HOME/sessions."""
    from agent import paths
    return os.environ.get("VOICE_MEMORY_DIR", str(paths.SESSIONS_DIR)) or None
TURN_LOG = os.environ.get("VOICE_TURN_LOG")   # optional JSONL path: one record per agent run (question, model, events, timings)
_DONE = object()


def _ms(dt: float | None) -> int | None:
    return None if dt is None else int(round(dt * 1000))


class VoiceState:
    def __init__(self, stt, tts, runner: AgentRunner, passcode: str, max_turns_per_min: int,
                 memory_dir: str | None = None):
        self.stt, self.tts, self.runner = stt, tts, runner
        self.passcode = passcode
        # conversation memory survives reconnects: one store file per passcode (hashed), see docs/memory.md
        self.memory = SessionStore(memory_dir) if memory_dir else None
        self.max_turns = max_turns_per_min
        self.turn_times: collections.deque[float] = collections.deque()
        # MLX speech work (Whisper + Kokoro) is serialised on one thread; the agent gets its own,
        # so the filler can be synthesised while the agent is still deciding.
        self.speech_pool = ThreadPoolExecutor(1, thread_name_prefix="speech")
        self.agent_pool = ThreadPoolExecutor(1, thread_name_prefix="agent")
        self.session: Session | None = None
        self.session_lock = asyncio.Lock()
        self.vocab = runner.vocab()
        self.warm_status: dict[str, Any] = {}
        self.default_model = runner.default_model()

    def allow_turn(self) -> bool:
        now = time.monotonic()
        while self.turn_times and now - self.turn_times[0] > 60:
            self.turn_times.popleft()
        if len(self.turn_times) >= self.max_turns:
            return False
        self.turn_times.append(now)
        return True


def log_turn(question: str, model: str | None, lane: int | None, events: list[dict], timings: dict):
    if not TURN_LOG:
        return
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "question": question, "model": model, "lane": lane,
           "events": events, "timings": {k: v for k, v in timings.items() if k != "lanes"}}
    try:
        with open(TURN_LOG, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        log.exception("turn log write failed")


class Session:
    def __init__(self, ws: WebSocket, state: VoiceState):
        self.ws, self.state = ws, state
        self.buf = bytearray()
        self.model = state.default_model
        self.compare: list[str] | None = None        # two model names: compare mode on
        self.lane_spoken: dict[int, str] = {}        # last compare turn's spoken answers, for play_lane
        self.conv = Conversation(state.memory, session_key(state.passcode))
        self.turn_task: asyncio.Task | None = None
        self.send_lock = asyncio.Lock()
        self.seq = 0
        self.closed = False

    @property
    def history(self) -> list[Any]:
        return self.conv.history

    # -- sending -------------------------------------------------------------
    async def send(self, event: dict):
        if self.closed:
            return
        async with self.send_lock:
            await self.ws.send_text(json.dumps(event, default=str))

    async def send_audio(self, pcm: bytes, text: str):
        if self.closed:
            return
        head = {"type": "audio", "sample_rate": TTS_RATE, "seq": self.seq, "text": text}
        async with self.send_lock:  # header + frame must stay adjacent
            await self.ws.send_text(json.dumps(head))
            await self.ws.send_bytes(pcm)
        self.seq += 1

    async def close(self, reason: str):
        try:
            await self.send({"type": "error", "message": reason})
            await self.ws.close(code=4000, reason=reason)
        except Exception:
            pass
        self.closed = True

    # -- receive loop --------------------------------------------------------
    async def run(self):
        await self.send({"type": "hello", "agent": self.state.runner.mode, "model": self.model,
                         "models": self.state.runner.models(), "compare": self.compare,
                         "memory": {"turns": len(self.conv.turns), "facts": len(self.conv.facts)},
                         "sample_rate_in": 16000, "sample_rate_out": TTS_RATE})
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if len(self.buf) < MAX_AUDIO_BYTES:
                        self.buf.extend(msg["bytes"])
                    continue
                if msg.get("text") is None:
                    continue
                try:
                    data = json.loads(msg["text"])
                except ValueError:
                    await self.send({"type": "error", "message": "bad json"})
                    continue
                await self.handle(data)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            self.closed = True
            self.conv.closed = True       # a turn still running on the agent thread is not recorded
            if self.turn_task and not self.turn_task.done():
                self.turn_task.cancel()

    async def handle(self, data: dict):
        kind = data.get("type")
        if kind == "start":                      # optional: client says a new utterance begins
            self.buf.clear()
        elif kind == "end":
            audio = bytes(self.buf)
            self.buf.clear()
            compare = await self.compare_models(data.get("compare"))
            self.start_turn(audio=audio, compare=compare)
        elif kind == "text":
            text = (data.get("text") or "").strip()
            if text:
                compare = await self.compare_models(data.get("compare"))
                self.start_turn(text=text, compare=compare)
        elif kind == "set_model":
            model = data.get("model")
            name = await self.resolve_model(model)
            if name is not None:
                self.model = name
                await self.send({"type": "model", "model": model, "resolved": name})
        elif kind == "set_compare":
            models = data.get("models")
            if not models:
                self.compare = None
                await self.send({"type": "compare", "models": None})
            else:
                names = await self.compare_models(models, strict=True)
                if names:
                    self.compare = names
                    await self.send({"type": "compare", "models": names})
        elif kind == "play_lane":
            try:
                text = self.lane_spoken.get(int(data.get("lane") or 0))
            except (TypeError, ValueError):
                text = None
            if text:
                self.start_turn(speak_only=text, replay=True)
            else:
                await self.send({"type": "error", "message": "nothing to play for that lane"})
        elif kind == "clear_memory":
            if self.turn_task and not self.turn_task.done():
                await self.send({"type": "error", "message": "busy: still answering the last question"})
            else:
                self.conv.clear("all")
                await self.send({"type": "memory_cleared", "scope": "all"})
        elif kind == "confirm_ticket":
            tid = data.get("ticket_id")
            await self.send({"type": "ticket_confirmed", "ticket_id": tid,
                             "status": "confirmed (paper, mock market: nothing is sent to a broker)"})
            self.start_turn(speak_only="Confirmed.")
        else:
            await self.send({"type": "error", "message": f"unknown message type {kind!r}"})

    async def resolve_model(self, model) -> str | None:
        try:
            return self.state.runner.resolve(str(model))
        except KeyError:
            await self.send({"type": "error", "message": f"unknown model {model!r}"})
        except RuntimeError as e:
            await self.send({"type": "error", "message": str(e)})
        return None

    async def compare_models(self, models=None, strict: bool = False) -> list[str] | None:
        """Two model names for a compare turn: `models` from the message, else the session's compare setting."""
        if models is None and not strict:
            return self.compare
        if not isinstance(models, list) or len(models) != 2:
            await self.send({"type": "error", "message": "compare needs exactly two models"})
            return None
        names = [await self.resolve_model(m) for m in models]
        return names if all(names) else None

    def start_turn(self, **kw):
        if self.turn_task and not self.turn_task.done():
            asyncio.create_task(self.send({"type": "error", "message": "busy: still answering the last question"}))
            return
        if "speak_only" not in kw and not self.state.allow_turn():
            asyncio.create_task(self.send({"type": "error", "message": "rate limit: too many questions this minute"}))
            return
        self.turn_task = asyncio.create_task(self.turn(**kw))

    # -- one turn ------------------------------------------------------------
    async def turn(self, audio: bytes | None = None, text: str | None = None, speak_only: str | None = None,
                   compare: list[str] | None = None, replay: bool = False):
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        timings: dict[str, Any] = {"stt": 0, "first_audio": None, "total": None}
        if replay:
            timings["replay"] = True
        try:
            if speak_only is not None:
                await self.speak_all(split_sentences(speak_only) if replay else [speak_only], t0, timings)
                return
            if audio is not None:
                transcript = await loop.run_in_executor(self.state.speech_pool, self.state.stt.transcribe,
                                                        audio, self.state.vocab)
                timings["stt"] = _ms(time.perf_counter() - t0)
                timings["audio_seconds"] = round(len(audio) / 32000, 2)
                await self.send({"type": "transcript", "text": transcript, "source": "audio"})
                if not transcript:
                    await self.send({"type": "error", "message": "didn't catch that"})
                    return
            else:
                transcript = text
                await self.send({"type": "transcript", "text": transcript, "source": "text"})
            if compare:
                await self.run_compare(transcript, t0, timings, compare)
            else:
                await self.run_agent(transcript, t0, timings)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("turn failed")
            await self.send({"type": "error", "message": f"server error: {e}"})
        finally:
            timings["total"] = _ms(time.perf_counter() - t0)
            try:
                await self.send({"type": "done", "timings": timings})
            except Exception:
                pass

    async def run_compare(self, question: str, t0: float, timings: dict, models: list[str]):
        """The same question through two models in turn. Every agent event carries `lane` (0/1) and `model`.
        Lane 0 speaks as usual; lane 1 is silent (its spoken answer is kept for `play_lane`). Lane 1 starts
        after lane 0's speech is synthesised, so the two never share the GPU. Only lane 0 enters history."""
        await self.send({"type": "compare_start", "models": models})
        self.lane_spoken = {}
        history = self.history                             # both lanes see the history from BEFORE this turn
        lanes = []
        for lane, model in enumerate(models):
            lt: dict[str, Any] = {}
            await self.run_agent(question, t0, timings if lane == 0 else lt, model=model, lane=lane,
                                 speak=lane == 0, record=lane == 0, history=history)
            src = timings if lane == 0 else lt
            info = {"lane": lane, "model": model, "agent": src.get("agent"), "first_tool": src.get("first_tool"),
                    "first_audio": src.get("first_audio") if lane == 0 else None, "decides": src.get("decides")}
            lanes.append(info)
            await self.send({"type": "lane_done", **info})
        timings["lanes"] = lanes

    async def run_agent(self, question: str, t0: float, timings: dict, model: str | None = None,
                        lane: int | None = None, speak: bool = True, record: bool = True,
                        history: list | None = None):
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        speech: asyncio.Queue = asyncio.Queue()
        flags = {"answer": False}

        def on_event(e: dict):
            loop.call_soon_threadsafe(events.put_nowait, e)

        model = model or self.model       # history None = the conversation's own (agent.memory.Conversation)

        runner = self.state.runner

        def work():
            try:
                return self.conv.ask(question, lambda q, h, f: runner.run(q, h, model, on_event, facts=f),
                                     on_event, world=runner.world(), record=record, history=history)
            except Exception as e:  # noqa: BLE001
                log.exception("agent failed")
                on_event({"type": "error", "message": f"agent error: {e}"})
                return None
            finally:
                on_event(_DONE)

        t_agent = time.perf_counter()
        fut = loop.run_in_executor(self.state.agent_pool, work)
        speaker = asyncio.create_task(self.speaker(speech, t0, timings, flags))
        filler_done = False
        seen: list[dict] = []
        try:
            while True:
                e = await events.get()
                if e is _DONE:
                    break
                kind = e.get("type") if isinstance(e, dict) else None
                if isinstance(e, dict):
                    seen.append(e)
                if lane is not None and isinstance(e, dict):
                    e = dict(e, lane=lane, model=model)
                await self.send(e)
                if kind == "tool_call" and timings.get("first_tool") is None:
                    timings["first_tool"] = _ms(time.perf_counter() - t_agent)
                say = None
                if kind == "filler" and not filler_done and e.get("text"):
                    filler_done = True
                    say = ("filler", e["text"])
                elif kind == "answer":
                    flags["answer"] = True
                    say = ("answer", e.get("spoken") or e.get("display") or "")
                elif kind == "error" and not flags["answer"] and not e.get("recoverable"):
                    flags["answer"] = True
                    say = ("answer", "Sorry, I couldn't get that.")
                if say and say[0] == "answer" and lane is not None:
                    self.lane_spoken[lane] = say[1]
                if say and speak:
                    speech.put_nowait(say)
            result = await fut
            timings["agent"] = _ms(time.perf_counter() - t_agent)
            decides = (getattr(result, "timings", None) or {}).get("decide") if result is not None else None
            if isinstance(decides, list):
                timings["decides"] = [_ms(d) for d in decides]
            log_turn(question, model, lane, seen, timings)
        finally:
            speech.put_nowait(None)
            await speaker

    async def speaker(self, speech: asyncio.Queue, t0: float, timings: dict, flags: dict):
        while True:
            item = await speech.get()
            if item is None:
                return
            kind, text = item
            if kind == "filler" and flags["answer"]:
                continue  # answer already here: the filler would only delay it
            await self.speak_all(split_sentences(text), t0, timings)

    async def speak_all(self, sentences: list[str], t0: float, timings: dict):
        loop = asyncio.get_running_loop()
        for s in sentences:
            if self.closed:
                return
            try:
                pcm = await loop.run_in_executor(self.state.speech_pool, self.state.tts.synth, s)
            except Exception as e:  # noqa: BLE001
                log.exception("tts failed")
                await self.send({"type": "error", "message": f"tts error: {e}"})
                return
            if not pcm:
                continue
            if timings.get("first_audio") is None:
                timings["first_audio"] = _ms(time.perf_counter() - t0)
            await self.send_audio(pcm, s)


def create_app(stt=None, tts=None, agent_mode: str = "auto", passcode: str | None = None,
               warm: bool = True, max_turns_per_min: int | None = None, memory_dir: str | None = "") -> FastAPI:
    """memory_dir: "" = memory_dir_default(), None = no persistence."""
    if memory_dir == "":
        memory_dir = memory_dir_default()
    if stt is None:
        from voice.stt import STT
        stt = STT()
    if tts is None:
        from voice.tts import TTS
        tts = TTS()
    state = VoiceState(stt, tts, AgentRunner(agent_mode),
                       passcode if passcode is not None else os.environ.get("VOICE_PASSCODE", "demo"),
                       max_turns_per_min or int(os.environ.get("VOICE_MAX_TURNS_PER_MIN", "6")), memory_dir)

    def _warm():
        for name, obj in (("stt", stt), ("tts", tts)):
            try:
                state.warm_status[name] = f"ok {obj.warm():.1f}s"
            except Exception as e:  # noqa: BLE001
                log.exception("warm %s failed", name)
                state.warm_status[name] = f"failed: {e}"
            log.info("warm %s: %s", name, state.warm_status[name])

    def _warm_agent():
        # Runs on the agent thread: MLX streams are per-thread, so load + warm where decide() will run.
        try:
            state.warm_status["agent"] = f"ok {state.runner.warm(state.default_model):.1f}s"
        except Exception as e:  # noqa: BLE001
            log.exception("warm agent failed")
            state.warm_status["agent"] = f"failed: {e}"
        log.info("warm agent: %s", state.warm_status["agent"])
        if os.environ.get("VOICE_WARM_ALL", "1") != "0":   # the rest of the picker, so switching is instant
            state.warm_status["models"] = state.runner.warm_all(state.default_model)
            log.info("warm models: %s", state.warm_status["models"])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warm:
            asyncio.get_running_loop().run_in_executor(state.speech_pool, _warm)
            if state.runner.mode == "real":
                asyncio.get_running_loop().run_in_executor(state.agent_pool, _warm_agent)
        yield
        state.speech_pool.shutdown(wait=False, cancel_futures=True)
        state.agent_pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="leaps voice", lifespan=lifespan)
    app.state.voice = state

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True, "agent": state.runner.mode,
            "stt_loaded": bool(getattr(stt, "loaded", False)),
            "tts_loaded": bool(getattr(tts, "loaded", False)),
            "warm": state.warm_status,
            "session_active": state.session is not None and not state.session.closed,
            "default_model": state.default_model, "models": state.runner.models(),
        })

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket, k: str = ""):
        await websocket.accept()
        if not secrets.compare_digest(k.encode(), state.passcode.encode()):
            await websocket.send_text(json.dumps({"type": "error", "message": "bad passcode"}))
            await websocket.close(code=4401, reason="bad passcode")
            return
        session = Session(websocket, state)
        async with state.session_lock:
            old = state.session
            state.session = session
        if old is not None and not old.closed:  # one session at a time: newest wins
            await old.close("replaced by a newer session")
        try:
            await session.run()
        finally:
            if state.session is session:
                state.session = None
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.close()
                except Exception:
                    pass

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("VOICE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--agent", choices=["auto", "fake", "real"], default="auto")
    ap.add_argument("--no-warm", action="store_true", help="load STT/TTS lazily on first use")
    ap.add_argument("--ssl-certfile")
    ap.add_argument("--ssl-keyfile")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn
    app = create_app(agent_mode=a.agent, warm=not a.no_warm)
    uvicorn.run(app, host=a.host, port=a.port, ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile,
                ws_max_size=4 * 1024 * 1024)


if __name__ == "__main__":
    main()
