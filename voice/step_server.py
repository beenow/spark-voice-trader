"""Step server for the on-device Android agent (android/): the phone runs ONLY the model; this runs the rest.

The phone owns generation (Llama 3.2 1B GGUF via llama.cpp). Everything else is the existing Python path,
unchanged and imported: agent.memory.Conversation -> agent.loop.run_turn (repairs, validators, slot fill,
memory, tool execution on mockmarket.World), with prompts rendered by agent.format (render_llama /
LlamaPromptBuilder), so the phone sees byte-identical prompts to the evaluated MLX path.

How: each turn runs run_turn in a worker thread with a RemoteBackend. Its decide() renders the prompt, hands it
to the phone and blocks until the phone posts the model's output back.

    POST /step   header X-Passcode: <passcode>
      {"session": s, "question": q}                -> starts a turn
      {"session": s, "model_output": text, "capped": bool, "stats": {...}}   -> continues it
      {"session": s, "reset": true}                -> forgets the conversation
    reply: {"prompt": text, "tokens": [ids], "max_tokens", "answer_max_tokens", "events": [...]}
       or  {"say": spoken, "display": text, "events": [...], "timings": {...}}
    GET /model/manifest.json, GET /model/<name>   (passcode; Range supported, so a download resumes)
       the GGUF (kind "llm") and the Kokoro-82M ONNX voice bundle (kind "tts": model, voice, tokenizer)
    GET /health

<home> is $VOICE_HOME via agent.paths when present, else ~/.leaps-voice-agent.
Run (voice venv). Binds 127.0.0.1 unless told otherwise; the owner picks the exposure (a LAN or Tailscale
address via --host; never a public tunnel). Over USB, `adb reverse tcp:8310 tcp:8310` also works.
    python -m voice.step_server --host <ip>
Model files: $STEP_MODEL_DIR (default <home>/gguf), names in $STEP_MODEL_FILES
(comma-separated, default llama-1b-loop-Q4_K_M.gguf). Only those names are ever served.
Passcode: read from $VOICE_PASSCODE_FILE (default <home>/.passcode), never logged.
Turn log: one JSON line per turn (question, each step's model output, tool calls, what was said, fallback,
outcome, timings, and which GGUF served it) appended to $STEP_TURN_LOG (default <home>/logs/step_turns.jsonl;
set it empty to turn it off). Local disk only. Without it a bad phone answer leaves nothing to diagnose from.
Provenance: a `<gguf>.provenance.json` sidecar (adapter path/sha, run id, iter, build command) beside a served
model is passed through in the manifest and logged at startup, so "which model is on the phone" is answerable.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import queue
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.backends import parse_llama, registry
from agent.format import LlamaPromptBuilder, render_llama
from agent.loop import run_turn
from agent.memory import Conversation

log = logging.getLogger("voice.step")

PORT = 8310
MAX_TOKENS = 160          # agent.backends.LlamaBackend defaults (the phone applies the same stop rules)
ANSWER_MAX_TOKENS = 70
PHONE_TIMEOUT_S = 120.0
SESSION_TTL_S = 3600.0
_DONE = object()
TURN_LOG_OUTPUT_CHARS = 2000   # per-step model output kept in the turn log (a step is <= MAX_TOKENS tokens)


def _home() -> Path:
    try:
        from agent import paths
        return Path(paths.HOME)
    except ImportError:
        return Path.home() / ".leaps-voice-agent"


def read_passcode() -> str:
    path = Path(os.path.expanduser(os.environ.get("VOICE_PASSCODE_FILE", str(_home() / ".passcode"))))
    code = path.read_text().strip()
    if not code:
        raise RuntimeError(f"empty passcode file {path}")
    return code


def model_files() -> dict[str, Path]:
    d = Path(os.path.expanduser(os.environ.get("STEP_MODEL_DIR", str(_home() / "gguf"))))
    names = [n.strip() for n in os.environ.get("STEP_MODEL_FILES", "llama-1b-loop-Q4_K_M.gguf").split(",") if n.strip()]
    return {n: d / n for n in names if n.endswith(".gguf") and "/" not in n and (d / n).is_file()}


# Kokoro-82M ONNX voice bundle for the phone's on-device TTS (Apache-2.0 weights): served name -> file, role.
KOKORO_FILES = {"kokoro-model.onnx": ("model.onnx", "model"), "kokoro-voice-af_heart.bin": ("voice.bin", "voice"),
                "kokoro-tokenizer.json": ("tokenizer.json", "tokenizer")}
KOKORO_VOICE, KOKORO_SPEED = "af_heart", 1.05     # the Mac web app's voice and speed (voice/tts.py)


def turn_log_path() -> Path | None:
    v = os.environ.get("STEP_TURN_LOG")
    if v is None:
        return _home() / "logs" / "step_turns.jsonl"
    return Path(os.path.expanduser(v)) if v.strip() else None


def read_provenance(gguf: Path) -> dict | None:
    """`<gguf>.provenance.json` written by the GGUF build (adapter, run id, iter); None if absent or unreadable."""
    side = gguf.with_name(gguf.name + ".provenance.json")
    try:
        v = json.loads(side.read_text())
        return v if isinstance(v, dict) else None
    except (OSError, ValueError):
        return None


class TurnLog:
    """Appends one JSON line per turn. A write failure is logged, never raised into the turn."""

    def __init__(self, path: Path | None, served: dict):
        self.path, self.served = path, served
        self.lock = threading.Lock()

    def write(self, rec: dict) -> None:
        if self.path is None:
            return
        line = json.dumps({"ts": dt.datetime.now().astimezone().isoformat(timespec="seconds"), **rec,
                           "served": self.served}, default=str)
        try:
            with self.lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError:
            log.exception("turn log write failed: %s", self.path)


def kokoro_files() -> dict[str, tuple[Path, str]]:
    """$STEP_KOKORO_DIR (default <home>/kokoro-onnx): model.onnx, voice.bin, tokenizer.json; all three or none."""
    d = Path(os.path.expanduser(os.environ.get("STEP_KOKORO_DIR", str(_home() / "kokoro-onnx"))))
    out = {n: (d / f, role) for n, (f, role) in KOKORO_FILES.items()}
    return out if all(p.is_file() for p, _ in out.values()) else {}


def sha256_of(path: Path) -> str:
    """sha256 of a model file, cached in a sidecar keyed by size + mtime (an 800 MB hash takes seconds)."""
    import hashlib
    st = path.stat()
    side = path.with_name(path.name + ".sha256")
    key = f"{st.st_size}:{int(st.st_mtime)}"
    try:
        k, h = side.read_text().split()
        if k == key:
            return h
    except (OSError, ValueError):
        pass
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    try:
        side.write_text(f"{key} {digest}\n")
    except OSError:
        pass
    return digest


def trim_two_sentences(answer: str) -> str:
    """LlamaBackend's cap: keep the first two sentences of an answer stopped by the sentence rule."""
    m = re.match(r"^(.*?[.!?])(?:\s+(.*?[.!?]))?(?=\s|$)", answer, re.S)
    return (m.group(1) + (" " + m.group(2) if m.group(2) else "")).strip() if m else answer


class PhoneTimeout(Exception):
    pass


class RemoteBackend:
    """decide() protocol of agent.backends, with generation done on the phone."""

    def __init__(self, pb: LlamaPromptBuilder, outbox: queue.Queue, inbox: queue.Queue, frame_line: bool):
        self.pb, self.outbox, self.inbox = pb, outbox, inbox
        self.frame_line = frame_line
        self.stats: list[dict] = []
        self.steps: list[dict] = []     # what the phone generated at each step, for the turn log
        self.abandoned: str | None = None  # set once the phone goes quiet or drops the turn; later steps fail fast

    def decide(self, messages: list[dict], tools_schema=None) -> dict:
        if self.abandoned:      # run_turn retries a failed decide; nobody is listening any more
            raise PhoneTimeout(self.abandoned)
        tokens = self.pb.tokens_for(messages, tools_schema)
        text = render_llama(messages, tools_schema, add_generation_prompt=True)
        self.outbox.put({"prompt": text, "tokens": tokens})
        try:
            reply = self.inbox.get(timeout=PHONE_TIMEOUT_S)
        except queue.Empty:
            self.abandoned = "phone did not answer"
            raise PhoneTimeout(self.abandoned)
        if reply is _DONE:
            self.abandoned = "turn dropped (new question or reset)"
            raise PhoneTimeout(self.abandoned)
        raw = str(reply.get("model_output") or "")
        for t in ("<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"):
            raw = raw.replace(t, "")
        out = parse_llama(raw)
        if reply.get("capped") and "answer" in out:
            out["answer"] = trim_two_sentences(out["answer"])
        st = dict(reply.get("stats") or {}, prompt_tokens=len(tokens),
                  kind="tool_calls" if "tool_calls" in out else ("answer" if "answer" in out else "error"))
        self.stats.append(st)
        self.steps.append({"step": len(self.stats), "prompt_tokens": len(tokens), "kind": st["kind"],
                           "capped": bool(reply.get("capped")), "model_output": raw[:TURN_LOG_OUTPUT_CHARS],
                           "stats": reply.get("stats") or {}})
        out.setdefault("raw", raw)
        return out


class Session:
    def __init__(self, world, frame_line: bool, pb: LlamaPromptBuilder, sid: str = "",
                 turn_log: TurnLog | None = None):
        self.conv = Conversation()
        self.sid, self.turn_log = sid, turn_log
        self.world, self.frame_line, self.pb = world, frame_line, pb
        self.outbox: queue.Queue = queue.Queue()
        self.inbox: queue.Queue = queue.Queue()
        self.events: list[dict] = []
        self.thread: threading.Thread | None = None
        self.backend: RemoteBackend | None = None
        self.lock = threading.Lock()
        self.touched = time.time()

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, question: str) -> None:
        self.outbox, self.inbox, self.events = queue.Queue(), queue.Queue(), []
        self.turn_events: list[dict] = []          # this turn's events so far, for the screen to show (nav)
        self.ticket_mark = len(self.world.tickets)
        self.question = question
        self.backend = RemoteBackend(self.pb, self.outbox, self.inbox, self.frame_line)
        be, world = self.backend, self.world

        def on_event(e: dict) -> None:
            self.events.append(e)

        def work():
            t0 = time.perf_counter()
            rec: dict = {"session": self.sid, "question": question}
            reply: dict | None = None
            try:
                res = self.conv.ask(question, lambda q, h, f: run_turn(q, h, world, be, on_event=on_event, facts=f,
                                                                       frame_line=self.frame_line),
                                    on_event, world=world)
                total = time.perf_counter() - t0
                rec.update(outcome="abandoned" if be.abandoned else "answered", say=res.spoken, display=res.display,
                           template=res.answer_template, tool_calls=_plain(res.tool_calls), fallback=bool(res.fallback))
                reply = {"result": res, "total_s": total}
            except PhoneTimeout as e:
                # the phone went quiet, or abandoned the turn (a new question or a reset arrived mid-turn)
                rec.update(outcome="abandoned", error=str(e))
                reply = {"error": f"PhoneTimeout: {e}"}
            except Exception as e:  # noqa: BLE001
                log.exception("turn failed")
                rec.update(outcome="error", error=f"{type(e).__name__}: {e}")
                reply = {"error": rec["error"]}
            finally:
                # log BEFORE the reply is released: once it is in the outbox the phone (or a test) may act on it
                if self.turn_log is not None:
                    try:
                        if be.abandoned:
                            rec.setdefault("error", f"PhoneTimeout: {be.abandoned}")
                        rec.update(steps=be.steps, n_steps=len(be.steps),
                                   mac_total_s=round(time.perf_counter() - t0, 3))
                        self.turn_log.write(rec)
                    except Exception:  # noqa: BLE001 -- the log must never cost the phone its answer
                        log.exception("turn log failed")
                if reply is not None:
                    be.outbox.put(reply)   # this turn's outbox, even if a newer turn has replaced self.outbox

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def next_reply(self) -> dict:
        item = self.outbox.get(timeout=PHONE_TIMEOUT_S + 30)
        events, self.events = self.events, []
        events = [_plain(e) for e in events]
        self.turn_events = getattr(self, "turn_events", []) + events
        nav = self.nav()
        seq = self.nav_seq(item["result"].spoken if "result" in item else "")
        if "prompt" in item:
            hint, self.last_hint = getattr(self, "last_hint", None), None
            if hint and len(self.backend.stats) == 0:      # first step of a turn: how much of the hint held
                toks = item["tokens"]
                i = next((k for k, (a, b) in enumerate(zip(hint, toks)) if a != b), min(len(hint), len(toks)))
                if i < len(hint) - 1:
                    tok = self.pb.tok
                    log.info("next_prefix miss at %d/%d: hint %r | prompt %r", i, len(hint),
                             tok.decode(hint[max(0, i - 8):i + 24]), tok.decode(toks[max(0, i - 8):i + 24]))
                else:
                    log.info("next_prefix hit: %d/%d tokens", i, len(toks))
            return {"prompt": item["prompt"], "tokens": item["tokens"], "max_tokens": MAX_TOKENS,
                    "answer_max_tokens": ANSWER_MAX_TOKENS, "step": len(self.backend.stats) + 1, "events": events,
                    **({"nav": nav} if nav else {}), **({"nav_seq": seq} if seq else {})}
        if "error" in item:
            return {"say": "Sorry, I couldn't get that.", "display": "", "error": item["error"], "events": events}
        res = item["result"]
        tm = getattr(res, "timings", {}) or {}
        return {"say": res.spoken, "display": res.display, "template": res.answer_template,
                "tool_calls": _plain(res.tool_calls), "fallback": bool(res.fallback), "events": events,
                "timings": {"mac_total_s": round(item["total_s"], 3),
                            "tools_s": [round(x, 4) for x in tm.get("tools") or []],
                            "decides": self.backend.stats if self.backend else []},
                "next_prefix": self._hint(), **({"nav": nav} if nav else {}), **({"nav_seq": seq} if seq else {})}

    def nav(self) -> dict | None:
        """The screen for this turn so far (voice.ui_api.nav_for), from the executed tool calls: sent with the
        reply that follows a tool step, so the phone shows it while the answer is generated and spoken."""
        from voice.ui_api import nav_for
        ok = {e.get("tool") for e in self.turn_events if e.get("type") == "tool_result" and e.get("ok")}
        calls = [{"tool": e["tool"], "args": e.get("args") or {}} for e in self.turn_events
                 if e.get("type") == "tool_call" and e.get("tool") in ok]
        new = list(self.world.tickets)[getattr(self, "ticket_mark", len(self.world.tickets)):]
        matches = [m.get("symbol") for e in self.turn_events
                   if e.get("type") == "tool_result" and e.get("tool") == "screen_watchlist"
                   for m in (e.get("words") or {}).get("matches", []) if isinstance(m, dict)]
        try:
            return nav_for(calls, new[-1] if new else None, matches, getattr(self, "question", ""))
        except Exception:  # noqa: BLE001 -- a screen hint must never cost the phone its answer
            log.exception("nav failed")
            return None

    def _calls(self) -> list[dict]:
        ok = {e.get("tool") for e in self.turn_events if e.get("type") == "tool_result" and e.get("ok")}
        return [{"tool": e["tool"], "args": e.get("args") or {}} for e in self.turn_events
                if e.get("type") == "tool_call" and e.get("tool") in ok]

    def nav_seq(self, spoken: str) -> list[dict] | None:
        """The screens of a multi-part answer, one per spoken sentence (voice.ui_api.nav_seq); the phone moves
        to each as its sentence starts playing. "nav" stays the one-screen version for older apps."""
        from voice.ui_api import nav_seq
        try:
            names = {s: u.name for s, u in self.world.universe.items()}
            return nav_seq(self._calls(), spoken, getattr(self, "question", ""), names)
        except Exception:  # noqa: BLE001 -- a screen hint must never cost the phone its answer
            log.exception("nav_seq failed")
            return None

    def _hint(self) -> list[int]:
        self.last_hint = self.next_prefix()
        return self.last_hint

    def next_prefix(self) -> list[int]:
        """Tokens the NEXT turn's first prompt will very likely start with: the system+tools prefix, the history
        window as it will be rendered (spoken answers, not what the phone generated), and the new user header
        and state line, cut just before the question. The phone prefills them while the answer plays, so the
        next turn only prefills its question. A hint only: the phone reuses whatever prefix actually matches."""
        try:
            from tools import schema
            from agent.format import llama_prefix_text
            from agent.loop import build_messages
            mark = "\u2063QQ\u2063"
            msgs = build_messages(mark, self.conv.history, facts=list(self.conv.facts), frame_line=self.frame_line)
            tools = schema()
            full = render_llama(msgs, tools, add_generation_prompt=True)
            pre, cut = llama_prefix_text(tools), full.rfind(mark)
            if cut <= 0 or not full.startswith(pre):
                return []
            return self.pb.prefix_tokens(tools) + self.pb.encode(full[len(pre):cut])
        except Exception:  # noqa: BLE001
            log.exception("next_prefix failed")
            return []


def _plain(v: Any) -> Any:
    import json
    return json.loads(json.dumps(v, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


class TicketRequest(BaseModel):
    session: str
    symbol: str
    direction: str = "buy"
    qty: int = 1
    type: str = "market"
    side: str | None = None
    strike: float | None = None
    expiry_rule: str | None = None


class StepRequest(BaseModel):
    session: str
    question: str | None = None
    model_output: str | None = None
    capped: bool = False
    stats: dict | None = None
    reset: bool = False


def create_app(model: str = "llama-1b-loop", passcode: str | None = None, tokenizer=None, world_factory=None) -> FastAPI:
    spec = next(m for m in registry()["models"] if m["name"] == model)
    if spec.get("family") != "llama":
        raise ValueError(f"{model} is not a Llama-format model")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(spec["model_dir"])
    pb = LlamaPromptBuilder(tokenizer)
    frame_line = bool(spec.get("frame_line"))
    code = passcode if passcode is not None else read_passcode()
    if world_factory is None:
        from mockmarket import World

        def world_factory():
            return World(seed=int(os.environ.get("VOICE_SEED", "7")), scenario=os.environ.get("VOICE_SCENARIO") or None)

    sessions: dict[str, Session] = {}
    glock = threading.Lock()
    app = FastAPI()

    @app.get("/health")
    def health():
        from tools import schema
        return {"ok": True, "model": model, "frame_line": frame_line, "sessions": len(sessions),
                "prefix_tokens": len(pb.prefix_tokens(schema()))}

    def check(x_passcode: str) -> None:
        if not secrets.compare_digest(x_passcode.encode(), code.encode()):
            raise HTTPException(401, "bad passcode")

    @app.get("/prefix")
    def prefix(x_passcode: str = Header(default="")):
        """The fixed system+tools prefix tokens: the phone prefills them once (KV reuse covers the rest)."""
        from tools import schema
        check(x_passcode)
        return {"tokens": pb.prefix_tokens(schema())}

    models = model_files()
    kokoro = kokoro_files()
    llm_files = [{"name": n, "url": f"/model/{n}", "kind": "llm", "size": p.stat().st_size, "sha256": sha256_of(p),
                  "provenance": read_provenance(p)} for n, p in models.items()]
    for f in llm_files:
        log.info("serving %s sha256=%s provenance=%s", f["name"], f["sha256"],
                 json.dumps(f["provenance"]) if f["provenance"] else "NONE (no .provenance.json sidecar)")
    turn_log = TurnLog(turn_log_path(), {"model": model, "files": [
        {"name": f["name"], "sha256": f["sha256"],
         "adapter": (f["provenance"] or {}).get("adapter"), "run_id": (f["provenance"] or {}).get("run_id"),
         "iter": (f["provenance"] or {}).get("iter")} for f in llm_files]})
    log.info("turn log: %s", turn_log.path or "off")
    manifest = {"files": llm_files
                + [{"name": n, "url": f"/model/{n}", "kind": "tts", "role": role, "size": p.stat().st_size,
                    "sha256": sha256_of(p)} for n, (p, role) in kokoro.items()],
                "tts": {"engine": "kokoro-82m-onnx", "voice": KOKORO_VOICE, "speed": KOKORO_SPEED} if kokoro else None}
    served = {**models, **{n: p for n, (p, _) in kokoro.items()}}

    @app.get("/model/manifest.json")
    def model_manifest(x_passcode: str = Header(default="")):
        check(x_passcode)
        return manifest

    @app.get("/model/{name}")
    def model_file(name: str, x_passcode: str = Header(default="")):
        check(x_passcode)
        if name not in served:
            raise HTTPException(404, "no such model")
        return FileResponse(served[name], media_type="application/octet-stream", filename=name)

    def session_for(sid: str) -> Session:
        with glock:
            now = time.time()
            for k in [k for k, s in sessions.items() if now - s.touched > SESSION_TTL_S and not s.busy()]:
                sessions.pop(k)
            sess = sessions.get(sid)
            if sess is None:
                sess = sessions[sid] = Session(world_factory(), frame_line, pb, sid, turn_log)
            sess.touched = now
            return sess

    # ---- trading UI screens (voice.ui_api): the same per-session World the agent's tools read
    from voice import ui_api

    def ui(sid: str, x_passcode: str, f):
        check(x_passcode)
        sess = session_for(sid)
        try:
            return _plain(f(sess))
        except KeyError as e:
            raise HTTPException(404, f"not found: {e}")
        except ValueError as e:
            raise HTTPException(400, str(e))

    ui_static = Path(__file__).parent / "ui_static"

    @app.get("/ui/app/{name}")
    def ui_app(name: str):
        """The trading UI page itself (no secrets in it: the passcode comes in the URL fragment, never sent here)."""
        f = ui_static / name
        if "/" in name or not f.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(f, headers={"Cache-Control": "no-store"})

    @app.get("/ui/home")
    def ui_home(session: str, x_passcode: str = Header(default="")):
        return ui(session, x_passcode, lambda s: ui_api.home(s.world, s.conv.watchlist))

    @app.get("/ui/symbol/{sym}")
    def ui_symbol(sym: str, session: str, x_passcode: str = Header(default="")):
        return ui(session, x_passcode, lambda s: ui_api.symbol_page(s.world, sym.upper()))

    @app.get("/ui/chain")
    def ui_chain(session: str, symbol: str, expiry_rule: str = "nearest", side: str = "call",
                 strike: float | None = None, x_passcode: str = Header(default="")):
        return ui(session, x_passcode, lambda s: ui_api.chain(s.world, symbol.upper(), expiry_rule, side, strike))

    @app.get("/ui/activity")
    def ui_activity(session: str, x_passcode: str = Header(default="")):
        return ui(session, x_passcode, lambda s: ui_api.activity(s.world))

    @app.get("/ui/ticket/{tid}")
    def ui_ticket(tid: str, session: str, x_passcode: str = Header(default="")):
        return ui(session, x_passcode, lambda s: ui_api.ticket(s.world, tid))

    @app.post("/ui/ticket/{tid}/confirm")
    def ui_confirm(tid: str, session: str, x_passcode: str = Header(default="")):
        return ui(session, x_passcode, lambda s: ui_api.confirm(s.world, tid))

    @app.post("/ui/ticket/{tid}/cancel")
    def ui_cancel(tid: str, session: str, x_passcode: str = Header(default="")):
        return ui(session, x_passcode, lambda s: ui_api.cancel(s.world, tid))

    @app.post("/ui/ticket")
    def ui_new_ticket(req: TicketRequest, x_passcode: str = Header(default="")):
        return ui(req.session, x_passcode, lambda s: ui_api.new_ticket(
            s.world, req.symbol.upper(), req.direction, req.qty, req.type, req.side, req.strike, req.expiry_rule))

    @app.post("/step")
    def step(req: StepRequest, x_passcode: str = Header(default="")):
        check(x_passcode)
        sess = session_for(req.session)
        with sess.lock:
            if req.reset:
                if sess.busy():
                    sess.inbox.put(_DONE)
                    sess.thread.join(timeout=5)
                with glock:
                    sessions.pop(req.session, None)
                return {"ok": True}
            if req.question is not None:
                if sess.busy():              # a turn the phone abandoned: end it first
                    sess.inbox.put(_DONE)
                    sess.thread.join(timeout=5)
                sess.start(req.question)
                return sess.next_reply()
            if req.model_output is not None:
                if not sess.busy():
                    raise HTTPException(409, "no turn in progress")
                sess.inbox.put({"model_output": req.model_output, "capped": req.capped, "stats": req.stats})
                return sess.next_reply()
        raise HTTPException(400, "need question, model_output or reset")

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 + adb reverse (default); 0.0.0.0 for LAN")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--model", default="llama-1b-loop")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn
    uvicorn.run(create_app(a.model), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
