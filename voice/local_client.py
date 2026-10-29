"""Mac fallback client: push-to-talk from the terminal over the same WS protocol.

  python -m voice.local_client [--url ws://localhost:8300/ws] [--k demo] [--wav file.wav]

Press Enter to start recording, Enter again to stop and send.
Type a question and press Enter to send it as text.
Commands: /base, /trained, /confirm, /quit.
--wav sends a 16 kHz mono PCM16 WAV as one utterance, prints the events, and exits.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import sys
import threading
import time
import wave

import numpy as np

IN_RATE, OUT_RATE, FRAME = 16000, 24000, 1600   # 100 ms frames


class Player:
    """Plays 24 kHz PCM16 chunks back to back."""

    def __init__(self, enabled=True):
        self.q: queue.Queue = queue.Queue()
        self.buf = np.zeros(0, dtype=np.int16)
        self.stream = None
        if enabled:
            import sounddevice as sd
            self.stream = sd.OutputStream(samplerate=OUT_RATE, channels=1, dtype="int16", callback=self._cb)
            self.stream.start()

    def _cb(self, out, frames, _t, _s):
        while len(self.buf) < frames and not self.q.empty():
            self.buf = np.concatenate([self.buf, self.q.get_nowait()])
        n = min(frames, len(self.buf))
        out[:n, 0] = self.buf[:n]
        out[n:, 0] = 0
        self.buf = self.buf[n:]

    def play(self, pcm: bytes):
        if self.stream is not None:
            self.q.put(np.frombuffer(pcm, dtype="<i2"))

    def stop(self):
        with self.q.mutex:
            self.q.queue.clear()
        self.buf = np.zeros(0, dtype=np.int16)


def show(e: dict):
    t = e.get("type")
    if t == "transcript":
        print(f"\n  you: {e.get('text')!r} ({e.get('source')})")
    elif t == "tool_call":
        print(f"  -> {e['tool']}({json.dumps(e.get('args'))})")
    elif t == "tool_result":
        print(f"     words={json.dumps(e.get('words'))} slots={json.dumps(e.get('slots_display'))}")
    elif t == "filler":
        print(f"  ({e.get('text')})")
    elif t == "answer":
        print(f"  agent: {e.get('display')}")
    elif t == "ticket":
        print(f"  TICKET {e.get('ticket_id')}: {json.dumps(e)}  (type /confirm)")
    elif t == "done":
        print(f"  timings: {e.get('timings')}")
    elif t in ("error", "ticket_confirmed", "hello", "model"):
        print(f"  [{t}] {json.dumps({k: v for k, v in e.items() if k != 'type'})}")


async def run_wav(url: str, path: str, play: bool):
    import websockets
    with wave.open(path) as w:
        assert w.getframerate() == IN_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16 kHz mono PCM16"
        pcm = w.readframes(w.getnframes())
    player = Player(play)
    async with websockets.connect(url, max_size=None) as ws:
        print(json.loads(await ws.recv()))
        t0 = time.perf_counter()
        for i in range(0, len(pcm), FRAME * 2):
            await ws.send(pcm[i:i + FRAME * 2])
        await ws.send(json.dumps({"type": "end"}))
        first = None
        async for m in ws:
            if isinstance(m, bytes):
                first = first or time.perf_counter() - t0
                player.play(m)
                continue
            e = json.loads(m)
            show(e)
            if e["type"] == "done":
                print(f"  client-side first audio after end: {first and round(first * 1000)} ms")
                break
    if play:
        await asyncio.sleep(4)


async def interactive(url: str):
    import sounddevice as sd
    import websockets
    player = Player(True)
    lines: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    threading.Thread(target=lambda: [loop.call_soon_threadsafe(lines.put_nowait, l.rstrip("\n")) for l in sys.stdin],
                     daemon=True).start()
    rec = {"on": False}
    last_ticket = {"id": None}

    async with websockets.connect(url, max_size=None) as ws:
        def mic_cb(indata, frames, _t, _s):
            if rec["on"]:
                data = bytes(indata)
                asyncio.run_coroutine_threadsafe(ws.send(data), loop)

        mic = sd.RawInputStream(samplerate=IN_RATE, channels=1, dtype="int16", blocksize=FRAME, callback=mic_cb)
        mic.start()

        async def reader():
            async for m in ws:
                if isinstance(m, bytes):
                    player.play(m)
                    continue
                e = json.loads(m)
                if e.get("type") == "ticket":
                    last_ticket["id"] = e.get("ticket_id")
                show(e)

        rt = asyncio.create_task(reader())
        print(__doc__.split("\n\n")[1])
        while True:
            line = (await lines.get()).strip()
            if line == "":
                if not rec["on"]:
                    player.stop()
                    await ws.send(json.dumps({"type": "start"}))
                    rec["on"] = True
                    print("  recording... (Enter to stop)")
                else:
                    await asyncio.sleep(0.15)
                    rec["on"] = False
                    await ws.send(json.dumps({"type": "end"}))
            elif line in ("/base", "/trained"):
                await ws.send(json.dumps({"type": "set_model", "model": line[1:]}))
            elif line == "/confirm":
                await ws.send(json.dumps({"type": "confirm_ticket", "ticket_id": last_ticket["id"]}))
            elif line == "/quit":
                break
            else:
                player.stop()
                await ws.send(json.dumps({"type": "text", "text": line}))
        rt.cancel()
        mic.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://localhost:8300/ws")
    ap.add_argument("--k", default=os.environ.get("VOICE_PASSCODE", "demo"))
    ap.add_argument("--wav")
    ap.add_argument("--no-play", action="store_true")
    a = ap.parse_args()
    url = f"{a.url}?k={a.k}"
    if a.wav:
        asyncio.run(run_wav(url, a.wav, not a.no_play))
    else:
        asyncio.run(interactive(url))


if __name__ == "__main__":
    main()
