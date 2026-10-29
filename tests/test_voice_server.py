"""WS round trips against the fake agent with STT/TTS mocked (no models load)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from voice.server import create_app
from voice.tts import split_sentences


class FakeSTT:
    loaded = True

    def __init__(self, text="How's my Nvidia position?"):
        self.text, self.calls = text, []

    def warm(self):
        return 0.0

    def transcribe(self, pcm16, vocab=None):
        self.calls.append(len(pcm16))
        return self.text


class FakeTTS:
    loaded = True

    def __init__(self):
        self.spoken = []

    def warm(self):
        return 0.0

    def synth(self, sentence):
        self.spoken.append(sentence)
        return b"\x01\x00" * 240  # 10 ms of PCM16


def make(**kw):
    stt, tts = FakeSTT(), FakeTTS()
    app = create_app(stt=stt, tts=tts, agent_mode="fake", passcode="pw", warm=False, **kw)
    return TestClient(app), stt, tts


def collect(ws):
    """Read until `done`; return (events, binary_frames)."""
    events, frames = [], []
    while True:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            frames.append(msg["bytes"])
            continue
        import json
        e = json.loads(msg["text"])
        events.append(e)
        if e["type"] == "done":
            return events, frames


def test_health_and_index():
    client, _, _ = make()
    with client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["agent"] == "fake"
        assert "<html" in client.get("/").text.lower()


def test_bad_passcode_rejected():
    client, _, _ = make()
    with client, client.websocket_connect("/ws?k=nope") as ws:
        assert ws.receive_json() == {"type": "error", "message": "bad passcode"}
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_typed_question_round_trip():
    client, stt, tts = make()
    with client, client.websocket_connect("/ws?k=pw") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "text", "text": "How's my Nvidia position?"})
        events, frames = collect(ws)
    types = [e["type"] for e in events]
    assert types[0] == "transcript" and events[0]["source"] == "text"
    for t in ("tool_call", "tool_result", "filler", "answer", "done"):
        assert t in types
    assert types.index("tool_call") < types.index("tool_result") < types.index("answer")
    # each binary frame is preceded by an audio header, seq increasing
    audio = [e for e in events if e["type"] == "audio"]
    assert len(audio) == len(frames) > 0
    assert [a["seq"] for a in audio] == list(range(len(audio)))
    assert all(a["sample_rate"] == 24000 for a in audio)
    # filler spoken first, then the answer sentence by sentence
    answer = next(e for e in events if e["type"] == "answer")
    assert tts.spoken[0] == "Checking your Nvidia position."
    assert tts.spoken[1:] == split_sentences(answer["spoken"])
    t = events[-1]["timings"]
    assert t["stt"] == 0 and t["first_audio"] is not None and t["total"] >= t["first_audio"]
    assert stt.calls == []


def test_audio_round_trip():
    client, stt, tts = make()
    with client, client.websocket_connect("/ws?k=pw") as ws:
        ws.receive_json()
        for _ in range(5):
            ws.send_bytes(b"\x00\x01" * 1600)  # 5 x 100 ms
        ws.send_json({"type": "end"})
        events, frames = collect(ws)
    assert stt.calls == [5 * 3200]
    assert events[0] == {"type": "transcript", "text": "How's my Nvidia position?", "source": "audio"}
    assert frames and events[-1]["type"] == "done"
    assert events[-1]["timings"]["audio_seconds"] == 0.5


def test_empty_transcript_reports_error():
    client, stt, _ = make()
    stt.text = ""
    with client, client.websocket_connect("/ws?k=pw") as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00\x00" * 160)
        ws.send_json({"type": "end"})
        events, frames = collect(ws)
    assert [e["type"] for e in events] == ["transcript", "error", "done"] and not frames


def test_ticket_and_confirm():
    client, _, tts = make()
    with client, client.websocket_connect("/ws?k=pw") as ws:
        ws.receive_json()
        ws.send_json({"type": "text", "text": "buy 2 Nvidia calls"})
        events, _ = collect(ws)
        ticket = next(e for e in events if e["type"] == "ticket")
        assert ticket["symbol"] == "NVDA" and ticket["qty"] == 2 and ticket["side"] == "call"
        ws.send_json({"type": "confirm_ticket", "ticket_id": ticket["ticket_id"]})
        events, frames = collect(ws)
    assert events[0]["type"] == "ticket_confirmed" and events[0]["ticket_id"] == ticket["ticket_id"]
    assert frames and tts.spoken[-1] == "Confirmed."


def test_set_model():
    client, _, _ = make()
    with client, client.websocket_connect("/ws?k=pw") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_model", "model": "trained"})
        e = ws.receive_json()
        assert e["type"] == "model" and e["model"] == "trained" and e["resolved"]
        ws.send_json({"type": "set_model", "model": "huge"})
        assert ws.receive_json()["type"] == "error"


def test_rate_limit():
    client, _, _ = make(max_turns_per_min=2)
    with client, client.websocket_connect("/ws?k=pw") as ws:
        ws.receive_json()
        for _ in range(2):
            ws.send_json({"type": "text", "text": "price of Apple"})
            collect(ws)
        ws.send_json({"type": "text", "text": "price of Apple"})
        e = ws.receive_json()
        assert e["type"] == "error" and "rate limit" in e["message"]


def test_newer_session_replaces_older():
    client, _, _ = make()
    with client:
        with client.websocket_connect("/ws?k=pw") as ws1:
            ws1.receive_json()
            with client.websocket_connect("/ws?k=pw") as ws2:
                assert ws2.receive_json()["type"] == "hello"
                e = ws1.receive_json()
                assert e["type"] == "error" and "replaced" in e["message"]
                ws2.send_json({"type": "text", "text": "price of Tesla"})
                events, _ = collect(ws2)
                assert events[-1]["type"] == "done"


def test_split_sentences():
    assert split_sentences("You're up. It's large! Down two percent?") == ["You're up. It's large!", "Down two percent?"]
    assert split_sentences("  ") == []
