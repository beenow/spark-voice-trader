"""Named models + compare mode over the WebSocket, with the REAL agent loop driven by ScriptedBackends
(one per model name; no model loads). STT/TTS are the fakes from test_voice_server."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from agent import ScriptedBackend
from tests.test_voice_server import FakeSTT, FakeTTS, collect
from voice.server import create_app


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    mdir = tmp_path / "glimmer"
    mdir.mkdir()
    ldir = tmp_path / "llama"
    ldir.mkdir()
    ads = {}
    for n in ("alt", "spark", "llama"):
        d = tmp_path / f"ad-{n}"
        d.mkdir()
        (d / "adapters.safetensors").write_bytes(b"")
        ads[n] = str(d)
    reg = {"default": "glimmer-base", "models": [
        {"name": "glimmer-base", "label": "Glimmer base", "family": "glimmer", "model_dir": str(mdir), "adapter": None},
        {"name": "glimmer-alt", "label": "Glimmer + Alt", "family": "glimmer", "model_dir": str(mdir),
         "adapter": ads["alt"]},
        {"name": "glimmer-spark", "label": "Glimmer + Spark", "family": "glimmer", "model_dir": str(mdir),
         "adapter": ads["spark"]},
        {"name": "llama-1b", "label": "Llama 1B", "family": "llama", "model_dir": str(ldir), "adapter": ads["llama"]},
        {"name": "llama-missing", "label": "missing", "family": "llama", "model_dir": str(ldir),
         "adapter": str(tmp_path / "nope")},
    ]}
    path = tmp_path / "models.json"
    path.write_text(json.dumps(reg))
    monkeypatch.setenv("VOICE_MODELS", str(path))
    monkeypatch.setenv("VOICE_ADAPTER", ads["spark"])
    monkeypatch.delenv("VOICE_DEFAULT_MODEL", raising=False)
    return reg


def quote_then_answer(text):
    """Scripted model: get_quote(NVDA), then `text` (may use {price}). Reusable across turns."""
    def step(messages):
        tail = messages[max(i for i, m in enumerate(messages) if m["role"] == "user"):]
        if not any(m["role"] == "tool" for m in tail):
            return {"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}}]}
        return {"answer": text}
    return ScriptedBackend([step] * 20)


def make(registry):
    stt, tts = FakeSTT(), FakeTTS()
    app = create_app(stt=stt, tts=tts, agent_mode="real", passcode="pw", warm=False)
    runner = app.state.voice.runner
    backends = {
        "glimmer-base": quote_then_answer("Base says Nvidia is at {price}."),
        "glimmer-alt": quote_then_answer("Alt says Nvidia is at {price}."),
        "glimmer-spark": quote_then_answer("Spark says Nvidia is at {price}."),
        "llama-1b": ScriptedBackend([{"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}},
                                                     {"tool": "get_levels", "args": {"symbol": "NVDA"}}]},
                                     {"answer": "Llama says {price}, support {support}."}] * 5),
    }
    runner._backends.update(backends)
    return TestClient(app), tts, backends


def test_hello_lists_models(registry):
    client, _, _ = make(registry)
    with client, client.websocket_connect("/ws?k=pw") as ws:
        hello = ws.receive_json()
    assert hello["model"] == "glimmer-base" and hello["compare"] is None
    by = {m["name"]: m for m in hello["models"]}
    assert set(by) == {"glimmer-base", "glimmer-alt", "glimmer-spark", "llama-1b", "llama-missing"}
    assert by["llama-1b"]["available"] and not by["llama-missing"]["available"]
    assert by["llama-1b"]["family"] == "llama" and by["llama-1b"]["label"] == "Llama 1B"


def test_set_named_model_and_aliases(registry):
    client, _, backends = make(registry)
    with client, client.websocket_connect("/ws?k=pw") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_model", "model": "llama-1b"})
        assert ws.receive_json() == {"type": "model", "model": "llama-1b", "resolved": "llama-1b"}
        ws.send_json({"type": "text", "text": "Nvidia price and support?"})
        events, _ = collect(ws)
        ans = next(e for e in events if e["type"] == "answer")
        assert ans["template"].startswith("Llama says") and not ans["fallback"]
        calls = [e["tool"] for e in events if e["type"] == "tool_call"]
        assert calls == ["get_quote", "get_levels"]                      # parallel calls: one decision
        assert len(backends["llama-1b"].seen) == 2
        # back-compat aliases: trained = the Glimmer whose adapter is $VOICE_ADAPTER, base = no adapter
        ws.send_json({"type": "set_model", "model": "trained"})
        assert ws.receive_json()["resolved"] == "glimmer-spark"
        ws.send_json({"type": "set_model", "model": "base"})
        assert ws.receive_json()["resolved"] == "glimmer-base"
        ws.send_json({"type": "set_model", "model": "llama-missing"})
        e = ws.receive_json()
        assert e["type"] == "error" and "not available" in e["message"]
        ws.send_json({"type": "set_model", "model": "gpt-9"})
        assert ws.receive_json()["type"] == "error"


def test_compare_two_models(registry):
    client, tts, backends = make(registry)
    with client, client.websocket_connect("/ws?k=pw") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_compare", "models": ["glimmer-alt", "llama-1b"]})
        assert ws.receive_json() == {"type": "compare", "models": ["glimmer-alt", "llama-1b"]}
        ws.send_json({"type": "text", "text": "How is Nvidia doing?"})
        events, frames = collect(ws)
        types = [e["type"] for e in events]
        assert types[0] == "transcript" and types[1] == "compare_start"
        assert types.count("answer") == 2 and types.count("lane_done") == 2 and types[-1] == "done"
        lane_ev = [e for e in events if e["type"] in ("tool_call", "tool_result", "answer", "filler")]
        assert all("lane" in e and "model" in e for e in lane_ev)
        lanes = [e["lane"] for e in lane_ev]
        assert lanes == sorted(lanes)                                     # lane 0 runs to the end, then lane 1
        a0, a1 = [e for e in events if e["type"] == "answer"]
        assert (a0["lane"], a0["model"]) == (0, "glimmer-alt") and a0["template"].startswith("Alt says")
        assert (a1["lane"], a1["model"]) == (1, "llama-1b") and a1["template"].startswith("Llama says")
        # audio only for lane 0: the filler + lane 0's answer; lane 1 is silent until asked
        assert frames and not any("Llama" in s for s in tts.spoken)
        assert any(s.startswith("Alt says") for s in tts.spoken)
        done = events[-1]["timings"]
        assert [x["model"] for x in done["lanes"]] == ["glimmer-alt", "llama-1b"]
        assert all(x["agent"] is not None and x["first_tool"] is not None for x in done["lanes"])
        assert done["lanes"][0]["first_audio"] is not None and done["lanes"][1]["first_audio"] is None
        # both models saw the same question; only lane 0 entered history
        assert backends["glimmer-alt"].seen[0][-1]["content"] == backends["llama-1b"].seen[0][-1]["content"]
        n_spoken = len(tts.spoken)
        ws.send_json({"type": "play_lane", "lane": 1})
        events, frames = collect(ws)
        assert frames and events[-1]["timings"].get("replay")
        assert tts.spoken[n_spoken].startswith("Llama says")
        # follow-up: history has ONE prior turn (lane 0's)
        ws.send_json({"type": "set_compare", "models": None})
        assert ws.receive_json() == {"type": "compare", "models": None}
        ws.send_json({"type": "set_model", "model": "glimmer-base"})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "And now?"})
        events, _ = collect(ws)
        assert "compare_start" not in [e["type"] for e in events]
        msgs = backends["glimmer-base"].seen[0]
        assert [m["role"] for m in msgs[:4]] == ["system", "user", "assistant", "user"]
        assert "Alt says" in msgs[2]["content"]


def test_compare_per_message_and_validation(registry):
    client, _, _ = make(registry)
    with client, client.websocket_connect("/ws?k=pw") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_compare", "models": ["glimmer-alt"]})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "text", "text": "Nvidia?", "compare": ["glimmer-spark", "glimmer-base"]})
        events, _ = collect(ws)
        assert [e["model"] for e in events if e["type"] == "answer"] == ["glimmer-spark", "glimmer-base"]
        ws.send_json({"type": "play_lane", "lane": 7})
        e = ws.receive_json()
        assert e["type"] == "error"
