"""Model-free unit tests for voice helpers."""
import numpy as np

from voice import fake_agent
from voice.agent_bridge import AgentRunner
from voice.stt import STT, build_prompt, pcm16_to_float
from voice.tts import float_to_pcm16


def test_pcm_roundtrip():
    x = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
    y = pcm16_to_float(float_to_pcm16(x))
    assert np.allclose(x, y, atol=1e-3)


def test_prompt_uses_caller_vocab():
    assert "Zebra Corp" in build_prompt(["Zebra Corp"])
    assert "Nvidia" in build_prompt(None)


def test_stt_skips_silence_and_taps_without_loading():
    stt = STT(model_dir="/nonexistent")
    assert stt.transcribe(b"\x00\x00" * 16000) == ""   # 1 s silence
    assert stt.transcribe(b"\x00\x10" * 1000) == ""    # tap, too short
    assert not stt.loaded


def test_fake_agent_event_order():
    ev = []
    fake_agent.run_turn("How's my Nvidia position?", on_event=ev.append, delay=0)
    types = [e["type"] for e in ev]
    assert types[:3] == ["tool_call", "filler", "tool_result"] and types[-1] == "answer"


def test_runner_fake_mode():
    r = AgentRunner("fake")
    ev = []
    out = r.run("price of Tesla", [], "base", ev.append)
    assert out["symbol"] == "TSLA" and ev[-1]["type"] == "answer"
