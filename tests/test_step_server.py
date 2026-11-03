"""voice/step_server.py: the phone-generation loop (a fake phone posts model outputs back)."""
from __future__ import annotations

import json

import pytest

from agent.backends import DEFAULT_LLAMA

pytestmark = pytest.mark.skipif(not (DEFAULT_LLAMA / "tokenizer.json").exists(), reason="Llama tokenizer not present")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    from fastapi.testclient import TestClient
    from transformers import AutoTokenizer
    from voice.step_server import create_app
    d = tmp_path_factory.mktemp("gguf")
    (d / "m.gguf").write_bytes(b"GGUF" + bytes(range(256)) * 4)
    (d / "m.gguf.provenance.json").write_text(json.dumps(PROV))
    os.environ["STEP_MODEL_DIR"], os.environ["STEP_MODEL_FILES"] = str(d), "m.gguf"
    os.environ["STEP_TURN_LOG"] = str(d / "logs" / "step_turns.jsonl")
    try:
        c = TestClient(create_app(passcode="pc", tokenizer=AutoTokenizer.from_pretrained(str(DEFAULT_LLAMA))))
        c.turn_log = d / "logs" / "step_turns.jsonl"
        return c
    finally:
        for k in ("STEP_MODEL_DIR", "STEP_MODEL_FILES", "STEP_TURN_LOG"):
            os.environ.pop(k)


H = {"X-Passcode": "pc"}
PROV = {"adapter": "adapters/llama-1b-loop-r2", "run_id": "2e517641", "iter": 12885}


def test_auth(client):
    assert client.post("/step", json={"session": "x", "question": "hi"}).status_code == 401
    assert client.get("/prefix").status_code == 401
    assert client.get("/model/manifest.json").status_code == 401


def test_turn_roundtrip(client):
    from agent.format import render_llama
    r = client.post("/step", json={"session": "t", "question": "What's Nvidia trading at?"}, headers=H).json()
    assert r["prompt"].endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    prefix = client.get("/prefix", headers=H).json()["tokens"]
    assert r["tokens"][:len(prefix)] == prefix
    r = client.post("/step", json={"session": "t", "model_output": '{"name": "get_quote", "parameters": {"symbol": "NVDA"}}'},
                    headers=H).json()
    assert "ipython" in r["prompt"] and any(e["type"] == "tool_call" for e in r["events"])
    r = client.post("/step", json={"session": "t", "model_output": "Nvidia is at {price}. It is {change_pct} today. More.",
                                   "capped": True}, headers=H).json()
    assert r["template"] == "Nvidia is at {price}. It is {change_pct} today."
    assert r["say"] and "{" not in r["say"] and r["tool_calls"][0]["tool"] == "get_quote"
    assert client.post("/step", json={"session": "t", "model_output": "x"}, headers=H).status_code == 409


def test_model_download(client):
    import hashlib
    m = client.get("/model/manifest.json", headers=H).json()["files"][0]
    body = client.get(m["url"], headers=H).content
    assert m["size"] == len(body) and m["sha256"] == hashlib.sha256(body).hexdigest()
    part = client.get(m["url"], headers={**H, "Range": "bytes=10-"})
    assert part.status_code == 206 and part.content == body[10:]
    assert client.get("/model/other.gguf", headers=H).status_code == 404


def test_next_prefix_is_next_turn_prefix(client):
    """The hint returned with an answer is (all but the question of) the next turn's first prompt."""
    s = "np"
    client.post("/step", json={"session": s, "question": "What's Nvidia trading at?"}, headers=H)
    client.post("/step", json={"session": s, "model_output": '{"name": "get_quote", "parameters": {"symbol": "NVDA"}}'},
                headers=H)
    r = client.post("/step", json={"session": s, "model_output": "Nvidia is at {price}. It is {change_pct} today."},
                    headers=H).json()
    hint = r["next_prefix"]
    nxt = client.post("/step", json={"session": s, "question": "And AMD?"}, headers=H).json()["tokens"]
    common = next((i for i, (a, b) in enumerate(zip(hint, nxt)) if a != b), min(len(hint), len(nxt)))
    assert len(hint) > 600 and common >= len(hint) - 1, (len(hint), common, len(nxt))
    client.post("/step", json={"session": s, "reset": True}, headers=H)


def _turns(client, session):
    lines = client.turn_log.read_text().splitlines() if client.turn_log.exists() else []
    return [r for r in map(json.loads, lines) if r["session"] == session]


def test_turn_log_records_what_the_phone_generated(client):
    """Regression (2026-11-03 phone test): two of three turns ended after the phone's step-1 output with no tool
    call, and the server had logged nothing but 'POST /step 200' -- the question, the model output and the spoken
    answer were unrecoverable. Every turn must leave one JSON line with all three and the served model."""
    s = "tl"
    q = "What is Apple trading at?"
    r = client.post("/step", json={"session": s, "question": q}, headers=H).json()
    out = "Apple is a large technology company."          # step 1 answers without calling get_quote
    for _ in range(4):
        if "say" in r:
            break
        r = client.post("/step", json={"session": s, "model_output": out, "capped": False,
                                       "stats": {"decode_tok_s": 30.0}}, headers=H).json()
    assert "say" in r
    rec = _turns(client, s)[-1]
    assert rec["question"] == q and rec["outcome"] == "answered"
    assert rec["say"] == r["say"] and rec["fallback"] == r["fallback"] and rec["tool_calls"] == r["tool_calls"]
    assert rec["n_steps"] >= 1 and rec["steps"][0]["model_output"] == out
    assert rec["steps"][0]["kind"] == "answer" and rec["steps"][0]["stats"] == {"decode_tok_s": 30.0}
    assert rec["steps"][0]["prompt_tokens"] > 500
    m = client.get("/model/manifest.json", headers=H).json()["files"][0]
    f = rec["served"]["files"][0]
    assert f["name"] == "m.gguf" and f["sha256"] == m["sha256"] and f["run_id"] == "2e517641" and f["iter"] == 12885
    client.post("/step", json={"session": s, "reset": True}, headers=H)


def test_turn_log_records_abandoned_turn(client):
    """A turn the phone drops (a new question mid-turn: cut-off speech, hands-free re-listen) is logged too."""
    s = "ab"
    client.post("/step", json={"session": s, "question": "What is"}, headers=H)
    client.post("/step", json={"session": s, "question": "What is Nvidia at?"}, headers=H)
    first = _turns(client, s)[0]
    assert first["question"] == "What is" and first["outcome"] == "abandoned" and first["n_steps"] == 0
    client.post("/step", json={"session": s, "reset": True}, headers=H)
    second = _turns(client, s)[1]
    assert second["question"] == "What is Nvidia at?" and second["outcome"] == "abandoned"


def test_manifest_carries_provenance(client):
    m = client.get("/model/manifest.json", headers=H).json()["files"][0]
    assert m["provenance"] == PROV
