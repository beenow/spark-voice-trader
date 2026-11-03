"""voice/step_server.py: the phone-generation loop (a fake phone posts model outputs back)."""
from __future__ import annotations

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
    os.environ["STEP_MODEL_DIR"], os.environ["STEP_MODEL_FILES"] = str(d), "m.gguf"
    try:
        return TestClient(create_app(passcode="pc", tokenizer=AutoTokenizer.from_pretrained(str(DEFAULT_LLAMA))))
    finally:
        os.environ.pop("STEP_MODEL_DIR"), os.environ.pop("STEP_MODEL_FILES")


H = {"X-Passcode": "pc"}


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
