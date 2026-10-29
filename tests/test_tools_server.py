import re

from fastapi.testclient import TestClient

from tools.server import create_app, seed_for

DIGIT_OUTSIDE_SLOT = re.compile(r"\d")


def test_call_returns_model_view_and_raw_values():
    client = TestClient(create_app())
    reply = client.post("/call", json={"tool": "get_quote", "arguments": {"symbol": "NVDA"}, "row_id": "r1"})
    assert reply.status_code == 200
    body = reply.json()
    view = body["result"]
    assert view["tool"] == "get_quote" and view["ok"] is True
    assert all(re.fullmatch(r"\{[a-z_0-9]+\}", s) for s in view["slots"])
    assert set(body["raw"]["slots"]) == {s[1:-1] for s in view["slots"]}
    assert set(body["raw"]["spoken"]) == set(body["raw"]["slots"])
    assert not any(DIGIT_OUTSIDE_SLOT.search(v) for v in body["raw"]["spoken"].values())
    # The model-visible result carries no digits outside slot names.
    stripped = re.sub(r"\{[A-Za-z_][A-Za-z0-9_]*\}", "", str(view["words"]))
    assert not DIGIT_OUTSIDE_SLOT.search(stripped)


def test_deterministic_per_row_and_seed():
    client = TestClient(create_app())
    ask = lambda **kw: client.post("/call", json={"tool": "get_quote", "arguments": {"symbol": "AAPL"}, **kw}).json()
    assert ask(row_id="row-a") == ask(row_id="row-a")
    assert ask(seed=11) == TestClient(create_app()).post(
        "/call", json={"tool": "get_quote", "arguments": {"symbol": "AAPL"}, "seed": 11}).json()
    assert seed_for("row-a", None) == seed_for("row-a", None) != seed_for("row-b", None)
    assert seed_for(None, 5) == 5


def test_unknown_tool_is_a_result_and_unknown_scenario_a_422():
    client = TestClient(create_app())
    body = client.post("/call", json={"tool": "nope", "arguments": {}}).json()
    assert body["result"]["ok"] is False and body["result"]["error"]
    assert client.post("/call", json={"tool": "get_quote", "arguments": {"symbol": "X"},
                                      "scenario": "no_such"}).status_code == 422
    assert client.get("/health").json()["ok"] is True
