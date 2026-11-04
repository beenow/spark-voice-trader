"""voice/ui_api.py (trading UI screens) and its /ui routes on the step server."""
from __future__ import annotations

import pytest

from mockmarket import World
from tools.impl import draft_order, get_levels, get_quote
from voice import ui_api


@pytest.fixture
def world():
    return World(seed=7)


def test_screens_match_the_tools(world):
    page = ui_api.symbol_page(world, "NVDA")
    assert page["price"] == get_quote(world, "NVDA").slots["price"]
    assert page["levels"]["support"] == get_levels(world, "NVDA").slots["support"]
    assert page["chart"]["3M"]["c"][-1] == page["price"]
    home = ui_api.home(world)
    assert home["total"] == round(world.total_equity(), 2)
    assert {p["symbol"] for p in home["positions"]} == {p.symbol for p in world.positions}
    assert [w["symbol"] for w in home["watchlist"]] == world.watchlist
    ch = ui_api.chain(world, "TSLA", "next_week", "put")
    assert ch["expiry"] == world.expiry_for("TSLA", "next_week").isoformat() and len(ch["rows"]) == 5


def test_ticket_confirm_roundtrip(world):
    tid = draft_order(world, "TSLA", "buy", 2, side="put", strike=415, expiry_rule="next_week").slots["ticket_id"]
    t = ui_api.ticket(world, tid)
    assert t["status"] == "needs confirmation" and t["qty"] == 2 and t["side"] == "put" and t["strike"] == 415
    n = len(world.orders)
    done = ui_api.confirm(world, tid)
    assert done["status"] == "submitted" and len(world.orders) == n + 1
    assert ui_api.confirm(world, tid)["order_id"] == done["order_id"] and len(world.orders) == n + 1   # idempotent
    act = ui_api.activity(world)
    assert act["orders"][0]["order_id"] == done["order_id"] and act["orders"][0]["voice"]
    assert act["tickets"][0]["ticket_id"] == tid
    with pytest.raises(KeyError):
        ui_api.ticket(world, "nope")


def test_manual_ticket(world):
    t = ui_api.new_ticket(world, "AAPL", "buy", 3)
    assert t["kind"] == "stock" and t["est_cost"] == round(world.quote("AAPL").price * 3, 2)
    assert ui_api.cancel(world, t["ticket_id"])["status"] == "cancelled"


def test_nav_for():
    nav = ui_api.nav_for
    assert nav([]) is None
    assert nav([{"tool": "draft_order", "args": {"symbol": "TSLA"}}], "T7-1") == \
        {"screen": "ticket", "ticket_id": "T7-1", "symbol": "TSLA"}
    n = nav([{"tool": "get_option_chain", "args": {"symbol": "TSLA", "expiry_rule": "next_week", "side": "put"}}])
    assert n["screen"] == "symbol" and n["tab"] == "options" and n["side"] == "put"
    n = nav([{"tool": "get_position", "args": {"symbol": "NVDA"}}, {"tool": "get_levels", "args": {"symbol": "NVDA"}}])
    assert n == {"screen": "symbol", "symbol": "NVDA", "tab": "overview", "sections": ["position", "levels"]}
    assert nav([{"tool": "get_orders", "args": {}}])["screen"] == "activity"
    assert nav([{"tool": "screen_watchlist", "args": {}}], matches=["PEP"])["highlight"] == ["PEP"]
    assert nav([{"tool": "what_if", "args": {"symbol": "ALL", "move_pct": -10}}])["section"] == "positions"
    two = nav([{"tool": "get_quote", "args": {"symbol": "AAPL"}}, {"tool": "get_quote", "args": {"symbol": "MSFT"}}])
    assert two["screen"] == "home" and two["highlight"] == ["AAPL", "MSFT"]


# ------------------------------------------------------------------ routes
from agent.backends import DEFAULT_LLAMA  # noqa: E402

needs_tok = pytest.mark.skipif(not (DEFAULT_LLAMA / "tokenizer.json").exists(), reason="Llama tokenizer not present")
H = {"X-Passcode": "pc"}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from transformers import AutoTokenizer
    from voice.step_server import create_app
    return TestClient(create_app(passcode="pc", tokenizer=AutoTokenizer.from_pretrained(str(DEFAULT_LLAMA))))


@needs_tok
def test_ui_routes_and_voice_ticket(client):
    s = "ui1"
    assert client.get(f"/ui/home?session={s}").status_code == 401
    home = client.get(f"/ui/home?session={s}", headers=H).json()
    assert home["positions"] and home["watchlist"]
    sym = client.get(f"/ui/symbol/nvda?session={s}", headers=H).json()
    assert sym["symbol"] == "NVDA" and sym["indicators"]["rsi_state"]
    assert client.get(f"/ui/symbol/ZZZZ?session={s}", headers=H).status_code == 404
    # a spoken order: the reply after the tool step carries the ticket screen, then confirm from the UI
    client.post("/step", json={"session": s, "question": "Buy two Tesla next week puts at four fifteen"}, headers=H)
    r = client.post("/step", json={"session": s, "model_output": '{"name": "draft_order", "parameters": {"symbol": '
                                   '"TSLA", "direction": "buy", "qty": 2, "side": "put", "strike": 415, '
                                   '"expiry_rule": "next_week", "type": "market"}}'}, headers=H).json()
    nav = r.get("nav")
    assert nav and nav["screen"] == "ticket", r
    t = client.get(f"/ui/ticket/{nav['ticket_id']}?session={s}", headers=H).json()
    assert t["symbol"] == "TSLA" and t["qty"] == 2
    done = client.post(f"/ui/ticket/{nav['ticket_id']}/confirm?session={s}", headers=H).json()
    assert done["status"] == "submitted"
    act = client.get(f"/ui/activity?session={s}", headers=H).json()
    assert act["orders"][0]["order_id"] == done["order_id"]
    m = client.post("/ui/ticket", json={"session": s, "symbol": "aapl", "qty": 1}, headers=H).json()
    assert m["symbol"] == "AAPL" and m["status"] == "needs confirmation"
