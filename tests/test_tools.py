import json
import math
import re

import pytest

from mockmarket import SCENARIOS, World
from tools import TOOL_SCHEMAS, kind_of, model_view, run_tool, schema
from tools.render import display, spoken

SPEC_TOOLS = ["get_quote", "get_position", "get_portfolio", "get_indicators", "get_levels", "get_option_chain",
              "get_orders", "get_news", "get_fundamentals", "get_market", "screen_watchlist", "what_if",
              "resolve_symbol", "draft_order"]

SLOT_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def calls_for(world):
    held = world.positions[0].symbol if world.positions else "AAPL"
    out = []
    for sym in ("NVDA", "MET", held, "ZZZZ"):
        out += [("get_quote", {"symbol": sym}), ("get_position", {"symbol": sym}),
                ("get_levels", {"symbol": sym}), ("get_news", {"symbol": sym}),
                ("get_fundamentals", {"symbol": sym})]
        for tf in ("1D", "1H", "5m"):
            out.append(("get_indicators", {"symbol": sym, "timeframe": tf}))
        for rule in ("this_week", "next_week", "monthly", "nearest"):
            out.append(("get_option_chain", {"symbol": sym, "expiry_rule": rule, "side": "put"}))
        out.append(("what_if", {"symbol": sym, "move_pct": -10}))
    for rb in ("pnl", "pnl_pct", "weight", "day_move"):
        out.append(("get_portfolio", {"rank_by": rb}))
    for st in ("open", "filled", "all"):
        out.append(("get_orders", {"status": st}))
    for f in ("movers", "breadth", "calendar"):
        out.append(("get_market", {"focus": f}))
    for c in ("oversold", "overbought", "overextended"):
        out.append(("screen_watchlist", {"condition": c}))
    out += [("what_if", {"symbol": "ALL", "move_pct": -5}),
            ("resolve_symbol", {"text": "en vidia"}), ("resolve_symbol", {"text": "met life"}),
            ("resolve_symbol", {"text": "o reilly"}), ("resolve_symbol", {"text": "qwertyuiop"}),
            ("draft_order", {"symbol": "NVDA", "direction": "buy", "qty": 5, "type": "market"}),
            ("draft_order", {"symbol": "AAPL", "direction": "sell", "qty": 2, "type": "limit", "side": "call",
                             "strike": 230, "expiry_rule": "monthly"})]
    return out


def test_schema_matches_spec():
    s = schema()
    assert [t["function"]["name"] for t in s] == SPEC_TOOLS
    assert set(TOOL_SCHEMAS) == set(SPEC_TOOLS)
    for t in s:
        assert t["type"] == "function"
        assert t["function"]["parameters"]["type"] == "object"
    json.dumps(s)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_no_digits_in_model_view(scenario):
    used = set()
    for seed in range(50):
        w = World(seed, scenario)
        for name, args in calls_for(w):
            res = run_tool(w, name, args)
            json.dumps(res)                                     # JSON-serialisable
            assert res["error"] != "tool failed", (name, args)
            used.add(name)
            view = json.dumps(model_view(res))
            stripped = SLOT_PLACEHOLDER.sub("", view)
            assert not re.search(r"\d", stripped), (seed, name, args, view)
            for slot, val in res["slots"].items():
                assert not re.search(r"\d", spoken(slot, val, name)), (name, slot, val)
                assert display(slot, val, name)
                if name != "draft_order":
                    assert kind_of(slot, name) != "text", (name, slot)
    assert used == set(SPEC_TOOLS)


def test_quote_price_equals_position_price():
    for seed in range(30):
        w = World(seed)
        for p in w.positions:
            if p.kind != "stock":
                continue
            q = run_tool(w, "get_quote", {"symbol": p.symbol})
            pos = run_tool(w, "get_position", {"symbol": p.symbol})
            price = q["slots"]["price"]
            implied = pos["slots"]["avg_cost"] + pos["slots"]["pnl"] / pos["slots"]["qty"]
            assert math.isclose(price, implied, rel_tol=1e-9, abs_tol=1e-6)


def test_errors_are_words():
    w = World(1)
    assert run_tool(w, "get_quote", {"symbol": "ZZZZ"})["error"] == "unknown symbol"
    assert run_tool(w, "get_option_chain", {"symbol": "MET", "expiry_rule": "this_week", "side": "call"})[
        "error"] == "no chain for that expiry"
    e = World(1, "empty_portfolio")
    assert run_tool(e, "get_portfolio", {"rank_by": "pnl"})["error"] == "empty portfolio"
    assert run_tool(w, "nope", {})["error"] == "unknown tool"


def test_levels_judgment():
    w = World(7)
    r = run_tool(w, "get_levels", {"symbol": "NVDA"})
    s = r["slots"]
    p = w.quote("NVDA").price
    assert s["support"] < p < s["resistance"]
    assert r["words"]["near_support"] == ("yes" if s["dist_support"] <= 0.02 else "no")


def test_resolve_symbol():
    w = World(1)
    r = run_tool(w, "resolve_symbol", {"text": "en vidia"})
    assert r["words"]["symbol"] == "NVDA" and "NVDL" in r["words"]["alternatives"]
    assert run_tool(w, "resolve_symbol", {"text": "MetLife"})["words"]["symbol"] == "MET"
    assert run_tool(w, "resolve_symbol", {"text": "N V D A"})["words"]["symbol"] == "NVDA"


def test_draft_order_only_tickets():
    w = World(2)
    n_orders = len(w.orders)
    r = run_tool(w, "draft_order", {"symbol": "NVDA", "direction": "buy", "qty": 10, "type": "market"})
    assert r["ok"] and r["words"]["status"] == "needs confirmation"
    tid = r["slots"]["ticket_id"]
    assert w.tickets[tid]["status"] == "needs confirmation"
    assert len(w.orders) == n_orders


def test_slot_names_in_view():
    w = World(1)
    r = run_tool(w, "get_quote", {"symbol": "NVDA"})
    v = model_view(r, {"price": "price_2"})
    assert "{price_2}" in v["slots"] and "{change_pct}" in v["slots"]
    assert set(v) == {"tool", "ok", "words", "slots", "error"}


def test_extra_tickers():
    w = World(7)
    assert run_tool(w, "get_quote", {"symbol": "VIX"})["ok"]
    assert run_tool(w, "get_levels", {"symbol": "VIX"})["error"] == "quote only symbol"
    for sym in ("APP", "ADBE", "ORCL", "AMAT", "XYZ", "SMH"):
        assert run_tool(w, "get_quote", {"symbol": sym})["ok"], sym
    for seed in range(10):
        w = World(seed)
        pos = run_tool(w, "get_position", {"symbol": "NVDA"})
        assert pos["words"]["expires"] in ("this week", "next week", "this month", "later") and "expiry" in pos["slots"]
        assert "expiry" in run_tool(w, "get_portfolio", {"rank_by": "pnl"})["slots"]
        ch = run_tool(w, "get_option_chain", {"symbol": "NVDA", "expiry_rule": "nearest", "side": "call"})
        assert ch["words"]["liquidity"] in ("thin", "normal", "deep") and all(f"oi_{i}" in ch["slots"] for i in range(1, 6))
