"""Watchlist: a code route (not a model tool) on the world's existing watchlist; show / add / remove in code, with
code-routed get_quote calls (lane 'code: watchlist'); per-session persistence; screens stay with the model."""
from __future__ import annotations

import re

from agent import ScriptedBackend, run_turn
from agent.loop import WATCH_LANE, watchlist_route
from agent.memory import Conversation, SessionStore
from mockmarket import World


def test_routes_and_non_routes():
    w = World(1)
    assert watchlist_route("what's on my watchlist", w) == ("show",)
    assert watchlist_route("how's my watch list doing today", w) == ("show",)
    assert watchlist_route("watchlist movers", w) == ("show",)
    assert watchlist_route("add Netflix to my watchlist", w) == ("add", "NFLX")
    assert watchlist_route("put Coinbase on the watchlist", w) == ("add", "COIN")
    assert watchlist_route("take Tesla off my watchlist", w) == ("remove", "TSLA")
    assert watchlist_route("Which names on my watchlist are overbought?", w) is None
    assert watchlist_route("add blorptech to my watchlist", w)[0] == "clarify"
    assert watchlist_route("remove it from my watchlist", w)[0] == "clarify"           # nothing discussed yet
    from types import SimpleNamespace
    assert watchlist_route("add it to my watchlist", w, [SimpleNamespace(symbol="AMD")]) == ("add", "AMD")
    for q in ("anything oversold on my watchlist", "what on my watchlist is overextended",
              "how's Nvidia doing", "add two contracts to the order", "what's the market doing"):
        assert watchlist_route(q, w) is None, q


def test_show_is_code_routed_and_ranked():
    w = World(4)
    be = ScriptedBackend([])                     # the model is never asked
    r = run_turn("what's on my watchlist", [], w, be)
    assert not be.seen
    calls = [e for e in r.events if e["type"] == "tool_call"]
    assert calls and all(e["lane"] == WATCH_LANE and e["tool"] == "get_quote" for e in calls)
    syms = [c["args"]["symbol"] for c in r.tool_calls]
    assert sorted(syms) == sorted(w.watchlist)
    assert [w.quote(s).change_pct for s in syms] == sorted((w.quote(s).change_pct for s in syms), reverse=True)
    assert r.turn.frame["symbols"] == syms                 # follow-ups ("the second one") see the spoken order
    assert not re.search(r"\d", r.spoken) and r.spoken.count(".") <= 2
    ans = [e for e in r.events if e["type"] == "answer"][0]
    assert ans["code"] and ans["lane"] == WATCH_LANE


def test_add_remove_persist(tmp_path):
    store = SessionStore(tmp_path)
    w = World(2)
    w.watchlist = [s for s in w.watchlist if s != "NFLX"]
    conv = Conversation(store, "ab12cd34")
    run = lambda q: conv.ask(q, lambda q_, h, f: run_turn(q_, h, w, ScriptedBackend([]), facts=f), world=w)
    assert "Added" in run("add Netflix to my watchlist").display
    assert "NFLX" in w.watchlist
    assert "already" in run("add Netflix to my watchlist").display
    gone = w.watchlist[0]
    assert "Took" in run(f"take {gone} off my watchlist").display and gone not in w.watchlist
    w2 = World(2)                                          # a reconnect: new world, same session
    from tools import run_tool
    screened = run_tool(w2, "screen_watchlist", {"condition": "oversold"})   # noqa: F841 (before the session loads)
    conv2 = Conversation(store, "ab12cd34")
    r = conv2.ask("what's on my watchlist", lambda q_, h, f: run_turn(q_, h, w2, ScriptedBackend([]), facts=f), world=w2)
    syms = {c["args"]["symbol"] for c in r.tool_calls}
    assert "NFLX" in syms and gone not in syms
