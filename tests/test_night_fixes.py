"""Night fixes (Nov 2): spoken 'hundred' strikes, day / ranking said, what_if follow-up in code,
ticket ids never read aloud, 'times times'. ScriptedBackend only; no model loads."""
from __future__ import annotations

import re

from agent import ScriptedBackend, run_turn
from agent.memory import resolve_followup, said_strikes, whatif_followup
from agent.slots import speakable
from mockmarket import World


def test_hundred_strikes_said():
    assert said_strikes("show me the six hundred calls on Netflix") == [600.0]
    assert said_strikes("Amazon puts around two twenty for next week") == [220.0]
    assert said_strikes("Google calls at 250") == [250.0]
    assert said_strikes("sell a hundred shares of Tesla") == []
    assert said_strikes("buy Intel calls at 3.50 limit") == []
    assert said_strikes("what if the market falls twelve percent") == []


def test_chain_strike_from_words():
    args, filled = resolve_followup("get_option_chain", {"symbol": "NFLX", "side": "call"},
                                    "the nine hundred calls on Netflix", [])
    assert args["strike"] == 900 and "said:strike" in filled


def test_orders_day_and_portfolio_rank_said():
    a, _ = resolve_followup("get_orders", {"status": "filled"}, "anything get filled today?", [])
    assert a["day"] == "today"
    a, _ = resolve_followup("get_portfolio", {"rank_by": "pnl"}, "how am I doing in percent", [])
    assert a["rank_by"] == "pnl_pct"
    a, _ = resolve_followup("get_portfolio", {"rank_by": "pnl"}, "what's my best position", [])
    assert a["rank_by"] == "pnl"                      # nothing named: the model's choice stands


def _whatif_turn(w, q="what if Nvidia falls 10 percent"):
    be = ScriptedBackend([{"tool_calls": [{"tool": "what_if", "args": {"symbol": "NVDA", "move_pct": -10}}]},
                          {"answer": "That would change your book by {pnl_change}."}])
    return run_turn(q, [], w, be)


def test_whatif_followup_reruns_in_code():
    w = World(3)
    first = _whatif_turn(w)
    hist = [first.turn]
    assert whatif_followup(hist, "and twenty") == {"symbol": "NVDA", "move_pct": -20}
    assert whatif_followup(hist, "what about if it rallies fifteen percent") == {"symbol": "NVDA", "move_pct": 15}
    assert whatif_followup(hist, "and the puts") is None
    be = ScriptedBackend([{"answer": "That would change your book by {pnl_change}."}])
    r = run_turn("and thirty", hist, w, be)
    assert [(c["tool"], c["args"]["move_pct"]) for c in r.tool_calls] == [("what_if", -30)]


def test_ticket_id_not_spoken():
    w = World(5)
    be = ScriptedBackend([
        {"tool_calls": [{"tool": "draft_order", "args": {"symbol": "MSFT", "direction": "buy", "qty": 3}}]},
        {"answer": "I've drafted a buy for three shares of Microsoft as {ticket_id}, awaiting your confirmation."}])
    r = run_turn("buy three shares of Microsoft", [], w, be)
    assert "ticket" not in r.spoken.lower() or "the ticket" in r.spoken.lower()
    assert " as " not in r.spoken and "awaiting your confirmation" in r.spoken
    assert any(ch.isdigit() for ch in r.display)      # the screen still shows the id


def test_ticket_id_phrasings():
    from agent.slots import fill

    class Book:
        def spoken(self, name):
            return "T one two"

    for t, want in (("Ticket {ticket_id} is ready to confirm.", "The ticket is ready to confirm."),
                    ("Your ticket {ticket_id} needs confirmation.", "The ticket needs confirmation."),
                    ("I drafted {ticket_id}.", "I drafted the ticket."),
                    ("Drafted ticket {ticket_id} to buy one put.", "Drafted the ticket to buy one put.")):
        assert fill(t, Book(), "spoken") == want


def test_times_times():
    assert speakable("volume is one point three times times average", World(1)) == \
        "volume is one point three times average"


def test_companion_calls_phrase_anchored():
    from agent.loop import companion_calls
    q = [{"tool": "get_quote", "args": {"symbol": "ORCL"}}]
    assert companion_calls("what's driving oracle today", q)[1] == "get_news"
    assert companion_calls("why is Netflix down so much", q)[0][-1] == {"tool": "get_news", "args": {"symbol": "ORCL"}}
    assert companion_calls("what's Oracle trading at", q)[1] is None            # plain quote: untouched
    both = q + [{"tool": "get_news", "args": {"symbol": "ORCL"}}]
    assert companion_calls("why is Oracle up", both)[1] is None                 # already a pair
    two = q + [{"tool": "get_quote", "args": {"symbol": "AMD"}}]
    assert companion_calls("why is Oracle up and AMD down", two)[1] is None     # two symbols: leave it
    ind = [{"tool": "get_indicators", "args": {"symbol": "APP", "timeframe": "1D"}}]
    assert companion_calls("walk me through the technicals on it", ind)[1] == "get_levels"
    assert companion_calls("what's the RSI on AppLovin", ind)[1] is None


def test_companion_call_runs_in_loop():
    w = World(4)
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}}]},
                          {"answer": "Nvidia is at {price}."}])
    r = run_turn("why is Nvidia moving today", [], w, be)
    assert [c["tool"] for c in r.tool_calls] == ["get_quote", "get_news"]


def test_strike_first_and_one_cent():
    from agent.slots import strike_first
    t = "Costco puts expiring {expiry} have high IV. The {strike_2} put is {bid_2} bid, {ask_2} ask."
    assert strike_first(t) == "The {strike_2} put is {bid_2} bid, {ask_2} ask. Costco puts expiring {expiry} have high IV."
    assert strike_first("The {strike_1} call is {bid_1} bid. IV is high.") == "The {strike_1} call is {bid_1} bid. IV is high."
    assert strike_first("Nvidia is at {price}. It is up today.") == "Nvidia is at {price}. It is up today."
    assert speakable("the call is one cents bid, three cents ask", World(1)) == "the call is one cent bid, three cents ask"


def _market_turn(w, q="how are stocks doing"):
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_market", "args": {"focus": "movers"}}]},
                          {"answer": "The market is {index_move} today."}])
    return run_turn(q, [], w, be)


def test_recovery_routes():
    from agent.loop import recovery_route
    w = World(2)
    hist = [_market_turn(w).turn]
    assert recovery_route("is anything holding up", hist, w) == ("get_market", {"focus": "breadth"})
    assert recovery_route("did my positions stop bleeding", [], w) == ("get_portfolio", {"rank_by": "day_move"})
    assert recovery_route("anything bouncing on my watchlist", [], w) == ("screen_watchlist", {"condition": "oversold"})
    assert recovery_route("anything green", [], w) is None                         # no context: the model
    assert recovery_route("is Tesla holding up", [], w) is None                    # a company named: the model's
    assert recovery_route("what's up", [], w) is None
    be = ScriptedBackend([{"answer": "Breadth is {index_move}."}])
    r = run_turn("anything that's stopped falling", hist, w, be)
    assert [(c["tool"], c["args"]) for c in r.tool_calls] == [("get_market", {"focus": "breadth"})]


def test_same_answer_to_a_new_question_is_not_reread():
    from agent.loop import SAME_AGAIN
    w = World(2)
    first = _market_turn(w)
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_market", "args": {"focus": "movers"}}]},
                          {"answer": "The market is {index_move} today."}])
    r = run_turn("and overall", [first.turn], w, be)
    assert r.spoken == SAME_AGAIN + first.spoken
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_market", "args": {"focus": "movers"}}]},
                          {"answer": "The market is {index_move} today."}])
    r = run_turn("how are stocks doing", [first.turn], w, be)          # the same question again: answer again
    assert r.spoken == first.spoken


def test_trailing_direction_after_percent():
    assert speakable("the index down zero point two percent down.", World(1)) == "the index down zero point two percent."
    assert speakable("up three percent today", World(1)) == "up three percent today"


def test_video_polish():
    w = World(6)
    r = run_turn("should I buy Tesla?", [], w, ScriptedBackend([]))
    assert not r.tool_calls and "Tesla" in r.spoken and "can't tell you" in r.spoken
    assert speakable("down nine percent for down about seven thousand dollars", w) == \
        "down nine percent for a loss of about seven thousand dollars"
    be = ScriptedBackend([
        {"tool_calls": [{"tool": "draft_order", "args": {"symbol": "TSLA", "direction": "buy", "qty": 2, "side": "put",
                                                          "strike": 430, "expiry_rule": "next_week"}}]},
        {"answer": "I've drafted two of the second strike puts as {ticket_id}, awaiting your confirmation."}])
    r = run_turn("buy two of the second strike", [], w, be)
    assert "four thirty put" in r.spoken and "second strike" not in r.spoken and not re.search(r"\d", r.spoken)
    from agent.slots import fill

    class Book:
        def spoken(self, name):
            return "four forty"

    assert fill("Most open interest is row four at {strike_4}.", Book(), "spoken") == "Most open interest is at four forty."


def test_duplicate_call_runs_once():
    w = World(7)
    call = {"tool": "get_option_chain", "args": {"symbol": "META", "side": "put"}}
    be = ScriptedBackend([{"tool_calls": [call, dict(call)]}, {"answer": "Meta puts expire {expiry}."}])
    r = run_turn("meta puts", [], w, be)
    assert [c["tool"] for c in r.tool_calls] == ["get_option_chain"]


def test_expiry_judgment_wording():
    from agent.slots import fill

    class Book:
        def spoken(self, name):
            return "November sixth"

    assert fill("Tesla puts expiring {expiry} are soon with high IV.", Book(), "spoken") == \
        "Tesla puts expiring November sixth are near-dated with high IV."
    assert fill("It is soon to tell.", Book(), "spoken") == "It is soon to tell."
