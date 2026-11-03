"""Night fixes (Nov 2): spoken 'hundred' strikes, day / ranking said, what_if follow-up in code,
ticket ids never read aloud, 'times times'. ScriptedBackend only; no model loads."""
from __future__ import annotations

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
