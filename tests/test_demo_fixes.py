"""Demo findings 2026-10-27: wrong symbol (OR for Oracle), strike-first chain answers, ungrounded fallback,
slot binding, labelled chain numbers."""
import re

from agent import ScriptedBackend, Turn, run_turn
from agent.loop import CLARIFY_UNGROUNDED, FALLBACK_DATA
from agent.slots import SlotBook, fill, label_chain
from agent.validator import binding_problems, correct_symbol, named_companies, rename_companies, tickers_in
from mockmarket import World
from tools import run_tool
from tools.render import spoken_kind

W = World(7)


def run(script, q, history=None):
    ev = []
    res = run_turn(q, history or [], W, ScriptedBackend(script), on_event=ev.append)
    return res, ev


def chain_call(sym="ORCL", rule="nearest"):
    return {"tool_calls": [{"tool": "get_option_chain", "args": {"symbol": sym, "expiry_rule": rule, "side": "call"}}]}


# ---------------------------------------------------------------- 1. symbol correction
def test_names_and_tickers_in_text():
    assert {s for s, _, _ in named_companies("I mean, Oracle.", W)} == {"ORCL"}
    assert {s for s, _, _ in named_companies("show me O'Reilly calls", W)} == {"ORLY"}
    assert {s for s, _, _ in named_companies("the nvidia leveraged ETF", W)} == {"NVDL"}
    assert named_companies("buy a block of shares", W) == []          # common word, lowercase
    assert {s for s, _, _ in tickers_in("ticker OR please", W)} == {"OR"}
    assert {s for s, _, _ in tickers_in("show me O R calls", W)} == {"OR"}
    assert tickers_in("calls or puts", W) == []


def test_correct_symbol_rules():
    # stale state carry-over: last turn was OR, the user now names Oracle
    assert correct_symbol(W, "OR", "I mean, Oracle.", last_symbol="OR") == "ORCL"
    # lookalikes even without state
    assert correct_symbol(W, "OR", "show me Oracle options") == "ORCL"
    assert correct_symbol(W, "MET", "how's Meta doing") == "META"
    assert correct_symbol(W, "NVDL", "Nvidia calls") == "NVDA"
    assert correct_symbol(W, "ORCL", "O'Reilly calls") == "ORLY"
    # explicit tickers are never overridden
    assert correct_symbol(W, "OR", "Oracle or the ticker OR?") is None
    assert correct_symbol(W, "OR", "show me O R and Oracle") is None
    # unrelated symbol, not stale: left alone (e.g. SPY next to a company)
    assert correct_symbol(W, "SPY", "is Apple beating the S and P") is None
    assert correct_symbol(W, "SPY", "is Apple beating the market") is None
    # already covered by another call this turn
    assert correct_symbol(W, "NVDA", "compare it with AMD", last_symbol="NVDA", used={"AMD"}) is None
    # nothing named: follow-ups keep the model's symbol
    assert correct_symbol(W, "OR", "what about puts", last_symbol="OR") is None


def test_loop_corrects_oracle_after_or_turn():
    hist = [Turn("Show me the OR calls", "The OR calls expire {expiry}.", symbol="OR", topic="options")]
    res, ev = run([chain_call("OR"), {"answer": "Oracle calls expire {expiry}; strikes run {strike_1} to {strike_5}."}],
                  "I mean, Oracle.", hist)
    call = next(e for e in ev if e["type"] == "tool_call")
    assert call["args"]["symbol"] == "ORCL" and call["corrected_from"] == "OR"
    assert res.tool_calls[0]["args"]["symbol"] == "ORCL" and res.turn.symbol == "ORCL"
    # the chain is Oracle's (weeklies: nearest expiry is this week's, not OR's monthly)
    assert res.slots["expiry"] == run_tool(W, "get_option_chain", {"symbol": "ORCL"})["slots"]["expiry"]
    assert not res.fallback and res.spoken.startswith("Oracle calls")


def test_loop_keeps_explicit_ticker():
    res, ev = run([chain_call("OR"), {"answer": "Osisko calls expire {expiry}."}], "I said the ticker OR, not Oracle")
    assert res.tool_calls[0]["args"]["symbol"] == "OR"
    assert "corrected_from" not in next(e for e in ev if e["type"] == "tool_call")


def test_answer_company_name_comes_from_tool_symbol():
    # the model names a company the data is not about: the name is replaced with the tool symbol's name
    assert rename_companies("O'Reilly calls expire {expiry}.", W, {"ORCL"}, "ORCL") == "Oracle calls expire {expiry}."
    assert rename_companies("ORLY calls expire {expiry}.", W, {"ORCL"}, "ORCL") == "Oracle calls expire {expiry}."
    assert rename_companies("Oracle is at {price}.", W, {"ORCL"}, "ORCL") == "Oracle is at {price}."
    res, _ = run([{"tool_calls": [{"tool": "get_quote", "args": {"symbol": "OR"}}]},
                  {"answer": "Oracle is at {price}."}], "how's OR")
    assert res.answer_template == "Osisko Gold Royalties is at {price}." and not res.fallback


# ---------------------------------------------------------------- 2. strike first
def _chain_book(n_runs=1):
    b = SlotBook()
    for _ in range(n_runs):
        r = run_tool(W, "get_option_chain", {"symbol": "TSLA"})
        b.add("get_option_chain", r["slots"], r["words"], {"symbol": "TSLA"})
    return b


def test_strike_first_rule_unit():
    b = _chain_book()
    bad = binding_problems("Tesla calls are {bid_1} bid, {ask_1} ask.", b, W)
    assert bad and bad[0].startswith("say the strike first") and "{strike_1}" in bad[0]
    assert binding_problems("Tesla calls: {bid_1} bid, {ask_1} ask at the {strike_1} strike.", b, W)
    assert binding_problems("Strikes run {strike_1} to {strike_5}; {bid_3} by {ask_3}.", b, W)   # wrong row
    assert binding_problems("{strike_1} is {bid_2} bid.", b, W)
    assert binding_problems("The {strike_1} strike is {bid_1} bid, {ask_1} ask, delta {delta_1}.", b, W) == []
    assert binding_problems("Strikes run {strike_1} to {strike_5}; {strike_3} is {bid_3} by {ask_3}.", b, W) == []
    assert binding_problems("Tesla calls expire {expiry}, {strike_1} through {strike_5}.", b, W) == []
    b2 = _chain_book(2)                                  # second run's rows carry the _2 suffix
    assert binding_problems("{strike_1} is {bid_1_2}.", b2, W)
    assert binding_problems("{strike_1_2} is {bid_1_2}.", b2, W) == []


def test_strike_first_retry_then_ok():
    res, ev = run([chain_call("TSLA"), {"answer": "Tesla calls are {bid_1} bid, {ask_1} ask."},
                   {"answer": "The {strike_1} strike is {bid_1} bid, {ask_1} ask."}], "Tesla calls")
    errs = [e for e in ev if e["type"] == "error"]
    assert len(errs) == 1 and "strike first" in errs[0]["message"]
    assert not res.fallback and res.answer_template.startswith("The {strike_1} strike")


def test_strike_first_fallback_is_code_built():
    res, _ = run([chain_call("TSLA"), {"answer": "Tesla calls are {bid_2} bid, {ask_2} ask."},
                  {"answer": "Still {bid_2} bid, {ask_2} ask at {strike_2}."}], "Tesla calls")
    assert res.fallback
    t = res.answer_template
    assert t.index("{strike_2}") < t.index("{bid_2}") < t.index("{ask_2}") and t.startswith("For Tesla calls")
    assert not re.search(r"\d", res.spoken) and "$" in res.display


def test_normal_chain_answer_costs_no_retry():
    res, ev = run([chain_call("TSLA"), {"answer": "Tesla calls expire {expiry}; {strike_1} is {bid_1} by {ask_1}."}],
                  "Tesla calls")
    assert not [e for e in ev if e["type"] == "error"] and not res.fallback
    assert len([1 for _ in ev if _["type"] == "answer"]) == 1


# ---------------------------------------------------------------- A. ungrounded answers
def test_ungrounded_answer_becomes_clarify():
    res, ev = run([{"answer": "Your portfolio is {day_pnl} and AppLovin is {day_move} at {price}."},
                   {"answer": "Your portfolio is down today, with AppLovin down at its current price."}],
                  "What is stop bleeding right now?")
    assert res.answer_template == CLARIFY_UNGROUNDED and res.fallback and not res.tool_calls
    retry = [m for m in res.events if m["type"] == "error"]
    assert retry and "unknown slots" in retry[0]["message"]


def test_ungrounded_retry_asks_for_a_tool_and_can_recover():
    be = ScriptedBackend([{"answer": "You're {day_pnl} today."},
                          {"tool_calls": [{"tool": "get_portfolio", "args": {}}]},
                          {"answer": "You're {day_pnl} today."}])
    res = run_turn("How am I doing?", [], W, be)
    retry_msg = be.seen[1][-1]["content"]
    assert retry_msg.startswith("You have no tool results yet") and "Call the tool" in retry_msg
    assert not res.fallback and res.answer_template == "You're {day_pnl} today."


def test_ungrounded_retry_may_ask_a_question():
    res, _ = run([{"answer": "AppLovin is at {price}."}, {"answer": "Do you mean AppLovin's price?"}], "app?")
    assert res.answer_template == "Do you mean AppLovin's price?"


# ---------------------------------------------------------------- B. slot binding
def _market_book():
    b = SlotBook()
    r = run_tool(W, "get_market", {"focus": "movers"})
    b.add("get_market", r["slots"], r["words"], {"focus": "movers"})
    return b, r["words"]


def test_binding_unit_market():
    b, w = _market_book()
    top, bottom = W.universe[w["top_name"]].name, W.universe[w["bottom_name"]].name
    assert binding_problems(f"{top} is the top mover today, up {{index_move}}.", b, W)
    assert binding_problems(f"{top} leads at {{bottom_move}}.", b, W)
    assert binding_problems(f"{top} is the top mover at {{top_move}}.", b, W) == []
    assert binding_problems(f"{w['top_name']} leads at {{top_move}}, {bottom} trails at {{bottom_move}}.", b, W) == []
    assert binding_problems("The market is down, with indexes at {index_move}.", b, W) == []
    assert binding_problems(f"The market is {{index_move}} while {top} leads at {{top_move}}.", b, W) == []
    assert binding_problems("It moved {index_move}.", b, W) == []          # nothing named: not checked


def test_binding_unit_symbol_tools():
    b = SlotBook()
    for s in ("NVDA", "AMD"):
        r = run_tool(W, "get_quote", {"symbol": s})
        b.add("get_quote", r["slots"], r["words"], {"symbol": s})
    assert binding_problems("Nvidia is at {price} and AMD is at {price_2}.", b, W) == []
    assert binding_problems("Nvidia is at {price_2}.", b, W)


def test_binding_retry_then_safe_line():
    w = run_tool(W, "get_market", {"focus": "movers"})["words"]
    top = W.universe[w["top_name"]].name
    res, ev = run([{"tool_calls": [{"tool": "get_market", "args": {"focus": "movers"}}]},
                   {"answer": f"{top} is the top mover today, up {{index_move}}."},
                   {"answer": f"{top} is the top mover today, up {{index_move}} again."}],
                  "Which names are leading the market right now?")
    assert [e for e in ev if e["type"] == "error" and "wrong name" in e["message"]]
    assert res.fallback and res.answer_template == FALLBACK_DATA


def test_binding_retry_fixed():
    w = run_tool(W, "get_market", {"focus": "movers"})["words"]
    top = W.universe[w["top_name"]].name
    res, _ = run([{"tool_calls": [{"tool": "get_market", "args": {"focus": "movers"}}]},
                  {"answer": f"{top} is the top mover today, up {{index_move}}."},
                  {"answer": f"{top} is the top mover today at {{top_move}}."}], "movers?")
    assert not res.fallback and "{top_move}" in res.answer_template


# ---------------------------------------------------------------- C. chain wording
def test_label_chain_and_spoken_numbers():
    t = "Row two at {strike_2} with {bid_2} by {ask_2} and {delta_2} at {iv_2}."
    assert label_chain(t) == "Row two at {strike_2} with bid {bid_2}, ask {ask_2} and delta {delta_2}, IV {iv_2}."
    assert label_chain("{strike_1} at {bid_1} bid, {ask_1} ask, delta of {delta_1}, IV at {iv_1}.") == \
        "{strike_1} at {bid_1} bid, {ask_1} ask, delta of {delta_1}, IV at {iv_1}."
    assert spoken_kind("greek", 0.59) == "point five nine"
    assert spoken_kind("vol", 0.581) == "fifty-eight percent"
    assert spoken_kind("ratio", 1.0) == "about average"
    b = _chain_book()
    sp = fill(t, b, "spoken")
    assert " bid " in sp and ", ask " in sp and "delta point" in sp and ", IV " in sp and not re.search(r"\d", sp)


def test_chain_fallback_keeps_iv_when_asked():
    res, _ = run([chain_call("TSLA"), {"answer": "Tesla IV is {iv_3} at the {strike_3} call."},
                  {"answer": "Tesla IV is {iv_3} at {strike_3}."}], "where's implied vol on tsla")
    assert res.fallback and "{strike_3} strike" in res.answer_template and "IV {iv_3}" in res.answer_template


def test_failed_tool_then_ungrounded_is_not_a_clarify():
    res, _ = run([{"tool_calls": [{"tool": "get_indicators", "args": {"symbol": "AAPL", "indicator": "macd"}}]},
                  {"answer": "Apple RSI is {rsi}."}, {"answer": "Apple RSI is {rsi}."}], "Apple technicals")
    assert res.fallback and res.answer_template != CLARIFY_UNGROUNDED


def test_ticket_answer_must_match_ticket_kind_and_side():
    from agent.loop import ticket_problems
    assert ticket_problems("I've drafted five Apple call tickets.", {"kind": "stock"})
    assert not ticket_problems("I've drafted five Apple shares.", {"kind": "stock"})
    assert not ticket_problems("Two Nvidia call contracts drafted.", {"kind": "option", "side": "call"})
    assert ticket_problems("Two Nvidia puts drafted.", {"kind": "option", "side": "call"})


def test_age_slot_does_not_repeat_ago():
    from agent.slots import fill

    class Book:
        def display(self, n): return "2h ago"
        def spoken(self, n): return "two hours ago"
    assert fill("A story from {age_1} ago.", Book(), "spoken") == "A story from two hours ago."
