import json
import re

from agent import ScriptedBackend, Turn, parse_glimmer, run_turn
from agent.loop import FALLBACK_DATA, FALLBACK_TICKET
from agent.slots import SlotBook
from agent.validator import check_answer, validate_call
from mockmarket import World


def run(script, q="How's my NVDA position and is it near support?", history=None, world=None, **kw):
    w = world or World(7)
    be = ScriptedBackend(script)
    ev = []
    res = run_turn(q, history or [], w, be, on_event=ev.append, **kw)
    return res, ev, be, w


def types(ev):
    return [e["type"] for e in ev]


def test_multi_step_turn():
    res, ev, be, _ = run([
        {"tool_calls": [{"tool": "get_position", "args": {"symbol": "NVDA"}}]},
        {"tool_calls": [{"tool": "get_levels", "args": {"symbol": "NVDA"}}]},
        {"answer": "You're {pnl} on Nvidia. It is {dist_support} above support at {support}."},
    ])
    assert types(ev) == ["tool_call", "filler", "tool_result", "tool_call", "tool_result", "answer"]
    assert ev[1]["text"] == "Checking your Nvidia position."
    assert not res.fallback
    assert "$" in res.display and not re.search(r"\d", res.spoken)
    assert ev[0]["step"] == 1 and ev[3]["step"] == 2
    # the model saw only slot names, never digits
    tool_msgs = [m for m in be.seen[-1] if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    for m in tool_msgs:
        assert not re.search(r"\d", re.sub(r"\{[a-z_0-9]+\}", "", m["content"]))
    assert res.turn.symbol == "NVDA" and res.turn.topic == "levels"


def test_parallel_calls_and_unique_slot_names():
    res, ev, be, _ = run([
        {"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}},
                        {"tool": "get_quote", "args": {"symbol": "AMD"}}]},
        {"answer": "Nvidia is at {price} and AMD is at {price_2}."},
    ], q="Compare NVDA and AMD")
    res_events = [e for e in ev if e["type"] == "tool_result"]
    assert "price" in res_events[0]["slots_display"] and "price_2" in res_events[1]["slots_display"]
    views = [json.loads(m["content"]) for m in be.seen[-1] if m["role"] == "tool"]
    assert "{price_2}" in views[1]["slots"]
    assert not res.fallback
    assert res.slots["price"] != res.slots["price_2"]


def test_follow_up_uses_history():
    w = World(7)
    r1, _, _, _ = run([{"tool_calls": [{"tool": "get_position", "args": {"symbol": "NVDA"}}]},
                       {"answer": "You're {pnl} on Nvidia."}], world=w)
    seen = {}

    def second(messages):
        seen["messages"] = messages
        return {"tool_calls": [{"tool": "get_indicators", "args": {"symbol": "NVDA", "indicator": "rsi"}}]}

    r2, ev, be, _ = run([second, {"answer": "RSI is {rsi}, which is neutral."}], q="And its RSI?",
                        history=[r1.turn], world=w)
    msgs = be.seen[0]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "last_symbol=NVDA" in msgs[-1]["content"] and "last_topic=position" in msgs[-1]["content"]
    assert "{" not in msgs[2]["content"]                       # old slots don't leak into history
    assert not r2.fallback


def test_history_capped_at_three():
    hist = [Turn(f"q{i}", "ok", symbol="AAPL", topic="quote") for i in range(5)]
    _, _, be, _ = run([{"answer": "Sure."}], history=hist)
    assert sum(1 for m in be.seen[0] if m["role"] == "assistant") == 3


def test_bad_tool_gets_one_retry():
    res, ev, be, _ = run([
        {"tool_calls": [{"tool": "get_price", "args": {"symbol": "NVDA"}}]},
        {"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}}]},
        {"answer": "Nvidia is at {price}."},
    ])
    errs = [e for e in ev if e["type"] == "error"]
    assert errs and errs[0]["message"] == "unknown tool"
    err_view = json.loads([m for m in be.seen[1] if m["role"] == "tool"][0]["content"])
    assert err_view["ok"] is False and err_view["error"] == "unknown tool"
    assert not res.fallback and "$" in res.display
    assert types(ev).count("filler") == 1


def test_two_bad_tools_fall_back():
    res, ev, _, _ = run([
        {"tool_calls": [{"tool": "get_price", "args": {}}]},
        {"tool_calls": [{"tool": "get_quote", "args": {"sym": "NVDA"}}]},
        {"answer": "never reached"},
    ], q="How is it doing and is it near support?")   # no company named: code cannot repair the missing symbol
    assert res.fallback
    assert ev[-1]["type"] == "answer"


def test_bad_args_and_symbol_resolution():
    w = World(7)
    assert validate_call(w, "get_quote", {}).error == "missing symbol"
    assert validate_call(w, "get_indicators", {"symbol": "NVDA", "timeframe": "weekly"}).ok is False
    c = validate_call(w, "draft_order", {"symbol": "nvda", "direction": "BUY", "qty": "5"})
    assert c.ok and c.args == {"symbol": "NVDA", "direction": "buy", "qty": 5, "type": "market"}
    c = validate_call(w, "get_quote", {"symbol": "Nvidia"})
    assert c.ok and c.args["symbol"] == "NVDA" and c.resolved
    c = validate_call(w, "get_quote", {"symbol": "QWERTY"})
    assert not c.ok and c.error == "unknown symbol"
    assert validate_call(w, "what_if", {"symbol": "ALL", "move_pct": "-10%"}).args["move_pct"] == -10.0


def test_digits_in_answer_retry_then_ok():
    res, ev, be, _ = run([
        {"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}}]},
        {"answer": "Nvidia is at 182 dollars."},
        {"answer": "Nvidia is at {price}."},
    ])
    assert [e["stage"] for e in ev if e["type"] == "error"] == ["answer"]
    assert not res.fallback
    feedback = be.seen[-1][-1]["content"]
    assert "digits" in feedback and "{price}" in feedback
    assert not re.search(r"\d", be.seen[-1][-2]["content"])     # bad answer echoed back without digits


def test_digits_twice_fallback():
    res, ev, _, _ = run([
        {"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}}]},
        {"answer": "Nvidia is at 182."},
        {"answer": "Nvidia is at {nope}."},
    ])
    assert res.fallback and res.answer_template == FALLBACK_DATA
    assert ev[-1]["type"] == "answer" and ev[-1]["fallback"] is True


def test_answer_checks():
    names = {"price", "pnl"}
    assert check_answer("At {price}.", names) == []
    assert "digits outside slots" in check_answer("Up 5 percent.", names)
    assert any("unknown slots" in p for p in check_answer("At {rsi}.", names))
    assert "more than two sentences" in check_answer("One. Two. Three.", names)
    assert check_answer("", names) == ["empty answer"]


def test_draft_order_never_sends():
    w = World(4)
    before = [o.__dict__.copy() for o in w.orders]
    res, ev, _, _ = run([
        {"tool_calls": [{"tool": "draft_order", "args": {"symbol": "NVDA", "direction": "buy", "qty": 10}}]},
        {"answer": "Done, I placed the order."},
        {"answer": "I sent it."},
    ], q="Buy ten Nvidia", world=w)
    tickets = [e for e in ev if e["type"] == "ticket"]
    assert len(tickets) == 1 and tickets[0]["status"] == "needs confirmation" and tickets[0]["qty"] == 10
    assert res.ticket["ticket_id"] in w.tickets
    assert res.fallback and res.answer_template == FALLBACK_TICKET
    assert [o.__dict__ for o in w.orders] == before


def test_draft_order_ok_answer():
    res, ev, _, _ = run([
        {"tool_calls": [{"tool": "draft_order", "args": {"symbol": "NVDA", "direction": "buy", "qty": 10}}]},
        {"answer": "I drafted a buy ticket for Nvidia; confirm to send it."},
    ], q="Buy ten Nvidia")
    assert not res.fallback and res.ticket


def test_max_steps_forces_answer():
    loop_call = {"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}}]}
    res, ev, be, _ = run([loop_call] * 3 + [loop_call, {"answer": "Nvidia is at {price_3}."}], max_steps=3)
    assert "No more tool calls" in be.seen[-1][-1]["content"]
    assert not res.fallback


def test_parse_error_retry():
    res, ev, _, _ = run([
        {"error": "could not parse tool call", "raw": "<atem:function_calls>garbage"},
        {"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}}]},
        {"answer": "Nvidia is at {price}."},
    ])
    assert ev[0]["type"] == "error" and ev[0]["stage"] == "parse"
    assert not res.fallback


def test_tool_error_result_is_not_a_retry():
    res, ev, _, _ = run([
        {"tool_calls": [{"tool": "get_portfolio", "args": {}}]},
        {"answer": "Your portfolio is empty."},
    ], world=World(1, "empty_portfolio"), q="How's my portfolio?")
    tr = [e for e in ev if e["type"] == "tool_result"][0]
    assert tr["ok"] is False and tr["error"] == "empty portfolio"
    assert not res.fallback


def test_slotbook_suffixes():
    b = SlotBook()
    assert b.add("get_quote", {"price": 1.0}) == {"price": "price"}
    assert b.add("get_quote", {"price": 2.0}) == {"price": "price_2"}
    assert b.add("get_quote", {"price": 3.0}) == {"price": "price_3"}


def test_parse_glimmer_formats():
    tc = (' to=get_quote<|message|><atem:function_calls>\n<atem:invoke name="get_quote">\n'
          '<atem:parameter name="symbol">NVDA</atem:parameter>\n</atem:invoke>\n</atem:function_calls>')
    assert parse_glimmer(tc) == {"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}}]}
    two = (' to=self<|message|>thinking...<|eom|><|start|>assistant to=what_if<|message|><atem:function_calls>\n'
           '<atem:invoke name="functions.what_if">\n<atem:parameter name="symbol">ALL</atem:parameter>\n'
           '<atem:parameter name="move_pct">-10</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>')
    assert parse_glimmer(two) == {"tool_calls": [{"tool": "what_if", "args": {"symbol": "ALL", "move_pct": -10}}]}
    assert parse_glimmer(" to=user<|message|>Nvidia is at {price}.<|eot|>") == {"answer": "Nvidia is at {price}."}
    assert parse_glimmer("<|message|>Hi there.") == {"answer": "Hi there."}
    assert "error" in parse_glimmer(" to=get_quote<|message|><atem:function_calls><atem:inv")
    assert parse_glimmer(' to=get_quote<|message|>{"symbol": "AAPL"}<|eom|>') == {
        "tool_calls": [{"tool": "get_quote", "args": {"symbol": "AAPL"}}]}


def test_spoken_direction_not_doubled():
    from agent.slots import speakable
    w = World(1)
    assert speakable("NVDA is up up zero point seven percent.", w) == "Nvidia is up zero point seven percent."
    assert speakable("It is up down two percent.", w) == "It is down two percent."
