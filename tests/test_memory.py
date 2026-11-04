"""Conversation memory (agent/memory.py): frames, follow-up resolution, result reuse, commands, pinned facts,
persistence across reconnects. ScriptedBackend dialogues; no model loads."""
from __future__ import annotations

import json
import re
import time

from fastapi.testclient import TestClient

from agent import ScriptedBackend, run_turn
from agent.loop import ASK_QTY, build_messages
from agent.memory import (STALE_S, Conversation, SessionStore, call_text, memory_command, parse_fact,
                          session_key)
from mockmarket import World
from tests.test_voice_models import make, quote_then_answer, registry  # noqa: F401  (fixture)
from tests.test_voice_server import collect


def chain_turn(w, history=None, q="tesla calls for next week"):
    be = ScriptedBackend([
        {"tool_calls": [{"tool": "get_option_chain", "args": {"symbol": "TSLA", "expiry_rule": "next_week",
                                                              "side": "call"}}]},
        {"answer": "Tesla calls expiring {expiry}: the {strike_2} strike is {bid_2} bid, {ask_2} ask."}])
    return run_turn(q, history or [], w, be)


def test_frame_is_filled_by_code():
    w = World(7)
    r = chain_turn(w)
    fr = r.turn.frame
    assert fr["calls"] == [{"tool": "get_option_chain", "args": r.tool_calls[0]["args"]}]
    assert fr["symbols"] == ["TSLA"]
    assert fr["option"]["side"] == "call" and fr["option"]["expiry_rule"] == "next_week"
    assert fr["option"]["said_strike"] == fr["chain_strikes"][2] == r.slots["strike_2"]
    assert "strike_2" in fr["slots"] and fr["answer"].startswith("Tesla calls")


def test_call_text_is_compact_and_stable():
    assert call_text("get_option_chain", {"symbol": "TSLA", "side": "call", "expiry_rule": "next_week"}) == \
        "get_option_chain(expiry_rule=next_week, side=call, symbol=TSLA)"
    # schema defaults the validator filled in are dropped; integral floats print as ints
    assert call_text("get_indicators", {"symbol": "NVDA", "timeframe": "1D"}) == "get_indicators(symbol=NVDA)"
    assert call_text("draft_order", {"symbol": "APP", "direction": "buy", "qty": 2, "strike": 650.0,
                                     "type": "market"}) == "draft_order(direction=buy, qty=2, strike=650, symbol=APP)"


def test_state_line_with_frame_only_for_frame_models():
    w = World(7)
    r = chain_turn(w)
    plain = build_messages("and the puts?", [r.turn])[-1]["content"]
    framed = build_messages("and the puts?", [r.turn], facts=["name Minh"], frame_line=True)[-1]["content"]
    assert plain == "[state: last_symbol=TSLA, last_topic=options]\nand the puts?"
    assert framed == ("[state: last_symbol=TSLA, last_topic=options, "
                      "last_call=get_option_chain(expiry_rule=next_week, side=call, symbol=TSLA), "
                      "facts=name Minh]\nand the puts?")
    assert build_messages("hi", [], facts=["name Minh"], frame_line=True)[-1]["content"] == "hi"


def test_buy_two_of_those_fills_side_expiry_and_said_strike():
    w = World(7)
    r1 = chain_turn(w)
    be = ScriptedBackend([{"tool_calls": [{"tool": "draft_order", "args": {"symbol": "TSLA", "direction": "buy",
                                                                           "qty": 2}}]},
                          {"answer": "I drafted a ticket for two Tesla calls; confirm to send."}])
    ev = []
    r2 = run_turn("buy two of those", [r1.turn], w, be, on_event=ev.append)
    call = next(e for e in ev if e["type"] == "tool_call")
    assert sorted(call["filled_from_frame"]) == ["expiry_rule", "side", "strike"]
    assert call["args"]["strike"] == r1.slots["strike_2"] and call["args"]["side"] == "call"
    assert r2.ticket is not None and not r2.fallback


def test_the_second_strike_and_symbol_from_frame():
    w = World(7)
    r1 = chain_turn(w)
    be = ScriptedBackend([{"tool_calls": [{"tool": "draft_order", "args": {"direction": "buy", "qty": 1}}]},
                          {"answer": "Drafted; it needs your confirmation."}])
    ev = []
    run_turn("put in a ticket for one at the second strike", [r1.turn], w, be, on_event=ev.append)
    call = next(e for e in ev if e["type"] == "tool_call")
    assert call["args"]["symbol"] == "TSLA" and call["args"]["strike"] == r1.turn.frame["chain_strikes"][2]


def test_question_words_beat_the_frame():
    w = World(7)
    r1 = chain_turn(w)
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_option_chain", "args": {"symbol": "TSLA"}}]},
                          {"answer": "Here are the {strike_1} puts."}])
    ev = []
    run_turn("and the puts expiring monthly", [r1.turn], w, be, on_event=ev.append)
    call = next(e for e in ev if e["type"] == "tool_call")
    assert "filled_from_frame" not in call            # side and expiry were said: nothing carried over
    assert sorted(call["from_question"]) == ["expiry_rule", "side"]
    assert (call["args"]["side"], call["args"]["expiry_rule"]) == ("put", "monthly")


def test_strike_read_as_qty_asks_for_quantity():
    w = World(7)
    be = ScriptedBackend([{"tool_calls": [{"tool": "draft_order", "args": {"symbol": "APP", "direction": "buy",
                                                                           "qty": 650, "side": "call"}}]}])
    r = run_turn("draft me a ticket to buy the 650 call for next week", [], w, be)
    assert r.answer_template == ASK_QTY and r.ticket is None and not r.fallback and r.tool_calls == []


def test_fresh_result_is_reused_stale_one_is_not():
    w = World(7)
    be = quote_then_answer("Nvidia is at {price}.")
    r1 = run_turn("where's nvidia trading", [], w, be)
    ev = []
    run_turn("and now?", [r1.turn], w, be, on_event=ev.append)
    assert next(e for e in ev if e["type"] == "tool_result").get("cached") is True
    r1.turn.frame["ts"] -= STALE_S["get_quote"] + 1
    ev = []
    run_turn("and now?", [r1.turn], w, be, on_event=ev.append)
    assert "cached" not in next(e for e in ev if e["type"] == "tool_result")


def test_commands_and_facts_are_code():
    assert memory_command("Start over.") == "all" and memory_command("ok forget that") == "last"
    assert memory_command("sorry, could you repeat that") == "repeat" and memory_command("forget about Tesla puts") is None
    w = World(7)
    assert parse_fact("call me Minh", w) == ("name Minh", "Got it, Minh.")
    assert parse_fact("I mostly trade Nvidia and AMD", w)[0] == "trades mostly NVDA AMD"
    assert parse_fact("my name is 12345", w) is None
    assert parse_fact("I mostly trade with account 5512", w) is None


def _conv_dialogue(conv, w, script, questions):
    be = ScriptedBackend(script)
    out = []
    for q in questions:
        out.append(conv.ask(q, lambda q_, h, f: run_turn(q_, h, w, be, facts=f), world=w))
    return out, be


def test_conversation_persists_and_forgets(tmp_path):
    store, key = SessionStore(tmp_path / "s"), session_key("pw")
    w = World(7)
    conv = Conversation(store, key)
    (r1, r2, r3), be = _conv_dialogue(conv, w, [
        {"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}}]}, {"answer": "Nvidia is at {price}."},
        {"answer": "Nvidia it is."}], ["where's nvidia trading", "call me Minh", "say that again"])
    assert r2.spoken == "Got it, Minh." and r3.spoken == "Got it, Minh."     # repeat: code, no model call
    assert len(be.script) == 1                                                 # the model was not asked
    assert conv.facts == ["name Minh"] and [t.question for t in conv.turns] == ["where's nvidia trading", "call me Minh"]
    assert conv.turns[-1].symbol == "NVDA"          # a pinned fact keeps what "it" refers to
    raw = store.path(key).read_text()
    assert "pw" not in store.path(key).name and json.loads(raw)["facts"] == ["name Minh"]
    # a new connection (new Conversation) sees the same memory
    again = Conversation(store, key)
    assert [t.question for t in again.turns] == ["where's nvidia trading", "call me Minh"] and again.facts == ["name Minh"]
    assert again.turns[0].frame["calls"][0]["tool"] == "get_quote"
    ev = []
    again.ask("forget that", lambda *a: None, on_event=ev.append)
    assert ev[0] == {"type": "memory_cleared", "scope": "last"}
    assert again.facts == [] and [t.question for t in Conversation(store, key).turns] == ["where's nvidia trading"]
    again.ask("start over", lambda *a: None)
    assert not store.path(key).exists() and Conversation(store, key).turns == []


def test_retention_drops_old_turns(tmp_path):
    store, key = SessionStore(tmp_path / "s"), session_key("pw")
    conv = Conversation(store, key)
    _conv_dialogue(conv, World(7), [{"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}}]},
                                    {"answer": "Nvidia is at {price}."}], ["nvidia"])
    d = json.loads(store.path(key).read_text())
    d["turns"][0]["ts"] = d["turns"][0]["frame"]["ts"] = time.time() - 2 * 24 * 3600
    store.path(key).write_text(json.dumps(d))
    assert Conversation(store, key).turns == []


def test_secrets_are_redacted_in_the_store(tmp_path):
    store, key = SessionStore(tmp_path / "s"), session_key("pw")
    conv = Conversation(store, key)
    _conv_dialogue(conv, World(7), [{"answer": "I can't use account numbers."}],
                   ["my account is 4111 1111 1111 1111 mail me at a@b.co"])
    raw = store.path(key).read_text()
    assert "4111" not in raw and "a@b.co" not in raw


def test_memory_survives_reconnect_over_websocket(registry):  # noqa: F811
    client, _, backends = make(registry)
    with client:
        with client.websocket_connect("/ws?k=pw") as ws:
            assert ws.receive_json()["memory"] == {"turns": 0, "facts": 0}
            ws.send_json({"type": "text", "text": "How is Nvidia doing?"})
            collect(ws)
        with client.websocket_connect("/ws?k=pw") as ws:
            assert ws.receive_json()["memory"] == {"turns": 1, "facts": 0}
            ws.send_json({"type": "text", "text": "And now?"})
            collect(ws)
            msgs = backends["glimmer-base"].seen[-1]
            assert [m["role"] for m in msgs[:4]] == ["system", "user", "assistant", "user"]
            assert msgs[1]["content"] == "How is Nvidia doing?"
            ws.send_json({"type": "clear_memory"})
            assert ws.receive_json() == {"type": "memory_cleared", "scope": "all"}
        with client.websocket_connect("/ws?k=pw") as ws:
            assert ws.receive_json()["memory"] == {"turns": 0, "facts": 0}
            ws.send_json({"type": "text", "text": "start over"})
            events, _ = collect(ws)
            types = [e["type"] for e in events]
            assert "memory_cleared" in types and any(e.get("code") for e in events if e["type"] == "answer")
            assert not any(re.search(r"\d", e.get("spoken", "")) for e in events if e["type"] == "answer")


def test_ordinal_strike_overrides_a_guessed_strike():
    w = World(7)
    r1 = chain_turn(w)
    be = ScriptedBackend([{"tool_calls": [{"tool": "draft_order", "args": {"symbol": "TSLA", "direction": "buy",
                                                                           "qty": 1, "strike": 1.0}}]},
                          {"answer": "Drafted; it needs your confirmation."}])
    ev = []
    run_turn("get me one at the third strike", [r1.turn], w, be, on_event=ev.append)
    call = next(e for e in ev if e["type"] == "tool_call")
    assert call["args"]["strike"] == r1.turn.frame["chain_strikes"][3] and "strike" in call["filled_from_frame"]


def test_frame_side_beats_a_model_guess_when_the_user_said_none():
    w = World(7)
    r1 = chain_turn(w)                                   # TSLA calls, next week
    r1.turn.frame["option"]["side"] = "put"              # as if the user had heard puts
    be = ScriptedBackend([{"tool_calls": [{"tool": "draft_order", "args": {"symbol": "TSLA", "direction": "buy",
                                                                           "qty": 2, "side": "call"}}]},
                          {"answer": "Drafted; it needs your confirmation."}])
    ev = []
    run_turn("I want two from the second strike row", [r1.turn], w, be, on_event=ev.append)
    call = next(e for e in ev if e["type"] == "tool_call")
    assert call["args"]["side"] == "put" and "side" in call["filled_from_frame"]
    # another company, carried over ("same for AMD"): the chain's side and expiry, not the model's
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_option_chain", "args": {"symbol": "AMD", "side": "call",
                                                                                "expiry_rule": "nearest"}}]},
                          {"answer": "AMD {side} chain."}])
    ev = []
    run_turn("same for AMD", [r1.turn], w, be, on_event=ev.append)
    call = next(e for e in ev if e["type"] == "tool_call")
    assert (call["args"]["side"], call["args"]["expiry_rule"]) == ("put", "next_week")
    # another company, a fresh request: the model's own choice stands
    ev = []
    be.script = [{"tool_calls": [{"tool": "get_option_chain", "args": {"symbol": "AMD", "side": "call"}}]},
                 {"answer": "AMD chain."}]
    run_turn("show me the AMD chain", [r1.turn], w, be, on_event=ev.append)
    call = next(e for e in ev if e["type"] == "tool_call")
    assert call["args"]["side"] == "call"


def test_said_side_and_expiry_beat_the_model():
    w = World(7)
    r1 = chain_turn(w)
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_option_chain", "args": {"symbol": "TSLA", "side": "call",
                                                                                "expiry_rule": "nearest"}}]},
                          {"answer": "Tesla puts."}])
    ev = []
    run_turn("and the puts for the monthly?", [r1.turn], w, be, on_event=ev.append)
    call = next(e for e in ev if e["type"] == "tool_call")
    assert (call["args"]["side"], call["args"]["expiry_rule"]) == ("put", "monthly")


def test_share_orders_stay_share_orders():
    w = World(7)
    r1 = chain_turn(w)
    be = ScriptedBackend([{"tool_calls": [{"tool": "draft_order", "args": {"symbol": "TSLA", "direction": "buy",
                                                                           "qty": 10}}]},
                          {"answer": "Drafted; it needs your confirmation."}])
    ev = []
    run_turn("buy 10 shares of it", [r1.turn], w, be, on_event=ev.append)
    call = next(e for e in ev if e["type"] == "tool_call")
    assert not ({"side", "strike", "expiry_rule"} & set(call["args"])) and "filled_from_frame" not in call


def test_two_chains_one_turn_strikes_follow_the_chain():
    from agent.memory import build_frame
    from agent.slots import SlotBook
    calls = [{"tool": "get_option_chain", "args": {"symbol": "TSLA", "side": "call", "expiry_rule": "next_week"}},
             {"tool": "get_option_chain", "args": {"symbol": "TSLA", "side": "put", "expiry_rule": "next_week"}}]
    res = [{"tool": "get_option_chain", "args": calls[0]["args"], "result": {"ok": True, "slots": {"strike_2": 410.0}, "words": {}}},
           {"tool": "get_option_chain", "args": calls[1]["args"], "result": {"ok": True, "slots": {"strike_2": 400.0}, "words": {}}}]
    fr = build_frame("tesla calls and puts", calls, SlotBook(), "The {strike_2_2} put.", "", res)
    assert fr["option"]["side"] == "put" and fr["chain_strikes"] == {2: 400} and fr["option"]["said_strike"] == 400


def test_verbs_are_not_sides_and_nearest_needs_an_expiry_noun():
    w = World(7)
    r1 = chain_turn(w)
    be = ScriptedBackend([{"tool_calls": [{"tool": "draft_order", "args": {"symbol": "TSLA", "direction": "buy", "qty": 3}}]},
                          {"answer": "Drafted; it needs your confirmation."}])
    ev = []
    run_turn("put in an order for three of the second strike", [r1.turn], w, be, on_event=ev.append)
    assert next(e for e in ev if e["type"] == "tool_call")["args"]["side"] == "call"
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_option_chain", "args": {"symbol": "TSLA"}}]}, {"answer": "ok"}])
    ev = []
    run_turn("which strike is nearest the money", [r1.turn], w, be, on_event=ev.append)
    assert next(e for e in ev if e["type"] == "tool_call")["args"]["expiry_rule"] == "next_week"


def test_quantity_question_completes_the_held_back_ticket():
    w = World(7)
    r1 = chain_turn(w)
    be = ScriptedBackend([{"tool_calls": [{"tool": "draft_order", "args": {"symbol": "TSLA", "direction": "buy"}}]}])
    r2 = run_turn("buy the second strike", [r1.turn], w, be)
    assert r2.answer_template == ASK_QTY and r2.turn.frame["pending"]["strike"] == r1.turn.frame["chain_strikes"][2]
    be = ScriptedBackend([{"answer": "Drafted three; it needs your confirmation."}])
    r3 = run_turn("three", [r1.turn, r2.turn], w, be)
    assert r3.ticket is not None and r3.tool_calls[0]["args"]["qty"] == 3
    assert r3.tool_calls[0]["args"]["strike"] == r1.turn.frame["chain_strikes"][2]
    # "50 calls" is a quantity, not a strike
    from agent.memory import qty_is_strike
    assert not qty_is_strike({"qty": 50, "side": "call"}, "buy 50 calls") and qty_is_strike({"qty": 650}, "the 650 call")


def test_no_account_reuse_after_a_ticket_and_clear_beats_a_running_turn(tmp_path):
    from agent.memory import cached_result
    w = World(7)
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_orders", "args": {}}]}, {"answer": "Your orders."},
                          {"tool_calls": [{"tool": "draft_order", "args": {"symbol": "NVDA", "direction": "buy", "qty": 1}}]},
                          {"answer": "Drafted; it needs your confirmation."}])
    r1 = run_turn("show my orders", [], w, be)
    r2 = run_turn("buy one share of nvidia", [r1.turn], w, be)
    assert cached_result("get_orders", r1.tool_calls[0]["args"], [r1.turn]) is not None
    assert cached_result("get_orders", r1.tool_calls[0]["args"], [r1.turn, r2.turn]) is None
    conv = Conversation(SessionStore(tmp_path / "s"), session_key("pw"))

    def run(q, h, f):            # the user clears memory while this turn is still running
        conv.clear("all")
        return r1
    conv.ask("show my orders", run, world=w)
    assert conv.turns == []
    conv.closed = True
    conv.ask("show my orders", lambda *a: r1, world=w)
    assert conv.turns == []


def test_name_stop_words_and_bad_store_file(tmp_path):
    assert parse_fact("call me later", World(7)) is None
    store, key = SessionStore(tmp_path / "s"), session_key("pw")
    store.path(key).write_text("[1, 2]")
    assert Conversation(store, key).turns == []
    store.path(key).write_text(json.dumps({"turns": [{"answer": "x"}], "facts": [3, "name Sam"]}))
    c = Conversation(store, key)
    assert c.turns == [] and c.facts == ["name Sam"]


def test_single_turn_option_order_repairs():
    w = World(7)
    be = ScriptedBackend([{"tool_calls": [{"tool": "draft_order", "args": {"symbol": "TSLA", "direction": "buy",
                                                                           "qty": 3, "type": "call",
                                                                           "expiry_rule": "next_week"}}]},
                          {"answer": "Drafted; it needs your confirmation."}])
    ev = []
    r = run_turn("buy 3 Tesla 450 calls next week", [], w, be, on_event=ev.append)
    call = next(e for e in ev if e["type"] == "tool_call")
    assert call["args"]["side"] == "call" and call["args"]["strike"] == 450 and call["args"]["type"] == "market"
    assert r.ticket is not None and "strike" in call["from_question"]


def test_said_words_fix_single_turn_args():
    from agent.memory import resolve_followup
    a, f = resolve_followup("get_option_chain", {"symbol": "TSLA", "expiry_rule": "next_week", "side": "call"},
                            "next week's put chain for TSLA", [])
    assert a["side"] == "put" and f == ["said:side"]
    a, _ = resolve_followup("get_option_chain", {"symbol": "AAPL", "side": "call"}, "Apple 235 calls next week, bid and ask", [])
    assert a["strike"] == 235 and a["expiry_rule"] == "next_week"
    a, _ = resolve_followup("get_indicators", {"symbol": "SMCI", "timeframe": "1H"}, "Supermicro rsi on the five minute", [])
    assert (a["indicator"], a["timeframe"]) == ("rsi", "5m")
    a, _ = resolve_followup("get_indicators", {"symbol": "LLY", "timeframe": "1H"}, "give me the Eli Lilly indicators", [])
    assert a["timeframe"] == "1D" and "indicator" not in a
    a, _ = resolve_followup("get_indicators", {"symbol": "PLTR"}, "is Palantir above the moving averages", [])
    assert a["indicator"] == "ema"


def test_unchecked_no_data_claim_gets_a_retry():
    w = World(7)
    be = ScriptedBackend([{"answer": "There's no Apple two thirty-five call for next week."},
                          {"tool_calls": [{"tool": "get_option_chain", "args": {"symbol": "AAPL", "strike": 235,
                                                                                "expiry_rule": "next_week"}}]},
                          {"answer": "The {strike_1} call is {bid_1} bid, {ask_1} ask."}])
    r = run_turn("Apple 235 calls next week, bid and ask", [], w, be)
    assert r.tool_calls and r.tool_calls[0]["tool"] == "get_option_chain" and not r.fallback
    # a plain decline is untouched
    be = ScriptedBackend([{"answer": "I can't tell you what to buy, but I can pull up data on any stock."}])
    r = run_turn("what should I buy today", [], w, be)
    assert r.tool_calls == [] and len(be.seen) == 1


def test_a_month_name_means_the_monthly():
    from agent.memory import resolve_followup
    a, _ = resolve_followup("get_option_chain", {"symbol": "AAPL", "expiry_rule": "nearest"},
                            "options for Apple maybe for November or something", [])
    assert a["expiry_rule"] == "monthly"


def test_unsupported_chart_timeframe_is_answered_in_code():
    w = World(7)
    be = ScriptedBackend([])
    r = run_turn("show me nvda on the 15 minute", [], w, be)
    assert r.tool_calls == [] and "five-minute, hourly and daily" in r.spoken and be.seen == []
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_indicators", "args": {"symbol": "NVDA", "timeframe": "5m"}}]},
                          {"answer": "Nvidia RSI is {rsi}."}])
    r = run_turn("five minute rsi on nvidia", [], w, be)
    assert r.tool_calls and r.tool_calls[0]["args"]["timeframe"] == "5m"
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_option_chain", "args": {"symbol": "NVDA", "expiry_rule": "monthly"}}]},
                          {"answer": "Nvidia monthly calls."}])
    r = run_turn("nvidia monthly calls", [], w, be)
    assert r.tool_calls          # "monthly" expiry is not a chart timeframe


def test_spoken_strikes():
    from agent.memory import resolve_followup, said_strikes
    assert said_strikes("sell four of the three seventys") == [370.0]
    assert said_strikes("buy 1 of the 280s") == [280.0]
    assert said_strikes("grab 5 of the apple two forty calls for next week") == [240.0]
    assert said_strikes("one seventy five puts") == [175.0]
    assert said_strikes("buy three shares") == [] and said_strikes("how did the market do in the 2020s") == [2020.0]
    a, f = resolve_followup("draft_order", {"symbol": "ADBE", "direction": "sell", "qty": 3, "strike": 37.0, "side": "put"},
                            "okay, sell three of those three seventys", [])
    assert a["strike"] == 370 and "said:strike" in f


def test_arithmetic_declined_in_code_and_orders_status():
    from agent.memory import resolve_followup
    w = World(7)
    for q in ("um what does two plus two come to", "can you add 3 and 4", "yo tell me two and two added together"):
        be = ScriptedBackend([])
        r = run_turn(q, [], w, be)
        assert r.tool_calls == [] and be.seen == [], q
    be = ScriptedBackend([{"tool_calls": [{"tool": "get_news", "args": {"symbol": "NVDA"}}]}, {"answer": "News."}])
    assert run_turn("news plus levels for nvidia", [], w, be).tool_calls
    a, _ = resolve_followup("get_orders", {"day": "yesterday", "status": "open"}, "how many orders did i send in yesterday", [])
    assert a["status"] == "all"
    a, _ = resolve_followup("get_orders", {"status": "all"}, "any open orders", [])
    assert a["status"] == "open"


def test_validator_tolerates_spelled_tickers_and_bad_optional_choices():
    from agent.memory import resolve_followup
    from agent.validator import validate_call
    w = World(7)
    chk = validate_call(w, "get_quote", {"symbol": "S M H"})
    assert chk.ok and chk.args["symbol"] == "SMH"
    chk = validate_call(w, "get_option_chain", {"symbol": "OR", "expiry_rule": "monthly expiry"})
    assert chk.ok and chk.args["symbol"] == "OR"
    a, _ = resolve_followup("get_orders", {"day": "yesterday"}, "how many orders did i send in yesterday", [])
    assert a["status"] == "all"


def test_repair_args_from_the_question():
    from agent.loop import repair_args
    w = World(7)
    assert repair_args("get_quote", {"symbol": "MSH"}, "how is S M H looking today", w)[0]["symbol"] == "SMH"
    assert repair_args("get_option_chain", {"strike": 205.0, "side": "put"}, "delta on those Boeing 205 puts?", w)[0]["symbol"] == "BA"
    assert repair_args("what_if", {"symbol": "BOK", "move_pct": -5}, "what happens to my book if the market drops 5 percent", w)[0]["symbol"] == "ALL"
    a, f = repair_args("get_option_chain", {"symbol": "OR", "expiry_rule": "call"}, "Osisko Gold calls", w)
    assert "expiry_rule" not in a and a["side"] == "call"
    # a known symbol the model chose is left alone
    assert repair_args("get_quote", {"symbol": "AAPL"}, "how's Apple", w)[1] == []


def test_repairs_switch_off(monkeypatch):
    from agent.memory import resolve_followup
    monkeypatch.setenv("VOICE_REPAIRS", "0")
    a, f = resolve_followup("get_option_chain", {"symbol": "TSLA", "side": "call"}, "next week's put chain for TSLA", [])
    assert a == {"symbol": "TSLA", "side": "call"} and f == []
    be = ScriptedBackend([{"answer": "Four."}])
    r = run_turn("what's three plus four", [], World(7), be)
    assert len(be.seen) == 1 and r.answer_template == "Four."


def test_those_means_the_priced_row():
    from agent.memory import build_frame
    from agent.slots import SlotBook
    calls = [{"tool": "get_option_chain", "args": {"symbol": "TSLA", "side": "call", "expiry_rule": "next_week"}}]
    res = [{"tool": "get_option_chain", "args": calls[0]["args"],
            "result": {"ok": True, "slots": {"strike_1": 410.0, "strike_2": 415.0, "strike_3": 420.0}, "words": {}}}]
    fr = build_frame("q", calls, SlotBook(), "Strikes run {strike_1} to {strike_3}; row two at {strike_2} is {bid_2} bid, {ask_2} ask.", "", res)
    assert fr["option"]["said_strike"] == 415


def test_stt_repairs_options_mishears():
    from agent.memory import stt_repairs

    class T:
        frame = {"calls": [{"tool": "get_option_chain", "args": {}}]}

    assert stt_repairs("Show me next week's Tesla cost", []) == "Show me next week's Tesla calls"
    assert stt_repairs("And the post", [T()]) == "And the puts"
    assert stt_repairs("And the post", []) == "And the post"                       # no options context
    assert stt_repairs("What's the cost of my position", []) == "What's the cost of my position"
    assert stt_repairs("What driving oracle today", []) == "What's driving oracle today"


def test_stt_repairs_leaves_normal_questions():
    from agent.memory import stt_repairs

    class T:
        frame = {"calls": [{"tool": "get_option_chain", "args": {}}]}

    assert stt_repairs("What could cause Tesla to drop this week", []) == "What could cause Tesla to drop this week"
    assert stt_repairs("What's my cost basis on Nvidia", [T()]) == "What's my cost basis on Nvidia"
    assert stt_repairs("How did Tesla do post-market", [T()]) == "How did Tesla do post-market"
    assert stt_repairs("What's the weekly cost?", []) == "What's the weekly calls?"   # expiry named + last word


def test_stt_repairs_oracle():
    from agent.memory import stt_repairs
    assert stt_repairs("What's driving allocal today", []) == "What's driving Oracle today"


def test_stt_repairs_was_driving():
    from agent.memory import stt_repairs
    assert stt_repairs("Was driving Oracle today", []) == "What's driving Oracle today"
    assert stt_repairs("Was driving over a goal today", []) == "What's driving Oracle today"
