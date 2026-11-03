"""Agent loop: model decides tool calls / answers; code runs tools, validates, renders."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from tools import model_view, run_tool, schema
from tools.result import ToolResult
from tools.numwords import int_words

from .format import tool_content
from .memory import (build_frame, cached_result, pending_order, whatif_followup, qty_is_strike, repairs_on, resolve_followup,
                     state_extras)
from .slots import SlotBook, filler_line, fill, speakable
from .validator import (SLOT_RE, binding_problems, check_answer, correct_symbol, named_companies, no_digits,
                        rename_companies, tickers_in, validate_call)

SYSTEM_PROMPT = """You are Leaps, a voice trading assistant. Answers are spoken aloud, so keep them short: at most two sentences.

Use tools to get data. Tool results contain judgment words (like "near_support": "yes") and slot names (like "{price}"). You never see the numbers.
In your final answer write every number ONLY as a slot placeholder copied exactly from the tool results, e.g. "Nvidia is at {price}, {change_pct} today." Never write digits yourself; if you repeat a number the user said, spell it out (five shares). Use the judgment words for judgments.
Company names and tickers (Nvidia, NVDA) can be passed straight as symbol; call resolve_symbol only when you can't tell which company is meant. If a tool returns an error, fix the call or tell the user briefly.
draft_order only drafts a ticket; nothing is ever sent until the user confirms. Never say an order was placed.
Call only the tools the question needs (usually one or two); when you need two, put both invokes in one function_calls block."""

TOPICS = {"get_quote": "quote", "get_position": "position", "get_portfolio": "portfolio",
          "get_indicators": "indicators", "get_levels": "levels", "get_option_chain": "options",
          "get_orders": "orders", "get_news": "news", "get_fundamentals": "fundamentals",
          "get_market": "market", "screen_watchlist": "watchlist", "what_if": "what if",
          "resolve_symbol": "symbol lookup", "draft_order": "order ticket"}


@dataclass
class Turn:
    question: str
    answer: str                         # answer template (with {slots})
    display: str = ""
    spoken: str = ""
    symbol: str | None = None
    topic: str | None = None
    tools: list[str] = field(default_factory=list)
    frame: dict | None = None           # code-filled conversation frame (agent.memory.build_frame)


@dataclass
class TurnResult:
    question: str
    answer_template: str
    display: str
    spoken: str
    events: list[dict]
    tool_calls: list[dict]
    slots: dict
    ticket: dict | None
    timings: dict
    fallback: bool
    turn: Turn


def _history_text(t: Turn) -> str:
    """Prior answers go back as their spoken form: natural, digit-free, no stale slot names."""
    return t.spoken or SLOT_RE.sub("that", t.answer)


def build_messages(question: str, history: list[Turn] | None, facts: list[str] | None = None,
                   frame_line: bool = False) -> list[dict]:
    """System + last <=3 turns (question, spoken answer) + the state line + question. `frame_line` (models
    trained with it) adds `last_call=` (the last tool-calling turn's calls) and `facts=` (pinned) to the line."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    hist = (history or [])[-3:]
    for t in hist:
        msgs.append({"role": "user", "content": t.question})
        msgs.append({"role": "assistant", "content": _history_text(t)})
    hint = ""
    if hist:
        last = hist[-1]
        parts = []
        if last.symbol:
            parts.append(f"last_symbol={last.symbol}")
        if last.topic:
            parts.append(f"last_topic={last.topic}")
        if frame_line:
            parts += state_extras(hist, facts)
        if parts:
            hint = "[state: " + ", ".join(parts) + "]\n"
    msgs.append({"role": "user", "content": hint + question})
    return msgs


# An answer that says data is missing / unavailable (only acceptable after a tool said so).
_NO_DATA_CLAIM = re.compile(r"\b(there(?:'s| is| are) no|there aren't any|i couldn't (?:pull|get|find|price|draft)|"
                            r"isn't available|not available|no (?:chain|weeklies|data|quote))\b", re.I)
# A question that asks for market / account / option data (a company, a ticker or option words).
_ASKS_FOR_DATA = re.compile(r"\b(calls?|puts?|strike|chain|options?|quote|price|bid|ask|delta|iv|rsi|position|"
                            r"orders?|news|earnings|support|resistance)\b|\b[A-Z]{2,5}\b")
# Chart timeframes the tools do not have (get_indicators: 5m, 1H, 1D).
# Plain arithmetic ("what's three plus four") with no company in it: off-topic, declined in code.
_NUMW = (r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
         r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)")
_ARITHMETIC = re.compile(r"\b" + _NUMW + r"\s+(?:plus|minus|times|divided\s+by|multiplied\s+by|over)\s+" + _NUMW + r"\b|"
                         r"\badd(?:ed)?\s+" + _NUMW + r"\s+(?:and|to)\s+" + _NUMW + r"\b|"
                         r"\b" + _NUMW + r"\s+and\s+" + _NUMW + r"\s+added\s+together\b", re.I)
# "should I buy Tesla?": advice is declined in code, with what the agent can do instead
_ADVICE = re.compile(r"^\W*(?:(?:so|hey|ok(?:ay)?|uh|um)\W+)*(?:should|would|do\s+you\s+think)\s+i\s+(?:buy|sell|short|"
                     r"get\s+into|get\s+out\s+of|dump|load\s+up\s+on)\b", re.I)
_UNSUPPORTED_TF = re.compile(r"\b(?:1|one|2|two|3|three|10|ten|15|fifteen|30|thirty)[\s-]*min(?:ute)?s?\b|"
                             r"\b(?:2|two|4|four)[\s-]*h(?:ou)?rs?\b|\bweekly\s+(?:chart|candles?|rsi|timeframe)\b|"
                             r"\b(?:on|the)\s+weekly\b|\bmonthly\s+(?:chart|candles?|rsi|timeframe)\b", re.I)
FALLBACK_TICKET = "I drafted that order ticket; it needs your confirmation before anything is sent."
FALLBACK_DATA = "I pulled that up, and the numbers are on your screen."
FALLBACK_NONE = "Sorry, I couldn't get that. Could you say it another way?"
# The model wanted numbers ({slots}) but never fetched any data: never let it state facts; ask instead.
CLARIFY_UNGROUNDED = "I haven't looked that up yet. Do you mean your positions, a stock, or the market movers?"
STRIKE_FIRST = "say the strike first"
ASK_QTY = "How many contracts do you want on that ticket?"
ASK_SHARES = "How many shares do you want on that ticket?"


def _symbols_in(obj, world, out: set) -> set:
    """Every universe ticker in a tool's args / words (strings, lists, dicts; also inside labels "BUY AAPL call")."""
    if isinstance(obj, str):
        out.update(t for t in re.findall(r"\b[A-Z]{1,5}\b", obj) if t in world.universe)
    elif isinstance(obj, dict):
        for v in obj.values():
            _symbols_in(v, world, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _symbols_in(v, world, out)
    return out


_BOOK = re.compile(r"\b(my\s+(?:book|portfolio|account|positions|holdings)|the\s+(?:whole\s+)?market|everything)\b", re.I)


# "why is X down", "what's driving X", "what's going on with X today": the trained answer is price + news;
# "the technicals on X": indicators + levels. A model that runs only the first adds the companion (same symbol).
_WHY_MOVE = re.compile(r"\bwhy\s+(?:is|are|'s|did|has)\b|\bwhat'?s\s+driving\b|\bwhat\s+is\s+driving\b|\bdriving\s+(?:it|that|this)\b"
                       r"|\bwhat'?s\s+going\s+on\s+with\b|\bwhy\b.*\b(?:moving|up|down|falling|dropping|rallying)\b", re.I)
_TECHNICALS = re.compile(r"\btechnicals?\b(?!\s+(?:indicator|rsi|ema))", re.I)


# "is anything holding up", "anything that stopped bleeding", "anything green": in a down market the user asks what
# is recovering. The book when they say so, the watchlist screen for their list (trained: "bouncing off the lows on my
# list" -> oversold), market breadth right after a market answer (the movers were just read out), else the model.
_RECOVER = re.compile(r"\bstop(?:s|ped|ping)?\s+(?:the\s+)?(?:bleed(?:ing)?|falling|dropping|sliding)\b|\bholding\s+up\b"
                      r"|\banything\s+(?:that'?s\s+|that\s+is\s+|is\s+)?(?:green|up|bouncing|recovering|rebounding)\b"
                      r"|\bwhat'?s\s+(?:green|bouncing|recovering|rebounding)\b|\bin\s+the\s+green\b|\bbounc(?:e|es|ing)\s+back\b", re.I)
_MY_BOOK = re.compile(r"\b(?:my\s+(?:book|positions?|portfolio|holdings?|stocks?|account)|i\s+(?:own|hold))\b", re.I)
_MY_LIST = re.compile(r"\bwatch\s*list\b|\bmy\s+list\b", re.I)


def recovery_route(question: str, history: list, world) -> tuple[str, dict] | None:
    """(tool, args) for a 'what is recovering' question with no company named, else None."""
    if not _RECOVER.search(question) or named_companies(question, world) or tickers_in(question, world):
        return None
    if _MY_LIST.search(question):
        return "screen_watchlist", {"condition": "oversold"}
    if _MY_BOOK.search(question):
        return "get_portfolio", {"rank_by": "day_move"}
    prev = [c["tool"] for c in ((getattr(history[-1], "frame", None) or {}).get("calls") or [])] if history else []
    if "get_market" in prev:
        return "get_market", {"focus": "breadth"}
    return None                     # no context: the model's own call (trained: "bouncing in the market" -> movers)


SAME_AGAIN = "Same as a moment ago: "


# ------------------------------------------------------------------ watchlist (a code route, not a model tool)
WATCH_LANE = "code: watchlist"
_WL = r"(?:my\s+|the\s+|our\s+)?watch\s*-?\s*list"
_WL_ADD = re.compile(r"\b(?:add|put|stick|throw|pop)\s+(?P<what>.+?)\s+(?:to|on|onto|in|into)\s+" + _WL + r"\b", re.I)
_WL_REMOVE = re.compile(r"\b(?:remove|take|drop|delete|pull|kick)\s+(?P<what>.+?)\s+(?:off(?:\s+of)?|from|out\s+of)\s+" + _WL + r"\b",
                        re.I)
_WL_SHOW = re.compile(r"^\W*(?:(?:hey|ok(?:ay)?|so|uh|um)\W+)*(?:(?:what'?s|what\s+is)\s+(?:on|in)\s+" + _WL +
                      r"|(?:show|read|give|tell)(?:\s+me)?\s+" + _WL + r"|(?:how'?s|how\s+is|how\s+are|check(?:\s+on)?)\s+" + _WL +
                      r"(?:\s+(?:doing|looking))?|" + _WL + r"(?:\s+(?:movers|update|check))?)(?:\s+(?:today|now|right\s+now))?\W*$", re.I)


def watchlist_route(question: str, world, history: list | None = None) -> tuple | None:
    """("show",) | ("add", sym) | ("remove", sym) | ("clarify", text) | None. Screens ("oversold on my watchlist")
    are not routed here: they stay with the model's screen_watchlist."""
    q = question or ""
    for kind, rx in (("add", _WL_ADD), ("remove", _WL_REMOVE)):
        m = rx.search(q)
        if m:
            what = m.group("what")
            syms = list(dict.fromkeys(x[0] for x in named_companies(what, world) + tickers_in(what, world)))
            if not syms and history and re.fullmatch(r"\s*(?:it|that|this|that\s+one|this\s+one|that\s+stock|this\s+stock)\s*",
                                                     what, re.I) and getattr(history[-1], "symbol", None):
                syms = [history[-1].symbol]                  # "add it to my watchlist": the company just discussed
            if len(syms) != 1:
                return ("clarify", "Which company should I " + ("add to" if kind == "add" else "take off") +
                        " your watchlist?")
            return (kind, syms[0])
    if _WL_SHOW.search(q):
        return ("show",)
    return None


def _watchlist_turn(question: str, route: tuple, world, emit, t_start: float) -> "TurnResult":
    """Run a watchlist route in code: quotes for every name (code-routed tool calls, lane 'code: watchlist'), a
    code-filled two-sentence answer. Add / remove update world.watchlist; the frame carries the new list so the
    Conversation persists it."""
    book, tool_calls, results = SlotBook(), [], []
    frame_extra: dict = {}
    wl = list(getattr(world, "watchlist", []) or [])
    if route[0] == "clarify":
        answer = route[1]
    elif route[0] in ("add", "remove"):
        sym = route[1]
        if route[0] == "add":
            if sym in wl:
                answer = f"{sym} is already on your watchlist."
            else:
                wl.append(sym)
                answer = f"Added {sym} to your watchlist."
        else:
            if sym not in wl:
                answer = f"{sym} isn't on your watchlist."
            else:
                wl.remove(sym)
                answer = f"Took {sym} off your watchlist."
        world.watchlist = wl
        frame_extra["watchlist"] = list(wl)
    elif not wl:
        answer = "Your watchlist is empty. Say add, then a company name, to start one."
    else:
        emit({"type": "filler", "text": "Checking your watchlist."})
        ranked = sorted(wl, key=lambda x: -world.quote(x).change_pct)
        names_by_sym = {}
        for sym in ranked:
            args = {"symbol": sym}
            emit({"type": "tool_call", "tool": "get_quote", "args": args, "step": 1, "lane": WATCH_LANE})
            res = run_tool(world, "get_quote", args)
            names = book.add("get_quote", res["slots"], res["words"], args)
            names_by_sym[sym] = names
            results.append({"tool": "get_quote", "args": dict(args), "result": res})
            emit({"type": "tool_result", "tool": "get_quote", "ok": res["ok"], "words": res["words"],
                  "error": res["error"], "slots_display": {names[k]: book.display(names[k]) for k in res["slots"]},
                  "step": 1, "lane": WATCH_LANE})
            tool_calls.append({"tool": "get_quote", "args": args, "ok": res["ok"], "step": 1, "lane": WATCH_LANE})
        chg = lambda x: "{" + names_by_sym[x]["change_pct"] + "}"
        uni = getattr(world, "universe", {}) or {}
        nm = lambda x: getattr(uni.get(x), "name", None) or x       # company names (a ticker like "V" reads badly)
        top = ranked[:3]
        answer = (f"Your watchlist has {int_words(len(ranked))} names; " +
                  ", ".join(f"{nm(x)} is {chg(x)}" for x in top[:-1]) + (" and " if len(top) > 1 else "") +
                  f"{nm(top[-1])} is {chg(top[-1])}.")
        if len(ranked) > 3:
            answer += f" Weakest is {nm(ranked[-1])}, {chg(ranked[-1])}."
    display = fill(answer, book, "display")
    spoken = speakable(fill(answer, book, "spoken"), world)
    emit({"type": "answer", "template": answer, "display": display, "spoken": spoken, "fallback": False,
          "code": True, "lane": WATCH_LANE})
    frame = build_frame(question, tool_calls, book, answer, spoken, results)
    frame.update(frame_extra)
    syms = [c["args"]["symbol"] for c in tool_calls]
    sym = syms[0] if syms else (route[1] if route[0] in ("add", "remove") else None)
    turn = Turn(question, answer, display, spoken, sym, "watchlist", [c["tool"] for c in tool_calls], frame)
    return TurnResult(question, answer, display, spoken, [], tool_calls, book.values(), None,
                      {"total": time.perf_counter() - t_start, "decide": [], "tools": [], "filler_at": None},
                      False, turn)


def ticket_sentence(ticket: dict, world) -> str:
    """An option ticket's answer, from the ticket itself. Numbers as words (the ticket is not a slot result)."""
    from tools.render import spoken as rspoken
    uni = getattr(world, "universe", {}) or {}
    name = getattr(uni.get(ticket["symbol"]), "name", None) or ticket["symbol"]
    qty = int(ticket.get("qty") or 0)
    side = ticket.get("side") or "call"
    exp = (" expiring " + rspoken("expiry", ticket["expiry"], "get_option_chain")) if ticket.get("expiry") else ""
    return (f"I've drafted a ticket to {ticket.get('direction') or 'buy'} {int_words(qty)} {name} "
            f"{rspoken('strike_1', ticket['strike'], 'get_option_chain')} {side}{'s' if qty != 1 else ''}{exp}. "
            "It needs your confirmation.")


def _norm_q(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (q or "").lower()).strip()


def companion_calls(question: str, calls: list) -> tuple[list, str | None]:
    """First decision only. The missing half of a trained parallel pair, phrase-anchored:
    why/driving/going-on + only get_quote(s) on ONE symbol -> + get_news; technicals + only get_indicators on ONE
    symbol -> + get_levels. Returns (calls, added tool or None)."""
    def parts(c):
        a = c.get("args") if c.get("args") is not None else c.get("arguments")
        return str(c.get("tool") or c.get("name") or "").split(".")[-1].strip(), a if isinstance(a, dict) else {}
    named = [parts(c) for c in calls]
    tools = {t for t, _ in named}
    syms = {str(a.get("symbol") or "").upper() for _, a in named}
    if len(syms) != 1 or "" in syms or "ALL" in syms:
        return calls, None
    sym = next(iter(named))[1].get("symbol")
    if tools == {"get_quote"} and _WHY_MOVE.search(question):
        return list(calls) + [{"tool": "get_news", "args": {"symbol": sym}}], "get_news"
    if tools == {"get_indicators"} and _TECHNICALS.search(question):
        return list(calls) + [{"tool": "get_levels", "args": {"symbol": sym}}], "get_levels"
    return calls, None


def repair_args(tool: str, args, question: str, world) -> tuple:
    """Arguments the model got wrong in ways the user's words settle (the call would otherwise fail validation):
    a missing or unknown symbol when the question names exactly one company / ticker; what_if on the user's book or
    the market -> ALL; a side written into expiry_rule. Returns (args, repaired keys)."""
    if not isinstance(args, dict) or not repairs_on():
        return args, []
    from tools.schema import TOOL_SCHEMAS
    props = TOOL_SCHEMAS.get(tool, {}).get("parameters", {}).get("properties", {})
    args, fixed = dict(args), []
    if "symbol" in props:
        sym = str(args.get("symbol") or "").upper().lstrip("$").strip()
        known = bool(sym) and (world.has(sym) or (tool == "what_if" and sym == "ALL"))
        if known and tool == "what_if" and sym != "ALL" and _BOOK.search(question) \
                and not (named_companies(question, world) or tickers_in(question, world)):
            args["symbol"], fixed = "ALL", fixed + ["symbol"]      # "if the market drops": the whole book
        elif not known:
            named = sorted({x[0] for x in named_companies(question, world) + tickers_in(question, world)})
            if tool == "what_if" and not named and _BOOK.search(question):
                args["symbol"], fixed = "ALL", fixed + ["symbol"]
            elif len(named) == 1:
                args["symbol"], fixed = named[0], fixed + ["symbol"]
    exp = str(args.get("expiry_rule") or "").lower().rstrip("s")
    if exp in ("call", "put"):
        args.pop("expiry_rule")
        if "side" in props and "side" not in args:
            args["side"] = exp
        fixed.append("expiry_rule")
    return args, fixed


def ticket_problems(text: str, ticket: dict) -> list[str]:
    """The answer must describe the ticket that was drafted: shares vs options, and the option side."""
    said = {sd for w, sd in (("call", "call"), ("put", "put")) if re.search(rf"\b{w}s?\b", text, re.I)}
    if ticket.get("kind") == "stock" and said:
        return ["the ticket is for shares, not options: don't say calls or puts"]
    side = ticket.get("side")
    if ticket.get("kind") == "option" and side and said - {side}:
        return [f"the ticket is for {side}s, not {(said - {side}).pop()}s"]
    return []


def chain_line(book: SlotBook, world, prefer: str | None = None, fields: set[str] | None = None) -> str | None:
    """Code-built option-chain answer that names the strike first, or None if no chain ran.
    `prefer`: a row slot name from the model's answer (its row is used), else the most-open-interest row.
    `fields`: row fields the model's answer used; IV / delta are added when it used them (bid/ask always)."""
    runs = sorted({m["run"] for m in book.meta.values() if m["run"][0] == "get_option_chain"})
    if not runs:
        return None
    run, row = runs[0], None
    if prefer and prefer in book.meta and book.meta[prefer].get("row"):
        run, row = book.meta[prefer]["run"], book.meta[prefer]["row"][1]
    words = book.run_words.get(run, {})
    if row is None:
        m = re.match(r"row (\w+)", str(words.get("most_open_interest", "")))
        row = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}.get(m.group(1) if m else "", 1)
    strike, bid, ask = (book.name_of(run, f"{f}_{row}") for f in ("strike", "bid", "ask"))
    if not (strike and bid and ask):
        return None
    sym = words.get("symbol")
    who = world.universe[sym].name + " " if sym in world.universe else ""
    side = {"call": "calls", "put": "puts"}.get(words.get("side"), "options")
    exp = book.name_of(run, "expiry")
    head = f"For {who}{side}" + (f" expiring {{{exp}}}" if exp else "")
    extra = "".join(f", {label} {{{n}}}" for f, label in (("iv", "IV"), ("delta", "delta"))
                    if f in (fields or ()) and (n := book.name_of(run, f"{f}_{row}")))
    return f"{head}, the {{{strike}}} strike is {{{bid}}} bid, {{{ask}}} ask{extra}."


def run_turn(question: str, history: list[Turn] | None, world, backend, max_steps: int = 4,
             on_event: Callable[[dict], None] | None = None, facts: list[str] | None = None,
             frame_line: bool | None = None) -> TurnResult:
    t_start = time.perf_counter()
    events: list[dict] = []
    history = [getattr(h, "turn", h) for h in (history or []) if isinstance(getattr(h, "turn", h), Turn)]

    def emit(ev: dict) -> None:
        events.append(ev)
        if on_event is not None:
            on_event(ev)

    tools_schema = schema()
    if frame_line is None:
        frame_line = bool(getattr(backend, "frame_line", False))
    messages = build_messages(question, history, facts=facts, frame_line=frame_line)
    results: list[dict] = []            # executed results, for the turn's frame (code-side only)
    ask_qty: str | None = None
    pending: dict | None = None         # the draft held back for a quantity (stored on the frame)
    # "three" after "How many contracts…?": the held-back draft, completed in code (the model is not asked)
    forced = pending_order(history, question)
    forced_tool = "draft_order"
    if forced is None:
        # "and twenty" after a what-if: re-run it with the new move, in code (the model would answer from memory)
        forced = whatif_followup(history, question)
        forced_tool = "what_if"
    if forced is None:
        wr = watchlist_route(question, world, history)
        if wr is not None:
            res = _watchlist_turn(question, wr, world, emit, t_start)
            res.events[:] = events
            return res
    code_text = None
    if forced is None and repairs_on():
        rr = recovery_route(question, history, world)
        if rr is not None:
            forced_tool, forced = rr[0], dict(rr[1])
    if not repairs_on():
        pass
    elif forced is None and _ADVICE.search(question):
        named = [x[0] for x in named_companies(question, world) + tickers_in(question, world)]
        uni = getattr(world, "universe", {}) or {}
        who = (getattr(uni.get(named[0]), "name", None) or named[0]) if len(set(named)) == 1 else None
        code_text = (f"I can't tell you whether to buy or sell, but I can pull up {who}'s price, chart or news."
                     if who else "I can't tell you what to buy or sell, but I can pull up any stock's price, chart or news.")
    elif forced is None and _UNSUPPORTED_TF.search(question):
        # a chart timeframe the tools don't have: say which ones exist, in code (no tool, no model)
        code_text = "I only have five-minute, hourly and daily charts. Want the daily?"
    elif forced is None and _ARITHMETIC.search(question) and not (named_companies(question, world) or tickers_in(question, world)):
        code_text = "I can only help with stocks, options and your account."
    if code_text is not None:
        text = code_text
        ev = {"type": "answer", "template": text, "display": text, "spoken": text, "fallback": False, "code": True}
        emit(ev)
        last_sym = history[-1].symbol if history else None
        topic = history[-1].topic if history else None
        turn = Turn(question, text, text, text, last_sym, topic, [], build_frame(question, [], None, text, text, []))
        return TurnResult(question, text, text, text, events, [], {}, None,
                          {"total": time.perf_counter() - t_start, "decide": [], "tools": [], "filler_at": None},
                          False, turn)
    book = SlotBook()
    tool_calls: list[dict] = []
    ticket: dict | None = None
    filler_sent = False
    tool_retry_used = False
    answer_retry_used = False
    forced_answer = False
    steps = 0
    call_n = 0
    decide_times: list[float] = []
    tool_times: list[float] = []
    first_filler_at = None
    answer = None
    fallback = False
    ungrounded_retry = False            # the retry was for slots the model never fetched (no tool ran)
    tried_tools = False                 # any tool call was made (even one that failed validation)
    last_text = ""
    last_problems: list[str] = []
    last_symbol = history[-1].symbol if history else None
    turn_symbols: set[str] = set()

    while True:
        t0 = time.perf_counter()
        try:
            if forced is not None:
                resp, forced = {"tool_calls": [{"tool": forced_tool, "args": forced}]}, None
            else:
                resp = backend.decide(messages, tools_schema)
        except Exception as e:  # noqa: BLE001
            resp = {"error": f"backend failure: {type(e).__name__}"}
        decide_times.append(time.perf_counter() - t0)

        if resp.get("error"):
            emit({"type": "error", "recoverable": True, "stage": "parse", "message": resp["error"]})
            if not tool_retry_used:
                tool_retry_used = True
                if resp.get("raw"):
                    messages.append({"role": "assistant", "content": resp["raw"][:2000]})
                messages.append({"role": "user", "content":
                                 "Error: your last message could not be parsed. Call a tool using the function "
                                 "call format, or answer in plain words with slot placeholders."})
                continue
            fallback = True
            break

        calls = resp.get("tool_calls") or []
        if calls and steps == 0 and not tool_calls and repairs_on():
            calls, _ = companion_calls(question, calls)
        if calls and repairs_on():
            # the same call twice in one decision (same tool, same arguments) runs once
            seen, uniq = set(), []
            for c in calls:
                k = json.dumps([c.get("tool") or c.get("name"), c.get("args") if c.get("args") is not None
                                else c.get("arguments")], sort_keys=True, default=str)
                if k not in seen:
                    seen.add(k)
                    uniq.append(c)
            calls = uniq
        if calls:
            if steps >= max_steps:
                if not forced_answer:
                    forced_answer = True
                    messages.append({"role": "user", "content":
                                     "No more tool calls. Answer now using the slots you already have."})
                    continue
                fallback = True
                break
            steps += 1
            tried_tools = True
            assistant_msg = {"role": "assistant", "content": "", "tool_calls": []}
            tool_msgs = []
            bad = False
            batch_syms = {str((c.get("args") or c.get("arguments") or {}).get("symbol", "")).upper()
                          for c in calls if isinstance(c.get("args") or c.get("arguments"), dict)}
            for c in calls:
                call_n += 1
                cid = f"call_{call_n}"
                raw_args = c.get("args") if c.get("args") is not None else c.get("arguments")
                tname = str(c.get("tool") or c.get("name") or "").split(".")[-1].strip()
                # follow-ups: fill what the model left implicit (symbol / side / expiry / "the second strike")
                exec_args, filled = resolve_followup(tname, raw_args, question, history)
                exec_args, fixed = repair_args(tname, exec_args, question, world)
                if tname == "draft_order" and isinstance(exec_args, dict) and "side" not in exec_args and history:
                    # "buy some Nvidia calls" -> (asked how many) -> "two": the side was said one turn back
                    prev = (getattr(history[-1], "frame", None) or {})
                    pq = str(prev.get("question") or "")
                    sides = {sd for w, sd in (("call", "call"), ("put", "put")) if re.search(rf"\b{w}s?\b", pq, re.I)}
                    bare = not (named_companies(question, world) or tickers_in(question, world))
                    if bare and not prev.get("calls") and len(sides) == 1 and not re.search(r"\bshares?\b", question, re.I):
                        exec_args = dict(exec_args, side=sides.pop())
                        fixed = list(fixed) + ["side"]
                if isinstance(exec_args, dict) and exec_args.get("symbol") not in (None, "", "ALL") and history:
                    # a bare answer to our own question ("five"): the company is the one named one turn back,
                    # not an older ticker carried in the frame
                    prev = (getattr(history[-1], "frame", None) or {})
                    if not prev.get("calls") and not named_companies(question, world) and not tickers_in(question, world):
                        pq = str(prev.get("question") or "")
                        named = {sym for sym, _, _ in named_companies(pq, world) + tickers_in(pq, world)}
                        if len(named) == 1 and exec_args["symbol"] not in named:
                            exec_args = dict(exec_args, symbol=named.pop())
                            fixed = list(fixed) + ["symbol"]
                filled = filled + ["said:" + k for k in fixed]
                chk = validate_call(world, c.get("tool") or c.get("name"), exec_args)
                if chk.name == "draft_order" and (chk.error == "missing qty" or (chk.ok and repairs_on() and qty_is_strike(chk.args, question))):
                    # no quantity said (or the strike was read as one): ask, don't draft
                    ea = exec_args if isinstance(exec_args, dict) else {}
                    ask_qty = ASK_QTY if ea.get("side") or ea.get("strike") else ASK_SHARES
                    pending = {k: v for k, v in ea.items() if k != "qty"}
                    if qty_is_strike(ea, question) and "strike" not in pending:
                        pending["strike"] = ea.get("qty")
                    break
                corrected_from = None
                if chk.ok and "symbol" in chk.args:
                    # the user's words name a company whose ticker differs from the model's (lookalike / stale state)
                    fix = correct_symbol(world, chk.args["symbol"], question, last_symbol,
                                         {tc["args"].get("symbol") for tc in tool_calls} | (batch_syms - {
                                             chk.args["symbol"], str(raw_args.get("symbol", "")).upper()}))
                    if fix:
                        corrected_from, chk.args["symbol"] = chk.args["symbol"], fix
                assistant_msg["tool_calls"].append(
                    # the model's own arguments (as emitted), so re-rendered history matches the trained format
                    {"id": cid, "type": "function",
                     "function": {"name": chk.name, "arguments": dict(raw_args) if isinstance(raw_args, dict)
                                  else {}}})
                emit({"type": "tool_call", "tool": chk.name, "args": chk.args if chk.ok else c.get("args"),
                      "step": steps, **({"corrected_from": corrected_from} if corrected_from else {}),
                      **({"filled_from_frame": [f for f in filled if not f.startswith("said:")]}
                         if any(not f.startswith("said:") for f in filled) else {}),
                      **({"from_question": [f[5:] for f in filled if f.startswith("said:")]}
                         if any(f.startswith("said:") for f in filled) else {})})
                if not chk.ok:
                    bad = True
                    res = ToolResult.fail(chk.name, chk.error, words=chk.words)
                    emit({"type": "error", "recoverable": True, "stage": "tool", "tool": chk.name, "message": chk.error})
                    names = {}
                else:
                    if not filler_sent:
                        filler_sent = True
                        first_filler_at = time.perf_counter() - t_start
                        emit({"type": "filler", "text": filler_line(chk.name, chk.args, world)})
                    t1 = time.perf_counter()
                    res = cached_result(chk.name, chk.args, history)     # still fresh from an earlier turn?
                    cached = res is not None
                    if res is None:
                        res = run_tool(world, chk.name, chk.args)
                    tool_times.append(time.perf_counter() - t1)
                    if chk.resolved or corrected_from:
                        res["words"] = dict(res["words"], resolved_from="spoken name")
                    names = book.add(chk.name, res["slots"], res["words"], chk.args)
                    results.append({"tool": chk.name, "args": dict(chk.args), "result": res})
                    _symbols_in([chk.args, res["words"]], world, turn_symbols)
                    emit({"type": "tool_result", "tool": chk.name, "ok": res["ok"], "words": res["words"],
                          "error": res["error"],
                          "slots_display": {names[s]: book.display(names[s]) for s in res["slots"]},
                          "step": steps, **({"cached": True} if cached else {})})
                    tool_calls.append({"tool": chk.name, "args": chk.args, "ok": res["ok"], "step": steps})
                    if chk.name == "draft_order" and res["ok"]:
                        tid = res["slots"]["ticket_id"]
                        ticket = dict(world.tickets[tid])
                        emit({"type": "ticket", **ticket})
                tool_msgs.append({"role": "tool", "tool_call_id": cid, "name": chk.name,
                                  "content": tool_content(model_view(res, names))})
            if ask_qty:
                break
            messages.append(assistant_msg)
            messages.extend(tool_msgs)
            if bad:
                if tool_retry_used:
                    fallback = True
                    break
                tool_retry_used = True
            continue

        text = (resp.get("answer") or "").strip()
        if repairs_on() and not tried_tools and not answer_retry_used and _NO_DATA_CLAIM.search(text) \
                and _ASKS_FOR_DATA.search(question):
            # "There's no Apple 235 call for next week" with no tool run: a claim about data it never looked at
            answer_retry_used = True
            messages.append({"role": "assistant", "content": no_digits(text) or "(empty)"})
            named = sorted({x[0] for x in named_companies(question, world) + tickers_in(question, world)})
            messages.append({"role": "user", "content": "You have not looked anything up yet, so you cannot say "
                             "the data is missing. Call the tool that gets it" +
                             (f" for {', '.join(named)}." if named else ".")})
            emit({"type": "error", "recoverable": True, "stage": "answer", "message": "unchecked no-data claim"})
            continue
        arg_syms = {c["args"]["symbol"] for c in tool_calls if c["args"].get("symbol") not in (None, "ALL")}
        if len(arg_syms) == 1:
            # the company the answer names comes from the data, not the model's free text
            text = rename_companies(text, world, turn_symbols, next(iter(arg_syms)))
        problems = check_answer(text, book.names(), draft_ran=ticket is not None)
        if not problems:
            problems = binding_problems(text, book, world)
        if not problems and ticket is not None:
            problems = ticket_problems(text, ticket)
        if problems:
            last_problems, last_text = problems, text
            emit({"type": "error", "recoverable": True, "stage": "answer", "message": "; ".join(problems)})
            if not answer_retry_used:
                answer_retry_used = True
                messages.append({"role": "assistant", "content": no_digits(text) or "(empty)"})
                if not book.names() and SLOT_RE.search(text):
                    ungrounded_retry = True
                    messages.append({"role": "user", "content":
                                     "You have no tool results yet, so you have no numbers or facts to state. Call "
                                     "the tool that gets this data, or ask the user one short clarifying question."})
                else:
                    avail = ", ".join("{" + s + "}" for s in sorted(book.names())) or "none"
                    messages.append({"role": "user", "content":
                                     "Rewrite your answer: " + "; ".join(problems) + ". Use only these slot "
                                     f"placeholders for numbers: {avail}; if you need other data, call a tool. "
                                     "No digits, at most two sentences."})
                continue
            fallback = True
            break
        answer = text
        break

    if ask_qty:
        answer, fallback = ask_qty, False
    if not fallback and answer is not None and ungrounded_retry and not book.names() and not tried_tools \
            and not answer.rstrip().endswith("?"):
        fallback, answer = True, CLARIFY_UNGROUNDED          # a fact with no data behind it: ask instead
    if fallback or answer is None:
        fallback = True
        line = None
        if any(p.startswith(STRIKE_FIRST) for p in last_problems):
            bad = re.findall(r"\{([A-Za-z_0-9]+)\}", next(p for p in last_problems if p.startswith(STRIKE_FIRST)))
            fields = {book.meta[n]["row"][0] for n in SLOT_RE.findall(last_text)
                      if n in book.meta and book.meta[n].get("row")}
            line = chain_line(book, world, prefer=next((b for b in bad if b in book), None), fields=fields)
        if line is None and ungrounded_retry and not book.names() and not tried_tools:
            line = CLARIFY_UNGROUNDED
        answer = line or (FALLBACK_TICKET if ticket else (FALLBACK_DATA if tool_calls else FALLBACK_NONE))
    display = fill(answer, book, "display")
    spoken = speakable(fill(answer, book, "spoken"), world)
    if (repairs_on() and ticket and ticket.get("kind") == "option" and ticket.get("strike") is not None
            and "{strike" not in answer and not re.search(r"\d", re.sub(r"\{[^}]*\}", "", answer))):
        # "I've drafted two of the second strike puts": the user needs to hear WHICH contract the ticket holds
        answer = ticket_sentence(ticket, world)
        display = fill(answer, book, "display")
        spoken = speakable(fill(answer, book, "spoken"), world)
    if (repairs_on() and history and not fallback and not ticket and spoken == getattr(history[-1], "spoken", None)
            and _norm_q(question) != _norm_q(getattr(history[-1], "question", ""))):
        # a different question, word-for-word the same answer: say so instead of silently re-reading it
        answer, display, spoken = SAME_AGAIN + answer, SAME_AGAIN + display, SAME_AGAIN + spoken
    emit({"type": "answer", "template": answer, "display": display, "spoken": spoken, "fallback": fallback})

    syms = [c["args"].get("symbol") for c in tool_calls if c["args"].get("symbol") not in (None, "ALL")]
    last_sym = syms[-1] if syms else (history[-1].symbol if history else None)
    topic = TOPICS.get(tool_calls[-1]["tool"]) if tool_calls else (history[-1].topic if history else None)
    frame = build_frame(question, tool_calls, book, answer, spoken, results)
    if ask_qty and pending:
        frame["pending"] = pending
    turn = Turn(question, answer, display, spoken, last_sym, topic, [c["tool"] for c in tool_calls], frame)
    timings = {"total": time.perf_counter() - t_start, "decide": decide_times, "tools": tool_times,
               "filler_at": first_filler_at}
    return TurnResult(question, answer, display, spoken, events, tool_calls, book.values(), ticket, timings,
                      fallback, turn)
