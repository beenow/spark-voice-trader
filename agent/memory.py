"""Conversation memory: the per-turn frame (filled by code), follow-up resolution, result reuse with a staleness
rule, memory commands ("start over", "forget that"), pinned user facts, and a per-passcode session store that
survives reconnects. See docs/memory.md.

Nothing here calls a model. The model sees only `state_extras()` (a compact `last_call=` / `facts=` addition to
the state line) and only when its backend was trained with it (`backend.frame_line`)."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from tools.schema import TOOL_SCHEMAS

# --------------------------------------------------------------------------- frame
FRAME_VERSION = 1
MAX_TURNS = 6                 # turns kept per session (the model sees the last 3)
MAX_FACTS = 5
MAX_WATCHLIST = 30
RETENTION_S = 24 * 3600       # turns older than this are dropped on load
# How long a fetched result may be reused instead of calling the tool again (seconds). Account data changes
# when orders fill; prices move; news and fundamentals are slow. draft_order / what_if / resolve_symbol: never.
STALE_S = {"get_quote": 60, "get_indicators": 60, "get_levels": 300, "get_option_chain": 60, "get_market": 60,
           "screen_watchlist": 300, "get_news": 900, "get_fundamentals": 3600, "get_position": 30,
           "get_portfolio": 30, "get_orders": 15}
CHAIN_ARGS = ("side", "expiry_rule")
ACCOUNT_TOOLS = ("get_position", "get_portfolio", "get_orders")


def _num(v: Any) -> Any:
    return int(v) if isinstance(v, float) and v.is_integer() else v


def visible_args(tool: str, args: dict) -> dict:
    """The call's arguments minus schema defaults the validator filled in (they add tokens and say nothing);
    an option's side and expiry are always kept, follow-ups ("and the puts") turn on them."""
    props = TOOL_SCHEMAS.get(tool, {}).get("parameters", {}).get("properties", {})
    return {k: _num(v) for k, v in args.items()
            if k in CHAIN_ARGS or not ("default" in props.get(k, {}) and props[k]["default"] == v)}


def call_text(tool: str, args: dict) -> str:
    """`get_option_chain(expiry_rule=next_week, side=call, symbol=TSLA)`: keys sorted, defaults dropped. The ONE
    function both inference (state line) and the training-data composer use, so the strings are identical."""
    a = visible_args(tool, args)
    return f"{tool}(" + ", ".join(f"{k}={a[k]}" for k in sorted(a)) + ")"


def build_frame(question: str, tool_calls: list[dict], book, answer: str, spoken: str,
                results: list[dict], now: float | None = None) -> dict:
    """The code-filled record of one turn. `results`: [{tool, args, result (ToolResult dict), names}] in call
    order (names = raw slot -> this turn's unique name). Values stay code-side; the model never sees them."""
    calls = [{"tool": c["tool"], "args": dict(c["args"])} for c in tool_calls]
    syms = []
    for c in calls:
        s = c["args"].get("symbol")
        if s and s != "ALL" and s not in syms:
            syms.append(s)
    option = {}
    for c in calls:
        if c["tool"] in ("get_option_chain", "draft_order"):
            option = {k: _num(c["args"][k]) for k in ("symbol", "side", "expiry_rule", "strike", "qty") if k in c["args"]}
    # every chain this turn, in run order (run k's slots carry the suffix _k for k >= 2)
    chains = []
    for r in results:
        if r["tool"] == "get_option_chain" and r["result"].get("ok"):
            strikes = {}
            for raw, val in (r["result"].get("slots") or {}).items():
                m = re.fullmatch(r"strike_([1-5])", raw)
                if m:
                    strikes[int(m.group(1))] = _num(val)
            mo = re.match(r"row (\w+)", str((r["result"].get("words") or {}).get("most_open_interest", "")))
            chains.append({"symbol": r["args"].get("symbol"), "side": r["args"].get("side"),
                           "expiry_rule": r["args"].get("expiry_rule"), "strikes": strikes,
                           "most_oi": {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}.get(mo.group(1) if mo else "")})
    chain = None                    # the chain the option context refers to: same symbol + side, else the last one
    if chains:
        chain = next((c for c in reversed(chains) if c["symbol"] == option.get("symbol") and c["side"] == option.get("side")),
                     chains[-1])
        if option.get("symbol") != chain["symbol"] or "side" not in option:
            option.update({k: chain[k] for k in ("symbol", "side", "expiry_rule") if chain.get(k)})
    chain_strikes, said_strike = (chain["strikes"] if chain else {}), None
    if chain:
        k = chains.index(chain) + 1
        # "those" = the contract the answer priced (its bid / ask row), else the first strike it named
        priced = [int(r) for f, r, run in re.findall(r"\{(bid|ask)_([1-5])(?:_(\d+))?\}", answer or "") if int(run or 1) == k]
        if priced and len(set(priced)) == 1:
            said_strike = chain_strikes.get(priced[0])
        for row, run in ([] if said_strike is not None else re.findall(r"\{strike_([1-5])(?:_(\d+))?\}", answer or "")):
            if int(run or 1) == k:
                said_strike = chain_strikes.get(int(row))
                break
        if said_strike is None and chain["most_oi"]:
            said_strike = chain_strikes.get(chain["most_oi"])
    if said_strike is not None:
        option["said_strike"] = said_strike
    return {"v": FRAME_VERSION, "ts": now if now is not None else time.time(), "question": question,
            "calls": calls, "symbols": syms, "option": option, "chain_strikes": chain_strikes,
            "slots": sorted(book.names()) if book is not None else [], "answer": answer, "spoken": spoken,
            "results": [{"tool": r["tool"], "args": dict(r["args"]), "result": r["result"]} for r in results]}


def state_extras(history: list, facts: list[str] | None = None) -> list[str]:
    """Extra `key=value` parts for the state line: the last tool-calling turn's calls (at most 2) and the
    pinned facts. Order and wording are part of the trained format (FORMAT.md)."""
    parts = []
    for t in reversed(history or []):
        fr = getattr(t, "frame", None) or {}
        if fr.get("calls"):
            parts.append("last_call=" + "; ".join(call_text(c["tool"], c["args"]) for c in fr["calls"][:2]))
            break
    if facts:
        parts.append("facts=" + "; ".join(facts[:MAX_FACTS]))
    return parts


# --------------------------------------------------------------------------- follow-up resolution (code side)
_ORD = {"first": 1, "1st": 1, "top": 1, "one": 1, "second": 2, "2nd": 2, "two": 2, "third": 3, "3rd": 3,
        "three": 3, "fourth": 4, "4th": 4, "four": 4, "fifth": 5, "5th": 5, "five": 5, "last": 5}
_ORD_RE = re.compile(r"\b(?:the\s+)?(first|1st|top|second|2nd|third|3rd|fourth|4th|fifth|5th|last)\s+(?:strike|one|row|contract)"
                     r"|\brow\s+(one|two|three|four|five)\b", re.I)
_DEICTIC = re.compile(r"\b(those|these|them|that one|that contract|that strike|it|that)\b", re.I)
_EXPIRY_WORDS = re.compile(r"\b(this week|next week|monthly|month|friday|expir\w*|weekly|weeklies)\b|"
                           r"\b(?:nearest|soonest|closest)\s+(?:expir\w*|date|one)", re.I)
# "calls" / "puts" / "the 650 call": a side. "put in an order", "call it three", "call me": verbs, not a side.
_SIDE_WORDS = re.compile(r"\b(calls|puts)\b|\b(call|put)\b(?!\s+(?:in|on|a|an|me|it|that|them|up|through|out|off|"
                         r"together|down|together|my|the\s+order)\b)", re.I)
_EXPIRY_SAID = {r"\bthis\s+week\b|\bthis\s+friday\b|\bweeklies\b": "this_week", r"\bnext\s+week\b|\bnext\s+friday\b": "next_week",
                r"\bmonthl(?:y|ies)\b|\bthis\s+month\b|\bthe\s+month\b|\b(?:january|february|march|april|june|july|august|"
                r"september|october|november|december)\b": "monthly",
                r"\b(?:nearest|soonest|closest)\s+(?:expir\w*|date)|\bexpir\w*\s+soonest\b": "nearest"}
_INDICATOR_SAID = {r"\brsi\b|\br\s*s\s*i\b|overbought|oversold": "rsi",
                   r"moving\s+averages?|\bema\b|\be\s*m\s*a\b|\b(?:9|21|50|nine|twenty-?one|fifty)[\s-]*day": "ema",
                   r"\bvolume\b": "volume"}
_TIMEFRAME_SAID = {r"\b(?:5|five)[\s-]*min(?:ute)?s?\b|\b5m\b": "5m", r"\bhourly\b|\b(?:1|one|an)[\s-]*hour\b|\b1h\b": "1H",
                   r"\bdaily\b|\bday\s+chart\b": "1D"}
_SHARES = re.compile(r"\b(shares?|stock)\b", re.I)
_OPTION_WORDS = re.compile(r"\b(calls?|puts?|contracts?|strike|options?|those|them|that one|these)\b", re.I)
_CARRY = re.compile(r"^\W*(?:ok(?:ay)?\W+|so\W+)?(?:(?:do\s+)?(?:the\s+)?same(?:\s+thing)?\b|now\s+do\b|and\b|what\s+about\b|"
                    r"how\s+about\b|i\s+mean\b|no\W+i\s+mean\b|not\s+that\s+one\b|sorry\b|wrong\s+one\b)", re.I)


def last_frame(history: list) -> dict | None:
    for t in reversed(history or []):
        fr = getattr(t, "frame", None)
        if fr and fr.get("calls"):
            return fr
    return None


def repairs_on() -> bool:
    """Code-side argument repairs (follow-up filling, words the user said, argument repair, code answers for
    timeframes / arithmetic, the no-data retry, validator coercions). VOICE_REPAIRS=0 turns them all off, so an
    eval can show what the model does on its own."""
    return os.environ.get("VOICE_REPAIRS", "1") != "0"


# Phone speech recognition mishears in an options conversation (seen on the Fold, Nov 4): "Tesla calls" ->
# "Tesla cost", "and the puts" -> "and the post" (which then went to get_news). Only rewritten when the turn is about
# options: the previous turn showed a chain or drafted an option, or this question names an expiry or a strike.
_OPTION_TALK = re.compile(r"\b(?:next\s+week'?s?|this\s+week'?s?|weekly|weeklies|monthly|expir\w*|strikes?|"
                          r"chain|options?|calls?|puts?)\b", re.I)
_HEARD_CALLS = re.compile(r"\b(?:cost|costs|cause|cars|coals|colds|cols)\b", re.I)
_HEARD_PUTS = re.compile(r"\b(?:post|posts|puds|pots|putts|put's)\b", re.I)


def stt_repairs(question: str, history: list | None = None) -> str:
    """Undo known speech-recognition mishears (see above) and "what driving X" -> "what's driving X"."""
    if not repairs_on():
        return question
    q = re.sub(r"^(\s*)(w)hat\s+(driving|moving|happening)\b", r"\1\2hat's \3", question, flags=re.I)
    fr = last_frame(history or [])
    options = bool(fr and any(c["tool"] in ("get_option_chain", "draft_order") for c in fr.get("calls", []))) \
        or bool(_OPTION_TALK.search(q))
    if options:
        q = _HEARD_CALLS.sub("calls", q)
        q = _HEARD_PUTS.sub("puts", q)
    return q


_RANK_SAID = {r"\bpercent(?:age)?\b|\bpct\b": "pnl_pct",
              r"\bby\s+weight\b|\b(?:largest|biggest)\s+(?:holding|position)s?\b|\bconcentrat": "weight",
              r"\b(?:moved|moving|movers?)\b.*\btoday\b|\btoday'?s?\s+(?:biggest\s+)?movers?\b": "day_move"}


def resolve_followup(tool: str, args: dict, question: str, history: list) -> tuple[dict, list[str]]:
    """Fill arguments the follow-up left implicit, from the last frame. Returns (args, filled keys; "said:k" =
    set from the question's own words — one side / one expiry named — which beat the model's value).
    Fills only what the model left OUT and the question does not state; the one override is an ordinal row
    reference ("the second strike"), which is resolved to that chain row's strike."""
    if not isinstance(args, dict) or not repairs_on():
        return args, []
    args, filled = dict(args), []
    if tool in ("draft_order", "get_option_chain"):
        # "buy 3 Tesla 450 calls", "Apple 235 calls next week", "sell three of the three seventies", "the 280s":
        # the one strike the user said (for an order, not the quantity) beats the model's number
        said = said_strikes(question)
        qty = args.get("qty")
        if len(said) == 1 and (tool == "get_option_chain" or (qty is not None and float(qty) != said[0])) \
                and args.get("strike") != said[0]:
            args["strike"], filled = _num(said[0]), filled + ["said:strike"]
    if tool == "get_orders":
        # "open orders" / "filled orders" say the status; "how many orders did I place / send yesterday" means all
        st = {v for k, v in ((r"\bopen\b|\bpending\b|\bworking\b", "open"), (r"\bfilled\b|\bexecuted\b", "filled"))
              if re.search(k, question, re.I)}
        if len(st) == 1 and args.get("status") != next(iter(st)):
            args["status"], filled = next(iter(st)), filled + ["said:status"]
        elif not st and re.search(r"\b(?:did|have)\s+i\s+(?:\w+\s+){0,2}(?:send|sent|place|placed|put|make|made|submit|enter)",
                                  question, re.I) and args.get("status") != "all":
            args["status"], filled = "all", filled + ["said:status"]
    if tool == "get_orders":
        # "anything filled today?" / "yesterday's orders": the one day the user named
        days = {d for d in ("today", "yesterday") if re.search(rf"\b{d}\b", question, re.I)}
        if len(days) == 1 and args.get("day") != next(iter(days)):
            args["day"], filled = next(iter(days)), filled + ["said:day"]
    if tool == "get_portfolio":
        # the one ranking the user named: "in percent", "by weight", "biggest movers today"
        rk = {v for k, v in _RANK_SAID.items() if re.search(k, question, re.I)}
        if len(rk) == 1 and args.get("rank_by") != next(iter(rk)):
            args["rank_by"], filled = next(iter(rk)), filled + ["said:rank_by"]
    if tool == "get_indicators":
        # the indicator and timeframe the user named; an unnamed timeframe is the default (daily)
        ind = {v for k, v in _INDICATOR_SAID.items() if re.search(k, question, re.I)}
        if len(ind) == 1 and args.get("indicator") != next(iter(ind)):
            args["indicator"], filled = next(iter(ind)), filled + ["said:indicator"]
        tf = {v for k, v in _TIMEFRAME_SAID.items() if re.search(k, question, re.I)}
        if len(tf) == 1 and args.get("timeframe") != next(iter(tf)):
            args["timeframe"], filled = next(iter(tf)), filled + ["said:timeframe"]
        elif not tf and args.get("timeframe") not in (None, "1D"):
            args["timeframe"], filled = "1D", filled + ["said:timeframe"]
    stock_order = tool == "draft_order" and (bool(_SHARES.search(question)) or not (
        any(k in args for k in ("side", "strike", "expiry_rule")) or _OPTION_WORDS.search(question)))
    if stock_order:                 # a share order: never turn it into an option order
        return args, filled
    if tool in ("get_option_chain", "draft_order"):
        # words the user SAID beat the model: one side named, one expiry named
        sides = {(a or b).lower().rstrip("s") for a, b in _SIDE_WORDS.findall(question)}
        if len(sides) == 1 and args.get("side") != next(iter(sides)):
            args["side"], filled = next(iter(sides)), filled + ["said:side"]
        exp = {v for k, v in _EXPIRY_SAID.items() if re.search(k, question, re.I)}
        if len(exp) == 1 and args.get("expiry_rule") != next(iter(exp)):
            args["expiry_rule"], filled = next(iter(exp)), filled + ["said:expiry_rule"]
    fr = last_frame(history)
    if not fr:
        return args, filled
    props = TOOL_SCHEMAS.get(tool, {}).get("parameters", {}).get("properties", {})
    prev_sym = fr["symbols"][-1] if fr["symbols"] else None
    if "symbol" in props and not args.get("symbol") and prev_sym:
        args["symbol"], filled = prev_sym, filled + ["symbol"]
    opt = fr.get("option") or {}
    same = opt.get("symbol") and opt.get("symbol") == str(args.get("symbol", "")).upper()
    if tool in ("get_option_chain", "draft_order") and opt:
        # The user heard one chain (or ticket). When the question says no side / expiry, they are that
        # chain's: for the same symbol, or when the question carries the request over ("same for AMD",
        # "what about AMD") the frame's value wins over the model's guess; otherwise it only fills a gap.
        carry = same or bool(_CARRY.match(question))
        for key, words in (("side", _SIDE_WORDS), ("expiry_rule", _EXPIRY_WORDS)):
            if opt.get(key) and not words.search(question) and (key not in args or (carry and args[key] != opt[key])):
                args[key], filled = opt[key], filled + [key]
    if tool == "draft_order" and same:
        m = _ORD_RE.search(question)
        n = _ORD.get((m.group(1) or m.group(2)).lower()) if m else None
        strikes = {int(k): v for k, v in (fr.get("chain_strikes") or {}).items()}
        if m and (m.group(1) or "").lower() == "last" and strikes:
            n = max(strikes)
        if n and n in strikes:
            # "the second strike" names a row of the chain the user just heard: that row's strike, whatever
            # number the model wrote (it never saw the values)
            if args.get("strike") != strikes[n]:
                args["strike"], filled = strikes[n], filled + ["strike"]
        elif "strike" not in args and _DEICTIC.search(question) and opt.get("said_strike") is not None:
            args["strike"], filled = opt["said_strike"], filled + ["strike"]
    return args, filled


_W1 = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_W10 = {"ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90}
_W10S = {k + "s": v for k, v in _W10.items()} | {k[:-1] + "ies": v for k, v in _W10.items()}   # "seventys" / "seventies"
_STRIKE_TAIL = r"(?:\s*(?:calls?|puts?|strike|c\b|p\b))"


def said_strikes(question: str) -> list[float]:
    """Strikes said in the question: "450 calls", "the 450 strike", "the 280s", "three seventy calls",
    "the three seventies", "one seventy five puts". Distinct values, in order."""
    out = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)" + _STRIKE_TAIL, question, re.I)]
    out += [float(x) for x in re.findall(r"\b(\d{2,4})s\b", question)]
    tens = "|".join(list(_W10) + list(_W10S))
    for h, t, o, tail in re.findall(r"\b(" + "|".join(_W1) + r")\s+(" + tens + r")(?:[\s-]+(" + "|".join(_W1) + r"))?\b(\s*(?:calls?|puts?|strike|s\b)?)",
                                    question, re.I):
        plural = t.lower() in _W10S
        if not (tail.strip() or plural):
            continue
        out.append(float(_W1[h.lower()] * 100 + (_W10S.get(t.lower()) or _W10[t.lower()]) + (_W1.get(o.lower(), 0) if o else 0)))
    # "the six hundred calls", "nine hundred puts", "the four hundreds"
    for h, plural, tail in re.findall(r"\b(" + "|".join(_W1) + r")\s+hundred(s?)\b(\s*(?:calls?|puts?|strike)?)", question, re.I):
        if tail.strip() or plural:
            out.append(float(_W1[h.lower()] * 100))
    # "puts around two twenty", "calls at 250", "the call near three ten"
    for m in re.finditer(r"\b(?:calls?|puts?|strike)\s+(?:at|around|near|by)\s+(?:the\s+)?(\d+(?:\.\d+)?|(?:"
                         + "|".join(_W1) + r")\s+(?:hundred|" + "|".join(_W10) + r")(?:[\s-]+(?:" + "|".join(_W1) + r"))?)\b",
                         question, re.I):
        if re.match(r"\s*(?:dollars?|bucks|cents|each|per\b|a\s+contract|limit)", question[m.end():], re.I):
            continue                                   # "calls at 3.50 limit": a price, not a strike
        v = _strike_words(m.group(1))
        if v is not None:
            out.append(v)
    return list(dict.fromkeys(out))


def _strike_words(text: str) -> float | None:
    """"450" / "five hundred" / "four fifty" / "two seventy five" -> the number, else None."""
    t = text.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return float(t)
    w = re.split(r"[\s-]+", t)
    if not w or w[0] not in _W1:
        return None
    if len(w) == 2 and w[1] == "hundred":
        return float(_W1[w[0]] * 100)
    if len(w) in (2, 3) and w[1] in _W10 and (len(w) == 2 or w[2] in _W1):
        return float(_W1[w[0]] * 100 + _W10[w[1]] + (_W1[w[2]] if len(w) == 3 else 0))
    return None


def qty_is_strike(args: dict, question: str) -> bool:
    """draft_order with qty = the number the user said as a STRIKE ("the 650 call") and no other number: the
    model read the strike as the quantity."""
    q = args.get("qty")
    if q is None:
        return False
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", question)
    try:
        qn = float(q)
    except (TypeError, ValueError):
        return False
    if args.get("strike") is not None and float(args["strike"]) != qn:
        return False
    strikes = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:strike|call\b|put\b|c\b|p\b)", question, re.I)]
    strikes += [float(x) for x in re.findall(r"\bthe\s+(\d+(?:\.\d+)?)\s*(?:calls|puts)\b", question, re.I)]
    return qn in strikes and len(nums) == 1 and qn >= 20


def cached_result(tool: str, args: dict, history: list, now: float | None = None) -> dict | None:
    """A result fetched in an earlier turn for the same tool + arguments and still fresh (STALE_S)."""
    ttl = STALE_S.get(tool)
    if not ttl:
        return None
    now = time.time() if now is None else now
    for t in reversed(history or []):
        fr = getattr(t, "frame", None) or {}
        if tool in ACCOUNT_TOOLS and any(c["tool"] == "draft_order" for c in fr.get("calls") or []):
            return None             # a ticket since then: account data may have changed
        if now - fr.get("ts", 0) > ttl:
            continue
        for r in fr.get("results") or []:
            if r["tool"] == tool and r["args"] == args and r["result"].get("ok"):
                res = copy.deepcopy(r["result"])
                (res.get("words") or {}).pop("resolved_from", None)     # that turn's tag, not this one's
                return res
    return None


def pending_order(history: list, question: str) -> dict | None:
    """The draft_order the last turn held back for a quantity, completed with the quantity this question
    gives ("three", "make it 5", "two contracts"), else None."""
    if not history:
        return None
    fr = getattr(history[-1], "frame", None) or {}
    pend = fr.get("pending")
    if not pend:
        return None
    m = re.fullmatch(r"\W*(?:ok(?:ay)?\W+|uh\W+|um\W+)?(?:make\s+it\s+|just\s+|let'?s\s+do\s+|do\s+)?"
                     r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|fifty|a\s+hundred)"
                     r"(?:\s+(?:contracts?|shares?|of\s+them|please))*\W*", question, re.I)
    if not m:
        return None
    w = m.group(1).lower()
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "twenty": 20, "fifty": 50, "a hundred": 100}
    qty = int(w) if w.isdigit() else words.get(re.sub(r"\s+", " ", w))
    return dict(pend, qty=qty) if qty else None


_PCT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
              "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty five": 25, "thirty": 30,
              "forty": 40, "fifty": 50}
_WHATIF_FU = re.compile(
    r"\W*(?:ok(?:ay)?\W+|uh\W+|um\W+)?(?:and|what\s+about|how\s+about|what\s+if|try|now)?\s*(?:if\s+)?(?:it\s+|that\s+)?"
    r"(?P<verb>drops?|falls?|goes\s+down|tanks?|rallies|rally|rises?|goes\s+up|jumps?)?\s*(?:by\s+)?"
    r"(?P<num>\d+(?:\.\d+)?|twenty[\s-]five|" + "|".join(sorted(_PCT_WORDS, key=len, reverse=True)) + r")"
    r"(?:\s*(?:percent|%|pct))?(?:\s+(?:instead|then))?\W*", re.I)


def whatif_followup(history: list, question: str) -> dict | None:
    """A bare re-run of the last turn's what_if with a new move ("and twenty", "what about fifteen percent",
    "what if it drops thirty"): same symbol, the previous direction unless a direction word is said."""
    if not history or not repairs_on():
        return None
    fr = getattr(history[-1], "frame", None) or {}
    wi = [c for c in fr.get("calls") or [] if c.get("tool") == "what_if"]
    if len(wi) != 1:
        return None
    m = _WHATIF_FU.fullmatch(question)
    if not m:
        return None
    w = re.sub(r"[\s-]+", " ", m.group("num").lower())
    n = float(w) if re.fullmatch(r"\d+(?:\.\d+)?", w) else _PCT_WORDS.get(w)
    if not n or n > 100:
        return None
    prev = wi[0]["args"]
    try:
        sign = -1.0 if float(prev.get("move_pct") or -1) < 0 else 1.0
    except (TypeError, ValueError):
        sign = -1.0
    verb = (m.group("verb") or "").lower()
    if verb:
        sign = -1.0 if re.match(r"drop|fall|goes\s+down|tank", verb) else 1.0
    if not prev.get("symbol"):
        return None
    return {"symbol": prev["symbol"], "move_pct": _num(sign * n)}


# --------------------------------------------------------------------------- commands + facts (code side)
_START_OVER = re.compile(r"^\W*(?:ok(?:ay)?\W+|hey\W+)?(?:let'?s\s+)?(?:start\s+over|start\s+fresh|forget\s+everything|"
                         r"clear\s+(?:the\s+|our\s+|my\s+)?(?:conversation|chat|memory|history)|new\s+conversation|reset)\W*$", re.I)
_FORGET_THAT = re.compile(r"^\W*(?:ok(?:ay)?\W+)?(?:forget\s+(?:that|it|what\s+i\s+(?:just\s+)?said)|scratch\s+that|"
                          r"never\s*mind(?:\s+that)?|disregard\s+that)\W*$", re.I)
_REPEAT = re.compile(r"^\W*(?:sorry\W+|um\W+|uh\W+)?(?:(?:can|could)\s+you\s+)?(?:say\s+(?:that|it)\s+again|repeat\s+(?:that|it)"
                     r"|come\s+again|what\s+was\s+that|i\s+didn'?t\s+(?:catch|hear)\s+(?:that|you|it))(?:\s+please)?\W*$", re.I)
_SECRET = re.compile(r"\d|@|password|passcode|\bpin\b|account|ssn|social security|routing|card number|secret|token", re.I)
_NAME = re.compile(r"^\W*(?:please\s+)?(?:call\s+me|my\s+name\s+is|i'?m\s+called)\s+([A-Za-z][A-Za-z'\-]{1,20})\W*$", re.I)
_NOT_NAMES = {"later", "back", "tomorrow", "now", "today", "tonight", "soon", "maybe", "anytime", "whenever",
              "again", "please", "when", "if", "after", "before", "sometime", "crazy", "out"}
_TRADES = re.compile(r"^\W*(?:fyi\W+|just\s+so\s+you\s+know\W+)?i\s+(?:mostly|mainly|usually|only|mostly\s+just)\s+trade\s+(.{2,60}?)\W*$"
                     r"|^\W*i\s+trade\s+(?:mostly|mainly)\s+(.{2,60}?)\W*$", re.I)


def memory_command(question: str) -> str | None:
    """'all' (start over), 'last' (forget that), 'repeat', or None."""
    q = question.strip()
    if _START_OVER.match(q):
        return "all"
    if _FORGET_THAT.match(q):
        return "last"
    if _REPEAT.match(q):
        return "repeat"
    return None


def parse_fact(question: str, world=None) -> tuple[str, str] | None:
    """(fact text, spoken acknowledgement) for a pinned preference, else None. Never anything secret-like."""
    q = question.strip()
    m = _NAME.match(q)
    if m and not _SECRET.search(m.group(1)) and m.group(1).lower() not in _NOT_NAMES:
        name = m.group(1).capitalize()
        return f"name {name}", f"Got it, {name}."
    m = _TRADES.match(q)
    if m:
        what = (m.group(1) or m.group(2)).strip()
        if _SECRET.search(what):
            return None
        syms = []
        if world is not None:
            from .validator import named_companies, tickers_in
            syms = [s for s, _, _ in sorted(named_companies(what, world) + tickers_in(what, world), key=lambda x: x[1])]
            syms = list(dict.fromkeys(syms))
        if syms:
            names = [world.universe[s].name for s in syms]
            return "trades mostly " + " ".join(syms), "Got it, I'll keep " + " and ".join(names) + " in mind."
        if re.fullmatch(r"(?:options|calls|puts|stocks|shares|etfs|crypto)(?:\s+and\s+\w+)?", what, re.I):
            return f"trades mostly {what.lower()}", f"Got it, mostly {what.lower()}."
    return None


# --------------------------------------------------------------------------- persistence
def session_key(passcode: str) -> str:
    """Store key: a salted hash of the passcode, never the passcode itself."""
    return hashlib.sha256(("leaps-voice-memory:" + passcode).encode()).hexdigest()[:24]


def _redact(text: str) -> str:
    text = re.sub(r"\S+@\S+", "[email]", text or "")
    return re.sub(r"\d[\d\s-]{6,}\d", "[number]", text)


class SessionStore:
    """One JSON file per session key under `root` (0600, written atomically): {"v", "turns": [...], "facts": [...]}.
    Turns are Turn fields + frame; turns older than RETENTION_S are dropped on load."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

    def path(self, key: str) -> Path:
        assert re.fullmatch(r"[0-9a-f]{8,64}", key), "bad session key"
        return self.root / f"{key}.json"

    def load(self, key: str) -> dict:
        """{"turns": [...], "facts": [...]}; a missing, unreadable or malformed file is an empty conversation."""
        try:
            d = json.loads(self.path(key).read_text())
            if not isinstance(d, dict):
                raise ValueError("not an object")
            cutoff = time.time() - RETENTION_S
            turns = [t for t in d.get("turns") or [] if isinstance(t, dict) and isinstance(t.get("question"), str)
                     and (t.get("ts") or (t.get("frame") or {}).get("ts") or 0) >= cutoff]
            facts = [f for f in d.get("facts") or [] if isinstance(f, str)][:MAX_FACTS]
            wl = d.get("watchlist")
            wl = [w for w in wl if isinstance(w, str)][:MAX_WATCHLIST] if isinstance(wl, list) else None
            return {"turns": turns, "facts": facts, "watchlist": wl}
        except (OSError, ValueError, TypeError, AttributeError):
            return {"turns": [], "facts": [], "watchlist": None}

    def save(self, key: str, turns: list[dict], facts: list[str], watchlist: list[str] | None = None) -> None:
        p = self.path(key)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-")
        try:
            with os.fdopen(fd, "w") as f:
                doc = {"v": FRAME_VERSION, "turns": turns[-MAX_TURNS:], "facts": facts[:MAX_FACTS]}
                if watchlist is not None:
                    doc["watchlist"] = list(watchlist)[:MAX_WATCHLIST]
                json.dump(doc, f, ensure_ascii=False, default=str)
            os.chmod(tmp, 0o600)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def clear(self, key: str) -> None:
        try:
            self.path(key).unlink()
        except FileNotFoundError:
            pass


def turn_to_dict(t, now: float | None = None) -> dict:
    """What is written to disk for one turn: text fields redacted; tool results only while still reusable."""
    now = time.time() if now is None else now
    fr = copy.deepcopy(getattr(t, "frame", None))
    if fr:
        for k in ("question", "answer", "spoken"):
            if isinstance(fr.get(k), str):
                fr[k] = _redact(fr[k])
        if now - fr.get("ts", 0) > max(STALE_S.values()):
            fr["results"] = []
    return {"question": _redact(t.question), "answer": _redact(t.answer), "display": _redact(t.display),
            "spoken": _redact(t.spoken), "symbol": t.symbol, "topic": t.topic, "tools": list(t.tools),
            "ts": (fr or {}).get("ts", now), "frame": fr}


def turn_from_dict(d: dict):
    from .loop import Turn
    return Turn(d["question"], d.get("answer", ""), d.get("display", ""), d.get("spoken", ""), d.get("symbol"),
                d.get("topic"), list(d.get("tools") or []), d.get("frame"))


class Conversation:
    """One user's conversation: history (Turn objects with frames), pinned facts, optional persistence.
    `ask()` handles memory commands, pinned facts and "say that again" in code; everything else goes to `run`
    (normally agent.loop.run_turn) with the history and facts."""

    def __init__(self, store: SessionStore | None = None, key: str | None = None):
        self.store, self.key = store, key
        self.turns: list = []
        self.facts: list[str] = []
        self.watchlist: list[str] | None = None   # None = the world's own watchlist
        self.gen = 0                 # bumped by clear(): a turn that started before a clear is not recorded
        self.closed = False          # set when the connection ends: a turn still running is not recorded
        self._lock = threading.Lock()
        if store is not None and key:
            d = store.load(key)
            for t in d["turns"]:
                try:
                    self.turns.append(turn_from_dict(t))
                except (KeyError, TypeError, ValueError):
                    continue
            self.facts = d["facts"]
            self.watchlist = d.get("watchlist")

    @property
    def history(self) -> list:
        return list(self.turns)

    def _persist(self) -> None:
        if self.store is not None and self.key:
            turns = [turn_to_dict(t) for t in self.turns if hasattr(t, "question") and hasattr(t, "tools")]
            if not turns and not self.facts and self.watchlist is None:
                self.store.clear(self.key)
            else:
                self.store.save(self.key, turns, self.facts, self.watchlist)

    def clear(self, scope: str = "all") -> None:
        with self._lock:
            self.gen += 1
            if scope == "all":
                self.turns, self.facts = [], []
            elif self.turns:
                last = self.turns.pop()
                fact = (getattr(last, "frame", None) or {}).get("pinned_fact")
                if fact in self.facts:
                    self.facts.remove(fact)
            self._persist()

    def record(self, result, gen: int | None = None) -> None:
        """Add a finished turn. A non-Turn result (the fake agent's dict) is kept in memory, not on disk."""
        t = getattr(result, "turn", None)
        if t is None:
            t = result if isinstance(result, dict) else None
        if t is None:
            return
        with self._lock:
            if self.closed or (gen is not None and gen != self.gen):
                return
            self.turns = (self.turns + [t])[-MAX_TURNS:]
            self._persist()

    def ask(self, question: str, run: Callable[..., Any], on_event: Callable[[dict], None] | None = None,
            world=None, record: bool = True, history: list | None = None):
        """`run(question, history, facts)` -> TurnResult. `history` overrides the conversation's (compare lane 1)."""
        emit = on_event or (lambda e: None)
        gen = self.gen
        fixed = stt_repairs(question, self.history if history is None else history)
        if fixed != question:
            emit({"type": "stt_repair", "heard": question, "used": fixed})
            question = fixed
        cmd = memory_command(question)
        if cmd in ("all", "last"):
            if record:
                self.clear(cmd)
            text = "Okay, starting fresh." if cmd == "all" else "Okay, I've forgotten that."
            emit({"type": "memory_cleared", "scope": cmd})
            return _code_reply(question, text, emit, record=False)
        if cmd == "repeat":
            hist = self.history if history is None else history
            prev = hist[-1] if hist else None
            said = (getattr(prev, "spoken", None) or getattr(prev, "display", None)) if prev is not None else None
            if isinstance(prev, dict):
                said = prev.get("spoken") or prev.get("display")
            text = said or "There's nothing to repeat yet. What would you like to know?"
            prev = prev if said else None
            return _code_reply(question, text, emit, record=False, replayed=prev is not None)
        fact = parse_fact(question, world)
        if fact is not None:
            if record and fact[0] not in self.facts:
                self.facts = (self.facts + [fact[0]])[-MAX_FACTS:]
            res = _code_reply(question, fact[1], emit, record=True)
            prev = (self.history if history is None else history)[-1:] or [None]
            if prev[0] is not None:            # a pinned fact does not change what "it" refers to
                res.turn.symbol, res.turn.topic = getattr(prev[0], "symbol", None), getattr(prev[0], "topic", None)
            res.turn.frame = {"v": FRAME_VERSION, "ts": time.time(), "question": question, "calls": [],
                              "symbols": [], "pinned_fact": fact[0]}
            if record:
                self.record(res, gen)
            return res
        if world is not None and self.watchlist is not None:
            world.watchlist = list(self.watchlist)          # this session's watchlist (screens use it too)
        result = run(question, self.history if history is None else history, list(self.facts))
        fr = getattr(getattr(result, "turn", None), "frame", None) or {}
        if record and isinstance(fr.get("watchlist"), list):
            self.watchlist = list(fr["watchlist"])          # an add / remove: persisted with the session
        if record:
            self.record(result, gen)
        return result


def _code_reply(question: str, text: str, emit, record: bool, replayed: bool = False):
    from .loop import Turn, TurnResult
    ev = {"type": "answer", "template": text, "display": text, "spoken": text, "fallback": False, "code": True}
    if replayed:
        ev["replayed"] = True
    emit(ev)
    turn = Turn(question, text, text, text, None, None, [], None)
    return TurnResult(question, text, text, text, [ev], [], {}, None, {"total": 0.0, "decide": [], "tools": []},
                      False, turn)


# --------------------------------------------------------------------------- training-data side
STATE_KEYS = ("last_symbol", "last_topic", "last_call", "facts")   # the pack's trace.render.state_keys (frame models)


def context_entries(history: list, facts: list[str] | None = None) -> list[dict]:
    """Framework row `context` for a turn whose history is `history` (Turn objects with frames): one entry per
    prior turn ({q, route, slots, state, answer=spoken}); the LAST entry's state carries the frame keys, so the
    framework's state line (pack state_keys = STATE_KEYS) is byte-identical to build_messages(frame_line=True)."""
    hist = list(history or [])[-3:]
    out = []
    for i, t in enumerate(hist):
        state = {k: v for k, v in (("last_symbol", t.symbol), ("last_topic", t.topic)) if v}
        if i == len(hist) - 1:
            for part in state_extras(hist, facts):
                k, _, v = part.partition("=")
                state[k] = v
        out.append({"q": t.question, "route": "tool" if t.tools else "chat",
                    "slots": {"symbol": t.symbol} if t.symbol else {}, "state": state,
                    "answer": t.spoken or t.answer})
    return out
