"""Tool-call and answer validation."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from tools import TOOL_SCHEMAS
from tools.impl import resolve_symbol

SLOT_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SENT_SPLIT = re.compile(r"[.!?]+(?:\s+|$)")
SENT_CLAIMS = re.compile(r"\b(placed|executed|submitted|sent|filled|bought|sold)\b", re.I)


def no_digits(s: str) -> str:
    return re.sub(r"\d", "", s)


@dataclass
class CallCheck:
    ok: bool
    name: str
    args: dict = field(default_factory=dict)
    error: str | None = None
    words: dict = field(default_factory=dict)
    resolved: tuple[str, str] | None = None   # (said, symbol) when a symbol was routed through resolve_symbol


def _coerce(spec: dict, v):
    t = spec.get("type")
    if t == "integer":
        if isinstance(v, bool):
            raise ValueError
        if isinstance(v, str):
            v = float(v.strip().replace(",", ""))
        if float(v) != int(float(v)):
            raise ValueError
        return int(float(v))
    if t == "number":
        if isinstance(v, bool):
            raise ValueError
        if isinstance(v, str):
            v = v.strip().replace(",", "").rstrip("%").lstrip("$")
        return float(v)
    if t == "string":
        if isinstance(v, (dict, list)) or v is None:
            raise ValueError
        v = str(v).strip()
        if "enum" in spec:
            low = {e.lower(): e for e in spec["enum"]}
            if v.lower() not in low:
                raise ValueError
            v = low[v.lower()]
        return v
    return v


def validate_call(world, name: str, args) -> CallCheck:
    """Tool exists, args match schema (with light coercion), symbol in universe (else resolve_symbol)."""
    name = (name or "").split(".")[-1].strip()
    if name not in TOOL_SCHEMAS:
        return CallCheck(False, no_digits(name) or "unknown", error="unknown tool")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return CallCheck(False, name, error="arguments must be an object")
    params = TOOL_SCHEMAS[name]["parameters"]
    props, req = params["properties"], params["required"]
    clean: dict = {}
    repairs = os.environ.get("VOICE_REPAIRS", "1") != "0"     # agent.memory.repairs_on (no import cycle)
    if repairs and name == "draft_order" and str(args.get("type", "")).lower() in ("call", "put", "calls", "puts"):
        # the option side written into `type` (market/limit): it is the side
        args = dict(args)
        args.setdefault("side", str(args.pop("type")).lower().rstrip("s"))
        args.pop("type", None)
    for k, v in args.items():
        if k not in props or v is None or v == "":
            continue                                   # unknown / empty args are dropped
        if repairs and k == "symbol" and isinstance(v, str) and re.fullmatch(r"\$?(?:[A-Za-z][ .\-]){1,4}[A-Za-z]", v.strip()):
            v = re.sub(r"[ .\-]", "", v.strip())      # "S M H" / "S.M.H" -> "SMH"
        if repairs and k == "expiry_rule" and isinstance(v, str) and v not in props[k].get("enum", []):
            low = v.lower()                            # "monthly expiry", "next friday", "this_friday" -> the rule
            v = ("monthly" if "month" in low else "next_week" if "next" in low else
                 "nearest" if ("near" in low or "soon" in low) else "this_week" if ("this" in low or "week" in low) else v)
        try:
            clean[k] = _coerce(props[k], v)
        except (ValueError, TypeError):
            return CallCheck(False, name, error=f"bad value for {no_digits(k)}")
    for k in req:
        if k not in clean:
            return CallCheck(False, name, error=f"missing {k}")
    for k, spec in props.items():
        if k not in clean and "default" in spec:
            clean[k] = spec["default"]
    resolved = None
    if "symbol" in clean:
        sym = clean["symbol"].upper().lstrip("$").strip()
        clean["symbol"] = sym
        if not (name == "what_if" and sym == "ALL") and not world.has(sym):
            r = resolve_symbol(world, clean["symbol"])
            if r.ok and r.words.get("confidence") == "high":
                resolved = (sym, r.words["symbol"])
                clean["symbol"] = r.words["symbol"]
            else:
                alts = r.words.get("alternatives", [])
                if r.ok:
                    alts = [r.words["symbol"]] + [a for a in alts if a != r.words["symbol"]]
                return CallCheck(False, name, error="unknown symbol", words={"alternatives": alts[:3]})
    if name == "draft_order" and clean.get("qty", 0) <= 0:
        return CallCheck(False, name, error="quantity must be positive")
    return CallCheck(True, name, clean, resolved=resolved)


def sentences(text: str) -> int:
    body = SLOT_RE.sub("X", text).strip()
    parts = [p for p in _SENT_SPLIT.split(body) if p.strip()]
    return len(parts)


def check_answer(text: str, slot_names, draft_ran: bool = False) -> list[str]:
    """Problems with a model answer template (empty list = ok)."""
    problems = []
    if not text or not text.strip():
        return ["empty answer"]
    used = SLOT_RE.findall(text)
    missing = [s for s in used if s not in slot_names]
    if missing:
        problems.append("unknown slots: " + ", ".join("{" + no_digits(s) + "}" for s in missing))
    if re.search(r"\d", SLOT_RE.sub("", text)):
        problems.append("digits outside slots")
    if "{" in SLOT_RE.sub("", text) or "}" in SLOT_RE.sub("", text):
        problems.append("malformed slot braces")
    if sentences(text) > 2:
        problems.append("more than two sentences")
    if draft_ran and SENT_CLAIMS.search(text) and "confirm" not in text.lower():
        problems.append("claims the order was sent; it is only drafted and needs confirmation")
    return problems


# ---------------------------------------------------------------- company names <-> tickers
_SPELLED = re.compile(r"^(?:[A-Za-z] )+[A-Za-z]$")            # "O R", "N V D A": a spoken ticker, not a name
# Aliases that are also everyday words: only count them when capitalised ("Block", not "a block of shares").
_COMMON = {"block", "square", "chase", "riot", "visa", "coke", "spy", "queues", "the nasdaq"}
_MARKET_WORDS = re.compile(r"\b(market|markets|index|indexes|indices|S&P|Nasdaq|Dow|broad|overall)\b", re.I)
_NAME_CACHE: dict[int, tuple[re.Pattern, dict[str, str]]] = {}


def _alias_key(s: str) -> str:
    return re.sub(r"[^a-z]+", " ", s.lower()).strip()


def _name_index(world) -> tuple[re.Pattern, dict[str, str]]:
    """One regex over every company name / spoken alias in the universe (longest first) + key -> symbol.
    Spelled tickers ("O R") are excluded: those are tickers (tickers_in)."""
    key = id(world.universe)
    if key not in _NAME_CACHE:
        amap: dict[str, str] = {}
        for sym, t in world.universe.items():
            for a in (t.name, *t.spoken):
                k = _alias_key(a)
                if len(k) < 2 or _SPELLED.match(a.strip()):
                    continue
                amap.setdefault(k, sym)
        parts = sorted(amap, key=len, reverse=True)
        pat = re.compile(r"(?<![A-Za-z])(?:" + "|".join(r"[^A-Za-z]{0,2}".join(map(re.escape, k.split()))
                                                        for k in parts) + r")(?![A-Za-z])", re.I)
        _NAME_CACHE[key] = (pat, amap)
    return _NAME_CACHE[key]


def named_companies(text: str, world, capitalised: bool = False) -> list[tuple[str, int, int]]:
    """[(symbol, start, end)] for company names / spoken aliases in `text` (not tickers)."""
    pat, amap = _name_index(world)
    out = []
    for m in pat.finditer(text or ""):
        k = _alias_key(m.group(0))
        sym = amap.get(k)
        if sym is None:
            continue
        if (capitalised or k in _COMMON) and not m.group(0)[:1].isupper():
            continue
        out.append((sym, m.start(), m.end()))
    return out


def tickers_in(text: str, world) -> list[tuple[str, int, int]]:
    """[(symbol, start, end)] for tickers written in caps (>=2 letters) or spelled out ("O R", "O. R.")."""
    out = []
    for m in re.finditer(r"(?<![A-Za-z])((?:[A-Za-z]\.? ){1,4}[A-Za-z])\.?(?![A-Za-z])", text or ""):
        sym = re.sub(r"[^A-Za-z]", "", m.group(1)).upper()
        if sym in world.universe:
            out.append((sym, m.start(), m.end()))
    for m in re.finditer(r"(?<![A-Za-z$])\$?([A-Z]{2,5})(?![A-Za-z])", text or ""):
        if m.group(1) in world.universe:
            out.append((m.group(1), m.start(), m.end()))
    for m in re.finditer(r"\b(?:ticker|symbol)\s+\$?([A-Za-z]{1,5})\b", text or "", re.I):
        if m.group(1).upper() in world.universe:
            out.append((m.group(1).upper(), m.start(), m.end()))
    return out


def _lookalike(a: str, b: str, world) -> bool:
    if b in getattr(world, "lookalikes", {}).get(a, ()) or a in getattr(world, "lookalikes", {}).get(b, ()):
        return True
    return a.startswith(b) or b.startswith(a) or (len(a) >= 3 and len(b) >= 3 and a[:2] == b[:2])


def correct_symbol(world, symbol: str, question: str, last_symbol: str | None = None,
                   used: set[str] | frozenset = frozenset()) -> str | None:
    """The ticker the user's words name when the model's `symbol` is a different one, else None.

    Conservative: only when the model's symbol is NOT itself spoken in the question (as a ticker or name), exactly
    one named company qualifies, that company isn't already covered by another call this turn, and the model's
    symbol is a lookalike of it (OR/ORCL, META/MET, NVDA/NVDL, ORLY/ORCL ...) or is just the previous turn's
    symbol carried over from the state line ("I mean, Oracle." after an OR turn)."""
    if not symbol or symbol == "ALL" or not question:
        return None
    named = {s for s, _, _ in named_companies(question, world)}
    spoken = named | {s for s, _, _ in tickers_in(question, world)}
    if not named or symbol in spoken:
        return None
    cands = {x for x in named - set(used) - {symbol}
             if _lookalike(symbol, x, world) or (last_symbol and symbol == last_symbol)}
    return cands.pop() if len(cands) == 1 else None


def rename_companies(text: str, world, symbols: set[str], target: str | None) -> str:
    """Make the answer's company names agree with the data: a company (name or caps ticker) named in the
    answer but not among this turn's `symbols` is replaced by `target`'s display name. Only used when the
    turn's tool calls were about exactly one symbol (`target`)."""
    if not text or not target or target not in world.universe:
        return text
    hits = [h for h in named_companies(text, world, capitalised=True) + tickers_in(text, world)
            if h[0] not in symbols and h[0] != target]
    if not hits:
        return text
    name = world.universe[target].name
    out, pos = [], 0
    for _, a, b in sorted(hits, key=lambda h: h[1]):
        if a < pos:
            continue
        out += [text[pos:a], name]
        pos = b
    return "".join(out) + text[pos:]


# ---------------------------------------------------------------- slot binding (which entity / row a slot is about)
_SENT_SPAN = re.compile(r"[^.!?;]+[.!?;]*")


def binding_problems(text: str, book, world) -> list[str]:
    """Conservative binding checks on an answer template (`book`: agent.slots.SlotBook with .meta).

    1. Option-chain rows: a row slot (bid/ask/iv/delta/oi _N) needs its strike_N earlier in the answer, and the
       last strike said before it must be that same row ("$26.30 bid" must follow "the $410 strike").
       oi_N only needs the latter when a strike was said.
    2. Entities: in a sentence that names a company (or the market), every slot bound to an entity must belong to
       one of the entities named in that sentence ("AppLovin ... up {index_move}" is wrong)."""
    meta = getattr(book, "meta", None)
    if not meta or not text:
        return []
    problems: list[str] = []
    used = [(m.group(1), m.start()) for m in SLOT_RE.finditer(text) if m.group(1) in meta]
    last_strike: dict = {}          # run -> row of the last strike slot seen
    said_strike: dict = {}          # run -> rows whose strike was said so far
    bad_rows = []
    for name, _ in used:
        mt = meta[name]
        if not mt.get("row"):
            continue
        field_, row = mt["row"]
        run = mt["run"]
        if field_ == "strike":
            last_strike[run] = row
            said_strike.setdefault(run, set()).add(row)
            continue
        if field_ == "oi":
            if run in last_strike and last_strike[run] != row:
                bad_rows.append(name)
            continue
        if row not in said_strike.get(run, set()) or last_strike.get(run) != row:
            bad_rows.append(name)
    if bad_rows:
        pairs = []
        for n in bad_rows[:3]:
            st = book.name_of(meta[n]["run"], "strike_%d" % meta[n]["row"][1]) if hasattr(book, "name_of") else None
            pairs.append("{" + (st or "strike") + "} right before {" + n + "}")
        problems.append("say the strike first: " + ", ".join(pairs))
    wrong = []
    for sm in _SENT_SPAN.finditer(text):
        sent = sm.group(0)
        slots_here = [n for n in SLOT_RE.findall(sent) if n in meta and meta[n].get("entity")]
        if not slots_here:
            continue
        body = SLOT_RE.sub(" ", sent)
        ents = {s for s, _, _ in named_companies(body, world, capitalised=True)} | \
               {s for s, _, _ in tickers_in(body, world)}
        if not ents:
            continue                                   # no company named: pronouns etc. are not checked
        if _MARKET_WORDS.search(body) or ents & {"SPY", "QQQ"}:
            ents.add("@market")
        wrong += [n for n in slots_here if meta[n]["entity"] not in ents]
    if wrong:
        problems.append("slot attached to the wrong name: " + ", ".join("{" + n + "}" for n in wrong[:3])
                        + " is not about the company named in that sentence")
    return problems
