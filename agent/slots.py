"""Per-turn slot book (unique names, `_2` suffix), answer filling, and the code-made filler line."""
from __future__ import annotations

import re
from collections import Counter

from tools import render
from tools import slots as tool_slots

from .validator import SLOT_RE


class SlotBook:
    """Holds every slot produced in one turn under a unique name."""

    def __init__(self):
        self.slots: dict[str, tuple[str, str, object]] = {}    # unique -> (tool, raw slot, value)
        self.meta: dict[str, dict] = {}                         # unique -> {"entity", "run", "row"}
        self.run_words: dict[tuple[str, int], dict] = {}        # (tool, run) -> that result's words
        self._tool_runs: Counter = Counter()

    def add(self, tool: str, slots: dict, words: dict | None = None, args: dict | None = None) -> dict[str, str]:
        """Register a result's slots; returns raw -> unique name map. `words`/`args` (optional) bind each slot to
        its entity (tools.slots.entity_of) for the answer's binding check."""
        self._tool_runs[tool] += 1
        n = self._tool_runs[tool]
        self.run_words[(tool, n)] = dict(words or {})
        names = {}
        for raw, val in slots.items():
            name = raw if n == 1 else f"{raw}_{n}"
            k = n
            while name in self.slots:                  # cross-tool collision
                k += 1
                name = f"{raw}_{k}"
            self.slots[name] = (tool, raw, val)
            self.meta[name] = {"entity": tool_slots.entity_of(tool, raw, words, args) if (words or args) else None,
                               "run": (tool, n), "row": tool_slots.row_of(raw) if tool == "get_option_chain"
                               else None}
            names[raw] = name
        return names

    def __contains__(self, name: str) -> bool:
        return name in self.slots

    def names(self) -> set[str]:
        return set(self.slots)

    def display(self, name: str) -> str:
        tool, raw, val = self.slots[name]
        return render.display(raw, val, tool)

    def spoken(self, name: str) -> str:
        tool, raw, val = self.slots[name]
        return render.spoken(raw, val, tool)

    def name_of(self, run: tuple[str, int], raw: str) -> str | None:
        """Unique slot name of raw slot `raw` from tool run `run` (None if absent)."""
        return next((k for k, (t, r, _) in self.slots.items() if r == raw and self.meta[k]["run"] == run), None)

    def values(self) -> dict:
        return {k: v[2] for k, v in self.slots.items()}


_ROWSLOT = r"_[1-5](?:_\d+)?"
_BIDASK = re.compile(r"\{(bid" + _ROWSLOT + r")\}\s*(?:by|/|x|×|,|-|–|and|to|over)\s*\{(ask" + _ROWSLOT + r")\}", re.I)
_DELTA = re.compile(r"\{(delta" + _ROWSLOT + r")\}")
_IV = re.compile(r"(\s+at\s+)?\{(iv" + _ROWSLOT + r")\}")


def label_chain(template: str) -> str:
    """Label unlabelled option-chain numbers so they are not read out bare: "{bid_2} by {ask_2}" ->
    "bid {bid_2}, ask {ask_2}"; "{delta_2}" -> "delta {delta_2}"; "at {iv_2}" -> ", IV {iv_2}". Text that
    already labels them ("{bid_1} bid", "delta of {delta_1}", "IV at {iv_1}") is left alone."""
    def before(m, n):
        return m.string[max(0, m.start() - n):m.start()].lower()

    def after(m, n):
        return m.string[m.end():m.end() + n].lower()

    def bidask(m):
        if "bid" in before(m, 20) or after(m, 5).strip().startswith("ask"):
            return m.group(0)
        return "bid {" + m.group(1) + "}, ask {" + m.group(2) + "}"

    def delta(m):
        if "delta" in before(m, 14) or after(m, 7).strip().startswith("delta"):
            return m.group(0)
        return "delta " + m.group(0)

    def iv(m):
        b, a = before(m, 26), after(m, 12).strip()
        if re.search(r"\biv\b|implied|vol", b) or re.match(r"(iv|implied|vol)\b", a):
            return m.group(0)
        return (", IV {" if m.group(1) else "IV {") + m.group(2) + "}"

    if "{" not in template:
        return template
    t = _BIDASK.sub(bidask, template)
    t = _DELTA.sub(delta, t)
    return _IV.sub(iv, t)


def fill(template: str, book: SlotBook, mode: str) -> str:
    """mode 'display' or 'spoken'. Strips a '$' typed before a slot and '%' after it; labels bare option-chain
    numbers (label_chain)."""
    def sub(m):
        return book.display(m.group(1)) if mode == "display" else book.spoken(m.group(1))
    t = re.sub(r"\$\s*(\{[A-Za-z_][A-Za-z0-9_]*\})", r"\1", label_chain(template))
    t = re.sub(r"(\{[A-Za-z_][A-Za-z0-9_]*\})\s*(%|percent\b)", r"\1", t)
    # age slots already end in "ago" ("two hours ago"): drop a template's own "ago" after them
    t = re.sub(r"(\{age_\d+(?:_\d+)?\})\s+ago\b", r"\1", t)
    return SLOT_RE.sub(sub, t)


def speakable(text: str, world) -> str:
    """Swap tickers for company names so TTS doesn't spell them; keep everything digit-free."""
    def sub(m):
        s = m.group(0)
        t = world.universe.get(s)
        return t.name if t and len(s) >= 2 and s not in ("OR",) else s
    text = re.sub(r"\b[A-Z]{2,5}\b", sub, text)
    # "up {change_pct}" renders as "up down two percent": keep the value's own direction word
    return re.sub(r"\b(?:up|down)\s+(up|down)\b", r"\1", text, flags=re.I)


_FILLER = {
    "get_position": "Checking your {name} position.",
    "get_quote": "Pulling up {name}.",
    "get_portfolio": "Looking at your portfolio.",
    "get_indicators": "Checking the {name} chart.",
    "get_levels": "Checking {name} levels.",
    "get_option_chain": "Pulling the {name} options chain.",
    "get_orders": "Checking your orders.",
    "get_news": "Checking {name} news.",
    "get_fundamentals": "Looking at {name} fundamentals.",
    "get_market": "Checking the market.",
    "screen_watchlist": "Scanning your watchlist.",
    "what_if": "Running the numbers.",
    "resolve_symbol": "Let me find that ticker.",
    "draft_order": "Drafting that order.",
}


def filler_line(tool: str, args: dict, world) -> str:
    sym = args.get("symbol")
    t = world.universe.get(sym) if sym else None
    name = t.name if t else ("that" if sym else "")
    if tool == "what_if" and sym == "ALL":
        return "Running the numbers on your portfolio."
    line = _FILLER.get(tool, "One moment.")
    if "{name}" in line and not name:
        return "One moment."
    return line.format(name=name)
