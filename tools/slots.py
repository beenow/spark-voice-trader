"""Per-tool slot-type map. Kinds are rendered by tools.render."""
from __future__ import annotations

import re

SLOT_TYPES: dict[str, dict[str, str]] = {
    "get_quote": {"price": "price", "change_pct": "signed_pct", "volume": "count"},
    "get_position": {"qty": "count", "avg_cost": "price", "pnl": "signed_money", "pnl_pct": "signed_pct",
                     "weight": "pct", "option_qty": "count", "expiry": "date"},
    "get_portfolio": {"total": "money", "day_pnl": "signed_money", "cash": "money", "expiry": "date"},
    "get_indicators": {"rsi": "index", "ema9": "price", "ema21": "price", "ema50": "price", "vol_ratio": "ratio"},
    "get_levels": {"support": "price", "resistance": "price", "dist_support": "pct", "dist_resistance": "pct"},
    "get_option_chain": {"expiry": "date", **{f"{k}_{i}": kind for i in range(1, 6)
                                              for k, kind in (("strike", "price"), ("bid", "price"),
                                                              ("ask", "price"), ("iv", "vol"),
                                                              ("delta", "greek"), ("oi", "count"))}},
    "get_orders": {},          # order_N_qty / order_N_price via patterns
    "get_news": {"age_1": "age", "age_2": "age"},
    "get_fundamentals": {"market_cap": "money", "pe": "index", "target": "price", "dist_target": "signed_pct",
                         "earnings_date": "date"},
    "get_market": {"top_move": "signed_pct", "bottom_move": "signed_pct", "index_move": "signed_pct",
                   "event_time": "time"},
    "screen_watchlist": {},
    "what_if": {"pnl_change": "signed_money", "new_total": "money"},
    "resolve_symbol": {},
    "draft_order": {"ticket_id": "text"},
}

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^order_\d+_qty$"), "count"),
    (re.compile(r"^order_\d+_price$"), "price"),
]

_GLOBAL: dict[str, str] = {}
for _tool, _m in SLOT_TYPES.items():
    for _s, _k in _m.items():
        _GLOBAL.setdefault(_s, _k)

_SUFFIX = re.compile(r"^(.*)_(\d+)$")


def _lookup(slot: str, tool: str | None) -> str | None:
    if tool and slot in SLOT_TYPES.get(tool, {}):
        return SLOT_TYPES[tool][slot]
    if slot in _GLOBAL:
        return _GLOBAL[slot]
    for pat, kind in _PATTERNS:
        if pat.match(slot):
            return kind
    return None


def kind_of(slot: str, tool: str | None = None) -> str:
    """Kind for a slot name; accepts the per-turn `_2` suffix and bare kind names."""
    from .render import KINDS
    if slot in KINDS:
        return slot
    k = _lookup(slot, tool)
    s = slot
    while k is None:
        m = _SUFFIX.match(s)
        if not m:
            break
        s = m.group(1)
        k = _lookup(s, tool)
    return k or "text"


# ---------------------------------------------------------------- slot -> entity binding (answer check)
# Which entity a slot's value belongs to, for tools whose slots are not all about the call's `symbol`.
# Values: "word:<key>" = the symbol in the result's words[key]; "@market" = the market / index, not a ticker;
# None = not tied to a named entity (skip). Tools not listed: every slot belongs to args["symbol"] when the
# tool takes a symbol, else None.
SLOT_ENTITY: dict[str, dict[str, str | None]] = {
    "get_market": {"top_move": "word:top_name", "bottom_move": "word:bottom_name", "index_move": "@market",
                   "event_time": None},
    "get_portfolio": {},
    "get_orders": {},
    "screen_watchlist": {},
    "what_if": {},
    "resolve_symbol": {},
    "draft_order": {"ticket_id": None},
}

# get_option_chain rows: strike_N, bid_N, ask_N, iv_N, delta_N, oi_N all describe row N.
ROW_SLOT = re.compile(r"^(strike|bid|ask|iv|delta|oi)_([1-5])$")


def entity_of(tool: str, raw_slot: str, words: dict | None, args: dict | None) -> str | None:
    """The entity (ticker, "@market") a raw slot of `tool` belongs to, or None when unbound."""
    spec = SLOT_ENTITY.get(tool)
    if spec is not None:
        e = spec.get(raw_slot)
        if e is None:
            return None
        if e.startswith("word:"):
            v = (words or {}).get(e[5:])
            return v if isinstance(v, str) and v else None
        return e
    sym = (args or {}).get("symbol")
    return sym if isinstance(sym, str) and sym and sym != "ALL" else None


def row_of(raw_slot: str) -> tuple[str, int] | None:
    """('bid', 3) for 'bid_3' (get_option_chain row slots), else None."""
    m = ROW_SLOT.match(raw_slot)
    return (m.group(1), int(m.group(2))) if m else None
