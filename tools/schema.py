"""JSON-schema tool definitions given to the model (OpenAI/HF `tools=` format)."""
from __future__ import annotations

SYM = {"type": "string", "description": "Ticker symbol, e.g. NVDA"}

_DEFS: list[tuple[str, str, dict, list[str]]] = [
    ("get_quote", "Current price, day change and volume for one symbol.", {"symbol": SYM}, ["symbol"]),
    ("get_position", "The user's position in one symbol: quantity, average cost, P&L, portfolio weight.",
     {"symbol": SYM}, ["symbol"]),
    ("get_portfolio", "Whole-portfolio summary: total value, today's P&L, cash, best/worst and ranked positions.",
     {"rank_by": {"type": "string", "enum": ["pnl", "pnl_pct", "weight", "day_move"], "default": "pnl"}}, []),
    ("get_indicators", "Technical indicators for a symbol: RSI, EMA 9/21/50, relative volume.",
     {"symbol": SYM, "timeframe": {"type": "string", "enum": ["1D", "1H", "5m"], "default": "1D"},
      "indicator": {"type": "string", "enum": ["rsi", "ema", "volume"],
                    "description": "Only this indicator; omit for all."}}, ["symbol"]),
    ("get_levels", "Nearest support and resistance for a symbol and whether price is near them (within two percent).",
     {"symbol": SYM}, ["symbol"]),
    ("get_option_chain", "Five strikes around the money (or around a strike) with bid/ask/IV/delta for one expiry.",
     {"symbol": SYM,
      "expiry_rule": {"type": "string", "enum": ["this_week", "next_week", "monthly", "nearest"],
                      "default": "nearest"},
      "side": {"type": "string", "enum": ["call", "put"], "default": "call"},
      "strike": {"type": "number", "description": "Center the chain on this strike (from the user's words)."}},
     ["symbol"]),
    ("get_orders", "The user's orders.",
     {"status": {"type": "string", "enum": ["open", "filled", "all"], "default": "open"},
      "day": {"type": "string", "enum": ["today", "yesterday"]}}, []),
    ("get_news", "Latest headlines for a symbol and their overall tone.", {"symbol": SYM}, ["symbol"]),
    ("get_fundamentals", "Market cap, P/E, analyst target, next earnings date for a symbol.", {"symbol": SYM},
     ["symbol"]),
    ("get_market", "Market overview: top/bottom movers, breadth, or the economic calendar.",
     {"focus": {"type": "string", "enum": ["movers", "breadth", "calendar"], "default": "movers"}}, []),
    ("screen_watchlist", "Scan the user's watchlist for a condition.",
     {"condition": {"type": "string", "enum": ["oversold", "overbought", "overextended"]}}, ["condition"]),
    ("what_if", "Estimate the portfolio impact if a symbol moves by a percent. Use symbol ALL for the whole portfolio.",
     {"symbol": SYM, "move_pct": {"type": "number", "description": "Percent move, e.g. -10 for a ten percent drop."}},
     ["symbol", "move_pct"]),
    ("resolve_symbol", "Map a spoken company name or unclear ticker to a symbol.",
     {"text": {"type": "string", "description": "What the user said."}}, ["text"]),
    ("draft_order", "Draft an order ticket for the user to confirm. It never sends the order.",
     {"symbol": SYM, "direction": {"type": "string", "enum": ["buy", "sell"]},
      "qty": {"type": "integer", "description": "Shares or contracts."},
      "type": {"type": "string", "enum": ["market", "limit"], "default": "market"},
      "side": {"type": "string", "enum": ["call", "put"], "description": "Only for options."},
      "strike": {"type": "number"},
      "expiry_rule": {"type": "string", "enum": ["this_week", "next_week", "monthly", "nearest"]}},
     ["symbol", "direction", "qty"]),
]

TOOL_SCHEMAS: dict[str, dict] = {
    name: {"name": name, "description": desc,
           "parameters": {"type": "object", "properties": props, "required": req}}
    for name, desc, props, req in _DEFS
}


def schema() -> list[dict]:
    """Tool list in the `tools=` format chat templates expect."""
    return [{"type": "function", "function": s} for s in TOOL_SCHEMAS.values()]
