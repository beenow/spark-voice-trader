"""Stand-in for `agent.loop.run_turn` until the real agent lands.

Same signature and event stream as SPEC.md: tool_call, tool_result, filler
(once, on the first tool call), answer, and ticket when an order is drafted.
Keyword-routed and deterministic; it exists so the server and page can be
exercised end to end.
"""
from __future__ import annotations

import itertools
import re
import time
from typing import Any, Callable

NAMES = {
    "NVDA": ["nvidia", "nvda", "en vidia"], "AAPL": ["apple", "aapl"], "TSLA": ["tesla", "tsla"],
    "MSFT": ["microsoft", "msft"], "AMZN": ["amazon", "amzn"], "META": ["meta", "facebook"],
    "AMD": ["amd"], "PLTR": ["palantir", "pltr"], "SPY": ["spy", "s&p", "market"],
}
SPOKEN = {"NVDA": "Nvidia", "AAPL": "Apple", "TSLA": "Tesla", "MSFT": "Microsoft", "AMZN": "Amazon",
          "META": "Meta", "AMD": "A M D", "PLTR": "Palantir", "SPY": "the S and P"}
_ticket_ids = itertools.count(1)


def _symbol(q: str) -> str | None:
    ql = q.lower()
    for sym, names in NAMES.items():
        if any(re.search(rf"\b{re.escape(n)}\b", ql) for n in names):
            return sym
    return None


def run_turn(question: str, history: list | None = None, world: Any = None, backend: Any = None,
             max_steps: int = 4, on_event: Callable[[dict], None] | None = None, delay: float = 0.4,
             **_: Any) -> dict:
    emit = on_event or (lambda e: None)
    q = question.lower()
    sym = _symbol(question)
    if sym is None and history:
        sym = (history[-1] or {}).get("symbol") if isinstance(history[-1], dict) else None
    sym = sym or "NVDA"
    name = SPOKEN[sym]

    if re.search(r"\b(buy|sell)\b", q):
        direction = "sell" if "sell" in q else "buy"
        m = re.search(r"\b(\d+)\b", q)
        qty = int(m.group(1)) if m else 1
        side = "call" if "call" in q else "put" if "put" in q else None
        args = {"symbol": sym, "direction": direction, "qty": qty, "type": "market"}
        if side:
            args.update(side=side, expiry_rule="next_week")
        emit({"type": "tool_call", "tool": "draft_order", "args": args, "step": 1})
        emit({"type": "filler", "text": f"Drafting that {name} order."})
        time.sleep(delay)
        tid = f"T{next(_ticket_ids):04d}"
        emit({"type": "tool_result", "tool": "draft_order", "words": {"status": "needs confirmation"},
              "slots_display": {"ticket_id": tid}, "step": 1})
        what = f"{qty} {sym} {side + 's' if side else 'shares'}"
        emit({"type": "answer", "template": "I drafted the order, ticket {ticket_id}. Tap confirm to send it.",
              "display": f"Drafted: {direction.upper()} {what} (market). Ticket {tid} needs confirmation.",
              "spoken": f"I drafted an order to {direction} {qty} {name} {side + 's' if side else 'shares'}. Tap confirm to send it."})
        emit({**args, "order_type": args["type"], "type": "ticket", "ticket_id": tid, "status": "needs confirmation"})
        return {"question": question, "symbol": sym, "tools": ["draft_order"], "ticket_id": tid}

    if any(w in q for w in ("position", "holding", "how's my", "how is my", "p&l")):
        emit({"type": "tool_call", "tool": "get_position", "args": {"symbol": sym}, "step": 1})
        emit({"type": "filler", "text": f"Checking your {name} position."})
        time.sleep(delay)
        emit({"type": "tool_result", "tool": "get_position",
              "words": {"direction": "up", "size": "large", "held": "yes"},
              "slots_display": {"qty": "120", "avg_cost": "$118.40", "pnl": "+$1,862.00", "pnl_pct": "+13.1%", "weight": "18.2%"},
              "step": 1})
        emit({"type": "tool_call", "tool": "get_quote", "args": {"symbol": sym}, "step": 2})
        time.sleep(delay / 2)
        emit({"type": "tool_result", "tool": "get_quote", "words": {"direction": "up", "move": "notable"},
              "slots_display": {"price": "$133.92", "change_pct": "+2.4%", "volume": "212.4M"}, "step": 2})
        emit({"type": "answer",
              "template": "You're up {pnl} on {qty} shares, {pnl_pct}. It's {weight} of the portfolio and up {change_pct} today.",
              "display": "You're up +$1,862.00 on 120 shares, +13.1%. It's 18.2% of the portfolio and up +2.4% today.",
              "spoken": f"You're up about eighteen hundred sixty dollars on your {name} shares, thirteen percent. "
                        "It's a large position, and it's up two point four percent today."})
        return {"question": question, "symbol": sym, "tools": ["get_position", "get_quote"]}

    emit({"type": "tool_call", "tool": "get_quote", "args": {"symbol": sym}, "step": 1})
    emit({"type": "filler", "text": f"Looking up {name}."})
    time.sleep(delay)
    emit({"type": "tool_result", "tool": "get_quote", "words": {"direction": "up", "move": "notable"},
          "slots_display": {"price": "$133.92", "change_pct": "+2.4%", "volume": "212.4M"}, "step": 1})
    emit({"type": "answer", "template": "{symbol} is at {price}, up {change_pct} today.",
          "display": f"{sym} is at $133.92, up +2.4% today.",
          "spoken": f"{name} is at about one thirty-four, up two point four percent today."})
    return {"question": question, "symbol": sym, "tools": ["get_quote"]}
