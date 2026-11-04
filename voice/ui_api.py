"""Screen data for the phone's trading UI (android/ WebView): home, symbol page, option chain, ticket, activity.

Read from the SAME per-session mockmarket.World the agent's tools use, through the tool functions themselves
where one exists (get_quote, get_indicators, get_levels, get_news, get_fundamentals, draft_order), so a number
on screen is the number the agent spoke. Plain dicts; the step server wraps them in routes.

`nav_for(tool_calls)` maps a finished turn's tool calls to the screen the app should show.
"""
from __future__ import annotations

import datetime as dt

from mockmarket.world import Order, World
from tools.impl import (draft_order, get_fundamentals, get_indicators, get_levels, get_news, get_option_chain,
                        get_quote)

EXPIRY_RULES = ("this_week", "next_week", "monthly", "nearest")


def _r(v: float, n: int = 2) -> float:
    return round(float(v), n)


def _name(world: World, sym: str) -> str:
    return world.universe[sym].name


def _row(world: World, sym: str) -> dict:
    q = world.quote(sym)
    return {"symbol": sym, "name": _name(world, sym), "price": q.price, "change_pct": _r(q.change_pct * 100, 2),
            "spark": [_r(v) for v in list(q.intraday["close"])]}


def home(world: World, watchlist: list[str] | None = None) -> dict:
    pos = []
    for p in world.positions:
        mv, pnl, day = world.market_value(p), world.pnl(p), world.day_pnl_of(p)
        cost = p.avg_cost * p.qty * p.mult
        label = p.symbol if p.kind == "stock" else (
            f"{p.symbol} {p.strike:g} {p.side} {p.expiry.strftime('%b %-d')}")
        pos.append({"symbol": p.symbol, "label": label, "kind": p.kind, "qty": p.qty, "avg_cost": p.avg_cost,
                    "price": _r(world.mark(p)), "value": _r(mv), "pnl": _r(pnl),
                    "pnl_pct": _r(pnl / cost * 100 if cost else 0.0), "day_pnl": _r(day),
                    "side": p.side, "strike": p.strike, "expiry": p.expiry.isoformat() if p.expiry else None})
    pos.sort(key=lambda r: -r["value"])
    total = world.total_equity()
    day_pnl = sum(world.day_pnl_of(p) for p in world.positions)
    idx = world.index_symbol
    return {"as_of": f"{world.today.isoformat()} {world.session_time()}", "total": _r(total),
            "day_pnl": _r(day_pnl), "day_pct": _r(day_pnl / max(total - day_pnl, 1.0) * 100),
            "cash": _r(world.cash), "positions": pos,
            "watchlist": [_row(world, s) for s in (watchlist or world.watchlist) if world.has(s)],
            "index": _row(world, idx) if world.has(idx) else None,
            "orders": activity(world)["orders"][:4]}


def symbol_page(world: World, sym: str) -> dict:
    if not world.has(sym):
        raise KeyError(sym)
    q = world.quote(sym)
    qt = get_quote(world, sym)
    page = {"symbol": sym, "name": _name(world, sym), "price": qt.slots["price"],
            "change_pct": _r(qt.slots["change_pct"] * 100), "move": qt.words.get("move"),
            "open": q.open, "high": q.high, "low": q.low, "prev_close": q.prev_close, "volume": q.volume,
            "chart": {"1D": {"t": list(q.intraday["time"]), "c": [_r(v) for v in q.intraday["close"]]},
                      "3M": {"c": [_r(v) for v in list(q.daily["close"][-63:]) + [q.price]]},
                      "6M": {"c": [_r(v) for v in list(q.daily["close"][-126:]) + [q.price]]}},
            "quote_only": world.universe[sym].quote_only, "options": bool(world.universe[sym].options),
            "positions": [p for p in home(world)["positions"] if p["symbol"] == sym]}
    if world.universe[sym].quote_only:
        return page
    ind = get_indicators(world, sym, "1D")
    lv = get_levels(world, sym)
    nw = get_news(world, sym)
    fd = get_fundamentals(world, sym)
    page["indicators"] = {"rsi": _r(ind.slots["rsi"], 1), "rsi_state": ind.words["rsi_state"],
                          "ema9": _r(ind.slots["ema9"]), "ema21": _r(ind.slots["ema21"]),
                          "ema50": _r(ind.slots["ema50"]), "trend": ind.words["trend"],
                          "vol_ratio": _r(ind.slots["vol_ratio"]), "volume": ind.words["volume"]}
    page["levels"] = {"support": lv.slots["support"], "resistance": lv.slots["resistance"],
                      "dist_support": _r(lv.slots["dist_support"] * 100), "near_support": lv.words["near_support"],
                      "dist_resistance": _r(lv.slots["dist_resistance"] * 100),
                      "near_resistance": lv.words["near_resistance"]}
    page["news"] = {"tone": nw.words["tone"],
                    "items": [{"headline": h, "age_hours": _r(nw.slots[f"age_{k}"], 1)}
                              for k, h in enumerate(nw.words["headlines"], 1)]}
    page["fundamentals"] = {"market_cap": fd.slots["market_cap"], "pe": _r(fd.slots["pe"], 1),
                            "target": fd.slots["target"], "dist_target": _r(fd.slots["dist_target"] * 100),
                            "earnings_date": fd.slots["earnings_date"], "valuation": fd.words["valuation"]}
    if page["options"]:
        page["expiries"] = [{"rule": r, "date": d.isoformat()} for r in EXPIRY_RULES
                            if (d := world.expiry_for(sym, r)) is not None]
    return page


def chain(world: World, sym: str, expiry_rule: str = "nearest", side: str = "call",
          strike: float | None = None) -> dict:
    res = get_option_chain(world, sym, expiry_rule, side, strike)
    if not res.ok:
        return {"symbol": sym, "side": side, "expiry_rule": expiry_rule, "error": res.error, "rows": []}
    exp = dt.date.fromisoformat(res.slots["expiry"])
    rows = world.chain(sym, exp, side, strike)
    return {"symbol": sym, "name": _name(world, sym), "side": side, "expiry_rule": expiry_rule,
            "expiry": exp.isoformat(), "price": world.quote(sym).price, "iv_level": res.words["iv_level"],
            "liquidity": res.words["liquidity"], "most_open_interest": res.words["most_open_interest"],
            "rows": [{"row": i, "strike": r["strike"], "bid": r["bid"], "ask": r["ask"], "mid": r["mid"],
                      "iv": _r(r["iv"] * 100, 1), "delta": _r(r["delta"], 2), "oi": r["oi"]}
                     for i, r in enumerate(rows, 1)]}


def ticket(world: World, tid: str) -> dict:
    t = world.tickets.get(tid)
    if t is None:
        raise KeyError(tid)
    out = dict(t)
    q = world.quote(t["symbol"])
    out.update(name=_name(world, t["symbol"]), price=q.price, change_pct=_r(q.change_pct * 100),
               buying_power=_r(world.cash))
    for k in ("est_price", "est_cost", "limit_price"):
        if k in out:
            out[k] = _r(out[k])
    return out


def new_ticket(world: World, symbol: str, direction: str, qty: int, type: str = "market", side: str | None = None,
               strike: float | None = None, expiry_rule: str | None = None) -> dict:
    """The Trade button: the same draft_order the agent calls, so a manual ticket looks like a spoken one."""
    res = draft_order(world, symbol, direction, qty, type, side, strike, expiry_rule)
    if not res.ok:
        raise ValueError(res.error)
    return ticket(world, res.slots["ticket_id"])


def confirm(world: World, tid: str) -> dict:
    """Mock submit: never leaves this process. The ticket becomes an open order (a market order fills)."""
    t = world.tickets.get(tid)
    if t is None:
        raise KeyError(tid)
    if t["status"] == "needs confirmation":
        n = sum(1 for o in world.orders if o.order_id.startswith("V")) + 1
        status = "filled" if t.get("order_type") == "market" else "open"
        exp = dt.date.fromisoformat(t["expiry"]) if t.get("expiry") else None
        world.orders.insert(0, Order(f"V{n}", t["symbol"], t["direction"], int(t["qty"]),
                                     round(float(t.get("limit_price") or t["est_price"]), 2), status, "today",
                                     t["kind"], t.get("side"), t.get("strike"), exp))
        t.update(status="submitted", order_id=f"V{n}", order_status=status,
                 submitted_at=world.session_time())
    return ticket(world, tid)


def cancel(world: World, tid: str) -> dict:
    t = world.tickets.get(tid)
    if t is None:
        raise KeyError(tid)
    if t["status"] == "needs confirmation":
        t["status"] = "cancelled"
    return ticket(world, tid)


def activity(world: World) -> dict:
    orders = []
    for o in world.orders:
        label = f"{o.symbol} shares" if o.kind == "stock" else (
            f"{o.symbol} {o.strike:g} {o.side} {o.expiry.strftime('%b %-d') if o.expiry else ''}".strip())
        orders.append({"order_id": o.order_id, "symbol": o.symbol, "direction": o.direction, "qty": o.qty,
                       "price": o.price, "status": o.status, "day": o.day, "kind": o.kind, "label": label,
                       "voice": o.order_id.startswith("V")})
    tickets = [ticket(world, tid) for tid in reversed(list(world.tickets))]
    return {"orders": orders, "tickets": tickets}


# ------------------------------------------------------------------------------------------- agent -> screen
_SYMBOL_SECTION = {"get_quote": "chart", "get_indicators": "indicators", "get_levels": "levels",
                   "get_news": "news", "get_fundamentals": "fundamentals", "get_position": "position"}
_HOME_SECTION = {"get_portfolio": "positions", "what_if": "positions", "screen_watchlist": "watchlist",
                 "get_market": "market"}


def chart_focus(calls: list[dict], question: str = "") -> tuple[list[str], str | None]:
    """What a technical question is about, for the chart to highlight: the overlays (rsi / ema / support /
    resistance) and the timeframe of the indicator call. Support or resistance alone when only one was asked."""
    focus, tf = [], None
    q = (question or "").lower()
    for c in calls:
        a = c.get("args") or {}
        if c["tool"] == "get_indicators":
            tf = tf or a.get("timeframe") or "1D"
            focus += {"rsi": ["rsi"], "ema": ["ema"], "volume": []}.get(a.get("indicator"), ["rsi", "ema"])
        elif c["tool"] == "get_levels":
            s, r = "support" in q, "resist" in q
            focus += ["support"] if s and not r else ["resistance"] if r and not s else ["support", "resistance"]
    return list(dict.fromkeys(focus)), tf


def nav_for(tool_calls: list[dict] | None, ticket_id: str | None = None, matches: list[str] | None = None,
            question: str = "") -> dict | None:
    """The screen for a turn: the most specific call wins (a ticket over a chain over a symbol page over home).
    `ticket_id`: the ticket draft_order created this turn; `matches`: screen_watchlist's symbols."""
    calls = [c for c in (tool_calls or []) if isinstance(c, dict) and c.get("tool")]
    if not calls:
        return None
    for c in calls:
        if c["tool"] == "draft_order" and ticket_id:
            return {"screen": "ticket", "ticket_id": ticket_id, "symbol": (c.get("args") or {}).get("symbol")}
    for c in calls:
        if c["tool"] == "get_option_chain":
            a = c.get("args") or {}
            return {"screen": "symbol", "symbol": a.get("symbol"), "tab": "options",
                    "expiry_rule": a.get("expiry_rule") or "nearest", "side": a.get("side") or "call",
                    "strike": a.get("strike")}
    syms = [(c.get("args") or {}).get("symbol") for c in calls if c["tool"] in _SYMBOL_SECTION]
    syms = [s for s in syms if s and s != "ALL"]
    if len(calls) <= 3 and syms and len(set(syms)) == 1:
        sections = [_SYMBOL_SECTION[c["tool"]] for c in calls if c["tool"] in _SYMBOL_SECTION]
        out = {"screen": "symbol", "symbol": syms[0], "tab": "overview", "sections": sections}
        focus, tf = chart_focus(calls, question)
        if focus or tf:
            out.update(focus=focus, timeframe=tf or "1D")       # the page shows the chart, not a card
        return out
    if any(c["tool"] == "get_orders" for c in calls):
        return {"screen": "activity"}
    if any(c["tool"] == "screen_watchlist" for c in calls):
        return {"screen": "home", "section": "watchlist", "highlight": list(matches or [])}
    if len(calls) > 3 and all(c["tool"] == "get_quote" for c in calls):        # the watchlist code route
        return {"screen": "home", "section": "watchlist", "highlight": []}
    if len(set(syms)) > 1:                                                      # a comparison
        return {"screen": "home", "section": "watchlist", "highlight": list(dict.fromkeys(syms))}
    sec = next((_HOME_SECTION[c["tool"]] for c in calls if c["tool"] in _HOME_SECTION), "positions")
    return {"screen": "home", "section": sec, "highlight": []}
