"""The 14 tools over mockmarket. Each returns a ToolResult: judgment words + raw slots (never shown to the model)."""
from __future__ import annotations

import datetime as dt
import difflib
import re

import numpy as np

from mockmarket.world import SESSION_BARS, World

from .result import ToolResult

ETFS = {"SPY", "QQQ", "NVDL", "SMH", "VIX"}   # excluded from movers / breadth


# ----------------------------------------------------------------- helpers
def _unknown(tool: str) -> ToolResult:
    return ToolResult.fail(tool, "unknown symbol")


def _check(world: World, tool: str, symbol: str) -> ToolResult | None:
    """Unknown symbol, or an index-like symbol (VIX) that only has a quote."""
    if not world.has(symbol):
        return _unknown(tool)
    if world.universe[symbol].quote_only:
        return ToolResult.fail(tool, "quote only symbol", words={"symbol": symbol})
    return None


def _expires(world: World, d: dt.date) -> str:
    fri = _fri(world)
    if d <= fri:
        return "this week"
    if d <= fri + dt.timedelta(days=7):
        return "next week"
    if (d.year, d.month) == (world.today.year, world.today.month):
        return "this month"
    return "later"


def _name(world: World, sym: str) -> str:
    return world.universe[sym].name


def ema(x: np.ndarray, n: int) -> float:
    a = 2.0 / (n + 1)
    e = float(x[0])
    for v in x[1:]:
        e = a * float(v) + (1 - a) * e
    return e


def rsi(x: np.ndarray, n: int = 14) -> float:
    d = np.diff(np.asarray(x, dtype=float))
    if len(d) < 2:
        return 50.0
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    n = min(n, len(d))
    au, ad = up[:n].mean(), dn[:n].mean()
    for u, w in zip(up[n:], dn[n:]):
        au = (au * (n - 1) + u) / n
        ad = (ad * (n - 1) + w) / n
    if ad == 0:
        return 100.0 if au > 0 else 50.0
    return float(100 - 100 / (1 + au / ad))


def _series(world: World, sym: str, tf: str) -> np.ndarray:
    q = world.quote(sym)
    if tf == "5m":
        return np.asarray(q.intraday["close"], dtype=float)
    if tf == "1H":
        return np.asarray(q.hourly_close, dtype=float)
    return np.concatenate([q.daily["close"], [q.price]])


def _fri(world: World) -> dt.date:
    return world.today + dt.timedelta(days=(4 - world.today.weekday()) % 7)


# ----------------------------------------------------------------- tools
def get_quote(world: World, symbol: str) -> ToolResult:
    if not world.has(symbol):
        return _unknown("get_quote")
    q = world.quote(symbol)
    ch = q.change_pct
    direction = "flat" if abs(ch) < 0.001 else ("up" if ch > 0 else "down")
    sd = world.universe[symbol].vol / np.sqrt(252)
    z = abs(ch) / sd
    move = "small" if z < 0.7 else ("notable" if z < 1.8 else "big")
    return ToolResult("get_quote", words={"symbol": symbol, "direction": direction, "move": move},
                      slots={"price": q.price, "change_pct": ch, "volume": q.volume})


def get_position(world: World, symbol: str) -> ToolResult:
    if (err := _check(world, "get_position", symbol)) is not None:
        return err
    if not world.positions:
        return ToolResult.fail("get_position", "empty portfolio")
    ps = world.positions_for(symbol)
    if not ps:
        return ToolResult("get_position", words={"symbol": symbol, "held": "no"}, slots={})
    p = ps[0]
    pnl = world.pnl(p)
    cost = p.avg_cost * p.qty * p.mult
    pnl_pct = pnl / cost if cost else 0.0
    weight = world.market_value(p) / world.total_equity()
    size = "small" if weight < 0.05 else ("medium" if weight < 0.15 else "large")
    words = {"symbol": symbol, "held": "yes", "direction": "up" if pnl >= 0 else "down", "size": size,
             "kind": "shares" if p.kind == "stock" else f"{p.side} options"}
    slots = {"qty": p.qty, "avg_cost": p.avg_cost, "pnl": pnl, "pnl_pct": pnl_pct, "weight": weight}
    opts = sorted((o for o in ps if o.kind == "option"), key=lambda o: o.expiry)
    if opts:                                   # nearest option expiry in this symbol (shares never expire)
        o = opts[0]
        words["expires"] = _expires(world, o.expiry)
        words["option_kind"] = f"{o.side} options"
        if p.kind == "stock":
            words["kind"] = f"shares and {o.side} options"
            slots["option_qty"] = o.qty
        slots["expiry"] = o.expiry.isoformat()
    return ToolResult("get_position", words=words, slots=slots)


def get_portfolio(world: World, rank_by: str = "pnl") -> ToolResult:
    if not world.positions:
        return ToolResult.fail("get_portfolio", "empty portfolio")
    total = world.total_equity()

    def key(p):
        if rank_by == "pnl_pct":
            c = p.avg_cost * p.qty * p.mult
            return world.pnl(p) / c if c else 0
        if rank_by == "weight":
            return world.market_value(p)
        if rank_by == "day_move":
            return world.quote(p.symbol).change_pct
        return world.pnl(p)

    ranked = sorted(world.positions, key=key, reverse=True)
    day = sum(world.day_pnl_of(p) for p in world.positions)
    syms = list(dict.fromkeys(p.symbol for p in ranked))
    words = {"best": syms[0], "worst": syms[-1], "day": "up" if day >= 0 else "down",
             "positions": syms, "ranked_by": rank_by}
    slots = {"total": total, "day_pnl": day, "cash": world.cash}
    opts = sorted((p for p in world.positions if p.kind == "option"), key=lambda p: p.expiry)
    if opts:                                   # what expires next, plus every option leg's expiry bucket
        words["expires"] = _expires(world, opts[0].expiry)
        words["next_expiring"] = f"{opts[0].symbol} {opts[0].side}s"
        words["option_expiries"] = [{"position": f"{p.symbol} {p.side}s", "expires": _expires(world, p.expiry)}
                                    for p in opts]
        slots["expiry"] = opts[0].expiry.isoformat()
    else:
        words["expires"] = "no options held"
    return ToolResult("get_portfolio", words=words, slots=slots)


def get_indicators(world: World, symbol: str, timeframe: str = "1D", indicator: str | None = None) -> ToolResult:
    if (err := _check(world, "get_indicators", symbol)) is not None:
        return err
    x = _series(world, symbol, timeframe)
    q = world.quote(symbol)
    r = rsi(x[-120:])
    e9, e21, e50 = ema(x[-150:], 9), ema(x[-150:], 21), ema(x[-150:], 50)
    if timeframe == "5m":
        v = np.asarray(q.intraday["volume"], dtype=float)
        vr = float(v[-1] / max(v.mean(), 1.0))
    else:
        vr = float(q.volume / max(q.avg_volume * world.bars_elapsed / SESSION_BARS, 1.0))
    words, slots = {"symbol": symbol, "timeframe": {"1D": "daily", "1H": "hourly", "5m": "five minute"}[timeframe]}, {}
    if indicator in (None, "rsi"):
        words["rsi_state"] = "oversold" if r < 30 else ("overbought" if r > 70 else "neutral")
        slots["rsi"] = r
    if indicator in (None, "ema"):
        words["trend"] = "above twenty-one EMA" if q.price >= e21 else "below twenty-one EMA"
        slots.update(ema9=e9, ema21=e21, ema50=e50)
    if indicator in (None, "volume"):
        words["volume"] = "light" if vr < 0.7 else ("heavy" if vr > 1.4 else "normal")
        slots["vol_ratio"] = vr
    return ToolResult("get_indicators", words=words, slots=slots)


def _pivots(a: np.ndarray, w: int, low: bool) -> list[float]:
    out = []
    for i in range(w, len(a) - w):
        seg = a[i - w:i + w + 1]
        if (low and a[i] == seg.min()) or (not low and a[i] == seg.max()):
            out.append(float(a[i]))
    return out


def levels(world: World, symbol: str) -> tuple[float, float]:
    q = world.quote(symbol)
    lo, hi, p = q.daily["low"][-90:], q.daily["high"][-90:], q.price
    sup = [v for v in _pivots(lo, 3, True) if v < p * 0.999] + ([q.low] if q.low < p * 0.999 else [])
    res = [v for v in _pivots(hi, 3, False) if v > p * 1.001] + ([q.high] if q.high > p * 1.001 else [])
    support = max(sup) if sup else float(min(lo.min(), p * 0.95))
    if res:
        resistance = min(res)
    else:
        step = world.strike_step(p) * 4
        resistance = float(np.ceil(p * 1.005 / step) * step)
    return round(support, 2), round(resistance, 2)


def get_levels(world: World, symbol: str) -> ToolResult:
    if (err := _check(world, "get_levels", symbol)) is not None:
        return err
    p = world.quote(symbol).price
    s, r = levels(world, symbol)
    ds, dr = (p - s) / p, (r - p) / p
    return ToolResult("get_levels",
                      words={"symbol": symbol, "near_support": "yes" if ds <= 0.02 else "no",
                             "near_resistance": "yes" if dr <= 0.02 else "no"},
                      slots={"support": s, "resistance": r, "dist_support": ds, "dist_resistance": dr})


def get_option_chain(world: World, symbol: str, expiry_rule: str = "nearest", side: str = "call",
                     strike: float | None = None) -> ToolResult:
    if (err := _check(world, "get_option_chain", symbol)) is not None:
        return err
    exp = world.expiry_for(symbol, expiry_rule)
    if exp is None:
        return ToolResult.fail("get_option_chain", "no chain for that expiry",
                               words={"symbol": symbol, "has_weeklies": "yes" if world.universe[symbol].weeklies
                                      else "no"})
    rows = world.chain(symbol, exp, side, strike)
    atm = min(rows, key=lambda r: abs(r["strike"] - world.quote(symbol).price))
    ratio = atm["iv"] / world.universe[symbol].iv
    iv_level = "low" if ratio < 0.9 else ("high" if ratio > 1.2 else "normal")
    dte = "this week" if exp <= _fri(world) else ("soon" if (exp - world.today).days <= 14 else "far")
    slots: dict = {}
    for i, r in enumerate(rows, 1):
        slots.update({f"strike_{i}": r["strike"], f"bid_{i}": r["bid"], f"ask_{i}": r["ask"],
                      f"iv_{i}": r["iv"], f"delta_{i}": r["delta"], f"oi_{i}": r["oi"]})
    slots["expiry"] = exp.isoformat()
    liquidity = "thin" if atm["oi"] < 1500 else ("deep" if atm["oi"] >= 20000 else "normal")
    most = max(range(len(rows)), key=lambda i: rows[i]["oi"]) + 1
    return ToolResult("get_option_chain",
                      words={"symbol": symbol, "side": side, "iv_level": iv_level, "days_to_expiry": dte,
                             "liquidity": liquidity, "most_open_interest": f"row {_ORD[most]}"},
                      slots=slots)


_ORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def get_orders(world: World, status: str = "open", day: str | None = None) -> ToolResult:
    sel = [o for o in world.orders if (status == "all" or o.status == status) and (day is None or o.day == day)]
    count = "none" if not sel else ("one" if len(sel) == 1 else "several")
    labels, slots = [], {}
    for i, o in enumerate(sel, 1):
        labels.append(f"{o.direction.upper()} {o.symbol} {o.side if o.kind == 'option' else 'shares'}"
                      f" ({o.status})")
        slots[f"order_{i}_qty"] = o.qty
        slots[f"order_{i}_price"] = o.price
    return ToolResult("get_orders", words={"count": count, "orders": labels}, slots=slots)


def get_news(world: World, symbol: str) -> ToolResult:
    if (err := _check(world, "get_news", symbol)) is not None:
        return err
    items = world.news[symbol]
    tones = {i["tone"] for i in items}
    tone = tones.pop() if len(tones) == 1 else "mixed"
    return ToolResult("get_news", words={"symbol": symbol, "headlines": [i["headline"] for i in items],
                                         "tone": tone},
                      slots={f"age_{k}": i["age_hours"] for k, i in enumerate(items, 1)})


def get_fundamentals(world: World, symbol: str) -> ToolResult:
    if (err := _check(world, "get_fundamentals", symbol)) is not None:
        return err
    q = world.quote(symbol)
    valuation = "cheap" if q.pe < 18 else ("rich" if q.pe > 35 else "fair")
    earnings = "this week" if q.earnings_date <= _fri(world) else "later"
    return ToolResult("get_fundamentals", words={"symbol": symbol, "valuation": valuation, "earnings": earnings},
                      slots={"market_cap": q.market_cap, "pe": q.pe, "target": round(q.target, 2),
                             "dist_target": q.target / q.price - 1, "earnings_date": q.earnings_date.isoformat()})


def get_market(world: World, focus: str = "movers") -> ToolResult:
    idx = world.quote(world.index_symbol).change_pct
    market_day = "flat" if abs(idx) < 0.001 else ("up" if idx > 0 else "down")
    if focus == "calendar":
        ev = sorted(world.events, key=lambda e: e.date)
        e = ev[0]
        delta = (e.date - world.today).days
        risk = "today" if delta == 0 else ("tomorrow" if delta == 1 else "none")
        when = ("today" if delta == 0 else "tomorrow" if delta == 1 else
                "this week" if e.date <= _fri(world) else "next week or later")
        return ToolResult("get_market", words={"next_event": e.name, "event_day": when, "event_risk": risk,
                                               "market_day": market_day},
                          slots={"event_time": e.time, "index_move": idx})
    stocks = [s for s in world.quotes if s not in ETFS]
    if focus == "breadth":
        adv = sum(world.quote(s).change_pct > 0 for s in stocks) / len(stocks)
        breadth = "strong" if adv > 0.65 else ("weak" if adv < 0.35 else "mixed")
        return ToolResult("get_market", words={"market_day": market_day, "breadth": breadth},
                          slots={"index_move": idx})
    ranked = sorted(stocks, key=lambda s: world.quote(s).change_pct)
    top, bot = ranked[-1], ranked[0]
    return ToolResult("get_market", words={"top_name": top, "bottom_name": bot, "market_day": market_day},
                      slots={"top_move": world.quote(top).change_pct, "bottom_move": world.quote(bot).change_pct,
                             "index_move": idx})


def screen_watchlist(world: World, condition: str = "oversold") -> ToolResult:
    matches = []
    for s in world.watchlist:
        x = _series(world, s, "1D")
        r = rsi(x[-120:])
        e21 = ema(x[-150:], 21)
        ext = world.quote(s).price / e21 - 1
        if condition == "oversold" and r < 35:
            matches.append({"symbol": s, "reason": "RSI deeply oversold" if r < 30 else "RSI near oversold"})
        elif condition == "overbought" and r > 65:
            matches.append({"symbol": s, "reason": "RSI overbought" if r > 70 else "RSI near overbought"})
        elif condition == "overextended" and abs(ext) > 0.06:
            matches.append({"symbol": s, "reason": "far above its twenty-one EMA" if ext > 0 else "far below its twenty-one EMA"})
    count = "none" if not matches else ("one" if len(matches) == 1 else "several")
    return ToolResult("screen_watchlist", words={"condition": condition, "count": count, "matches": matches})


def what_if(world: World, symbol: str, move_pct: float) -> ToolResult:
    if not world.positions:
        return ToolResult.fail("what_if", "empty portfolio")
    m = float(move_pct) / 100.0
    if symbol == "ALL":
        ps = world.positions
    else:
        if (err := _check(world, "what_if", symbol)) is not None:
            return err
        ps = world.positions_for(symbol)
        if not ps:
            return ToolResult.fail("what_if", "not held", words={"symbol": symbol, "held": "no"})
    total = world.total_equity()
    change = 0.0
    for p in ps:
        spot = world.quote(p.symbol).price * (1 + m)
        change += (world.mark(p, spot=spot) - world.mark(p)) * p.qty * p.mult
    frac = abs(change) / total
    impact = "minor" if frac < 0.01 else ("moderate" if frac < 0.05 else "severe")
    return ToolResult("what_if", words={"symbol": symbol, "impact": impact,
                                        "direction": "gain" if change >= 0 else "loss"},
                      slots={"pnl_change": change, "new_total": total + change})


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def resolve_symbol(world: World, text: str) -> ToolResult:
    raw = (text or "").strip()
    up = re.sub(r"[^A-Za-z]", "", raw).upper()
    if raw.upper() in world.universe or (re.fullmatch(r"[A-Z ]+", raw.strip()) and up in world.universe
                                         and len(raw.split()) > 1):
        sym = raw.upper() if raw.upper() in world.universe else up
        return ToolResult("resolve_symbol", words={"symbol": sym, "name": _name(world, sym), "confidence": "high",
                                                   "alternatives": list(world.lookalikes.get(sym, ()))})
    key = _norm(raw)
    scores: dict[str, float] = {}
    for sym, t in world.universe.items():
        cands = [sym.lower(), t.name] + list(t.spoken)
        best = max(difflib.SequenceMatcher(None, key, _norm(c)).ratio() for c in cands)
        scores[sym] = best
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not key or ranked[0][1] < 0.5:
        return ToolResult.fail("resolve_symbol", "unknown symbol",
                               words={"alternatives": [s for s, _ in ranked[:3]]})
    best, sc = ranked[0]
    second = ranked[1][1]
    conf = "high" if sc >= 0.85 and sc - second >= 0.08 else "low"
    alts = list(dict.fromkeys(list(world.lookalikes.get(best, ())) + [s for s, v in ranked[1:4] if v >= 0.5]))
    return ToolResult("resolve_symbol", words={"symbol": best, "name": _name(world, best), "confidence": conf,
                                               "alternatives": alts[:3]})


def draft_order(world: World, symbol: str, direction: str, qty: int, type: str = "market",
                side: str | None = None, strike: float | None = None,
                expiry_rule: str | None = None) -> ToolResult:
    if (err := _check(world, "draft_order", symbol)) is not None:
        return err
    if int(qty) <= 0:
        return ToolResult.fail("draft_order", "quantity must be positive")
    q = world.quote(symbol)
    order = {"symbol": symbol, "direction": direction, "qty": int(qty), "order_type": type, "kind": "stock"}
    if side:
        exp = world.expiry_for(symbol, expiry_rule or "nearest")
        if exp is None:
            return ToolResult.fail("draft_order", "no chain for that expiry")
        k = world.strike_grid(symbol, float(strike) if strike else q.price, n=1)[0]
        mark = world.option_mark(symbol, k, exp, side)
        order.update(kind="option", side=side, strike=k, expiry=exp.isoformat(), est_price=mark,
                     est_cost=mark * int(qty) * 100)
    else:
        order.update(est_price=q.price, est_cost=q.price * int(qty))
    if type == "limit":
        order["limit_price"] = order["est_price"]
    tid = world.new_ticket(order)
    label = f"{direction.upper()} {symbol} {side if side else 'shares'}"
    return ToolResult("draft_order", words={"status": "needs confirmation", "order": label},
                      slots={"ticket_id": tid})


TOOLS = {
    "get_quote": get_quote, "get_position": get_position, "get_portfolio": get_portfolio,
    "get_indicators": get_indicators, "get_levels": get_levels, "get_option_chain": get_option_chain,
    "get_orders": get_orders, "get_news": get_news, "get_fundamentals": get_fundamentals,
    "get_market": get_market, "screen_watchlist": screen_watchlist, "what_if": what_if,
    "resolve_symbol": resolve_symbol, "draft_order": draft_order,
}
