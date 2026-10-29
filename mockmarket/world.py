"""Seeded mock market. `World(seed, scenario)` is fully deterministic for a (seed, scenario) pair.

Pure numpy/math. Prices are generated once at construction; every tool reads the same numbers,
so get_quote's price is exactly the price used in position P&L.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np

from .bs import bs
import zlib

from .universe import BY_SYMBOL, CORE, EXTRA, INDEX_SYMBOL, LOOKALIKES, UNIVERSE, Ticker

SCENARIOS = (None, "gap_down", "earnings_tomorrow", "empty_portfolio")
TODAY = dt.date(2026, 10, 27)          # a Tuesday; fixed so worlds are reproducible
SESSION_BARS = 78                       # 5-min bars, 9:30-16:00
DAILY_BARS = 200
MULT = 100                              # option contract multiplier
# names with very deep listed options (at-the-money open interest in the tens of thousands)
DEEP_OPTIONS = {"NVDA", "TSLA", "AAPL", "SPY", "QQQ", "AMD", "PLTR", "AMZN", "META", "MSFT", "COIN", "SOFI", "HOOD",
                "APP", "SMCI", "GOOGL", "INTC", "MU", "AVGO", "NFLX"}


@dataclass
class Quote:
    symbol: str
    price: float
    prev_close: float
    open: float
    high: float
    low: float
    volume: int
    avg_volume: int
    iv: float
    daily: dict            # arrays: open, high, low, close, volume (DAILY_BARS completed sessions)
    intraday: dict         # arrays: open, high, low, close, volume, time (bars so far today)
    hourly_close: np.ndarray
    market_cap: float
    pe: float
    target: float
    earnings_date: dt.date

    @property
    def change_pct(self) -> float:
        return self.price / self.prev_close - 1.0


@dataclass
class Position:
    symbol: str
    kind: str                      # "stock" | "option"
    qty: int
    avg_cost: float                # per share / per contract-share (option premium)
    side: str | None = None        # call | put
    strike: float | None = None
    expiry: dt.date | None = None

    @property
    def mult(self) -> int:
        return MULT if self.kind == "option" else 1


@dataclass
class Order:
    order_id: str
    symbol: str
    direction: str                 # buy | sell
    qty: int
    price: float
    status: str                    # open | filled
    day: str                       # today | yesterday
    kind: str = "stock"
    side: str | None = None
    strike: float | None = None
    expiry: dt.date | None = None


@dataclass
class Event:
    name: str
    date: dt.date
    time: str                      # "08:30"


_GOOD = ["{n} beats expectations as demand stays strong", "Analysts lift their view on {n}",
         "{n} unveils new product line to upbeat reviews", "Big fund adds to its {n} stake",
         "{n} wins a major contract"]
_BAD = ["{n} slips after a cautious outlook", "Regulators open a probe into {n}",
        "Analyst downgrades {n} on valuation", "{n} faces supply chain delays",
        "Insider selling at {n} draws attention"]
_MIX = ["{n} holds investor day with few surprises", "{n} reshuffles its leadership team",
        "Options traders split on {n} ahead of earnings", "{n} trades sideways as sector rotates"]

_EVENTS = ["consumer price index report", "Fed rate decision", "jobs report",
           "retail sales report", "GDP report", "producer price index report"]


def third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    first_fri = d + dt.timedelta(days=(4 - d.weekday()) % 7)
    return first_fri + dt.timedelta(days=14)


class World:
    def __init__(self, seed: int = 7, scenario: str | None = None):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; one of {SCENARIOS}")
        self.seed = int(seed)
        self.scenario = scenario
        self.rng = np.random.default_rng(self.seed)
        self.today = TODAY
        self.bars_elapsed = int(self.rng.integers(18, SESSION_BARS + 1))
        self.universe: dict[str, Ticker] = dict(BY_SYMBOL)
        self.lookalikes = LOOKALIKES
        self.tickets: dict[str, dict] = {}
        self._ticket_n = 0
        self._build_quotes()
        self._build_portfolio()
        self._build_orders()
        self._build_news()
        self._build_calendar()
        self.watchlist = self._pick(12, exclude=("SPY", "QQQ"))

    # ------------------------------------------------------------------ build
    def _pick(self, n: int, exclude=()) -> list[str]:
        syms = [t.symbol for t in CORE if t.symbol not in exclude]
        idx = self.rng.choice(len(syms), size=min(n, len(syms)), replace=False)
        return [syms[i] for i in sorted(idx)]

    def _build_quotes(self) -> None:
        rng = self.rng
        n = DAILY_BARS
        mkt = rng.normal(0.0004, 0.010, n)                       # shared market factor
        mkt_gap = -0.03 if self.scenario == "gap_down" else float(rng.normal(0, 0.004))
        mkt_intraday = rng.normal(0, 0.0009, SESSION_BARS)
        self.quotes: dict[str, Quote] = {}
        self._rng_x = np.random.default_rng([self.seed, 1])      # separate stream for EXTRA tickers
        for t in CORE + EXTRA:
            if t is EXTRA[0]:
                rng = self._rng_x
            sd = t.vol / math.sqrt(252)
            beta = float(rng.uniform(0.6, 1.5))
            regime = float(rng.choice([-1.0, 0.0, 0.0, 1.0]))   # recent trend regime: pushes RSI around
            rets = beta * mkt + rng.normal(0, sd * 0.8, n)
            rets[-15:] += regime * sd * 0.55
            start = t.base_price * math.exp(rng.normal(0, 0.15))
            close = start * np.exp(np.cumsum(rets))
            close *= t.base_price * math.exp(rng.normal(0, 0.06)) / close[-1]   # anchor near base price
            opn = np.concatenate([[close[0] * math.exp(-rets[0])], close[:-1]]) * np.exp(rng.normal(0, sd * 0.25, n))
            wig = np.abs(rng.normal(0, sd * 0.5, n))
            high = np.maximum(opn, close) * np.exp(wig)
            low = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(0, sd * 0.5, n)))
            avg_vol = int(rng.lognormal(math.log(3e9 / t.base_price), 0.6))
            vol = (avg_vol * rng.lognormal(0, 0.35, n)).astype(np.int64)
            prev_close = float(close[-1])

            # today's session
            gap = beta * mkt_gap + float(rng.normal(0, sd * 0.35))
            if self.scenario == "gap_down":
                gap -= float(rng.uniform(0.0, 0.03))
            k = self.bars_elapsed
            i_rets = beta * mkt_intraday[:k] + rng.normal(0, sd / math.sqrt(SESSION_BARS) * 0.9, k)
            i_open0 = prev_close * math.exp(gap)
            i_close = i_open0 * np.exp(np.cumsum(i_rets))
            i_open = np.concatenate([[i_open0], i_close[:-1]])
            ih = np.maximum(i_open, i_close) * np.exp(np.abs(rng.normal(0, sd * 0.06, k)))
            il = np.minimum(i_open, i_close) * np.exp(-np.abs(rng.normal(0, sd * 0.06, k)))
            x = np.linspace(-1, 1, SESSION_BARS)[:k]
            prof = (1.0 + 2.2 * x ** 2) / (SESSION_BARS * (1 + 2.2 / 3))   # U-shaped volume
            act = float(rng.lognormal(0, 0.45)) * (1.6 if self.scenario == "gap_down" else 1.0)
            i_vol = (avg_vol * prof * act * rng.lognormal(0, 0.3, k)).astype(np.int64)
            times = [f"{(9 * 60 + 30 + 5 * j) // 60:02d}:{(9 * 60 + 30 + 5 * j) % 60:02d}" for j in range(k)]

            price = round(float(i_close[-1]), 2)
            i_close[-1] = price
            # hourly series ending at the current price (synthetic, for 1H indicators)
            h_rets = rng.normal(regime * sd * 0.05, sd / math.sqrt(6.5), 200)
            hourly = price * np.exp(np.cumsum(h_rets[::-1]))[::-1] / math.exp(0)
            hourly = hourly * price / hourly[-1]

            shares_out = float(rng.uniform(0.3, 25.0)) * 1e9 / max(t.base_price / 100.0, 0.3)
            pe = float(rng.uniform(9, 65))
            target = price * math.exp(rng.normal(0.07, 0.12))
            e_days = int(rng.integers(4, 70))
            earnings = self.today + dt.timedelta(days=e_days)
            iv = t.iv * float(rng.uniform(0.85, 1.25)) * (1.35 if self.scenario == "gap_down" else 1.0)

            self.quotes[t.symbol] = Quote(
                symbol=t.symbol, price=price, prev_close=round(prev_close, 2),
                open=round(i_open0, 2), high=round(float(max(ih.max(), price)), 2),
                low=round(float(min(il.min(), price)), 2), volume=int(i_vol.sum()), avg_volume=avg_vol,
                iv=iv,
                daily={"open": opn, "high": high, "low": low, "close": close, "volume": vol},
                intraday={"open": i_open, "high": ih, "low": il, "close": i_close, "volume": i_vol, "time": times},
                hourly_close=hourly, market_cap=price * shares_out, pe=pe, target=target,
                earnings_date=earnings,
            )

    def _build_portfolio(self) -> None:
        rng = self.rng
        self.positions: list[Position] = []
        self.cash = round(float(rng.uniform(5_000, 80_000)), 2)
        if self.scenario == "empty_portfolio":
            self.positions = []
            return
        n = int(rng.integers(8, 16))
        held = ["NVDA"] + [s for s in self._pick(n + 2, exclude=("NVDA", "SPY", "QQQ"))][: n - 1]
        for sym in held:
            q = self.quotes[sym]
            if sym != "NVDA" and rng.random() < 0.3 and self.universe[sym].options:
                side = "call" if rng.random() < 0.7 else "put"
                exp = self.expiries(sym)
                expiry = exp[min(len(exp) - 1, int(rng.integers(1, 5)))]
                strike = self.strike_grid(sym, q.price * (1 + (0.05 if side == "call" else -0.05)))[2]
                mark = self.option_mark(sym, strike, expiry, side)
                avg = round(max(0.05, mark * math.exp(rng.normal(0, 0.35))), 2)
                self.positions.append(Position(sym, "option", int(rng.integers(1, 11)), avg, side, strike, expiry))
            else:
                qty = int(rng.integers(1, 41)) * 5
                avg = round(q.price * math.exp(rng.normal(-0.03, 0.18)), 2)
                self.positions.append(Position(sym, "stock", qty, avg))
        # NVDA is always held as shares AND a call position (owner's book: "I bought some calls on Nvidia"), so
        # "when do my Nvidia options expire" is answerable. Separate RNG stream keeps the rest of the book unchanged.
        rx = np.random.default_rng([self.seed, 2])
        if len(self.positions) >= 15:
            self.positions.pop()
        q = self.quotes["NVDA"]
        exp = self.expiries("NVDA")
        expiry = exp[int(rx.integers(0, 5))]
        strike = self.strike_grid("NVDA", q.price * 1.05)[2]
        mark = self.option_mark("NVDA", strike, expiry, "call")
        avg = round(max(0.05, mark * math.exp(rx.normal(0, 0.35))), 2)
        self.positions.append(Position("NVDA", "option", int(rx.integers(1, 11)), avg, "call", strike, expiry))
        if self.scenario == "earnings_tomorrow":
            for p in self.positions[:3]:
                self.quotes[p.symbol].earnings_date = self.today + dt.timedelta(days=1)

    def _build_orders(self) -> None:
        rng = self.rng
        self.orders: list[Order] = []
        for i in range(int(rng.integers(0, 6))):
            sym = self._pick(1, exclude=("SPY",))[0]
            q = self.quotes[sym]
            status = "open" if rng.random() < 0.5 else "filled"
            day = "today" if (status == "open" or rng.random() < 0.6) else "yesterday"
            direction = "buy" if rng.random() < 0.65 else "sell"
            if rng.random() < 0.35:
                side = "call" if rng.random() < 0.6 else "put"
                exp = self.expiries(sym)[0]
                strike = self.strike_grid(sym, q.price)[2]
                price = round(self.option_mark(sym, strike, exp, side) * float(rng.uniform(0.9, 1.05)), 2)
                self.orders.append(Order(f"O{i + 1}", sym, direction, int(rng.integers(1, 6)), price, status, day,
                                         "option", side, strike, exp))
            else:
                price = round(q.price * float(rng.uniform(0.95, 1.02)), 2)
                self.orders.append(Order(f"O{i + 1}", sym, direction, int(rng.integers(1, 21)) * 5, price, status, day))

    def _build_news(self) -> None:
        rng = self.rng
        self.news: dict[str, list[dict]] = {}
        for t in CORE + EXTRA:
            if t is EXTRA[0]:
                rng = self._rng_x
            ch = self.quotes[t.symbol].change_pct
            p_good = 0.5 + max(-0.35, min(0.35, ch * 10))
            items = []
            for _ in range(2):
                r = rng.random()
                pool, tone = ((_GOOD, "positive") if r < p_good * 0.85 else
                              (_BAD, "negative") if r < 0.85 else (_MIX, "mixed"))
                items.append({"headline": pool[int(rng.integers(len(pool)))].format(n=t.name), "tone": tone,
                              "age_hours": float(rng.uniform(0.3, 30))})
            items.sort(key=lambda x: x["age_hours"])
            self.news[t.symbol] = items

    def _build_calendar(self) -> None:
        rng = self.rng
        self.events: list[Event] = []
        for i, off in enumerate(sorted(rng.choice(range(0, 10), size=3, replace=False))):
            name = _EVENTS[int(rng.integers(len(_EVENTS)))]
            hh, mm = (8, 30) if "Fed" not in name else (14, 0)
            self.events.append(Event(name, self.today + dt.timedelta(days=int(off)), f"{hh:02d}:{mm:02d}"))

    # --------------------------------------------------------------- queries
    def has(self, symbol: str) -> bool:
        return symbol in self.quotes

    def quote(self, symbol: str) -> Quote:
        return self.quotes[symbol]

    def session_time(self) -> str:
        m = 9 * 60 + 30 + 5 * self.bars_elapsed
        return f"{m // 60:02d}:{m % 60:02d}"

    def strike_step(self, price: float) -> float:
        for lim, step in ((25, 0.5), (60, 1.0), (200, 2.5), (500, 5.0), (1000, 10.0)):
            if price < lim:
                return step
        return 20.0

    def strike_grid(self, symbol: str, center: float, n: int = 5) -> list[float]:
        step = self.strike_step(self.quotes[symbol].price)
        mid = round(center / step) * step
        return [round(mid + (i - n // 2) * step, 2) for i in range(n)]

    def expiries(self, symbol: str) -> list[dt.date]:
        t = self.universe[symbol]
        if not t.options:
            return []
        out = set()
        if t.weeklies:
            fri = self.today + dt.timedelta(days=(4 - self.today.weekday()) % 7)
            for w in range(8):
                out.add(fri + dt.timedelta(weeks=w))
        y, m = self.today.year, self.today.month
        for _ in range(7):
            tf = third_friday(y, m)
            if tf >= self.today:
                out.add(tf)
            m += 1
            if m > 12:
                y, m = y + 1, 1
        return sorted(out)

    def expiry_for(self, symbol: str, rule: str) -> dt.date | None:
        exps = self.expiries(symbol)
        if not exps:
            return None
        fri = self.today + dt.timedelta(days=(4 - self.today.weekday()) % 7)
        if rule == "nearest":
            return exps[0]
        if rule == "this_week":
            return fri if fri in exps else None
        if rule == "next_week":
            nf = fri + dt.timedelta(weeks=1)
            return nf if nf in exps else None
        if rule == "monthly":
            monthlies = [e for e in exps if e.day >= 15 and e.day <= 21 and e.weekday() == 4]
            return monthlies[0] if monthlies else None
        return None

    def years_to(self, expiry: dt.date) -> float:
        # expire at 16:00 on expiry day; time left today counts
        frac_left = (SESSION_BARS - self.bars_elapsed) / SESSION_BARS * (6.5 / 24)
        return max(((expiry - self.today).days + frac_left) / 365.0, 1e-4)

    def strike_iv(self, symbol: str, strike: float, expiry: dt.date) -> float:
        q = self.quotes[symbol]
        m = math.log(strike / q.price)
        term = 1.0 + 0.15 / math.sqrt(max((expiry - self.today).days, 1))
        return q.iv * term * (1.0 + 1.6 * m * m - 0.25 * m)

    def option_greeks(self, symbol: str, strike: float, expiry: dt.date, side: str, spot: float | None = None,
                      day_shift: int = 0) -> dict:
        q = self.quotes[symbol]
        spot = q.price if spot is None else spot
        t = self.years_to(expiry) + day_shift / 365.0
        iv = self.strike_iv(symbol, strike, expiry)
        g = bs(spot, strike, t, iv, side)
        g["iv"] = iv
        return g

    def option_mark(self, symbol: str, strike: float, expiry: dt.date, side: str, spot: float | None = None,
                    day_shift: int = 0) -> float:
        return round(self.option_greeks(symbol, strike, expiry, side, spot, day_shift)["price"], 2)

    def chain(self, symbol: str, expiry: dt.date, side: str, strike: float | None = None) -> list[dict]:
        q = self.quotes[symbol]
        center = strike if strike is not None else q.price
        rows = []
        for k in self.strike_grid(symbol, center):
            if k <= 0:
                continue
            g = self.option_greeks(symbol, k, expiry, side)
            mid = g["price"]
            half = max(0.01, mid * 0.015 + 0.02)
            rows.append({"strike": k, "bid": round(max(0.01, mid - half), 2), "ask": round(mid + half, 2),
                         "iv": g["iv"], "delta": g["delta"], "mid": round(mid, 2),
                         "oi": self.open_interest(symbol, k, expiry, side)})
        return rows

    def open_interest(self, symbol: str, strike: float, expiry: dt.date, side: str) -> int:
        """Deterministic open interest: scales with the name's liquidity, peaks at the money, decays with tenor."""
        q = self.quotes[symbol]
        t = self.universe[symbol]
        base = (40_000 if symbol in DEEP_OPTIONS else 8_000 if t.weeklies else 1_200)
        money = abs(math.log(strike / q.price)) / max(self.universe[symbol].iv, 0.1)
        tenor = 1.0 / (1.0 + max((expiry - self.today).days, 0) / 60.0)
        h = zlib.crc32(f"{self.seed}|{symbol}|{strike}|{expiry}|{side}".encode()) / 2 ** 32
        noise = math.exp((h - 0.5) * 1.2) * (0.7 if side == "put" else 1.0)
        return max(0, int(base * math.exp(-6.0 * money ** 2) * (0.5 + tenor) * noise))

    # ---------------------------------------------------------- portfolio math
    def mark(self, p: Position, spot: float | None = None, day_shift: int = 0) -> float:
        if p.kind == "stock":
            return self.quotes[p.symbol].price if spot is None else spot
        return self.option_mark(p.symbol, p.strike, p.expiry, p.side, spot, day_shift)

    def market_value(self, p: Position) -> float:
        return self.mark(p) * p.qty * p.mult

    def pnl(self, p: Position) -> float:
        return (self.mark(p) - p.avg_cost) * p.qty * p.mult

    def day_pnl_of(self, p: Position) -> float:
        q = self.quotes[p.symbol]
        prev = (q.prev_close if p.kind == "stock"
                else self.option_mark(p.symbol, p.strike, p.expiry, p.side, q.prev_close, day_shift=1))
        return (self.mark(p) - prev) * p.qty * p.mult

    def total_equity(self) -> float:
        return self.cash + sum(self.market_value(p) for p in self.positions)

    def positions_for(self, symbol: str) -> list[Position]:
        return [p for p in self.positions if p.symbol == symbol]

    def new_ticket(self, order: dict) -> str:
        self._ticket_n += 1
        tid = f"T{self.seed}-{self._ticket_n}"
        self.tickets[tid] = dict(order, ticket_id=tid, status="needs confirmation")
        return tid

    @property
    def index_symbol(self) -> str:
        return INDEX_SYMBOL
