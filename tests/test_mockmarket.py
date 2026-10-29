import math

import numpy as np
import pytest

from mockmarket import LOOKALIKES, SCENARIOS, UNIVERSE, World
from mockmarket.bs import bs
from mockmarket.world import DAILY_BARS


def test_universe_size_and_lookalikes():
    syms = {t.symbol for t in UNIVERSE}
    assert 38 <= len(syms) <= 60
    for a, b in [("OR", "ORLY"), ("META", "MET"), ("NVDA", "NVDL")]:
        assert a in syms and b in syms
        assert b in LOOKALIKES[a]
    assert all(t.spoken for t in UNIVERSE)
    assert "Nvidia" in next(t for t in UNIVERSE if t.symbol == "NVDA").spoken


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_deterministic(scenario):
    a, b = World(11, scenario), World(11, scenario)
    for s in a.quotes:
        assert a.quote(s).price == b.quote(s).price
        assert np.array_equal(a.quote(s).daily["close"], b.quote(s).daily["close"])
    assert [(p.symbol, p.qty, p.avg_cost) for p in a.positions] == [(p.symbol, p.qty, p.avg_cost) for p in b.positions]
    assert a.cash == b.cash
    assert [o.__dict__ for o in a.orders] == [o.__dict__ for o in b.orders]


def test_seeds_differ():
    assert World(1).quote("NVDA").price != World(2).quote("NVDA").price


def test_bars_and_portfolio():
    for seed in range(20):
        w = World(seed)
        q = w.quote("AAPL")
        assert len(q.daily["close"]) == DAILY_BARS
        assert len(q.intraday["close"]) == w.bars_elapsed
        assert q.intraday["close"][-1] == q.price
        assert q.low <= q.price <= q.high
        assert 8 <= len(w.positions) <= 15
        assert any(p.kind == "option" for p in w.positions) or seed  # options appear in most worlds
        assert w.positions_for("NVDA")
        assert w.total_equity() > w.cash


def test_options_present_somewhere():
    assert any(p.kind == "option" for s in range(10) for p in World(s).positions)


def test_scenarios():
    assert World(3, "empty_portfolio").positions == []
    gd, base = World(3, "gap_down"), World(3)
    assert gd.quote("SPY").change_pct < -0.01
    assert gd.quote("SPY").change_pct < base.quote("SPY").change_pct
    e = World(3, "earnings_tomorrow")
    held = {p.symbol for p in e.positions}
    assert any((e.quote(s).earnings_date - e.today).days == 1 for s in held)
    with pytest.raises(ValueError):
        World(3, "nope")


def test_black_scholes_parity_and_greeks():
    S, K, T, iv, r = 100.0, 105.0, 0.25, 0.4, 0.04
    c, p = bs(S, K, T, iv, "call", r), bs(S, K, T, iv, "put", r)
    assert math.isclose(c["price"] - p["price"], S - K * math.exp(-r * T), abs_tol=1e-6)
    assert 0 < c["delta"] < 1 and -1 < p["delta"] < 0
    assert c["gamma"] > 0 and c["vega"] > 0 and c["theta"] < 0


def test_chain_and_expiries():
    w = World(7)
    nv = w.expiries("NVDA")
    assert w.expiry_for("NVDA", "this_week") == nv[0]
    assert w.expiry_for("MET", "this_week") is None           # monthlies only
    assert w.expiry_for("MET", "monthly") is not None
    rows = w.chain("NVDA", nv[1], "call")
    assert len(rows) == 5
    assert all(r["bid"] < r["ask"] for r in rows)
    deltas = [r["delta"] for r in rows]
    assert deltas == sorted(deltas, reverse=True)


def test_pnl_uses_quote_price():
    w = World(5)
    for p in w.positions:
        if p.kind == "stock":
            assert math.isclose(w.pnl(p), (w.quote(p.symbol).price - p.avg_cost) * p.qty)
