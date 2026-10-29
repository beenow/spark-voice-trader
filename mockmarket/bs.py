"""Black-Scholes price + greeks (pure math, no scipy)."""
from __future__ import annotations

import math


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs(spot: float, strike: float, t_years: float, iv: float, side: str, r: float = 0.04) -> dict:
    """Return {price, delta, gamma, theta (per day), vega (per vol point)}."""
    t = max(t_years, 1e-6)
    iv = max(iv, 1e-4)
    sq = iv * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / sq
    d2 = d1 - sq
    disc = math.exp(-r * t)
    if side == "call":
        price = spot * _ncdf(d1) - strike * disc * _ncdf(d2)
        delta = _ncdf(d1)
        theta = -spot * _npdf(d1) * iv / (2 * math.sqrt(t)) - r * strike * disc * _ncdf(d2)
    else:
        price = strike * disc * _ncdf(-d2) - spot * _ncdf(-d1)
        delta = _ncdf(d1) - 1.0
        theta = -spot * _npdf(d1) * iv / (2 * math.sqrt(t)) + r * strike * disc * _ncdf(-d2)
    gamma = _npdf(d1) / (spot * sq)
    vega = spot * _npdf(d1) * math.sqrt(t) / 100.0
    intrinsic = max(0.0, spot - strike) if side == "call" else max(0.0, strike - spot)
    return {"price": max(price, intrinsic, 0.01), "delta": delta, "gamma": gamma,
            "theta": theta / 365.0, "vega": vega}
