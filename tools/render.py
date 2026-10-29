"""Slot rendering: `display()` for the screen, `spoken()` for TTS (no digits, no cents, sensible rounding)."""
from __future__ import annotations

import datetime as dt
import re

from .numwords import decimal_words, int_words, ordinal_words, price_words

KINDS = ("price", "pct", "signed_pct", "money", "signed_money", "count", "ratio", "index", "greek",
         "vol", "date", "time", "age", "text")

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]


def _as_date(v) -> dt.date:
    if isinstance(v, dt.date):
        return v
    return dt.date.fromisoformat(str(v)[:10])


def _short_num(x: float) -> str:
    a = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= div:
            return f"{x / div:.1f}{suf}"
    return f"{x:,.0f}"


def display_kind(kind: str, value) -> str:
    if value is None:
        return "n/a"
    if kind == "price":
        return f"${value:,.2f}"
    if kind in ("pct", "vol"):
        return f"{value * 100:.1f}%"
    if kind == "signed_pct":
        return f"{value * 100:+.1f}%"
    if kind == "money":
        a = abs(value)
        if a >= 1e6:
            return ("-" if value < 0 else "") + "$" + _short_num(a)
        return f"${value:,.0f}" if a >= 1000 else f"${value:,.2f}"
    if kind == "signed_money":
        sign = "+" if value >= 0 else "-"
        a = abs(value)
        body = _short_num(a) if a >= 1e6 else (f"{a:,.0f}" if a >= 1000 else f"{a:,.2f}")
        return f"{sign}${body}"
    if kind == "count":
        return _short_num(value) if abs(value) >= 1e6 else f"{int(round(value)):,}"
    if kind == "ratio":
        return f"{value:.1f}x"
    if kind == "index":
        return f"{value:.1f}"
    if kind == "greek":
        return f"{value:.2f}"
    if kind == "date":
        d = _as_date(value)
        return f"{_MONTHS[d.month - 1][:3]} {d.day}"
    if kind == "time":
        hh, mm = (int(x) for x in str(value).split(":")[:2])
        return f"{(hh - 1) % 12 + 1}:{mm:02d} {'AM' if hh < 12 else 'PM'}"
    if kind == "age":
        h = float(value)
        if h < 1:
            return f"{max(1, int(round(h * 60)))}m ago"
        if h < 24:
            return f"{int(round(h))}h ago"
        return f"{int(round(h / 24))}d ago"
    return str(value)


def _about(exact: float, rounded: float, tol: float) -> str:
    return "about " if abs(exact - rounded) > tol else ""


def _money_words(a: float) -> str:
    """Unsigned money, rounded for speech."""
    if a < 1:
        return int_words(int(round(a * 100))) + " cents"
    if a < 1000:
        n = int(round(a))
        return _about(a, n, 0.5) + int_words(n) + (" dollar" if n == 1 else " dollars")
    if a < 10_000:
        n = int(round(a / 100.0)) * 100
        return _about(a, n, 1) + int_words(n) + " dollars"
    if a < 1e6:
        n = int(round(a / 1000.0))
        if n >= 1000:
            return "about one million dollars"
        return _about(a, n * 1000, 1) + int_words(n) + " thousand dollars"
    for div, name in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if a >= div:
            v = round(a / div, 1)
            return "about " + decimal_words(v, 1) + " " + name + " dollars"
    raise AssertionError


def _pct_words(p: float) -> str:
    """p already in percent units, unsigned."""
    v = round(abs(p), 1)
    return decimal_words(v, 1) + " percent"


def spoken_kind(kind: str, value) -> str:
    if value is None:
        return "not available"
    if kind == "price":
        exact = abs(float(value))
        w = price_words(exact)
        rounded_away = (exact >= 20 and abs(exact - round(exact)) >= 0.005) or \
                       (1 <= exact < 20 and abs(exact * 20 - round(exact * 20)) > 1e-6)
        return ("about " if rounded_away else "") + w
    if kind == "pct":
        return _pct_words(value * 100)
    if kind == "signed_pct":
        p = value * 100
        if abs(p) < 0.05:
            return "flat"
        return ("up " if p > 0 else "down ") + _pct_words(p)
    if kind == "money":
        return ("minus " if value < 0 else "") + _money_words(abs(value))
    if kind == "signed_money":
        if abs(value) < 0.5:
            return "flat"
        return ("up " if value > 0 else "down ") + _money_words(abs(value))
    if kind == "count":
        n = abs(float(value))
        if n >= 1e6:
            for div, name in ((1e9, "billion"), (1e6, "million")):
                if n >= div:
                    return "about " + decimal_words(round(n / div, 1), 1) + " " + name
        if n >= 10_000:
            return "about " + int_words(int(round(n / 1000.0))) + " thousand"
        return int_words(int(round(n)))
    if kind == "ratio":
        r = round(float(value), 1)
        if 0.9 <= r <= 1.1:
            return "about average"                       # "one times" says nothing; "normal" already does
        return decimal_words(r, 1) + " times"
    if kind == "index":
        return int_words(int(round(float(value))))
    if kind == "greek":
        w = decimal_words(round(float(value), 2), 2)     # "zero point five nine" -> "point five nine"
        return re.sub(r"^(minus )?zero point", r"\1point", w)
    if kind == "vol":                                    # implied vol: whole percent ("fifty-eight percent")
        return int_words(int(round(float(value) * 100))) + " percent"
    if kind == "date":
        d = _as_date(value)
        return f"{_MONTHS[d.month - 1]} {ordinal_words(d.day)}"
    if kind == "time":
        hh, mm = (int(x) for x in str(value).split(":")[:2])
        h12 = (hh - 1) % 12 + 1
        mins = "" if mm == 0 else (" oh " + int_words(mm) if mm < 10 else " " + int_words(mm))
        return int_words(h12) + mins + (" a m" if hh < 12 else " p m")
    if kind == "age":
        h = float(value)
        if h < 1:
            m = max(1, int(round(h * 60)))
            return f"{int_words(m)} minute{'s' if m != 1 else ''} ago"
        if h < 24:
            n = int(round(h))
            return f"{int_words(n)} hour{'s' if n != 1 else ''} ago"
        n = int(round(h / 24))
        return f"{int_words(n)} day{'s' if n != 1 else ''} ago"
    return re.sub(r"\d", lambda m: " " + int_words(int(m.group())) + " ", str(value)).replace("  ", " ").strip()


def display(slot: str, value, tool: str | None = None) -> str:
    """Screen text for a slot; kind inferred from the per-tool slot-type map."""
    from .slots import kind_of
    return display_kind(kind_of(slot, tool), value)


def spoken(slot: str, value, tool: str | None = None) -> str:
    """TTS text for a slot (never contains digits)."""
    from .slots import kind_of
    return spoken_kind(kind_of(slot, tool), value)
