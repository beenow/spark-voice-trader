"""Integers and decimals to English words (for TTS-safe spoken text)."""
from __future__ import annotations

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
        "nine": "ninth", "twelve": "twelfth"}


def int_words(n: int) -> str:
    n = int(n)
    if n < 0:
        return "minus " + int_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else "-" + _ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + ("" if r == 0 else " " + int_words(r))
    for div, name in ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= div:
            hi, r = divmod(n, div)
            return int_words(hi) + " " + name + ("" if r == 0 else " " + int_words(r))
    raise AssertionError


def decimal_words(x: float, places: int = 1) -> str:
    """1.8 -> 'one point eight'; 2.0 -> 'two'; 0.05 (places=2) -> 'zero point zero five'."""
    neg = x < 0
    s = f"{abs(x):.{places}f}"
    whole, _, frac = s.partition(".")
    frac = frac.rstrip("0")
    out = int_words(int(whole))
    if frac:
        out += " point " + " ".join(_ONES[int(c)] for c in frac)
    return ("minus " + out) if neg else out


def ordinal_words(n: int) -> str:
    w = int_words(n)
    head, sep, last = w.rpartition("-") if "-" in w.split(" ")[-1] else w.rpartition(" ")
    if last in _ORD:
        last = _ORD[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last = last + "th"
    return head + sep + last


def price_words(x: float) -> str:
    """Trader-style price, no cents: 182.4 -> 'one eighty-two'; 45.67 -> 'forty-six'; 3.45 -> 'three forty-five'."""
    x = abs(x)
    if x < 1:
        c = int(round(x * 100))
        return int_words(c) + " cents"
    if x < 20:
        r = round(x * 20) / 20                     # nearest nickel
        d, c = int(r), int(round((r - int(r)) * 100))
        if c == 0:
            return int_words(d)
        return int_words(d) + " " + (("oh " + _ONES[c]) if c < 10 else int_words(c))
    n = int(round(x))
    if n < 100:
        return int_words(n)
    if n < 1000:
        h, r = divmod(n, 100)
        if r == 0:
            return int_words(n)
        return _ONES[h] + " " + (("oh " + _ONES[r]) if r < 10 else int_words(r))
    if n < 10000:
        hi, r = divmod(n, 100)
        if r == 0:
            return int_words(n)
        if hi % 10 == 0:                           # 2005 -> two thousand five
            return int_words(n)
        return int_words(hi) + " " + (("oh " + _ONES[r]) if r < 10 else int_words(r))
    return int_words(n)
