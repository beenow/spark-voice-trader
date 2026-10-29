import re

from tools.numwords import int_words, ordinal_words, price_words
from tools.render import display, display_kind, spoken, spoken_kind


def test_display():
    assert display("support", 182.4, "get_levels") == "$182.40"
    assert display("change_pct", -0.023, "get_quote") == "-2.3%"
    assert display("dist_support", 0.018, "get_levels") == "1.8%"
    assert display("pnl", -12865.2, "get_position") == "-$12,865"
    assert display("price_2", 10.0) == "$10.00"          # turn suffix resolved
    assert display("order_3_price", 4.5) == "$4.50"
    assert display_kind("date", "2026-10-28") == "Oct 28"
    assert display_kind("time", "14:00") == "2:00 PM"


def test_spoken_examples():
    assert spoken("support", 182.4, "get_levels") == "about one eighty-two"
    assert spoken("change_pct", -0.023, "get_quote") == "down two point three percent"
    assert spoken("dist_support", 0.018) == "one point eight percent"
    assert spoken_kind("price", 182.0) == "one eighty-two"
    assert spoken_kind("price", 1205.3) == "about twelve oh five"
    assert spoken_kind("price", 3.45) == "three forty-five"
    assert spoken_kind("price", 0.45) == "forty-five cents"
    assert spoken_kind("signed_money", 12345.67) == "up about twelve thousand dollars"
    assert spoken_kind("signed_money", -3391.0) == "down about three thousand four hundred dollars"
    assert spoken_kind("money", 412.0) == "four hundred twelve dollars"
    assert spoken_kind("money", 2.3e12) == "about two point three trillion dollars"
    assert spoken_kind("signed_pct", 0.0001) == "flat"
    assert spoken_kind("count", 45_123_456) == "about forty-five point one million"
    assert spoken_kind("ratio", 1.84) == "one point eight times"
    assert spoken_kind("date", "2026-10-28") == "October twenty-eighth"
    assert spoken_kind("time", "08:30") == "eight thirty a m"
    assert spoken_kind("age", 3.2) == "three hours ago"


def test_no_cents_no_digits():
    for v in [0.07, 1.0, 9.99, 19.97, 20.5, 99.99, 182.4, 999.9, 1000.0, 2005.0, 12345.6, 512.04]:
        s = spoken_kind("price", v)
        assert "cent" not in s or v < 1
        assert not re.search(r"\d", s)
    for v in [-1e9, -5e5, 0.3, 999.4, 5e7]:
        assert not re.search(r"\d", spoken_kind("signed_money", v))


def test_numwords():
    assert int_words(0) == "zero"
    assert int_words(1234567) == "one million two hundred thirty-four thousand five hundred sixty-seven"
    assert ordinal_words(1) == "first" and ordinal_words(22) == "twenty-second" and ordinal_words(30) == "thirtieth"
    assert price_words(105.0) == "one oh five"
