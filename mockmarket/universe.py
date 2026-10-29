"""Static ticker universe: symbol, display name, spoken aliases, sector, base price, annual vol, base IV.

Symbols never contain digits. Lookalike pairs (OR/ORLY, META/MET, NVDA/NVDL ...) are deliberate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ticker:
    symbol: str
    name: str
    spoken: tuple[str, ...]
    sector: str
    base_price: float
    vol: float        # annualised realised vol
    iv: float         # base implied vol
    weeklies: bool = True
    options: bool = True
    quote_only: bool = False      # index-like (VIX): get_quote works, other symbol tools answer "quote only symbol"


_T = Ticker
CORE: tuple[Ticker, ...] = (
    _T("NVDA", "Nvidia", ("Nvidia", "en vidia", "N V D A", "invidia"), "semis", 182.0, 0.50, 0.48),
    _T("NVDL", "GraniteShares Nvidia Long", ("N V D L", "nvidia leveraged", "nvdl"), "etf", 78.0, 1.00, 0.95, weeklies=False),
    _T("AAPL", "Apple", ("Apple", "A A P L"), "tech", 228.0, 0.25, 0.24),
    _T("MSFT", "Microsoft", ("Microsoft", "M S F T"), "tech", 512.0, 0.24, 0.23),
    _T("AMZN", "Amazon", ("Amazon", "A M Z N"), "consumer", 221.0, 0.32, 0.31),
    _T("GOOGL", "Alphabet", ("Google", "Alphabet", "G O O G L"), "tech", 246.0, 0.30, 0.29),
    _T("META", "Meta", ("Meta", "Facebook", "M E T A"), "tech", 735.0, 0.36, 0.35),
    _T("MET", "MetLife", ("MetLife", "Met Life", "M E T"), "financials", 79.0, 0.22, 0.21, weeklies=False),
    _T("TSLA", "Tesla", ("Tesla", "T S L A"), "auto", 432.0, 0.60, 0.58),
    _T("AMD", "AMD", ("A M D", "Advanced Micro Devices", "AMD"), "semis", 162.0, 0.50, 0.49),
    _T("AVGO", "Broadcom", ("Broadcom", "A V G O"), "semis", 348.0, 0.42, 0.40),
    _T("INTC", "Intel", ("Intel", "I N T C"), "semis", 37.0, 0.45, 0.47),
    _T("MU", "Micron", ("Micron", "M U"), "semis", 196.0, 0.55, 0.52),
    _T("TSM", "Taiwan Semiconductor", ("T S M", "Taiwan Semi", "TSMC"), "semis", 291.0, 0.38, 0.36),
    _T("NFLX", "Netflix", ("Netflix", "N F L X"), "media", 1205.0, 0.35, 0.33),
    _T("DIS", "Disney", ("Disney", "D I S"), "media", 112.0, 0.26, 0.25, weeklies=False),
    _T("JPM", "JPMorgan", ("JPMorgan", "J P Morgan", "Chase"), "financials", 303.0, 0.22, 0.21),
    _T("BAC", "Bank of America", ("Bank of America", "B A C"), "financials", 51.0, 0.26, 0.25),
    _T("GS", "Goldman Sachs", ("Goldman", "Goldman Sachs", "G S"), "financials", 781.0, 0.28, 0.27),
    _T("V", "Visa", ("Visa", "V"), "financials", 344.0, 0.20, 0.19),
    _T("MA", "Mastercard", ("Mastercard", "M A"), "financials", 569.0, 0.21, 0.20),
    _T("XOM", "Exxon", ("Exxon", "Exxon Mobil", "X O M"), "energy", 113.0, 0.24, 0.23),
    _T("CVX", "Chevron", ("Chevron", "C V X"), "energy", 154.0, 0.24, 0.23),
    _T("OXY", "Occidental", ("Occidental", "O X Y"), "energy", 44.0, 0.33, 0.32, weeklies=False),
    _T("OR", "Osisko Gold Royalties", ("Osisko", "O R", "Osisko Gold"), "materials", 31.0, 0.38, 0.40, weeklies=False),
    _T("ORLY", "O'Reilly Automotive", ("O'Reilly", "O Reilly", "O R L Y", "Oh Reilly"), "consumer", 103.0, 0.24, 0.23, weeklies=False),
    _T("COST", "Costco", ("Costco", "C O S T"), "consumer", 921.0, 0.20, 0.19),
    _T("WMT", "Walmart", ("Walmart", "W M T"), "consumer", 103.0, 0.20, 0.19),
    _T("KO", "Coca-Cola", ("Coke", "Coca Cola", "K O"), "staples", 69.0, 0.15, 0.15, weeklies=False),
    _T("PEP", "PepsiCo", ("Pepsi", "PepsiCo", "P E P"), "staples", 148.0, 0.17, 0.17, weeklies=False),
    _T("UNH", "UnitedHealth", ("United Health", "UnitedHealth", "U N H"), "health", 352.0, 0.35, 0.34),
    _T("LLY", "Eli Lilly", ("Lilly", "Eli Lilly", "L L Y"), "health", 815.0, 0.33, 0.32),
    _T("PFE", "Pfizer", ("Pfizer", "P F E"), "health", 25.0, 0.24, 0.24, weeklies=False),
    _T("PLTR", "Palantir", ("Palantir", "P L T R"), "tech", 184.0, 0.70, 0.66),
    _T("COIN", "Coinbase", ("Coinbase", "C O I N"), "financials", 352.0, 0.75, 0.70),
    _T("SHOP", "Shopify", ("Shopify", "S H O P"), "tech", 163.0, 0.55, 0.52),
    _T("UBER", "Uber", ("Uber", "U B E R"), "tech", 96.0, 0.40, 0.38),
    _T("BA", "Boeing", ("Boeing", "B A"), "industrials", 216.0, 0.34, 0.33),
    _T("CAT", "Caterpillar", ("Caterpillar", "C A T"), "industrials", 527.0, 0.30, 0.29),
    _T("SPY", "S&P five hundred ETF", ("S P Y", "spy", "the S and P", "S and P"), "etf", 671.0, 0.15, 0.14),
    _T("QQQ", "Nasdaq one hundred ETF", ("Q Q Q", "queues", "the Nasdaq", "triple Q"), "etf", 611.0, 0.19, 0.18),
    _T("SOFI", "SoFi", ("SoFi", "So Fi", "S O F I"), "financials", 28.0, 0.60, 0.62, weeklies=False),
    _T("HOOD", "Robinhood", ("Robinhood", "H O O D"), "financials", 138.0, 0.70, 0.68),
)

# Extra real tickers that traders ask about. Built from a separate RNG stream after CORE, and never picked for
# the portfolio/watchlist, so worlds over CORE stay identical to before they were added.
EXTRA: tuple[Ticker, ...] = (
    _T("APP", "AppLovin", ("AppLovin", "App Lovin", "A P P"), "tech", 600.0, 0.65, 0.62),
    _T("ADBE", "Adobe", ("Adobe", "A dobie", "A D B E"), "tech", 350.0, 0.32, 0.31),
    _T("ORCL", "Oracle", ("Oracle", "O R C L"), "tech", 280.0, 0.40, 0.38),
    _T("AMAT", "Applied Materials", ("Applied Materials", "A M A T"), "semis", 200.0, 0.42, 0.40),
    _T("APLD", "Applied Digital", ("Applied Digital", "A P L D"), "tech", 25.0, 0.95, 0.90),
    _T("SMCI", "Super Micro Computer", ("Super Micro", "Supermicro", "S M C I"), "tech", 45.0, 0.75, 0.72),
    _T("SLB", "SLB", ("Schlumberger", "S L B"), "energy", 34.0, 0.32, 0.31),
    _T("RIOT", "Riot Platforms", ("Riot", "Riot Platforms", "R I O T"), "financials", 18.0, 0.85, 0.82),
    _T("CRM", "Salesforce", ("Salesforce", "Sales Force", "C R M"), "tech", 245.0, 0.32, 0.31),
    _T("PYPL", "PayPal", ("PayPal", "Pay Pal", "P Y P L"), "financials", 68.0, 0.35, 0.34),
    _T("COP", "ConocoPhillips", ("Conoco", "ConocoPhillips", "C O P"), "energy", 92.0, 0.28, 0.27),
    _T("XYZ", "Block", ("Block", "Square", "X Y Z"), "financials", 75.0, 0.50, 0.48),
    _T("SMH", "VanEck Semiconductor ETF", ("S M H", "semiconductor ETF", "semis ETF"), "etf", 330.0, 0.30, 0.29),
    _T("VIX", "Cboe Volatility Index", ("the VIX", "VIX", "V I X", "volatility index"), "index", 17.0, 1.00, 1.00,
       weeklies=False, options=False, quote_only=True),
)

UNIVERSE: tuple[Ticker, ...] = CORE + EXTRA

BY_SYMBOL: dict[str, Ticker] = {t.symbol: t for t in UNIVERSE}

LOOKALIKES: dict[str, tuple[str, ...]] = {
    "OR": ("ORLY",), "ORLY": ("OR",),
    "META": ("MET",), "MET": ("META",),
    "NVDA": ("NVDL",), "NVDL": ("NVDA",),
    "MA": ("MU",), "MU": ("MA",),
    "COST": ("CAT",), "CAT": ("COST",),
}

INDEX_SYMBOL = "SPY"

for _t in UNIVERSE:
    assert not any(c.isdigit() for c in _t.symbol + _t.name + "".join(_t.spoken)), _t
