from .universe import BY_SYMBOL, LOOKALIKES, UNIVERSE, Ticker
from .world import SCENARIOS, TODAY, World

__all__ = ["World", "SCENARIOS", "TODAY", "UNIVERSE", "BY_SYMBOL", "LOOKALIKES", "Ticker"]


def spoken_vocab() -> list[str]:
    """Company names, then tickers, then other spoken aliases, used to bias speech recognition. Ordered so a
    length-capped prompt (voice.stt.build_prompt) keeps every company name first."""
    names = sorted(UNIVERSE, key=lambda t: t.sector in ("etf", "index"))     # long fund names last
    out = [t.name for t in names] + [t.symbol for t in UNIVERSE]
    for t in UNIVERSE:
        out += list(t.spoken)
    return list(dict.fromkeys(out))
