"""Whisper speech-to-text (mlx-whisper), loaded once and kept warm.

Input: 16 kHz mono PCM16 bytes. Output: transcript text.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from agent import paths

MODELS = paths.MODELS_DIR
DEFAULT_MODEL = Path(os.environ.get("VOICE_WHISPER_MODEL", MODELS / "whisper-large-v3-turbo"))
SAMPLE_RATE = 16000

# Fallback vocabulary when the caller passes none: biases Whisper toward tickers
# and company names it would otherwise mangle ("en vidia", "O'Reilly", "Palantir").
BUILTIN_VOCAB = [
    "NVDA", "Nvidia", "AAPL", "Apple", "TSLA", "Tesla", "MSFT", "Microsoft",
    "AMZN", "Amazon", "META", "Meta", "GOOGL", "Alphabet", "AMD", "NFLX", "Netflix",
    "PLTR", "Palantir", "AVGO", "Broadcom", "ORLY", "O'Reilly", "COIN", "Coinbase",
    "SPY", "QQQ", "SMCI", "Super Micro", "UBER", "Uber", "JPM", "JPMorgan",
    "RSI", "EMA", "P&L", "calls", "puts", "strike", "expiry", "support", "resistance",
]

MIN_SECONDS = 0.3      # shorter than this: treat as an accidental tap
MIN_RMS = 1e-3         # quieter than this: silence (Whisper hallucinates "Thank you." on it)


# Trading words go first: they are what the agent acts on ("buying" misheard as "uploading", 2026-10-27).
TRADING_TERMS = [
    # "buy 2 ... 650 strike": a digit example keeps Whisper writing quantities/strikes as digits (without it,
    # the word list nudged it to "ten shares", "three hundred calls")
    "buy 2", "buying", "sell 10", "selling", "300 calls", "650 call", "put", "puts", "strike", "expiry", "option chain",
    "open a ticket", "draft an order", "limit order", "shares", "position", "portfolio", "P&L",
    "RSI", "EMA", "support", "resistance", "IV", "delta",
]
_SPELLED = re.compile(r"^(?:[A-Za-z] )+[A-Za-z]$")
PROMPT_CHARS = 750          # ~215 Whisper tokens: inside the 223-token prompt window, so nothing is cut


def build_prompt(vocab: list[str] | None) -> str:
    """Trading terms, then tickers / company names (spelled-out "N V D A" forms and lower-case aliases skipped:
    they add length, not recognition), capped at PROMPT_CHARS."""
    words = list(TRADING_TERMS)
    for w in vocab or BUILTIN_VOCAB:
        w = w.strip()
        if w and not _SPELLED.match(w) and not w.islower() and w not in words:
            words.append(w)
    head = "A trader asks about stocks and options: "
    out = []
    n = len(head)
    for w in words:
        if n + len(w) + 2 > PROMPT_CHARS:
            break
        out.append(w)
        n += len(w) + 2
    return head + ", ".join(out) + "."


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:
        pcm = pcm[:-1]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class STT:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL, vocab: list[str] | None = None):
        self.model_dir = str(model_dir)
        self.vocab = vocab
        self.loaded = False
        self._lock = threading.Lock()

    def warm(self) -> float:
        """Load weights and run one dummy decode so the first real turn is fast."""
        t = time.perf_counter()
        import mlx_whisper  # noqa: F401  (heavy import, deferred)
        with self._lock:
            self._run(np.zeros(SAMPLE_RATE, dtype=np.float32), None)
            self.loaded = True
        return time.perf_counter() - t

    def _run(self, audio: np.ndarray, prompt: str | None) -> str:
        import mlx_whisper
        out = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_dir,
            language="en",
            task="transcribe",
            initial_prompt=prompt,
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4),
            no_speech_threshold=0.6,
            verbose=None,
        )
        return (out.get("text") or "").strip()

    def transcribe(self, pcm16: bytes, vocab: list[str] | None = None) -> str:
        audio = pcm16_to_float(pcm16)
        if len(audio) < MIN_SECONDS * SAMPLE_RATE:
            return ""
        if float(np.sqrt(np.mean(audio ** 2))) < MIN_RMS:
            return ""
        with self._lock:
            text = self._run(audio, build_prompt(vocab or self.vocab))
            self.loaded = True
        # Whisper sometimes echoes the prompt back on near-silence.
        if text.startswith("A trader asks about"):
            return ""
        return text
