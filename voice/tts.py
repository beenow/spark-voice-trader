"""Kokoro-82M text-to-speech via mlx-audio, loaded once.

`stream(text)` yields 24 kHz mono PCM16 bytes, one chunk per sentence, so the
first sentence can play while later ones synthesise.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from agent import paths

MODELS = paths.MODELS_DIR
DEFAULT_MODEL = Path(os.environ.get("VOICE_KOKORO_MODEL", MODELS / "Kokoro-82M"))
DEFAULT_VOICE = os.environ.get("VOICE_TTS_VOICE", "af_heart")
SAMPLE_RATE = 24000

_SENT_RE = re.compile(r"(?<=[.!?;])\s+")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    # Merge very short fragments ("OK.") into the next sentence: fewer, smoother chunks.
    out: list[str] = []
    for p in parts:
        if out and len(out[-1]) < 12:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def float_to_pcm16(audio) -> bytes:
    a = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (a * 32767.0).astype("<i2").tobytes()


class TTS:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL, voice: str = DEFAULT_VOICE, speed: float = 1.05):
        self.model_dir = Path(model_dir)
        self.voice = voice
        self.speed = speed
        self.model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _load(self):
        if self.model is None:
            from mlx_audio.tts.utils import load_model
            self.model = load_model(self.model_dir)
        return self.model

    def warm(self) -> float:
        t = time.perf_counter()
        self.synth("Ready.")
        return time.perf_counter() - t

    def synth(self, sentence: str) -> bytes:
        """Synthesise one sentence to PCM16 bytes."""
        with self._lock:
            model = self._load()
            chunks = []
            for res in model.generate(text=sentence, voice=self.voice, speed=self.speed,
                                      lang_code="a", split_pattern=r"\n+"):
                chunks.append(np.asarray(res.audio, dtype=np.float32).reshape(-1))
        if not chunks:
            return b""
        return float_to_pcm16(np.concatenate(chunks))

    def stream(self, text: str) -> Iterator[bytes]:
        for s in split_sentences(text):
            pcm = self.synth(s)
            if pcm:
                yield pcm
