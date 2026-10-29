"""Where local weights, adapters and state live. Everything is overridable from the environment.

VOICE_HOME        base directory for local state (default ~/.leaps-voice-agent)
VOICE_MODELS_DIR  model weights (default $VOICE_HOME/models): Muse-Glimmer-30B-4bit, whisper-large-v3-turbo,
                  Kokoro-82M, Llama-3.2-1B-Instruct-bf16
VOICE_MODELS      the named-model registry JSON (default $VOICE_HOME/models.json; see models.example.json)
VOICE_MEMORY_DIR  conversation store (default $VOICE_HOME/sessions; empty = memory off)
"""
from __future__ import annotations

import os
from pathlib import Path


def _p(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    return Path(os.path.expanduser(v)) if v else default


HOME = _p("VOICE_HOME", Path.home() / ".leaps-voice-agent")
MODELS_DIR = _p("VOICE_MODELS_DIR", HOME / "models")
ADAPTERS_DIR = _p("VOICE_ADAPTERS_DIR", HOME / "adapters")
REGISTRY = _p("VOICE_MODELS", HOME / "models.json")
SESSIONS_DIR = HOME / "sessions"
GLIMMER_DIR = MODELS_DIR / "Muse-Glimmer-30B-4bit"
LLAMA_DIR = _p("VOICE_LLAMA_MODEL", MODELS_DIR / "Llama-3.2-1B-Instruct-bf16")
