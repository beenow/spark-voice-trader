"""Picks the agent the server talks to: the real `agent.loop.run_turn` or voice.fake_agent.

mode: "fake" | "real" | "auto" (real when `agent.loop` imports, else fake).
The real agent's world/backend constructors are looked up lazily and tolerantly
because agent/ and mockmarket/ are built in parallel; set VOICE_SEED,
VOICE_GLIMMER_MODEL and VOICE_ADAPTER to steer them.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("voice.agent")

from agent import paths

MODELS = paths.MODELS_DIR
GLIMMER_DIR = os.environ.get("VOICE_GLIMMER_MODEL", str(paths.GLIMMER_DIR))
ADAPTER_DIR = os.environ.get("VOICE_ADAPTER", str(paths.ADAPTERS_DIR / "latest"))


def real_available() -> bool:
    try:
        importlib.import_module("agent.loop")
        return True
    except Exception:
        return False


def _first_attr(modules: list[str], names: list[str]):
    for m in modules:
        try:
            mod = importlib.import_module(m)
        except Exception:
            continue
        for n in names:
            if hasattr(mod, n):
                return getattr(mod, n)
    return None


class AgentRunner:
    def __init__(self, mode: str = "auto"):
        if mode == "auto":
            mode = "real" if real_available() else "fake"
        self.mode = mode
        self._backends: dict[str, Any] = {}
        self._world = None
        if mode == "real":
            from agent.loop import run_turn  # noqa: F401  (fail fast if broken)
            self._run_turn = run_turn
        elif mode == "fake":
            from voice.fake_agent import run_turn
            self._run_turn = run_turn
        else:
            raise ValueError(f"unknown agent mode {mode!r}")
        log.info("agent mode: %s", self.mode)

    # -- real-agent plumbing -------------------------------------------------
    def _world_obj(self):
        if self._world is None and self.mode == "real":
            World = _first_attr(["mockmarket", "mockmarket.world"], ["World"])
            if World is None:
                raise RuntimeError("mockmarket.World not importable")
            self._world = World(seed=int(os.environ.get("VOICE_SEED", "7")), scenario=os.environ.get("VOICE_SCENARIO") or None)
        return self._world

    # -- named models (agent.backends.registry; aliases base/trained) ----------
    def models(self) -> list[dict]:
        """[{name, label, family, available}] for the picker. Fake mode lists the same names (all usable)."""
        try:
            from agent.backends import list_models
            out = list_models()
        except Exception:  # noqa: BLE001
            out = [{"name": n, "label": n, "family": "glimmer", "available": True} for n in ("base", "trained")]
        if self.mode != "real":
            out = [dict(m, available=True) for m in out]
        return out

    def default_model(self) -> str:
        env = os.environ.get("VOICE_DEFAULT_MODEL")
        try:
            from agent.backends import registry
            reg = registry()
            names = {m["name"]: m for m in reg["models"]}
            if env and env in names:
                return env
            d = reg.get("default")
            if d and (names[d]["available"] or self.mode != "real"):
                return d
            return next((n for n, m in names.items() if m["available"]), d or "base")
        except Exception:  # noqa: BLE001
            return env or "base"

    def resolve(self, model: str) -> str:
        """Canonical name for `model` (KeyError when unknown, RuntimeError when not on disk)."""
        try:
            from agent.backends import resolve_model
            name = resolve_model(model)
        except KeyError:
            raise
        except Exception:  # noqa: BLE001
            if model in ("base", "trained"):
                return model
            raise KeyError(model)
        if self.mode == "real":
            spec = next((m for m in self.models() if m["name"] == name), None)
            if spec is not None and not spec["available"]:
                raise RuntimeError(f"model {name} is not available")
        return name

    def warm_all(self, first: str | None = None) -> dict[str, str]:
        """Load + warm every available model (first the default), so switching and compare are instant."""
        import time
        out: dict[str, str] = {}
        if self.mode != "real":
            return out
        names = [m["name"] for m in self.models() if m["available"]]
        if first in names:
            names.remove(first)
            names.insert(0, first)
        for n in names:
            t0 = time.perf_counter()
            try:
                self._backend(n)
                out[n] = f"ok {time.perf_counter() - t0:.1f}s"
            except Exception as e:  # noqa: BLE001
                log.exception("warm %s failed", n)
                out[n] = f"failed: {e}"
        return out

    def _backend(self, model: str):
        if self.mode != "real":
            return None
        try:
            model = self.resolve(model)
        except KeyError:
            pass
        if model not in self._backends:
            # Prefer a factory if agent/ provides one; else construct GlimmerBackend directly.
            factory = _first_attr(["agent.backends", "agent.backend", "agent.models", "agent"], ["make_backend"])
            if factory is not None:
                self._backends[model] = factory(model)
            else:
                Glimmer = _first_attr(["agent.backends", "agent.backend", "agent.models", "agent.glimmer", "agent"],
                                      ["GlimmerBackend"])
                if Glimmer is None:
                    raise RuntimeError("agent GlimmerBackend not importable")
                adapter = ADAPTER_DIR if model == "trained" and Path(ADAPTER_DIR).exists() else None
                if model == "trained" and adapter is None:
                    log.warning("trained adapter %s missing; using base weights", ADAPTER_DIR)
                # Share one base model across base/trained when the backend allows it.
                self._backends[model] = Glimmer(GLIMMER_DIR, adapter=adapter)
        return self._backends[model]

    def warm(self, model: str = "base") -> float:
        """Load + warm the real backend (prefix cache, kernels) so the first question is fast."""
        import time
        t0 = time.perf_counter()
        if self.mode == "real":
            self._world_obj()
            self._backend(model)
        return time.perf_counter() - t0

    def vocab(self) -> list[str] | None:
        """Ticker + spoken names from mockmarket, if it exposes them."""
        fn = _first_attr(["mockmarket", "mockmarket.universe", "tools"], ["spoken_vocab", "stt_vocab"])
        if fn is None:
            return None
        try:
            return list(fn())
        except Exception:
            return None

    def world(self):
        """The real agent's world (None in fake mode)."""
        return self._world_obj() if self.mode == "real" else None

    def run(self, question: str, history: list, model: str, on_event: Callable[[dict], None],
            facts: list[str] | None = None) -> Any:
        kwargs: dict[str, Any] = dict(history=history, on_event=on_event, facts=facts)
        if self.mode == "real":
            kwargs.update(world=self._world_obj(), backend=self._backend(model))
        params = inspect.signature(self._run_turn).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in params}
        return self._run_turn(question, **kwargs)
