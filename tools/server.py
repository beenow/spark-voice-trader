"""HTTP tool executor for teacher-trace generation and live evals (any trainer can call it).

    POST /call {tool, arguments, row_id?, seed?, scenario?}
      -> {"result": model_view(...), "raw": {"slots": {...}, "words": {...}, "spoken": {...}}}

`result` is exactly what the model sees (`tools.model_view`: words + slot NAMES,
never a digit); `raw` carries the slot values for logging and rendering only.

Deterministic: the world is `mockmarket.World(seed, scenario)`. `seed` defaults
to a stable hash of `row_id`, so every call for one row - across steps, retries
and re-runs - sees the same market. Worlds are cached per (seed, scenario), so a
row's later calls see what its earlier ones did (a drafted ticket, say).

Run:  python -m uvicorn tools.server:app --host 127.0.0.1 --port 8040
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mockmarket import SCENARIOS, World
from tools import TOOLS, model_view, run_tool
from tools.render import spoken

PORT = 8040
_WORLD_CACHE_SIZE = 64


class CallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    row_id: Optional[str] = None
    seed: Optional[int] = None
    scenario: Optional[str] = None


def seed_for(row_id: Optional[str], seed: Optional[int]) -> int:
    """The explicit seed, else a stable 31-bit hash of the row id, else 7."""
    if seed is not None:
        return int(seed)
    if row_id:
        return int(hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
    return 7


def _plain(value: Any) -> Any:
    """JSON-safe copy (numpy scalars, dates -> plain values)."""
    return json.loads(
        json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    )


def create_app() -> FastAPI:
    app = FastAPI(title="leaps-voice-agent tools")
    worlds: "OrderedDict[tuple[int, Optional[str]], World]" = OrderedDict()
    lock = threading.Lock()

    def world_for(seed: int, scenario: Optional[str]) -> World:
        key = (seed, scenario)
        with lock:
            world = worlds.get(key)
            if world is None:
                world = World(seed=seed, scenario=scenario)
                worlds[key] = world
                while len(worlds) > _WORLD_CACHE_SIZE:
                    worlds.popitem(last=False)
            else:
                worlds.move_to_end(key)
            return world

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "tools": sorted(TOOLS)}

    @app.post("/call")
    def call(body: CallRequest) -> dict:
        if body.scenario is not None and body.scenario not in SCENARIOS:
            raise HTTPException(422, f"unknown scenario {body.scenario!r}")
        world = world_for(seed_for(body.row_id, body.seed), body.scenario)
        with lock:  # a World is not thread-safe (rng, tickets)
            result = run_tool(world, body.tool, dict(body.arguments))
        return {
            "result": _plain(model_view(result)),
            "raw": {
                "slots": _plain(result["slots"]),
                "words": _plain(result["words"]),
                # Each slot as it is SPOKEN (no digits): what a prior turn's
                # answer sounds like when it comes back as history.
                "spoken": {name: spoken(name, value, body.tool) for name, value in result["slots"].items()},
            },
        }

    return app


app = create_app()
