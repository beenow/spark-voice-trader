"""ToolResult + model_view (the only thing the model sees from a tool)."""
from __future__ import annotations

import json
from typing import Any


class ToolResult(dict):
    """JSON-serialisable dict: {tool, ok, words, slots, error}."""

    def __init__(self, tool: str, ok: bool = True, words: dict | None = None, slots: dict | None = None,
                 error: str | None = None):
        super().__init__(tool=tool, ok=ok, words=dict(words or {}), slots=dict(slots or {}), error=error)

    @classmethod
    def fail(cls, tool: str, error: str, words: dict | None = None) -> "ToolResult":
        return cls(tool, ok=False, words=words, slots={}, error=error)

    tool = property(lambda self: self["tool"])
    ok = property(lambda self: self["ok"])
    words = property(lambda self: self["words"])
    slots = property(lambda self: self["slots"])
    error = property(lambda self: self["error"])


def model_view(result: dict, names: dict[str, str] | None = None) -> dict[str, Any]:
    """What the model sees. `names` maps raw slot name -> per-turn unique slot name."""
    names = names or {}
    return {
        "tool": result["tool"],
        "ok": result["ok"],
        "words": result["words"],
        "slots": ["{" + names.get(s, s) + "}" for s in result["slots"]],
        "error": result["error"],
    }


def model_view_json(result: dict, names: dict[str, str] | None = None) -> str:
    return json.dumps(model_view(result, names), ensure_ascii=False)
