"""Tools over mockmarket: schemas, implementations, slots, rendering."""
from __future__ import annotations

from . import render
from .impl import TOOLS
from .result import ToolResult, model_view, model_view_json
from .schema import TOOL_SCHEMAS, schema
from .slots import SLOT_TYPES, kind_of


def run_tool(world, name: str, args: dict) -> ToolResult:
    """Run a tool with already-validated args. Never raises."""
    fn = TOOLS.get(name)
    if fn is None:
        return ToolResult.fail(name, "unknown tool")
    try:
        return fn(world, **args)
    except Exception:  # noqa: BLE001 - tool errors become words, not crashes
        return ToolResult.fail(name, "tool failed")


__all__ = ["TOOLS", "TOOL_SCHEMAS", "SLOT_TYPES", "ToolResult", "kind_of", "model_view", "model_view_json",
           "render", "run_tool", "schema"]
