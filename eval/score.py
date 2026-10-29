"""Per-check scoring of one agent turn against a gold list of tool calls. Each check is reported on its own,
never blended.

tool_set    the multiset of tools called equals the gold's (any order)
arguments   every gold call was made with at least the gold's arguments
step_count  as many calls as the gold
max_steps   no more than 4 calls
contract    the answer template uses only slots this turn's results offered, and has no digits outside slots
action      tools were called if and only if the gold has calls
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

CHECKS = ("tool_set", "arguments", "step_count", "max_steps", "contract", "action")
_SLOT = re.compile(r"\{([a-z_0-9]+)\}")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) < 1e-9
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def covers(made: dict, want: dict) -> bool:
    """`made` has every key of `want` with the same value (strings case-insensitive, numbers numerically)."""
    return all(k in made and _same(made[k], v) for k, v in (want or {}).items())


def score_turn(gold: list[dict], calls: list[dict], template: str, slots: set[str] | dict) -> dict[str, bool]:
    made = [{"tool": c["tool"], "args": dict(c.get("args") or {})} for c in calls]
    unused, args_ok = list(made), True
    for g in gold:
        hit = next((c for c in unused if c["tool"] == g["tool"] and covers(c["args"], g.get("args") or {})), None)
        if hit is None:
            args_ok = False
        else:
            unused.remove(hit)
    used = set(_SLOT.findall(template or ""))
    contract = used <= set(slots) and not re.search(r"\d", _SLOT.sub("", template or ""))
    return {
        "tool_set": Counter(c["tool"] for c in made) == Counter(g["tool"] for g in gold),
        "arguments": args_ok,
        "step_count": len(made) == len(gold),
        "max_steps": len(made) <= 4,
        "contract": contract,
        "action": bool(made) == bool(gold),
    }


def summarize(results: list[dict]) -> dict:
    """{"overall": {check: pass_rate}, "by_category": {cat: {check: pass_rate}}, "n": rows}."""
    def rates(rows):
        return {k: round(100.0 * sum(r["checks"][k] for r in rows) / len(rows), 1) for k in CHECKS} if rows else {}
    cats = sorted({r["category"] for r in results})
    return {"n": len(results), "overall": rates(results),
            "by_category": {c: dict(n=sum(r["category"] == c for r in results),
                                    **rates([r for r in results if r["category"] == c])) for c in cats}}
