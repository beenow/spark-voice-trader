"""Automatic trace filters: a teacher trace is kept only if every check passes.

    python -m training.filters traces.jsonl [--gold gold.json] > kept.jsonl

Checks (each reported on its own):
  schema     every call names a real tool and its arguments pass agent.validator.validate_call
  max_steps  at most 4 calls
  contract   the answer uses only slots its results offered and has no digits outside slots
  no_send    no step claims an order was sent (draft_order only ever drafts a ticket)
  gold       (with --gold {id: [{"tool","args"}]}) tool set, arguments, step count and action match the gold
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

from eval.score import score_turn

_SLOT = re.compile(r"\{([a-z_0-9]+)\}")


def _schema_ok(steps, world) -> bool:
    from agent.validator import validate_call
    for st in steps:
        call = st["call"]
        try:
            chk = validate_call(world, call["tool"], dict(call.get("arguments") or {}))
        except Exception:  # noqa: BLE001
            return False
        if not getattr(chk, "ok", False):
            return False
    return True


def check_trace(tr: dict, gold: list[dict] | None = None, world=None) -> dict[str, bool]:
    steps = tr.get("steps") or []
    offered = {s.strip("{}") for st in steps for s in (st.get("result") or {}).get("slots") or []}
    answer = tr.get("answer") or ""
    out = {
        "schema": _schema_ok(steps, world),
        "max_steps": len(steps) <= 4,
        "contract": set(_SLOT.findall(answer)) <= offered and not re.search(r"\d", _SLOT.sub("", answer)),
        "no_send": not any(re.search(r"\b(sent|placed|submitted|executed)\b", json.dumps(st.get("result") or {}), re.I)
                           for st in steps if st["call"]["tool"] == "draft_order"),
    }
    if gold is not None:
        calls = [{"tool": st["call"]["tool"], "args": st["call"].get("arguments") or {}} for st in steps]
        g = score_turn(gold, calls, answer, offered)
        out["gold"] = all(g[k] for k in ("tool_set", "arguments", "step_count", "action"))
    return out


def main(argv=None) -> int:
    from mockmarket import World
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces")
    ap.add_argument("--gold", default=None)
    a = ap.parse_args(argv)
    gold = json.load(open(a.gold)) if a.gold else None
    fails: Counter = Counter()
    kept = total = 0
    for line in open(a.traces):
        if not line.strip():
            continue
        tr = json.loads(line)
        total += 1
        res = check_trace(tr, gold.get(tr["id"]) if gold else None, World(int(tr.get("seed", 7))))
        bad = [k for k, v in res.items() if not v]
        fails.update(bad)
        if not bad:
            kept += 1
            sys.stdout.write(json.dumps(tr, ensure_ascii=False) + "\n")
    print(json.dumps({"traces": total, "kept": kept, "failed_by_check": dict(fails)}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
