"""Run the synthetic eval sample through the agent loop and score each turn.

    python -m eval.run_eval                          # oracle backend: replays the gold (checks the harness itself)
    python -m eval.run_eval --model llama-1b-loop    # a named model from the registry ($VOICE_MODELS)
    python -m eval.run_eval --rows my_rows.json --out results.json

Each row runs through agent.memory.Conversation + agent.loop.run_turn on mockmarket.World(seed), the same path
the voice server uses. `context` questions (follow-ups) run first and are not scored.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from agent.backends import ScriptedBackend, make_backend
from agent.loop import run_turn
from agent.memory import Conversation
from mockmarket import World

from .score import CHECKS, score_turn, summarize

SAMPLE = Path(__file__).with_name("sample.json")


def oracle(gold: list[dict]) -> ScriptedBackend:
    """Calls the gold tools in one parallel step, then answers without numbers."""
    if not gold:
        return ScriptedBackend([{"answer": "I can help with quotes, positions, charts, options and order tickets."}])
    return ScriptedBackend([{"tool_calls": [{"tool": g["tool"], "args": dict(g.get("args") or {})} for g in gold]},
                            {"answer": "Here is what I found."}])


def run_rows(rows: list[dict], seed: int, model: str | None = None) -> list[dict]:
    backend = make_backend(model) if model else None
    out = []
    for row in rows:
        world = World(seed)
        conv = Conversation()
        for q in row.get("context") or []:
            be = backend or ScriptedBackend([{"answer": "Okay."}])
            conv.ask(q, lambda qq, h, f: run_turn(qq, h, world, be, facts=f), world=world)
        be = backend or oracle(row["gold"])
        t0 = time.perf_counter()
        res = conv.ask(row["question"], lambda qq, h, f: run_turn(qq, h, world, be, facts=f), world=world)
        dt = time.perf_counter() - t0
        calls = [{"tool": c["tool"], "args": c.get("args") or {}} for c in (res.tool_calls or [])]
        checks = score_turn(row["gold"], calls, res.answer_template, res.slots or {})
        out.append({"id": row["id"], "category": row["category"], "question": row["question"], "calls": calls,
                    "answer": res.spoken, "checks": checks, "seconds": round(dt, 3)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default=str(SAMPLE))
    ap.add_argument("--model", default=None, help="registry model name; omit for the oracle backend")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    data = json.loads(Path(a.rows).read_text())
    results = run_rows(data["rows"], int(data.get("seed", 7)), a.model)
    summ = summarize(results)
    secs = [r["seconds"] for r in results]
    summ["latency_s"] = {"p50": round(statistics.median(secs), 3), "max": round(max(secs), 3)}
    print(json.dumps({"model": a.model or "oracle", **summ}, indent=1))
    for r in results:
        bad = [k for k in CHECKS if not r["checks"][k]]
        if bad:
            print(f"FAIL {r['id']} {bad}: {r['question']!r} -> {r['calls']}")
    if a.out:
        Path(a.out).write_text(json.dumps({"model": a.model or "oracle", "summary": summ, "rows": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
