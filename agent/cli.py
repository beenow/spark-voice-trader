"""python -m agent.cli --seed 7 --backend scripted|named|glimmer "How's my NVDA position and is it near support?"

  --backend named --name llama-1b-loop   a model from the registry ($VOICE_MODELS, see models.example.json)

Several questions in one invocation run as consecutive turns through agent.memory.Conversation, the same
path the voice server uses (history, frames, memory commands, pinned facts; nothing is written to disk).
"""
from __future__ import annotations

import argparse
import json
import time

from mockmarket import World

from .backends import DEFAULT_GLIMMER, GlimmerBackend, ScriptedBackend, make_backend
from .loop import run_turn
from .memory import Conversation


def demo_script() -> ScriptedBackend:
    """Canned run for the default question (position + levels)."""
    def first(messages):
        q = messages[-1]["content"]
        import re
        m = re.search(r"\b([A-Z]{1,5})\b", q)
        sym = m.group(1) if m else "NVDA"
        return {"tool_calls": [{"tool": "get_position", "args": {"symbol": sym}},
                               {"tool": "get_levels", "args": {"symbol": sym}}]}
    return ScriptedBackend([first, {"answer": "You're {pnl} on the position, {pnl_pct}. Support is {support}, "
                                              "{dist_support} below the price."}])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="+")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--backend", choices=["glimmer", "named", "scripted"], default="scripted")
    ap.add_argument("--name", default="llama-1b-loop", help="registry model name for --backend named")
    ap.add_argument("--json", action="store_true", help="print one JSON line per turn (calls, spoken, timings)")
    ap.add_argument("--model", default=str(DEFAULT_GLIMMER))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--mode", choices=["base", "trained"], default=None,
                    help="decoding mode (default: trained when --adapter is given, else base)")
    ap.add_argument("--force-first-tool", action="store_true", help="base mode: step one must be a tool call")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="print raw model output")
    a = ap.parse_args(argv)

    world = World(a.seed, a.scenario)
    t0 = time.perf_counter()
    if a.backend == "glimmer":
        backend = GlimmerBackend(a.model, adapter=a.adapter, mode=a.mode, verbose=a.verbose,
                                 force_first_tool=a.force_first_tool or None)
        print(f"[load] {backend.load_s:.1f}s  model={a.model}")
    elif a.backend == "named":
        backend = make_backend(a.name)
        print(f"[load] model={a.name}")
    else:
        backend = None
    conv = Conversation()
    for q in a.questions:
        if a.backend == "scripted":
            backend = demo_script()
        print(f"\n=== Q: {q}")
        t_turn = time.perf_counter()
        n_stats = len(getattr(backend, "stats", []))

        def on_event(ev):
            if not a.json:
                print(f"[{(time.perf_counter() - t_turn) * 1000:7.0f} ms] {json.dumps(ev, default=str)}", flush=True)

        res = conv.ask(q, lambda qq, h, f: run_turn(qq, h, world, backend, max_steps=a.max_steps, on_event=on_event,
                                                    facts=f), on_event=on_event, world=world)
        if a.json:
            print(json.dumps({"q": q, "calls": [{"tool": c.get("tool"), "args": c.get("args")}
                                                for c in (res.tool_calls or [])],
                              "spoken": res.spoken, "fallback": res.fallback,
                              "total_s": round(res.timings.get("total") or 0, 3)}, default=str), flush=True)
            continue
        print(f"--- display: {res.display}")
        print(f"--- spoken:  {res.spoken}")
        print(f"--- timings: total={res.timings.get('total') or 0:.2f}s decide={[round(x, 2) for x in res.timings.get('decide') or []]}"
              f" filler_at={res.timings.get('filler_at') and round(res.timings['filler_at'], 2)}")
        for i, st in enumerate(getattr(backend, "stats", [])[n_stats:], 1):
            print(f"--- step {i}: {json.dumps(st)}")
    print(f"\n[done] wall {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
