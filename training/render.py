"""Render teacher traces (trace_schema.json) into Llama 3.2 training examples.

    python -m training.render traces.jsonl --out data/ [--tokenizer $VOICE_LLAMA_MODEL] [--valid-frac 0.05]

Writes data/train.jsonl and data/valid.jsonl, one {"tokens": [...], "offset": k} per assistant decision. Only
tokens[offset:] (the model's own call lines or answer, plus <|eot_id|>) are trained. The prompt is rendered by
agent.format.render_llama, the same function inference uses, so training and serving are byte-identical
(tests/test_format_parity.py). Paraphrases of one question share a `group` and land in the same split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agent import paths
from agent.format import grouped_trace_messages, llama_examples
from agent.loop import SYSTEM_PROMPT
from agent.memory import STATE_KEYS
from tools import schema


def prefix_messages(tr: dict) -> list[dict]:
    """system + up to 3 earlier turns (question, spoken answer) + the question with its state line."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    ctx = (tr.get("context") or [])[-3:]
    for c in ctx:
        msgs += [{"role": "user", "content": c["question"]}, {"role": "assistant", "content": c["spoken"]}]
    q = tr["question"]
    st = tr.get("state") or {}
    if ctx and st:
        q = "[state: " + ", ".join(f"{k}={st[k]}" for k in STATE_KEYS if st.get(k)) + "]\n" + q
    msgs.append({"role": "user", "content": q})
    return msgs


def step_groups(steps: list[dict]) -> list[list[dict]]:
    """Steps sharing a `turn` were one teacher decision (parallel calls)."""
    groups: list[list[dict]] = []
    last = object()
    for st in steps:
        item = {"tool": st["call"]["tool"], "args": dict(st["call"].get("arguments") or {}),
                "view": dict(st.get("result") or {})}
        turn = st.get("turn")
        if groups and turn is not None and turn == last:
            groups[-1].append(item)
        else:
            groups.append([item])
        last = turn
    return groups


def examples_for(tr: dict, tokenizer) -> list[tuple[list[int], int]]:
    msgs = grouped_trace_messages(prefix_messages(tr), step_groups(tr.get("steps") or []), tr["answer"])
    return llama_examples(msgs, tokenizer, schema())


def is_valid(group: str, frac: float) -> bool:
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=str(paths.LLAMA_DIR))
    ap.add_argument("--valid-frac", type=float, default=0.05)
    a = ap.parse_args(argv)
    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(Path(a.tokenizer))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = {"train": 0, "valid": 0}
    with (out / "train.jsonl").open("w") as ft, (out / "valid.jsonl").open("w") as fv:
        for line in open(a.traces):
            if not line.strip():
                continue
            tr = json.loads(line)
            split = "valid" if is_valid(tr.get("group") or tr["id"], a.valid_frac) else "train"
            for tokens, offset in examples_for(tr, tok):
                (fv if split == "valid" else ft).write(json.dumps({"tokens": tokens, "offset": offset}) + "\n")
                n[split] += 1
    print(json.dumps(n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
