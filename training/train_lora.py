"""LoRA-train Llama 3.2 1B on the Llama-format voice-agent examples (training/render.py) with mlx_lm.

  python -m training.train_lora training/lora_config.yaml [--dry-run]

Llama 3.2's own chat template adds a dated system line, puts the full JSON tool schema in the user turn
(~2k tokens) and refuses parallel calls, so it cannot express the compact, parallel-call format a 1B model
needs. Here the examples are pre-tokenised by agent.format.render_llama (the same renderer inference uses) and
trained with mlx_lm's own trainer; the mask is one prompt offset per example.

Writes <adapter_path>/{adapters.safetensors (best val), last_adapters.safetensors, adapter_config.json,
progress.jsonl}. --dry-run: load data + config, print the plan (iters, s/iter estimate), no model.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import os

import yaml


class Examples:
    def __init__(self, path: Path):
        self.items = [(e["tokens"], e["offset"]) for e in map(json.loads, path.open())]

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)


def plan(cfg, train, valid):
    per_epoch = len(train) // cfg["batch_size"]
    iters = cfg["epochs"] * per_epoch
    opt_steps = iters // cfg["grad_accumulation_steps"]
    toks = sum(len(t) for t, _ in train.items) * cfg["epochs"]
    # 1.24B params, frozen base: ~4N FLOPs/token (forward + activation grads; no checkpoint recompute)
    # at ~12 TFLOPS effective bf16 on the M4 Max -> ~2,400 tokens/s; assume 1,500-3,000.
    est = {f"{r} tok/s": round(toks / r / 3600, 2) for r in (1500, 2400, 3000)}
    return {"train_examples": len(train), "valid_examples": len(valid), "iters": iters, "per_epoch": per_epoch,
            "optimizer_steps": opt_steps, "train_tokens_total": toks,
            "s_per_iter_at_2400": round(toks / iters / 2400, 2), "hours": est}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="GPU smoke: this many iters at batch 1 into --out")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(os.path.expandvars(Path(a.config).read_text()))
    cfg["model"] = os.path.expanduser(cfg["model"])
    if "$" in cfg["model"]:
        raise SystemExit(f"model path not set: {cfg['model']} (export VOICE_LLAMA_MODEL or edit the config)")
    if a.smoke:
        cfg.update(batch_size=1, grad_accumulation_steps=1, warmup_steps=1, evals_per_epoch=1, steps_per_report=1)
    if a.out:
        cfg["adapter_path"] = a.out
    data, out = Path(cfg["data"]), Path(cfg["adapter_path"])
    train, valid = Examples(data / "train.jsonl"), Examples(data / "valid.jsonl")
    if a.smoke:
        train.items, valid.items = train.items[:a.smoke], valid.items[:2]
    p = plan(cfg, train, valid)
    print(json.dumps(p, indent=1), flush=True)
    if a.dry_run:
        assert max(len(t) for t, _ in train.items + valid.items) <= cfg["max_seq_length"]
        assert all(0 < o < len(t) for t, o in train.items + valid.items)
        print("dry run ok", flush=True)
        return

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.trainer import TrainingArgs, train as mlx_train
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

    mx.random.seed(cfg["seed"])
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load(cfg["model"])
    model.freeze()
    linear_to_lora_layers(model, cfg["num_layers"], cfg["lora_parameters"])
    print_trainable_parameters(model)
    (out / "adapter_config.json").write_text(json.dumps({
        "model": cfg["model"], "fine_tune_type": "lora", "num_layers": cfg["num_layers"],
        "lora_parameters": cfg["lora_parameters"], "format": "llama-voice-v1 (agent/format.py render_llama)",
        "data": str(data), "config": cfg, "plan": p}, indent=1))

    iters, accum = p["iters"], cfg["grad_accumulation_steps"]
    steps = p["optimizer_steps"]
    warm = cfg["warmup_steps"]
    sched = optim.join_schedules(
        [optim.linear_schedule(cfg.get("warmup_init", cfg["learning_rate"] / 50), cfg["learning_rate"], warm),
         optim.cosine_decay(cfg["learning_rate"], max(1, steps - warm), cfg["min_learning_rate"])], [warm])
    opt = optim.Adam(learning_rate=sched)
    eval_every = max(1, p["per_epoch"] // cfg["evals_per_epoch"])
    prog = (out / "progress.jsonl").open("a")
    mean_len = sum(len(t) for t, _ in train.items) / len(train)
    t_start = time.perf_counter()
    best = {"val": math.inf, "iter": None}

    class CB(TrainingCallback):
        def on_train_loss_report(self, info):
            el = time.perf_counter() - t_start
            it = info["iteration"]
            rec = {"t": time.strftime("%H:%M:%S"), "iter": it, "of": iters, "epoch": round(it / p["per_epoch"], 2),
                   "train_loss": round(info["train_loss"], 4), "lr": info["learning_rate"],
                   "s_per_iter": round(1 / info["iterations_per_second"], 3),
                   "tok_s": round(info["iterations_per_second"] * cfg["batch_size"] * mean_len),
                   "peak_gb": round(info["peak_memory"], 2), "elapsed_min": round(el / 60, 1),
                   "eta_min": round(el / it * (iters - it) / 60, 1)}
            prog.write(json.dumps(rec) + "\n")
            prog.flush()
            print("[train]", json.dumps(rec), flush=True)

        def on_val_loss_report(self, info):
            it, vl = info["iteration"], info["val_loss"]
            rec = {"t": time.strftime("%H:%M:%S"), "iter": it, "val_loss": round(vl, 4), "best": vl < best["val"]}
            if vl < best["val"] and it > 0:
                best.update(val=vl, iter=it)
                mx.save_safetensors(str(out / "adapters.safetensors"),
                                    dict(tree_flatten(model.trainable_parameters())))
            prog.write(json.dumps(rec) + "\n")
            prog.flush()
            print("[val]", json.dumps(rec), flush=True)

    args = TrainingArgs(batch_size=cfg["batch_size"], iters=iters, val_batches=-1,
                        steps_per_report=cfg["steps_per_report"], steps_per_eval=eval_every,
                        steps_per_save=10 ** 9, max_seq_length=cfg["max_seq_length"],
                        adapter_file=str(out / "last_adapters.safetensors"),
                        grad_checkpoint=cfg["grad_checkpoint"], grad_accumulation_steps=accum)
    mlx_train(model, opt, train, valid, args=args, training_callback=CB())
    mx.save_safetensors(str(out / "last_adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    if not (out / "adapters.safetensors").exists():
        shutil.copy(out / "last_adapters.safetensors", out / "adapters.safetensors")
    done = {"done": True, "best_val": best["val"], "best_iter": best["iter"],
            "minutes": round((time.perf_counter() - t_start) / 60, 1)}
    prog.write(json.dumps(done) + "\n")
    prog.close()
    print("[done]", json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
