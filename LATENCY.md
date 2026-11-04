# Glimmer latency notes

Setup: Muse-Glimmer-30B-4bit with mlx_lm 0.32.2 on this Mac. Decoding runs at about 20-22 tokens/s and uses about 16 GB of memory at peak. Seed 7, run through `agent.loop.run_turn`.

| change | what it bought |
|---|---|
| Baseline (first smoke run, reasoning on, `reasoning_strength=low`) | "Position + support" took **44.2 s**. Steps took 15.3 s, 2.9 s, 11.1 s and 14.8 s. The first tool call came at 15.3 s. The model writes 50-300 tokens of private reasoning (`to=self`) on every step. |
| (b) Prefill and snapshot the system+tools prompt (~2,000 tokens) once at load (`warmup()`, ~10 s) and restore it every turn. Within a turn, keep the KV cache so only new tokens are processed. | The 10.9 s spent on the first step's prompt is now paid once at load. Later steps process only 20-100 new tokens, about 0.6 s each. |
| (a1) Just banning the `=self` token | **Failed.** The model answered "I can check that for you." without calling any tools, or got around the ban with `to self` / `to="self`. |
| (a2) Constrain every assistant header to `to=<tool>` or `to=user`. The first step of a turn must be a tool call, and `resolve_symbol` is left out because the validator already resolves names. | Headers are always valid and nothing is spent on reasoning, but the model **picks the wrong tools** (e.g. `screen_watchlist` four times). Without a reasoning step it chooses poorly. |
| (a3) Same as (a2), plus a short **pre-written reasoning message** ("User asks: … Call the one or two tools that answer this") that is processed as prompt instead of generated | Tool choice is right again. First tool call at **1.5 s**. The two-tool turn takes **6.5 s** (3 steps: 1.5, 2.0 and 3.0 s). Single-tool turns take 3.4-5.3 s. |
| Force the fixed `<atem:function_calls><atem:invoke name=".."><atem:parameter name="` text after a tool header and process it in one pass | Cuts each tool step from ~45 to ~23 generated tokens (~20 fewer), about **1 s per tool step**. |
| (d) Cap answers at 70 tokens or two sentences, then trim to the last whole sentence | Answer steps take 1.4-3.0 s instead of 11-15 s (most of that gain comes from dropping the reasoning). |
| (c) DFlash speculative decoding | **Not used.** The official drafter exists (`meta-models/Muse-Glimmer-30B-assistant`: 5 layers, 16-token blocks, reads hidden states from target layers 1, 13, 25, 37 and 49). mlx_lm 0.32.2 has no `muse_glimmer_assistant` model and no DFlash path; its speculative decoding only accepts a standalone causal draft model. Porting it is real work. It could roughly double decode speed. |

What's left: each tool step costs ~0.6 s to process the tool result plus ~1 s to decode ~23 tokens. Answers cost about 50 ms per token. The model still makes one tool call per step even when asked to make two. A parallel call would bring the two-tool turn to about 4.5 s. Fine-tuning for brevity and parallel calls should close that gap.
Turn the old behaviour back on with `GlimmerBackend(think=True)`.

## Modes (added later)
The pre-written reasoning message, the hidden `resolve_symbol` and the forced first tool are now **`mode="base"` only**. The forced first tool is also off by default, because it broke the no-tool rows. Adapters run in `mode="trained"`, which follows the canonical format in FORMAT.md. It keeps the forced function_calls prefix and the answer cap.
