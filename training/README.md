# Training: Muse Spark traces to a Llama 3.2 1B adapter

The student learns from **traces**: complete runs of the agent loop on one question, written by the teacher,
**Muse Spark** (`muse-spark-1.3`), through the Meta Model API. The served adapter was trained this way, and the
files here are the contract and the recipe.

| file | what |
|---|---|
| `trace_schema.json` | one trace: question, optional context turns + state line, steps (call + real tool result), answer |
| `example_traces.jsonl` | three example traces built on `mockmarket.World(7)` (format reference, not training data) |
| `filters.py` | the automatic checks a trace must pass to be kept |
| `render.py` | traces -> Llama examples (`{"tokens", "offset"}`, one per decision), the same renderer inference uses |
| `lora_config.yaml` | the LoRA recipe of the served adapter |
| `train_lora.py` | trains a LoRA with mlx-lm's trainer on `render.py` output, keeping the best-validation checkpoint |

## How a trace is made with Muse Spark

1. **A labelled question.** Each question carries the gold tool calls that answer it (e.g. "How's my NVDA position,
   and is it near support?" -> `get_position(NVDA)` + `get_levels(NVDA)`). Questions get spoken-style paraphrases
   (typos, misheard names like "in video" for Nvidia, lookalike tickers), and follow-up questions get real earlier
   turns produced by the agent loop, so their state line (`last_symbol`, `last_topic`, `last_call`, `facts`) is
   exactly what inference builds.
2. **Teacher run.** Muse Spark gets the agent's system prompt, the 14 tool schemas (`tools.schema()`, OpenAI-style
   function tools) and the question with its context. Each tool call it makes is validated and executed against the
   mock tools (`tools/server.py`, `POST /call`, with a seed fixed per question so every step sees the same market).
   It receives the `model_view` of each result: judgment words and slot names, never a number. It ends with an answer
   template such as "You're up {pnl} on Nvidia. Support is {support}, {dist_support} away." Calls the teacher makes
   in one reply are recorded with the same `turn`, so the student learns parallel calls.
   Muse Spark is a reasoning model: its reasoning tokens count toward `max_tokens`, so the budget per call is set
   generously; the API caches the long shared prompt prefix automatically.
3. **Filter** (`filters.py`). Kept only if: every call names a real tool with schema-valid arguments; at most 4
   steps; the answer uses only slots its results offered and has no digits of its own; no order was sent; and the
   tools, arguments, step count and action match the gold label.
4. **Render** (`render.py`). One example per assistant decision, rendered in the compact Llama format of
   `agent/format.py` (FORMAT.md). Only the model's own tokens are trained. Paraphrase groups stay in one split.
5. **Train** (`train_lora.py training/lora_config.yaml`). LoRA rank 16 / scale 2 on all 16 layers, 3 epochs, cosine
   lr 2e-4 -> 2e-6 after 50 warmup steps, batch 2 x 4, best-validation checkpoint.

The served adapter used 4,971 accepted Spark traces: single-turn questions with paraphrases, 896 multi-turn
follow-ups, and two small targeted rounds aimed at misses seen in live testing. Total Muse Spark spend for all
traces was about $70 of the $150 hackathon credit.

In the build, the teacher calls, filtering and training jobs were run by a private agent framework (prior work, not
published) using this same schema, renderer and recipe. `render.py` and `train_lora.py` reproduce those last two steps
without it.

```bash
python -m training.filters traces.jsonl > kept.jsonl
python -m training.render kept.jsonl --out data/
VOICE_LLAMA_MODEL=~/.leaps-voice-agent/models/Llama-3.2-1B-Instruct-bf16 python -m training.train_lora training/lora_config.yaml
```
