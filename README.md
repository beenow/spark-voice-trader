# Leaps Voice Agent

A voice trading assistant that decides its own tool calls. You ask a question out loud (through Ray-Ban Meta
glasses paired to a phone, or the phone itself), and a **Llama 3.2 1B** model running locally on a Mac Studio picks
one to four tools, reads the results and speaks a short answer back, usually in under a second of model time. The
model was taught by **Muse Spark**, which wrote the tool-call traces it was fine-tuned on.

Market data is synthetic on purpose. One seeded generator backs all 14 tools, so prices, positions, option
chains and P&L are fake but consistent with each other. The point is to show that a small local model can choose
and chain tools correctly, not that the numbers are real.

> Not financial advice. The agent reports mock data and drafts order tickets that need a confirmation outside the
> model. Voice alone never sends an order.

## The demo

1. "How's my NVDA position, and is it near support?" The model calls `get_position` and `get_levels` in parallel.
   The answer you hear: "You hold one hundred twenty-five shares at about one fifty-eight, up seventeen point two
   percent… Yes, it's near support at about one eighty-three, just zero point eight percent away."
2. "What about its RSI?" A follow-up. The ticker carries over: `get_indicators(NVDA, rsi)`.
3. "Show me next week's Tesla calls." "And the puts?" "Buy two of the second strike." The chain, the same chain for
   puts, then a ticket for two 415 puts that **needs confirmation**.
4. "Should I buy Tesla?" No tool, no advice.

The phone page shows the live trace (each tool call, its result and the answer template) next to the spoken answer.

## How it works

```
phone / glasses mic ─► WebSocket ─► Whisper (MLX) ─► agent loop ─► answer template ─► code fills numbers ─► Kokoro (MLX) ─► speaker
                                                       │   ▲
                                         Llama 3.2 1B  │   │ words + {slot} names, never numbers
                                         + LoRA        ▼   │
                                                  14 tools over mockmarket
```

- **The model decides, code checks.** At each step the model emits tool calls (one JSON line each; several lines
  are parallel calls) or a final answer. Code validates every call against the schema, resolves tickers, runs the
  tool and returns only judgment words and slot names (`"near_support": "yes"`, `{support}`). The loop stops at an
  answer or after 4 steps.
- **The model never sees a number.** It writes "Support is {support}, {dist_support} away." Code fills in the values,
  once for the screen ("$182.40") and once for speech ("about one eighty-two"). A trained model can't invent a price.
- **Speed.** The system prompt and tool list (about 565 tokens) are prefilled once and cached; each step only reads
  the new tool results. A code-made filler line ("Checking your Nvidia position.") starts playing on the first tool
  call. Speech is synthesised one sentence at a time.

Diagrams and details:

- [docs/voice-path.html](docs/voice-path.html): the speech path, from the mic to the spoken answer.
- [docs/training.html](docs/training.html): how the student is trained.
- [SPEC.md](SPEC.md): the tool contract, agent loop, validator and WebSocket protocol.
- [FORMAT.md](FORMAT.md): the exact prompt formats, which are checked token for token in tests.

### Layout

```
mockmarket/   seeded generator: ~57 tickers, 5-minute and daily bars, portfolio, Black-Scholes option chains, news, events
tools/        14 tool schemas and implementations; slots, judgment words, display and spoken rendering
agent/        loop, validator, model backends (Llama, Glimmer, scripted), slot filler, conversation memory, CLI
voice/        FastAPI server, Whisper STT, Kokoro TTS, WebSocket, the phone page (voice/static/index.html)
training/     trace schema, trace filters, renderer (traces -> Llama examples), LoRA config + trainer
eval/         eval harness + a synthetic 20-question sample
tests/        pytest
docs/         diagrams, memory design
```

## Meta models used

| role | model | how |
|---|---|---|
| Teacher | **Muse Spark** (`muse-spark-1.3`) | Meta Model API. Runs the agent loop against the mock tools and writes the traces the student learns from. |
| Student (served) | **Llama 3.2 1B Instruct** + LoRA | Runs locally with MLX on the Mac Studio. Trained only on Muse Spark traces. |
| Comparison | **Muse Glimmer 30B** (4-bit), base and + a Spark-trained LoRA | Same tools, same test, locally with MLX. |

Speech runs locally too: Whisper large-v3-turbo (MLX) in, Kokoro-82M (MLX) out. The only cloud call is Muse Spark
while writing training traces; answering a question uses no cloud model.

## Results

Frozen test set: **238 trader questions**, never trained on, run live against the mock tools and scored per check
against gold labels (the checks are never blended into one number). Latency is the p50 time per answer on the Mac
Studio (M4 Max).

**Served agent loop** (what the phone gets: model + code checks), 2026-10-28:

| system | tool choice | arguments | step count | answer contract | p50 / p90 |
|---|---|---|---|---|---|
| **Llama 3.2 1B + Spark LoRA, multi-turn traces (`llama-1b-loop`, served)** | **90.8%** | **89.9%** | 93.3% | 100% | **0.68 / 1.17 s** |
| Llama 3.2 1B + Spark LoRA, single-turn traces | 88.2% | 87.0% | 91.6% | 100% | 0.68 / 1.16 s |

**Model alone** (trace loop, no code repairs), same 238 questions:

| system | date | tool choice | arguments | step count | answer contract | p50 |
|---|---|---|---|---|---|---|
| Muse Glimmer 30B, base | 2026-10-26 | 61.8% | 58.4% | 67.7% | 74.8% | 5.08 s |
| Muse Glimmer 30B + Spark LoRA | 2026-10-27 | 77.7% | 79.0% | 84.0% | 95.8% | 4.19 s |
| Llama 3.2 1B + Spark LoRA, single-turn traces | 2026-10-28 | 87.4% | 78.1% | 91.6% | 99.2% | 0.64 s |

- **tool choice**: the multiset of tools called equals the gold's. **arguments**: every gold call was made with at
  least the gold's arguments. **step count**: as many calls as the gold. **answer contract**: the answer uses only
  slots the results offered and has no digits of its own.
- Training on Muse Spark traces lifts Glimmer from 61.8% to 77.7% tool choice. The 1B Llama student, trained on the
  same kind of traces, beats both Glimmer systems while answering about six times faster, which is what a voice
  answer needs.
- The test questions and their labels are private and not in this repo. To try the harness, run it on the synthetic
  sample: `./run.sh eval` (the served adapter passes all 20; the sample is a smoke test, too small to measure with).

## Multi-turn and memory

Follow-ups work: "and the puts?", "what about next week?", "buy two of those", "the second strike", "say that
again", "call me Minh", "start over". Each turn leaves a small frame written by code (the calls that ran, the
company, option side, expiry, strikes and what was said). Code fills only what the user left out, and the model
gets a one-line summary of the last call (`last_call=get_option_chain(expiry_rule=next_week, side=call,
symbol=TSLA)`), so it learns to repeat the last call with only what changed. Fresh results are reused for a short
window, and the conversation survives a reconnect for a day.

On a multi-turn dev set of 295 follow-up turns, training on Spark's multi-turn traces took follow-up tool choice
from 73.6% to **97.3%** and arguments from 72.9% to **96.6%**, with the ticker carried over correctly 99.6% of the time.

Details: [docs/memory.md](docs/memory.md).

## Safety

- **Orders become tickets.** `draft_order` only creates a ticket marked "needs confirmation". Confirming happens on
  the phone screen, outside the model, and the voice path never sends an order. A ticket with no quantity is not
  drafted: the agent asks how many.
- **No advice.** "Should I buy X?" gets a decline and an offer of data, not a recommendation.
- **Synthetic data only.** No brokerage connection, no real account, no real prices.
- **No invented numbers.** The model only ever writes slot names; every value comes from a tool result.
- **Local memory is small and redacted.** Pinned facts never store digits, e-mail addresses or account, password
  or card words. Conversations are stored per passcode under a salted hash, never the passcode itself.

## Run it locally

Apple silicon Mac, Python 3.12.

```bash
./run.sh setup                      # .venv + requirements.txt
./run.sh test                       # tests that need model weights skip when they are absent

# weights (about 2.5 GB for the 1B student; Whisper and Kokoro for voice)
mkdir -p ~/.leaps-voice-agent/models ~/.leaps-voice-agent/adapters
huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-bf16 --local-dir ~/.leaps-voice-agent/models/Llama-3.2-1B-Instruct-bf16
huggingface-cli download mlx-community/whisper-large-v3-turbo --local-dir ~/.leaps-voice-agent/models/whisper-large-v3-turbo
huggingface-cli download prince-canuma/Kokoro-82M --local-dir ~/.leaps-voice-agent/models/Kokoro-82M
huggingface-cli download <HF_REPO_ID> --local-dir ~/.leaps-voice-agent/adapters/llama-1b-loop
cp models.example.json ~/.leaps-voice-agent/models.json

./run.sh ask "How's my NVDA position, and is it near support?" "What about its RSI?"   # text only
VOICE_PASSCODE=choose-one ./run.sh serve                                               # http://127.0.0.1:8300
```

Open the page, enter the passcode, hold the button to talk (or type). For the phone, put the server behind HTTPS
(the browser needs it for the microphone), for example with Tailscale Serve or Funnel. For the glasses, pair them to
the phone as a Bluetooth headset; the page then uses their mic and speakers.

Paths are configurable: `VOICE_HOME`, `VOICE_MODELS_DIR`, `VOICE_MODELS`, `VOICE_LLAMA_MODEL`,
`VOICE_MEMORY_DIR` (see `agent/paths.py`). The Glimmer comparison needs `mlx-community/Muse-Glimmer-30B-4bit`
(about 17 GB) in `$VOICE_MODELS_DIR`.

## Training

How the student was made (diagram: [docs/training.html](docs/training.html); details: [training/README.md](training/README.md)):

1. Trader questions, each labelled with the tools that answer it, plus paraphrases and generated follow-ups.
2. Muse Spark runs the agent loop on each question against the mock tools, so every tool result in a trace is a
   real generator output, and writes the spoken answer as a template with slots.
3. Filters keep a trace only if its tools, arguments, steps and action match the label and its answer uses only real
   slots with no digits (`training/filters.py`).
4. The renderer turns each trace into one Llama example per decision, byte-identical to the inference prompt
   (`training/render.py`, checked in `tests/test_format_parity.py`).
5. LoRA on Llama 3.2 1B with MLX on the Mac Studio (`training/lora_config.yaml`), then the eval gate above.

## Built during the hackathon (Oct 26 – Nov 4, 2026)

Everything in this repository was written during the hackathon: the mock market, the 14 tools, the agent loop and
validator, the Llama and Glimmer backends, the prompt formats, conversation memory, the voice server and phone page,
the training renderer, filters and recipe, the eval harness and the docs. The adapter was trained during the event
on Muse Spark traces generated during the event.

Prior context, not part of this submission: **Leapsify**, an existing trading app by the same author, which is where
the idea and the trader questions come from; and a **private agent framework** (prior work, not published) that ran
the trace generation and training jobs locally on the Mac Studio. This repo does not depend on either: the trace
schema, filters, renderer and LoRA recipe here are enough to reproduce the pipeline. The trader questions, their labels
and the test questions stay private; only aggregate scores are reported.

## Adapter weights

The served adapter (`llama-1b-loop`, LoRA for Llama 3.2 1B Instruct) is published separately with a model card:
**<HF_REPO_URL>** (link added at release). It is distributed under the Llama 3.2 Community License.

## License

Code: MIT (see [LICENSE](LICENSE)). Model weights are under their own licenses: Llama 3.2 Community License for the
Llama base and this adapter; Meta's license for Muse Glimmer.
