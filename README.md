# Leaps Voice Agent

A voice trading assistant that decides its own tool calls and drives the app for you. You talk to it hands-free
(through Ray-Ban Meta glasses paired to a phone, or the phone itself). A **Llama 3.2 1B** model, running **on the
phone**, picks one to four tools, reads the results, speaks a short answer and opens the matching screen of a small
trading app, **Leaps Trader**: the chart with the support line it just mentioned, the options chain on the strike it
named, or a pre-filled order ticket for you to review and send. The model was taught by **Muse Spark**, which wrote
the tool-call traces it was fine-tuned on. The same model also runs on a Mac behind a web page (`voice/`).

Market data is synthetic on purpose. One seeded generator backs all 14 tools, so prices, positions, option
chains and P&L are fake but consistent with each other. The point is to show that a small local model can choose
and chain tools correctly, not that the numbers are real.
The watchlist ("what's on my watchlist", "add Netflix to my watchlist") is a code route, not a model tool: code runs the
quotes and fills the answer, and those calls are marked `lane: "code: watchlist"` in the event trace.

> Not financial advice. The agent reports mock data and drafts order tickets that need a confirmation outside the
> model. Voice alone never sends an order.

## The demo

One tap starts a conversation; after that you just talk (a question ends after a short pause, and the mic reopens
when the answer has been spoken). In Leaps Trader:

| You say | Tools the model calls | What you hear | What the app shows |
|---|---|---|---|
| "How's my Nvidia position, and is it near support?" | `get_position` + `get_levels` (parallel) | "You hold one hundred twenty-five shares at about one fifty-eight, up seventeen point two percent... Yes, it's near support at about one eighty-three, just zero point eight percent away." | Nvidia: the position card while the first sentence plays, then the support line pulsing on the chart as the second starts |
| "What about its RSI?" | `get_indicators(NVDA, rsi)`: the ticker carries over | "Nvidia's RSI is thirty-seven, neutral on the daily." | The RSI panel under the chart |
| "Show me next week's Tesla calls." / "And the puts?" | `get_option_chain` twice | the strike with the most open interest, bid, ask, delta, IV | The options tab, next week's expiry, calls then puts, the strike highlighted |
| "Buy two of the second strike." | `draft_order` | "I've drafted a ticket to buy two Tesla four fifteen puts expiring November sixth. It needs your confirmation." | A pre-filled order ticket. **You tap Confirm**; voice never sends an order |
| "Show me my orders." | `get_orders` | the Tesla order | Activity, the order marked "via voice" |
| "Which names on my watchlist are overbought?" | `screen_watchlist` | four names | Home, the matches highlighted in the watchlist |
| "Should I buy Tesla?" | none | a decline and an offer of price, chart or news | stays put |
| "What happens to my account if the market drops ten percent?" | `what_if(ALL, -10)` | about twenty-two thousand dollars lost | Home, positions |

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
voice/        FastAPI server, Whisper STT, Kokoro TTS, WebSocket, the phone page (voice/static/index.html),
              the step server for the Android app, and Leaps Trader's screens (ui_api.py, ui_static/)
training/     trace schema, trace filters, renderer (traces -> Llama examples), LoRA config + trainer
eval/         eval harness + a synthetic 20-question sample
tests/        pytest
docs/         diagrams, memory design
```

## On-device Android app

The same trained Llama 3.2 1B also runs **on the phone**. `android/` is a Kotlin app for a Galaxy Z Fold
(Snapdragon 8 Elite Gen 5):

- **On the phone:** speech-to-text (Android SpeechRecognizer), the Llama 3.2 1B student (llama.cpp, Q4_K_M GGUF,
  808 MB, the adapter fused into the base), and speech out. This repo speaks with Android TextToSpeech; the demo
  video uses a proprietary on-device Kokoro-82M voice (prior work, not included) plugged in through `OnDeviceVoice`.
- **On the Mac (`voice/step_server.py`):** prompt rendering, the code checks, slot filling, memory and the tools.
  The phone only generates, so every check is the same code that was evaluated.
- **Speed:** the fixed system+tools prefix (577 tokens) is prefilled once and its KV cache saved, so a turn only
  prefills the new tokens. Model time is about 1 s per turn, with the demo voice's first audio 0.7-1.0 s later.
- **Quality:** the 4-bit GGUF scores 90.8 tool_set / 89.1 args on the test split, against 90.8 / 89.9 for the
  full-precision MLX model.

See [android/README.md](android/README.md) for the build and what runs where.

## Leaps Trader: the agent drives the app

The Android app is a small trading app (Home with the account, positions and watchlist; a symbol page with the chart,
EMA 9/21/50, labelled support and resistance and an RSI panel, the position, news, fundamentals and an options tab;
an order ticket; Activity). Its screens are a web view fed by the same mock market the tools read
(`voice/ui_api.py`, `/ui/*` on the step server), so every number on screen matches the spoken one.

- **Navigation comes from the tool calls, not from the model's words.** When a turn's tools have run, code maps them
  to a screen (`nav_for`): `draft_order` opens the ticket, `get_option_chain` the options tab on that expiry, side and
  strike, `get_levels` / `get_indicators` the chart with the line or panel asked about, `get_position` the position
  card, `get_orders` Activity, `screen_watchlist` and `what_if` Home.
- **Multi-part answers move with the voice.** For a turn with several tools, `nav_seq` ties each screen to the
  sentence that speaks its numbers, and the app changes screen as that sentence starts playing.
- **Orders stay tickets.** The ticket is pre-filled from the voice request; Confirm is a tap on the screen. In this
  demo Confirm marks the mock ticket filled; there is no brokerage connection.

## In production (Leapsify)

This agent is the prototype for the assistant in the [Leapsify](https://leapsify.com) trading app. The production
version uses the same design: a small on-device Llama student, a closed tool-calling loop, slot-filled answers and
code checks. It runs over a private catalogue of about 50 real tools, covering quotes, option chains, indicators
and levels, news, fundamentals, portfolio and positions, orders, watchlists, alerts and in-app navigation.

- **Account tools run only on the phone.** Positions, orders and balances never leave the device or reach a server.
- **Market data tools run on the server.**
- **Orders are always tickets.** The user confirms each one on screen; voice never places a trade.

The production tools, their schemas and their training data are private. The 14 tools in this repo are a public
stand-in over a simulated market, so anyone can run and evaluate the agent end to end without a brokerage account.

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
huggingface-cli download leapsify/spark-voice-trader-llama-3.2-1b-lora --local-dir ~/.leaps-voice-agent/adapters/llama-1b-loop
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

## Built during the hackathon (Oct 26 - Nov 4, 2026)

Everything in this repository was written during the hackathon: the mock market, the 14 tools, the agent loop and
validator, the Llama and Glimmer backends, the prompt formats, conversation memory, the voice server and phone page,
the training renderer, filters and recipe, the eval harness and the docs. The adapter was trained during the event
on Muse Spark traces generated during the event. Not included: the on-device neural voice used in the demo video
(Kokoro-82M with a phonemizer from Leapsify, prior work); the app here speaks with Android TextToSpeech instead.

Prior context, not part of this submission: **Leapsify**, an existing trading app by the same author, which is where
the idea and the trader questions come from; and a **private agent framework** (prior work, not published) that ran
the trace generation and training jobs locally on the Mac Studio. This repo does not depend on either: the trace
schema, filters, renderer and LoRA recipe here are enough to reproduce the pipeline. The trader questions, their labels
and the test questions stay private; only aggregate scores are reported.

## Adapter weights

The served adapter (`llama-1b-loop`, LoRA for Llama 3.2 1B Instruct) is published separately with a model card:
**https://huggingface.co/leapsify/spark-voice-trader-llama-3.2-1b-lora**. It is distributed under the Llama 3.2 Community License.

## License

Code: MIT (see [LICENSE](LICENSE)). Model weights are under their own licenses: Llama 3.2 Community License for the
Llama base and this adapter; Meta's license for Muse Glimmer.
