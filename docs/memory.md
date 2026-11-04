# Conversation memory

How the voice agent keeps track of a conversation: what it remembers about each turn, what the model is shown,
which follow-ups code resolves and which the model resolves, when an earlier result is reused, how memory is
stored across reconnects, and how multi-turn behaviour is trained and measured.

Code: `agent/memory.py` (frame, resolution, reuse, commands, facts, store), `agent/loop.py` (`run_turn`,
`build_messages`), `voice/server.py` (one `Conversation` per session). Tests: `tests/test_memory.py`.

## 1. The frame

Every turn ends with a **frame**: a record filled by code from what actually ran, never by the model.
It lives on `Turn.frame` and is built by `agent.memory.build_frame`.

| field | type | filled from | used for |
|---|---|---|---|
| `v` | int | constant (`FRAME_VERSION`) | store format version |
| `ts` | float (epoch s) | clock at the end of the turn | staleness rule, retention |
| `question` | str | the transcript | history, store |
| `calls` | list of `{tool, args}` | the executed calls (after validation, symbol correction and follow-up filling) | `last_call=` on the state line; follow-up resolution |
| `symbols` | list of str | `symbol` args in call order | what "it" / "its" / "that" refer to |
| `option` | `{symbol, side, expiry_rule, strike, qty, said_strike}` | the last `get_option_chain` / `draft_order` call; `said_strike` = the strike of the row the answer priced (its bid / ask), else the first strike it named, else the most-open-interest row | "and the puts", "next week instead", "buy two of those" |
| `chain_strikes` | `{row: strike}` | the chain result's `strike_1..5` | "the second strike", "row two" |
| `slots` | list of str | this turn's slot names (`{price}`, `{strike_2}` ...) | debugging, eval |
| `answer` | str | the answer template (with `{slots}`) | history, eval |
| `spoken` | str | the spoken answer | history (what the model sees), "say that again" |
| `results` | list of `{tool, args, result}` | the full tool results | reuse (section 4); code-side only |
| `pinned_fact` | str (only on fact turns) | `parse_fact` | "forget that" removes the fact too |

The model never sees `results`, `chain_strikes` or any value: tool results reach the model as slot names and
judgment words only, exactly as in a single turn.

## 2. What the model sees per turn

The prompt is the system prompt (cached), the last **3** turns as `user` (question) + `assistant` (the
**spoken** answer, so no stale slot names and no digits), then the current question with a one-line state hint:

```
[state: last_symbol=TSLA, last_topic=options]
and the puts?
```

Models trained with the frame (`"frame_line": true` in the model registry) get two more keys on that line:

```
[state: last_symbol=TSLA, last_topic=options, last_call=get_option_chain(expiry_rule=next_week, side=call, symbol=TSLA), facts=name Minh]
and the puts?
```

- `last_call=`: the last tool-calling turn among those 3, at most two calls joined by `; `. Each call is
  `agent.memory.call_text`: tool name, keys sorted, schema defaults dropped (except `side` and
  `expiry_rule`), whole-number floats printed as ints. About 15-30 tokens.
- `facts=`: pinned user facts (section 6), `; `-separated, at most 5.

The line replaces nothing else: old turns stay as short spoken sentences (≈ 20-40 tokens each), so three turns of
context add roughly 100-150 tokens to a ≈ 600-token prompt whose system part is cached.

Training and inference produce this line from the same code. Training rows carry it as the last context
entry's `state` (`agent.memory.context_entries`) with keys `[last_symbol, last_topic, last_call, facts]`, and
the renderer (`training/render.py`) writes `[state: k=v, ...]` from them, byte-identical to the loop's line. A
model trained without the keys keeps the old line, so switching it on never surprises an older adapter.

## 3. Follow-up resolution: code vs model

The model decides **what** to do: which tool, which company, and what changed ("puts" instead of calls, AMD
instead of Nvidia). Code then fills in the parts the user left implicit, from the last frame. Code runs
before validation (`resolve_followup`). It overrides the model only where the model could not have known better or contradicts the user:
an ordinal strike (it never saw the values), a side or expiry the user said, and a side or expiry the user did not
say but just heard. Every fill is reported on the `tool_call` event as `"filled_from_frame": [keys]`.

| follow-up | example | who | rule |
|---|---|---|---|
| pronoun symbol | "what about its RSI" | model (state `last_symbol`); code backstop | if the model's call has no `symbol`, code uses the last frame's symbol |
| change one option field | "and the puts?", "what about next week", "same for AMD" | model picks what changed; code carries the rest | for `get_option_chain` / `draft_order`, when the question names no side (no expiry), the last chain's side (expiry) is used: it replaces the model's guess for the same symbol or a carried-over request ("same for", "what about", "and", "I mean"); for a fresh request on another symbol it only fills a gap |
| side / expiry the user said | "and the puts for the monthly?" | code | for `get_option_chain` / `draft_order`, exactly one side (calls / puts / "the 650 call"; not the verbs "put in", "call it") or one expiry phrase (this week, next week, monthly, nearest expiry) in the question sets that argument, whatever the model wrote; reported as `"from_question": [keys]` |
| "buy 2 of those" | after a chain answer | model: `draft_order(symbol, buy, qty=2)`; code: side, expiry, strike | strike = `said_strike`, the strike the last answer named, when the question says those / that / it / them |
| "the second strike" / "row two" | after a chain answer | code | strike = that chain row's strike, even if the model wrote another number (it never saw the values) |
| "compare it with AMD" | after an Nvidia RSI turn | model | the same tool for both symbols; the earlier symbol's call is usually served from the reuse cache (section 4) |
| company correction | "I mean, Oracle." after an OR turn | model, then code | `validator.correct_symbol` redirects a lookalike or stale-state symbol to the company the user named |
| missing quantity | "open a ticket for the 650 call" | code | a `draft_order` with no `qty`, or whose `qty` is the number said as a strike ("the 650 call", "650 strike"; "50 calls" stays a quantity), is not drafted; the answer is "How many contracts do you want on that ticket?" and the held-back ticket is kept on the frame (`pending`) |
| the quantity answer | "three", "make it 5" | code | completes the held-back ticket with that quantity, without asking the model which ticket |
| strike said | "buy 3 Tesla 450 calls", "the 280s", "three of the three seventies" | code | the one strike the user said (digits or spoken: "three seventy", "one seventy five"; plural "the 280s") replaces the model's number (for an order, not the quantity) (`from_question`) |
| order status said | "did I put in any orders yesterday", "any open orders" | code | `get_orders` gets the status the user named (open / filled); "did I place / send / put in" means all |
| indicator / timeframe said | "five minute RSI on Supermicro", "above the moving averages" | code | `get_indicators` gets the one indicator (RSI / moving averages / volume) and timeframe (5 minute / hourly / daily) the user named; an unnamed timeframe is daily (`from_question`) |
| share orders | "buy 10 shares of it" | code | a `draft_order` that says shares / stock, or has no option field and no option words, never gets a side, expiry or strike from the frame |

## 4. use_history: reuse vs re-call

When the model asks for a call whose tool and arguments match a result from an earlier turn (one of the last
6), code answers it from that stored result if it is still fresh, and does not call the tool. The
`tool_result` event carries `"cached": true`. The model sees the same view (slot names and words) either way, so
the answer path does not change. The numbers stay consistent with what was just said, and the tool round-trip
is skipped.

Staleness (`STALE_S`, seconds since the earlier turn ended):

| data | fresh for | why |
|---|---|---|
| quote, indicators, option chain, market | 60 s | prices move |
| levels, watchlist scan | 5 min | computed from daily bars |
| news | 15 min | slow |
| fundamentals | 1 h | slow |
| position, portfolio | 30 s | changes when orders fill |
| orders | 15 s | the most volatile account data |
| `draft_order`, `what_if`, `resolve_symbol` | never | they act, or depend on the moment |

Account data (positions, portfolio, orders) is never reused across a turn that drafted a ticket. A reused result
drops the earlier turn's per-turn tags (e.g. `resolved_from`). Data that has gone stale, or was never fetched, is
called again. "Say that again" / "repeat that" / "what was
that" never reaches the model: code replays the last spoken answer (`"replayed": true`).

## 5. Persistence

A `Conversation` holds the turns (with frames) and the pinned facts for one user. The voice server makes one per
WebSocket session and loads it from a `SessionStore`, so a reconnect continues the same conversation.

- **Key**: one conversation per passcode. The file name is a salted SHA-256 of the passcode (24 hex
  characters). The passcode itself is never written.
- **Location**: `$VOICE_MEMORY_DIR`, default `$VOICE_HOME/sessions/` (`~/.leaps-voice-agent/sessions/`), on the Mac that
  runs the server. The directory is mode 0700 and the file is 0600. An empty `VOICE_MEMORY_DIR` turns
  persistence off, and memory then lives only for the connection.
- **Format**: one JSON file, `{"v": 1, "turns": [...], "facts": [...]}`. Each turn holds question, answer
  template, display, spoken, symbol, topic, tools and frame. Files are written atomically (temp file +
  rename).
- **Retention**: the last 6 turns and at most 5 facts are kept. On load, turns older than 24 h are dropped, and a
  malformed file is treated as an empty conversation. Stored tool results are dropped once older than the
  longest freshness window (1 h), since they can no longer be reused. Nothing leaves the machine.
- **Concurrency**: a turn that started before a clear, or whose connection has closed, is not recorded, so a
  late-finishing turn cannot bring cleared memory back or overwrite a newer session's.
- **Never stored**: audio, the passcode, any API key. Digit runs of 8 or more and e-mail addresses are redacted
  from every stored text field (question, answer, display, spoken). A "fact" containing digits, "@", or account / password / PIN / card / routing / token
  words is refused. Tool values in `results` come from the seeded mock market, not a brokerage.
- `hello` reports `"memory": {"turns": n, "facts": n}` so the page can show that a conversation is being
  continued.

## 6. Pinned facts

Code recognises short self-statements and keeps them as facts. Each one gets a code acknowledgement with no
model call:

| said | fact | reply |
|---|---|---|
| "call me Minh" / "my name is Minh" | `name Minh` | "Got it, Minh." |
| "I mostly trade Nvidia and AMD" | `trades mostly NVDA AMD` | "Got it, I'll keep Nvidia and AMD in mind." |
| "I mostly trade options" | `trades mostly options` | "Got it, mostly options." |

A fact turn keeps the previous turn's symbol, so "it" still means what it meant before. Frame models see the
facts on the state line.

## 7. Clear and forget

| said (or sent) | effect | events |
|---|---|---|
| "start over", "forget everything", "clear the conversation", "new conversation", "reset" - or `{"type":"clear_memory"}` | all turns and facts deleted; the store file is removed | `memory_cleared` `scope: "all"`, then a code answer "Okay, starting fresh." (the WebSocket message gets only the event) |
| "forget that", "scratch that", "never mind" | the last turn removed, plus the fact it pinned | `memory_cleared` `scope: "last"`, then "Okay, I've forgotten that." |

Commands are matched against the whole utterance, so "forget about Tesla puts" is an ordinary question. Command
turns are not added to history.

## 8. Multi-turn training traces (Muse Spark)

1. **Dialogues.** A small grammar generates 2-4-turn dialogues over the mock universe. The
   first turn is a quote, chain, technicals, news, fundamentals or position question. Each follow-up is drawn
   from: pronoun tool switch, symbol switch, lookalike correction, side or expiry change, order from a chain
   (explicit strike, "those", "the second strike"), compare, what-if, repeat. Training and evaluation use two
   **disjoint phrase banks**, so no eval phrasing is ever trained on.
2. **History by the real loop.** The earlier turns of each dialogue run through `agent.loop.run_turn` (via
   `Conversation`) with a scripted model: the gold calls, then a code-written answer built from the slots the
   results offered. Frames, follow-up filling, facts and spoken answers are therefore exactly what inference
   produces. About 15% of dialogues start with a pinned fact.
3. **Rows.** Each follow-up turn becomes one training row. The row holds the question, labels, the gold calls,
   and `context = context_entries(earlier turns, facts)`, which carries `last_call` / `facts` on its last
   entry's state. "Say that again" turns are code at inference, so they are not rows.
4. **Teacher.** Muse Spark 1.3 (`muse-spark-1.3`, Meta Model API) traces each row, with the same tools live and the
   same state line. Its instruction adds the follow-up rules: repeat `last_call` with only what changed;
   "compare" calls the tool for both symbols; "those" / "the second strike" draft from the chain the last
   answer described; ask for a missing quantity. The trace filters (`training/filters.py`) must pass: tool set, arguments
   against the gold, contract (only offered slots, no digits) and action. Spend is capped by the project's
   budget.
5. **Student.** The rows join the single-turn Spark traces in the Llama 3.2 1B LoRA run. The renderer is the
   one inference uses.

## 9. Multi-turn eval

The multi-turn eval runs each dialogue through `agent.loop.run_turn`, the served path, in one seeded world. Every
turn sees the model's **own** earlier turns as history, so a wrong turn 2 can make turn 3 wrong, as on the
phone. Each turn is scored against its gold with the same checks as single-turn rows (tool set, arguments,
action, contract).

Reported for turn 1 and for follow-up turns (turn ≥ 2, the target), and per follow-up kind:

- **tool_set** and **arguments**. Target for follow-ups: tool_set ≥ 90%.
- **symbol_carried**: on follow-ups whose gold names symbols, the executed calls' symbol set equals the
  gold's. Target ≥ 98%.
- **fallback_rate**, and latency p50 / p90 per turn depth (turn ≥ 3 has two earlier turns in the prompt;
  turn ≥ 4 has three).

There are two sets. The **dev set** (140 dialogues, 435 turns, 295 follow-ups) comes from the eval phrase
bank. The **live set** comes from conversations reported from real use, re-phrased, with gold labels. Neither
is ever trained on.

## 10. Latency

- Frame building, resolution, the reuse lookup and the store write are plain Python over a handful of dicts,
  far below 1 ms. The store write is one small JSON file per turn.
- Three turns of context add roughly 100-150 prompt tokens, about 30 of them for `last_call=`. Only the new
  tokens are prefilled, because the cached prefix covers the system prompt and tools.
- A reused result skips the tool call. A code-handled command, fact or repeat skips the model entirely.

Measured with the served Llama 3.2 1B adapter on the Mac Studio (multi-turn dev set, agent time per turn): follow-up
turns p50 0.65 s / p90 1.10 s, and turns with two or more earlier turns in the prompt p50 0.64 s (section 12).

## 11. Known limits

- Reuse matches exact arguments only. "Its RSI" after a full indicators call re-fetches, although the RSI
  was in that result.
- "Those" means the contract the last answer **priced** (its bid / ask row), else the first strike it named, else
  the most-open-interest row. If neither applies, the model's own strike is used, and a missing one is left to the
  tool.
- Ordinals cover first to fifth, "top", "last" (the chain's last row) and "row one" to "row five". "The cheaper one" and "the one
  with the most volume" are left to the model.
- Facts are a small fixed set of patterns. Anything else stays an ordinary question.
- One conversation per passcode (the server has one passcode and one session at a time): two phones or tabs
  that share a passcode share, and take turns writing, one memory.
- The mock market is regenerated from its seed on restart. Stored results stay internally consistent, but after
  a restart they may differ from a fresh call, which is one more reason for the short freshness windows.

## 12. Results (2026-10-28, Llama 3.2 1B trained on Muse Spark traces only)

Multi-turn dev set (140 dialogues, 295 follow-up turns), agent loop with the memory code, same code for both rows:

| adapter | follow-up tool_set | arguments | symbol carried | fallback | p50 / p90 per follow-up |
|---|---|---|---|---|---|
| Llama 1B + Spark, single-turn traces only, frame line off | 73.6% | 72.9% | 93.5% | 3.4% | 0.64 / 1.17 s |
| **llama-1b-loop**: + multi-turn Spark traces, frame line on (served) | **97.3%** | **96.6%** | **99.6%** | 0.0% | 0.65 / 1.10 s |

The memory code alone moves what code can decide (an explicit strike, "the second strike", a side or expiry
change). The rest ("buy two of those", "compare it with X", lookalike corrections, symbol switches) needed the
multi-turn traces: the model learns to repeat `last_call` with only what changed.
