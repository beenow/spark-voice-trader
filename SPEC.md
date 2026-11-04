# Leaps Voice Agent - interface contract

Local build on Apple silicon. Python 3.12 (mlx-lm 0.32.2 with `muse_glimmer`, mlx-whisper, mlx-audio, fastapi, uvicorn; see requirements.txt).
Weights live in `$VOICE_MODELS_DIR` (default `~/.leaps-voice-agent/models`, see `agent/paths.py`): `Muse-Glimmer-30B-4bit`, `Llama-3.2-1B-Instruct-bf16`, `whisper-large-v3-turbo`, `Kokoro-82M`.

## Layout

```
mockmarket/   seeded trading-data generator (the ONLY mock part)
tools/        14 tool schemas + implementations over mockmarket; slots, judgment words, rendering
agent/        loop, validator, model backends (glimmer via mlx_lm, scripted stub), slot filler
voice/        FastAPI server, Whisper STT, Kokoro TTS, WebSocket, web page (voice/static/index.html)
training/     trace schema, renderer, trace filters, LoRA config + trainer (teacher: Muse Spark)
eval/         harness + synthetic sample (eval/sample.json)
tests/        pytest
```

## Tool result contract

Every tool returns `ToolResult` (JSON-serialisable dict):

```json
{
  "tool": "get_levels",
  "ok": true,
  "words":  {"near_support": "yes", "near_resistance": "no"},   // judgment words the MODEL sees
  "slots":  {"support": 182.4, "dist_support": 0.018},           // raw values: model sees only the slot NAMES
  "error":  null                                                 // or "unknown symbol" / "empty portfolio" / "no chain for that expiry"
}
```

`model_view(result)` = what the model sees: `{"tool", "ok", "words", "slots": ["{support}", "{dist_support}"], "error"}`. Symbols, ticker lists and reasons are words, not slots. No digits ever reach the model from a tool.

Slot names are unique per turn: if the same tool runs twice in one turn, the second's slots get a suffix `_2` (`{price_2}`).

Rendering (`tools/render.py`): `display(slot, value)` → "$182.40", "-2.3%", "1.8%"; `spoken(slot, value)` → "about one eighty-two", "down two point three percent". Slot kind is inferred from a per-tool slot-type map (price, pct, signed_pct, money, signed_money, count, ratio, index, greek, vol (implied vol), date, time, age, text).

## The 14 tools

| tool | arguments | slots | words |
|---|---|---|---|
| get_quote | symbol | price, change_pct, volume | direction(up/down/flat), move(small/notable/big) |
| get_position | symbol | qty, avg_cost, pnl, pnl_pct, weight; option_qty, expiry (date) when options are held in the symbol | direction(up/down), size(small/medium/large), held(yes/no), kind, expires(this week/next week/this month/later) for options |
| get_portfolio | rank_by? (pnl, pnl_pct, weight, day_move) | total, day_pnl, cash, expiry (nearest option expiry) | best, worst (symbols), day(up/down), positions (list of symbols, ranked), expires(this week/next week/this month/later/no options held), next_expiring, option_expiries [{position, expires}] |
| get_indicators | symbol, timeframe(1D/1H/5m)=1D, indicator?(rsi/ema/volume) | rsi, ema9, ema21, ema50, vol_ratio | rsi_state(oversold/neutral/overbought), trend(above/below 21 EMA), volume(light/normal/heavy) |
| get_levels | symbol | support, resistance, dist_support, dist_resistance | near_support(yes/no), near_resistance(yes/no) (within 2%) |
| get_option_chain | symbol, expiry_rule(this_week/next_week/monthly/nearest), side(call/put), strike? | strike_1..5, bid_1..5, ask_1..5, iv_1..5, delta_1..5, oi_1..5 (open interest), expiry | iv_level(low/normal/high), days_to_expiry(this week/soon/far), liquidity(thin/normal/deep, ATM open interest), most_open_interest("row two" = suffix _2) |
| get_orders | status(open/filled/all)=open, day?(today/yesterday) | per order: order_1_qty, order_1_price … | count(none/one/several), orders (list "BUY AAPL call") |
| get_news | symbol | age_1, age_2 | headlines (text list, synthetic), tone(positive/negative/mixed) |
| get_fundamentals | symbol | market_cap, pe, target, dist_target, earnings_date | valuation(cheap/fair/rich), earnings(this week/later) |
| get_market | focus(movers/breadth/calendar)=movers | top_move, bottom_move, index_move, event_time | top_name, bottom_name, market_day(up/down/flat), event_1(name), event_risk(none/today/tomorrow) |
| screen_watchlist | condition(oversold/overbought/overextended) | - | matches: [{symbol, reason}] |
| what_if | symbol, move_pct (number, allowed as an argument) | pnl_change, new_total | impact(minor/moderate/severe) |
| resolve_symbol | text | - | symbol, confidence(high/low), alternatives |
| draft_order | symbol, direction(buy/sell), qty, type(market/limit), side?(call/put), strike?, expiry_rule? | ticket_id | status: "needs confirmation" (always; never sends) |

Arguments may contain numbers (qty, strike, move_pct) - they come from the user's words. Tool results never show numbers to the model.

## mockmarket

`World(seed:int, scenario:str|None)` - deterministic. ~40 core tickers + 14 extra real tickers (APP, ADBE, ORCL, AMAT, APLD, SMCI, SLB, RIOT, CRM, PYPL, COP, XYZ=Block, SMH; VIX is quote-only: other symbol tools return error "quote only symbol"); extras use a separate RNG stream and are never in the portfolio/watchlist (base price, vol, sector, spoken names e.g. NVDA: ["Nvidia","en vidia","N V D A"]), lookalikes (OR/ORLY, META/MET, NVDA/NVDL). 5-min bars for one session + 200 daily bars. Portfolio 8-15 positions (shares + options; NVDA always held as shares AND calls), cash. Option open interest is deterministic per (seed, contract). Options via Black-Scholes with per-ticker IV + greeks. Scenarios: gap_down, earnings_tomorrow, empty_portfolio, (unknown symbol / no chain handled by tools). Consistent: get_quote price == price used in get_position P&L.

## Agent loop (`agent/loop.py`)

```python
run_turn(question:str, history:list[Turn], world:World, backend:Backend, max_steps=4, on_event=callable) -> TurnResult
```

Events emitted via `on_event(dict)` in order: `{"type":"tool_call","tool","args","step"}`, `{"type":"tool_result","tool","words","slots_display":{...},"step"}`, `{"type":"filler","text"}` (once, on first tool call: code-made, e.g. "Checking your Nvidia position."), `{"type":"answer","template","display","spoken"}`, `{"type":"ticket", ...}` when draft_order ran, `{"type":"error",...}`.

Backend protocol: `backend.decide(messages:list[dict], tools_schema) -> {"tool_calls":[{"tool","args"}]} | {"answer":"text with {slots}"}`. Backends: `GlimmerBackend(model_dir, adapter=None)` (mlx_lm, model's own chat template + tool format, constrained/validated JSON), `LlamaBackend(model_dir, adapter=None)` (Llama 3.2 1B via mlx_lm, compact Llama format with JSON call lines and parallel calls, see FORMAT.md), `ScriptedBackend` (tests). `make_backend(name)` builds a named model (see Voice WebSocket).

`VOICE_REPAIRS=0` turns every code-side repair off (follow-up filling from the frame and the user's words, `repair_args`, the code answers for timeframes / arithmetic, the no-data retry, the quantity-as-strike ask, validator coercions), so an eval can show the model on its own. Memory commands, facts, repeat and result reuse stay on.

Argument repair (`loop.repair_args`, before validation): a missing or unknown `symbol` becomes the one company / ticker the question names ("S M H" -> SMH); `what_if` on "my book / portfolio / the market" with an unknown symbol becomes `ALL`; a side written into `expiry_rule` moves to `side`. Validator: spaced tickers ("S M H") are joined; a loose `expiry_rule` ("monthly expiry", "next friday") maps to the rule. A `draft_order` whose `type` is "call"/"put" (the side in the wrong field) gets that as `side` and the default type. Tool exists, args match schema, symbol in universe (else route through resolve_symbol), one retry with an error result. Answer check: every `{slot}` used exists in this turn's results; no digits outside slots (else one retry, then a safe fallback line). Answer ≤ 2 sentences.

Symbol correction (`validator.correct_symbol`, before the tool runs): when the user's words name a company (universe names / spoken aliases, `named_companies`) whose ticker X differs from the model's `symbol`, the call is redirected to X. Only if: the model's symbol is not itself spoken in the question (caps ticker, spelled "O R", "ticker OR"); exactly one named company qualifies and no other call this turn already covers it; and the model's symbol is a lookalike of X (universe lookalikes, shared prefix: OR/ORCL, META/MET, NVDA/NVDL, ORLY/ORCL) or is the previous turn's `last_symbol` (stale state: "I mean, Oracle." after an OR turn). The `tool_call` event then carries `"corrected_from"`, and the result's words get `resolved_from`.

Company names in answers come from the data: when the turn's calls were about one symbol, a company the answer names (capitalised name or caps ticker) that appears in none of the turn's args/words is replaced by that symbol's display name (`rename_companies`) before the answer check.

Binding check (`validator.binding_problems`, after the basic answer check; one retry, then fallback):
- Option-chain rows: a `bid_N`/`ask_N`/`iv_N`/`delta_N` slot needs `strike_N` (same row, same chain run) earlier in the answer, and the last strike said before it must be that row; `oi_N` only must not follow a different row's strike. Fallback after a failed retry is a code-built line that states the strike first: "For Tesla calls expiring {expiry}, the {strike_1} strike is {bid_1} bid, {ask_1} ask."
- Entities: each slot is bound to an entity via `tools.slots.entity_of` (`SLOT_ENTITY`: per-tool slot → `word:<key>` / `@market` / unbound; default = the call's `symbol`). E.g. get_market `top_move`→`top_name`, `bottom_move`→`bottom_name`, `index_move`→`@market`. In a sentence that names a company (or ticker), every bound slot must belong to an entity named in that sentence (`@market` counts as named when the sentence says market/index/S&P/Nasdaq). Sentences that name nothing are not checked. Fallback: the safe data line.

Code answers (no tool, no model): a chart timeframe the tools do not have (15 / 30 minute, 4 hour, weekly / monthly chart) gets "I only have five-minute, hourly and daily charts. Want the daily?"; plain arithmetic with no company in it ("two plus two") gets "I can only help with stocks, options and your account."

Unchecked "no data" claims: if no tool has run, the question asks for data (a ticker, or option / quote / position words) and the answer says the data is missing ("there's no … call", "I couldn't pull …"), the model gets one retry telling it to call the tool (naming the company / ticker the user said).

Ungrounded answers: if no tool has run and the answer uses slots, the retry tells the model it has no data and must call a tool or ask a clarifying question. If, after that retry, there is still no tool data and the answer is not a question, the turn ends with the clarify line `CLARIFY_UNGROUNDED` ("I haven't looked that up yet. Do you mean your positions, a stock, or the market movers?"), never a factual statement.

Chain wording (`agent.slots.label_chain`, applied in `fill` for display and spoken): bare `{bid_N} by {ask_N}` → "bid X, ask Y"; bare `{delta_N}` → "delta X"; bare `{iv_N}` ("at {iv_N}") → ", IV X". Spoken kinds: greeks drop the leading zero ("point five nine"); implied vol (`vol` kind) is a whole percent ("fifty-eight percent"); a ratio of about 1 (0.9-1.1) is "about average", not "one times".

History: last 3 turns as messages + state hint `last_symbol`, `last_topic`. Models trained with the frame (`models.json` `"frame_line": true` → `backend.frame_line`) also get `last_call=` (the last tool-calling turn's calls, `agent.memory.call_text`) and `facts=` (pinned user facts) on that line.

### Conversation memory (`agent/memory.py`, details in docs/memory.md)

- **Frame.** Every turn's `Turn.frame` is filled by code (`build_frame`): calls with executed args, symbols, option context (symbol, side, expiry_rule, strike, qty, `said_strike` = the strike the answer named), the chain's strike per row, slot names, answer template + spoken, the results (code-side only; values never reach the model) and a timestamp.
- **Follow-up resolution (code, `resolve_followup`, before validation).** Fills only arguments the model left out and the question does not state: `symbol` from the last frame; for `get_option_chain` / `draft_order`, `side` and `expiry_rule` from the last chain when the question names none (replacing the model's value for the same symbol or a carried-over request - "same for", "what about", "and", "I mean" - otherwise only filling a gap); for `draft_order`, `strike` from "the second strike" / "row two" (the chain's row) or from "those / that / it" (the strike the last answer named). For `draft_order` / `get_option_chain`, the one strike said (digits or spoken, "the 280s", "three seventy calls"; for an order, not the quantity) sets `strike`; for `get_orders`, a status said (open / filled) is set and "did I place / send" means `all`; a month name means the monthly expiry; for `get_indicators`, the one indicator (RSI, moving averages, volume) and timeframe (5 minute, hourly, daily) named are set, and an unnamed timeframe is `1D`. Exactly one side word / one expiry phrase in the question sets `side` / `expiry_rule` on those two tools whatever the model wrote (`"from_question": [keys]`). The `tool_call` event carries `"filled_from_frame": [keys]`.
- **Quantity.** A `draft_order` with no quantity, or whose `qty` is the number the user said as a strike ("the 650 call", `qty_is_strike`), is not drafted: the answer is the code line "How many contracts (shares) do you want on that ticket?" and the held-back args go on the frame (`pending`); a bare quantity next ("three", "make it 5") completes that ticket in code (`pending_order`). A share order ("shares", "stock", or no option field and no option words) never gets side / expiry / strike from the frame.
- **Reuse (use_history).** A call with the same tool + arguments as a result fetched in the last turns is answered from that result while it is fresh (`STALE_S`: quotes, indicators, chains, market 60 s; levels, watchlist 5 min; news 15 min; fundamentals 1 h; positions, portfolio 30 s; orders 15 s; `draft_order`, `what_if`, `resolve_symbol` never; account data never across a turn that drafted a ticket). The `tool_result` event carries `"cached": true`.
- **Commands (code, no model call).** "start over" / "forget everything" / "clear the conversation" → all turns and facts cleared; "forget that" / "scratch that" / "never mind" → the last turn (and the fact it pinned) dropped. Both emit `{"type":"memory_cleared","scope":"all"|"last"}` then a code `answer` (`"code": true`). "Say that again" / "repeat that" replays the last spoken answer (`"replayed": true`) without the model.
- **Pinned facts.** "Call me Minh", "my name is …", "I mostly trade Nvidia and AMD", "I mostly trade options": kept as short facts (≤5), acknowledged by code. Anything with digits, "@", or account / password / PIN / card words is never stored.
- **Persistence.** `Conversation` (per connection) loads and saves through `SessionStore`: one JSON file per passcode under `$VOICE_MEMORY_DIR` (default `$VOICE_HOME/sessions`, empty = off), named by a salted SHA-256 of the passcode (never the passcode), mode 0600, atomic writes; last 6 turns + facts, turns older than 24 h dropped on load; digit runs and e-mail addresses are redacted from stored questions.

## Voice WebSocket (`/ws?k=PASSCODE`)

STT (`voice/stt.py`): Whisper's `initial_prompt` = "A trader asks about stocks and options: " + trading terms with a digit cue ("buy 2", "sell 10", "300 calls", "650 call": keeps quantities and strikes as digits), then buying, put, strike, expiry, option chain, open a ticket, draft an order, RSI, EMA, …) + company names, then tickers (`mockmarket.spoken_vocab`, names first), capped at `PROMPT_CHARS` so it stays inside Whisper's 223-token prompt window (~215 tokens). Measured on Kokoro-synthesised questions: no STT latency change (254-268 ms either way); "AppLovin" now recognised (was "app love an").

Phone → server: binary frames = 16 kHz mono PCM16 audio while held; text `{"type":"end"}` on release; `{"type":"confirm_ticket","ticket_id"}`; `{"type":"set_model","model":NAME}`; `{"type":"text","text":"..."}` (typed question, for testing without audio); `{"type":"clear_memory"}` (same as saying "start over"; answered with `memory_cleared`).
Server → phone: text events = `transcript` + all agent events above; binary frames = reply audio (24 kHz mono PCM16) chunks, each preceded by a text event `{"type":"audio","sample_rate":24000,"seq":n}`; `{"type":"done","timings":{"stt","first_audio","total"}}`; `{"type":"memory_cleared","scope":"all"|"last"}`. `hello` carries `"memory": {"turns", "facts"}` (what this passcode's stored conversation holds on connect).

### Named models

`hello` carries `"model"` (the session's current model) and `"models": [{"name","label","family":"glimmer"|"llama","available"}]`, for the picker. Names come from `agent.backends.registry()` (`$VOICE_MODELS`, default `$VOICE_HOME/models.json`; see `models.example.json`: `glimmer-base`, `glimmer-spark`, `llama-1b-loop`); `available` is false when weights or the adapter are not on disk.

- `{"type":"set_model","model":NAME}` → `{"type":"model","model":NAME,"resolved":CANONICAL}`, or `error` (unknown / not available).
- Back-compat: `"base"` = the Glimmer without an adapter, `"trained"` = the Glimmer whose adapter is `$VOICE_ADAPTER` (else the first available Glimmer adapter). Old clients that only send `base|trained` and read `type` keep working.
- All Glimmer models share ONE loaded Glimmer (~17 GB): LoRA adapters are hot-swapped on it (base = all-zero LoRA, exactly the base model), each with its own cached system prefix. Llama 1B (~2.5 GB bf16) is a separate model. All available models are loaded and warmed at start (`VOICE_WARM_ALL=0` to skip), so switching is instant.

### Compare mode

The same question through two models in turn, each with its own trace, answer and timings.

- `{"type":"set_compare","models":[A,B]}` → `{"type":"compare","models":[A,B]}`; `{"type":"set_compare","models":null}` turns it off. While on, every `text` / `end` is a compare turn. A single `text` / `end` may instead carry `"compare":[A,B]`.
- A compare turn emits `transcript`, `{"type":"compare_start","models":[A,B]}`, then lane 0's agent events, `{"type":"lane_done","lane":0,"model":A,"agent","first_tool","first_audio","decides"}` (ms), then lane 1's events and its `lane_done`, then `done` with `timings.lanes` (both `lane_done` records). Every agent event of a compare turn carries `"lane"` (0/1) and `"model"`.
- Audio: only lane 0 speaks (filler + answer). Lane 1 runs after lane 0's speech is synthesised, so the two never share the GPU and lane 1's timings are clean. `{"type":"play_lane","lane":1}` speaks lane 1's answer on demand (audio frames + a `done` with `timings.replay=true`).
- History: both lanes see the history from before the turn; only lane 0's result is added to it.

One session at a time; passcode from env `VOICE_PASSCODE`; port 8300. The conversation (history + pinned facts) belongs to the passcode, not the connection: a reconnect continues it. Default model: `$VOICE_DEFAULT_MODEL`, else the registry's `default`.
