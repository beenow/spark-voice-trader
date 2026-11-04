# Glimmer trained format (canonical)

`tests/test_format_parity.py` checks this format token for token. It renders traces through `chat_template.jinja` the same way the trainer does (`apply_chat_template(messages, tools=..., return_dict=False, **kwargs)`, prompt cut at `add_generation_prompt=True`). It then compares that with the stream that trained-mode inference builds (`agent.format.PromptBuilder`, the same object `GlimmerBackend` uses). The comparison covers four things:

- the whole trace;
- every decision-point prompt, which is exactly the trainer's mask offset for that assistant turn;
- the cached system+tools prefix;
- tool-result wrapping.

All of them match for:

- two-tool traces;
- calls with numeric arguments;
- a no-tool greeting;
- a follow-up with history.

The template adds **no** `to=self` block when an assistant message has no `reasoning_content`.

## What the trainer must render

Every message must be **exactly** as below. The safe way to get this is to build the messages with `agent.format.trace_messages(question, steps, answer, history)` and then render them with `render_trace(...)`, or hand them to mlx_lm's ChatDataset. Steps are `{tool, args, view}`.

| item | must be |
|---|---|
| `tools=` | `tools.schema()`: all 14 tools, in the same order, with the same text |
| template kwargs | `chat_template_kwargs: {reasoning_strength: "low"}`. If it is left unset, the template writes "Reasoning strength: high." and the system prefix no longer matches. `enable_thinking` is ignored by this template. |
| system | `agent.loop.SYSTEM_PROMPT`, verbatim. Don't add a context line. |
| history | The last ≤3 turns as `user` (the question) + `assistant` `{"content": <that turn's SPOKEN answer>}` |
| user | `"[state: last_symbol=X, last_topic=Y]\n" + question` (`build_messages`). Leave the state line out when there is no history. Frame models (`frame_line`) append `, last_call=T(k=v, …)[; T2(…)]` and `, facts=f1; f2` (see below). |
| tool call | One call per assistant message: `{"role":"assistant","content":"","tool_calls":[{"id":"call_k","type":"function","function":{"name":T,"arguments":{…}}}]}`. `arguments` is a mapping holding exactly what the model should emit. Don't fill in defaults; use ints as ints (`-10`, not `-10.0`, which renders differently). Don't add `reasoning_content`. |
| tool result | `{"role":"tool","tool_call_id":"call_k","name":T,"content": json.dumps(model_view, sort_keys=True, ensure_ascii=False)}`. It is applied to the **model_view** (`tool, ok, words, slots:["{name}"], error`), never to raw values. Slot names already carry the per-turn `_2` suffix. |
| answer | `{"role":"assistant","content": <template with {slots}, ≤2 sentences, no digits>}`. It renders as ` to=user<|message|>…<|eot|>`. |

How these render:

- Tool calls render as `<|start|>assistant to=T<|message|><atem:function_calls>\n<atem:invoke name="T">\n<atem:parameter name="k">v</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>`.
- Results render as `<|start|>tool T<|message|><tool_output name="T">\n…\n</tool_output><|eot|>`.

No-tool rows (greeting, clarify, decline, chat) are just system + user + answer.

## Inference modes (`GlimmerBackend(mode=...)`)

**`trained`** (the default when `adapter=` is given):

- Assistant headers are constrained to ` to=<any of the 14 tools>` or ` to=user`.
- Nothing is forced on step 1 and there is no reasoning stub.
- After a tool header, the template-invariant prefix `<atem:function_calls>\n<atem:invoke name="T">\n` (14-15 tokens) is forced. It is the longest token prefix shared by every argument set, so it never creates a token boundary the model wasn't trained on.
- After `</atem:function_calls>`, only `<|eot|>` (done) or `<|eom|>` (another call) may follow.
- Answers are capped at 70 tokens / 2 sentences.

**`base`** (the honest baseline for the before/after chart):

- Same constraints as trained mode, plus a pre-written reasoning message in place of the model's slow `to=self` channel: "If this needs market or account data, call the one or two tools…; otherwise reply briefly".
- `resolve_symbol` is hidden from the model.
- No forced first tool (`force_first_tool=False`).

Measured on the base model:

- **Without forcing:** the tool questions were identical to the forced run. All four no-tool rows were correct: greeting, thanks, the clarify ("Which ticker…?") and the joke.
- **With forcing:** it drafted an AAPL order for "Buy some shares." and called tools for greetings.
- So base mode does no-tool rows and should be scored on them.
- Known base weakness: "next week's Apple calls" makes it loop on `screen_watchlist`.

## Pack note

The 3,072-token sequence length fits. A 5-step LoRA run (rank 16, 16 layers, grad checkpointing, batch 1, rows of 3,032-3,091 tokens) peaked at 28.5 GB footprint and took about 29 s per step.

# Llama 3.2 format (`LlamaBackend`, `agent.format.render_llama`)

Llama 3.2 1B Instruct does not use its HF chat template here. That template adds a dated "Cutting Knowledge / Today Date" system line (the prompt would change every day), puts the full JSON tool schema in the user turn (~2,000 tokens), and refuses parallel calls. A 1B model degrades with long prompts, so the format is a compact one built from Llama's own special tokens. `render_llama` is the ONE renderer for training examples and inference prompts, so they are byte-identical by construction. `tests/test_format_parity.py` checks the token streams: whole trace, every decision prompt equal to its training mask offset, the cached prefix, and parallel-call regrouping.

| item | rendering |
|---|---|
| system | `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n` + `LLAMA_SYSTEM` + `\n\n` + `llama_tools_text(tools)` + `<|eot_id|>`. A system message passed in (the Glimmer `SYSTEM_PROMPT` from the loop) is replaced. About 565 tokens, prefilled once and cached. |
| tools | One line per tool: `name(arg, opt?=a\|b, num:number): description`, generated from `tools.schema()`. |
| history, user | Same messages as Glimmer (`build_messages`, `training/render.py`): last ≤3 turns as user + spoken answer, state line on the current question. |
| tool calls | One assistant message per decision. Each call is one line, `{"name": T, "parameters": {…}}` (`json.dumps`, `ensure_ascii=False`). Several lines = parallel calls. |
| tool results | One `ipython` message per call, in call order: `json.dumps({"tool","words","slots"[,"error"]}, sort_keys=True)`. It is the model_view without `ok`. Keys are sorted because DB jsonb and the live tools order `words` differently. Slot names only, never values. |
| answer | Assistant text with `{slots}`, then `<|eot_id|>`. |

**Training examples.** `llama_examples(messages, tokenizer)` gives one `(tokens, offset)` per assistant decision after the last user message. Only `tokens[offset:]` is trained: the assistant body + `<|eot_id|>`. History answers are never targets.

**Parallel calls.** Teacher traces store steps flat. `turns` (teacher replies) recovers the grouping: one call turn means the calls were parallel. An eval that stores each call as its own assistant+tool pair restores the model's own grouping with `regroup_calls` before building the next prompt.

**Inference.** Greedy, no constrained decoding. The KV cache keeps the longest common token prefix with the last stream (trimmed otherwise), so the system prefix and earlier steps are never re-prefilled. Answers stop at 2 sentences / 70 tokens.

# Frame state line (models trained with `frame_line`)

Models whose registry entry has `"frame_line": true` get two more keys on the state line, in this order and only when present:

- `last_call=` the last tool-calling turn among the last 3: at most 2 calls, `; `-separated, each `agent.memory.call_text(tool, args)` = `tool(k=v, …)` with keys sorted, schema defaults dropped (except `side` / `expiry_rule`), integral floats as ints. Example: `last_call=get_option_chain(expiry_rule=next_week, side=call, symbol=TSLA)`.
- `facts=` pinned user facts, `; `-separated, e.g. `facts=name Minh; trades mostly NVDA`.

Full line: `[state: last_symbol=TSLA, last_topic=options, last_call=get_option_chain(expiry_rule=next_week, side=call, symbol=TSLA)]\nand the puts?`

Training rows carry these as the LAST context entry's `state` (`agent.memory.context_entries`), with keys `agent.memory.STATE_KEYS` (`last_symbol, last_topic, last_call, facts`), so `training/render.py` writes the same line the loop builds at inference. Rows without the keys render exactly as before. Models trained without them keep the old line (`frame_line` false).
