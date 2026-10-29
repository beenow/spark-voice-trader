"""Canonical trained format for Glimmer + the token-stream builder shared by inference and training.

CANONICAL TRAINED FORMAT (see FORMAT.md). Messages are OpenAI-shaped and rendered by the model's own
chat_template.jinja with `tools=tools.schema()` and TEMPLATE_KWARGS:

    system     SYSTEM_PROMPT (agent.loop) — the template appends "Reasoning strength: low." + tool schemas
    user/assistant pairs for the last <=3 turns (assistant content = that turn's SPOKEN answer)
    user       "[state: last_symbol=X, last_topic=Y]\\n" + question   (state line only when there is history)
    assistant  {"content": "", "tool_calls": [ONE call, arguments = a mapping]}   -> to=<tool> + <atem:...>
    tool       {"tool_call_id", "name", "content": tool_content(model_view)}
    ...
    assistant  {"content": answer template with {slots}}                         -> to=user

No `to=self` reasoning, no reasoning_content, nothing forced. Every assistant turn starts directly with
` to=<tool>` or ` to=user`.

Pure: needs only a tokenizer (no model, no MLX).
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

TEMPLATE_KWARGS: dict[str, Any] = {"reasoning_strength": "low"}


def tool_content(view: Mapping[str, Any]) -> str:
    """A tool message's content: model_view as canonical JSON (keys sorted, non-ASCII kept).
    The trainer must write tool results with exactly these bytes."""
    return json.dumps(view, sort_keys=True, ensure_ascii=False)


def assistant_call(call_id: str, tool: str, args: Mapping[str, Any]) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": tool, "arguments": dict(args)}}]}


def trace_messages(question: str, steps: Sequence[Mapping[str, Any]], answer: str | None,
                   history: Sequence[Any] | None = None) -> list[dict]:
    """Canonical messages for one turn.

    steps: [{"tool": str, "args": {...}, "view": model_view dict}, ...] — one tool call per assistant message.
    history: agent.loop.Turn objects (or TurnResults) for the previous turns.
    """
    from .loop import build_messages
    msgs = build_messages(question, history)
    for i, st in enumerate(steps, 1):
        cid = f"call_{i}"
        msgs.append(assistant_call(cid, st["tool"], st.get("args") or {}))
        msgs.append({"role": "tool", "tool_call_id": cid, "name": st["tool"], "content": tool_content(st["view"])})
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def render_trace(messages: Sequence[Mapping[str, Any]], tokenizer, tools: list | None = None,
                 add_generation_prompt: bool = False) -> str:
    """Render canonical messages exactly as training must see them."""
    from tools import schema
    return tokenizer.apply_chat_template(list(messages), tools=tools if tools is not None else schema(),
                                         add_generation_prompt=add_generation_prompt, tokenize=False,
                                         **TEMPLATE_KWARGS)


def _lcp(seqs: Iterable[list[int]]) -> list[int]:
    seqs = list(seqs)
    out = []
    for col in zip(*seqs):
        if all(c == col[0] for c in col):
            out.append(col[0])
        else:
            break
    return out


class PromptBuilder:
    """Builds the exact token stream inference feeds the model (tokenizer only)."""

    def __init__(self, tokenizer, template_kwargs: Mapping[str, Any] | None = None):
        self.tok = tokenizer
        self.kwargs = dict(TEMPLATE_KWARGS if template_kwargs is None else template_kwargs)
        self.eot = tokenizer.convert_tokens_to_ids("<|eot|>")
        self.eom = tokenizer.convert_tokens_to_ids("<|eom|>")
        self.start = tokenizer.convert_tokens_to_ids("<|start|>")
        self.msg = tokenizer.convert_tokens_to_ids("<|message|>")
        self.asst_open = self.encode("<|start|>assistant")
        self._seq_cache: dict[tuple, list[list[int]]] = {}
        self._forced: dict[str, list[int]] = {}
        self.last_msgs: list[dict] | None = None
        self.last_tokens: list[int] | None = None
        self.last_ended = True

    # -- primitives ---------------------------------------------------------------
    def render(self, messages, tools, gen_prompt: bool) -> str:
        return self.tok.apply_chat_template(list(messages), tools=tools, add_generation_prompt=gen_prompt,
                                            tokenize=False, **self.kwargs)

    def encode(self, s: str) -> list[int]:
        return list(self.tok.encode(s, add_special_tokens=False))

    def prefix_tokens(self, messages, tools) -> list[int]:
        """System+tools prefix (the part cached once at load)."""
        sys_only = [m for m in messages[:1] if m.get("role") == "system"]
        return self.encode(self.render(sys_only, tools, False)) if sys_only else []

    def full_tokens(self, messages, tools, gen_prompt: bool = True) -> list[int]:
        """Whole-render tokenization, split after the system prefix (specials make the split exact)."""
        full = self.render(messages, tools, gen_prompt)
        sys_only = [m for m in messages[:1] if m.get("role") == "system"]
        pre = self.render(sys_only, tools, False) if sys_only else ""
        if pre and full.startswith(pre):
            return self.encode(pre) + self.encode(full[len(pre):])
        return self.encode(full)

    def tokens_for(self, messages, tools) -> list[int]:
        """Prompt for the next decision. Continues the previous stream (the model's own tokens stay as
        generated) when `messages` extends the last call's messages by our assistant turn + new messages."""
        n = len(self.last_msgs) if self.last_msgs is not None else -1
        if (self.last_msgs is not None and len(messages) > n + 1 and messages[:n] == self.last_msgs
                and messages[n].get("role") == "assistant"):
            head = self.render(messages[:n + 1], tools, False)
            full = self.render(messages, tools, True)
            if full.startswith(head):
                end = [] if self.last_ended else [self.eot]
                return self.last_tokens + end + self.encode(full[len(head):])
        return self.full_tokens(messages, tools, True)

    def commit(self, messages, tokens_with_generation: list[int], ended: bool) -> None:
        import copy
        self.last_msgs = copy.deepcopy(list(messages))
        self.last_tokens = list(tokens_with_generation)
        self.last_ended = ended

    def reset(self) -> None:
        self.last_msgs, self.last_tokens, self.last_ended = None, None, True

    # -- constrained decoding helpers ----------------------------------------------
    def header_seqs(self, names: Sequence[str]) -> list[list[int]]:
        key = tuple(names)
        if key not in self._seq_cache:
            self._seq_cache[key] = [self.encode(" to=" + n + "<|message|>") for n in names]
        return self._seq_cache[key]

    def call_text(self, name: str, args: Mapping[str, Any]) -> str:
        """The template's rendering of one call body (after '<|message|>')."""
        body = '<atem:function_calls>\n<atem:invoke name="' + name + '">\n'
        for k, v in args.items():
            if isinstance(v, bool):
                v = "true" if v else "false"
            elif isinstance(v, (dict, list)):
                v = json.dumps(v)
            body += f'<atem:parameter name="{k}">{v}</atem:parameter>\n'
        return body + "</atem:invoke>\n</atem:function_calls>"

    def forced_chunk(self, name: str) -> list[int]:
        """Tokens that ALWAYS follow ' to=<name><|message|>' in the template's tokenization: the longest
        common token prefix over representative argument sets, so forcing never creates a token boundary
        the trained model hasn't seen."""
        if name not in self._forced:
            samples = [{}, {"symbol": "NVDA"}, {"symbol": "AAPL", "timeframe": "1D"}, {"condition": "oversold"},
                       {"rank_by": "pnl"}, {"text": "en vidia"}, {"status": "open"}, {"focus": "movers"}]
            self._forced[name] = _lcp(self.encode(self.call_text(name, a)) for a in samples)
        return self._forced[name]


# =============================================================================================== Llama 3.2
"""LLAMA TRAINED FORMAT (see FORMAT.md, "Llama 3.2 format"). Llama 3.2 special tokens, own compact renderer
(the HF template is not used: it adds a date line, puts the tool JSON in the user turn and refuses parallel
calls). One function, `render_llama`, builds both the training examples and the inference prompt, so they are
byte-identical by construction; tests/test_format_parity.py checks the token streams too.

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n LLAMA_SYSTEM + compact tool list <|eot_id|>
    user / assistant(spoken answer) pairs for the last <=3 turns
    user       "[state: ...]\\n" + question
    assistant  one JSON object per line, {"name": T, "parameters": {...}}  (several lines = parallel calls)
    ipython    llama_tool_content(model_view), one message per call, in call order
    ...
    assistant  answer template with {slots}
"""

LLAMA_BOS = "<|begin_of_text|>"
LLAMA_EOT = "<|eot_id|>"

LLAMA_SYSTEM = """You are Leaps, a voice trading assistant. Replies are spoken: at most two short sentences.
To get data, reply only with tool calls, one JSON object per line: {"name": "tool", "parameters": {...}}. Put every call you need now on its own line.
Results give judgment words and slot names like {price}, never numbers. In your answer write every number only as a slot copied from the results; never write digits. Spell out numbers the user said (five shares).
Pass company names or tickers as symbol; use resolve_symbol only when unsure which company. draft_order only drafts a ticket the user must confirm; never say an order was placed. If a needed detail is missing, ask one short question. Never give advice or picks."""


def _llama_header(role: str) -> str:
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n"


def llama_tools_text(tools: Sequence[Mapping[str, Any]]) -> str:
    """Compact tool list: `name(arg, opt?=a|b, num:number): description` — about a quarter of the JSON schema."""
    lines = []
    for t in tools:
        f = t.get("function", t)
        props = (f.get("parameters") or {}).get("properties") or {}
        req = set((f.get("parameters") or {}).get("required") or [])
        args = []
        for k, spec in props.items():
            a = k if k in req else k + "?"
            if spec.get("enum"):
                a += "=" + "|".join(spec["enum"])
            elif spec.get("type") in ("number", "integer"):
                a += ":" + spec["type"]
            args.append(a)
        lines.append(f"{f['name']}({', '.join(args)}): {f.get('description', '')}")
    return "Tools:\n" + "\n".join(lines)


def llama_tool_content(view: Mapping[str, Any]) -> str:
    """An ipython message: the model_view without the redundant ok flag (error only when there is one),
    canonical JSON with sorted keys (DB jsonb and live tools order `words` differently)."""
    d = {"tool": view.get("tool"), "words": view.get("words") or {}, "slots": list(view.get("slots") or [])}
    if view.get("error"):
        d["error"] = view["error"]
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def llama_call_line(name: str, args: Mapping[str, Any]) -> str:
    return json.dumps({"name": name, "parameters": dict(args)}, ensure_ascii=False)


def _calls_of(m: Mapping[str, Any]) -> list[tuple[str, dict]]:
    out = []
    for c in m.get("tool_calls") or []:
        fn = c.get("function", c)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        out.append((fn.get("name") or c.get("tool"), dict(args or {})))
    return out


def _tool_view(m: Mapping[str, Any]) -> dict:
    c = m.get("content")
    if isinstance(c, Mapping):
        return dict(c)
    try:
        v = json.loads(c)
        return v if isinstance(v, dict) else {"tool": m.get("name"), "words": {"text": str(c)}, "slots": []}
    except (TypeError, ValueError):
        return {"tool": m.get("name"), "words": {"text": str(c)}, "slots": []}


def render_llama(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None = None,
                 add_generation_prompt: bool = False) -> str:
    """Render OpenAI-shaped messages in the Llama trained format. A leading system message is REPLACED by
    LLAMA_SYSTEM + the compact tool list (the loop and the framework pass the Glimmer SYSTEM_PROMPT)."""
    if tools is None:
        from tools import schema
        tools = schema()
    tools = list(tools)
    out = [LLAMA_BOS, _llama_header("system"), LLAMA_SYSTEM, "\n\n", llama_tools_text(tools), LLAMA_EOT]
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "assistant" and m.get("tool_calls"):
            body = "\n".join(llama_call_line(n, a) for n, a in _calls_of(m))
            out += [_llama_header("assistant"), body, LLAMA_EOT]
        elif role == "tool":
            out += [_llama_header("ipython"), llama_tool_content(_tool_view(m)), LLAMA_EOT]
        else:
            out += [_llama_header("assistant" if role == "assistant" else "user"), str(m.get("content") or ""),
                    LLAMA_EOT]
    if add_generation_prompt:
        out.append(_llama_header("assistant"))
    return "".join(out)


def llama_prefix_text(tools: Sequence[Mapping[str, Any]] | None = None) -> str:
    """The system+tools prefix every prompt starts with (prefilled once and cached)."""
    return render_llama([], tools)


def grouped_trace_messages(prefix: Sequence[Mapping[str, Any]], step_groups: Sequence[Sequence[Mapping[str, Any]]],
                           answer: str | None) -> list[dict]:
    """prefix (system, history, user) + one assistant message per GROUP of parallel calls, its tool results
    after it in call order, then the answer. Steps are {"tool", "args", "view"}."""
    msgs = [dict(m) for m in prefix]
    k = 0
    for group in step_groups:
        calls, results = [], []
        for st in group:
            k += 1
            cid = f"call_{k}"
            calls.append({"id": cid, "type": "function",
                          "function": {"name": st["tool"], "arguments": dict(st.get("args") or {})}})
            results.append({"role": "tool", "tool_call_id": cid, "name": st["tool"],
                            "content": tool_content(st["view"])})
        msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
        msgs.extend(results)
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def regroup_calls(messages: Sequence[Mapping[str, Any]], sizes: Sequence[int]) -> list[dict]:
    """A trace eval that appends each call as its own assistant+tool pair. Merge them back into the
    groups the model actually emitted (`sizes`: calls per decision, in order) so the prompt continues the
    model's own stream. Only the pairs after the last user message are touched."""
    msgs = [dict(m) for m in messages]
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    head, tail = msgs[:last_user + 1], msgs[last_user + 1:]
    out, i = [], 0
    for n in sizes:
        pairs = []
        while len(pairs) < n and i + 1 < len(tail) and tail[i].get("role") == "assistant" \
                and len(tail[i].get("tool_calls") or []) == 1 and tail[i + 1].get("role") == "tool":
            pairs.append((tail[i], tail[i + 1]))
            i += 2
        if not pairs:
            break
        out.append({"role": "assistant", "content": "",
                    "tool_calls": [c for a, _ in pairs for c in a["tool_calls"]]})
        out.extend(t for _, t in pairs)
    return head + out + tail[i:]


def llama_examples(messages: Sequence[Mapping[str, Any]], tokenizer,
                   tools: Sequence[Mapping[str, Any]] | None = None) -> list[tuple[list[int], int]]:
    """Training examples for one trace: (tokens, prompt_len) per assistant turn after the last user message
    (history answers are context, never targets). Only tokens[prompt_len:] — the assistant body + <|eot_id|>
    — are trained. The prompt is exactly what inference feeds (`LlamaPromptBuilder.tokens_for`)."""
    pb = LlamaPromptBuilder(tokenizer)
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    out = []
    for i, m in enumerate(messages):
        if i > last_user and m.get("role") == "assistant":
            prompt = pb.tokens_for(messages[:i], tools)
            full = pb.encode(render_llama(messages[:i + 1], tools))
            if full[:len(prompt)] != prompt:
                raise ValueError("prompt is not a token prefix of the trained turn")
            out.append((full, len(prompt)))
    return out


class LlamaPromptBuilder:
    """Token streams for the Llama format (tokenizer only, no model)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.eot = tokenizer.convert_tokens_to_ids(LLAMA_EOT)
        self.eom = tokenizer.convert_tokens_to_ids("<|eom_id|>")
        self.eos_ids = {self.eot, self.eom, tokenizer.convert_tokens_to_ids("<|end_of_text|>")}
        self._prefix: dict[str, list[int]] = {}

    def encode(self, s: str) -> list[int]:
        return list(self.tok.encode(s, add_special_tokens=False))

    def prefix_tokens(self, tools=None) -> list[int]:
        key = json.dumps(tools, sort_keys=True) if tools is not None else ""
        if key not in self._prefix:
            self._prefix[key] = self.encode(llama_prefix_text(tools))
        return self._prefix[key]

    def tokens_for(self, messages, tools=None) -> list[int]:
        """Prompt for the next decision: system+tools prefix tokens + the rest (the prefix ends in a special
        token, so splitting the encode there is exact)."""
        full = render_llama(messages, tools, add_generation_prompt=True)
        pre = llama_prefix_text(tools)
        assert full.startswith(pre)
        return self.prefix_tokens(tools) + self.encode(full[len(pre):])

    def reset(self) -> None:          # API parity with PromptBuilder (stateless here)
        pass


def step_messages_split(prefix: Sequence[Mapping[str, Any]], step_groups: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict]:
    """A flat trace eval's shape: every call its own assistant+tool pair (groups flattened). Test helper."""
    return grouped_trace_messages(prefix, [[st] for g in step_groups for st in g], None)


def framework_examples(prefix: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], answer: str,
                       tools: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Training renderer hook for a trace in step form (loadable from this one file: it imports nothing
    beyond the stdlib). Steps are `{"call": {"tool", "arguments"}, "result": model_view[, "turn"]}`;
    steps sharing a `turn` are one decision (parallel calls). Returns one `{"prompt", "completion"}` per
    assistant decision, rendered by `render_llama` exactly as inference renders it."""
    groups: list[list[dict]] = []
    last: Any = object()
    for st in steps:
        item = {"tool": st["call"]["tool"], "args": dict(st["call"].get("arguments") or {}),
                "view": dict(st.get("result") or {})}
        turn = st.get("turn")
        if groups and turn is not None and turn == last:
            groups[-1].append(item)
        else:
            groups.append([item])
        last = turn
    msgs = grouped_trace_messages(prefix, groups, answer)
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    out = []
    for i, m in enumerate(msgs):
        if i > last_user and m.get("role") == "assistant":
            prompt = render_llama(msgs[:i], tools, add_generation_prompt=True)
            full = render_llama(msgs[:i + 1], tools)
            assert full.startswith(prompt)
            out.append({"prompt": prompt, "completion": full[len(prompt):]})
    return out
