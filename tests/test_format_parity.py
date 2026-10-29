"""Trained-format parity: the token stream trained-mode inference builds == the chat template's rendering
of the same trace (what the agent framework trains on). Needs only the tokenizer, never the model."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent import paths, run_turn
from agent.format import TEMPLATE_KWARGS, PromptBuilder, assistant_call, render_trace, tool_content
from mockmarket import World
from tools import schema

MODEL = paths.GLIMMER_DIR
pytestmark = pytest.mark.skipif(not (MODEL / "tokenizer.json").exists(), reason="Glimmer tokenizer not present")


@pytest.fixture(scope="module")
def tok():
    from mlx_lm.utils import load_tokenizer
    return load_tokenizer(MODEL)


def framework_tokens(tok, messages, gen_prompt=False):
    """Exactly how the trainer tokenizes a chat example (tokens and mask offset)."""
    return list(tok.apply_chat_template(messages, tools=schema(), add_generation_prompt=gen_prompt,
                                        return_dict=False, **TEMPLATE_KWARGS))


class SimTrained:
    """Plays a perfectly trained model: emits the template's own tokens for the scripted next message and
    checks each emission against the trained-mode decoding constraints. Builds the prompt with the same
    PromptBuilder the GlimmerBackend uses."""

    def __init__(self, tok, script):
        self.tok, self.pb, self.script = tok, PromptBuilder(tok), list(script)
        self.prompts: list[list[int]] = []
        self.expected: list[dict] = []         # canonical trace messages, built independently of the loop
        self.stream: list[int] = []

    def decide(self, messages, tools):
        item = self.script.pop(0)
        prompt = self.pb.tokens_for(messages, tools)
        self.prompts.append(prompt)
        if not self.expected:
            self.expected = [dict(m) for m in messages]
        else:                                  # loop appended tool results since our last call
            self.expected += [dict(m) for m in messages[len(self._seen):]]
        if "tool_calls" in item:
            c = item["tool_calls"][0]
            nxt = assistant_call(f"call_{sum(1 for m in self.expected if m['role'] == 'tool') + 1}",
                                 c["tool"], c["args"])
        else:
            nxt = {"role": "assistant", "content": item["answer"]}
        p_str = render_trace(self.expected, self.tok, add_generation_prompt=True)
        f_str = render_trace(self.expected + [nxt], self.tok)
        assert f_str.startswith(p_str)
        gen = self.pb.encode(f_str[len(p_str):])
        # trained-mode constraints: header is one of to=<tool>/to=user; tool body starts with the forced chunk
        names = [t["function"]["name"] for t in tools] + ["user"]
        heads = self.pb.header_seqs(names)
        head = next((h for h in heads if gen[:len(h)] == h), None)
        assert head is not None, self.tok.decode(gen[:8])
        if "tool_calls" in item:
            forced = self.pb.forced_chunk(c["tool"])
            assert len(forced) >= 5
            assert gen[len(head):len(head) + len(forced)] == forced
        assert gen[-1] == self.pb.eot
        self.expected.append(nxt)
        self._seen = list(messages) + [nxt]
        self.stream = prompt + gen
        self.pb.commit(messages, prompt + gen, ended=True)
        return item


def run(tok, q, script, history=None, world=None):
    be = SimTrained(tok, script)
    res = run_turn(q, history or [], world or World(7), be)
    return be, res


TRACES = [
    ("How's my NVDA position and is it near support?",
     [{"tool_calls": [{"tool": "get_position", "args": {"symbol": "NVDA"}}]},
      {"tool_calls": [{"tool": "get_levels", "args": {"symbol": "NVDA"}}]},
      {"answer": "You're {pnl} on Nvidia, and it's {dist_support} above support at {support}."}]),
    ("What if Nvidia drops ten percent?",
     [{"tool_calls": [{"tool": "what_if", "args": {"symbol": "NVDA", "move_pct": -10}}]},
      {"answer": "That would be {pnl_change}, a moderate hit."}]),
    ("Hi there!", [{"answer": "Hi! Ask me about a stock, your positions, or the market."}]),
    ("What's the five minute RSI on AMD?",
     [{"tool_calls": [{"tool": "get_indicators", "args": {"symbol": "AMD", "timeframe": "5m", "indicator": "rsi"}}]},
      {"answer": "AMD's five minute RSI is {rsi}."}]),
]


@pytest.mark.parametrize("q,script", TRACES)
def test_stream_equals_template(tok, q, script):
    be, res = run(tok, q, script)
    assert not res.fallback
    final = framework_tokens(tok, be.expected)
    assert be.stream == final                                       # whole trace, token for token
    pb = PromptBuilder(tok)
    prefix = pb.prefix_tokens(be.expected, schema())
    assert final[:len(prefix)] == prefix                            # cached system+tools prefix
    # every decision point == the framework's mask offset for that assistant turn
    asst = [i for i, m in enumerate(be.expected) if m["role"] == "assistant"]
    assert len(asst) == len(be.prompts)
    for i, prompt in zip(asst, be.prompts):
        assert prompt == framework_tokens(tok, be.expected[:i], gen_prompt=True)
    text = tok.decode(final, skip_special_tokens=False)
    assert "to=self" not in text and "Reasoning strength: low." in text
    for m in be.expected:
        if m["role"] == "tool":
            assert m["content"] == tool_content(__import__("json").loads(m["content"]))   # canonical JSON


def test_follow_up_with_history(tok):
    w = World(7)
    _, r1 = run(tok, *TRACES[0], world=w)
    be, r2 = run(tok, "And its RSI?", [
        {"tool_calls": [{"tool": "get_indicators", "args": {"symbol": "NVDA", "indicator": "rsi"}}]},
        {"answer": "RSI is {rsi}, neutral."}], history=[r1], world=w)
    assert be.stream == framework_tokens(tok, be.expected)
    assert be.expected[2] == {"role": "assistant", "content": r1.spoken}
    assert be.expected[3]["content"].startswith("[state: last_symbol=NVDA, last_topic=levels]\n")


def test_forced_chunk_is_prefix_for_every_tool(tok):
    pb = PromptBuilder(tok)
    for t in schema():
        name = t["function"]["name"]
        forced = pb.forced_chunk(name)
        for args in ({}, {"symbol": "TSLA"}, {"move_pct": -5, "symbol": "ALL"}, {"qty": 3, "symbol": "MSFT"}):
            full = pb.encode(pb.call_text(name, args))
            assert full[:len(forced)] == forced
        # call_text reproduces the template's own rendering
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                assistant_call("call_1", name, {"symbol": "NVDA"})]
        assert pb.call_text(name, {"symbol": "NVDA"}) in render_trace(msgs, tok)


# ============================================================================ Llama 3.2 format parity
LLAMA = paths.LLAMA_DIR
needs_llama = pytest.mark.skipif(not (LLAMA / "tokenizer.json").exists(), reason="Llama tokenizer not present")


@pytest.fixture(scope="module")
def ltok():
    from mlx_lm.utils import load_tokenizer
    return load_tokenizer(LLAMA)


class SimLlama:
    """Plays a perfectly trained Llama: for each decision it emits exactly the tokens the training render has
    after the prompt (so the stream is what a trained model would produce), built with the SAME
    LlamaPromptBuilder the LlamaBackend uses."""

    def __init__(self, tok, script):
        from agent.format import LlamaPromptBuilder
        self.tok, self.pb, self.script = tok, LlamaPromptBuilder(tok), list(script)
        self.prompts: list[list[int]] = []
        self.seen: list[list[dict]] = []
        self.stream: list[int] = []

    def decide(self, messages, tools):
        from agent.format import render_llama
        item = self.script.pop(0)
        prompt = self.pb.tokens_for(messages, tools)
        self.prompts.append(prompt)
        self.seen.append([dict(m) for m in messages])
        if "tool_calls" in item:
            nxt = {"role": "assistant", "content": "", "tool_calls": [
                {"id": "x", "type": "function", "function": {"name": c["tool"], "arguments": c["args"]}}
                for c in item["tool_calls"]]}
        else:
            nxt = {"role": "assistant", "content": item["answer"]}
        full = self.pb.encode(render_llama(list(messages) + [nxt], tools))
        assert full[:len(prompt)] == prompt                       # prompt is a token prefix of the trained turn
        gen = full[len(prompt):]
        assert gen[-1] == self.pb.eot
        self.stream = full
        return item


def run_llama(tok, q, script, history=None, world=None):
    be = SimLlama(tok, script)
    res = run_turn(q, history or [], world or World(7), be)
    return be, res


def groups_from(messages, script):
    """The trace as the TRAINING side stores it: groups of {tool, args, view} (views = the tool messages)."""
    import json as _j
    tools_msgs = [m for m in messages if m["role"] == "tool"]
    groups, k = [], 0
    for item in script:
        if "tool_calls" in item:
            g = []
            for c in item["tool_calls"]:
                g.append({"tool": c["tool"], "args": c["args"], "view": _j.loads(tools_msgs[k]["content"])})
                k += 1
            groups.append(g)
    return groups


LLAMA_TRACES = [
    ("How's my NVDA position and is it near support?",                       # parallel: one decision, two calls
     [{"tool_calls": [{"tool": "get_position", "args": {"symbol": "NVDA"}},
                      {"tool": "get_levels", "args": {"symbol": "NVDA"}}]},
      {"answer": "You're {pnl} on Nvidia, and it's {dist_support} above support at {support}."}]),
    ("How's my NVDA position and is it near support?",                       # sequential: two decisions
     [{"tool_calls": [{"tool": "get_position", "args": {"symbol": "NVDA"}}]},
      {"tool_calls": [{"tool": "get_levels", "args": {"symbol": "NVDA"}}]},
      {"answer": "You're {pnl} on Nvidia, near support at {support}."}]),
    ("What if Nvidia drops ten percent?",
     [{"tool_calls": [{"tool": "what_if", "args": {"symbol": "NVDA", "move_pct": -10}}]},
      {"answer": "That would be {pnl_change}, a moderate hit."}]),
    ("Hi there!", [{"answer": "Hi! Ask me about a stock, your positions, or the market."}]),
]


@needs_llama
@pytest.mark.parametrize("q,script", LLAMA_TRACES)
def test_llama_stream_equals_training_examples(ltok, q, script):
    from agent.format import LLAMA_SYSTEM, grouped_trace_messages, llama_examples, render_llama
    from agent.loop import build_messages
    be, res = run_llama(ltok, q, [dict(x) for x in script])
    assert not res.fallback
    final_msgs = be.seen[-1]
    groups = groups_from(final_msgs, script)
    # training side: built independently from the stored trace (prefix + groups + answer)
    train_msgs = grouped_trace_messages(build_messages(q, None), groups, script[-1]["answer"])
    ex = llama_examples(train_msgs, ltok)
    assert len(ex) == len(be.prompts) == len(script)
    for (tokens, off), prompt in zip(ex, be.prompts):
        assert tokens[:off] == prompt                               # decision prompt == training mask offset
    assert ex[-1][0] == be.stream                                   # whole trace, token for token
    text = render_llama(train_msgs)
    assert text.count("<|begin_of_text|>") == 1 and LLAMA_SYSTEM in text
    assert "Cutting Knowledge" not in text and "Today Date" not in text
    assert "You are Leaps, a voice trading assistant. Answers are spoken aloud" not in text   # Glimmer prompt replaced
    for t in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        assert ltok.convert_tokens_to_ids(t) in ex[-1][0]
    assert not any(ch.isdigit() for m in final_msgs if m["role"] == "tool"
                   for ch in __import__("json").dumps(__import__("json").loads(m["content"])["slots"]))


@needs_llama
def test_llama_follow_up_with_history(ltok):
    from agent.format import grouped_trace_messages, llama_examples
    from agent.loop import build_messages
    w = World(7)
    _, r1 = run_llama(ltok, *LLAMA_TRACES[0], world=w)
    script = [{"tool_calls": [{"tool": "get_indicators", "args": {"symbol": "NVDA", "indicator": "rsi"}}]},
              {"answer": "RSI is {rsi}, neutral."}]
    be, r2 = run_llama(ltok, "And its RSI?", [dict(x) for x in script], history=[r1], world=w)
    msgs = build_messages("And its RSI?", [r1.turn])
    assert msgs[2] == {"role": "assistant", "content": r1.spoken}
    ex = llama_examples(grouped_trace_messages(msgs, groups_from(be.seen[-1], script), script[-1]["answer"]), ltok)
    assert ex[-1][0] == be.stream and [t[:o] for t, o in ex] == be.prompts
    assert len(ex) == 2                                             # the history answer is never a target


@needs_llama
def test_llama_prefix_cache_and_regroup(ltok):
    from agent.format import LlamaPromptBuilder, grouped_trace_messages, regroup_calls, step_messages_split
    pb = LlamaPromptBuilder(ltok)
    pre = pb.prefix_tokens(schema())
    be, _ = run_llama(ltok, *LLAMA_TRACES[0])
    for p in be.prompts:
        assert p[:len(pre)] == pre                                  # every prompt starts with the cached prefix
    assert len(pre) < 700                                           # compact: a 1B model degrades with long prompts
    # the framework eval splits parallel calls into one assistant+tool pair each; regroup restores the stream
    groups = groups_from(be.seen[-1], LLAMA_TRACES[0][1])
    from agent.loop import build_messages
    prefix = build_messages(LLAMA_TRACES[0][0], None)
    grouped = grouped_trace_messages(prefix, groups, None)
    split = step_messages_split(prefix, groups)
    assert pb.tokens_for(regroup_calls(split, [2]), schema()) == pb.tokens_for(grouped, schema())
    assert pb.tokens_for(split, schema()) != pb.tokens_for(grouped, schema())


def test_parse_llama_calls_and_answers():
    from agent.backends import parse_llama
    from agent.format import llama_call_line
    two = llama_call_line("get_quote", {"symbol": "NVDA"}) + "\n" + llama_call_line("what_if", {"symbol": "ALL", "move_pct": -10})
    assert parse_llama(two) == {"tool_calls": [{"tool": "get_quote", "args": {"symbol": "NVDA"}},
                                               {"tool": "what_if", "args": {"symbol": "ALL", "move_pct": -10}}]}
    assert parse_llama('<|python_tag|>{"name": "get_market", "parameters": {}}') == \
        {"tool_calls": [{"tool": "get_market", "args": {}}]}
    assert parse_llama("Nvidia is at {price}.") == {"answer": "Nvidia is at {price}."}
    assert "error" in parse_llama('{"name": "get_quote", "parameters": {"symbol"')
    assert "error" in parse_llama("")


@needs_llama
def test_framework_renderer_hook_equals_llama_examples(ltok):
    """agent.format.framework_examples (the framework's training.renderer) == llama_examples, token for token."""
    from agent.format import LlamaPromptBuilder, framework_examples, grouped_trace_messages, llama_examples
    from agent.loop import build_messages
    q, script = LLAMA_TRACES[0]
    be, _ = run_llama(ltok, q, [dict(x) for x in script])
    groups = groups_from(be.seen[-1], script)
    prefix = build_messages(q, None)
    steps = [{"call": {"tool": s["tool"], "arguments": s["args"]}, "result": s["view"], "turn": k}
             for k, g in enumerate(groups, 1) for s in g]
    fx = framework_examples(prefix, steps, script[-1]["answer"], schema())
    ref = llama_examples(grouped_trace_messages(prefix, groups, script[-1]["answer"]), ltok)
    pb = LlamaPromptBuilder(ltok)
    assert [(pb.encode(e["prompt"] + e["completion"]), len(pb.encode(e["prompt"]))) for e in fx] == ref
