"""training/: filters keep good traces and reject bad ones; the renderer gives one example per decision."""
import json
from pathlib import Path

import pytest

from agent import paths
from mockmarket import World
from training.filters import check_trace

EXAMPLES = Path(__file__).resolve().parents[1] / "training" / "example_traces.jsonl"


def traces():
    return [json.loads(x) for x in EXAMPLES.read_text().splitlines() if x.strip()]


def test_example_traces_pass_filters():
    for tr in traces():
        assert all(check_trace(tr, world=World(7)).values()), tr["id"]


def test_filters_reject_digits_unknown_slots_and_bad_tools():
    tr = traces()[0]
    assert not check_trace(dict(tr, answer="You're up 3400 dollars."), world=World(7))["contract"]
    assert not check_trace(dict(tr, answer="It is {nope}."), world=World(7))["contract"]
    bad = dict(tr, steps=[{"call": {"tool": "send_order", "arguments": {}}, "result": tr["steps"][0]["result"]}])
    assert not check_trace(bad, world=World(7))["schema"]
    gold = [{"tool": "get_position", "args": {"symbol": "NVDA"}}]
    assert not check_trace(tr, gold, World(7))["gold"]          # gold wants one call, the trace made two


@pytest.mark.skipif(not (paths.LLAMA_DIR / "tokenizer.json").exists(), reason="Llama tokenizer not present")
def test_render_one_example_per_decision():
    from mlx_lm.utils import load_tokenizer
    from training.render import examples_for
    tok = load_tokenizer(paths.LLAMA_DIR)
    n = [len(examples_for(tr, tok)) for tr in traces()]
    assert n == [2, 2, 1]                     # parallel calls + answer; follow-up call + answer; no-tool answer
    tokens, offset = examples_for(traces()[1], tok)[0]
    text = tok.decode(tokens[:offset])
    assert "[state: last_symbol=NVDA, last_topic=levels]\nWhat about its RSI?" in text
    assert '"name": "get_indicators"' in tok.decode(tokens[offset:])
