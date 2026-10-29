"""The synthetic eval sample and its scorer: the oracle (gold replay) passes every check, wrong calls fail."""
import json

from eval.run_eval import SAMPLE, run_rows
from eval.score import CHECKS, score_turn, summarize


def test_sample_is_synthetic_and_well_formed():
    data = json.loads(SAMPLE.read_text())
    rows = data["rows"]
    assert 15 <= len(rows) <= 30 and len({r["id"] for r in rows}) == len(rows)
    from tools import TOOL_SCHEMAS
    assert all(g["tool"] in TOOL_SCHEMAS for r in rows for g in r["gold"])


def test_oracle_passes_every_check():
    data = json.loads(SAMPLE.read_text())
    s = summarize(run_rows(data["rows"], data["seed"]))
    assert s["overall"] == {k: 100.0 for k in CHECKS}


def test_scorer_rejects_wrong_tool_args_and_digits():
    gold = [{"tool": "get_quote", "args": {"symbol": "MSFT"}}]
    ok = score_turn(gold, [{"tool": "get_quote", "args": {"symbol": "msft"}}], "Microsoft is at {price}.", {"price"})
    assert all(ok.values())
    bad = score_turn(gold, [{"tool": "get_quote", "args": {"symbol": "META"}}], "It is at 412 dollars.", {"price"})
    assert bad["tool_set"] and not bad["arguments"] and not bad["contract"]
    none = score_turn([], [{"tool": "get_portfolio", "args": {}}], "Here.", set())
    assert not none["action"] and not none["tool_set"]
