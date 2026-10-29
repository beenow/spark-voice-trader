"""LlamaBackend's decode/cache logic and GlimmerBackend's adapter hot-swap, on a TINY random Llama on the CPU
(no real weights, no GPU). The tiny model shares the Llama 3.2 tokenizer so prompts are real."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent import paths
from agent.format import LlamaPromptBuilder
from tools import schema

LLAMA = paths.LLAMA_DIR
pytestmark = pytest.mark.skipif(not (LLAMA / "tokenizer.json").exists(), reason="Llama tokenizer not present")


@pytest.fixture(scope="module")
def cpu():
    import mlx.core as mx
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield mx
    mx.set_default_device(prev)


def tiny(mx, vocab):
    from mlx_lm.models.llama import Model, ModelArgs
    mx.random.seed(0)
    m = Model(ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                        num_attention_heads=2, num_key_value_heads=1, rms_norm_eps=1e-5, vocab_size=vocab))
    mx.eval(m.parameters())
    return m


@pytest.fixture(scope="module")
def tok():
    from mlx_lm.utils import load_tokenizer
    return load_tokenizer(LLAMA)


def make_llama(mx, tok, model):
    from agent.backends import LlamaBackend
    be = LlamaBackend.__new__(LlamaBackend)
    be.model, be.tokenizer, be.adapter, be.mode, be.model_dir = model, tok, None, "base", "tiny"
    be.pb = LlamaPromptBuilder(tok)
    be.eos_ids = set(be.pb.eos_ids)
    be.max_tokens, be.answer_max_tokens, be.keep_cache, be.verbose = 12, 70, True, False
    be.stats, be._cache, be._cache_tokens, be.warm_s = [], None, [], 0.0
    return be


def test_llama_cache_reuse_matches_fresh(cpu, tok):
    mx = cpu
    model = tiny(mx, len(tok))
    be = make_llama(mx, tok, model)
    m1 = [{"role": "system", "content": "x"}, {"role": "user", "content": "How's Nvidia?"}]
    out1 = be.decide(m1, schema())
    assert be.stats[-1]["reused_tokens"] == 0
    m2 = m1 + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_quote", "arguments": {"symbol": "NVDA"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "get_quote",
         "content": '{"error": null, "ok": true, "slots": ["{price}"], "tool": "get_quote", "words": {"move": "small"}}'}]
    out2 = be.decide(m2, schema())
    st = be.stats[-1]
    prefix = len(be.pb.prefix_tokens(schema()))
    assert st["reused_tokens"] >= prefix and st["prefilled_tokens"] < st["prompt_tokens"] - prefix
    fresh = make_llama(mx, tok, model)
    assert fresh.decide(m2, schema())["raw"] == out2["raw"]        # cached == uncached, token for token
    # a new turn trims back to the shared prefix
    out3 = be.decide(m1[:1] + [{"role": "user", "content": "And Tesla?"}], schema())
    assert be.stats[-1]["reused_tokens"] >= prefix
    assert all(k in out1 for k in ("raw",)) and "raw" in out3


def test_glimmer_adapter_hot_swap(cpu, tok, tmp_path):
    """activate() swaps LoRA weights on one loaded model: base == all-zero LoRA == the plain model; each
    adapter == that adapter loaded directly; per-adapter prefix snapshots survive switching."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from types import SimpleNamespace

    from agent.backends import GlimmerBackend
    vocab = 512
    plain = tiny(mx, vocab)
    x = mx.array([[1, 2, 3, 4, 5]])
    ref_base = plain(x)
    cfg = {"rank": 4, "scale": 2.0, "dropout": 0.0, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
    adapters = {}
    for n, seed in (("a", 1), ("b", 2)):
        m = tiny(mx, vocab)
        m.freeze()
        linear_to_lora_layers(m, 2, cfg)
        mx.random.seed(seed)
        w = {k: mx.random.normal(v.shape) * 0.5 for k, v in tree_flatten(m.trainable_parameters())}
        d = tmp_path / n
        d.mkdir()
        mx.save_safetensors(str(d / "adapters.safetensors"), w)
        m.load_weights(list(w.items()), strict=False)
        adapters[n] = (str(d), m(x))
    model = tiny(mx, vocab)
    model.freeze()
    linear_to_lora_layers(model, 2, cfg)
    model.load_weights(list(mx.load(adapters["a"][0] + "/adapters.safetensors").items()), strict=False)
    be = GlimmerBackend.__new__(GlimmerBackend)
    be.model, be.adapter, be.mode = model, adapters["a"][0], "trained"
    be.pb = SimpleNamespace(reset=lambda: None)
    be._cache, be._cache_tokens = None, []
    be._prefix_key, be._prefix_tokens, be._prefix_state = "ka", [1], "state-a"
    close = lambda p, q: float(mx.abs(p - q).max()) < 1e-4  # noqa: E731
    assert close(model(x), adapters["a"][1])
    be.activate(None, "base")
    assert close(model(x), ref_base) and be.mode == "base" and be.exclude_tools == ("resolve_symbol",)
    assert be._prefix_state is None                                 # base has its own prefix (not a's)
    be._prefix_key, be._prefix_tokens, be._prefix_state = "k0", [0], "state-0"
    be.activate(adapters["b"][0], "trained")
    assert close(model(x), adapters["b"][1]) and be.exclude_tools == ()
    be.activate(adapters["a"][0], "trained")
    assert close(model(x), adapters["a"][1]) and be._prefix_state == "state-a"
    be.activate(None, "base")
    assert close(model(x), ref_base) and be._prefix_state == "state-0"
    bad = tmp_path / "bad"
    bad.mkdir()
    mx.save_safetensors(str(bad / "adapters.safetensors"), {"model.layers.0.self_attn.q_proj.lora_a": mx.zeros((3, 3))})
    with pytest.raises(RuntimeError):
        be.activate(str(bad), "trained")
