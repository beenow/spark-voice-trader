"""Model backends. Protocol: decide(messages, tools_schema) -> {"tool_calls":[{"tool","args"}]} | {"answer": str}
| {"error": str, "raw": str}."""
from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from agent import paths

MODELS = paths.MODELS_DIR
DEFAULT_GLIMMER = paths.GLIMMER_DIR
DEFAULT_ADAPTER = paths.ADAPTERS_DIR / "latest"
DEFAULT_LLAMA = paths.LLAMA_DIR
DEFAULT_LLAMA_ADAPTER = paths.ADAPTERS_DIR / "llama-1b-loop"
DEFAULT_REGISTRY = paths.REGISTRY


class ScriptedBackend:
    """Replays a script. Each item is a response dict or a callable(messages) -> response dict."""

    def __init__(self, script: list[dict | Callable[[list[dict]], dict]]):
        self.script = list(script)
        self.seen: list[list[dict]] = []
        self.stats: list[dict] = []

    def decide(self, messages: list[dict], tools_schema=None) -> dict:
        self.seen.append(copy.deepcopy(messages))
        if not self.script:
            return {"error": "script exhausted"}
        item = self.script.pop(0)
        return copy.deepcopy(item(messages) if callable(item) else item)


# ---------------------------------------------------------------- Glimmer output parsing
_INVOKE = re.compile(r"<atem:invoke\b[^>]*?\bname=\"(?P<name>[^\"]+)\">(?P<body>.*?)(?:</atem:invoke>|$)", re.S)
_PARAM = re.compile(r"<atem:parameter\b[^>]*?\bname=\"(?P<name>[^\"]+)\"[^>]*?>(?P<value>.*?)</atem:parameter>",
                    re.S)
_SEG = re.compile(r"^(?P<header>[^<]*?)<\|message\|>(?P<body>.*?)(?:<\|eot\|>|<\|eom\|>|<\|end_of_text\|>|$)",
                  re.S)


def _pval(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw.strip()


def parse_glimmer(text: str) -> dict:
    """Parse Glimmer's raw completion (decoded WITH special tokens, starting after '<|start|>assistant')."""
    segments = [s for s in text.split("<|start|>assistant")]
    calls, answer, saw_atem = [], None, "<atem:" in text
    for seg in segments:
        m = _SEG.match(seg)
        if not m:
            if seg.strip() and "<|message|>" not in seg and not calls:
                # no header at all: treat as plain answer text unless it carries a tool call
                if "<atem:" not in seg:
                    answer = re.sub(r"<\|[a-z_]+\|>", "", seg).strip() or answer
            continue
        header, body = m.group("header").strip(), m.group("body")
        recipient = header[3:].strip() if header.startswith("to=") else (header or "user")
        if recipient == "self":
            continue                                       # reasoning channel
        invokes = list(_INVOKE.finditer(body))
        if invokes:
            for inv in invokes:
                args = {p.group("name"): _pval(p.group("value")) for p in _PARAM.finditer(inv.group("body"))}
                calls.append({"tool": inv.group("name").split(".")[-1], "args": args})
            continue
        if recipient not in ("user", "") and "<atem:" not in body:
            # 'to=tool' with a JSON body instead of atem XML
            try:
                args = json.loads(body.strip() or "{}")
                if isinstance(args, dict):
                    calls.append({"tool": recipient.split(".")[-1], "args": args})
                    continue
            except ValueError:
                pass
        if "<atem:" not in body:
            answer = body.strip()
    if calls:
        return {"tool_calls": calls}
    if saw_atem:
        return {"error": "could not parse tool call", "raw": text}
    if answer:
        return {"answer": answer}
    return {"error": "empty response", "raw": text}


# ---------------------------------------------------------------- Glimmer (mlx_lm)
class GlimmerBackend:
    """Muse Glimmer via mlx_lm, using the model's own chat template + <atem:function_calls> tool format.

    mode="trained" (default when an adapter is given) — the CANONICAL TRAINED FORMAT (agent/format.py):
      assistant headers constrained to ' to=<tool>' / ' to=user'; no forced first tool; no reasoning stub;
      after a tool header the template-invariant '<atem:function_calls>...' prefix is forced; after
      '</atem:function_calls>' only <|eot|> (done) or <|eom|> (another call) may follow.
    mode="base" — honest best-effort baseline for the untrained model: same constraints plus a short
      pre-written reasoning message standing in for the model's own slow `to=self` channel, and
      resolve_symbol hidden. `force_first_tool` (default False) makes step one a tool call.
    Both: KV cache kept within a turn; system+tools prefix prefilled once at load and restored per turn;
    spoken answers stop after two sentences / answer_max_tokens.
    """

    CTL = ("Rewrite your answer", "No more tool calls", "Error: your last message")

    def __init__(self, model_dir: str | os.PathLike = DEFAULT_GLIMMER, adapter: str | None = None,
                 mode: str | None = None, max_tokens: int = 200, keep_cache: bool = True, verbose: bool = False,
                 answer_max_tokens: int = 70, warm: bool = True, force_first_tool: bool | None = None,
                 force_boilerplate: bool = True, think: bool = False):
        import mlx.core as mx  # noqa: F401
        from mlx_lm import load

        from .format import PromptBuilder

        self.mode = mode or ("trained" if adapter else "base")
        if self.mode not in ("base", "trained"):
            raise ValueError("mode must be 'base' or 'trained'")
        self.model_dir = str(model_dir)
        self.adapter = adapter
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(self.model_dir, adapter_path=adapter)
        self.load_s = time.perf_counter() - t0
        self.max_tokens = max_tokens
        self.keep_cache = keep_cache
        self.verbose = verbose
        self.think = think                       # True: unconstrained (model may reason) — diagnostics only
        self.answer_max_tokens = answer_max_tokens
        self.force_first_tool = bool(force_first_tool) if force_first_tool is not None else False
        self.force_boilerplate = force_boilerplate
        self.canned_thought = self.mode == "base"
        self.exclude_tools = ("resolve_symbol",) if self.mode == "base" else ()
        self.pb = PromptBuilder(self.tokenizer)
        tk = self.tokenizer
        self.eot, self.eom = self.pb.eot, self.pb.eom
        self.eos_ids = set(getattr(tk, "eos_token_ids", None) or []) | {self.eot,
                                                                          tk.convert_tokens_to_ids("<|end_of_text|>")}
        self._msg_id = self.pb.msg
        self.stats: list[dict] = []
        self._cache = None
        self._cache_tokens: list[int] = []
        self._prefix_key = None
        self._prefix_tokens: list[int] | None = None
        self._prefix_state = None
        self.warm_s = 0.0
        if warm:
            self.warmup()

    def warmup(self) -> None:
        """Prefill + snapshot the system+tools prefix and compile kernels, so the first turn is fast."""
        from tools import schema
        from .loop import SYSTEM_PROMPT
        t0 = time.perf_counter()
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Hi"}]
        saved = self.max_tokens
        self.max_tokens = 4
        try:
            self.decide(msgs, schema())
        finally:
            self.max_tokens = saved
            self.stats.clear()
            self._cache, self._cache_tokens = None, []
            self.pb.reset()
        self.warm_s = time.perf_counter() - t0

    # -- adapter hot-swap (several LoRA adapters on ONE loaded base) --------------
    def _adapter_file(self, adapter) -> Path:
        return Path(adapter) / "adapters.safetensors"

    def activate(self, adapter: str | None, mode: str) -> None:
        """Make `adapter` (None = base weights) and `mode` current. The model must have been loaded with SOME
        adapter (so its LoRA layers exist); adapters must share that LoRA config. Base = all LoRA params zero,
        which adds exactly zero to every layer. Each adapter keeps its own system-prefix KV snapshot (the
        prefix KV depends on the weights), so switching back is instant."""
        import mlx.core as mx
        if mode not in ("base", "trained"):
            raise ValueError("mode must be 'base' or 'trained'")
        key = str(adapter) if adapter else None
        if not hasattr(self, "_active"):
            self._active = str(self.adapter) if self.adapter else None
            self._prefix_store: dict = {}
            self._lora_cache: dict = {}
        if key != self._active:
            if self.adapter is None:
                raise RuntimeError("this Glimmer was loaded without an adapter; it cannot swap one in")
            if key not in self._lora_cache:
                if key is None:
                    ref = self._lora_cache.get(str(self.adapter)) or mx.load(str(self._adapter_file(self.adapter)))
                    self._lora_cache[None] = {k: mx.zeros_like(v) for k, v in ref.items()}
                else:
                    self._lora_cache[key] = mx.load(str(self._adapter_file(key)))
                if self._active is not None and self._active not in self._lora_cache:
                    self._lora_cache[self._active] = mx.load(str(self._adapter_file(self._active)))
            new = self._lora_cache[key]
            from mlx.utils import tree_flatten
            have = {k: v.shape for k, v in tree_flatten(self.model.parameters()) if "lora_" in k}
            bad = [k for k, v in new.items() if have.get(k) != v.shape]
            if bad:
                raise RuntimeError(f"adapter {key} does not match the loaded LoRA layout ({bad[0]})")
            self._prefix_store[self._active] = (self._prefix_key, self._prefix_tokens, self._prefix_state)
            self.model.load_weights(list(new.items()), strict=False)
            mx.eval(self.model.parameters())
            self._prefix_key, self._prefix_tokens, self._prefix_state = self._prefix_store.get(key, (None, None, None))
            self._active = key
            self._cache, self._cache_tokens = None, []
            self.pb.reset()
        if mode != self.mode:
            self._cache, self._cache_tokens = None, []
            self.pb.reset()
        self.mode = mode
        self.canned_thought = mode == "base"
        self.exclude_tools = ("resolve_symbol",) if mode == "base" else ()

    # -- cache ---------------------------------------------------------------
    def _forward(self, cache, tokens: list[int]):
        import mlx.core as mx
        logits = None
        for i in range(0, len(tokens), 2048):
            logits = self.model(mx.array(tokens[i:i + 2048])[None], cache=cache)
            if i + 2048 < len(tokens):
                mx.eval([c.state for c in cache])
        return logits[:, -1, :]

    def _snapshot(self, cache):
        import mlx.core as mx
        snap = [tuple(mx.array(a) if isinstance(a, mx.array) else a for a in c.state) for c in cache]
        mx.eval([x for st in snap for x in st if isinstance(x, mx.array)])
        return snap

    def _restore(self, snap):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)
        for c, state in zip(cache, snap):
            c.state = tuple(mx.array(a) if isinstance(a, mx.array) else a for a in state)
        return cache

    def _cache_ok(self, cache, n: int) -> bool:
        from mlx_lm.models.cache import KVCache
        return all(c.offset == n for c in cache if isinstance(c, KVCache))

    def _prepare(self, tokens: list[int], messages, tools):
        """Return (cache, suffix tokens, reused count)."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        if (self.keep_cache and self._cache is not None and len(tokens) > len(self._cache_tokens)
                and tokens[:len(self._cache_tokens)] == self._cache_tokens
                and self._cache_ok(self._cache, len(self._cache_tokens))):
            k = len(self._cache_tokens)
            return self._cache, tokens[k:], k
        if self.keep_cache:
            sys_only = [m for m in messages[:1] if m.get("role") == "system"]
            if sys_only:
                key = (json.dumps(sys_only, sort_keys=True), json.dumps(tools, sort_keys=True))
                if self._prefix_key != key:
                    pre = self.pb.prefix_tokens(messages, tools)
                    cache = make_prompt_cache(self.model)
                    self._forward(cache, pre)
                    mx.eval([c.state for c in cache])
                    self._prefix_key, self._prefix_tokens = key, pre
                    self._prefix_state = self._snapshot(cache)
                p = self._prefix_tokens
                if len(tokens) > len(p) + 1 and tokens[:len(p)] == p:
                    return self._restore(self._prefix_state), tokens[len(p):], len(p)
        return make_prompt_cache(self.model), tokens, 0

    def _pick(self, logits, allowed: set[int] | None) -> int:
        import mlx.core as mx
        if not allowed:
            return int(mx.argmax(logits, axis=-1).item())
        idx = sorted(allowed)
        sub = logits[0, mx.array(idx)]
        return idx[int(mx.argmax(sub).item())]

    # -- decide --------------------------------------------------------------
    def decide(self, messages: list[dict], tools_schema=None) -> dict:
        import mlx.core as mx
        from mlx_lm.generate import generation_stream, wired_limit

        t0 = time.perf_counter()
        tokens = self.pb.tokens_for(messages, tools_schema)
        user_idx = [i for i, m in enumerate(messages) if m.get("role") == "user"
                    and not str(m.get("content", "")).startswith(self.CTL)]
        last_user = user_idx[-1] if user_idx else 0
        first_step = not any(m.get("role") == "tool" for m in messages[last_user:])
        tool_names = [t["function"]["name"] for t in (tools_schema or [])
                      if t["function"]["name"] not in self.exclude_tools]
        names = tool_names + ([] if (self.force_first_tool and first_step) else ["user"])
        seqs = [] if self.think else self.pb.header_seqs(names)
        call_seqs = [] if self.think else self.pb.header_seqs(tool_names)
        names_by_seq = {tuple(q): n for q, n in zip(seqs, names)}
        names_by_seq.update({tuple(q): n for q, n in zip(call_seqs, tool_names)})
        prompt_len_no_stub = len(tokens)
        if self.canned_thought and not self.think:
            # base mode only: a short pre-written reasoning message replaces the model's own to=self channel
            q = str(messages[last_user].get("content", "")).split("]\n")[-1].strip()
            thought = (f"User asks: {q}\nIf this needs market or account data, call the one or two tools that "
                       "answer it; otherwise reply to the user briefly." if first_step else
                       "If the tool results answer the question, reply to the user in at most two short "
                       "sentences using the slot placeholders; otherwise call the missing tool.")
            tokens = tokens + self.pb.encode(" to=self<|message|>" + thought + "<|eom|><|start|>assistant")
        cache, suffix, reused = self._prepare(tokens, messages, tools_schema)

        mx.reset_peak_memory()
        ids: list[int] = []
        fed = 0
        ended, capped, ans_start, n_forced = False, False, None, 0
        t_first = None
        state, hdr, cur_seqs = "header", [], seqs        # prompt ends with '<|start|>assistant'
        with wired_limit(self.model, [generation_stream]):
            logits = self._forward(cache, suffix)
            while len(ids) < self.max_tokens:
                allowed = None
                if state == "header" and cur_seqs:
                    allowed = {q[len(hdr)] for q in cur_seqs if len(q) > len(hdr) and q[:len(hdr)] == hdr} or None
                elif state == "after_calls":
                    allowed = {self.eot, self.eom}
                tok = self._pick(logits, allowed)
                if t_first is None:
                    t_first = time.perf_counter()
                ids.append(tok)
                if tok in self.eos_ids:
                    ended = True
                    break
                forced: list[int] = []
                if state == "header":
                    hdr.append(tok)
                    if tok == self._msg_id:
                        name = names_by_seq.get(tuple(hdr))
                        state = "call" if name and name != "user" else "body"
                        if state == "call" and self.force_boilerplate:
                            forced = self.pb.forced_chunk(name)
                elif state == "after_calls":                  # tok == eom: another call follows
                    forced = list(self.pb.asst_open)
                    state, hdr, cur_seqs = "header", [], call_seqs
                ids.extend(forced)
                n_forced += len(forced)
                text = self.tokenizer.decode(ids, skip_special_tokens=False)
                seg = text.rsplit("<|start|>assistant", 1)[-1]
                if state == "call" and seg.endswith("</atem:function_calls>"):
                    if self.mode == "base":
                        break                                 # base model: stop after the first block
                    state = "after_calls"
                if state == "body" and not self.think and tok == self.eom:
                    ended = True
                    break
                if self.think and tok == self.eom and not seg.lstrip().startswith("to=self"):
                    ended = True
                    break
                if self.think and "</atem:function_calls>" in text[-40:]:
                    break
                if state == "body" and ans_start is None and seg.lstrip().startswith("to=user<|message|>"):
                    ans_start = len(ids)
                if ans_start is not None:
                    body = seg.split("<|message|>", 1)[-1]
                    if len(ids) - ans_start >= self.answer_max_tokens or (
                            len(re.findall(r"[.!?](?:\s|$)", body)) >= 2 and body.rstrip()[-1:] in ".!?"):
                        capped = True
                        break
                logits = self._forward(cache, [tok] + forced)
                fed = len(ids)
        t_end = time.perf_counter()
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        self._cache, self._cache_tokens = cache, tokens + ids[:fed]
        self.pb.commit(messages, tokens + ids, ended)
        out = parse_glimmer(text)
        if capped and "answer" in out:
            m = re.match(r"^(.*?[.!?])(?:\s+(.*?[.!?]))?(?=\s|$)", out["answer"], re.S)
            if m:
                out["answer"] = (m.group(1) + (" " + m.group(2) if m.group(2) else "")).strip()
        st = {"mode": self.mode, "prompt_tokens": prompt_len_no_stub, "reused_tokens": reused,
              "prefilled_tokens": len(suffix), "gen_tokens": len(ids) - n_forced, "forced_tokens": n_forced,
              "first_token_s": round((t_first or t_end) - t0, 3), "total_s": round(t_end - t0, 3),
              "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
              "kind": "tool_calls" if "tool_calls" in out else ("answer" if "answer" in out else "error")}
        self.stats.append(st)
        if self.verbose:
            print("[glimmer raw]", repr(text), flush=True)
        out.setdefault("raw", text)
        return out




# ---------------------------------------------------------------- Llama 3.2 (mlx_lm)
def parse_llama(text: str) -> dict:
    """Parse a Llama-format completion (special tokens already stripped): JSON call lines, or an answer."""
    body = text.replace("<|python_tag|>", "").strip()
    if not body.startswith("{"):
        return {"answer": body} if body else {"error": "empty response", "raw": text}
    calls = []
    for line in body.splitlines():
        line = line.strip().rstrip(",;")
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            return {"error": "could not parse tool call", "raw": text}
        if not isinstance(d, dict) or not (d.get("name") or d.get("tool")):
            return {"error": "could not parse tool call", "raw": text}
        args = d.get("parameters", d.get("arguments", d.get("args"))) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        calls.append({"tool": str(d.get("name") or d.get("tool")).split(".")[-1],
                      "args": args if isinstance(args, dict) else {}})
    return {"tool_calls": calls} if calls else {"error": "could not parse tool call", "raw": text}


class LlamaBackend:
    """Llama 3.2 Instruct via mlx_lm, in the Llama trained format (agent/format.py `render_llama`: compact
    system prompt + tool list, JSON call lines, several lines = parallel calls, ipython results).

    Same decide() protocol and slot contract as GlimmerBackend (the model sees slot names only). Greedy.
    KV cache: one cache kept across decisions and turns; each prompt reuses the longest common token prefix
    with what the cache holds (trimmed back otherwise), so the system+tools prefix is prefilled once (warmup)
    and, within a turn, only the new tool results are processed. Answers stop after two sentences /
    answer_max_tokens."""

    def __init__(self, model_dir: str | os.PathLike = DEFAULT_LLAMA, adapter: str | None = None,
                 max_tokens: int = 160, answer_max_tokens: int = 70, keep_cache: bool = True,
                 verbose: bool = False, warm: bool = True):
        import mlx.core as mx  # noqa: F401
        from mlx_lm import load

        from .format import LlamaPromptBuilder
        self.model_dir, self.adapter = str(model_dir), adapter
        self.mode = "trained" if adapter else "base"
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(self.model_dir, adapter_path=adapter)
        self.load_s = time.perf_counter() - t0
        self.pb = LlamaPromptBuilder(self.tokenizer)
        self.eos_ids = set(self.pb.eos_ids) | set(getattr(self.tokenizer, "eos_token_ids", None) or [])
        self.max_tokens, self.answer_max_tokens = max_tokens, answer_max_tokens
        self.keep_cache, self.verbose = keep_cache, verbose
        self.stats: list[dict] = []
        self._cache = None
        self._cache_tokens: list[int] = []
        self.warm_s = 0.0
        if warm:
            self.warmup()

    def warmup(self) -> None:
        from tools import schema
        from .loop import SYSTEM_PROMPT
        t0 = time.perf_counter()
        saved = self.max_tokens
        self.max_tokens = 4
        try:
            self.decide([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Hi"}], schema())
        finally:
            self.max_tokens = saved
            self.stats.clear()
        self.warm_s = time.perf_counter() - t0

    def _forward(self, cache, tokens: list[int]):
        import mlx.core as mx
        logits = None
        for i in range(0, len(tokens), 1024):
            logits = self.model(mx.array(tokens[i:i + 1024])[None], cache=cache)
            if i + 1024 < len(tokens):
                mx.eval([c.state for c in cache])
        return logits[:, -1, :]

    def _prepare(self, tokens: list[int]):
        """(cache, suffix to feed, reused count): reuse the longest common prefix with the cached stream."""
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
        if self.keep_cache and self._cache is not None and self._cache_tokens:
            have = self._cache_tokens
            k = 0
            n = min(len(have), len(tokens) - 1)          # always feed at least one token
            while k < n and have[k] == tokens[k]:
                k += 1
            drop = len(have) - k
            if k > 0 and (drop == 0 or can_trim_prompt_cache(self._cache)):
                if drop:
                    trim_prompt_cache(self._cache, drop)
                return self._cache, tokens[k:], k
        return make_prompt_cache(self.model), tokens, 0

    def decide(self, messages: list[dict], tools_schema=None) -> dict:
        import mlx.core as mx
        from mlx_lm.generate import generation_stream, wired_limit
        t0 = time.perf_counter()
        tokens = self.pb.tokens_for(messages, tools_schema)
        cache, suffix, reused = self._prepare(tokens)
        mx.reset_peak_memory()
        ids: list[int] = []
        capped = ended = False
        t_first = None
        fed_all = True
        with wired_limit(self.model, [generation_stream]):
            logits = self._forward(cache, suffix)
            while len(ids) < self.max_tokens:
                tok = int(mx.argmax(logits, axis=-1).item())
                if t_first is None:
                    t_first = time.perf_counter()
                ids.append(tok)
                fed_all = False
                if tok in self.eos_ids:
                    ended = True
                    break
                text = self.tokenizer.decode(ids)
                body = text.lstrip()
                if body and not body.startswith("{") and (
                        len(ids) >= self.answer_max_tokens
                        or (len(re.findall(r"[.!?](?:\s|$)", body)) >= 2 and body.rstrip()[-1:] in ".!?")):
                    capped = True
                    break
                logits = self._forward(cache, [tok])
                fed_all = True
        t_end = time.perf_counter()
        fed = ids if fed_all else ids[:-1]
        self._cache, self._cache_tokens = cache, tokens + fed
        text = self.tokenizer.decode([i for i in ids if i not in self.eos_ids])
        out = parse_llama(text)
        if capped and "answer" in out:
            m = re.match(r"^(.*?[.!?])(?:\s+(.*?[.!?]))?(?=\s|$)", out["answer"], re.S)
            if m:
                out["answer"] = (m.group(1) + (" " + m.group(2) if m.group(2) else "")).strip()
        self.stats.append({"mode": self.mode, "family": "llama", "prompt_tokens": len(tokens), "reused_tokens": reused,
                           "prefilled_tokens": len(suffix), "gen_tokens": len(ids), "forced_tokens": 0,
                           "first_token_s": round((t_first or t_end) - t0, 3), "total_s": round(t_end - t0, 3),
                           "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2), "ended": ended,
                           "kind": "tool_calls" if "tool_calls" in out else ("answer" if "answer" in out else "error")})
        if self.verbose:
            print("[llama raw]", repr(text), flush=True)
        out.setdefault("raw", text)
        return out


# ---------------------------------------------------------------- named models (voice picker, eval)
ALIASES = ("base", "trained")


def _builtin_registry() -> dict:
    env_adapter = os.environ.get("VOICE_ADAPTER")
    return {"default": "glimmer-base", "models": [
        {"name": "glimmer-base", "label": "Glimmer base", "family": "glimmer", "adapter": None},
        {"name": "glimmer-trained", "label": "Glimmer trained", "family": "glimmer",
         "adapter": env_adapter or str(DEFAULT_ADAPTER)},
        {"name": "llama-1b", "label": "Llama 1B", "family": "llama", "adapter": str(DEFAULT_LLAMA_ADAPTER)},
    ]}


def registry() -> dict:
    """Named models. Read from $VOICE_MODELS (default $VOICE_HOME/models.json,
    see models.example.json) — {"default": name, "models": [{name, label, family: glimmer|llama, model_dir?,
    adapter?: dir|null, mode?: base|trained}]} — else a built-in list. Each model is `available` when its
    weights (and adapter, if any) are on disk."""
    path = Path(os.environ.get("VOICE_MODELS") or DEFAULT_REGISTRY)
    try:
        reg = json.loads(path.read_text())
    except (OSError, ValueError):
        reg = _builtin_registry()
    out = []
    for m in reg.get("models") or []:
        m = dict(m)
        fam = m.get("family", "glimmer")
        m.setdefault("label", m["name"])
        m.setdefault("model_dir", os.environ.get("VOICE_GLIMMER_MODEL", str(DEFAULT_GLIMMER)) if fam == "glimmer"
                     else os.environ.get("VOICE_LLAMA_MODEL", str(DEFAULT_LLAMA)))
        m["adapter"] = os.path.expanduser(m["adapter"]) if m.get("adapter") else None
        m.setdefault("mode", "trained" if m["adapter"] else "base")
        m["available"] = Path(m["model_dir"]).exists() and (
            m["adapter"] is None or (Path(m["adapter"]) / "adapters.safetensors").exists())
        out.append(m)
    env = os.environ.get("VOICE_ADAPTER")
    if env and not any(m["adapter"] and os.path.realpath(m["adapter"]) == os.path.realpath(os.path.expanduser(env))
                       for m in out):
        mdir = os.environ.get("VOICE_GLIMMER_MODEL", str(DEFAULT_GLIMMER))
        ad = os.path.expanduser(env)
        out.append({"name": "glimmer-trained", "label": "Glimmer trained", "family": "glimmer", "model_dir": mdir,
                    "adapter": ad, "mode": "trained",
                    "available": Path(mdir).exists() and (Path(ad) / "adapters.safetensors").exists()})
    names = [m["name"] for m in out]
    default = reg.get("default") if reg.get("default") in names else (names[0] if names else None)
    return {"default": default, "models": out}


def list_models() -> list[dict]:
    """[{name, label, family, available}] for the phone picker."""
    return [{k: m[k] for k in ("name", "label", "family", "available")} for m in registry()["models"]]


def resolve_model(name: str) -> str:
    """Canonical model name. Back-compat aliases: 'base' = the first Glimmer without an adapter; 'trained' =
    the Glimmer whose adapter is $VOICE_ADAPTER, else the first available Glimmer with an adapter."""
    reg = registry()["models"]
    by = {m["name"]: m for m in reg}
    if name in by:
        return name
    glim = [m for m in reg if m.get("family", "glimmer") == "glimmer"]
    if name == "base":
        hit = next((m for m in glim if not m["adapter"]), None)
    elif name == "trained":
        env = os.environ.get("VOICE_ADAPTER")
        env = os.path.realpath(os.path.expanduser(env)) if env else None
        hit = next((m for m in glim if m["adapter"] and env and os.path.realpath(m["adapter"]) == env), None) \
            or next((m for m in glim if m["adapter"] and m["available"]), None) \
            or next((m for m in glim if not m["adapter"]), None)
    else:
        hit = None
    if hit is None:
        raise KeyError(f"unknown model {name!r}")
    return hit["name"]


class GlimmerView:
    """One named Glimmer model (adapter + mode) on a SHARED loaded Glimmer: adapters are hot-swapped
    (`GlimmerBackend.activate`), so base + every adapter cost one ~17 GB model."""

    def __init__(self, shared: GlimmerBackend, adapter: str | None, mode: str, name: str):
        self.shared, self.adapter, self.mode, self.name = shared, adapter, mode, name

    def _on(self):
        self.shared.activate(self.adapter, self.mode)
        return self.shared

    def decide(self, messages, tools_schema=None):
        return self._on().decide(messages, tools_schema)

    def warmup(self):
        self._on().warmup()

    def __getattr__(self, k):                     # stats, pb, warm_s, _cache ... of the shared backend
        return getattr(self.shared, k)


_SHARED: dict = {}
_GLIMMERS: dict = {}


def make_backend(model: str = "base"):
    """Factory used by voice/: a named model (registry()) or the aliases 'base' / 'trained'. Cached per name.

    All Glimmer models share ONE loaded Glimmer (loaded with the first available adapter so LoRA layers exist;
    adapters are hot-swapped). If no adapter is on disk, Glimmer loads plain and only base is usable."""
    if model == "scripted":
        return ScriptedBackend([])
    name = resolve_model(model)
    if name in _SHARED:
        return _SHARED[name]
    reg = registry()["models"]
    spec = next(m for m in reg if m["name"] == name)
    if not spec["available"]:
        raise RuntimeError(f"model {name} is not available (weights or adapter missing)")
    if spec["family"] == "llama":
        be = LlamaBackend(spec["model_dir"], adapter=spec["adapter"])
    else:
        mdir = spec["model_dir"]
        if mdir not in _GLIMMERS:
            first = next((m["adapter"] for m in reg if m["family"] == "glimmer" and m["model_dir"] == mdir
                          and m["adapter"] and m["available"]), None)
            _GLIMMERS[mdir] = GlimmerBackend(mdir, adapter=first, mode=spec["mode"], warm=False)
        be = GlimmerView(_GLIMMERS[mdir], spec["adapter"], spec["mode"], name)
        be.warmup()
    be.frame_line = bool(spec.get("frame_line"))      # trained with the frame state line (FORMAT.md)
    _SHARED[name] = be
    return be
