#!/usr/bin/env python3
"""Rust engine against the JAX engine on the same chip: logits after the prompt, then greedy tokens.

    python3.13 tools/parity.py jax  --out ref.json     # the JAX engine of ../tpu-jax-v5e1-2b
    python3.13 tools/parity.py rust --out rust.json    # this rig's jaxrust_rs
    python3 tools/parity.py compare ref.json rust.json

Each side runs in its own process: one process holds the TPU. Both read the same prompts through
the same chat template, with the leading <bos> the JAX engine adds, and decode greedily for a fixed
128 tokens with no stop token. The JSON keeps the last-position logits of the first prompt and
every token id, so the comparison is computed here, not read off by eye.
"""

import argparse
import array
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "google/gemma-4-E2B-it"
REVISION = "4adf15748de9e1a0097b6274cee9c15836c41c37"
PLUGIN = os.environ.get("PJRT_PLUGIN", "/usr/local/lib/python3.13/dist-packages/libtpu/libtpu.so")
NEW_TOKENS = 128
CONTEXT = 256
PROMPTS = [
    "Explain TPU buffer donation in JAX in three sentences.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "List five prime numbers greater than 100.",
    "Translate 'The weather is lovely today' into French, German and Spanish.",
    "A train leaves at 3:15 pm and arrives at 6:40 pm. How long is the trip?",
    "Describe how a hash table handles collisions.",
]
REPEATS = 3


def prompt_ids(tok, text):
    out = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    ids = [int(t) for t in out]
    if ids[0] != tok.bos_token_id:  # what JaxGemmaEngine.generate_stream does
        ids = [tok.bos_token_id] + ids
    return ids


def run_jax(out):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tpu-jax-v5e1-2b"))
    import jax
    import jax.numpy as jnp
    from jax_engine import JaxGemmaEngine
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    eng = JaxGemmaEngine(model_id=MODEL, kv_cache_dtype="bf16", quant_mode="fp16", max_model_len=CONTEXT)
    eng.load()
    eng.bos_token_id = tok.bos_token_id
    first = prompt_ids(tok, PROMPTS[0])
    from ports.gemma4.jax_e_model import pad_to_tpu_v6e_bucket
    padded, valid = pad_to_tpu_v6e_bucket(jnp.asarray(first, jnp.int32)[None, :])
    last, _, _ = eng._jit_prefill(model=eng.model, prompt_ids=padded, prompt_valid=valid, params=eng.params,
                                  max_new_tokens=1, quant_mode=eng.quant_mode, cache_dtype=eng.cache_dtype,
                                  window_kv=eng.window_kv)
    logits = [float(x) for x in jax.device_get(last[0]).astype("float32")]
    rows = []
    for p in PROMPTS:
        ids = prompt_ids(tok, p)
        eng.generate(ids, max_new_tokens=NEW_TOKENS, temperature=0.0, eos_token_ids=None)
        rates = []
        for _ in range(REPEATS):
            toks, st = eng.generate(ids, max_new_tokens=NEW_TOKENS, temperature=0.0, eos_token_ids=None)
            rates.append(st.decode_tok_per_s)
        rows.append({"prompt": p, "ids": ids, "tokens": [int(t) for t in toks], "decode_tok_s": statistics.median(rates)})
        print(f"jax  {rows[-1]['decode_tok_s']:7.1f} tok/s  {p[:50]}", flush=True)
    json.dump({"engine": "jax", "device": str(jax.devices()[0]), "weight_bytes": eng.weight_bytes,
               "logits0": logits, "prompts": rows}, open(out, "w"))


def run_rust(out):
    sys.path.insert(0, os.path.join(HERE, "rust", "build"))
    import jaxrust_rs
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    plat = jaxrust_rs.init(PLUGIN)
    print("platform", plat, "build", jaxrust_rs.BUILD, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    path = snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors"])
    t0 = time.perf_counter()
    eng = jaxrust_rs.Engine(path, CONTEXT)
    print(f"engine {eng!r} in {time.perf_counter() - t0:.1f}s, {eng.timings()}", flush=True)
    first = prompt_ids(tok, PROMPTS[0])
    t0, raw = eng.logits(first)
    logits = list(array.array("f", raw))
    rows = []
    for p in PROMPTS:
        ids = prompt_ids(tok, p)
        eng.decode(ids, NEW_TOKENS)
        rates = []
        for _ in range(REPEATS):
            r = eng.decode(ids, NEW_TOKENS)
            rates.append(len(r.tokens) / r.decode_s)
        rows.append({"prompt": p, "ids": ids, "tokens": list(r.tokens), "decode_tok_s": statistics.median(rates),
                     "prefill_s": r.prefill_s})
        print(f"rust {rows[-1]['decode_tok_s']:7.1f} tok/s  prefill {r.prefill_s * 1e3:6.1f} ms  {p[:50]}", flush=True)
    json.dump({"engine": "rust", "platform": plat, "build": jaxrust_rs.BUILD, "weight_bytes": eng.weight_bytes,
               "cache_bytes": eng.cache_bytes, "timings": eng.timings(), "logits0": logits, "prompts": rows},
              open(out, "w"))


def compare(ref_path, got_path):
    ref, got = json.load(open(ref_path)), json.load(open(got_path))
    a, b = ref["logits0"], got["logits0"]
    diff = [abs(x - y) for x, y in zip(a, b)]
    am = max(range(len(a)), key=a.__getitem__)
    bm = max(range(len(b)), key=b.__getitem__)
    def top(v, k=10):
        return sorted(range(len(v)), key=v.__getitem__, reverse=True)[:k]
    overlap = len(set(top(a)) & set(top(b)))
    def lsm(v):
        m = max(v)
        z = math.log(sum(math.exp(x - m) for x in v)) + m
        return [x - z for x in v]
    la, lb = lsm(a), lsm(b)
    kl = sum(math.exp(x) * (x - y) for x, y in zip(la, lb))
    print(f"logits after prompt 0: max |diff| {max(diff):.4f}, mean |diff| {sum(diff) / len(diff):.5f}, "
          f"argmax {am} vs {bm}, top-10 overlap {overlap}/10, KL(ref||got) {kl:.2e}")
    same_ids = sum(x["ids"] == y["ids"] for x, y in zip(ref["prompts"], got["prompts"]))
    identical, prefixes = 0, []
    for x, y in zip(ref["prompts"], got["prompts"]):
        n = 0
        for s, t in zip(x["tokens"], y["tokens"]):
            if s != t:
                break
            n += 1
        prefixes.append(n)
        identical += x["tokens"] == y["tokens"]
    k = len(ref["prompts"])
    print(f"prompts tokenized the same: {same_ids} of {k}; completions identical: {identical} of {k}; "
          f"matching prefix per prompt: {prefixes} (mean {sum(prefixes) / k:.1f} of {NEW_TOKENS})")
    rj = statistics.median(p["decode_tok_s"] for p in ref["prompts"])
    rr = statistics.median(p["decode_tok_s"] for p in got["prompts"])
    print(f"decode tok/s (median over prompts): {ref['engine']} {rj:.1f}, {got['engine']} {rr:.1f}, ratio {rr / rj:.3f}")
    print(f"weights on device: {ref['engine']} {ref['weight_bytes'] / 1e9:.3f} GB, {got['engine']} {got['weight_bytes'] / 1e9:.3f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("side", choices=["jax", "rust", "compare"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.side == "jax":
        run_jax(a.out)
    elif a.side == "rust":
        run_rust(a.out)
    else:
        compare(*a.files)


if __name__ == "__main__":
    main()
