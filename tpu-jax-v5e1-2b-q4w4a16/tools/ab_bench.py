#!/usr/bin/env python3
"""One arm of the E2B checkpoint A/B: load a checkpoint, decode fixed prompts greedily, write JSON.

    python3.13 tools/ab_bench.py --model google/gemma-4-E2B-it --quant-mode fp16 --out bf16.json

Every arm gets the same prompts, the same chat template call, greedy sampling and a fixed
128-token completion (no EOS stop), so decode work is identical across arms. One untimed
warmup generation per prompt length compiles the programs; then each prompt runs REPEATS
times. Token ids are kept so arms can be compared position by position (tools/ab_compare.py).
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402

from jax_engine import JaxGemmaEngine  # noqa: E402

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
NEW_TOKENS = 128
REPEATS = 3


def chat_ids(tok, text):
    out = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant-mode", required=True, choices=["w4a16", "fp16"])
    ap.add_argument("--kv-cache-dtype", default="bf16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.perf_counter()
    eng = JaxGemmaEngine(model_id=args.model, kv_cache_dtype=args.kv_cache_dtype,
                         quant_mode=args.quant_mode, max_model_len=args.max_model_len)
    eng.load()
    load_s = time.perf_counter() - t0
    eng.bos_token_id = getattr(tok, "bos_token_id", None)
    mem = eng.memory_stats()
    print(f"loaded {args.model} in {load_s:.1f}s, weights {mem['weight_bytes'] / 1e9:.3f} GB, "
          f"HBM {mem['hbm_bytes_in_use'] / 1e9:.3f} / {mem['hbm_bytes_limit'] / 1e9:.3f} GB", flush=True)

    rows = []
    for p in PROMPTS:
        ids = chat_ids(tok, p)
        eng.generate(ids, max_new_tokens=NEW_TOKENS, temperature=0.0, eos_token_ids=None)  # compile
        runs = []
        for _ in range(REPEATS):
            toks, st = eng.generate(ids, max_new_tokens=NEW_TOKENS, temperature=0.0, eos_token_ids=None)
            runs.append({"prefill_ms": st.prefill_ms, "decode_ms": st.decode_ms,
                         "completion_tokens": st.completion_tokens, "decode_tok_s": st.decode_tok_per_s})
        rows.append({
            "prompt": p,
            "prompt_ids": ids,
            "tokens": [int(t) for t in toks],
            "text": tok.decode(toks, skip_special_tokens=True),
            "runs": runs,
            "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in runs),
            "prefill_ms_median": statistics.median(r["prefill_ms"] for r in runs),
        })
        print(f"{rows[-1]['decode_tok_s_median']:7.1f} tok/s  prefill {rows[-1]['prefill_ms_median']:7.1f} ms  "
              f"{p[:50]}", flush=True)

    result = {
        "model": args.model,
        "quant_mode": args.quant_mode,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_model_len": args.max_model_len,
        "new_tokens": NEW_TOKENS,
        "repeats": REPEATS,
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "load_s": load_s,
        "weight_bytes": mem["weight_bytes"],
        "hbm_bytes_in_use": mem["hbm_bytes_in_use"],
        "hbm_bytes_limit": mem["hbm_bytes_limit"],
        "decode_tok_s_median": statistics.median(r["decode_tok_s_median"] for r in rows),
        "prefill_ms_median": statistics.median(r["prefill_ms_median"] for r in rows),
        "prompts": rows,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"median decode {result['decode_tok_s_median']:.1f} tok/s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
