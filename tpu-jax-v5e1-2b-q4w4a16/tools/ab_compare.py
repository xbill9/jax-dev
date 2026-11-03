#!/usr/bin/env python3
"""Compare ab_bench.py arms against a reference arm. Stdlib only; all counts are computed here.

    python3 tools/ab_compare.py bf16.json q4w4a16.json w4a16.json

The first file is the reference. Per arm: median decode tok/s and its ratio to the reference,
weight and HBM bytes, whether every prompt tokenized identically, how many of the completions
match the reference token for token, and the mean length of the matching prefix.
"""

import json
import sys


def main(paths):
    arms = [json.load(open(p)) for p in paths]
    ref = arms[0]
    head = f"{'arm':<48} {'tok/s':>7} {'x ref':>6} {'weights GB':>10} {'HBM GB':>7} {'prompts same':>12} {'identical':>9} {'mean prefix':>11}"
    print(head)
    for a in arms:
        same_prompt = sum(x["prompt_ids"] == y["prompt_ids"] for x, y in zip(a["prompts"], ref["prompts"]))
        identical = 0
        prefixes = []
        for x, y in zip(a["prompts"], ref["prompts"]):
            n = 0
            for s, t in zip(x["tokens"], y["tokens"]):
                if s != t:
                    break
                n += 1
            prefixes.append(n)
            identical += x["tokens"] == y["tokens"]
        name = f"{a['model']} ({a['quant_mode']})"
        print(f"{name:<48} {a['decode_tok_s_median']:7.1f} {a['decode_tok_s_median'] / ref['decode_tok_s_median']:6.3f} "
              f"{a['weight_bytes'] / 1e9:10.3f} {a['hbm_bytes_in_use'] / 1e9:7.3f} "
              f"{same_prompt:>5} of {len(ref['prompts'])} {identical:>3} of {len(ref['prompts'])} "
              f"{sum(prefixes) / len(prefixes):7.1f}/{ref['new_tokens']}")


if __name__ == "__main__":
    main(sys.argv[1:])
