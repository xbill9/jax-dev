#!/usr/bin/env python3
"""Time any OpenAI-compatible server the same way: streamed chat, one request at a time.

    python3 tools/serve_bench.py run --url http://127.0.0.1:8000 --label vllm --out vllm.json
    python3 tools/serve_bench.py compare vllm.json jax.json rust.json

Every engine gets the same 8 prompts (those of tools/parity.py) through its own chat
template, greedy (temperature 0), exactly --max-tokens new tokens (`ignore_eos`), streamed
with `stream_options.include_usage`. One untimed request per prompt, then --repeats timed.

Per request, measured at the client:
- first token: request sent to the first chunk carrying text.
- decode tok/s: (completion_tokens - 1) / (finish chunk - first text chunk), with
  completion_tokens from the server's usage chunk, so chunks that carry several tokens
  still count right.
Per prompt the median over repeats; per engine the median over prompts. Standard library only.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from parity import PROMPTS


def one(url, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = last = None
    text, usage = [], None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    first = first or time.perf_counter()
                    text.append(piece)
                if c.get("finish_reason"):
                    # The clock stops at the finish chunk, not the last chunk with text: past the
                    # end of the answer (ignore_eos) tokens can decode to empty text.
                    last = time.perf_counter()
    n = usage["completion_tokens"]
    return {
        "first_token_s": first - t0,
        "decode_tok_s": (n - 1) / (last - first),
        "completion_tokens": n,
        "prompt_tokens": usage["prompt_tokens"],
        "text": "".join(text),
    }


def run(a):
    model = (
        a.model
        or json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
    )
    rows = []
    for p in PROMPTS:
        one(a.url, model, p, a.max_tokens)  # untimed
        reps = [one(a.url, model, p, a.max_tokens) for _ in range(a.repeats)]
        row = {
            "prompt": p,
            "prompt_tokens": reps[0]["prompt_tokens"],
            "completion_tokens": [r["completion_tokens"] for r in reps],
            "decode_tok_s": statistics.median(r["decode_tok_s"] for r in reps),
            "first_token_ms": 1000
            * statistics.median(r["first_token_s"] for r in reps),
            "text": reps[0]["text"],
            "texts_repeat_identical": len({r["text"] for r in reps}) == 1,
        }
        rows.append(row)
        print(
            f"{a.label}: {row['decode_tok_s']:.1f} tok/s, first token {row['first_token_ms']:.1f} ms, "
            f"{row['prompt_tokens']} prompt tokens: {p[:40]}",
            flush=True,
        )
    out = {
        "label": a.label,
        "url": a.url,
        "model": model,
        "max_tokens": a.max_tokens,
        "repeats": a.repeats,
        "note": a.note,
        "prompts": rows,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


def compare(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    base = runs[0]
    print(
        f"{'engine':<10} {'decode tok/s':>13} {'range':>15} {'first token ms':>15} {'tokens':>7} {'same text as ' + base['label']:>18}"
    )
    med = {}
    for r in runs:
        rates = [p["decode_tok_s"] for p in r["prompts"]]
        ttft = [p["first_token_ms"] for p in r["prompts"]]
        toks = sorted({t for p in r["prompts"] for t in p["completion_tokens"]})
        same = sum(
            p["text"] == q["text"] for p, q in zip(r["prompts"], base["prompts"])
        )
        med[r["label"]] = statistics.median(rates)
        print(
            f"{r['label']:<10} {med[r['label']]:>13.1f} {min(rates):>7.1f}-{max(rates):<7.1f} "
            f"{statistics.median(ttft):>15.1f} {','.join(map(str, toks)):>7} {same:>14} of {len(r['prompts'])}"
        )
    print(
        "prompt tokens per engine:",
        {r["label"]: [p["prompt_tokens"] for p in r["prompts"]] for r in runs},
    )
    print(
        "repeats identical:",
        {
            r["label"]: sum(p["texts_repeat_identical"] for p in r["prompts"])
            for r in runs
        },
    )
    for r in runs[1:]:
        print(
            f"ratio {r['label']} / {base['label']}: {med[r['label']] / med[base['label']]:.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--model", default=None)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--note", default="")
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a.files)


if __name__ == "__main__":
    main()
