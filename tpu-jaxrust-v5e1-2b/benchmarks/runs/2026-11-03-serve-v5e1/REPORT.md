# vLLM, the JAX engine and the Rust engine through one client, E2B bf16, one TPU v5e chip

Run 2026-11-03 on a flex-start `ct5lp-hightpu-1t` VM (`jax-v5e1-2b`, `us-west4-a`), deleted after the run. Checkpoint `google/gemma-4-E2B-it` at `4adf1574` for all three. Versions in `versions.txt`: JAX 0.12.0.1 and libtpu 0.0.51 for the JAX and Rust engines, vLLM 0.33.0rc1.dev430 (`vllm/vllm-tpu@sha256:f1f925f5…`) for vLLM.

Method (`tools/serve_bench.py`, `run.sh`): each engine served on port 8000 in turn, alone on the chip, and timed by the same client. Same 8 prompts as `tools/parity.py`, each through the server's own chat template, greedy, exactly 128 new tokens (`ignore_eos`), streamed, one request at a time. One untimed request per prompt, then 3 timed. First token is request sent to first text chunk; decode tok/s is (completion tokens − 1) over first text chunk to finish chunk, with the token count from the server's usage chunk. Per prompt the median of 3, per engine the median over prompts.

| Engine | Server | Context | Decode tok/s | Range over prompts | First token, ms |
| :--- | :--- | ---: | ---: | :--- | ---: |
| vLLM | `vllm serve`, `--max-num-seqs 16 --no-enable-prefix-caching` | 2048 | **148.7** | 148.7–148.7 | 13.0 |
| Rust engine | `rust_openai_server.py` | 2048 | **148.4** | 147.8–148.6 | **8.9** |
| Python JAX engine | `../../../../tpu-jax-v5e1-2b/jax_openai_server.py --quant-mode fp16` | 2048 | 126.7 | 126.0–128.4 | 12.5 |

#### Findings

- 🟢 **The Rust engine decodes at vLLM's rate**, 148.4 against 148.7 tok/s (ratio 0.998), and reaches its first token in 8.9 ms against vLLM's 13.0.
- ⚠️ **The Python JAX engine runs at 0.852x vLLM**, 126.7 tok/s, with a first token level with vLLM's (12.5 ms).
- 🟢 **Same prompts in, same tokens out.** All three tokenized every prompt to the same length (17–35 tokens) and returned exactly 128 tokens every time; each engine repeated its own text exactly across the 3 runs.
- 🟢 **The three agree on 6 of 8 completions.** The JAX and Rust engines agree with each other on 7 of 8. Both part from vLLM at the same two points, near-ties between phrasings: prompt 2 at character 542 ("as a new capital city" against "as a planned city") and prompt 6 at character 127.
- **Earlier figures for comparison**, each measured a different way: the in-engine timing of `../2026-11-03-parity-v5e1` (context 256) gave Rust 150.7 and JAX 131.6 tok/s; the September vLLM sweep in `~/gemma4-dev/jev-tpu-v5e1` gave 144 tok/s at one request on an older vLLM image.

#### Scope

One request at a time; vLLM's batching, the reason it exists, is not exercised. bf16 weights only. Decoding past the end of the answer (`ignore_eos`) keeps every engine at 128 tokens, so the text after a model's own stop point is filler; the agreement counts include it.

#### Files

`vllm.json`, `jax.json`, `rust.json` (per prompt: prompt tokens, token counts, rate, first token, text), `*.log` (client), `*.server.log`, `compare.txt` (computed by `tools/serve_bench.py compare`), `run.sh`, `versions.txt`.
