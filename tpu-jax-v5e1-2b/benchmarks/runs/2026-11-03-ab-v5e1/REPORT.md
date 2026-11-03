# E2B checkpoint A/B on one TPU v5e chip, pure JAX

Run 2026-11-03 on a flex-start `ct5lp-hightpu-1t` VM (`jax-v5e1-2b`, `us-west4-a`), JAX 0.12.0.1, libtpu 0.0.51, Python 3.13. One process per arm, one after another on the same chip. Engine: this rig's `jax_engine.py`, byte-identical across `tpu-jax-v5e1-2b` and `tpu-jax-v5e1-2b-q4w4a16`.

Method (`tools/ab_bench.py`): 8 fixed prompts through each model's own chat template, greedy, exactly 128 new tokens with no EOS stop, bf16 KV cache, `max_model_len` 4096. One untimed generation per prompt compiles; then 3 timed repeats. Each prompt's figure is the median of its 3 repeats; the arm's figure is the median over the 8 prompts. Comparisons by `tools/ab_compare.py`.

| Arm | `--quant-mode` | Decode tok/s | Range over prompts | Prefill ms | Weights GB | HBM in use GB | Load s |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `google/gemma-4-E2B-it` (`4adf1574`) | `fp16` (dense bf16) | 132.6 | 130.3–132.8 | 7.4 | 9.257 | 9.260 | 64.7 |
| `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` (`315aec06`) | `w4a16` | 103.0 | 102.2–103.2 | 23.2 | 6.559 | 6.561 | 39.2 |
| `google/gemma-4-E2B-it-qat-w4a16-ct` | `w4a16` | 103.1 | 101.6–103.4 | 23.2 | 6.559 | 6.562 | 51.9 |

HBM limit reported by the device: 16.909 GB. Load time includes the Hub download on a fresh VM.

#### Findings

- **bf16 decodes 1.29x faster than either 4-bit build in this engine on v5e** (132.6 against 103.0 tok/s), and prefills 3.1x faster (7.4 against 23.2 ms). The 4-bit arms ran the engine's default `w4a16_impl="reference"`, which unpacks int4 inside every matmul; which part of that is slower than the dense path was not measured.
- **4-bit saves 2.70 GB of HBM** (6.56 against 9.26 GB resident), headroom the KV cache can use on a 16.9 GB chip.
- **The repack and Google's export run at the same speed and size** (103.0 against 103.1 tok/s, 6.559 GB each), as on the Colab v5e (93.3 against 93.7 tok/s, `colab/runs/2026-10-09/`).
- **Their outputs differ.** All 8 prompts tokenize identically; none of the 8 completions match token for token, and they agree for 26.6 of 128 tokens on average before diverging.
- **Against bf16, neither 4-bit build is a quantization of it.** Both are QAT weights, a separate training run, so agreement with `google/gemma-4-E2B-it` (20.5 and 20.6 of 128 tokens) measures QAT against the release, not the cost of 4 bits. All three produce coherent answers (`*.json`, field `text`).

#### Files

`bf16.json`, `q4w4a16.json`, `w4a16.json` (per prompt: prompt ids, token ids, text, each timed run), their `.log`s, `run.sh`, `compare.txt`.
