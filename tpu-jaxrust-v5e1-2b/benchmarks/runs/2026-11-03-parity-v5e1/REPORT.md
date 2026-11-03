# Rust engine against the JAX engine, E2B bf16, one TPU v5e chip

Run 2026-11-03 on a flex-start `ct5lp-hightpu-1t` VM (`jax-v5e1-2b`, `us-west4-a`), libtpu 0.0.51 (PJRT API 0.116), JAX 0.12.0.1 for the reference, Python 3.13, rustc 1.101.0-nightly (2026-11-02). Checkpoint `google/gemma-4-E2B-it` at `4adf1574` for both.

Method (`tools/parity.py`): the JAX engine of `tpu-jax-v5e1-2b` (`--quant-mode fp16`, bf16 KV) and this rig's `jaxrust_rs`, each in its own process. Same 8 prompts through the same chat template with the leading `<bos>` the JAX engine adds, greedy, exactly 128 new tokens with no stop token, context 256 on both. One untimed decode per prompt, then 3 timed repeats; each prompt's rate is the median of its repeats, the engine's the median over prompts. Logits are the last-position, softcapped logits after prompt 0.

| | JAX engine | Rust engine |
| :--- | ---: | ---: |
| Decode tok/s, median over 8 prompts | 131.6 | **150.7** |
| Decode tok/s, range over prompts | 130.1–132.2 | 150.7–151.0 |
| Prefill ms, median | 7.4 (the same day's A/B run, `../tpu-jax-v5e1-2b/benchmarks/runs/2026-11-03-ab-v5e1`) | 7.8 |
| Weights on device | 9.257 GB | 9.257 GB |
| KV cache on device | | 4.72 MB |
| Weight upload / decode compile | | 2.3 s / 5.7 s |

#### Findings

- 🟢 **The Rust engine decodes 1.145x faster than the JAX engine** on the same chip, checkpoint, prompts and context (150.7 against 131.6 tok/s).
- 🟢 **Its logits match.** After prompt 0: same argmax, the same top 10 tokens, KL(JAX‖Rust) 1.31e-6, largest absolute difference 0.231 and mean 0.073 on a softcapped ±30 scale.
- 🟢 **7 of 8 completions are identical** to the JAX engine's for all 128 tokens.
- ⚠️ **Prompt 7 diverges at token 42**, where the two take different near-equal continuations ("Time until 4:00 pm" against "Next full hour: 4:00 pm"); both stay coherent.
- **Where the two programs differ:** the Rust decode step returns the argmax token from the same program, where the JAX engine samples in a separate call; and Rust attends over the whole 256-slot cache at every step, where the JAX cache is the prompt bucket plus the 128 new tokens, also 256 here. Which part accounts for the speed difference was not measured.

#### Files

`jax.json`, `rust.json` (logits after prompt 0, per-prompt ids, tokens and rates), their logs, `compare.txt` (the comparison above, computed by `tools/parity.py compare`), `run.sh`.
