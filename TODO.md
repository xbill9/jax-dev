# TODO

Open work across the jax-dev rigs, as of 2026-11-03. Each item says what it needs; the ones marked 💳 need a billed flex-start TPU VM.

#### MTP drafter in the Rust engine (`tpu-jaxrust-v5e1-2b`)

No rig in this tree does speculative decoding. The drafter to use is `google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant` (158 MB), or its W4A16 repack from `~/gemma4-dev/local-vllm-3050oem-2b-q4w4a16emb4/repack/repack_drafter.py` (143 MB).

- [ ] Drafter graph: 4 decoder layers at width 256, with projections to and from the target's width (vLLM `gemma4_mtp.py`). It reads the target's hidden state, so the target's decode program must return it.
- [ ] Verify program: the target runs over the drafted tokens, keeps the accepted prefix and rolls the KV cache back past the first rejection.
- [ ] Parity against vLLM's MTP: same prompts, same accepted tokens.
- [ ] 💳 Acceptance rate and decode tok/s on a v5e-1. Measured elsewhere, for reference: on this chip under vLLM, Google's QAT assistant drafter was slower than no drafter on every target, and `E2B-it-assistant` reached 395 tok/s at one request on W8A8 (`~/gemma4-dev/jev-tpu-v5e1/README.md`). Its acceptance on a bf16 target is unmeasured; the q4w4a16 repack is the QAT-matched target.

#### JAX survey paper

Enough is measured for an E2B comparison on v5e-1 (`tpu-jaxrust-v5e1-2b/benchmarks/runs/2026-11-03-serve-v5e1/`, `tpu-jax-v5e1-2b/benchmarks/runs/2026-11-03-ab-v5e1/`). Gaps:

- [ ] 💳 26B on v6e-1 (`tpu-jax-v6e1-26b-q4_0`): only a memory plan exists (15.25 GB), no decode rate.
- [ ] 💳 E2B q4w4a16 on v6e-1 (`tpu-jax-v6e1-2b-q4w4a16`): only a checkpoint header check.
- [ ] 💳 E2B bf16 and the Rust engine on v6e-1, timed with `tools/serve_bench.py` like the v5e run.
- [ ] 💳 4B w4a16 on v5e-1 (`tpu-jax-v5e1-4b-w4a16`): never run. Its `server.py` still has the old v5e provisioning bug the three `tpu.env` rigs fixed; port that first.
- [ ] Check the 12B v6e-1 report's 33.9 tok/s against its raw files before quoting it (`benchmarks/runs/2026-07-31-gemma4-12b-v6e1`).
- [ ] Use only the HTTP figures from the July E2B v6e-1 run (140.5 / 143.3 tok/s); its report says its decode table predates a working model.
- [ ] Inferentia2 (`tpu-jax-inf2-2b-w4a16`): a two-step decode only, no speed figure. One sentence at most, or a real run.
- [ ] PyTorch-Rust on v5e-1 (`~/torchtpu-dev/tpu-pytorchrust-v5e1-2b-q4w4a16emb4`): its 11.5–12.2 tok/s comes from demo notes; time it with `serve_bench.py` before it goes in a table.
- [ ] Say in the paper that the v6e figures (July) and the v5e figures (November) come from different software versions.

#### Engine gaps

- [ ] JAX and Rust engines: no concurrent requests, so no comparison with vLLM beyond one request.
- [ ] JAX and Rust engines: no accuracy suite run (vLLM rows have GSM8K, BFCL and the suite in `~/gemma4-dev/jev-tpu-v5e1`).
- [ ] Rust engine: greedy only, no 4-bit path.
- [ ] JAX engine: 0.852x vLLM over HTTP; where the time goes is unmeasured.
