---
title: "Gemma 4 E2B in JAX on One TPU v5e: bf16 or 4-Bit, Python or Rust, and How It Stands Against vLLM"
published: false
description: "Gemma 4 E2B served three ways on one Cloud TPU v5e chip: a pure-JAX engine in Python, the same graph written in Rust and compiled by the libtpu that JAX loads, and vLLM, all timed by one client. bf16 beats 4-bit in the JAX engine, the Rust engine matches the JAX engine's tokens, and over HTTP it decodes at vLLM's rate with the fastest first token."
tags: jax, tpu, gemma, rust
cover_image: https://raw.githubusercontent.com/xbill9/jax-dev/main/article/devto-cover.3ce8c43b.jpg
---

This article provides a step by step guide to serving Gemma 4 E2B on a single Cloud TPU v5e chip with JAX. A pure-JAX engine serves the model from Python, a second engine builds the same graph in Rust and hands it to the libtpu compiler that JAX itself uses, and vLLM serves the same checkpoint as the reference. Every number below was measured on one flex-start v5e-1 VM on 2026-11-03, and every run is committed with its logs.

https://github.com/xbill9/jax-dev

---

#### Why One v5e Chip?

A v5e-1 is the smallest Cloud TPU you can rent: one chip, 16.909 GB of HBM, in a `ct5lp-hightpu-1t` VM. Gemma 4 E2B in bf16 puts 9.257 GB of weights on it, which leaves room for the KV cache with no quantization at all. That makes it the place to ask three questions with nothing else in the way:

- Does 4-bit pay off for a model that already fits in bf16?
- How much of the JAX engine's time is Python?
- Where does a hand-written JAX engine stand against vLLM on the same chip?

---

#### At This Point You Should Have…

- A Google Cloud project with the TPU API enabled and flex-start quota for v5e (`TPUV5sLitepodPerProjectPerZoneForTPUAPI`) in `us-west4-a`
- An authenticated `gcloud` with the alpha components
- The repository cloned, and its `tpu-jax-v5e1-2b` directory as the working directory

No Hugging Face token is needed: `google/gemma-4-E2B-it` and both 4-bit builds are public.

---

#### Step 1 — Provision a Flex-Start v5e-1

Each rig ships an MCP server, `server.py`, whose tools wrap the `gcloud` calls. The rig's `tpu.env` sets `ACCELERATOR_TYPE=v5litepod-1`, and the server derives the machine type from it. `workload="jax"` installs Python 3.13 and `jax[tpu]` on a bare VM with no Docker:

```text
$ python3 -c 'import asyncio, server as s; print(asyncio.run(s.create_tpu_vm_instance(instance_name="jax-v5e1-2b", zone="us-west4-a", accelerator="v5litepod-1", workload="jax", max_run_duration="4h", request_valid_for="30m")))'
Executing gcloud command: gcloud compute instances create jax-v5e1-2b --project=<project-id> --zone=us-west4-a --machine-type=ct5lp-hightpu-1t --provisioning-model=FLEX_START --request-valid-for-duration=30m --max-run-duration=4h --instance-termination-action=DELETE ... --boot-disk-size=200GB --scopes=cloud-platform
🚀 Flex-start TPU VM `jax-v5e1-2b` (ct5lp-hightpu-1t, 1 chip(s)) created in us-west4-a; installing Python 3.13 + `jax[tpu]`.
NAME         ZONE        MACHINE_TYPE      PREEMPTIBLE  STATUS
jax-v5e1-2b  us-west4-a  ct5lp-hightpu-1t               RUNNING
```

`wait_for_jax_ready` watches the serial console until the startup script has seen a TPU device from JAX:

```text
🟢 JAX environment ready on `jax-v5e1-2b`. Verify with `verify_jax_tpu`; run workloads over SSH with `python3.13`.

$ python3.13 -c "import jax; print(jax.__version__, jax.devices())"
0.12.0.1 [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
```

The flex-start VM deletes itself after `--max-run-duration`, so a forgotten VM stops billing on its own.

---

#### Step 2 — Load E2B Into the Pure-JAX Engine

`jax_engine.py` is a Gemma 4 E2B decoder written directly in JAX: jitted prefill, a static KV cache, and one jitted decode step per token. It reads the Hugging Face safetensors itself, so no PyTorch is installed. `--quant-mode fp16` is its dense path, which keeps the bf16 weights as they are:

```text
$ python3.13 jax_openai_server.py --model google/gemma-4-E2B-it --quant-mode fp16 --max-model-len 2048
JAX devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
Loading tokenizer: google/gemma-4-E2B-it
Loaded 9.26 GB of parameters on TPU_0(process=0,(0,0,0,0)) in 5.4s (KV cache: bf16)
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The server speaks the OpenAI chat API, streamed or not, so any OpenAI client can talk to it.

---

#### Step 3 — bf16 or 4-Bit?

Two 4-bit builds of E2B load into the same engine with `--quant-mode w4a16`: Google's `google/gemma-4-E2B-it-qat-w4a16-ct`, and `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`, a repack that stores the QAT grid exactly. `tools/ab_bench.py` runs each in its own process on 8 prompts, 128 greedy tokens each, one untimed run and three timed:

```text
loaded google/gemma-4-E2B-it in 64.7s, weights 9.257 GB, HBM 9.260 / 16.909 GB
loaded xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text in 39.2s, weights 6.559 GB, HBM 6.561 / 16.909 GB
loaded google/gemma-4-E2B-it-qat-w4a16-ct in 51.9s, weights 6.559 GB, HBM 6.562 / 16.909 GB

arm                                                tok/s  x ref weights GB  HBM GB
google/gemma-4-E2B-it (fp16)                       132.6  1.000      9.257   9.260
xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text (w4a16)   103.0  0.777      6.559   6.561
google/gemma-4-E2B-it-qat-w4a16-ct (w4a16)         103.1  0.777      6.559   6.562
```

bf16 decodes 1.29x faster than either 4-bit build and prefills in 7.4 ms against 23.2 ms. The 4-bit builds save 2.70 GB of HBM. This engine unpacks the int4 weights inside every matmul, and on a chip where bf16 already fits, that unpacking costs more time than the smaller weights save.

The repack and Google's export tie on speed and size. Their outputs differ: none of the 8 completions match token for token, and they agree for 26.6 of 128 tokens on average. Which of the two is closer to the trained QAT weights is measured in the earlier article listed under References.

---

#### Step 4 — The Same Graph, Built From Rust

JAX traces Python into StableHLO and hands it to the TPU's PJRT plugin, `libtpu.so`, to compile and run. The `tpu-jaxrust-v5e1-2b` rig writes that StableHLO directly. `rust/src/gemma.rs` builds the forward pass op by op, following `jax_e_model.py` down to where JAX promotes bf16 to f32:

```rust
fn rms_norm(b: &mut Builder, x: &V, w: Option<&V>, eps: f64) -> V {
    let dt = x.dt;
    let last = x.rank() as i64 - 1;
    let d = x.dim(-1) as f64;
    let xf = b.convert(x, DType::F32);
    let sq = b.mul(&xf, &xf);
    let sum = b.reduce_sum(&sq, &[last]);
    let var = b.scale(&sum, 1.0 / d);
    let var = b.add_scalar(&var, eps);
    let r = b.rsqrt(&var);
    ...
```

`rust/src/hlo.rs` prints the module as MLIR text, and `rust/src/pjrt.c`, a small C shim over the vendored PJRT C API header, opens the same `libtpu.so` that JAX loaded and compiles it. A PyO3 module, `jaxrust_rs`, hands the result to Python for the tokenizer:

```text
platform {'plugin': '/usr/local/lib/python3.13/dist-packages/libtpu/libtpu.so', 'header_api': (0, 117), 'plugin_api': (0, 116), 'devices': 1}
build {'rustc': 'rustc 1.101.0-nightly (1c3a65069 2026-11-02)', 'pjrt_header': '0.117', 'crate': '0.1.0'}
engine Engine(context=256, weight_bytes=9257336832, cache_bytes=4718592) in 8.4s, {'load_s': 2.318838774, 'compile_s': 5.73853671}
```

The weights upload in 2.3 s and the decode program compiles in 5.7 s. There is no JAX in this path, and the Rust crate has no ML dependencies: `pyo3`, `safetensors`, `memmap2` and `serde_json`.

---

#### Step 5 — Does Rust Produce the Same Tokens as JAX?

`tools/parity.py` runs each engine in its own process on the same 8 prompts, with the same chat template and the same leading `<bos>`, and compares the last-position logits after the first prompt and every greedy token:

```text
logits after prompt 0: max |diff| 0.2306, mean |diff| 0.07346, argmax 7216 vs 7216, top-10 overlap 10/10, KL(ref||got) 1.31e-06
prompts tokenized the same: 8 of 8; completions identical: 7 of 8; matching prefix per prompt: [128, 128, 128, 128, 128, 128, 42, 128] (mean 117.2 of 128)
decode tok/s (median over prompts): jax 131.6, rust 150.7, ratio 1.145
weights on device: jax 9.257 GB, rust 9.257 GB
```

Both engines pick the same next token with the same top 10, 1.31e-6 apart in KL divergence. Seven of eight 128-token completions match token for token; the eighth parts at token 42, where "Time until 4:00 pm" and "Next full hour: 4:00 pm" were nearly tied. Timed inside each engine, Rust decodes 1.145x faster on identical weights.

---

#### Step 6 — Serve All Three Through One Client

The Rust engine gets an OpenAI-style server of its own, `rust_openai_server.py`, and vLLM 0.33 runs from the `vllm/vllm-tpu` image on the same VM. Each server takes port 8000 alone, and `tools/serve_bench.py` times all three the same way: streamed chat, one request at a time, temperature 0, exactly 128 tokens (`ignore_eos`), context 2048, 8 prompts with one untimed and three timed requests each:

```text
$ python3 tools/serve_bench.py compare vllm.json jax.json rust.json
engine      decode tok/s           range  first token ms  tokens  same text as vllm
vllm               148.7   148.7-148.7              13.0     128              8 of 8
jax                126.7   126.0-128.4              12.5     128              6 of 8
rust               148.4   147.8-148.6               8.9     128              6 of 8
prompt tokens per engine: {'vllm': [20, 23, 23, 21, 20, 24, 35, 17], 'jax': [20, 23, 23, 21, 20, 24, 35, 17], 'rust': [20, 23, 23, 21, 20, 24, 35, 17]}
repeats identical: {'vllm': 8, 'jax': 8, 'rust': 8}
ratio jax / vllm: 0.852
ratio rust / vllm: 0.998
```

The Rust engine decodes at vLLM's rate, 148.4 against 148.7 tokens per second, and reaches the first token in 8.9 ms against 13.0. The Python JAX engine decodes at 0.852x vLLM with a first token level with it. All three tokenize every prompt to the same length, and each repeats its own text exactly. The JAX and Rust engines part from vLLM at the same two near-ties, in prompts 2 and 6.

---

#### 🔎 Tip: Time Every Engine by the Same Clock

Engines report their own speed in their own way: inside the decode loop, per HTTP response, with or without the first token. One client gives every engine the same definition. In `serve_bench.py`, decode speed is completion tokens minus one, over the time from the first text chunk to the finish chunk, with the token count read from the server's usage chunk. The clock stops at the finish chunk because, past the end of the answer, `ignore_eos` tokens can decode to empty text. The same Rust engine reads 150.7 tokens per second timed inside the engine at context 256, and 148.4 through this client at context 2048.

---

#### 🔎 Tip: libtpu Wants One Replica and One Partition

PJRT's compile call takes a serialized `xla.CompileOptionsProto`. With empty options, libtpu asks for zero cores and refuses to compile ("Requested total cores: 0"). Six bytes set one replica and one partition:

```rust
/// `xla.CompileOptionsProto { executable_build_options { num_replicas: 1, num_partitions: 1 } }`
/// serialized by hand: field 3 (length 4) holding fields 4 and 5, varint 1 each.
const COMPILE_OPTIONS: [u8; 6] = [0x1a, 0x04, 0x20, 0x01, 0x28, 0x01];
```

Two rules of the MLIR text format surface at the same compile call: a float literal needs a decimal point (`2.0e0`), and an empty integer array is written `array<i64>`.

---

#### Compare and Contrast

| One v5e chip, E2B | JAX engine, bf16 | JAX engine, 4-bit | Rust engine, bf16 | vLLM 0.33, bf16 |
|---|---:|---:|---:|---:|
| Decode tok/s, timed in the engine | 🥈 131.6 | 🥉 103.0 | 🥇 150.7 | — |
| Decode tok/s, one client | 🥉 126.7 | not run | 🥈 148.4 | 🥇 148.7 |
| First token, one client | 12.5 ms | not run | 🥇 8.9 ms | 13.0 ms |
| Weights on the chip | 9.257 GB | 🥇 6.559 GB | 9.257 GB | — |
| Same text as vLLM | 6 of 8 | not run | 6 of 8 | — |
| Concurrent requests | no | no | no | yes |

---

#### So, Which One?

For one user at a time, the Rust engine: it decodes at vLLM's rate, answers first, and puts the same 9.257 GB on the chip. For JAX work in Python, the bf16 JAX engine: 0.852x vLLM, with the whole model written in JAX you can read and change. For more than one user, vLLM, which batches requests and is the only one of the three that does. On this chip keep E2B in bf16: the 4-bit builds save 2.70 GB and decode at 0.777x its speed in this engine.

---

#### Summary

The goal of this article was to measure Gemma 4 E2B in JAX on one TPU v5e chip: bf16 against 4-bit, Python against Rust, and both against vLLM. The key to the solution was holding everything else fixed: one VM, one checkpoint revision, the same 8 prompts through each server's own chat template, and one client timing every engine. The results were:

- 🟢 The Rust engine, the same graph built from Rust and compiled by JAX's own libtpu, decodes at vLLM's rate (0.998x) with the fastest first token, 8.9 ms
- 🟢 It matches the JAX engine: same argmax and top 10, KL 1.31e-6, 7 of 8 completions token for token
- 🟢 bf16 E2B fits a v5e chip (HBM 9.260 / 16.909 GB in use) and decodes 1.29x faster than 4-bit in the JAX engine
- ⚠️ The Python JAX engine decodes at 0.852x vLLM over HTTP; where its time goes was not measured
- ⚠️ The repack and Google's 4-bit export run at the same speed and size, with different outputs
- ❌ Neither JAX engine takes concurrent requests, so vLLM's batching is not compared here

Scope: one flex-start `ct5lp-hightpu-1t` VM in `us-west4-a` (one TPU v5 lite chip, 16.909 GB HBM) on 2026-11-03; JAX 0.12.0.1 and libtpu 0.0.51 for both JAX-side engines, vLLM 0.33.0rc1.dev430 from `vllm/vllm-tpu@sha256:f1f925f5…`; `google/gemma-4-E2B-it` at revision `4adf1574`. Every figure is the median over 8 prompts of the median of 3 timed runs, greedy, 128 tokens, one request at a time. The in-engine runs used context 4096 (bf16 against 4-bit: 132.6 tok/s for bf16) and 256 (JAX against Rust: 131.6 for bf16); the one-client run used 2048 for all three. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed run files.

The strategy for serving Gemma 4 E2B with JAX on a single TPU v5e chip was validated with an incremental step by step approach.

---

#### References

- The monorepo: https://github.com/xbill9/jax-dev
- The pure-JAX rig: https://github.com/xbill9/jax-dev/tree/main/tpu-jax-v5e1-2b
- The Rust rig: https://github.com/xbill9/jax-dev/tree/main/tpu-jaxrust-v5e1-2b
- bf16 against 4-bit run: https://github.com/xbill9/jax-dev/tree/main/tpu-jax-v5e1-2b/benchmarks/runs/2026-11-03-ab-v5e1
- JAX against Rust run: https://github.com/xbill9/jax-dev/tree/main/tpu-jaxrust-v5e1-2b/benchmarks/runs/2026-11-03-parity-v5e1
- One-client run: https://github.com/xbill9/jax-dev/tree/main/tpu-jaxrust-v5e1-2b/benchmarks/runs/2026-11-03-serve-v5e1
- The repack against Google's 4-bit export, on a Colab v5e: https://dev.to/gde/gemma-4-e2b-in-pure-jax-on-a-colab-tpu-googles-4-bit-export-against-an-exact-repack-4dle
- The model: https://huggingface.co/google/gemma-4-E2B-it
- The repack: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text
- Google's 4-bit export: https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct
- StableHLO specification: https://openxla.org/stablehlo/spec
- PJRT C API header: https://github.com/openxla/xla/blob/261d4d4979f65a89b614a6a5baae1790ce51f048/xla/pjrt/c/pjrt_c_api.h
