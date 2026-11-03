# Gemma 4 E2B in Rust on a TPU v5e, through PJRT

This rig serves `google/gemma-4-E2B-it` (bf16) on one Cloud TPU v5e chip with the forward pass written in Rust. The graph is built in `rust/src/gemma.rs`, emitted as StableHLO text by `rust/src/hlo.rs`, and compiled and run by libtpu's PJRT plugin, the same library JAX loads, through a small C shim over the PJRT C API. A PyO3 module, `jaxrust_rs`, hands it to Python for the tokenizer and the serving loop. There is no JAX, no XLA build and no third-party ML crate in the model path.

The graph follows the pure-JAX engine of [`tpu-jax-v5e1-2b`](../tpu-jax-v5e1-2b) operation for operation, including where JAX promotes bf16 to f32, so the two can be compared token for token on the same chip.

#### Measured on one v5e chip

`benchmarks/runs/2026-11-03-parity-v5e1/`: same checkpoint, chip, 8 prompts, 128 greedy tokens each, context 256, each engine in its own process.

| | JAX engine (`tpu-jax-v5e1-2b`) | Rust engine (this rig) |
| :--- | ---: | ---: |
| Decode tok/s, median | 131.6 | **150.7** |
| Weights on device | 9.257 GB | 9.257 GB |
| Completions identical to the JAX engine | | 7 of 8 |

- 🟢 **1.145x the JAX engine's decode rate.**
- 🟢 **Same answers.** After the first prompt both pick the same token with the same top 10; KL between their distributions is 1.31e-6. Seven of eight 128-token completions match token for token.
- ⚠️ **One prompt diverges at token 42**, at a near-tie between two phrasings; both continue coherently.
- 🟢 **Fast start.** Weights upload in 2.3 s from the local Hub cache, and the decode program compiles in 5.7 s.

---

#### Step 1 — Provision a v5e-1

Use the bundled MCP server and skill (`tpu-jaxrust-v5e1-2b`, `tpu-jaxrust-v5e1-2b-management`) with `workload="jax"`, which installs Python 3.13 and `jax[tpu]`. The engine needs only the `libtpu` that comes with it; `tpu.env` names its path as `PJRT_PLUGIN`. `tpu.env` makes the server request `ct5lp-hightpu-1t` in `us-west4-a`.

```bash
./project-setup.sh /path/to/project --project <gcp-project-id>
make skill-install
```

#### Step 2 — Build the module on the VM

```bash
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain nightly
. ~/.cargo/env
PYO3_PYTHON=python3.13 make rust     # rust/build/jaxrust_rs.so
```

The build needs a C compiler and no TPU: `build.rs` compiles `rust/src/pjrt.c` against the vendored `rust/vendor/pjrt_c_api.h`, and the plugin is opened at run time.

#### Step 3 — Run it

```python
import sys; sys.path.insert(0, "rust/build")
import jaxrust_rs
from huggingface_hub import snapshot_download

jaxrust_rs.init("/usr/local/lib/python3.13/dist-packages/libtpu/libtpu.so")
print(jaxrust_rs.selftest())                       # [1.0, 3.0, 5.0, 7.0]
path = snapshot_download("google/gemma-4-E2B-it", allow_patterns=["*.json", "*.safetensors"])
eng = jaxrust_rs.Engine(path, 1024)                # context: a multiple of 128
r = eng.decode(prompt_ids, 128, stop=[1, 106])     # ids from the tokenizer's chat template
```

`rust/PYO3.md` describes every call and how values cross between Python and Rust.

#### Step 4 — Check it against the JAX engine

```bash
python3.13 tools/parity.py jax  --out jax.json     # needs ../tpu-jax-v5e1-2b beside this rig
python3.13 tools/parity.py rust --out rust.json
python3 tools/parity.py compare jax.json rust.json
```

---

#### 🔎 Tip: reading a compile error

PJRT reports a bad module as a line and column. `jaxrust_rs.emit(model_dir, "prefill", 128, 256)` (or `"decode"`) returns the exact text the engine compiles, so the failing line can be read directly. Two rules came out of the first run: float literals need a decimal point (`2.0e0`), and an empty array attribute is `array<i64>`.

#### Scope

One v5e chip, one sequence at a time, greedy decoding, text only, bf16 weights and KV cache. The engine has no HTTP server yet, no sampling beyond argmax, and no 4-bit path; the prefill compiles once per 128-token prompt bucket.

---

#### Repository map

| Path | Purpose |
| :--- | :--- |
| `rust/src/gemma.rs` | Config, weight list, the forward graph, prefill and decode programs, the decode loop |
| `rust/src/hlo.rs` | StableHLO emitter with JAX's bf16/f32 promotion |
| `rust/src/pjrt.c`, `rust/src/pjrt.rs` | PJRT C API shim and its Rust wrapper |
| `rust/src/weights.rs` | safetensors through mmap |
| `rust/src/bindings.rs` | Every PyO3 item |
| `rust/vendor/pjrt_c_api.h` | PJRT C API 0.117, openxla/xla `261d4d49` |
| `tools/parity.py` | JAX engine against the Rust engine |
| `tests/test_rust_bindings.py` | Binding tests that need no TPU |
| `tpu.env` | Authoritative `MODEL_NAME`, `PJRT_PLUGIN`, `ACCELERATOR_TYPE`, zone |
| `server.py`, `skills/`, `.claude/skills/` | MCP server and management skill, shared with the sibling rigs |

#### References

- JAX sibling: [`tpu-jax-v5e1-2b`](../tpu-jax-v5e1-2b)
- Torch counterpart: `~/torchtpu-dev/tpu-pytorchrust-v5e1-2b-q4w4a16emb4` (Rust over ATen, PyO3)
- [StableHLO spec](https://openxla.org/stablehlo/spec)
- [PJRT C API header](https://github.com/openxla/xla/blob/261d4d4979f65a89b614a6a5baae1790ce51f048/xla/pjrt/c/pjrt_c_api.h)
- Naming: `~/gemma4-dev/NAMING.md`, slot 2 `jaxrust`
