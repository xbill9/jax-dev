# Gemma 4 E2B bf16 Baseline on a Single TPU v5e with Pure JAX

This rig serves `google/gemma-4-E2B-it`, the reference bf16 release, on one Cloud TPU v5e-1 through the pure-JAX engine of [`tpu-jax-v6e1-2b-w4a16`](../tpu-jax-v6e1-2b-w4a16). It is the baseline for the 4-bit rigs in this tree: same engine, same `server.py`, same chip as [`tpu-jax-v5e1-2b-q4w4a16`](../tpu-jax-v5e1-2b-q4w4a16), so a run on both measures the weights alone.

| | `tpu-jax-v5e1-2b` | `tpu-jax-v5e1-2b-q4w4a16` |
| :--- | :--- | :--- |
| Checkpoint | `google/gemma-4-E2B-it` (`4adf1574`) | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` (`315aec06`) |
| Weights on disk | 10.25 GB, one file, vision and audio included | 6.56 GB, text only |
| `--quant-mode` default | `fp16`: dense matmuls, weights bf16 | `w4a16`: int4 unpacked per matmul |
| `tpu.env` | `v5litepod-1`, `us-west4-a` | `v5litepod-1`, `us-west4-a` |

`~/gemma4-dev/tpu-jax-v5e1-2b` is a separate rig with the same name. That tree compares rigs across Gemma runtimes; this one compares JAX rigs against each other.

#### What is verified

`tools/check_checkpoint.py` range-reads the checkpoint's safetensors header from the Hub and applies the loader's rules (`benchmarks/runs/2026-11-03-header-check/output.txt`):

```
google/gemma-4-E2B-it@4adf1574: 2011 tensors, 0 packed, layers 35, kv-shared from 15
```

All 276 projections the E2B decoder has are present as dense bf16 weights, which is the path `--quant-mode fp16` runs, and none is missing. The engine drops the 1,411 vision and audio tensors as the file loads.

- 🟢 **A bf16 E2B fits a v5e chip in this engine.** Colab notebook 02 loaded Google's `-qat-q4_0-unquantized`, a bf16 E2B with the same tensor shapes, on one v5e: 9.26 GB of weights resident, 135.0 tok/s decode (`colab/runs/2026-10-09/02_e2b_repack_vs_stock_on_jax_output.ipynb`). The same run measured the 4-bit repack at 93.3 tok/s.
- ⚠️ **This checkpoint has not been loaded in JAX yet,** and this rig has not provisioned.

---

#### Step 1 — Provision a v5e-1

Use the bundled MCP server and skill (`tpu-jax-v5e1-2b`, `tpu-jax-v5e1-2b-management`) with `workload="jax"`. `tpu.env` makes it request `ct5lp-hightpu-1t` on the GCE path, or `v5litepod-1` with runtime `v2-alpha-tpuv5-lite` on the queued-resource path, in `us-west4-a`. The checkpoint is ungated, so no Hugging Face token is needed.

```bash
./project-setup.sh /path/to/project --project <gcp-project-id>
make skill-install
```

#### Step 2 — Serve bf16

On the TPU VM:

```bash
python3 -m pip install fastapi uvicorn pydantic huggingface_hub safetensors \
  flax transformers sentencepiece 'jinja2>=3.1'

python3 jax_openai_server.py \
  --kv-cache-dtype int8 \
  --max-model-len 8192 \
  --port 8000
```

`--model` defaults to `google/gemma-4-E2B-it` and `--quant-mode` to `fp16`. `/health` reports `"weights": "bf16"`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [{"role": "user", "content": "Explain TPU buffer donation."}],
    "stream": true
  }'
```

#### Step 3 — Run the tests

```bash
python3 -m unittest discover -s tests -v
```

The CPU tests use a tiny synthetic checkpoint. jaxlib needs a CPU with AVX2.

---

#### 🔎 Tip: the A/B

Serve this rig and `tpu-jax-v5e1-2b-q4w4a16` on the same chip type with the same flags apart from the checkpoint and `--quant-mode`, and compare decode tok/s, resident weight bytes (`tpu_jax_weight_bytes` on `/metrics`) and greedy agreement on the same prompts. Google's `-qat-w4a16-ct` loads in either rig with `--model google/gemma-4-E2B-it-qat-w4a16-ct --quant-mode w4a16` as a third arm.

#### Scope

One v5e-1 chip (16 GB HBM), batch-1 serving, text only, the engine as of the parent rig on 2026-11-03.

---

#### Repository map

| Path | Purpose |
| :--- | :--- |
| `tpu.env` | Authoritative `MODEL_NAME`, `QUANT_MODE`, `ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`, zone |
| `jax_openai_server.py` | OpenAI-compatible completions, chat, SSE, health and metrics; defaults to bf16 E2B and `fp16` |
| `jax_engine.py` | Stateful generation engine |
| `ports/gemma4/jax_e_model.py` | Gemma 4 E2B forward, cached decode, quantized KV |
| `ports/gemma4/jax_e_loader.py` | Torch-free safetensors loader, dense and W4A16 |
| `tools/check_checkpoint.py` | Header-only check of a Hub checkpoint against the loader |
| `tests/` | CPU correctness and API tests |
| `server.py`, `skills/`, `.claude/skills/` | MCP server and management skill; `server.py` is shared with the `q4w4a16` rigs |

Measurements of the engine itself live in the parent: [`../tpu-jax-v6e1-2b-w4a16/benchmarks/runs/`](../tpu-jax-v6e1-2b-w4a16/benchmarks/runs/).

#### References

- Parent rig: [`tpu-jax-v6e1-2b-w4a16`](../tpu-jax-v6e1-2b-w4a16)
- 4-bit siblings: [`tpu-jax-v5e1-2b-q4w4a16`](../tpu-jax-v5e1-2b-q4w4a16), [`tpu-jax-v6e1-2b-q4w4a16`](../tpu-jax-v6e1-2b-q4w4a16)
- Checkpoint: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- Colab measurement: [`colab/notebooks/02_e2b_repack_vs_stock_on_jax.ipynb`](../colab/notebooks/02_e2b_repack_vs_stock_on_jax.ipynb)
- Naming: `~/gemma4-dev/NAMING.md`; a bare four-slot name means the reference weights
