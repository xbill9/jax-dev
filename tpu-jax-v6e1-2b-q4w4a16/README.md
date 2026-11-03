# Gemma 4 E2B QAT Q4_0 Repack on a Single TPU with Pure JAX

This rig serves `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` on one Cloud TPU v6e-1 through the pure-JAX engine of [`tpu-jax-v6e1-2b-w4a16`](../tpu-jax-v6e1-2b-w4a16). The checkpoint is Google's quantization-aware-trained (QAT) E2B, repacked from `google/gemma-4-E2B-it-qat-q4_0-unquantized` into the compressed-tensors W4A16 layout with each group's trained step recovered, so the stored int4 levels are the ones Google trained. The `-text` build drops the vision and audio towers, which the engine never reads.

The engine, loader and model are byte-for-byte the parent rig's. **The only change is the default checkpoint** (`MODEL_ID` in `jax_openai_server.py`, `MODEL_NAME` in `tpu.env`), which is why slot 5 of the directory name is the only slot that differs from the parent (`q4w4a16` against `w4a16`, per `~/gemma4-dev/NAMING.md`).

#### Why this checkpoint

Google's `-qat-w4a16-ct` export re-scales every group to max|w| / 7.5, which moves each QAT weight by about 6.7%. The repack keeps the trained grid. Measured in this engine on one Colab TPU v5e chip (`colab/runs/2026-10-09/02_e2b_repack_vs_stock_on_jax_output.ipynb`, against the multimodal repack `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct` at `da64116f`):

| Build | Relative error vs QAT values | Mean KL vs QAT model | Same top token | Decode tok/s | Greedy tokens equal to QAT |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Google `-qat-w4a16-ct` | 0.0667 | 0.07141 | 85.96% | 93.7 | 5 of 128 |
| Repack | 0.0019 | 0.00021 | 99.29% | 93.3 | 128 of 128 |

- 🟢 **342.6x lower KL** against the QAT model, at 0.996x the decode speed.
- 🟢 **Accuracy, under vLLM on one v5e chip:** repack 67.8%, Google's export 65.5% on the 3,880-record suite (+2.4 points, 95% range +1.4 to +3.4); bf16 E2B 68.5% (`~/gemma4-dev/QUANTIZATION.md`).
- 🟢 **Smaller download:** 6.56 GB of safetensors for `-text` (`315aec06`), against 8.32 GB for Google's export.
- ⚠️ **Not yet run on a v6e-1.** The engine files are unchanged since the commit the Colab run used, so the result above applies to this code. The v6e-1 throughput figures in the parent's reports were measured on Google's export.

#### What is verified for the `-text` build

`tools/check_checkpoint.py` range-reads every safetensors header from the Hub and applies the loader's rules: int32 `weight_packed`, bf16 `weight_scale` of shape `[out, in/32]`, every projection the loader looks up present, and the text decoder under `model.language_model.`. Output in `benchmarks/runs/2026-11-03-header-check/output.txt`:

```
xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text@315aec06: 1092 tensors, 276 packed, layers 35, kv-shared from 15
```

276 is every projection the E2B decoder has: 7 per layer across 35 layers, plus `k_proj`/`v_proj` on the 15 layers that own a KV cache, plus `per_layer_model_projection`. None falls back to bf16 and none is missing. The flat `gemma4_text` config parses through `config_from_hf`, which falls back to the top level when `text_config` is absent.

Not run: a load and generate of the `-text` build in JAX. Do that first on the chip (Step 2 below).

---

#### Step 1 — Provision a v6e-1

Use the bundled MCP server and skill (`tpu-jax-v6e1-2b-q4w4a16`, `tpu-jax-v6e1-2b-q4w4a16-management`) with `workload="jax"`. The startup script installs a current CPython and `jax[tpu]` and checks that a TPU device is visible. The checkpoint is public, so no Hugging Face token is needed.

```bash
./project-setup.sh /path/to/project --project <gcp-project-id>
make skill-install
```

#### Step 2 — Serve the repack

On the TPU VM:

```bash
python3 -m pip install fastapi uvicorn pydantic huggingface_hub safetensors \
  flax transformers sentencepiece 'jinja2>=3.1'

python3 jax_openai_server.py \
  --kv-cache-dtype int8 \
  --quant-mode w4a16 \
  --max-model-len 8192 \
  --port 8000
```

`--model` defaults to the repack. Pass `--model google/gemma-4-E2B-it-qat-w4a16-ct` to serve Google's export from the same process for an A/B.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text",
    "messages": [{"role": "user", "content": "Explain TPU buffer donation."}],
    "stream": true
  }'
```

#### Step 3 — Run the tests

```bash
python3 -m unittest discover -s tests -v
```

The CPU tests use a tiny synthetic checkpoint in the same packed layout. jaxlib needs a CPU with AVX2.

---

#### 🔎 Tip: other repacks

- **`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct`** (multimodal, `da64116f`) passes the same header check and is the build the Colab table measured. The engine skips its vision and audio tensors as each shard loads.
- **`-text-emb4` and `-text-ple4` do not load.** They store `embed_tokens` or the per-layer embeddings as int4, and the loader reads those tables as bf16 only. They need their own rig and a packed-embedding lookup.

#### Scope

One v6e-1 chip, batch-1 serving, text only, the engine as of the parent rig on 2026-11-03. Speculative decoding with Gemma 4's drafter models is not implemented in this engine.

---

#### Repository map

| Path | Purpose |
| :--- | :--- |
| `tpu.env` | Authoritative `MODEL_NAME`, `ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE` |
| `jax_openai_server.py` | OpenAI-compatible completions, chat, SSE, health and metrics; `MODEL_ID` is the repack |
| `jax_engine.py` | Stateful generation engine |
| `ports/gemma4/jax_e_model.py` | Gemma 4 E2B forward, cached decode, quantized KV |
| `ports/gemma4/jax_e_loader.py` | Torch-free safetensors W4A16 loader |
| `tools/check_checkpoint.py` | Header-only check of a Hub checkpoint against the loader |
| `tests/` | CPU correctness and API tests |
| `server.py`, `skills/`, `.claude/skills/` | MCP server and management skill for TPU provisioning |

Measurements of the engine itself (buffer donation, int8 KV, kernel sweeps, HTTP serving) live in the parent: [`../tpu-jax-v6e1-2b-w4a16/benchmarks/runs/`](../tpu-jax-v6e1-2b-w4a16/benchmarks/runs/).

#### References

- Parent rig: [`tpu-jax-v6e1-2b-w4a16`](../tpu-jax-v6e1-2b-w4a16)
- Repack tool: `~/gemma4-dev/jev-tpu-31b/repack_q4_0.py`
- Checkpoint: [xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text)
- Colab measurement: [`colab/notebooks/02_e2b_repack_vs_stock_on_jax.ipynb`](../colab/notebooks/02_e2b_repack_vs_stock_on_jax.ipynb)
- Naming: `~/gemma4-dev/NAMING.md`, slot 5, `q4w4a16`
