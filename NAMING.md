# Naming

Rig directories here follow the scheme in [`gemma4-dev/NAMING.md`](https://github.com/xbill9/gemma4-dev/blob/main/NAMING.md), the one spec for every rig tree (`~/gemma4-dev`, `~/torchtpu-dev`, `~/jax-dev`):

```
<platform>-<runtime>-<hardware>-<model>[-<encoding>]
```

Read that file before naming, renaming or adding a rig. This file records only what is specific to this tree.

#### Inventory

| Directory | Imported from | Platform | Hardware | Checkpoint | Notes |
| --- | --- | --- | --- | --- | --- |
| `tpu-jax-v6e1-2b-w4a16` | `xbill9/tpu-jax` | Cloud TPU API (queued resources) | v6e-1 | `google/gemma-4-E2B-it-qat-w4a16-ct` | The original pure-JAX engine; also carries the merged inf2 port |
| `tpu-jax-v5e1-4b-w4a16` | `xbill9/tpu-jax-4b` | Cloud TPU API | v5e-1 | `google/gemma-4-E4B-it-qat-w4a16-ct` | Its README still says v6e-1; the engine and server were retargeted to v5e-1 on 2026-08-05 |
| `tpu-jax-v6e1-12b-w4a16` | `xbill9/tpu-jax-12b` | Cloud TPU API | v6e-1 | `google/gemma-4-12B-it-qat-w4a16-ct` | Engine code. Its measurements live in the artifact rig of the same name in gemma4-dev |
| `tpu-jax-v6e1-26b-q4_0` | `xbill9/tpu-jax-26b` | Cloud TPU API | v6e-1 (target) | `gemma-4-26B-A4B-it-qat-q4_0-unquantized` | Engine code. Verified on CPU only; `v6e1` names the target. Measurements in the gemma4-dev artifact rig of the same name |
| `tpu-jax-v6e1-31b-w4a16` | `xbill9/tpu-jax-31b` | Cloud TPU API | v6e-1 | `google/gemma-4-31B-it-qat-w4a16-ct` | Engine code. Measurements in the gemma4-dev artifact rig of the same name |
| `tpu-jax-inf2-2b-w4a16` | `xbill9/tpu-jax-inf2` | AWS Neuron (slot `tpu`) | inf2 | `google/gemma-4-E2B-it-qat-w4a16-ct` | JAX on Inferentia2 (`jax_neuron/`). gemma4-dev's `tpu-jax-inf2-2b` is a separate rig serving bf16 `gemma-4-E2B-it` |
| `legacy/tpu-jax-v5e1-2b` | `xbill9/tpu-jax-v5e1-2b` | Cloud TPU API | v5e-1 | `google/gemma-4-E2B-it` | Frozen. The live rig is `gemma4-dev/tpu-jax-v5e1-2b`, imported 2026-08-06 and developed since |
| `legacy/tpu-jax-v6e1-2b` | `xbill9/tpu-jax-v6e1-2b` | Cloud TPU API | v6e-1 | `google/gemma-4-E2B-it-qat-w4a16-ct` | Frozen. The live rig is `gemma4-dev/tpu-jax-v6e1-2b`, migrated to Compute Engine 2026-08-19 |
| `colab/` | `xbill9/colab-jax` | — | Colab runtime | — | Outside the scheme: nothing in it is provisioned |

The old repos had no `tpu.env`; slot values come from each repo's README, its retarget commits and the defaults in `jax_openai_server.py` / `jax_engine.py`. Every `server.py` still defaults to `gemma-4-31B-it` on `v6e-8` with `TENSOR_PARALLEL_SIZE=8`, drift inherited from the template, so never read a value off `server.py` defaults.

#### Owed: derived names

The scheme requires the plugin, MCP server key and skill to be named after the directory (`<rig>`, `mcp__<rig>__*`, `<rig>-management`). Every rig here still registers as plugin `tpu-management`, MCP server `tpu-devops`, skill `tpu-management`, and shares `~/.cache/tpu_zones_status.md`. Two of them loaded at once collide, and `make skill-install` in one overwrites the other. Rename one rig per commit, following the sweep in gemma4-dev's NAMING.md ("The name is a claim about `tpu.env`"). Until that is done there is no root `marketplace.json`.

`legacy/` rigs are frozen and keep their old internal names.
