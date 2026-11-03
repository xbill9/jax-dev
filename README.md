# jax-dev

Gemma 4 on pure JAX: one directory per rig, each serving one checkpoint on one accelerator shape, plus the Colab notebooks.

Each rig directory is a former standalone repo, imported here with its full git history (`git log -- <dir>`). Directory names follow the rig naming scheme; [`NAMING.md`](NAMING.md) maps every old repo to its new directory.

| Directory | What it is |
| --- | --- |
| [`tpu-jax-v6e1-2b-w4a16`](tpu-jax-v6e1-2b-w4a16) | Gemma 4 E2B QAT W4A16 on one TPU v6e chip, the original pure-JAX engine |
| [`tpu-jax-v5e1-4b-w4a16`](tpu-jax-v5e1-4b-w4a16) | Gemma 4 E4B QAT W4A16 on one TPU v5e chip |
| [`tpu-jax-v6e1-12b-w4a16`](tpu-jax-v6e1-12b-w4a16) | 12B QAT W4A16 port of the engine, TPU v6e-1 |
| [`tpu-jax-v6e1-26b-q4_0`](tpu-jax-v6e1-26b-q4_0) | 26B A4B port, verified on CPU, targeting TPU v6e-1 |
| [`tpu-jax-v6e1-31b-w4a16`](tpu-jax-v6e1-31b-w4a16) | 31B QAT W4A16 port, TPU v6e-1 |
| [`tpu-jax-inf2-2b-w4a16`](tpu-jax-inf2-2b-w4a16) | E2B QAT W4A16 through JAX on AWS Inferentia2 |
| [`colab/`](colab) | Apache-2.0 Colab notebooks: JAX on TPU, and serving Gemma 4 E2B on one chip |
| [`legacy/`](legacy) | Frozen copies of `tpu-jax-v5e1-2b` and `tpu-jax-v6e1-2b`; the live rigs are in [gemma4-dev](https://github.com/xbill9/gemma4-dev) |

Install any rig as a Claude Code plugin with `/plugin marketplace add xbill9/jax-dev`; each brings a management skill and an MCP server named after its directory.

The rigs are siblings: they share ancestry and have diverged, and nothing is imported across a rig boundary. Read a rig's own `CLAUDE.md` before working in it.

Related trees: [gemma4-dev](https://github.com/xbill9/gemma4-dev) (vLLM, JAX, PyTorch and llama.cpp rigs across TPU, GPU and local hardware) and [torchtpu-dev](https://github.com/xbill9/torchtpu-dev) (TorchTPU).

#### License

Apache-2.0, see [`LICENSE`](LICENSE).
