# CLAUDE.md

Monorepo of pure-JAX Gemma 4 rigs, `github.com/xbill9/jax-dev`. Each top-level `tpu-jax-*` directory is a former standalone repo imported with full history on 2026-11-03; `colab/` is the former `xbill9/colab-jax`; `legacy/` holds two frozen repos whose live versions are in `~/gemma4-dev`.

- **Naming:** read `NAMING.md` here, then `~/gemma4-dev/NAMING.md` (the spec) before naming, renaming or adding a rig.
- **Rigs are siblings.** They share a template and have diverged. Read the rig's own `CLAUDE.md`, and run its tests from inside it (`python3 -m unittest discover -s tests -v`).
- **Only `tpu-jax-v6e1-2b-q4w4a16`, `tpu-jax-v5e1-2b-q4w4a16` and `tpu-jax-v5e1-2b` have a `tpu.env`** (their shared `server.py` loads it and derives runtime, quota metric and machine type from `ACCELERATOR_TYPE`; keep the three copies identical).
- **Rig names are unique within this tree.** `tpu-jax-v5e1-2b` here and `~/gemma4-dev/tpu-jax-v5e1-2b` are different rigs; that tree compares across Gemma runtimes, this one across JAX rigs (`NAMING.md`). Every other `server.py` defaults to `gemma-4-31B-it` / `v6e-8` / TP=8 regardless of the directory name; the values in `NAMING.md`'s inventory for those come from READMEs, retarget commits and the engine/server defaults.
- **Each rig registers under its own directory name**: MCP server `<rig>`, skill `<rig>-management`, zone cache `~/.cache/<rig>/`. The inf2 rig also ships its TPU template as `<rig>-tpu` (`NAMING.md`). The root `.claude-plugin/marketplace.json` lists every rig; keep it and each rig's `plugin.json` in step.
- **`legacy/` is frozen.** Change the live rig in gemma4-dev instead.
- Use the system `python3`; never create a virtualenv.
- **Git:** the git root is this monorepo, so `git add .` inside a rig stages only that rig. Commit straight to `main`; no branches or pull requests.
