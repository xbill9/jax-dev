# CLAUDE.md

Monorepo of pure-JAX Gemma 4 rigs, `github.com/xbill9/jax-dev`. Each top-level `tpu-jax-*` directory is a former standalone repo imported with full history on 2026-11-03; `colab/` is the former `xbill9/colab-jax`; `legacy/` holds two frozen repos whose live versions are in `~/gemma4-dev`.

- **Naming:** read `NAMING.md` here, then `~/gemma4-dev/NAMING.md` (the spec) before naming, renaming or adding a rig.
- **Rigs are siblings.** They share a template and have diverged. Read the rig's own `CLAUDE.md`, and run its tests from inside it (`python3 -m unittest discover -s tests -v`).
- **No `tpu.env` exists in these rigs yet**, and every `server.py` defaults to `gemma-4-31B-it` / `v6e-8` / TP=8 regardless of the directory name. The values in `NAMING.md`'s inventory come from READMEs, retarget commits and the engine/server defaults.
- **Every rig still registers as `tpu-devops` / `tpu-management`.** Do not load two at once. The per-rig rename is owed (`NAMING.md`).
- **`legacy/` is frozen.** Change the live rig in gemma4-dev instead.
- Use the system `python3`; never create a virtualenv.
- **Git:** the git root is this monorepo, so `git add .` inside a rig stages only that rig. Commit straight to `main`; no branches or pull requests.
