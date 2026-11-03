# The PyO3 bindings

The model is Rust. Python hosts the process and the tokenizer. [PyO3](https://github.com/PyO3/pyo3) 0.30 builds this crate as a native Python module, `jaxrust_rs`, and turns Rust functions and structs into Python functions and classes.

The TPU is reached through libtpu's PJRT plugin, the shared library JAX loads. The engine opens it with `dlopen`, so the module links nothing TPU-specific and builds anywhere; only `init` needs the plugin, and only `Engine` needs a TPU.

| File | Job | Python objects |
| --- | --- | --- |
| `src/pjrt.c` | Flat C functions over the PJRT C API: open the plugin, compile, upload, execute, download | none |
| `src/pjrt.rs` | `Buffer` and `Exe` over those functions, freed on drop | none |
| `src/hlo.rs` | The StableHLO emitter, with JAX's bf16/f32 promotion | none |
| `src/weights.rs` | safetensors through mmap, one tensor in host memory at a time | none |
| `src/gemma.rs` | The model graph, the KV cache, the decode loop | none |
| `src/bindings.rs` | Every PyO3 item | all of them |
| `python/jaxrust_rs.pyi` | The type stub | |

#### How the module is built and loaded

| Piece | Setting | Why |
| --- | --- | --- |
| `[lib] crate-type` | `cdylib` | A shared library Python can `import` |
| `pyo3` features | `extension-module` | Does not link libpython; the interpreter that imports the module provides it |
| `build.rs` | Compiles `src/pjrt.c` (C11) against `vendor/pjrt_c_api.h`, links `-ldl`, records the header's PJRT version and rustc | The plugin is a run-time choice, so the build needs no TPU and no JAX |
| Toolchain | `rust-toolchain.toml`: the newest nightly; edition 2024 | |
| `#[pymodule] fn jaxrust_rs` | in `src/lib.rs` | The function name is the module name and must match the file name: `make rust` copies `libjaxrust_rs.so` to `jaxrust_rs.so` |

`jaxrust_rs.BUILD` records the rustc, the PJRT header version and the crate version. `init` returns the plugin's own PJRT version beside the header's; a plugin older than the header works (`vendor/README.md`).

#### What Python can call

| Python | Rust | Arguments in | Result out |
| --- | --- | --- | --- |
| `jaxrust_rs.__version__` | `env!("CARGO_PKG_VERSION")` | | `str` |
| `jaxrust_rs.BUILD` | set by `build.rs` | | `dict`: `rustc`, `pjrt_header`, `crate` |
| `jaxrust_rs.BUCKET` | `gemma::BUCKET` | | `int`, 128 |
| `init(plugin)` | `#[pyfunction]` | `str` path to libtpu.so | `dict`: `plugin`, `header_api`, `plugin_api`, `devices` |
| `selftest()` | `#[pyfunction]` | | `list[float]`, `[1, 3, 5, 7]` |
| `emit(model_dir, kind, s, t)` | `#[pyfunction]` | path, `"prefill"` or `"decode"`, `int`, `int` | `str`, the StableHLO module |
| `Engine(model_dir, context)` | `#[pyclass(unsendable)]`, `#[new]` | path, `int` (multiple of 128) | an `Engine` |
| `Engine.context`, `.weight_bytes`, `.cache_bytes` | `#[getter]` | | `int` |
| `Engine.timings()` | `#[pymethods]` | | `dict`: `load_s`, `compile_s` |
| `Engine.logits(ids)` | `#[pymethods]` | `list[int]` | `tuple[int, bytes]`: next token, f32 logits |
| `Engine.decode(ids, max_new, stop=[], on_tokens=None)` | `#[pymethods]`, `#[pyo3(signature)]` | `list[int]`, `int`, `list[int]`, callable or `None` | `DecodeResult` |
| `DecodeResult` | `#[pyclass(frozen, get_all)]` | | `.tokens`, `.finish`, `.prefill_s`, `.decode_s` |

#### The crossings

**1. Token ids are plain Python lists.** `Vec<i64>` arguments accept any sequence of ints, and results come back as `list[int]`. Each generated token reaches the host as one 4-byte read of the device's argmax; logits cross only from `logits`, as `bytes` of little-endian f32 (`array.array("f", raw)` reads them).

**2. Device buffers never cross.** Weights, KV cache and compiled programs live in the `Engine` and are freed with it. Python sees only ids, bytes and numbers.

**3. Long calls release the GIL.** `init`, `selftest`, `Engine(...)`, `logits` and `decode` run inside `py.detach(...)`. A closure that runs without the GIL must be `Send`; `Buffer` and `Exe` are, because PJRT handles may be used from any thread.

**4. The callback takes the GIL back.** `decode` calls `on_tokens(list[int])` as each token reaches the host, inside `Python::attach(...)`. A truthy return continues; a false one ends the decode with `finish="cancelled"`. If the callback raises, the decode stops and `decode` raises that exception.

#### Errors

Argument problems (empty ids, context not a multiple of 128, too many tokens for the context, a missing directory) raise `ValueError` before any device work. Failures from the plugin (dlopen, compile, execute) raise `RuntimeError` with PJRT's message; a compile error names a line and column in the module `emit` returns. `Engine` called from a thread other than its creator's raises `pyo3_runtime.PanicException`, a `BaseException`.
