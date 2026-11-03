//! Gemma 4 E2B in Rust on a TPU through PJRT, as a Python extension module named `jaxrust_rs`.
//!
//! The forward pass is written here and emitted as StableHLO text; libtpu's PJRT plugin, the
//! one JAX loads, compiles and runs it. No JAX, no Python and no XLA build in the model path.
//!
//! Five files, five jobs:
//! - `pjrt.c` / `pjrt.rs` are the PJRT C API calls the engine makes (client, compile, buffers,
//!   execute), against the vendored `pjrt_c_api.h`. No Python.
//! - `hlo.rs` emits StableHLO with JAX's bf16/f32 promotion. No Python.
//! - `weights.rs` reads safetensors through mmap. No Python.
//! - `gemma.rs` is the model, the KV cache and the decode loop. No Python.
//! - `bindings.rs` is every PyO3 item: the only place Python objects appear. `PYO3.md` beside
//!   `Cargo.toml` describes each crossing, and `python/jaxrust_rs.pyi` is the type stub.
mod bindings;
mod gemma;
mod hlo;
mod pjrt;
mod weights;

use pyo3::prelude::*;

/// The module Python imports as `jaxrust_rs`. Its name has to match the shared library's file
/// name (`jaxrust_rs.so`), which is why the build copies `libjaxrust_rs.so` to that name.
#[pymodule]
fn jaxrust_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bindings::register(m)
}
