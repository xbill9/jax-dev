//! Every PyO3 item in the crate. `gemma.rs`, `hlo.rs`, `pjrt.rs` and `weights.rs` never see a
//! Python object; this file turns Python arguments into Rust values on the way in and Rust
//! values into Python objects on the way out. `PYO3.md` describes each crossing.
use crate::gemma::{self, BUCKET, Config, Decoded, Gemma};
use crate::pjrt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

fn rt(e: String) -> PyErr {
    PyRuntimeError::new_err(e)
}

/// `init(plugin_path) -> dict`: opens the PJRT plugin (libtpu.so) once per process and reports
/// the header's and the plugin's PJRT API versions and the device count.
#[pyfunction]
fn init(py: Python<'_>, plugin: String) -> PyResult<Py<PyAny>> {
    let p = py.detach(move || pjrt::init(&plugin)).map_err(rt)?;
    let d = PyDict::new(py);
    d.set_item("plugin", p.plugin)?;
    d.set_item("header_api", p.header_api)?;
    d.set_item("plugin_api", p.plugin_api)?;
    d.set_item("devices", p.devices)?;
    Ok(d.into_any().unbind())
}

/// `selftest() -> list[float]`: compiles and runs y = x * 2 + 1 on x = [0, 1, 2, 3] through
/// the same emitter and PJRT path as the model. Needs `init` first.
#[pyfunction]
fn selftest(py: Python<'_>) -> PyResult<Vec<f32>> {
    py.detach(|| -> pjrt::Result<Vec<f32>> {
        use crate::hlo::{Builder, DType};
        let mut b = Builder::new();
        let x = b.param(&[4], DType::F32, None);
        let y = b.scale(&x, 2.0);
        let y = b.add_scalar(&y, 1.0);
        let exe = pjrt::Exe::compile(&b.finish(&[&y]))?;
        let bytes: Vec<u8> = [0f32, 1.0, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let xb = pjrt::Buffer::upload(pjrt::Elem::F32, &bytes, &[4])?;
        let out = exe.run(&[&xb])?;
        out[0].read_f32s(4)
    })
    .map_err(rt)
}

/// `emit(config_json, kind, s, t) -> str`: the StableHLO text of a program, for inspection.
/// `kind` is "prefill" or "decode". Shapes come from the checkpoint at load, so this takes
/// them from the same weight list the engine uses, read from `model_dir`.
#[pyfunction]
fn emit(model_dir: std::path::PathBuf, kind: &str, s: i64, t: i64) -> PyResult<String> {
    let text = std::fs::read_to_string(model_dir.join("config.json"))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cfg = Config::from_json(&json).map_err(PyValueError::new_err)?;
    let shapes = gemma::weight_shapes(&model_dir, &cfg).map_err(rt)?;
    match kind {
        "prefill" => Ok(gemma::prefill_module(&cfg, &shapes, s, t)),
        "decode" => Ok(gemma::decode_module(&cfg, &shapes, t)),
        _ => Err(PyValueError::new_err("kind must be prefill or decode")),
    }
}

/// What `Engine.decode` returns. Frozen: Python reads the fields and cannot change them.
#[pyclass(frozen, get_all, name = "DecodeResult")]
struct PyDecoded {
    /// Generated token ids, without the stop token.
    tokens: Vec<i64>,
    /// "stop", "length" or "cancelled".
    finish: &'static str,
    /// Seconds to read the prompt and produce the first token.
    prefill_s: f64,
    /// Seconds for the tokens after the first.
    decode_s: f64,
}

#[pymethods]
impl PyDecoded {
    fn __repr__(&self) -> String {
        format!(
            "DecodeResult(tokens=<{}>, finish={:?}, prefill_s={:.3}, decode_s={:.3})",
            self.tokens.len(),
            self.finish,
            self.prefill_s,
            self.decode_s
        )
    }
}

impl From<Decoded> for PyDecoded {
    fn from(d: Decoded) -> Self {
        PyDecoded {
            tokens: d.tokens,
            finish: d.finish.as_str(),
            prefill_s: d.prefill_s,
            decode_s: d.decode_s,
        }
    }
}

/// `Engine(model_dir, context)`: Gemma 4 E2B with every weight on the device and the decode
/// program compiled. `init` must have opened the plugin first.
///
/// `unsendable`: the engine owns one KV cache that one sequence at a time writes into. Any
/// thread other than the one that built it gets `pyo3_runtime.PanicException`, which derives
/// from BaseException. A server builds the engine on one dedicated thread and sends every
/// request to that thread.
#[pyclass(unsendable)]
struct Engine(Gemma);

impl Engine {
    fn check(&self, ids: &[i64], max_new: i64) -> PyResult<()> {
        if ids.is_empty() {
            return Err(PyValueError::new_err("ids is empty"));
        }
        if max_new < 1 {
            return Err(PyValueError::new_err("max_new must be at least 1"));
        }
        let need = ids.len() as i64 + max_new;
        if need > self.0.context() {
            return Err(PyValueError::new_err(format!(
                "{} prompt tokens + {max_new} new tokens = {need}, over the context of {}",
                ids.len(),
                self.0.context()
            )));
        }
        let bucket = (ids.len() as i64 + BUCKET - 1) / BUCKET * BUCKET;
        if bucket > self.0.context() {
            return Err(PyValueError::new_err(format!(
                "the prompt's {bucket}-token bucket is over the context of {}",
                self.0.context()
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl Engine {
    /// Reads the checkpoint, uploads every weight and compiles the decode program. `context`
    /// is the KV cache length in tokens, a positive multiple of 128. Runs without the GIL.
    #[new]
    fn new(py: Python<'_>, model_dir: std::path::PathBuf, context: i64) -> PyResult<Self> {
        if context <= 0 || context % BUCKET != 0 {
            return Err(PyValueError::new_err(format!(
                "context must be a positive multiple of {BUCKET}"
            )));
        }
        if !model_dir.is_dir() {
            return Err(PyValueError::new_err(format!(
                "{} is not a directory",
                model_dir.display()
            )));
        }
        let g = py
            .detach(move || Gemma::load(&model_dir, context))
            .map_err(rt)?;
        Ok(Engine(g))
    }

    #[getter]
    fn context(&self) -> i64 {
        self.0.context()
    }

    /// Bytes of weights on the device.
    #[getter]
    fn weight_bytes(&self) -> i64 {
        self.0.weight_bytes
    }

    /// Bytes of KV cache on the device.
    #[getter]
    fn cache_bytes(&self) -> i64 {
        self.0.cache_bytes()
    }

    /// `{"load_s": ..., "compile_s": ...}`: seconds to upload the weights and to compile the
    /// decode program.
    fn timings(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("load_s", self.0.load_s)?;
        d.set_item("compile_s", self.0.compile_s)?;
        Ok(d.into_any().unbind())
    }

    /// Reads `ids` and returns `(first_token, logits)`: the greedy next token and the last
    /// position's softcapped logits as little-endian f32 bytes.
    fn logits(&mut self, py: Python<'_>, ids: Vec<i64>) -> PyResult<(i64, Py<PyAny>)> {
        self.check(&ids, 1)?;
        let g = &mut self.0;
        let (tok, lg) = py.detach(|| g.read_prompt(&ids)).map_err(rt)?;
        let bytes: Vec<u8> = lg.iter().flat_map(|x| x.to_le_bytes()).collect();
        Ok((tok, PyBytes::new(py, &bytes).into_any().unbind()))
    }

    /// Greedy decode that ends at a token in `stop`, at `max_new`, or when `on_tokens`
    /// returns a false value. Each token is passed to `on_tokens(list[int])` with the GIL
    /// taken back for the call; if it raises, the decode stops and the exception is raised.
    #[pyo3(signature = (ids, max_new, stop=Vec::new(), on_tokens=None))]
    fn decode(
        &mut self,
        py: Python<'_>,
        ids: Vec<i64>,
        max_new: i64,
        stop: Vec<i64>,
        on_tokens: Option<Py<PyAny>>,
    ) -> PyResult<PyDecoded> {
        self.check(&ids, max_new)?;
        let g = &mut self.0;
        let mut failed: Option<PyErr> = None;
        let done = py.detach(|| {
            g.decode(&ids, max_new, &stop, |batch| {
                let Some(cb) = &on_tokens else { return true };
                Python::attach(|py| {
                    let call = PyList::new(py, batch).and_then(|l| cb.call1(py, (l,)));
                    match call.and_then(|r| r.is_truthy(py)) {
                        Ok(go_on) => go_on,
                        Err(e) => {
                            failed = Some(e);
                            false
                        }
                    }
                })
            })
        });
        if let Some(e) = failed {
            return Err(e);
        }
        Ok(done.map_err(rt)?.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "Engine(context={}, weight_bytes={}, cache_bytes={})",
            self.0.context(),
            self.0.weight_bytes,
            self.0.cache_bytes()
        )
    }
}

/// Adds every item above to the module. One list, so `PYO3.md` and the stub can be checked
/// against it.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let build = PyDict::new(py);
    build.set_item("rustc", env!("JAXRUST_RS_RUSTC"))?;
    build.set_item("pjrt_header", env!("JAXRUST_RS_PJRT_HEADER"))?;
    build.set_item("crate", env!("CARGO_PKG_VERSION"))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("BUILD", build)?;
    m.add("BUCKET", BUCKET)?;
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(selftest, m)?)?;
    m.add_function(wrap_pyfunction!(emit, m)?)?;
    m.add_class::<Engine>()?;
    m.add_class::<PyDecoded>()?;
    Ok(())
}
