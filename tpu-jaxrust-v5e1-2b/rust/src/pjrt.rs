//! Rust over the C shim in `pjrt.c`: a process-wide PJRT client on the plugin's first device,
//! device buffers that free themselves, and compiled programs. No Python here.
use std::ffi::{CStr, CString, c_char, c_int, c_long, c_longlong, c_void};
use std::sync::OnceLock;

unsafe extern "C" {
    fn pj_last_error() -> *const c_char;
    fn pj_header_version(major: *mut c_int, minor: *mut c_int) -> c_int;
    fn pj_init(path: *const c_char, major: *mut c_int, minor: *mut c_int, n: *mut c_int) -> c_int;
    fn pj_compile(m: *const c_char, len: usize, o: *const c_char, olen: usize) -> *mut c_void;
    fn pj_num_outputs(exe: *mut c_void) -> c_long;
    fn pj_free_exe(exe: *mut c_void);
    fn pj_upload(dt: c_int, data: *const c_void, dims: *const i64, nd: usize) -> *mut c_void;
    fn pj_execute(
        exe: *mut c_void,
        args: *const *mut c_void,
        nargs: usize,
        outs: *mut *mut c_void,
        nd: *const i64,
        nnd: usize,
    ) -> c_int;
    fn pj_download(buf: *mut c_void, dst: *mut c_void, size: usize) -> c_int;
    fn pj_device_bytes(buf: *mut c_void) -> c_longlong;
    fn pj_free(buf: *mut c_void);
}

pub type Result<T> = std::result::Result<T, String>;

fn err(what: &str) -> String {
    let msg = unsafe { CStr::from_ptr(pj_last_error()) }.to_string_lossy();
    format!("{what}: {msg}")
}

/// `xla.CompileOptionsProto { executable_build_options { num_replicas: 1, num_partitions: 1 } }`
/// serialized by hand: field 3 (length 4) holding fields 4 and 5, varint 1 each. Empty options
/// leave both at 0, which the TPU plugin rejects ("Requested total cores: 0").
const COMPILE_OPTIONS: [u8; 6] = [0x1a, 0x04, 0x20, 0x01, 0x28, 0x01];

/// What `init` found: the header's and the plugin's PJRT API versions and the device count.
#[derive(Clone, Debug)]
pub struct Platform {
    pub plugin: String,
    pub header_api: (i32, i32),
    pub plugin_api: (i32, i32),
    pub devices: i32,
}

static PLATFORM: OnceLock<Platform> = OnceLock::new();

/// Opens the PJRT plugin at `path` once per process. Later calls return the first result;
/// a different path is an error, since one process holds one client.
pub fn init(path: &str) -> Result<Platform> {
    if let Some(p) = PLATFORM.get() {
        if p.plugin != path {
            return Err(format!("PJRT already initialized from {}", p.plugin));
        }
        return Ok(p.clone());
    }
    let c = CString::new(path).map_err(|e| e.to_string())?;
    let (mut hm, mut hn, mut pm, mut pn, mut n) = (0, 0, 0, 0, 0);
    unsafe {
        pj_header_version(&mut hm, &mut hn);
        if pj_init(c.as_ptr(), &mut pm, &mut pn, &mut n) != 0 {
            return Err(err("PJRT init"));
        }
    }
    let p = Platform {
        plugin: path.to_string(),
        header_api: (hm, hn),
        plugin_api: (pm, pn),
        devices: n,
    };
    let _ = PLATFORM.set(p.clone());
    Ok(p)
}

/// Element types this crate moves to and from the device. Codes match `to_pjrt` in pjrt.c.
#[derive(Clone, Copy, Debug)]
pub enum Elem {
    F32 = 0,
    BF16 = 1,
    S32 = 2,
}

/// A buffer on the device, freed on drop. PJRT buffers may be used from any thread.
pub struct Buffer(*mut c_void);
unsafe impl Send for Buffer {}

impl Buffer {
    /// Copies a dense row-major host array of `dims` to the device. `bytes` must hold exactly
    /// the product of `dims` elements of `elem`.
    pub fn upload(elem: Elem, bytes: &[u8], dims: &[i64]) -> Result<Buffer> {
        let size = match elem {
            Elem::BF16 => 2,
            _ => 4,
        };
        let n: i64 = dims.iter().product();
        if bytes.len() as i64 != n * size {
            return Err(format!(
                "upload: {} bytes for dims {dims:?} of {elem:?}",
                bytes.len()
            ));
        }
        let p = unsafe {
            pj_upload(
                elem as c_int,
                bytes.as_ptr().cast(),
                dims.as_ptr(),
                dims.len(),
            )
        };
        if p.is_null() {
            return Err(err("upload"));
        }
        Ok(Buffer(p))
    }

    pub fn scalar_i32(x: i32) -> Result<Buffer> {
        Buffer::upload(Elem::S32, &x.to_le_bytes(), &[])
    }

    pub fn vec_i32(xs: &[i32]) -> Result<Buffer> {
        let bytes: Vec<u8> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
        Buffer::upload(Elem::S32, &bytes, &[xs.len() as i64])
    }

    pub fn zeros_bf16(dims: &[i64]) -> Result<Buffer> {
        let n: i64 = dims.iter().product();
        Buffer::upload(Elem::BF16, &vec![0u8; (n * 2) as usize], dims)
    }

    /// Copies the buffer to the host, waiting for the device. `size` is its byte length.
    pub fn download(&self, size: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; size];
        if unsafe { pj_download(self.0, out.as_mut_ptr().cast(), size) } != 0 {
            return Err(err("download"));
        }
        Ok(out)
    }

    pub fn read_i32(&self) -> Result<i32> {
        let b = self.download(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_f32s(&self, n: usize) -> Result<Vec<f32>> {
        let b = self.download(n * 4)?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Bytes the buffer occupies on the device.
    pub fn device_bytes(&self) -> i64 {
        unsafe { pj_device_bytes(self.0) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { pj_free(self.0) }
    }
}

/// A compiled program on the device.
pub struct Exe {
    ptr: *mut c_void,
    pub outputs: usize,
}
unsafe impl Send for Exe {}

impl Exe {
    /// Compiles a StableHLO module given as MLIR text.
    pub fn compile(mlir: &str) -> Result<Exe> {
        let p = unsafe {
            pj_compile(
                mlir.as_ptr().cast(),
                mlir.len(),
                COMPILE_OPTIONS.as_ptr().cast(),
                COMPILE_OPTIONS.len(),
            )
        };
        if p.is_null() {
            return Err(err("compile"));
        }
        let n = unsafe { pj_num_outputs(p) };
        if n < 0 {
            unsafe { pj_free_exe(p) };
            return Err(err("compile: output count"));
        }
        Ok(Exe {
            ptr: p,
            outputs: n as usize,
        })
    }

    /// Runs the program. Arguments the module aliases to an output (`tf.aliasing_output`) are
    /// donated: their buffers are consumed and must not be read again, only dropped.
    pub fn run(&self, args: &[&Buffer]) -> Result<Vec<Buffer>> {
        let ptrs: Vec<*mut c_void> = args.iter().map(|b| b.0).collect();
        let mut outs: Vec<*mut c_void> = vec![std::ptr::null_mut(); self.outputs];
        let rc = unsafe {
            pj_execute(
                self.ptr,
                ptrs.as_ptr(),
                ptrs.len(),
                outs.as_mut_ptr(),
                std::ptr::null(),
                0,
            )
        };
        if rc != 0 {
            return Err(err("execute"));
        }
        Ok(outs.into_iter().map(Buffer).collect())
    }
}

impl Drop for Exe {
    fn drop(&mut self) {
        unsafe { pj_free_exe(self.ptr) }
    }
}
