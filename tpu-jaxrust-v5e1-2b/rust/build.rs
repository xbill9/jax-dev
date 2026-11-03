//! Compiles src/pjrt.c against the vendored PJRT C API header and records the toolchain. The
//! plugin (libtpu.so) is opened at run time with dlopen, so nothing here links against it.
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/pjrt.c");
    println!("cargo:rerun-if-changed=vendor/pjrt_c_api.h");
    cc::Build::new()
        .file("src/pjrt.c")
        .include("vendor")
        .flag("-std=c11")
        .warnings(true)
        .compile("pjrt_shim");
    println!("cargo:rustc-link-lib=dl");

    let header = std::fs::read_to_string("vendor/pjrt_c_api.h").unwrap();
    let ver = |key: &str| {
        header
            .lines()
            .find_map(|l| l.strip_prefix(&format!("#define {key} ")))
            .unwrap_or("?")
            .trim()
            .to_string()
    };
    println!(
        "cargo:rustc-env=JAXRUST_RS_PJRT_HEADER={}.{}",
        ver("PJRT_API_MAJOR"),
        ver("PJRT_API_MINOR")
    );
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let v = Command::new(rustc)
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=JAXRUST_RS_RUSTC={v}");
}
