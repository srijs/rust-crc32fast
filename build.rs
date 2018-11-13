extern crate rustc_version;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let version = rustc_version::version().unwrap();
    let target = ::std::env::var("TARGET").unwrap();

    if target.starts_with("x86") && version >= (1, 27, 0).into() {
        println!("cargo:rustc-cfg=pclmulqdq");
    }
}
