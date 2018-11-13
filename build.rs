extern crate rustc_version;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let version = rustc_version::version().unwrap();

    if version >= (1, 27, 0).into() {
        println!("cargo:rustc-cfg=crc32fast_stdarchx86");
    }
}
