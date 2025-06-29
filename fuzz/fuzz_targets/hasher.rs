#![no_main]

use crc32fast::Hasher;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut hasher = Hasher::internal_new_specialized(0, 0).unwrap();
    hasher.update(data);
    hasher.finalize();
});
