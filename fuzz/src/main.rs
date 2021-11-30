#[macro_use]
extern crate afl;
extern crate crc32fast;

use crc32fast::Hasher;

fn main() {
    let hasher_init = Hasher::internal_new_specialized(0, 0).unwrap();
    fuzz!(|data: &[u8]| {
        let mut hasher = hasher_init.clone();
        hasher.update(data);
        hasher.finalize();
    });
}
