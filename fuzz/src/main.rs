
use crc32fast::Hasher;

fn main() {
    let hasher_init = Hasher::internal_new_specialized(0, 0).unwrap();
    afl::fuzz!(|data: &[u8]| {
        let mut hasher = hasher_init.clone();
        hasher.update(data);
        hasher.finalize();
    });
}
