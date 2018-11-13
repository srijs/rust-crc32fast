#[macro_use]
extern crate bencher;
extern crate crc32fast;
extern crate rand;

use bencher::Bencher;
use crc32fast::Hasher;
use rand::Rng;

fn bench(b: &mut Bencher, size: usize, hasher_init: Hasher) {
    let mut bytes = vec![0u8; size];
    rand::thread_rng().fill_bytes(&mut bytes);

    b.iter(|| {
        let mut hasher = hasher_init.clone();
        hasher.update(&bytes);
        bencher::black_box(hasher.finalize())
    });

    b.bytes = size as u64;
}

fn bench_kilobyte_baseline(b: &mut Bencher) {
    bench(b, 1024, Hasher::internal_new_baseline())
}

fn bench_kilobyte_pclmulqdq(b: &mut Bencher) {
    #[cfg(pclmulqdq)]
    bench(b, 1024, Hasher::internal_new_pclmulqdq().unwrap())
}

fn bench_megabyte_baseline(b: &mut Bencher) {
    bench(b, 1024 * 1024, Hasher::internal_new_baseline())
}

fn bench_megabyte_pclmulqdq(b: &mut Bencher) {
    #[cfg(pclmulqdq)]
    bench(b, 1024 * 1024, Hasher::internal_new_pclmulqdq().unwrap())
}

benchmark_group!(
    bench_baseline,
    bench_kilobyte_baseline,
    bench_megabyte_baseline
);
benchmark_group!(
    bench_pclmulqdq,
    bench_kilobyte_pclmulqdq,
    bench_megabyte_pclmulqdq
);
benchmark_main!(bench_baseline, bench_pclmulqdq);
