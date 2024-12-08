#[macro_use]
extern crate bencher;
extern crate crc32_v2;
extern crate crc32fast;
extern crate rand;

use bencher::Bencher;
use crc32_v2::crc32;
use crc32fast::Hasher;
use rand::Rng;

fn bench_crc32_v2(b: &mut Bencher, size: usize) {
    let mut bytes = vec![0u8; size];
    rand::thread_rng().fill(&mut bytes[..]);

    b.iter(|| {
        bencher::black_box(crc32(bencher::black_box(0), bencher::black_box(&bytes)));
    });

    b.bytes = size as u64;
}

fn bench_byte_crc32_v2(b: &mut Bencher) {
    bench_crc32_v2(b, 1)
}

fn bench_kilobyte_crc32_v2(b: &mut Bencher) {
    bench_crc32_v2(b, 1024)
}

fn bench_megabyte_crc32_v2(b: &mut Bencher) {
    bench_crc32_v2(b, 1024 * 1024)
}

fn bench(b: &mut Bencher, size: usize, hasher_init: Hasher) {
    let mut bytes = vec![0u8; size];
    rand::thread_rng().fill(&mut bytes[..]);

    b.iter(|| {
        let mut hasher = hasher_init.clone();
        hasher.update(&bytes);
        bencher::black_box(hasher.finalize())
    });

    b.bytes = size as u64;
}

fn bench_byte_baseline(b: &mut Bencher) {
    bench(b, 1, Hasher::internal_new_baseline(0, 0))
}

fn bench_kilobyte_baseline(b: &mut Bencher) {
    bench(b, 1024, Hasher::internal_new_baseline(0, 0))
}

fn bench_byte_specialized(b: &mut Bencher) {
    bench(b, 1, Hasher::internal_new_specialized(0, 0).unwrap())
}

fn bench_kilobyte_specialized(b: &mut Bencher) {
    bench(b, 1024, Hasher::internal_new_specialized(0, 0).unwrap())
}

fn bench_megabyte_baseline(b: &mut Bencher) {
    bench(b, 1024 * 1024, Hasher::internal_new_baseline(0, 0))
}

fn bench_megabyte_specialized(b: &mut Bencher) {
    bench(
        b,
        1024 * 1024,
        Hasher::internal_new_specialized(0, 0).unwrap(),
    )
}

benchmark_group!(
    bench_baseline,
    bench_byte_baseline,
    bench_kilobyte_baseline,
    bench_megabyte_baseline,
);
benchmark_group!(
    bench_specialized,
    bench_byte_specialized,
    bench_kilobyte_specialized,
    bench_megabyte_specialized
);
benchmark_group!(
    bench_crc32,
    bench_byte_crc32_v2,
    bench_kilobyte_crc32_v2,
    bench_megabyte_crc32_v2
);
benchmark_main!(bench_baseline, bench_specialized, bench_crc32);
