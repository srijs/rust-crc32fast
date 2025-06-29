use bencher::Bencher;
use crc32fast::Hasher;
use rand::Rng;

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

fn bench_kilobyte_baseline(b: &mut Bencher) {
    bench(b, 1024, Hasher::internal_new_baseline(0, 0))
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

fn bench_combine(b: &mut Bencher) {
    let h1 = Hasher::new_with_initial_len(0x663DF39A, 0x6DD2EBDA9F9A0C29);
    let h2 = Hasher::new_with_initial_len(0x24DE685D, 0x5221FBD076875711);

    b.iter(|| {
        let mut h = h1.clone();
        h.combine(&h2);
        bencher::black_box(h);
    })
}

bencher::benchmark_group!(
    bench_baseline,
    bench_kilobyte_baseline,
    bench_megabyte_baseline
);
bencher::benchmark_group!(
    bench_specialized,
    bench_kilobyte_specialized,
    bench_megabyte_specialized
);
bencher::benchmark_group!(bench_combine_group, bench_combine);
bencher::benchmark_main!(bench_baseline, bench_specialized, bench_combine_group);
