use core::arch::aarch64 as arch;

#[derive(Clone)]
pub struct State(());

impl State {
    #[cfg(not(feature = "std"))]
    pub fn new() -> Option<Self> {
        if cfg!(target_feature = "crc") {
            // SAFETY: The conditions above ensure that all
            //         required instructions are supported by the CPU.
            Some(Self(()))
        } else {
            None
        }
    }

    #[cfg(feature = "std")]
    pub fn new() -> Option<Self> {
        if std::arch::is_aarch64_feature_detected!("crc") {
            // SAFETY: The condition above ensures that all
            //         required instructions are supported by the CPU.
            Some(Self(()))
        } else {
            None
        }
    }

    pub fn update(&self, state: u32, buf: &[u8]) -> u32 {
        // SAFETY: Calling a `#[target_feature]` function is only unsafe because the CPU might lack
        // that feature, and a `State` only exists if `new` found the `crc` feature.
        unsafe { calculate(state, buf) }
    }
}

// Conservative crossover: 64 words per stream gives the interleaved loop enough work to
// amortize its two `combine` calls (3 * 64 * 8 bytes = 1.5 KiB).
const MIN_TRIPLE_QUADS: usize = 192;

/// Reads the `i`-th 8-byte word of `b` as a little-endian `u64`, the byte order `crc32x` expects.
#[inline(always)]
fn quad(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap())
}

// target_feature is necessary to allow rustc to inline the crc32* wrappers
#[target_feature(enable = "crc")]
unsafe fn calculate(crc: u32, data: &[u8]) -> u32 {
    // Split the input into unaligned leading bytes, 8-byte-aligned words, and trailing bytes.
    let head = data.as_ptr().align_offset(8).min(data.len());
    let (pre_quad, rest) = data.split_at(head);
    let (quads, post_quad) = rest.split_at(rest.len() / 8 * 8);
    let n_quads = quads.len() / 8;

    // Running (finalized) CRC.
    let mut r = crc;

    // Unaligned leading bytes.
    r = crc_bytes(r, pre_quad);

    if n_quads >= MIN_TRIPLE_QUADS {
        // A single `crc32d` chain is latency-bound: each instruction depends on the previous
        // result. Splitting the aligned region into three equal, contiguous blocks and advancing
        // three independent accumulators in lockstep keeps several `crc32d` in flight, then
        // `combine` reassembles them (appending the zero-seeded later blocks to the first).
        let block_len = n_quads / 3 * 8;

        let (a, rest) = quads.split_at(block_len);
        let (b, rest) = rest.split_at(block_len);
        let (c, rest) = rest.split_at(block_len);

        // Internal (bit-inverted) states. A carries the running CRC; B and C start from zero.
        let mut c0 = !r;
        let mut c1 = !0u32;
        let mut c2 = !0u32;

        // Unroll each stream by 4 words while keeping the three chains independent.
        let mut chunks_a = a.chunks_exact(32);
        let mut chunks_b = b.chunks_exact(32);
        let mut chunks_c = c.chunks_exact(32);
        for ((qa, qb), qc) in (&mut chunks_a).zip(&mut chunks_b).zip(&mut chunks_c) {
            for k in 0..4 {
                c0 = arch::__crc32d(c0, quad(qa, k));
                c1 = arch::__crc32d(c1, quad(qb, k));
                c2 = arch::__crc32d(c2, quad(qc, k));
            }
        }
        for ((qa, qb), qc) in chunks_a
            .remainder()
            .chunks_exact(8)
            .zip(chunks_b.remainder().chunks_exact(8))
            .zip(chunks_c.remainder().chunks_exact(8))
        {
            c0 = arch::__crc32d(c0, quad(qa, 0));
            c1 = arch::__crc32d(c1, quad(qb, 0));
            c2 = arch::__crc32d(c2, quad(qc, 0));
        }

        r = crate::combine::combine(!c0, !c1, block_len as u64);
        r = crate::combine::combine(r, !c2, block_len as u64);

        // Words left over after the three equal blocks.
        r = crc_quads(r, rest);
    } else {
        r = crc_quads(r, quads);
    }

    // Unaligned trailing bytes.
    crc_bytes(r, post_quad)
}

/// CRCs a byte slice whose length is a multiple of 8, a word at a time.
#[target_feature(enable = "crc")]
unsafe fn crc_quads(crc: u32, quads: &[u8]) -> u32 {
    let mut c = !crc;

    // unrolling increases performance by a lot
    let mut quad_iter = quads.chunks_exact(64);
    for chunk in &mut quad_iter {
        c = arch::__crc32d(c, quad(chunk, 0));
        c = arch::__crc32d(c, quad(chunk, 1));
        c = arch::__crc32d(c, quad(chunk, 2));
        c = arch::__crc32d(c, quad(chunk, 3));
        c = arch::__crc32d(c, quad(chunk, 4));
        c = arch::__crc32d(c, quad(chunk, 5));
        c = arch::__crc32d(c, quad(chunk, 6));
        c = arch::__crc32d(c, quad(chunk, 7));
    }
    c = quad_iter
        .remainder()
        .chunks_exact(8)
        .fold(c, |acc, q| arch::__crc32d(acc, quad(q, 0)));

    !c
}

#[target_feature(enable = "crc")]
unsafe fn crc_bytes(crc: u32, bytes: &[u8]) -> u32 {
    let c = bytes.iter().fold(!crc, |acc, &b| arch::__crc32b(acc, b));
    !c
}

#[cfg(test)]
mod test {
    quickcheck::quickcheck! {
        fn check_against_baseline(init: u32, chunks: Vec<(Vec<u8>, usize)>) -> bool {
            let mut baseline = init;
            let mut aarch64 = init;
            let spec = super::State::new().expect("not supported");

            for (chunk, mut offset) in chunks {
                // simulate random alignments by offsetting the slice by up to 15 bytes
                offset &= 0xF;
                if chunk.len() <= offset {
                    baseline = crate::baseline::update_fast_16(baseline, &chunk);
                    aarch64 = spec.update(aarch64, &chunk);
                } else {
                    baseline = crate::baseline::update_fast_16(baseline, &chunk[offset..]);
                    aarch64 = spec.update(aarch64, &chunk[offset..]);
                }
            }
            aarch64 == baseline
        }
    }

    // Exercises the 3-way interleaved path across the sizes and alignments where it switches.
    #[test]
    fn check_large_inputs_against_baseline() {
        let mut data = vec![0u8; 8200];
        let mut s: u32 = 0x1234_5678;
        for byte in data.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *byte = (s >> 24) as u8;
        }

        for &len in &[
            0usize, 1, 7, 8, 15, 16, 63, 64, 127, 128, 1535, 1536, 1537, 4096, 8192, 8199,
        ] {
            for &offset in &[0usize, 1, 3, 7, 8, 15] {
                if offset + len > data.len() {
                    continue;
                }
                let slice = &data[offset..offset + len];
                let baseline = crate::baseline::update_fast_16(0, slice);
                let spec = super::State::new().expect("not supported");
                let specialized = spec.update(0, slice);

                assert_eq!(
                    specialized, baseline,
                    "mismatch for len={len} offset={offset}",
                );
            }
        }
    }
}
