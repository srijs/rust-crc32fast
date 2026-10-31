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
        // SAFETY: The `State::new` constructor ensures that all
        //         required instructions are supported by the CPU.
        unsafe { calculate(state, buf) }
    }
}

// Bytes per stream in the 3-way loops. Merging the streams appends zero bytes to the earlier CRC
// states, which for a fixed length is a multiplication by a constant: four lookups in a table built
// at compile time. Shorter blocks would help cores with a slow `crc32x` (Apple M1), but cost a
// little on cores where one chain already keeps up (Neoverse N2), so the 3-way loop starts at
// 3 * SHORT = 1.5 KiB, as before.
const LONG: usize = 32768;
const SHORT: usize = 512;

// The 3-way loop advances each chain four words at a time.
const _: () = assert!(LONG % 32 == 0 && SHORT % 32 == 0);

static LONG_SHIFT: [[u32; 256]; 4] = crate::combine::shift_table(LONG as u64);
static SHORT_SHIFT: [[u32; 256]; 4] = crate::combine::shift_table(SHORT as u64);

// target_feature is necessary to allow rustc to inline the crc32* wrappers
#[target_feature(enable = "crc")]
pub unsafe fn calculate(crc: u32, data: &[u8]) -> u32 {
    let (pre_quad, mut quads, post_quad) = data.align_to::<u64>();

    // Unaligned leading bytes, then switch to the internal (bit-inverted) state.
    let mut c0 = !crc_bytes(crc, pre_quad);

    // A single `crc32d` chain is latency-bound: each instruction depends on the previous result.
    // While enough words remain, advance three independent chains over three consecutive blocks of
    // `n` words. B and C start from a zero state, so appending them to A is linear: shift A's state
    // past B's bytes and XOR in B's state, then the same for C.
    for &(n, shift) in &[(LONG / 8, &LONG_SHIFT), (SHORT / 8, &SHORT_SHIFT)] {
        while quads.len() >= 3 * n {
            let (a, rest) = quads.split_at(n);
            let (b, rest) = rest.split_at(n);
            let (c, rest) = rest.split_at(n);
            quads = rest;

            let mut c1 = 0;
            let mut c2 = 0;
            for ((qa, qb), qc) in a
                .chunks_exact(4)
                .zip(b.chunks_exact(4))
                .zip(c.chunks_exact(4))
            {
                for k in 0..4 {
                    c0 = arch::__crc32d(c0, qa[k]);
                    c1 = arch::__crc32d(c1, qb[k]);
                    c2 = arch::__crc32d(c2, qc[k]);
                }
            }
            c0 = crate::combine::shift(shift, c0) ^ c1;
            c0 = crate::combine::shift(shift, c0) ^ c2;
        }
    }

    // Remaining words, then unaligned trailing bytes.
    let r = crc_quads(!c0, quads);
    crc_bytes(r, post_quad)
}

#[target_feature(enable = "crc")]
unsafe fn crc_quads(crc: u32, quads: &[u64]) -> u32 {
    let mut c = !crc;

    // unrolling increases performance by a lot
    let mut quad_iter = quads.chunks_exact(8);
    for chunk in &mut quad_iter {
        c = arch::__crc32d(c, chunk[0]);
        c = arch::__crc32d(c, chunk[1]);
        c = arch::__crc32d(c, chunk[2]);
        c = arch::__crc32d(c, chunk[3]);
        c = arch::__crc32d(c, chunk[4]);
        c = arch::__crc32d(c, chunk[5]);
        c = arch::__crc32d(c, chunk[6]);
        c = arch::__crc32d(c, chunk[7]);
    }
    c = quad_iter
        .remainder()
        .iter()
        .fold(c, |acc, &q| arch::__crc32d(acc, q));

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
