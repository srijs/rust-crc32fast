//! Specialized checksum code for the x86 CPU architecture, based on the efficient algorithm described
//! in the following whitepaper:
//!
//! Gopal, V., Ozturk, E., Guilford, J., Wolrich, G., Feghali, W., Dixon, M., & Karakoyunlu, D. (2009).
//! _Fast CRC computation for generic polynomials using PCLMULQDQ instruction_. Intel.
//! (Mirror link: <https://fossies.org/linux/zlib-ng/doc/crc-pclmulqdq.pdf>, accessed 2024-05-20)
//!
//! Throughout the code, this work is referred to as "the paper".
//!
//! On top of the 128-bit `PCLMULQDQ` implementation, two wider variants use `VPCLMULQDQ` to fold
//! several independent 128-bit streams per instruction: an `AVX2` variant over 256-bit `YMM`
//! registers (8 streams) and an `AVX-512` variant over 512-bit `ZMM` registers (16 streams). Both
//! rely on `VPCLMULQDQ` intrinsics stabilized in Rust 1.89 and are only compiled when the
//! `stable_vpclmulqdq` cfg is set by `build.rs`, leaving the crate MSRV unchanged otherwise. The
//! best variant supported by the running CPU is chosen at runtime.
//!
//! Each `unsafe fn` here is unsafe only because it needs the CPU features in its
//! `#[target_feature]` list (PCLMULQDQ for `reduce128`). None of them has a memory-safety
//! precondition: every load is bounds-checked.

#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

/// Which SIMD implementation to use, chosen once at construction from the CPU's features.
#[derive(Clone, Copy)]
enum Kind {
    /// 128-bit `PCLMULQDQ`, fold by 4.
    Sse,
    /// 256-bit `VPCLMULQDQ`, 8 streams.
    #[cfg(stable_vpclmulqdq)]
    Avx2,
    /// 512-bit `VPCLMULQDQ`, 16 streams.
    #[cfg(stable_vpclmulqdq)]
    Avx512,
}

#[cfg(feature = "std")]
impl Kind {
    /// Encodes `kind` for `State::detect`'s cache, which reserves 0 for "not detected yet".
    #[inline]
    fn encode(kind: Option<Kind>) -> u8 {
        match kind {
            None => 1,
            Some(Kind::Sse) => 2,
            #[cfg(stable_vpclmulqdq)]
            Some(Kind::Avx2) => 3,
            #[cfg(stable_vpclmulqdq)]
            Some(Kind::Avx512) => 4,
        }
    }

    /// Decodes what `encode` returned.
    #[inline]
    fn decode(code: u8) -> Option<Kind> {
        match code {
            2 => Some(Kind::Sse),
            #[cfg(stable_vpclmulqdq)]
            3 => Some(Kind::Avx2),
            #[cfg(stable_vpclmulqdq)]
            4 => Some(Kind::Avx512),
            _ => {
                debug_assert_eq!(code, 1, "code from `encode` has no `decode` arm");
                None
            }
        }
    }
}

#[derive(Clone)]
pub struct State {
    kind: Kind,
}

impl State {
    #[cfg(not(feature = "std"))]
    #[inline]
    fn detect() -> Option<Kind> {
        if cfg!(target_feature = "pclmulqdq")
            && cfg!(target_feature = "sse2")
            && cfg!(target_feature = "sse4.1")
            && cfg!(target_feature = "ssse3")
        {
            #[cfg(stable_vpclmulqdq)]
            {
                if cfg!(target_feature = "avx512f") && cfg!(target_feature = "vpclmulqdq") {
                    return Some(Kind::Avx512);
                }
                if cfg!(target_feature = "avx2") && cfg!(target_feature = "vpclmulqdq") {
                    return Some(Kind::Avx2);
                }
            }

            return Some(Kind::Sse);
        }

        None
    }

    /// The kernel for this CPU, if any. Finding it takes up to seven feature checks, each a load
    /// and a branch. `Hasher::new` asks every time, and `crc32fast::hash` constructs a `Hasher` for
    /// every call, so the answer is cached: after the first call, this is one load.
    #[cfg(feature = "std")]
    #[inline]
    fn detect() -> Option<Kind> {
        use core::sync::atomic::{AtomicU8, Ordering};

        // 0 = not detected yet, else `Kind::encode` of the result. Racing threads store the same
        // value, so relaxed ordering is enough.
        static CACHE: AtomicU8 = AtomicU8::new(0);
        match CACHE.load(Ordering::Relaxed) {
            0 => {
                let kind = Self::detect_uncached();
                CACHE.store(Kind::encode(kind), Ordering::Relaxed);
                kind
            }
            code => Kind::decode(code),
        }
    }

    #[cfg(feature = "std")]
    #[cold]
    fn detect_uncached() -> Option<Kind> {
        if is_x86_feature_detected!("pclmulqdq")
            && is_x86_feature_detected!("sse2")
            && is_x86_feature_detected!("sse4.1")
            && is_x86_feature_detected!("ssse3")
        {
            #[cfg(stable_vpclmulqdq)]
            {
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("vpclmulqdq") {
                    return Some(Kind::Avx512);
                }
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("vpclmulqdq") {
                    return Some(Kind::Avx2);
                }
            }

            return Some(Kind::Sse);
        }

        None
    }

    #[inline]
    pub fn new() -> Option<Self> {
        // SAFETY: `detect` only returns a `Kind` whose instructions the CPU supports.
        Self::detect().map(|kind| Self { kind })
    }

    #[inline]
    pub fn update(&self, state: u32, buf: &[u8]) -> u32 {
        // SAFETY: The kernels are only unsafe because the CPU might lack their target features.
        // `detect` checked every feature listed on the kernel for `self.kind`, and `kind` can't be
        // changed after `new`.
        unsafe {
            match self.kind {
                Kind::Sse => calculate(state, buf),
                #[cfg(stable_vpclmulqdq)]
                Kind::Avx2 => calculate_avx2(state, buf),
                #[cfg(stable_vpclmulqdq)]
                Kind::Avx512 => calculate_avx512(state, buf),
            }
        }
    }
}

const K1: i64 = 0x154442bd4;
const K2: i64 = 0x1c6e41596;
const K3: i64 = 0x1751997d0;
const K4: i64 = 0x0ccaa009e;

/// Barrett reduction constants: `floor(x^95 / G(x))` and `G(x)`, bit-reflected.
const BARRETT_MU: i64 = 0xb4e5b025f7011641_u64 as i64;
const BARRETT_G: i64 = 0x1db710641;

// The wider kernels have progressively more streams to initialize and collapse. Keep their
// crossovers conservative so short inputs do not pay that fixed cost.
const MIN_FOLD_BY_4_BYTES: usize = 128;
#[cfg(stable_vpclmulqdq)]
const MIN_AVX512_BYTES: usize = 2 * 1024;

// Fold constants for the wider strides. For a fold by `D` bits the pair is
// `(reflect(x^(D+32) mod P), reflect(x^(D-32) mod P))`, as for `K1`/`K2` (D = 512) and
// `K3`/`K4` (D = 128). AVX2 uses 8 streams (D = 1024), AVX-512 uses 16 streams (D = 2048).
#[cfg(stable_vpclmulqdq)]
const K_1024_LOW: i64 = 0x1e88ef372;
#[cfg(stable_vpclmulqdq)]
const K_1024_HIGH: i64 = 0x14a7fe880;
#[cfg(stable_vpclmulqdq)]
const K_2048_LOW: i64 = 0x11542778a;
#[cfg(stable_vpclmulqdq)]
const K_2048_HIGH: i64 = 0x1322d1430;

#[target_feature(
    enable = "pclmulqdq",
    enable = "sse2",
    enable = "sse4.1",
    enable = "ssse3"
)]
unsafe fn calculate(crc: u32, mut data: &[u8]) -> u32 {
    // Below 16 bytes there isn't even a single full block to load, so use the scalar fallback.
    if data.len() < 16 {
        return crate::baseline::update_short(crc, data);
    }
    let input = data;

    // For 16..127 bytes a single-accumulator fold-by-1 is enough; the fold-by-4 setup below only
    // pays off once there are several 64-byte groups.
    if data.len() < MIN_FOLD_BY_4_BYTES {
        let mut x = get(&mut data);
        x = arch::_mm_xor_si128(x, arch::_mm_cvtsi32_si128(!crc as i32));
        return reduce_128_to_crc(x, input, data);
    }

    // Step 1: fold by 4 loop
    let mut x3 = get(&mut data);
    let mut x2 = get(&mut data);
    let mut x1 = get(&mut data);
    let mut x0 = get(&mut data);

    // fold in our initial value, part of the incremental crc checksum
    x3 = arch::_mm_xor_si128(x3, arch::_mm_cvtsi32_si128(!crc as i32));

    let k1k2 = arch::_mm_set_epi64x(K2, K1);
    while data.len() >= 64 {
        x3 = reduce128(x3, get(&mut data), k1k2);
        x2 = reduce128(x2, get(&mut data), k1k2);
        x1 = reduce128(x1, get(&mut data), k1k2);
        x0 = reduce128(x0, get(&mut data), k1k2);
    }

    let k3k4 = arch::_mm_set_epi64x(K4, K3);
    let mut x = reduce128(x3, x2, k3k4);
    x = reduce128(x, x1, k3k4);
    x = reduce128(x, x0, k3k4);

    reduce_128_to_crc(x, input, data)
}

/// 256-bit `VPCLMULQDQ` variant: 8 streams across four `YMM` registers (two lanes each), folding
/// two streams per carry-less multiply.
#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
#[target_feature(
    enable = "pclmulqdq",
    enable = "sse2",
    enable = "sse4.1",
    enable = "ssse3",
    enable = "avx",
    enable = "avx2",
    enable = "vpclmulqdq"
)]
unsafe fn calculate_avx2(crc: u32, mut data: &[u8]) -> u32 {
    // Too small for the wide loop; use the 128-bit path.
    if data.len() < 256 {
        return calculate(crc, data);
    }
    let input = data;

    // First 128 bytes as 8 streams (four YMM registers, two lanes each).
    let mut v0 = get256(&mut data);
    let mut v1 = get256(&mut data);
    let mut v2 = get256(&mut data);
    let mut v3 = get256(&mut data);

    // Fold the initial CRC into the lowest-offset stream.
    v0 = arch::_mm256_xor_si256(
        v0,
        arch::_mm256_castsi128_si256(arch::_mm_cvtsi32_si128(!crc as i32)),
    );

    let k = arch::_mm256_set_epi64x(K_1024_HIGH, K_1024_LOW, K_1024_HIGH, K_1024_LOW);
    while data.len() >= 128 {
        v0 = reduce256(v0, get256(&mut data), k);
        v1 = reduce256(v1, get256(&mut data), k);
        v2 = reduce256(v2, get256(&mut data), k);
        v3 = reduce256(v3, get256(&mut data), k);
    }

    // Collapse the 8 streams, in increasing byte offset, folding each into the next by 128 bits.
    let k3k4 = arch::_mm_set_epi64x(K4, K3);
    let mut x = arch::_mm256_castsi256_si128(v0);
    x = reduce128(x, arch::_mm256_extracti128_si256(v0, 1), k3k4);
    x = reduce128(x, arch::_mm256_castsi256_si128(v1), k3k4);
    x = reduce128(x, arch::_mm256_extracti128_si256(v1, 1), k3k4);
    x = reduce128(x, arch::_mm256_castsi256_si128(v2), k3k4);
    x = reduce128(x, arch::_mm256_extracti128_si256(v2, 1), k3k4);
    x = reduce128(x, arch::_mm256_castsi256_si128(v3), k3k4);
    x = reduce128(x, arch::_mm256_extracti128_si256(v3, 1), k3k4);

    reduce_128_to_crc(x, input, data)
}

/// 512-bit `VPCLMULQDQ` variant: 16 streams across four `ZMM` registers (four lanes each), folding
/// four streams per carry-less multiply.
#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
#[target_feature(
    enable = "pclmulqdq",
    enable = "sse2",
    enable = "sse4.1",
    enable = "ssse3",
    enable = "avx",
    enable = "avx2",
    enable = "vpclmulqdq",
    enable = "avx512f"
)]
unsafe fn calculate_avx512(crc: u32, mut data: &[u8]) -> u32 {
    // Use a conservative crossover because the 16-stream setup and collapse are costly near 1 KiB.
    if data.len() < MIN_AVX512_BYTES {
        return calculate_avx2(crc, data);
    }
    let input = data;

    // First 256 bytes as 16 streams (four ZMM registers, four lanes each).
    let mut v0 = get512(&mut data);
    let mut v1 = get512(&mut data);
    let mut v2 = get512(&mut data);
    let mut v3 = get512(&mut data);

    // Fold the initial CRC into the lowest-offset stream.
    v0 = arch::_mm512_xor_si512(
        v0,
        arch::_mm512_castsi128_si512(arch::_mm_cvtsi32_si128(!crc as i32)),
    );

    let k = arch::_mm512_set_epi64(
        K_2048_HIGH,
        K_2048_LOW,
        K_2048_HIGH,
        K_2048_LOW,
        K_2048_HIGH,
        K_2048_LOW,
        K_2048_HIGH,
        K_2048_LOW,
    );
    while data.len() >= 256 {
        v0 = reduce512(v0, get512(&mut data), k);
        v1 = reduce512(v1, get512(&mut data), k);
        v2 = reduce512(v2, get512(&mut data), k);
        v3 = reduce512(v3, get512(&mut data), k);
    }

    // Collapse the 16 streams, in increasing byte offset, folding each into the next by 128 bits.
    let k3k4 = arch::_mm_set_epi64x(K4, K3);
    let mut x = arch::_mm512_castsi512_si128(v0);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v0, 1), k3k4);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v0, 2), k3k4);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v0, 3), k3k4);
    x = reduce128(x, arch::_mm512_castsi512_si128(v1), k3k4);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v1, 1), k3k4);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v1, 2), k3k4);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v1, 3), k3k4);
    x = reduce128(x, arch::_mm512_castsi512_si128(v2), k3k4);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v2, 1), k3k4);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v2, 2), k3k4);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v2, 3), k3k4);
    x = reduce128(x, arch::_mm512_castsi512_si128(v3), k3k4);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v3, 1), k3k4);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v3, 2), k3k4);
    x = reduce128(x, arch::_mm512_extracti32x4_epi32(v3, 3), k3k4);

    // A substantial remainder is faster through SSE's complete fold-by-4 kernel than through the
    // serial fold-by-1 tail below. Finalizing and restarting is valid because CRC updates compose.
    if data.len() >= MIN_FOLD_BY_4_BYTES {
        let crc = reduce_128_to_crc(x, input, &[]);
        return calculate(crc, data);
    }

    reduce_128_to_crc(x, input, data)
}

/// Finishes a fold: `x` is `input` folded up to `rest`, its unconsumed tail. Folds any remaining
/// 16-byte chunks into `x`, then a final partial (`< 16` byte) block, reduces from 128 to 32 bits
/// with a Barrett reduction, and returns the CRC. Shared by all of the fold implementations.
/// `input` is at least 16 bytes long.
#[target_feature(
    enable = "pclmulqdq",
    enable = "sse2",
    enable = "sse4.1",
    enable = "ssse3"
)]
unsafe fn reduce_128_to_crc(mut x: arch::__m128i, input: &[u8], mut rest: &[u8]) -> u32 {
    let k3k4 = arch::_mm_set_epi64x(K4, K3);

    // Fold by 1 over any remaining whole 16-byte blocks.
    while rest.len() >= 16 {
        x = reduce128(x, get(&mut rest), k3k4);
    }

    // Fold a final partial block of `n` (1..=15) bytes. `x` and those bytes are `16 + n` bytes of
    // message: the first `n` are folded forward by 128 bits, into the last 16. The last 16 are
    // the high `16 - n` bytes of `x` followed by the final `n` bytes of the input, which one load
    // ending at the end of `input` reads. That load overlaps bytes already folded in, which the
    // blend drops. `input` is at least 16 bytes long, so it stays in bounds.
    //
    // Byte `i` of `shl` is `i + n - 16`. As a `pshufb` mask it moves the low `n` bytes of `x` to
    // the top and zeroes the rest. `shr` is `shl` with each byte's high bit flipped: it moves the
    // high `16 - n` bytes of `x` to the bottom, and its high bits mark the `n` bytes above them,
    // where the blend takes the loaded bytes.
    let n = rest.len();
    if n > 0 {
        let seq = arch::_mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let shl = arch::_mm_add_epi8(seq, arch::_mm_set1_epi8(n as i8 - 16));
        let shr = arch::_mm_xor_si128(shl, arch::_mm_set1_epi8(-128));

        let first = arch::_mm_shuffle_epi8(x, shl);
        let tail =
            arch::_mm_loadu_si128(input[input.len() - 16..].as_ptr() as *const arch::__m128i);
        let last = arch::_mm_blendv_epi8(arch::_mm_shuffle_epi8(x, shr), tail, shr);
        x = reduce128(first, last, k3k4);
    }

    // Reduce the 128-bit message polynomial to its CRC: multiply by `x^32` and reduce modulo
    // `G(x)`. Fold the low 64 bits (the high-order terms, since the bits are reflected) into the
    // high 64, leaving 96 bits. A Barrett reduction with a 64-bit quotient estimate,
    // `floor(x^95 / G(x))`, takes those to 32 without folding them to 64 first.
    //
    // libdeflate's `crc32_pclmul_template.h` and the Linux kernel's `crc-pclmul-template.S` (with
    // `n=32` and `LSB_CRC=1`) do the same; see there for the math.
    let x = arch::_mm_xor_si128(
        arch::_mm_clmulepi64_si128(x, k3k4, 0x10),
        arch::_mm_srli_si128(x, 8),
    );
    let barrett = arch::_mm_set_epi64x(BARRETT_G, BARRETT_MU);
    let t = arch::_mm_clmulepi64_si128(x, barrett, 0x00);
    let t = arch::_mm_clmulepi64_si128(t, barrett, 0x10);
    !(arch::_mm_extract_epi32(arch::_mm_xor_si128(x, t), 2) as u32)
}

#[inline]
unsafe fn reduce128(a: arch::__m128i, b: arch::__m128i, keys: arch::__m128i) -> arch::__m128i {
    let t1 = arch::_mm_clmulepi64_si128(a, keys, 0x00);
    let t2 = arch::_mm_clmulepi64_si128(a, keys, 0x11);
    arch::_mm_xor_si128(arch::_mm_xor_si128(b, t1), t2)
}

#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
#[target_feature(enable = "avx2", enable = "vpclmulqdq")]
#[inline]
unsafe fn reduce256(a: arch::__m256i, b: arch::__m256i, keys: arch::__m256i) -> arch::__m256i {
    let t1 = arch::_mm256_clmulepi64_epi128(a, keys, 0x00);
    let t2 = arch::_mm256_clmulepi64_epi128(a, keys, 0x11);
    arch::_mm256_xor_si256(arch::_mm256_xor_si256(b, t1), t2)
}

#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
#[inline]
unsafe fn reduce512(a: arch::__m512i, b: arch::__m512i, keys: arch::__m512i) -> arch::__m512i {
    let t1 = arch::_mm512_clmulepi64_epi128(a, keys, 0x00);
    let t2 = arch::_mm512_clmulepi64_epi128(a, keys, 0x11);
    arch::_mm512_xor_si512(arch::_mm512_xor_si512(b, t1), t2)
}

// The loads below index `a[..N]` before reading through its pointer, so a miscounted length panics
// instead of reading out of bounds. The callers' loop conditions already imply the check, so it
// compiles away: the kernels' machine code is unchanged.

/// Loads the next 16 bytes of `a` as a vector and advances `a` past them.
#[inline]
fn get(a: &mut &[u8]) -> arch::__m128i {
    // SAFETY: `a[..16]` is 16 readable bytes, `_mm_loadu_si128` doesn't need them aligned, and SSE2
    // is enabled for the whole build (`specialized/mod.rs` only compiles this module when it is).
    let r = unsafe { arch::_mm_loadu_si128(a[..16].as_ptr() as *const arch::__m128i) };
    *a = &a[16..];
    r
}

#[cfg(stable_vpclmulqdq)]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn get256(a: &mut &[u8]) -> arch::__m256i {
    let r = arch::_mm256_loadu_si256(a[..32].as_ptr() as *const arch::__m256i);
    *a = &a[32..];
    r
}

#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn get512(a: &mut &[u8]) -> arch::__m512i {
    let r = arch::_mm512_loadu_si512(a[..64].as_ptr() as *const _);
    *a = &a[64..];
    r
}

#[cfg(test)]
mod test {
    quickcheck::quickcheck! {
        fn check_against_baseline(init: u32, chunks: Vec<(Vec<u8>, usize)>) -> bool {
            let mut baseline = init;
            let mut pclmulqdq = init;
            let spec = super::State::new().expect("not supported");

            for (chunk, mut offset) in chunks {
                // simulate random alignments by offsetting the slice by up to 15 bytes
                offset &= 0xF;
                if chunk.len() <= offset {
                    baseline = crate::baseline::update_fast_16(baseline, &chunk);
                    pclmulqdq = spec.update(pclmulqdq, &chunk);
                } else {
                    baseline = crate::baseline::update_fast_16(baseline, &chunk[offset..]);
                    pclmulqdq = spec.update(pclmulqdq, &chunk[offset..]);
                }
            }
            pclmulqdq == baseline
        }
    }

    // Exercises the wide fold paths across the sizes and alignments where the strategy switches.
    #[test]
    fn check_large_inputs_against_baseline() {
        let mut data = vec![0u8; 8200];
        let mut s: u32 = 0x1234_5678;
        for b in data.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (s >> 24) as u8;
        }

        for &len in &[
            0usize, 1, 15, 16, 17, 63, 64, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024,
            1025, 2047, 2048, 2049, 2175, 2176, 2303, 2304, 4096, 8192, 8199,
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
