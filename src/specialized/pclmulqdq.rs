//! Specialized checksum code for the x86 CPU architecture, using carry-less multiplication to
//! "fold" the input, as described in:
//!
//! Gopal, V., Ozturk, E., Guilford, J., Wolrich, G., Feghali, W., Dixon, M., & Karakoyunlu, D. (2009).
//! _Fast CRC computation for generic polynomials using PCLMULQDQ instruction_. Intel.
//! (Mirror link: <https://fossies.org/linux/zlib-ng/doc/crc-pclmulqdq.pdf>, accessed 2024-05-20)
//!
//! The kernel's structure follows libdeflate's `lib/x86/crc32_pclmul_template.h` by Eric Biggers
//! (MIT licensed): eight vector accumulators in the main loop, a tree of folds (eight vectors to
//! four, two, one, then down to 128 bits) that consumes the leftover input on the way, a final
//! partial block folded with one overlapping load, and a three-multiply reduction from 128 bits to
//! the 32-bit CRC.
//!
//! `fold_crc` is written once against the `Vector` trait and compiled for three vector widths:
//! 128-bit `PCLMULQDQ`, and 256-bit (`AVX2`) and 512-bit (`AVX-512`) `VPCLMULQDQ`, which fold two
//! and four 128-bit lanes per instruction. The wider two rely on intrinsics stabilized in Rust 1.89
//! and are only compiled when `build.rs` sets the `stable_vpclmulqdq` cfg, leaving the crate MSRV
//! unchanged otherwise. The best variant supported by the running CPU is chosen at runtime.
//!
//! Each `unsafe fn` here is unsafe only because it needs the CPU features in its
//! `#[target_feature]` list, or those of the function it's inlined into. None of them has a
//! memory-safety precondition: every load is bounds-checked.

#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

use arch::__m128i;
#[cfg(stable_vpclmulqdq)]
use arch::{__m256i, __m512i};

/// Which SIMD implementation to use, chosen once at construction from the CPU's features.
#[derive(Clone, Copy)]
enum Kind {
    /// 128-bit `PCLMULQDQ`.
    Sse,
    /// 256-bit `VPCLMULQDQ`.
    #[cfg(stable_vpclmulqdq)]
    Avx2,
    /// 256- and 512-bit `VPCLMULQDQ` with `AVX-512`.
    #[cfg(stable_vpclmulqdq)]
    Avx512,
    /// As `Avx512`, but with 512-bit vectors only for long inputs, for CPUs that downclock too
    /// eagerly when they're used.
    #[cfg(stable_vpclmulqdq)]
    Avx512Ymm,
}

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
            #[cfg(stable_vpclmulqdq)]
            Some(Kind::Avx512Ymm) => 5,
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
            #[cfg(stable_vpclmulqdq)]
            5 => Some(Kind::Avx512Ymm),
            _ => None,
        }
    }

    /// The `AVX-512` kind for this CPU.
    #[cfg(stable_vpclmulqdq)]
    #[inline]
    fn avx512() -> Self {
        if eager_zmm_downclocking() {
            Kind::Avx512Ymm
        } else {
            Kind::Avx512
        }
    }
}

/// Whether this is one of the Intel CPUs whose frequency drops too eagerly when 512-bit registers
/// are used only occasionally: Ice Lake and Tiger Lake, the list libdeflate uses.
#[cfg(stable_vpclmulqdq)]
#[allow(unused_unsafe)] // `__cpuid` is a safe function on newer toolchains
fn eager_zmm_downclocking() -> bool {
    // SAFETY: Only called for CPUs with AVX-512, which all have the CPUID instruction.
    let vendor = unsafe { arch::__cpuid(0) };
    // "GenuineIntel", in EBX, EDX, ECX order.
    if (vendor.ebx, vendor.edx, vendor.ecx) != (0x756e_6547, 0x4965_6e69, 0x6c65_746e) {
        return false;
    }
    // SAFETY: As above.
    let eax = unsafe { arch::__cpuid(1) }.eax;
    let family = (eax >> 8) & 0xf;
    let model = ((eax >> 4) & 0xf) | ((eax >> 12) & 0xf0);
    // Ice Lake server (106, 108) and client (126), Tiger Lake (140, 141).
    family == 6 && matches!(model, 106 | 108 | 126 | 140 | 141)
}

#[derive(Clone)]
pub struct State {
    kind: Kind,
}

impl State {
    /// The kernel for this CPU, if any. Finding it takes several feature checks, and CPUID on
    /// `AVX-512` CPUs, which is slow in a virtual machine. `Hasher::new` asks every time, and
    /// `crc32fast::hash` constructs a `Hasher` for every call, so the answer is cached: after the
    /// first call, this is one load.
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

    #[cfg(not(feature = "std"))]
    #[cold]
    fn detect_uncached() -> Option<Kind> {
        if cfg!(target_feature = "pclmulqdq")
            && cfg!(target_feature = "sse2")
            && cfg!(target_feature = "sse4.1")
            && cfg!(target_feature = "ssse3")
        {
            #[cfg(stable_vpclmulqdq)]
            {
                if cfg!(target_feature = "avx512f")
                    && cfg!(target_feature = "avx512vl")
                    && cfg!(target_feature = "vpclmulqdq")
                {
                    return Some(Kind::avx512());
                }
                if cfg!(target_feature = "avx2") && cfg!(target_feature = "vpclmulqdq") {
                    return Some(Kind::Avx2);
                }
            }

            return Some(Kind::Sse);
        }

        None
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
                if is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512vl")
                    && is_x86_feature_detected!("vpclmulqdq")
                {
                    return Some(Kind::avx512());
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
                Kind::Avx512 => calculate_avx512::<MIN_ZMM_BYTES>(state, buf),
                #[cfg(stable_vpclmulqdq)]
                Kind::Avx512Ymm => calculate_avx512::<MIN_ZMM_BYTES_DOWNCLOCKING>(state, buf),
            }
        }
    }
}

/// Multipliers for folding a 128-bit lane forward across `D` bits, as `(low, high)`: the lane's low
/// and high 64-bit halves are multiplied by `x^(D+32) mod G(x)` and `x^(D-32) mod G(x)`
/// respectively, bit-reflected, and the products XORed into the lane `D` bits further on.
type Mults = (i64, i64);

/// The multipliers for folding across `bits` bits, computed at compile time by
/// `combine::fold_keys`, as the aarch64 kernel's are.
const fn fold_by(bits: u64) -> Mults {
    let keys = crate::combine::fold_keys(bits);
    (keys[0] as i64, keys[1] as i64)
}

const FOLD_128: Mults = fold_by(128);
const FOLD_256: Mults = fold_by(256);
const FOLD_512: Mults = fold_by(512);
const FOLD_1024: Mults = fold_by(1024);
#[cfg(stable_vpclmulqdq)]
const FOLD_2048: Mults = fold_by(2048);
#[cfg(stable_vpclmulqdq)]
const FOLD_4096: Mults = fold_by(4096);

/// Barrett reduction constants: `floor(x^95 / G(x))` and `G(x)`, bit-reflected.
const BARRETT_MU: i64 = 0xb4e5b025f7011641_u64 as i64;
const BARRETT_G: i64 = 0x1db710641;

/// Inputs this long and up use 512-bit vectors in the `AVX-512` kernel; shorter ones use 256-bit
/// vectors with the same features.
#[cfg(stable_vpclmulqdq)]
const MIN_ZMM_BYTES: usize = 64;

/// As `MIN_ZMM_BYTES`, on CPUs that downclock eagerly when 512-bit vectors are used, so that short
/// inputs don't use them: the cutover the `AVX-512` kernel had on every CPU before.
#[cfg(stable_vpclmulqdq)]
const MIN_ZMM_BYTES_DOWNCLOCKING: usize = 2 * 1024;

/// SSE kernel: 128-bit `PCLMULQDQ`.
#[target_feature(
    enable = "pclmulqdq",
    enable = "sse2",
    enable = "sse4.1",
    enable = "ssse3"
)]
unsafe fn calculate(crc: u32, data: &[u8]) -> u32 {
    // Below 16 bytes there isn't even a single full block to load, so use the scalar fallback.
    if data.len() < 16 {
        return crate::baseline::update_short(crc, data);
    }
    fold_crc::<__m128i>(crc, data)
}

/// `AVX2` kernel: 256-bit `VPCLMULQDQ`, folding two 128-bit lanes per instruction.
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
unsafe fn calculate_avx2(crc: u32, data: &[u8]) -> u32 {
    if data.len() < 16 {
        return crate::baseline::update_short(crc, data);
    }
    fold_crc::<__m256i>(crc, data)
}

/// `AVX-512` kernel: 512-bit `VPCLMULQDQ` (four 128-bit lanes per instruction) for inputs of
/// `MIN_ZMM` bytes or more, and 256-bit for shorter ones. Either way, `AVX-512VL` lets the
/// compiler merge each fold's two XORs into one `vpternlogq` and use 32 vector registers.
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
    enable = "avx512f",
    enable = "avx512vl"
)]
unsafe fn calculate_avx512<const MIN_ZMM: usize>(crc: u32, data: &[u8]) -> u32 {
    let n = data.len();
    if n < 16 {
        return crate::baseline::update_short(crc, data);
    }
    if n < MIN_ZMM {
        fold_crc::<__m256i>(crc, data)
    } else {
        fold_crc::<__m512i>(crc, data)
    }
}

/// A vector of one or more 128-bit lanes for `fold_crc`. Every method is `#[inline(always)]` so
/// that it compiles with the target features of the kernel that `fold_crc` is inlined into.
trait Vector: Copy {
    /// Width in bytes.
    const LEN: usize;
    /// Multipliers for folding across one, two, four, and eight vectors.
    const M1: Mults;
    const M2: Mults;
    const M4: Mults;
    const M8: Mults;

    /// Loads the first `LEN` bytes of `a`.
    unsafe fn load(a: &[u8]) -> Self;
    /// Zero-extends a 128-bit vector.
    unsafe fn from_128(x: __m128i) -> Self;
    unsafe fn xor(self, other: Self) -> Self;
    /// Repeats `m` in every 128-bit lane.
    unsafe fn splat(m: Mults) -> Self;
    /// Folds each lane of `self` forward by the distance `m` was computed for, and adds the
    /// result to `dst`.
    unsafe fn fold(self, dst: Self, m: Self) -> Self;
    /// Folds the lanes together into one, consuming any whole vectors of a narrower width that
    /// remain in `rest`.
    unsafe fn reduce(self, rest: &mut &[u8]) -> __m128i;
}

impl Vector for __m128i {
    const LEN: usize = 16;
    const M1: Mults = FOLD_128;
    const M2: Mults = FOLD_256;
    const M4: Mults = FOLD_512;
    const M8: Mults = FOLD_1024;

    #[inline(always)]
    unsafe fn load(a: &[u8]) -> Self {
        arch::_mm_loadu_si128(a[..16].as_ptr() as *const __m128i)
    }

    #[inline(always)]
    unsafe fn from_128(x: __m128i) -> Self {
        x
    }

    #[inline(always)]
    unsafe fn xor(self, other: Self) -> Self {
        arch::_mm_xor_si128(self, other)
    }

    #[inline(always)]
    unsafe fn splat(m: Mults) -> Self {
        arch::_mm_set_epi64x(m.1, m.0)
    }

    #[inline(always)]
    unsafe fn fold(self, dst: Self, m: Self) -> Self {
        let lo = arch::_mm_clmulepi64_si128(self, m, 0x00);
        let hi = arch::_mm_clmulepi64_si128(self, m, 0x11);
        arch::_mm_xor_si128(arch::_mm_xor_si128(dst, lo), hi)
    }

    #[inline(always)]
    unsafe fn reduce(self, _rest: &mut &[u8]) -> __m128i {
        self
    }
}

#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
impl Vector for __m256i {
    const LEN: usize = 32;
    const M1: Mults = FOLD_256;
    const M2: Mults = FOLD_512;
    const M4: Mults = FOLD_1024;
    const M8: Mults = FOLD_2048;

    #[inline(always)]
    unsafe fn load(a: &[u8]) -> Self {
        arch::_mm256_loadu_si256(a[..32].as_ptr() as *const __m256i)
    }

    #[inline(always)]
    unsafe fn from_128(x: __m128i) -> Self {
        arch::_mm256_zextsi128_si256(x)
    }

    #[inline(always)]
    unsafe fn xor(self, other: Self) -> Self {
        arch::_mm256_xor_si256(self, other)
    }

    #[inline(always)]
    unsafe fn splat(m: Mults) -> Self {
        arch::_mm256_set_epi64x(m.1, m.0, m.1, m.0)
    }

    #[inline(always)]
    unsafe fn fold(self, dst: Self, m: Self) -> Self {
        let lo = arch::_mm256_clmulepi64_epi128(self, m, 0x00);
        let hi = arch::_mm256_clmulepi64_epi128(self, m, 0x11);
        arch::_mm256_xor_si256(arch::_mm256_xor_si256(dst, lo), hi)
    }

    #[inline(always)]
    unsafe fn reduce(self, _rest: &mut &[u8]) -> __m128i {
        let lo = arch::_mm256_castsi256_si128(self);
        let hi = arch::_mm256_extracti128_si256(self, 1);
        lo.fold(hi, __m128i::splat(FOLD_128))
    }
}

#[cfg(stable_vpclmulqdq)]
#[allow(clippy::incompatible_msrv)] // intrinsics are gated to rustc >= 1.89 by build.rs
impl Vector for __m512i {
    const LEN: usize = 64;
    const M1: Mults = FOLD_512;
    const M2: Mults = FOLD_1024;
    const M4: Mults = FOLD_2048;
    const M8: Mults = FOLD_4096;

    #[inline(always)]
    unsafe fn load(a: &[u8]) -> Self {
        arch::_mm512_loadu_si512(a[..64].as_ptr() as *const _)
    }

    #[inline(always)]
    unsafe fn from_128(x: __m128i) -> Self {
        arch::_mm512_zextsi128_si512(x)
    }

    #[inline(always)]
    unsafe fn xor(self, other: Self) -> Self {
        arch::_mm512_xor_si512(self, other)
    }

    #[inline(always)]
    unsafe fn splat(m: Mults) -> Self {
        arch::_mm512_set_epi64(m.1, m.0, m.1, m.0, m.1, m.0, m.1, m.0)
    }

    #[inline(always)]
    unsafe fn fold(self, dst: Self, m: Self) -> Self {
        let lo = arch::_mm512_clmulepi64_epi128(self, m, 0x00);
        let hi = arch::_mm512_clmulepi64_epi128(self, m, 0x11);
        arch::_mm512_xor_si512(arch::_mm512_xor_si512(dst, lo), hi)
    }

    #[inline(always)]
    unsafe fn reduce(self, rest: &mut &[u8]) -> __m128i {
        let m = __m256i::splat(FOLD_256);
        let lo = arch::_mm512_castsi512_si256(self);
        let hi = arch::_mm512_extracti64x4_epi64(self, 1);
        let mut y = lo.fold(hi, m);
        if rest.len() >= 32 {
            y = y.fold(__m256i::load(rest), m);
            *rest = &rest[32..];
        }
        y.reduce(rest)
    }
}

/// Computes the CRC of `data`, which must be at least 16 bytes long, folding it with `V`-wide
/// vectors.
///
/// Inputs of eight vectors or more run through a main loop over eight accumulators. Every
/// shorter input, and the leftover of a longer one, enters the tree of folds at the level its
/// length calls for: each level halves the number of accumulators, then folds in one more group of
/// vectors if that many bytes remain. So the leftover is consumed a group at a time instead of a
/// 16-byte block at a time, and the final partial block takes one more multiply.
#[inline(always)]
unsafe fn fold_crc<V: Vector>(crc: u32, data: &[u8]) -> u32 {
    debug_assert!(data.len() >= 16);
    let len = V::LEN;
    let init = arch::_mm_cvtsi32_si128(!crc as i32);
    let mut rest = data;

    let x = if data.len() >= len {
        let first = V::load(rest).xor(V::from_128(init));
        let v = if data.len() >= 2 * len {
            let (mut v0, v1) = if data.len() >= 4 * len {
                let mut v0 = first;
                let mut v1 = V::load(&rest[len..]);
                let mut v2 = V::load(&rest[2 * len..]);
                let mut v3 = V::load(&rest[3 * len..]);
                rest = &rest[4 * len..];

                if rest.len() >= 4 * len {
                    let mut v4 = V::load(rest);
                    let mut v5 = V::load(&rest[len..]);
                    let mut v6 = V::load(&rest[2 * len..]);
                    let mut v7 = V::load(&rest[3 * len..]);
                    rest = &rest[4 * len..];

                    let m8 = V::splat(V::M8);
                    while rest.len() >= 8 * len {
                        v0 = v0.fold(V::load(rest), m8);
                        v1 = v1.fold(V::load(&rest[len..]), m8);
                        v2 = v2.fold(V::load(&rest[2 * len..]), m8);
                        v3 = v3.fold(V::load(&rest[3 * len..]), m8);
                        v4 = v4.fold(V::load(&rest[4 * len..]), m8);
                        v5 = v5.fold(V::load(&rest[5 * len..]), m8);
                        v6 = v6.fold(V::load(&rest[6 * len..]), m8);
                        v7 = v7.fold(V::load(&rest[7 * len..]), m8);
                        rest = &rest[8 * len..];
                    }

                    // Eight accumulators to four.
                    let m4 = V::splat(V::M4);
                    v0 = v0.fold(v4, m4);
                    v1 = v1.fold(v5, m4);
                    v2 = v2.fold(v6, m4);
                    v3 = v3.fold(v7, m4);
                    if rest.len() >= 4 * len {
                        v0 = v0.fold(V::load(rest), m4);
                        v1 = v1.fold(V::load(&rest[len..]), m4);
                        v2 = v2.fold(V::load(&rest[2 * len..]), m4);
                        v3 = v3.fold(V::load(&rest[3 * len..]), m4);
                        rest = &rest[4 * len..];
                    }
                }

                // Four to two.
                let m2 = V::splat(V::M2);
                v0 = v0.fold(v2, m2);
                v1 = v1.fold(v3, m2);
                if rest.len() >= 2 * len {
                    v0 = v0.fold(V::load(rest), m2);
                    v1 = v1.fold(V::load(&rest[len..]), m2);
                    rest = &rest[2 * len..];
                }
                (v0, v1)
            } else {
                let v1 = V::load(&rest[len..]);
                rest = &rest[2 * len..];
                (first, v1)
            };

            // Two to one.
            let m1 = V::splat(V::M1);
            v0 = v0.fold(v1, m1);
            if rest.len() >= len {
                v0 = v0.fold(V::load(rest), m1);
                rest = &rest[len..];
            }
            v0
        } else {
            rest = &rest[len..];
            first
        };

        // One vector to 128 bits.
        v.reduce(&mut rest)
    } else {
        // Shorter than one vector (only when `V` is wider than 128 bits): start at 128 bits.
        let x = __m128i::load(rest).xor(init);
        rest = &rest[16..];
        x
    };

    reduce_128_to_crc(x, data, rest)
}

/// Finishes a fold: `x` is `input` folded up to `rest`, its unconsumed tail. Folds any remaining
/// 16-byte chunks into `x`, then a final partial (`< 16` byte) block, reduces from 128 to 32 bits
/// with a Barrett reduction, and returns the CRC. `input` is at least 16 bytes long.
#[inline(always)]
unsafe fn reduce_128_to_crc(mut x: __m128i, input: &[u8], mut rest: &[u8]) -> u32 {
    let m = __m128i::splat(FOLD_128);

    // Fold by 1 over any remaining whole 16-byte blocks: at most two.
    while rest.len() >= 16 {
        x = x.fold(__m128i::load(rest), m);
        rest = &rest[16..];
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
        let tail = __m128i::load(&input[input.len() - 16..]);
        let last = arch::_mm_blendv_epi8(arch::_mm_shuffle_epi8(x, shr), tail, shr);
        x = first.fold(last, m);
    }

    // Reduce the 128-bit message polynomial to its CRC: multiply by `x^32` and reduce modulo
    // `G(x)`. Fold the low 64 bits (the high-order terms, since the bits are reflected) into the
    // high 64, leaving 96 bits. A Barrett reduction with a 64-bit quotient estimate,
    // `floor(x^95 / G(x))`, takes those to 32 without folding them to 64 first.
    //
    // libdeflate's `crc32_pclmul_template.h` and the Linux kernel's `crc-pclmul-template.S` (with
    // `n=32` and `LSB_CRC=1`) do the same; see there for the math.
    let x = arch::_mm_xor_si128(
        arch::_mm_clmulepi64_si128(x, m, 0x10),
        arch::_mm_srli_si128(x, 8),
    );
    let barrett = arch::_mm_set_epi64x(BARRETT_G, BARRETT_MU);
    let t = arch::_mm_clmulepi64_si128(x, barrett, 0x00);
    let t = arch::_mm_clmulepi64_si128(t, barrett, 0x10);
    !(arch::_mm_extract_epi32(arch::_mm_xor_si128(x, t), 2) as u32)
}

#[cfg(test)]
mod test {
    use super::{Kind, State};

    /// Every kernel the running CPU supports, so the narrower ones get tested on wider CPUs too.
    fn supported() -> impl Iterator<Item = State> {
        #[cfg(stable_vpclmulqdq)]
        let avx512 = matches!(
            State::new().expect("not supported").kind,
            Kind::Avx512 | Kind::Avx512Ymm
        );
        let kinds = [
            Some(Kind::Sse),
            #[cfg(stable_vpclmulqdq)]
            (is_x86_feature_detected!("avx2") && is_x86_feature_detected!("vpclmulqdq"))
                .then_some(Kind::Avx2),
            #[cfg(stable_vpclmulqdq)]
            avx512.then_some(Kind::Avx512),
            #[cfg(stable_vpclmulqdq)]
            avx512.then_some(Kind::Avx512Ymm),
        ];
        kinds.into_iter().flatten().map(|kind| State { kind })
    }

    quickcheck::quickcheck! {
        fn check_against_baseline(init: u32, chunks: Vec<(Vec<u8>, usize)>) -> bool {
            supported().all(|spec| {
                let mut baseline = init;
                let mut specialized = init;
                for (chunk, offset) in &chunks {
                    // simulate random alignments by offsetting the slice by up to 15 bytes
                    let chunk = &chunk[(offset & 0xF).min(chunk.len())..];
                    baseline = crate::baseline::update_fast_16(baseline, chunk);
                    specialized = spec.update(specialized, chunk);
                }
                specialized == baseline
            })
        }
    }

    // Exercises every kernel across the sizes and alignments where the strategy switches.
    #[test]
    fn check_large_inputs_against_baseline() {
        let mut data = vec![0u8; 8200];
        let mut s: u32 = 0x1234_5678;
        for b in data.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (s >> 24) as u8;
        }

        let mut lens: Vec<usize> = (0..=160).collect();
        for &edge in &[
            192usize, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
        ] {
            lens.extend(edge - 17..=edge + 17);
        }
        for spec in supported() {
            for &len in &lens {
                for &offset in &[0usize, 1, 3, 7, 8, 15] {
                    if offset + len > data.len() {
                        continue;
                    }
                    let slice = &data[offset..offset + len];
                    for &init in &[0u32, 0xDEAD_BEEF] {
                        assert_eq!(
                            spec.update(init, slice),
                            crate::baseline::update_fast_16(init, slice),
                            "mismatch for len={len} offset={offset} init={init:#x}",
                        );
                    }
                }
            }
        }
    }
}
