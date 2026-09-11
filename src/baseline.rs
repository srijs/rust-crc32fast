use crate::table::CRC32_TABLE;

#[derive(Clone)]
pub struct State {
    state: u32,
}

impl State {
    pub fn new(state: u32) -> Self {
        State { state }
    }

    pub fn update(&mut self, buf: &[u8]) {
        self.state = update_fast_16(self.state, buf);
    }

    pub fn finalize(self) -> u32 {
        self.state
    }

    pub fn reset(&mut self) {
        self.state = 0;
    }

    pub fn combine(&mut self, other: u32, amount: u64) {
        self.state = crate::combine::combine(self.state, other, amount);
    }
}

/// # Safety
///
/// Requires `sse`; `ptr` may be invalid (hint, never dereferenced)
#[inline]
#[cfg(all(
    target_feature = "sse",
    any(target_arch = "x86", target_arch = "x86_64"),
))]
unsafe fn prefetch(ptr: *const u8) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_mm_prefetch, _MM_HINT_T0};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

    _mm_prefetch::<_MM_HINT_T0>(ptr.cast::<i8>());
}

/// # Safety
///
/// No-op: `ptr` is never used, so any value is safe.
#[inline(always)]
#[cfg(not(all(
    target_feature = "sse",
    any(target_arch = "x86", target_arch = "x86_64"),
)))]
const unsafe fn prefetch(_: *const u8) {}

#[rustfmt::skip]
macro_rules! block {
    ($current:expr, $crc:expr) => {{
        let a = unsafe { $current.add(0).read_unaligned() } ^ $crc;
        let b = unsafe { $current.add(1).read_unaligned() };
        let c = unsafe { $current.add(2).read_unaligned() };
        let d = unsafe { $current.add(3).read_unaligned() };

        $current = unsafe { $current.add(4) };

        let [a0, a1, a2, a3] = a.to_be_bytes();
        let [b0, b1, b2, b3] = b.to_be_bytes();
        let [c0, c1, c2, c3] = c.to_be_bytes();
        let [d0, d1, d2, d3] = d.to_be_bytes();

        $crc = CRC32C_TABLE[ 0][d0 as usize]
            ^  CRC32C_TABLE[ 1][d1 as usize]
            ^  CRC32C_TABLE[ 2][d2 as usize]
            ^  CRC32C_TABLE[ 3][d3 as usize]
            ^  CRC32C_TABLE[ 4][c0 as usize]
            ^  CRC32C_TABLE[ 5][c1 as usize]
            ^  CRC32C_TABLE[ 6][c2 as usize]
            ^  CRC32C_TABLE[ 7][c3 as usize]
            ^  CRC32C_TABLE[ 8][b0 as usize]
            ^  CRC32C_TABLE[ 9][b1 as usize]
            ^  CRC32C_TABLE[10][b2 as usize]
            ^  CRC32C_TABLE[11][b3 as usize]
            ^  CRC32C_TABLE[12][a0 as usize]
            ^  CRC32C_TABLE[13][a1 as usize]
            ^  CRC32C_TABLE[14][a2 as usize]
            ^  CRC32C_TABLE[15][a3 as usize];
    }};
}

pub(crate) fn update_fast_16(prev: u32, buf: &[u8]) -> u32 {
    const UNROLL: usize = 4;
    const BYTES_AT_ONCE: usize = 16 * UNROLL;
    const PREFETCH_AHEAD: usize = 256;

    let mut crc = !prev;
    #[allow(clippy::cast_ptr_alignment)]
    let mut current = buf.as_ptr().cast::<u32>();
    let mut length = buf.len();

    while length >= BYTES_AT_ONCE + PREFETCH_AHEAD {
        // SAFETY: offset within bounds per loop guard; prefetch never dereferences.
        unsafe {
            prefetch(current.cast::<u8>().add(PREFETCH_AHEAD));
        }

        for _ in 0..UNROLL {
            // SAFETY: loop guard ensures 64 bytes remain for these 4 reads.
            block!(current, crc);
        }

        length -= BYTES_AT_ONCE;
    }

    while length >= 16 {
        // SAFETY: loop guard ensures 16 bytes remain for these 4 reads.
        // from_le makes the decoded word endianness-independent.
        block!(current, crc);

        length -= 16;
    }

    if length == 0 {
        return !crc;
    }

    // SAFETY: `current`/`length` describe the exact unread tail of `buf`.
    let tail = unsafe { core::slice::from_raw_parts(current.cast::<u8>(), length) };
    update_slow(!crc, tail)
}

pub(crate) fn update_slow(prev: u32, buf: &[u8]) -> u32 {
    let mut crc = !prev;

    for &byte in buf.iter() {
        crc = CRC32_TABLE[0][((crc as u8) ^ byte) as usize] ^ (crc >> 8);
    }

    !crc
}

#[cfg(test)]
mod test {
    #[test]
    fn slow() {
        assert_eq!(super::update_slow(0, b""), 0);

        // test vectors from the iPXE project (input and output are bitwise negated)
        assert_eq!(super::update_slow(!0x12345678, b""), !0x12345678);
        assert_eq!(super::update_slow(!0xffffffff, b"hello world"), !0xf2b5ee7a);
        assert_eq!(super::update_slow(!0xffffffff, b"hello"), !0xc9ef5979);
        assert_eq!(super::update_slow(!0xc9ef5979, b" world"), !0xf2b5ee7a);

        // Some vectors found on Rosetta code
        assert_eq!(super::update_slow(0, b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"), 0x190A55AD);
        assert_eq!(super::update_slow(0, b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF"), 0xFF6CAB0B);
        assert_eq!(super::update_slow(0, b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1A\x1B\x1C\x1D\x1E\x1F"), 0x91267E8A);
    }

    quickcheck::quickcheck! {
        fn fast_16_is_the_same_as_slow(crc: u32, bytes: Vec<u8>) -> bool {
            super::update_fast_16(crc, &bytes) == super::update_slow(crc, &bytes)
        }
    }
}
