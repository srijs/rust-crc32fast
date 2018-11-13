#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

#[derive(Clone)]
pub struct State {
    xmm_crc0: arch::__m128i,
    xmm_crc1: arch::__m128i,
    xmm_crc2: arch::__m128i,
    xmm_crc3: arch::__m128i,
    xmm_crc_part: arch::__m128i,
}

impl State {
    pub fn new() -> Option<Self> {
        if std::is_x86_feature_detected!("pclmulqdq")
            && std::is_x86_feature_detected!("sse")
            && std::is_x86_feature_detected!("sse2")
            && std::is_x86_feature_detected!("ssse3")
            && std::is_x86_feature_detected!("sse4.1")
        {
            // SAFETY: The conditions above ensure that all
            //         required instructions are supported by the CPU.
            Some(unsafe { Self::init() })
        } else {
            None
        }
    }

    pub fn update(&mut self, buf: &[u8]) {
        // SAFETY: The `State::new` constructor ensures that all
        //         required instructions are supported by the CPU.
        unsafe { self.fold(buf) }
    }

    pub fn finalize(mut self) -> u32 {
        // SAFETY: The `State::new` constructor ensures that all
        //         required instructions are supported by the CPU.
        unsafe { self.fold_512to32() }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn init() -> Self {
        debug_assert!(std::is_x86_feature_detected!("sse2"));

        State {
            xmm_crc0: arch::_mm_cvtsi32_si128(0x9db42487u32 as i32),
            xmm_crc1: arch::_mm_setzero_si128(),
            xmm_crc2: arch::_mm_setzero_si128(),
            xmm_crc3: arch::_mm_setzero_si128(),
            xmm_crc_part: arch::_mm_setzero_si128(),
        }
    }

    #[target_feature(enable = "sse", enable = "sse2", enable = "pclmulqdq")]
    unsafe fn fold_1(&mut self) {
        debug_assert!(std::is_x86_feature_detected!("sse"));
        debug_assert!(std::is_x86_feature_detected!("sse2"));
        debug_assert!(std::is_x86_feature_detected!("pclmulqdq"));

        let xmm_fold4 =
            arch::_mm_set_epi32(0x00000001, 0x54442bd4, 0x00000001, 0xc6e41596u32 as i32);

        let x_tmp3 = self.xmm_crc3;

        self.xmm_crc3 = self.xmm_crc0;
        self.xmm_crc0 = arch::_mm_clmulepi64_si128(self.xmm_crc0, xmm_fold4, 0x01);
        self.xmm_crc3 = arch::_mm_clmulepi64_si128(self.xmm_crc3, xmm_fold4, 0x10);
        let ps_crc0 = arch::_mm_castsi128_ps(self.xmm_crc0);
        let ps_crc3 = arch::_mm_castsi128_ps(self.xmm_crc3);
        let ps_res = arch::_mm_xor_ps(ps_crc0, ps_crc3);

        self.xmm_crc0 = self.xmm_crc1;
        self.xmm_crc1 = self.xmm_crc2;
        self.xmm_crc2 = x_tmp3;
        self.xmm_crc3 = arch::_mm_castps_si128(ps_res);
    }

    #[target_feature(enable = "sse", enable = "sse2", enable = "pclmulqdq")]
    unsafe fn fold_2(&mut self) {
        debug_assert!(std::is_x86_feature_detected!("sse"));
        debug_assert!(std::is_x86_feature_detected!("sse2"));
        debug_assert!(std::is_x86_feature_detected!("pclmulqdq"));

        let xmm_fold4 =
            arch::_mm_set_epi32(0x00000001, 0x54442bd4, 0x00000001, 0xc6e41596u32 as i32);

        let x_tmp3 = self.xmm_crc3;
        let x_tmp2 = self.xmm_crc2;

        self.xmm_crc3 = self.xmm_crc1;
        self.xmm_crc1 = arch::_mm_clmulepi64_si128(self.xmm_crc1, xmm_fold4, 0x01);
        self.xmm_crc3 = arch::_mm_clmulepi64_si128(self.xmm_crc3, xmm_fold4, 0x10);
        let ps_crc3 = arch::_mm_castsi128_ps(self.xmm_crc3);
        let ps_crc1 = arch::_mm_castsi128_ps(self.xmm_crc1);
        let ps_res31 = arch::_mm_xor_ps(ps_crc3, ps_crc1);

        self.xmm_crc2 = self.xmm_crc0;
        self.xmm_crc0 = arch::_mm_clmulepi64_si128(self.xmm_crc0, xmm_fold4, 0x01);
        self.xmm_crc2 = arch::_mm_clmulepi64_si128(self.xmm_crc2, xmm_fold4, 0x10);
        let ps_crc0 = arch::_mm_castsi128_ps(self.xmm_crc0);
        let ps_crc2 = arch::_mm_castsi128_ps(self.xmm_crc2);
        let ps_res20 = arch::_mm_xor_ps(ps_crc0, ps_crc2);

        self.xmm_crc0 = x_tmp2;
        self.xmm_crc1 = x_tmp3;
        self.xmm_crc2 = arch::_mm_castps_si128(ps_res20);
        self.xmm_crc3 = arch::_mm_castps_si128(ps_res31);
    }

    #[target_feature(enable = "sse", enable = "sse2", enable = "pclmulqdq")]
    unsafe fn fold_3(&mut self) {
        debug_assert!(std::is_x86_feature_detected!("sse"));
        debug_assert!(std::is_x86_feature_detected!("sse2"));
        debug_assert!(std::is_x86_feature_detected!("pclmulqdq"));

        let xmm_fold4 =
            arch::_mm_set_epi32(0x00000001, 0x54442bd4, 0x00000001, 0xc6e41596u32 as i32);

        let x_tmp3 = self.xmm_crc3;

        self.xmm_crc3 = self.xmm_crc2;
        self.xmm_crc2 = arch::_mm_clmulepi64_si128(self.xmm_crc2, xmm_fold4, 0x01);
        self.xmm_crc3 = arch::_mm_clmulepi64_si128(self.xmm_crc3, xmm_fold4, 0x10);
        let ps_crc2 = arch::_mm_castsi128_ps(self.xmm_crc2);
        let ps_crc3 = arch::_mm_castsi128_ps(self.xmm_crc3);
        let ps_res32 = arch::_mm_xor_ps(ps_crc2, ps_crc3);

        self.xmm_crc2 = self.xmm_crc1;
        self.xmm_crc1 = arch::_mm_clmulepi64_si128(self.xmm_crc1, xmm_fold4, 0x01);
        self.xmm_crc2 = arch::_mm_clmulepi64_si128(self.xmm_crc2, xmm_fold4, 0x10);
        let ps_crc1 = arch::_mm_castsi128_ps(self.xmm_crc1);
        let ps_crc2 = arch::_mm_castsi128_ps(self.xmm_crc2);
        let ps_res21 = arch::_mm_xor_ps(ps_crc1, ps_crc2);

        self.xmm_crc1 = self.xmm_crc0;
        self.xmm_crc0 = arch::_mm_clmulepi64_si128(self.xmm_crc0, xmm_fold4, 0x01);
        self.xmm_crc1 = arch::_mm_clmulepi64_si128(self.xmm_crc1, xmm_fold4, 0x10);
        let ps_crc0 = arch::_mm_castsi128_ps(self.xmm_crc0);
        let ps_crc1 = arch::_mm_castsi128_ps(self.xmm_crc1);
        let ps_res10 = arch::_mm_xor_ps(ps_crc0, ps_crc1);

        self.xmm_crc0 = x_tmp3;
        self.xmm_crc1 = arch::_mm_castps_si128(ps_res10);
        self.xmm_crc2 = arch::_mm_castps_si128(ps_res21);
        self.xmm_crc3 = arch::_mm_castps_si128(ps_res32);
    }

    #[target_feature(enable = "sse", enable = "sse2", enable = "pclmulqdq")]
    unsafe fn fold_4(&mut self) {
        debug_assert!(std::is_x86_feature_detected!("sse"));
        debug_assert!(std::is_x86_feature_detected!("sse2"));
        debug_assert!(std::is_x86_feature_detected!("pclmulqdq"));

        let xmm_fold4 =
            arch::_mm_set_epi32(0x00000001, 0x54442bd4, 0x00000001, 0xc6e41596u32 as i32);

        let mut x_tmp0 = self.xmm_crc0;
        let mut x_tmp1 = self.xmm_crc1;
        let mut x_tmp2 = self.xmm_crc2;
        let mut x_tmp3 = self.xmm_crc3;

        self.xmm_crc0 = arch::_mm_clmulepi64_si128(self.xmm_crc0, xmm_fold4, 0x01);
        x_tmp0 = arch::_mm_clmulepi64_si128(x_tmp0, xmm_fold4, 0x10);
        let ps_crc0 = arch::_mm_castsi128_ps(self.xmm_crc0);
        let ps_t0 = arch::_mm_castsi128_ps(x_tmp0);
        let ps_res0 = arch::_mm_xor_ps(ps_crc0, ps_t0);

        self.xmm_crc1 = arch::_mm_clmulepi64_si128(self.xmm_crc1, xmm_fold4, 0x01);
        x_tmp1 = arch::_mm_clmulepi64_si128(x_tmp1, xmm_fold4, 0x10);
        let ps_crc1 = arch::_mm_castsi128_ps(self.xmm_crc1);
        let ps_t1 = arch::_mm_castsi128_ps(x_tmp1);
        let ps_res1 = arch::_mm_xor_ps(ps_crc1, ps_t1);

        self.xmm_crc2 = arch::_mm_clmulepi64_si128(self.xmm_crc2, xmm_fold4, 0x01);
        x_tmp2 = arch::_mm_clmulepi64_si128(x_tmp2, xmm_fold4, 0x10);
        let ps_crc2 = arch::_mm_castsi128_ps(self.xmm_crc2);
        let ps_t2 = arch::_mm_castsi128_ps(x_tmp2);
        let ps_res2 = arch::_mm_xor_ps(ps_crc2, ps_t2);

        self.xmm_crc3 = arch::_mm_clmulepi64_si128(self.xmm_crc3, xmm_fold4, 0x01);
        x_tmp3 = arch::_mm_clmulepi64_si128(x_tmp3, xmm_fold4, 0x10);
        let ps_crc3 = arch::_mm_castsi128_ps(self.xmm_crc3);
        let ps_t3 = arch::_mm_castsi128_ps(x_tmp3);
        let ps_res3 = arch::_mm_xor_ps(ps_crc3, ps_t3);

        self.xmm_crc0 = arch::_mm_castps_si128(ps_res0);
        self.xmm_crc1 = arch::_mm_castps_si128(ps_res1);
        self.xmm_crc2 = arch::_mm_castps_si128(ps_res2);
        self.xmm_crc3 = arch::_mm_castps_si128(ps_res3);
    }

    #[target_feature(
        enable = "sse",
        enable = "sse2",
        enable = "ssse3",
        enable = "pclmulqdq"
    )]
    unsafe fn partial_fold(&mut self, len: usize) {
        debug_assert!(std::is_x86_feature_detected!("sse"));
        debug_assert!(std::is_x86_feature_detected!("sse2"));
        debug_assert!(std::is_x86_feature_detected!("ssse3"));
        debug_assert!(std::is_x86_feature_detected!("pclmulqdq"));

        let pshufb_shf_table = PSHUFB_SHF_TABLE.0.as_ptr() as *const arch::__m128i;

        let xmm_fold4 =
            arch::_mm_set_epi32(0x00000001, 0x54442bd4, 0x00000001, 0xc6e41596u32 as i32);
        let xmm_mask3 = arch::_mm_set1_epi32(0x80808080u32 as i32);

        let xmm_shl = arch::_mm_load_si128(pshufb_shf_table.add(len - 1));
        // why this was originally two statements??
        let xmm_shr = arch::_mm_xor_si128(xmm_shl, xmm_mask3);

        let xmm_a0_0 = arch::_mm_shuffle_epi8(self.xmm_crc0, xmm_shl);

        self.xmm_crc0 = arch::_mm_shuffle_epi8(self.xmm_crc0, xmm_shr);
        let xmm_tmp1 = arch::_mm_shuffle_epi8(self.xmm_crc1, xmm_shl);
        self.xmm_crc0 = arch::_mm_or_si128(self.xmm_crc0, xmm_tmp1);

        self.xmm_crc1 = arch::_mm_shuffle_epi8(self.xmm_crc1, xmm_shr);
        let xmm_tmp2 = arch::_mm_shuffle_epi8(self.xmm_crc2, xmm_shl);
        self.xmm_crc1 = arch::_mm_or_si128(self.xmm_crc1, xmm_tmp2);

        self.xmm_crc2 = arch::_mm_shuffle_epi8(self.xmm_crc2, xmm_shr);
        let xmm_tmp3 = arch::_mm_shuffle_epi8(self.xmm_crc3, xmm_shl);
        self.xmm_crc2 = arch::_mm_or_si128(self.xmm_crc2, xmm_tmp3);

        self.xmm_crc3 = arch::_mm_shuffle_epi8(self.xmm_crc3, xmm_shr);
        self.xmm_crc_part = arch::_mm_shuffle_epi8(self.xmm_crc_part, xmm_shl);
        self.xmm_crc3 = arch::_mm_or_si128(self.xmm_crc3, self.xmm_crc_part);
        let xmm_a0_1 = arch::_mm_clmulepi64_si128(xmm_a0_0, xmm_fold4, 0x10);
        let xmm_a0_0 = arch::_mm_clmulepi64_si128(xmm_a0_0, xmm_fold4, 0x01);

        let ps_crc3 = arch::_mm_castsi128_ps(self.xmm_crc3);
        let psa0_0 = arch::_mm_castsi128_ps(xmm_a0_0);
        let psa0_1 = arch::_mm_castsi128_ps(xmm_a0_1);

        let mut ps_res = arch::_mm_xor_ps(ps_crc3, psa0_0);
        ps_res = arch::_mm_xor_ps(ps_res, psa0_1);

        self.xmm_crc3 = arch::_mm_castps_si128(ps_res);
    }

    #[inline(always)]
    unsafe fn partial_load(&mut self, ptr: *const u8, len: usize) {
        ::std::ptr::copy_nonoverlapping(
            ptr as *const u8,
            &mut self.xmm_crc_part as *const arch::__m128i as *mut u8,
            len,
        );
    }

    #[target_feature(
        enable = "sse",
        enable = "sse2",
        enable = "ssse3",
        enable = "pclmulqdq"
    )]
    unsafe fn fold(&mut self, bytes: &[u8]) {
        debug_assert!(std::is_x86_feature_detected!("sse"));
        debug_assert!(std::is_x86_feature_detected!("sse2"));
        debug_assert!(std::is_x86_feature_detected!("ssse3"));
        debug_assert!(std::is_x86_feature_detected!("pclmulqdq"));

        let mut src = bytes.as_ptr();
        let mut len = bytes.len();

        if len == 0 {
            return;
        }

        if len < 16 {
            self.partial_load(src, len);
            return self.partial_fold(len);
        }

        let algn_diff: u64 = (0 - src as i64) as u64 & 0xF;
        if algn_diff > 0 {
            self.xmm_crc_part = arch::_mm_loadu_si128(src as *const arch::__m128i);
            src = src.add(algn_diff as usize);
            len -= algn_diff as usize;

            if len == 0 {
                return;
            }

            self.partial_fold(algn_diff as usize);
        }

        while len >= 64 {
            let msrc = src as *const arch::__m128i;

            let xmm_t0 = arch::_mm_load_si128(msrc);
            let xmm_t1 = arch::_mm_load_si128(msrc.add(1));
            let xmm_t2 = arch::_mm_load_si128(msrc.add(2));
            let xmm_t3 = arch::_mm_load_si128(msrc.add(3));

            self.fold_4();

            self.xmm_crc0 = arch::_mm_xor_si128(self.xmm_crc0, xmm_t0);
            self.xmm_crc1 = arch::_mm_xor_si128(self.xmm_crc1, xmm_t1);
            self.xmm_crc2 = arch::_mm_xor_si128(self.xmm_crc2, xmm_t2);
            self.xmm_crc3 = arch::_mm_xor_si128(self.xmm_crc3, xmm_t3);

            src = src.add(64);
            len -= 64;
        }

        if len >= 48 {
            let msrc = src as *const arch::__m128i;

            let xmm_t0 = arch::_mm_load_si128(msrc);
            let xmm_t1 = arch::_mm_load_si128(msrc.add(1));
            let xmm_t2 = arch::_mm_load_si128(msrc.add(2));

            self.fold_3();

            self.xmm_crc1 = arch::_mm_xor_si128(self.xmm_crc1, xmm_t0);
            self.xmm_crc2 = arch::_mm_xor_si128(self.xmm_crc2, xmm_t1);
            self.xmm_crc3 = arch::_mm_xor_si128(self.xmm_crc3, xmm_t2);

            len -= 48;
            if len == 0 {
                return;
            }

            self.partial_load(msrc.add(3) as *const u8, len);
        } else if len >= 32 {
            let msrc = src as *const arch::__m128i;

            let xmm_t0 = arch::_mm_load_si128(msrc);
            let xmm_t1 = arch::_mm_load_si128(msrc.add(1));

            self.fold_2();

            self.xmm_crc2 = arch::_mm_xor_si128(self.xmm_crc2, xmm_t0);
            self.xmm_crc3 = arch::_mm_xor_si128(self.xmm_crc3, xmm_t1);

            len -= 32;
            if len == 0 {
                return;
            }

            self.partial_load(msrc.add(2) as *const u8, len);
        } else if len >= 16 {
            let msrc = src as *const arch::__m128i;

            let xmm_t0 = arch::_mm_load_si128(msrc);

            self.fold_1();

            self.xmm_crc3 = arch::_mm_xor_si128(self.xmm_crc3, xmm_t0);

            len -= 16;
            if len == 0 {
                return;
            }

            self.partial_load(msrc.add(1) as *const u8, len);
        } else {
            let msrc = src as *const arch::__m128i;

            if len == 0 {
                return;
            }

            self.partial_load(msrc.add(0) as *const u8, len);
        }

        self.partial_fold(len);
    }

    #[target_feature(enable = "sse2", enable = "sse4.1", enable = "pclmulqdq")]
    unsafe fn fold_512to32(&mut self) -> u32 {
        debug_assert!(std::is_x86_feature_detected!("sse2"));
        debug_assert!(std::is_x86_feature_detected!("sse4.1"));
        debug_assert!(std::is_x86_feature_detected!("pclmulqdq"));

        let crc_k = CRC_K.0.as_ptr() as *const arch::__m128i;

        let xmm_mask = arch::_mm_load_si128(CRC_MASK.0.as_ptr() as *const arch::__m128i);
        let xmm_mask2 = arch::_mm_load_si128(CRC_MASK2.0.as_ptr() as *const arch::__m128i);

        let mut crc_fold: arch::__m128i;

        // k1
        crc_fold = arch::_mm_load_si128(crc_k);

        let x_tmp0 = arch::_mm_clmulepi64_si128(self.xmm_crc0, crc_fold, 0x10);
        self.xmm_crc0 = arch::_mm_clmulepi64_si128(self.xmm_crc0, crc_fold, 0x01);
        self.xmm_crc1 = arch::_mm_xor_si128(self.xmm_crc1, x_tmp0);
        self.xmm_crc1 = arch::_mm_xor_si128(self.xmm_crc1, self.xmm_crc0);

        let x_tmp1 = arch::_mm_clmulepi64_si128(self.xmm_crc1, crc_fold, 0x10);
        self.xmm_crc1 = arch::_mm_clmulepi64_si128(self.xmm_crc1, crc_fold, 0x01);
        self.xmm_crc2 = arch::_mm_xor_si128(self.xmm_crc2, x_tmp1);
        self.xmm_crc2 = arch::_mm_xor_si128(self.xmm_crc2, self.xmm_crc1);

        let x_tmp2 = arch::_mm_clmulepi64_si128(self.xmm_crc2, crc_fold, 0x10);
        self.xmm_crc2 = arch::_mm_clmulepi64_si128(self.xmm_crc2, crc_fold, 0x01);
        self.xmm_crc3 = arch::_mm_xor_si128(self.xmm_crc3, x_tmp2);
        self.xmm_crc3 = arch::_mm_xor_si128(self.xmm_crc3, self.xmm_crc2);

        // k5
        crc_fold = arch::_mm_load_si128(crc_k.add(1));

        self.xmm_crc0 = self.xmm_crc3;
        self.xmm_crc3 = arch::_mm_clmulepi64_si128(self.xmm_crc3, crc_fold, 0);
        self.xmm_crc0 = arch::_mm_srli_si128(self.xmm_crc0, 8);
        self.xmm_crc3 = arch::_mm_xor_si128(self.xmm_crc3, self.xmm_crc0);

        self.xmm_crc0 = self.xmm_crc3;
        self.xmm_crc3 = arch::_mm_slli_si128(self.xmm_crc3, 4);
        self.xmm_crc3 = arch::_mm_clmulepi64_si128(self.xmm_crc3, crc_fold, 0x10);
        self.xmm_crc3 = arch::_mm_xor_si128(self.xmm_crc3, self.xmm_crc0);
        self.xmm_crc3 = arch::_mm_and_si128(self.xmm_crc3, xmm_mask2);

        // k7
        self.xmm_crc1 = self.xmm_crc3;
        self.xmm_crc2 = self.xmm_crc3;
        crc_fold = arch::_mm_load_si128(crc_k.add(2));

        self.xmm_crc3 = arch::_mm_clmulepi64_si128(self.xmm_crc3, crc_fold, 0);
        self.xmm_crc3 = arch::_mm_xor_si128(self.xmm_crc3, self.xmm_crc2);
        self.xmm_crc3 = arch::_mm_and_si128(self.xmm_crc3, xmm_mask);

        self.xmm_crc2 = self.xmm_crc3;
        self.xmm_crc3 = arch::_mm_clmulepi64_si128(self.xmm_crc3, crc_fold, 0x10);
        self.xmm_crc3 = arch::_mm_xor_si128(self.xmm_crc3, self.xmm_crc2);
        self.xmm_crc3 = arch::_mm_xor_si128(self.xmm_crc3, self.xmm_crc1);

        let crc = arch::_mm_extract_epi32(self.xmm_crc3, 2) as u32;
        return !crc;
    }
}

#[repr(align(16))]
struct Align16<T>(T);

static PSHUFB_SHF_TABLE: Align16<[u32; 60]> = Align16([
    0x84838281, 0x88878685, 0x8c8b8a89, 0x008f8e8d, /* shl 15 (16 - 1)/shr1 */
    0x85848382, 0x89888786, 0x8d8c8b8a, 0x01008f8e, /* shl 14 (16 - 3)/shr2 */
    0x86858483, 0x8a898887, 0x8e8d8c8b, 0x0201008f, /* shl 13 (16 - 4)/shr3 */
    0x87868584, 0x8b8a8988, 0x8f8e8d8c, 0x03020100, /* shl 12 (16 - 4)/shr4 */
    0x88878685, 0x8c8b8a89, 0x008f8e8d, 0x04030201, /* shl 11 (16 - 5)/shr5 */
    0x89888786, 0x8d8c8b8a, 0x01008f8e, 0x05040302, /* shl 10 (16 - 6)/shr6 */
    0x8a898887, 0x8e8d8c8b, 0x0201008f, 0x06050403, /* shl  9 (16 - 7)/shr7 */
    0x8b8a8988, 0x8f8e8d8c, 0x03020100, 0x07060504, /* shl  8 (16 - 8)/shr8 */
    0x8c8b8a89, 0x008f8e8d, 0x04030201, 0x08070605, /* shl  7 (16 - 9)/shr9 */
    0x8d8c8b8a, 0x01008f8e, 0x05040302, 0x09080706, /* shl  6 (16 -10)/shr10*/
    0x8e8d8c8b, 0x0201008f, 0x06050403, 0x0a090807, /* shl  5 (16 -11)/shr11*/
    0x8f8e8d8c, 0x03020100, 0x07060504, 0x0b0a0908, /* shl  4 (16 -12)/shr12*/
    0x008f8e8d, 0x04030201, 0x08070605, 0x0c0b0a09, /* shl  3 (16 -13)/shr13*/
    0x01008f8e, 0x05040302, 0x09080706, 0x0d0c0b0a, /* shl  2 (16 -14)/shr14*/
    0x0201008f, 0x06050403, 0x0a090807, 0x0e0d0c0b, /* shl  1 (16 -15)/shr15*/
]);

static CRC_K: Align16<[u32; 12]> = Align16([
    0xccaa009e, 0x00000000, /* rk1 */
    0x751997d0, 0x00000001, /* rk2 */
    0xccaa009e, 0x00000000, /* rk5 */
    0x63cd6124, 0x00000001, /* rk6 */
    0xf7011640, 0x00000001, /* rk7 */
    0xdb710640, 0x00000001, /* rk8 */
]);

static CRC_MASK: Align16<[u32; 4]> = Align16([0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000]);

static CRC_MASK2: Align16<[u32; 4]> = Align16([0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]);

#[cfg(test)]
mod test {
    quickcheck! {
        fn check_against_baseline(chunks: Vec<(Vec<u8>, usize)>) -> bool {
            let mut baseline = super::super::super::baseline::State::new();
            let mut pclmulqdq = super::State::new().expect("not supported");
            for (chunk, mut offset) in chunks {
                // simulate random alignments by offsetting the slice by up to 15 bytes
                offset = offset & 0xF;
                if chunk.len() <= offset {
                    baseline.update(&chunk);
                    pclmulqdq.update(&chunk);
                } else {
                    baseline.update(&chunk[offset..]);
                    pclmulqdq.update(&chunk[offset..]);
                }
            }
            pclmulqdq.finalize() == baseline.finalize()
        }
    }
}
