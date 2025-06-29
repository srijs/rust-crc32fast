const POLY: u32 = 0xedb88320;

static X2N_TABLE: [u32; 32] = [
    0x00800000, 0x00008000, 0xedb88320, 0xb1e6b092, 0xa06a2517, 0xed627dae, 0x88d14467, 0xd7bbfe6a,
    0xec447f11, 0x8e7ea170, 0x6427800e, 0x4d47bae0, 0x09fe548f, 0x83852d0f, 0x30362f1a, 0x7b5a9cc3,
    0x31fec169, 0x9fec022a, 0x6c8dedc4, 0x15d6874d, 0x5fde7a4e, 0xbad90e37, 0x2e4e5eef, 0x4eaba214,
    0xa8a472c0, 0x429a969e, 0x148d302a, 0xc40ba6d0, 0xc4e22c3c, 0x40000000, 0x20000000, 0x08000000,
];

// Calculates a(x) multiplied by b(x) modulo p(x), where p(x) is the CRC polynomial,
// reflected. For speed, this requires that a not be zero.
fn multiply(a: u32, mut b: u32) -> u32 {
    let mut m = 1u32 << 31;
    let mut p = 0u32;

    loop {
        if (a & m) != 0 {
            p ^= b;
            if (a & (m - 1)) == 0 {
                break;
            }
        }
        m >>= 1;
        if b & 1 != 0 {
            b = (b >> 1) ^ POLY;
        } else {
            b >>= 1;
        }
    }

    p
}

pub(crate) fn combine(crc1: u32, crc2: u32, len2: u64) -> u32 {
    let mut p = 1u32 << 31; // x^0 == 1
    let n = 64 - len2.leading_zeros();

    for i in 0..n {
        if (len2 >> i & 1) != 0 {
            p = multiply(X2N_TABLE[(i & 0x1F) as usize], p);
        }
    }

    multiply(p, crc1) ^ crc2
}

#[test]
fn golden() {
    assert_eq!(
        combine(0xB8AD0532, 0x804754D9, 0x19B77C403D9D90EE),
        940758956
    );
    assert_eq!(
        combine(0xF310DC54, 0x8B65DF79, 0x2F0327F1309076FF),
        3454617599
    );
}
