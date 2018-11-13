# Fast CRC32 (IEEE) checksum computation

## Usage

```rust
extern crate crc32fast;

use crc32fast::Hasher;

let mut hasher = Hasher::new();
hasher.update(b"foo bar baz");
let checksum = hasher.finalize();

```

## Performance

This crate contains multiple CRC32 implementations:

- A fast baseline implementation which processes up to 16 bytes per iteration
- An optimized implementation for modern `x86` using `sse` and `pclmulqdq` instructions

Calling the `Hasher::new` constructor at runtime will perform a feature detection to select the most
optimal implementation for the current CPU feature set.

| crate                               | version | variant   | ns/iter | MB/s | 
|-------------------------------------|---------|-----------|---------|------|
| [crc](https://crates.io/crates/crc) | 1.8.1   | n/a       |   4,926 |  207 |
| crc32fast (this crate)              | 1.0.0   | baseline  |     683 | 1499 |
| crc32fast (this crate)              | 1.0.0   | pclmulqdq |     140 | 7314 |

## Memory Safety

Due to the use of SIMD intrinsics for the optimized implementations, this crate contains some amount of `unsafe` code.

In order to ensure memory safety, the relevant code has been fuzz tested using [afl.rs](https://github.com/rust-fuzz/afl.rs) with millions of iterations in both `debug` and `release` build settings. You can inspect the test setup in the `fuzz` sub-directory, which also has instructions on how to run the tests yourself.

Even though the fuzzing has not revealed any safety bugs yet, please don't hesitate to file an issue if you run into any crashes or other unexpected behaviour.
