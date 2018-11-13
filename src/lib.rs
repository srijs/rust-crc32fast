//! ## Example
//!
//! ```rust
//! extern crate crc32fast;
//!
//! use crc32fast::Hasher;
//!
//! let mut hasher = Hasher::new();
//! hasher.update(b"foo bar baz");
//! let checksum = hasher.finalize();
//! ```
//!
//! ## Performance
//!
//! This crate contains multiple CRC32 implementations:
//!
//! - A fast baseline implementation which processes up to 16 bytes per iteration
//! - An optimized implementation for modern `x86` using `sse` and `pclmulqdq` instructions
//!
//! Calling the `Hasher::new` constructor at runtime will perform a feature detection to select the most
//! optimal implementation for the current CPU feature set.

#[deny(missing_docs)]

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use std::fmt;
use std::hash;

mod baseline;
mod specialized;
mod table;

#[derive(Clone)]
enum State {
    Baseline(baseline::State),
    Specialized(specialized::State),
}

#[derive(Clone)]
/// Represents an in-progress CRC32 computation.
pub struct Hasher {
    state: State,
}

impl Hasher {
    /// Create a new `Hasher`.
    ///
    /// This will perform a CPU feature detection at runtime to select the most
    /// optimal implementation for the current processor architecture.
    pub fn new() -> Self {
        Self::internal_new_specialized().unwrap_or_else(|| Self::internal_new_baseline())
    }

    #[doc(hidden)]
    // Internal-only API. Don't use.
    pub fn internal_new_baseline() -> Self {
        Hasher {
            state: State::Baseline(baseline::State::new()),
        }
    }

    #[doc(hidden)]
    // Internal-only API. Don't use.
    pub fn internal_new_specialized() -> Option<Self> {
        {
            if let Some(state) = specialized::State::new() {
                return Some(Hasher {
                    state: State::Specialized(state),
                });
            }
        }
        None
    }

    /// Process the given byte slice and update the hash state.
    pub fn update(&mut self, buf: &[u8]) {
        match self.state {
            State::Baseline(ref mut state) => state.update(buf),
            State::Specialized(ref mut state) => state.update(buf),
        }
    }

    /// Finalize the hash state and return the computed CRC32 value.
    pub fn finalize(self) -> u32 {
        match self.state {
            State::Baseline(state) => state.finalize(),
            State::Specialized(state) => state.finalize(),
        }
    }
}

impl fmt::Debug for Hasher {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("crc32fast::Hasher").finish()
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl hash::Hasher for Hasher {
    fn write(&mut self, bytes: &[u8]) {
        self.update(bytes)
    }

    fn finish(&self) -> u64 {
        self.clone().finalize() as u64
    }
}
