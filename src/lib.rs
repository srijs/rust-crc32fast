#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use std::fmt;
use std::hash;

mod baseline;
#[cfg(pclmulqdq)]
mod pclmulqdq;
mod table;

#[derive(Clone)]
enum State {
    Baseline(baseline::State),
    #[cfg(pclmulqdq)]
    Pclmulqdq(pclmulqdq::State),
}

#[derive(Clone)]
pub struct Hasher {
    state: State,
}

impl Hasher {
    pub fn new() -> Self {
        Self::internal_new_pclmulqdq().unwrap_or_else(|| Self::internal_new_baseline())
    }

    #[doc(hidden)]
    pub fn internal_new_baseline() -> Self {
        Hasher {
            state: State::Baseline(baseline::State::new()),
        }
    }

    #[doc(hidden)]
    pub fn internal_new_pclmulqdq() -> Option<Self> {
        #[cfg(pclmulqdq)]
        {
            if let Some(state) = pclmulqdq::State::new() {
                return Some(Hasher {
                    state: State::Pclmulqdq(state),
                });
            }
        }
        None
    }

    pub fn update(&mut self, buf: &[u8]) {
        match self.state {
            State::Baseline(ref mut state) => state.update(buf),
            #[cfg(pclmulqdq)]
            State::Pclmulqdq(ref mut state) => state.update(buf),
        }
    }

    pub fn finalize(self) -> u32 {
        match self.state {
            State::Baseline(state) => state.finalize(),
            #[cfg(pclmulqdq)]
            State::Pclmulqdq(state) => state.finalize(),
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
