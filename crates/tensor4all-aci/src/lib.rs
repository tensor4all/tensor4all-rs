#![warn(missing_docs)]
//! Alternating Cross Interpolation for elementwise tensor-train operations.
//!
//! This crate ports the Rust public API for
//! `AlternatingCrossInterpolation.jl`, originally authored by
//! Marc Ritter <mritter@flatironinstitute.org> and contributors.

mod batch;
mod error;
mod options;
#[allow(dead_code)]
mod random_tt;
mod result;
#[allow(dead_code)]
mod scalar;
#[allow(dead_code)]
pub(crate) mod validation;

pub use batch::ElementwiseBatch;
pub use error::{AciError, Result};
pub use options::AciOptions;
#[allow(unused_imports)]
pub(crate) use random_tt::initial_guess;
pub use result::AciResult;

#[cfg(test)]
mod tests;
