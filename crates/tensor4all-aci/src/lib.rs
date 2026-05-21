#![warn(missing_docs)]
//! Alternating Cross Interpolation for elementwise tensor-train operations.
//!
//! This crate ports the Rust public API for
//! `AlternatingCrossInterpolation.jl`, originally authored by
//! Marc Ritter <mritter@flatironinstitute.org> and contributors.

mod batch;
mod error;
mod options;
mod result;

pub use batch::ElementwiseBatch;
pub use error::{AciError, Result};
pub use options::AciOptions;
pub use result::AciResult;

#[cfg(test)]
mod tests;
