#![warn(missing_docs)]
//! Low-level LUCI / rrLU substrate.
//!
//! The dense LU path materializes candidate matrices for exact pivot selection.
//! It uses the configured tensor backend for square complete-pivoting LU and
//! the internal rrLU implementation for rectangular fallback coverage.

pub(crate) mod block_rook;
// The dense kernel is exercised by the pivot-kernel tests; keep it out of the
// production dead-code budget without hiding warnings elsewhere.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod dense;
pub(crate) mod error;
pub(crate) mod factors;
pub(crate) mod kernel;
pub(crate) mod scalar;
pub(crate) mod source;
pub(crate) mod types;

pub(crate) use error::{MatrixLuciError, Result};
pub(crate) use kernel::PivotKernel;
pub use scalar::MatrixLuciScalar;
pub(crate) use types::{PivotKernelOptions, PivotSelectionCore};
