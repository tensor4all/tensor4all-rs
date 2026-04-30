#![warn(missing_docs)]
//! Low-level LUCI / rrLU substrate.
//!
//! The dense LU path materializes candidate matrices for exact pivot selection.
//! It uses the configured tensor backend for square complete-pivoting LU and
//! the internal rrLU implementation for rectangular fallback coverage.

pub mod block_rook;
pub mod dense;
pub mod error;
pub mod factors;
pub mod kernel;
pub mod scalar;
pub mod source;
pub mod types;

pub use block_rook::LazyBlockRookKernel;
pub use dense::DenseLuKernel;
pub use error::{MatrixLuciError, Result};
pub use factors::CrossFactors;
pub use kernel::PivotKernel;
pub use scalar::Scalar;
pub use source::{CandidateMatrixSource, DenseMatrixSource, LazyMatrixSource};
pub use types::{DenseOwnedMatrix, PivotKernelOptions, PivotSelectionCore};
