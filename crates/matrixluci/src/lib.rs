#![warn(missing_docs)]
//! Low-level LUCI / rrLU substrate.
//!
//! The dense LU path in this crate is a tensor4all-owned port derived from
//! `faer` full-pivoting LU ideas and primitives. Keep that attribution explicit
//! in user-facing documentation.

pub mod block_rook;
pub mod dense;
pub mod error;
pub mod kernel;
pub mod scalar;
pub mod source;
pub mod types;

pub use block_rook::LazyBlockRookKernel;
pub use dense::DenseFaerLuKernel;
pub use error::{MatrixLuciError, Result};
pub use kernel::PivotKernel;
pub use scalar::Scalar;
pub use source::{CandidateMatrixSource, DenseMatrixSource, LazyMatrixSource};
pub use types::{DenseOwnedMatrix, PivotKernelOptions, PivotSelectionCore};
