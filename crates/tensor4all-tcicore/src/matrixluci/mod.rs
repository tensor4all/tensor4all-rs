//! Temporary shim re-exporting the standalone `matrixluci` crate.
//!
//! This keeps `tensor4all_tcicore::matrixluci` stable while the low-level
//! implementation is absorbed into this crate.

pub use ::matrixluci::block_rook;
pub use ::matrixluci::dense;
pub use ::matrixluci::error;
pub use ::matrixluci::factors;
pub use ::matrixluci::kernel;
pub use ::matrixluci::scalar;
pub use ::matrixluci::source;
pub use ::matrixluci::types;

pub use ::matrixluci::error::{MatrixLuciError, Result};
pub use ::matrixluci::factors::CrossFactors;
pub use ::matrixluci::kernel::PivotKernel;
pub use ::matrixluci::scalar::Scalar;
pub use ::matrixluci::source::{CandidateMatrixSource, DenseMatrixSource, LazyMatrixSource};
pub use ::matrixluci::types::{DenseOwnedMatrix, PivotKernelOptions, PivotSelectionCore};
pub use ::matrixluci::DenseFaerLuKernel;
pub use ::matrixluci::LazyBlockRookKernel;
