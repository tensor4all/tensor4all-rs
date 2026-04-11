//! Pivot-kernel traits.
//!
//! The [`PivotKernel`] trait abstracts pivot selection strategies.
//! Implementations include [`DenseFaerLuKernel`](super::DenseFaerLuKernel)
//! (full-pivoting LU via faer) and
//! [`LazyBlockRookKernel`](super::LazyBlockRookKernel) (residual-based
//! rook search for lazy sources).

use crate::matrixluci::scalar::Scalar;
use crate::matrixluci::source::CandidateMatrixSource;
use crate::matrixluci::types::{PivotKernelOptions, PivotSelectionCore};

/// Kernel that selects pivot rows and columns from a candidate matrix.
///
/// Different implementations choose pivots using different strategies
/// (dense full-pivoting LU, lazy rook search, etc.).
pub trait PivotKernel<T: Scalar> {
    /// Factorize the candidate matrix and return pivot-only output.
    fn factorize<S: CandidateMatrixSource<T>>(
        &self,
        source: &S,
        options: &PivotKernelOptions,
    ) -> crate::matrixluci::Result<PivotSelectionCore>;
}

#[cfg(test)]
mod tests;
