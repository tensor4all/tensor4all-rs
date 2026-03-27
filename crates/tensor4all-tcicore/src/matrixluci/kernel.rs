//! Pivot-kernel traits.

use crate::matrixluci::scalar::Scalar;
use crate::matrixluci::source::CandidateMatrixSource;
use crate::matrixluci::types::{PivotKernelOptions, PivotSelectionCore};

/// Kernel that selects pivot rows and columns.
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
