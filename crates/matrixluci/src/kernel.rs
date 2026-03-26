//! Pivot-kernel traits.

use crate::scalar::Scalar;
use crate::source::CandidateMatrixSource;
use crate::types::{PivotKernelOptions, PivotSelectionCore};

/// Kernel that selects pivot rows and columns.
pub trait PivotKernel<T: Scalar> {
    /// Factorize the candidate matrix and return pivot-only output.
    fn factorize<S: CandidateMatrixSource<T>>(
        &self,
        source: &S,
        options: &PivotKernelOptions,
    ) -> crate::Result<PivotSelectionCore>;
}

#[cfg(test)]
mod tests;
