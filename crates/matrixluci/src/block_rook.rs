//! Lazy pivot-kernel implementations.

use crate::dense::DenseFaerLuKernel;
use crate::kernel::PivotKernel;
use crate::source::CandidateMatrixSource;
use crate::types::{PivotKernelOptions, PivotSelectionCore};
use crate::Result;
use num_complex::{Complex32, Complex64};

/// Correctness-first lazy kernel placeholder.
///
/// The current implementation reuses the dense kernel path, which already
/// materializes non-dense sources through `CandidateMatrixSource::get_block`.
/// A dedicated block-rook search loop will replace this fallback in a follow-up
/// change.
#[derive(Default)]
pub struct LazyBlockRookKernel;

macro_rules! impl_lazy_block_rook_kernel {
    ($t:ty) => {
        impl PivotKernel<$t> for LazyBlockRookKernel {
            fn factorize<S: CandidateMatrixSource<$t>>(
                &self,
                source: &S,
                options: &PivotKernelOptions,
            ) -> Result<PivotSelectionCore> {
                DenseFaerLuKernel.factorize(source, options)
            }
        }
    };
}

impl_lazy_block_rook_kernel!(f32);
impl_lazy_block_rook_kernel!(f64);
impl_lazy_block_rook_kernel!(Complex32);
impl_lazy_block_rook_kernel!(Complex64);

#[cfg(test)]
mod tests;
