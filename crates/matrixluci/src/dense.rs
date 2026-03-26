//! Dense pivot-kernel implementations.

use crate::kernel::PivotKernel;
use crate::scalar::Scalar;
use crate::source::{materialize_source, CandidateMatrixSource};
use crate::{PivotKernelOptions, PivotSelectionCore, Result};
use faer::MatRef;
use num_complex::{Complex32, Complex64};

/// Dense LU kernel backed by faer.
#[derive(Default)]
pub struct DenseFaerLuKernel;

impl DenseFaerLuKernel {
    fn compute_pivot_errors(
        diag_abs: &[f64],
        nrows: usize,
        ncols: usize,
        options: &PivotKernelOptions,
    ) -> Vec<f64> {
        let full_rank = nrows.min(ncols);
        if full_rank == 0 {
            return vec![0.0];
        }

        let max_rank = options.max_rank.min(full_rank);
        let mut accepted = Vec::new();
        let mut max_error = 0.0f64;
        let mut last_error = f64::NAN;
        let mut rank = 0usize;

        while rank < max_rank {
            let pivot_abs = diag_abs.get(rank).copied().unwrap_or(0.0);
            last_error = pivot_abs;

            if rank > 0 && (pivot_abs < options.rel_tol * max_error || pivot_abs < options.abs_tol)
            {
                break;
            }

            if pivot_abs < f64::EPSILON {
                if rank == 0 {
                    last_error = pivot_abs;
                }
                break;
            }

            max_error = max_error.max(pivot_abs);
            accepted.push(pivot_abs);
            rank += 1;
        }

        if rank >= full_rank {
            last_error = 0.0;
        } else if rank == max_rank && rank > 0 {
            // Preserve current matrixci semantics for max_rank stopping.
            last_error = accepted[rank - 1];
        }

        accepted.push(last_error);
        accepted
    }
}

macro_rules! impl_dense_kernel {
    ($t:ty) => {
        impl PivotKernel<$t> for DenseFaerLuKernel {
            fn factorize<S: CandidateMatrixSource<$t>>(
                &self,
                source: &S,
                options: &PivotKernelOptions,
            ) -> Result<PivotSelectionCore> {
                let run = |data: &[$t]| -> Result<PivotSelectionCore> {
                    let nrows = source.nrows();
                    let ncols = source.ncols();
                    let mat = MatRef::from_column_major_slice(data, nrows, ncols);
                    let lu = mat.full_piv_lu();

                    let rank_cap = nrows.min(ncols);
                    let u = lu.U();
                    let mut diag_abs = Vec::with_capacity(rank_cap);
                    for i in 0..rank_cap {
                        diag_abs.push(u[(i, i)].abs_sq().sqrt());
                    }

                    let pivot_errors =
                        DenseFaerLuKernel::compute_pivot_errors(&diag_abs, nrows, ncols, options);
                    let rank = pivot_errors.len().saturating_sub(1);

                    let (row_fwd, _) = lu.P().arrays();
                    let (col_fwd, _) = lu.Q().arrays();

                    Ok(PivotSelectionCore {
                        row_indices: row_fwd.iter().take(rank).copied().collect(),
                        col_indices: col_fwd.iter().take(rank).copied().collect(),
                        pivot_errors,
                        rank,
                    })
                };

                if let Some(data) = source.dense_column_major_slice() {
                    run(data)
                } else {
                    let materialized = materialize_source(source);
                    run(materialized.as_slice())
                }
            }
        }
    };
}

impl_dense_kernel!(f32);
impl_dense_kernel!(f64);
impl_dense_kernel!(Complex32);
impl_dense_kernel!(Complex64);

#[cfg(test)]
mod tests;
