//! Dense pivot-kernel implementations.

use crate::matrixlu::pivot_is_undividable;
use crate::matrixluci::kernel::PivotKernel;
use crate::matrixluci::scalar::MatrixLuciScalar;
use crate::matrixluci::source::{materialize_source, CandidateMatrixSource};
use crate::matrixluci::{MatrixLuciError, PivotKernelOptions, PivotSelectionCore, Result};
use crate::scalar::Scalar as LegacyScalar;
use crate::{rrlu, RrLUOptions};
use num_complex::{Complex32, Complex64};
use tensor4all_tensorbackend::{capped_full_piv_lu, CappedFullPivLuScalar, Matrix};

/// Dense full-pivoting LU kernel.
///
/// Materializes the source matrix and performs dense pivot selection. Square
/// matrices use the backend's complete-pivoting LU stopped at the caller's rank
/// cap ([`tensor4all_tensorbackend::capped_full_piv_lu`]); rectangular matrices
/// fall back to the internal rrLU implementation until the provider stack
/// exposes rectangular complete pivoting.
#[derive(Default)]
pub struct DenseLuKernel;

impl DenseLuKernel {
    fn is_no_truncation(options: &PivotKernelOptions, full_rank: usize) -> bool {
        options.max_bond_dim >= full_rank && options.rel_tol == 0.0 && options.abs_tol == 0.0
    }

    fn compute_pivot_errors<T: LegacyScalar>(
        diag_abs: &[f64],
        nrows: usize,
        ncols: usize,
        options: &PivotKernelOptions,
    ) -> Vec<f64> {
        let full_rank = nrows.min(ncols);
        if full_rank == 0 {
            return vec![0.0];
        }

        if Self::is_no_truncation(options, full_rank) {
            let mut pivot_errors = Vec::with_capacity(full_rank + 1);
            for &pivot_abs in diag_abs.iter().take(full_rank) {
                if pivot_is_undividable::<T>(pivot_abs) {
                    pivot_errors.push(pivot_abs);
                    return pivot_errors;
                }
                pivot_errors.push(pivot_abs);
            }
            pivot_errors.push(0.0);
            return pivot_errors;
        }

        let max_bond_dim = options.max_bond_dim.min(full_rank);
        let mut accepted = Vec::new();
        let mut max_error = 0.0f64;
        let mut last_error = f64::NAN;
        let mut rank = 0usize;

        while rank < max_bond_dim {
            let pivot_abs = diag_abs.get(rank).copied().unwrap_or(0.0);
            last_error = pivot_abs;

            if rank > 0 && (pivot_abs < options.rel_tol * max_error || pivot_abs < options.abs_tol)
            {
                break;
            }

            if pivot_is_undividable::<T>(pivot_abs) {
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
        } else if rank == max_bond_dim {
            // Complete pivoting makes the next diagonal the maximum of the
            // remaining Schur complement, including the zero-rank ceiling.
            last_error = diag_abs.get(rank).copied().unwrap_or(0.0);
        }

        accepted.push(last_error);
        accepted
    }

    fn factorize_with_rrlu<T: MatrixLuciScalar + LegacyScalar>(
        data: &[T],
        nrows: usize,
        ncols: usize,
        options: &PivotKernelOptions,
    ) -> Result<PivotSelectionCore> {
        let lu_options = RrLUOptions {
            max_bond_dim: options.max_bond_dim,
            rel_tol: options.rel_tol,
            abs_tol: options.abs_tol,
            left_orthogonal: options.left_orthogonal,
        };
        let matrix = Matrix::from_col_major_vec(nrows, ncols, data.to_vec());
        let lu =
            rrlu(&matrix, Some(lu_options)).map_err(|err| MatrixLuciError::InvalidArgument {
                message: format!("dense rrLU factorization failed: {err}"),
            })?;

        Ok(PivotSelectionCore {
            row_indices: lu.row_indices(),
            col_indices: lu.col_indices(),
            pivot_errors: lu.pivot_errors(),
            rank: lu.npivots(),
        })
    }

    /// Complete-pivoting LU of a square input, stopped once the rank cap is met.
    ///
    /// The factorization is the backend's ([`capped_full_piv_lu`]); this kernel keeps only the
    /// LUCI decision of which pivots to accept. It hands the backend the rank cap and nothing else,
    /// one pivot past the cap, so that the residual [`Self::compute_pivot_errors`] reads from
    /// `diag_abs[rank]` is the pivot the factorization itself would take next rather than a rule
    /// re-derived here. The tolerance decision stays in [`Self::compute_pivot_errors`], because
    /// faer rejects a pivot at *or below* its tolerance where this kernel rejects only a strictly
    /// smaller one (`|pivot| < tol`), and measures it against the first pivot where this kernel
    /// keeps a running maximum.
    ///
    /// Returns the selection together with the raw pivot count the backend reported, not a count
    /// clamped to the cap. That number is the evidence a cap stops the elimination instead of
    /// factoring to full rank and truncating afterwards: a bounded run reports the cap plus the one
    /// extra pivot asked for, not `n`. Clamping it here would make the two indistinguishable from
    /// the outside.
    fn factorize_square_with_backend<T: MatrixLuciScalar + CappedFullPivLuScalar>(
        data: &[T],
        n: usize,
        options: &PivotKernelOptions,
    ) -> Result<(PivotSelectionCore, usize)> {
        if data.iter().any(|&value| !value.abs_val().is_finite()) {
            return Err(MatrixLuciError::NaNEncountered {
                matrix: "dense input",
            });
        }
        if n == 0 {
            return Ok((
                PivotSelectionCore {
                    row_indices: Vec::new(),
                    col_indices: Vec::new(),
                    pivot_errors: vec![0.0],
                    rank: 0,
                },
                0,
            ));
        }

        let cap = options.max_bond_dim.min(n);
        // One pivot past the cap is the residual `compute_pivot_errors` reads (`diag_abs[rank]`). A
        // cap at or above `n` asks for no bound at all. One extra elimination step costs
        // `O((n - cap)^2)`.
        let ask = if cap < n { cap + 1 } else { cap };
        let lu = capped_full_piv_lu(
            Matrix::from_col_major_vec(n, n, data.to_vec()),
            (cap < n).then_some(ask),
        )
        .map_err(|err| MatrixLuciError::InvalidArgument {
            message: format!("dense capped full-pivoting LU failed: {err}"),
        })?;
        let pivots = lu.rank();

        // The pivots taken, and - when the elimination stopped for its own reason (a zero or
        // subnormal Schur complement) before it could reveal the extra pivot - the trailing diagonal
        // it left un-eliminated. That stop predates the rank cap, so the untruncated run reads the
        // same entries; reading them keeps the subnormal contract the residual shortcut would change.
        let read = if pivots < ask { n } else { pivots };
        let diag_abs = lu.diagonal()[..read]
            .iter()
            .map(|value| value.abs_val())
            .collect::<Vec<_>>();
        if diag_abs.iter().any(|value| !value.is_finite()) {
            return Err(MatrixLuciError::NaNEncountered {
                matrix: "dense diagonal",
            });
        }

        let pivot_errors = Self::compute_pivot_errors::<T>(&diag_abs, n, n, options);
        let rank = pivot_errors.len().saturating_sub(1);
        Ok((
            PivotSelectionCore {
                row_indices: lu.row_perm()[..rank].to_vec(),
                col_indices: lu.col_perm()[..rank].to_vec(),
                pivot_errors,
                rank,
            },
            // Raw count: the cap plus the extra pivot asked for above when the cap binds, and the
            // backend's own stop when it stopped first.
            pivots,
        ))
    }
}

macro_rules! impl_dense_kernel {
    ($t:ty) => {
        impl PivotKernel<$t> for DenseLuKernel {
            fn factorize<S: CandidateMatrixSource<$t>>(
                &self,
                source: &S,
                options: &PivotKernelOptions,
            ) -> Result<PivotSelectionCore> {
                let run = |data: &[$t]| -> Result<PivotSelectionCore> {
                    let nrows = source.nrows();
                    let ncols = source.ncols();
                    if nrows == ncols {
                        DenseLuKernel::factorize_square_with_backend(data, nrows, options)
                            .map(|(selection, _)| selection)
                    } else {
                        DenseLuKernel::factorize_with_rrlu(data, nrows, ncols, options)
                    }
                };

                if let Some(data) = source.dense_column_major_slice() {
                    run(data)
                } else {
                    let materialized = materialize_source(source);
                    run(materialized.as_col_major_slice())
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
