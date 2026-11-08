//! Dense pivot-kernel implementations.

use crate::matrixlu::pivot_is_undividable;
use crate::matrixluci::kernel::PivotKernel;
use crate::matrixluci::scalar::MatrixLuciScalar;
use crate::matrixluci::source::{materialize_source, CandidateMatrixSource};
use crate::matrixluci::{MatrixLuciError, PivotKernelOptions, PivotSelectionCore, Result};
use crate::scalar::Scalar as LegacyScalar;
use crate::{rrlu, RrLUOptions};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::full_pivoting::factor::{self as faer_flu, FullPivLuParams};
use faer::{Auto, Par};
use num_complex::{Complex32, Complex64};
use tensor4all_tensorbackend::Matrix;

/// Dense full-pivoting LU kernel.
///
/// Materializes the source matrix and performs dense pivot selection. Square
/// matrices use a complete-pivoting LU from faer, stopped at the caller's rank
/// cap; rectangular matrices fall back to the internal rrLU implementation
/// until the provider stack exposes rectangular complete pivoting.
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
    /// Returns the selection together with the number of pivots the factorization actually took,
    /// which is the evidence that a rank cap stops the elimination instead of factoring to full
    /// rank and truncating afterwards.
    ///
    /// The rank cap is the only bound handed to faer. The tolerance decision stays in
    /// [`Self::compute_pivot_errors`] and runs on the pivot magnitudes this factorization produced,
    /// for two reasons: faer rejects a pivot at *or below* its tolerance where this kernel rejects
    /// only a strictly smaller one (`|pivot| < tol`), and a tolerance-driven stop would leave the
    /// trailing residual below unmeasured. With every option at its no-truncation setting the call
    /// is the same factorization the tenferro path performed (faer 0.28 and 0.33 agree bit for bit
    /// with no option set), so the selection and `pivot_errors` are unchanged.
    fn factorize_square_with_faer<T: MatrixLuciScalar + faer::traits::ComplexField>(
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
        let params = FullPivLuParams {
            max_rank: (cap < n).then_some(cap),
            ..<FullPivLuParams as Auto<T>>::auto()
        };
        let mut lu = faer::Mat::<T>::from_fn(n, n, |i, j| data[i + n * j]);
        let (mut row_perm, mut row_perm_inv) = (vec![0usize; n], vec![0usize; n]);
        let (mut col_perm, mut col_perm_inv) = (vec![0usize; n], vec![0usize; n]);
        let mut scratch = MemBuffer::new(faer_flu::lu_in_place_scratch::<usize, T>(
            n,
            n,
            Par::Seq.shape(),
            params.into(),
        ));
        let (info, _, _) = faer_flu::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            Par::Seq,
            MemStack::new(&mut scratch),
            params.into(),
        );
        let pivots = info.rank.min(n);

        let mut diag_abs = (0..pivots)
            .map(|i| lu[(i, i)].abs_val())
            .collect::<Vec<_>>();
        if pivots < n {
            // A truncated factorization leaves the trailing `(n - r) x (n - r)` block as the Schur
            // complement of the pivots taken, so its largest magnitude is the residual an
            // untruncated run meets next (its `U[r, r]` once complete pivoting has swapped that
            // entry into place). `compute_pivot_errors` reads it from `diag_abs[rank]`.
            let residual = (pivots..n)
                .flat_map(|j| (pivots..n).map(move |i| (i, j)))
                .map(|(i, j)| lu[(i, j)].abs_val())
                .fold(0.0f64, f64::max);
            diag_abs.push(residual);
        }
        if diag_abs.iter().any(|value| !value.is_finite()) {
            return Err(MatrixLuciError::NaNEncountered {
                matrix: "dense diagonal",
            });
        }

        let pivot_errors = Self::compute_pivot_errors::<T>(&diag_abs, n, n, options);
        let rank = pivot_errors.len().saturating_sub(1);
        Ok((
            PivotSelectionCore {
                row_indices: row_perm[..rank].to_vec(),
                col_indices: col_perm[..rank].to_vec(),
                pivot_errors,
                rank,
            },
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
                        DenseLuKernel::factorize_square_with_faer(data, nrows, options)
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
