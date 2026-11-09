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
    /// Returns the selection together with the number of pivots the rank cap accepted, which is the
    /// evidence that a cap stops the elimination instead of factoring to full rank and truncating
    /// afterwards.
    ///
    /// The rank cap is the only bound handed to faer, one pivot past the cap, so that the residual
    /// [`Self::compute_pivot_errors`] reads from `diag_abs[rank]` is the pivot faer itself would take
    /// next rather than a rule re-derived here. The tolerance decision stays in
    /// [`Self::compute_pivot_errors`] and runs on the pivot magnitudes this factorization produced,
    /// because faer rejects a pivot at *or below* its tolerance where this kernel rejects only a
    /// strictly smaller one (`|pivot| < tol`). The call itself is the factorization the previous path
    /// performed through `full_piv_lu_matrix_owned` in the same lock; the parity test compares those
    /// two consumers of faer, which is what it can prove - the cross-version claim (0.28.2 against
    /// 0.33.0) was measured when the lock still held 0.28.2 and cannot be re-derived from this one.
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
        // One pivot past the cap is what `compute_pivot_errors` reads as the residual
        // (`diag_abs[rank]`), so the value comes from faer's own selection instead of a guess about
        // its criterion: the search maximises `abs1` (`|re| + |im|` for complex), and for a tiny
        // trailing block it does not answer the block's largest modulus either - `diag(10, 3,
        // 2.5 + 2.5i)` capped at 1 picks 3, not the 3.53 complex entry nor its larger `abs1`. One
        // extra elimination step costs `O((n - cap)^2)`.
        let ask = if cap < n { cap + 1 } else { cap };
        let params = FullPivLuParams {
            // A `max_rank` at or above `n` still asks faer to reveal the rank it stopped at, which is
            // how the subnormal stop below becomes visible instead of being reported as `n`.
            max_rank: (cap < n).then_some(ask),
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
        if pivots < ask {
            // faer stopped for its own reason - a zero or subnormal Schur complement - before it
            // could reveal the extra pivot, and that stop predates the rank cap: the untruncated
            // run reports `rank = n`, leaves the trailing block un-eliminated, and reading its
            // diagonal is what the previous path did. Reproducing that diagonal keeps the
            // subnormal contract, which the residual shortcut above would change.
            diag_abs.extend((pivots..n).map(|i| lu[(i, i)].abs_val()));
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
            // The pivots the rank cap accepted, not the extra one asked for above.
            pivots.min(cap),
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
