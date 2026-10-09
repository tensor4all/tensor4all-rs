//! Reconstruction and dense residual validation of a prescribed cross.

use super::{factors_to_public, map_backend_error, MatrixLuciFactors};
use crate::{
    matrixluci::factors::CrossFactors, matrixluci::source::DenseMatrixSource,
    matrixluci::types::PivotSelectionCore, MatrixCIError, MatrixLuciScalar, Result, RrLUOptions,
    Scalar,
};
use tensor4all_tensorbackend::{mat_mul, Matrix};

/// Reconstruct and validate a dense cross with prescribed pivot rows or columns.
///
/// `a` is column-major. Each supplied selection contains distinct in-bounds
/// indices. Supply both selections with equal nonzero lengths to fix the
/// complete cross, or leave exactly one empty to select that axis with LUCI
/// from the prescribed opposite axis. The resulting pivot block must be
/// numerically nonsingular. With `left_orthogonal`, the left factor interpolates
/// pivot rows; otherwise the right factor interpolates pivot columns.
/// The returned `pivot_errors` contains one value: the measured
/// maximum absolute entry of `a - left * right`, in the input's units.
/// Callers can use this value to validate a previous selection against their
/// unchanged tolerance. Storage is O(rows(a) * cols(a) + (rows(a) + cols(a)) * rank).
///
/// # Errors
///
/// Returns typed errors for invalid shapes, both-empty/mismatched/duplicate/out-of-
/// bounds selections, non-finite values, a singular pivot block, or a failed
/// backend solve. Singularity is checked after scaling the pivot block to unit
/// maximum magnitude, using rrLU's existing numerical pivot floor.
///
/// # Examples
///
/// ```
/// use tensor4all_core::matrix_luci_factors_from_pivots;
/// use tensor4all_tensorbackend::{from_vec2d, mat_mul};
/// let a = from_vec2d(vec![vec![1.0_f64, 2.0, 3.0], vec![2.0, 4.0, 6.0]]);
/// let cross = matrix_luci_factors_from_pivots(&a, &[0], &[1], true).unwrap();
/// assert_eq!(cross.row_indices, vec![0]);
/// assert!(cross.pivot_errors[0] < 1e-12);
/// let reconstructed = mat_mul(&cross.left, &cross.right).unwrap();
/// for (&expected, &actual) in a.as_col_major_slice().iter()
///     .zip(reconstructed.as_col_major_slice()) {
///     assert!((expected - actual).abs() < 1e-12);
/// }
/// let completed = matrix_luci_factors_from_pivots(&a, &[], &[1], true).unwrap();
/// assert_eq!(completed.col_indices, vec![1]);
/// assert_eq!(completed.row_indices, vec![1]);
/// assert!(completed.pivot_errors[0] < 1e-12);
/// ```
pub fn matrix_luci_factors_from_pivots<T>(
    a: &Matrix<T>,
    rows: &[usize],
    cols: &[usize],
    left_orthogonal: bool,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + MatrixLuciScalar,
{
    crate::matrixlu::validate_col_major_matrix_len(
        a.nrows(),
        a.ncols(),
        a.as_col_major_slice().len(),
    )?;
    if (rows.is_empty() && cols.is_empty())
        || (!rows.is_empty() && !cols.is_empty() && rows.len() != cols.len())
    {
        return Err(MatrixCIError::InvalidArgument {
            message: "fixed cross requires one nonempty axis or equally sized pivot selections"
                .into(),
        });
    }
    validate_axes(a.nrows(), a.ncols(), rows, cols)?;
    if a.as_col_major_slice()
        .iter()
        .any(|&x| !Scalar::abs_val(x).is_finite())
    {
        return Err(MatrixCIError::NaNEncountered {
            matrix: "fixed-cross input".into(),
        });
    }
    let rank = rows.len().max(cols.len());
    if rank > a.nrows().min(a.ncols()) {
        return Err(MatrixCIError::SingularMatrix);
    }
    if super::matrix_luci_factors_working_bytes::<T>(a.nrows(), a.ncols(), rank).is_none() {
        return Err(MatrixCIError::InvalidArgument {
            message: "fixed-cross working size overflow".into(),
        });
    }
    // Completing one axis preserves the candidate matrix and its tensor-leg
    // dimensions. Extending that matrix with old samples would break nested
    // CI gauges on adjacent edges. Release the restricted LUCI buffers before
    // reconstructing and measuring the complete cross.
    let completed;
    let (rows, cols) = if rows.is_empty() || cols.is_empty() {
        let rows_fixed = !rows.is_empty();
        let mut restricted = if rows_fixed {
            Matrix::zeros(rank, a.ncols())
        } else {
            Matrix::zeros(a.nrows(), rank)
        };
        for col in 0..restricted.ncols() {
            for row in 0..restricted.nrows() {
                restricted[[row, col]] = if rows_fixed {
                    a[[rows[row], col]]
                } else {
                    a[[row, cols[col]]]
                };
            }
        }
        let scale = restricted
            .as_col_major_slice()
            .iter()
            .copied()
            .map(Scalar::abs_val)
            .fold(0.0_f64, f64::max);
        if scale == 0.0 {
            return Err(MatrixCIError::SingularMatrix);
        }
        for value in restricted.as_col_major_mut_slice() {
            *value = Scalar::div_real(*value, scale);
        }
        let selected = super::matrix_luci_factors_from_matrix_owned(
            restricted,
            Some(RrLUOptions {
                rel_tol: 0.0,
                abs_tol: 0.0,
                max_bond_dim: rank,
                left_orthogonal: true,
            }),
        )?;
        if selected.rank != rank {
            return Err(MatrixCIError::SingularMatrix);
        }
        completed = if rows_fixed {
            selected.col_indices
        } else {
            selected.row_indices
        };
        if rows_fixed {
            (rows, completed.as_slice())
        } else {
            (completed.as_slice(), cols)
        }
    } else {
        (rows, cols)
    };
    let selection = PivotSelectionCore {
        row_indices: rows.to_vec(),
        col_indices: cols.to_vec(),
        rank: rows.len(),
        pivot_errors: Vec::new(),
    };
    let source = DenseMatrixSource::from_column_major(a.as_col_major_slice(), a.nrows(), a.ncols());
    let factors = CrossFactors::from_source(&source, &selection).map_err(map_backend_error)?;
    let scale = factors
        .pivot
        .as_col_major_slice()
        .iter()
        .copied()
        .map(Scalar::abs_val)
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(MatrixCIError::SingularMatrix);
    }
    let mut pivot = factors.pivot.clone();
    for x in pivot.as_col_major_mut_slice() {
        *x = Scalar::div_real(*x, scale);
    }
    let lu = crate::rrlu_mut(
        &mut pivot,
        Some(RrLUOptions {
            rel_tol: 0.0,
            abs_tol: 0.0,
            max_bond_dim: rows.len(),
            left_orthogonal: true,
        }),
    )?;
    if lu.npivots() != rows.len() {
        return Err(MatrixCIError::SingularMatrix);
    }
    drop(lu);
    drop(pivot);
    let mut result = factors_to_public(selection, factors, left_orthogonal)?;
    // These entries are mathematically the identity. Preserve that exact
    // interpolation invariant instead of retaining solve round-off on sample
    // axes; the full residual below validates the resulting factors.
    for pivot in 0..rows.len() {
        for bond in 0..rows.len() {
            let identity = <T as Scalar>::from_f64(if pivot == bond { 1.0 } else { 0.0 });
            if left_orthogonal {
                result.left[[rows[pivot], bond]] = identity;
            } else {
                result.right[[bond, cols[pivot]]] = identity;
            }
        }
    }
    let reconstructed = mat_mul(&result.left, &result.right)
        .map_err(|error| super::backend_linalg_error(error.to_string()))?;
    let mut residual = 0.0_f64;
    for (&expected, &actual) in a
        .as_col_major_slice()
        .iter()
        .zip(reconstructed.as_col_major_slice())
    {
        let error = Scalar::abs_val(expected - actual);
        if !error.is_finite() {
            return Err(MatrixCIError::NaNEncountered {
                matrix: "fixed-cross residual".into(),
            });
        }
        residual = residual.max(error);
    }
    result.pivot_errors.push(residual);
    Ok(result)
}

#[cfg(test)]
mod tests;

pub(super) fn validate_axes(
    nrows: usize,
    ncols: usize,
    rows: &[usize],
    cols: &[usize],
) -> Result<()> {
    for (position, &row) in rows.iter().enumerate() {
        if row >= nrows {
            return Err(MatrixCIError::IndexOutOfBounds {
                row,
                col: 0,
                nrows,
                ncols,
            });
        }
        if rows[..position].contains(&row) {
            return Err(MatrixCIError::DuplicatePivotRow { row });
        }
    }
    for (position, &col) in cols.iter().enumerate() {
        if col >= ncols {
            return Err(MatrixCIError::IndexOutOfBounds {
                row: 0,
                col,
                nrows,
                ncols,
            });
        }
        if cols[..position].contains(&col) {
            return Err(MatrixCIError::DuplicatePivotCol { col });
        }
    }
    Ok(())
}
