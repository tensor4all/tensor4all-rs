//! Accuracy-checked retention of preferred pivots within a fixed-rank cross.

use super::{map_backend_error, matrix_luci_factors_from_pivots, MatrixLuciFactors};
use crate::{
    matrixluci::factors::CrossFactors, matrixluci::source::DenseMatrixSource,
    matrixluci::types::PivotSelectionCore, MatrixCIError, MatrixLuciScalar, Result, Scalar,
};
use tensor4all_tensorbackend::{mat_mul, Matrix};

/// Restore preferred rows and columns in a dense cross without increasing its rank.
///
/// `a` is column-major; `seed_rows` and `seed_cols` define a nonempty,
/// nonsingular cross of equal length. `old_rows` and `old_cols` are independent
/// ordered preferences: either can be empty or contain only surviving samples.
/// Every selection must contain distinct in-bounds indices. `tolerance` is a
/// finite nonnegative maximum absolute entrywise error, in `a`'s units.
/// `left_orthogonal` chooses which factor interpolates its selected axis.
///
/// The seed is reconstructed and its full residual is measured first. If it
/// exceeds `tolerance`, it is returned unchanged. Otherwise, single-pivot
/// replacements increase overlap with the preferences while preserving the
/// seed rank. A rank-one residual prediction screens trials; admissible
/// positions are tried in descending absolute determinant ratio, with residual
/// and index tie breaks. Each accepted trial is independently reconstructed
/// and checked over the full
/// matrix. Selected preferred pivots are never displaced. Columns are tried
/// before rows, and passes repeat until no further replacement is accepted.
/// The overlap increases strictly, so at most twice the seed rank replacements
/// can be accepted. This is a local search, not a minimum-distance or globally
/// optimal cross guarantee. The returned `pivot_errors` contains one measured
/// full residual, rather than a pivot spectrum.
///
/// Storage is O(rows(a)*cols(a) + (rows(a)+cols(a))*rank). This optional search
/// can cost more than ordinary LUCI; callers should first try an admissible
/// complete previous cross and reserve conservative reconstruction scratch.
///
/// # Errors
///
/// Returns typed errors for invalid matrix storage, non-finite values or
/// tolerance, negative tolerance, duplicate/out-of-bounds preferences,
/// malformed or singular seed selections, or failed backend linear algebra.
/// Numerically singular replacement trials are skipped.
///
/// # Examples
///
/// ```
/// use tensor4all_core::matrix_luci_factors_with_preferred_pivots;
/// use tensor4all_tensorbackend::from_vec2d;
/// let a = from_vec2d(vec![
///     vec![1.0_f64, 2.0, 3.0],
///     vec![2.0, 4.01, 6.0],
///     vec![3.0, 6.0, 9.0],
/// ]);
/// let factors = matrix_luci_factors_with_preferred_pivots(
///     &a, &[2], &[2], &[1], &[1], true, 0.02,
/// ).unwrap();
/// assert_eq!(factors.rank, 1);
/// assert_eq!(factors.row_indices, vec![2]);
/// assert_eq!(factors.col_indices, vec![1]);
/// assert!((factors.pivot_errors[0] - 0.015).abs() < 1e-12);
/// ```
pub fn matrix_luci_factors_with_preferred_pivots<T>(
    a: &Matrix<T>,
    seed_rows: &[usize],
    seed_cols: &[usize],
    old_rows: &[usize],
    old_cols: &[usize],
    left_orthogonal: bool,
    tolerance: f64,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + MatrixLuciScalar,
{
    crate::matrixlu::validate_col_major_matrix_len(
        a.nrows(),
        a.ncols(),
        a.as_col_major_slice().len(),
    )?;
    super::selected::validate_axes(a.nrows(), a.ncols(), old_rows, old_cols)?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(MatrixCIError::InvalidArgument {
            message: "preferred-pivot maximum error must be finite and nonnegative".into(),
        });
    }
    if seed_rows.is_empty() || seed_cols.is_empty() || seed_rows.len() != seed_cols.len() {
        return Err(MatrixCIError::InvalidArgument {
            message: "preferred-pivot search requires equally sized nonempty seed axes".into(),
        });
    }
    let mut current = matrix_luci_factors_from_pivots(a, seed_rows, seed_cols, left_orthogonal)?;
    if current.pivot_errors[0] > tolerance || (old_rows.is_empty() && old_cols.is_empty()) {
        return Ok(current);
    }
    let source = DenseMatrixSource::from_column_major(a.as_col_major_slice(), a.nrows(), a.ncols());
    let mut error = Matrix::zeros(a.nrows(), a.ncols());
    refresh_residual(a, &current, &mut error)?;
    loop {
        let mut changed = false;
        for row_axis in [false, true] {
            let old = if row_axis { old_rows } else { old_cols };
            // Rejected candidates leave the cross unchanged. Reuse its
            // solved coefficients until an accepted replacement invalidates
            // them, rather than gathering and solving the same block again.
            let mut cached_coefficients = None;
            for &candidate in old {
                let selected = if row_axis {
                    &current.row_indices
                } else {
                    &current.col_indices
                };
                if selected.contains(&candidate) {
                    continue;
                }
                let coefficients = match cached_coefficients.take() {
                    Some(coefficients) => coefficients,
                    None => {
                        let selection = PivotSelectionCore {
                            row_indices: current.row_indices.clone(),
                            col_indices: current.col_indices.clone(),
                            rank: current.rank,
                            pivot_errors: Vec::new(),
                        };
                        let cross = CrossFactors::from_source(&source, &selection)
                            .map_err(map_backend_error)?;
                        if row_axis {
                            cross.cols_solve_pivot()
                        } else {
                            cross.solve_pivot_rows()
                        }
                        .map_err(map_backend_error)?
                    }
                };
                let mut trials = Vec::new();
                for position in 0..current.rank {
                    if old.contains(&selected[position]) {
                        continue;
                    }
                    let denominator = if row_axis {
                        coefficients[[candidate, position]]
                    } else {
                        coefficients[[position, candidate]]
                    };
                    let magnitude = Scalar::abs_val(denominator);
                    if magnitude == 0.0 || !magnitude.is_finite() {
                        continue;
                    }
                    let weights = (0..if row_axis { a.ncols() } else { a.nrows() })
                        .map(|i| {
                            (if row_axis {
                                error[[candidate, i]]
                            } else {
                                error[[i, candidate]]
                            }) / denominator
                        })
                        .collect::<Vec<_>>();
                    let mut predicted = 0.0_f64;
                    'scan: for col in 0..a.ncols() {
                        for row in 0..a.nrows() {
                            let correction = if row_axis {
                                coefficients[[row, position]] * weights[col]
                            } else {
                                weights[row] * coefficients[[position, col]]
                            };
                            let value = Scalar::abs_val(error[[row, col]] - correction);
                            if !value.is_finite() {
                                predicted = f64::INFINITY;
                                break 'scan;
                            }
                            predicted = predicted.max(value);
                            if predicted > tolerance {
                                break 'scan;
                            }
                        }
                    }
                    if predicted <= tolerance {
                        trials.push((position, predicted, magnitude));
                    }
                }
                trials.sort_by(|a, b| {
                    b.2.total_cmp(&a.2)
                        .then(a.1.total_cmp(&b.1))
                        .then(a.0.cmp(&b.0))
                });
                let mut accepted = false;
                for (position, _, _) in trials {
                    let mut rows = current.row_indices.clone();
                    let mut cols = current.col_indices.clone();
                    if row_axis {
                        rows[position] = candidate;
                    } else {
                        cols[position] = candidate;
                    }
                    match matrix_luci_factors_from_pivots(a, &rows, &cols, left_orthogonal) {
                        Ok(next) if next.pivot_errors[0] <= tolerance => {
                            current = next;
                            refresh_residual(a, &current, &mut error)?;
                            accepted = true;
                            changed = true;
                            break;
                        }
                        Ok(_) | Err(MatrixCIError::SingularMatrix) => {}
                        Err(e) => return Err(e),
                    }
                }
                cached_coefficients = if accepted { None } else { Some(coefficients) };
            }
        }
        if !changed {
            break;
        }
    }
    Ok(current)
}

fn refresh_residual<T: Scalar + MatrixLuciScalar>(
    a: &Matrix<T>,
    factors: &MatrixLuciFactors<T>,
    residual: &mut Matrix<T>,
) -> Result<()> {
    let approximation = mat_mul(&factors.left, &factors.right)
        .map_err(|e| super::backend_linalg_error(e.to_string()))?;
    for ((error, &expected), &actual) in residual
        .as_col_major_mut_slice()
        .iter_mut()
        .zip(a.as_col_major_slice())
        .zip(approximation.as_col_major_slice())
    {
        *error = expected - actual;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
