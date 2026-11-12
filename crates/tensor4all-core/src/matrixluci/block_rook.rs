//! Lazy pivot-kernel implementations.

use crate::matrixlu::pivot_is_undividable;
use crate::matrixluci::error::MatrixLuciError;
use crate::matrixluci::factors::{load_block, subtract_inplace};
use crate::matrixluci::kernel::PivotKernel;
use crate::matrixluci::scalar::MatrixLuciScalar;
use crate::matrixluci::source::CandidateMatrixSource;
use crate::matrixluci::types::{PivotKernelOptions, PivotSelectionCore};
use crate::matrixluci::Result;
use num_complex::{Complex32, Complex64};
use tensor4all_tensorbackend::{mat_mul, solve_matrix, Matrix};

/// Lazy pivot kernel based on residual row/column rook search.
///
/// Selects pivots by computing residual blocks on demand, avoiding full
/// matrix materialization. Suitable for large matrices accessed via
/// [`LazyMatrixSource`](super::source::LazyMatrixSource).
/// An exactly zero starting residual column is skipped. Certifying an all-zero
/// residual may inspect every remaining entry, one column at a time.
///
/// A rook pivot is the largest entry of its own residual row and column only,
/// so its magnitude can be far below the maximum of the remaining residual.
/// Before the factorization stops (tolerance or undividable pivot), the
/// remaining residual block is therefore scanned exactly. If its maximum still
/// passes the stopping test the factorization stops; otherwise that maximum
/// becomes the next pivot and the rook search continues.
///
/// The trailing `pivot_errors` entry is the exact maximum magnitude of the
/// remaining residual block, which is the dense kernel's meaning, for every
/// way the factorization ends. Each scan costs one evaluation of every
/// unselected entry, again one column at a time.
#[derive(Default)]
pub struct LazyBlockRookKernel;

/// Subtract the eliminated part from `A[rows, cols]`.
///
/// `pivot` is `A[selected_rows, selected_cols]` and `a_rj` is
/// `A[rows, selected_cols]`; loading them once lets a multi-column scan share
/// them instead of re-reading the pivot operands per column block.
fn residual_from<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    rows: &[usize],
    cols: &[usize],
    selected_rows: &[usize],
    pivot: &Matrix<T>,
    a_rj: &Matrix<T>,
) -> Result<Matrix<T>> {
    let mut residual = load_block(source, rows, cols);
    if selected_rows.is_empty() {
        return Ok(residual);
    }

    let a_ic = load_block(source, selected_rows, cols);
    eliminate(&mut residual, pivot, a_rj, &a_ic)?;
    Ok(residual)
}

/// Subtract `a_rj * pivot^{-1} * a_ic` from `residual` in place.
fn eliminate<T: MatrixLuciScalar>(
    residual: &mut Matrix<T>,
    pivot: &Matrix<T>,
    a_rj: &Matrix<T>,
    a_ic: &Matrix<T>,
) -> Result<()> {
    let solved = solve_matrix(pivot, a_ic).map_err(|err| MatrixLuciError::InvalidArgument {
        message: format!("residual pivot solve failed: {err}"),
    })?;
    let approx = mat_mul(a_rj, &solved).map_err(|err| MatrixLuciError::InvalidArgument {
        message: format!("residual multiplication failed: {err}"),
    })?;
    subtract_inplace(residual, &approx);
    Ok(())
}

/// Copy column `col` of `matrix` into an `nrows x 1` matrix.
fn column_of<T: MatrixLuciScalar>(matrix: &Matrix<T>, col: usize) -> Matrix<T> {
    let data = (0..matrix.nrows()).map(|row| matrix[[row, col]]).collect();
    Matrix::from_col_major_vec(matrix.nrows(), 1, data)
}

/// Copy row `row` of `matrix` into a `1 x ncols` matrix.
fn row_of<T: MatrixLuciScalar>(matrix: &Matrix<T>, row: usize) -> Matrix<T> {
    let data = (0..matrix.ncols()).map(|col| matrix[[row, col]]).collect();
    Matrix::from_col_major_vec(1, matrix.ncols(), data)
}

/// The blocks of `A` that stay fixed during one rook walk.
///
/// Within a walk only the current row and column move; the selected and
/// remaining index sets do not. The pivot block `A[I, J]`, the pivot columns
/// `A[remaining_rows, J]` and the pivot rows `A[I, remaining_cols]` are
/// therefore read once per walk, and each column or row block takes its
/// operands from them. The operands passed to the solve and the product have
/// the same shapes and values as a fresh load, so the residuals are bitwise
/// those of a walk that reloads every block.
struct WalkOperands<T: MatrixLuciScalar> {
    /// `A[I, J]`.
    pivot: Matrix<T>,
    /// `A[remaining_rows, J]`.
    pivot_cols: Matrix<T>,
    /// `A[I, remaining_cols]`.
    pivot_rows: Matrix<T>,
}

impl<T: MatrixLuciScalar> WalkOperands<T> {
    /// Load the walk invariants; `None` before the first pivot, when the
    /// residual is the matrix itself.
    fn load<S: CandidateMatrixSource<T>>(
        source: &S,
        remaining_rows: &[usize],
        remaining_cols: &[usize],
        selected_rows: &[usize],
        selected_cols: &[usize],
    ) -> Option<Self> {
        if selected_rows.is_empty() {
            return None;
        }
        Some(Self {
            pivot: load_block(source, selected_rows, selected_cols),
            pivot_cols: load_block(source, remaining_rows, selected_cols),
            pivot_rows: load_block(source, selected_rows, remaining_cols),
        })
    }
}

/// Residual column `A[remaining_rows, remaining_cols[col_pos]]`.
fn residual_column<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
    col_pos: usize,
    operands: Option<&WalkOperands<T>>,
) -> Result<Matrix<T>> {
    let mut residual = load_block(source, remaining_rows, &[remaining_cols[col_pos]]);
    if let Some(ops) = operands {
        let a_ic = column_of(&ops.pivot_rows, col_pos);
        eliminate(&mut residual, &ops.pivot, &ops.pivot_cols, &a_ic)?;
    }
    Ok(residual)
}

/// Residual row `A[remaining_rows[row_pos], remaining_cols]`.
fn residual_row<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
    row_pos: usize,
    operands: Option<&WalkOperands<T>>,
) -> Result<Matrix<T>> {
    let mut residual = load_block(source, &[remaining_rows[row_pos]], remaining_cols);
    if let Some(ops) = operands {
        let a_rj = row_of(&ops.pivot_cols, row_pos);
        eliminate(&mut residual, &ops.pivot, &a_rj, &ops.pivot_rows)?;
    }
    Ok(residual)
}

fn argmax_abs<T: MatrixLuciScalar>(matrix: &Matrix<T>) -> (usize, usize, f64) {
    let mut best_row = 0usize;
    let mut best_col = 0usize;
    let mut best_abs = -1.0f64;
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            let value = matrix[[row, col]].abs_val();
            if !value.is_finite() {
                return (row, col, value);
            }
            if value > best_abs {
                best_row = row;
                best_col = col;
                best_abs = value;
            }
        }
    }
    (best_row, best_col, best_abs.max(0.0))
}

fn remaining_indices(total: usize, selected: &[usize]) -> Vec<usize> {
    let mut used = vec![false; total];
    for &idx in selected {
        used[idx] = true;
    }
    (0..total).filter(|&idx| !used[idx]).collect()
}

fn rook_pivot<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
    selected_rows: &[usize],
    selected_cols: &[usize],
) -> Result<(usize, usize, f64)> {
    let operands = WalkOperands::load(
        source,
        remaining_rows,
        remaining_cols,
        selected_rows,
        selected_cols,
    );
    let operands = operands.as_ref();
    let mut current_col_pos;
    let mut current_row_pos;
    // INVARIANT: a zero fiber does not certify a zero residual. Scan lazily
    // until a nonzero starting column is found; never materialize the full
    // remaining matrix. The ordinary nonzero-column path performs no extra
    // block evaluations.
    let mut columns = 0..remaining_cols.len();
    loop {
        let Some(col_pos) = columns.next() else {
            return Ok((remaining_rows[0], remaining_cols[0], 0.0));
        };
        current_col_pos = col_pos;
        let col_residual =
            residual_column(source, remaining_rows, remaining_cols, col_pos, operands)?;
        let (best_row_pos, _, best_abs) = argmax_abs(&col_residual);
        // Skip only an exactly zero fiber. Non-finite magnitudes must reach
        // the kernel's existing numerical-error check, not look like zeros.
        if best_abs != 0.0 {
            current_row_pos = best_row_pos;
            break;
        }
    }

    let max_steps = remaining_rows.len() + remaining_cols.len() + 1;
    for _ in 0..max_steps {
        let row_residual = residual_row(
            source,
            remaining_rows,
            remaining_cols,
            current_row_pos,
            operands,
        )?;
        let (_, best_col_pos, best_abs) = argmax_abs(&row_residual);

        if best_col_pos == current_col_pos {
            return Ok((
                remaining_rows[current_row_pos],
                remaining_cols[current_col_pos],
                best_abs,
            ));
        }
        current_col_pos = best_col_pos;
        let col_residual = residual_column(
            source,
            remaining_rows,
            remaining_cols,
            current_col_pos,
            operands,
        )?;
        let (best_row_pos, _, _) = argmax_abs(&col_residual);
        current_row_pos = best_row_pos;
    }

    let row_residual = residual_row(
        source,
        remaining_rows,
        remaining_cols,
        current_row_pos,
        operands,
    )?;
    let (_, best_col_pos, best_abs) = argmax_abs(&row_residual);
    Ok((
        remaining_rows[current_row_pos],
        remaining_cols[best_col_pos],
        best_abs,
    ))
}

/// Exact largest magnitude of the residual block over the unselected rows and
/// columns.
///
/// The block is evaluated in column chunks of one pivot width, so it is never
/// materialized in one piece. An exact maximum over the block costs one
/// evaluation of every remaining entry; the pivot operands are loaded once.
fn residual_max_abs<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
    selected_rows: &[usize],
    selected_cols: &[usize],
) -> Result<f64> {
    residual_argmax_abs(
        source,
        remaining_rows,
        remaining_cols,
        selected_rows,
        selected_cols,
    )
    .map(|(_, _, value)| value)
}

/// Location (as source row and column indices) and exact magnitude of the
/// largest entry of the residual block over the unselected rows and columns.
///
/// Returns [`MatrixLuciError::NaNEncountered`] if any residual entry is not
/// finite, so a NaN can never be mistaken for a small residual. For an
/// all-zero block the location is the first remaining entry and the
/// magnitude is `0.0`.
fn residual_argmax_abs<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
    selected_rows: &[usize],
    selected_cols: &[usize],
) -> Result<(usize, usize, f64)> {
    let (pivot, a_rj) = if selected_rows.is_empty() {
        (Matrix::zeros(0, 0), Matrix::zeros(remaining_rows.len(), 0))
    } else {
        (
            load_block(source, selected_rows, selected_cols),
            load_block(source, remaining_rows, selected_cols),
        )
    };
    let chunk_width = selected_rows.len().max(1);
    let mut best = (0usize, 0usize, 0.0f64);
    for (chunk_idx, chunk) in remaining_cols.chunks(chunk_width).enumerate() {
        let residual = residual_from(source, remaining_rows, chunk, selected_rows, &pivot, &a_rj)?;
        let (row_pos, col_pos, value) = argmax_abs(&residual);
        if !value.is_finite() {
            return Err(MatrixLuciError::NaNEncountered {
                matrix: "rook residual",
            });
        }
        if value > best.2 {
            best = (row_pos, chunk_idx * chunk_width + col_pos, value);
        }
    }
    Ok((remaining_rows[best.0], remaining_cols[best.1], best.2))
}

fn factorize_lazy<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    options: &PivotKernelOptions,
) -> Result<PivotSelectionCore> {
    let nrows = source.nrows();
    let ncols = source.ncols();
    let full_rank = nrows.min(ncols);
    if full_rank == 0 {
        return Ok(PivotSelectionCore {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            pivot_errors: vec![0.0],
            rank: 0,
        });
    }

    let max_bond_dim = options.max_bond_dim.min(full_rank);
    let mut selected_rows = Vec::with_capacity(max_bond_dim);
    let mut selected_cols = Vec::with_capacity(max_bond_dim);
    let mut accepted = Vec::with_capacity(max_bond_dim + 1);
    let mut max_error = 0.0f64;
    let mut last_error = f64::NAN;

    while selected_rows.len() < max_bond_dim {
        let remaining_rows = remaining_indices(nrows, &selected_rows);
        let remaining_cols = remaining_indices(ncols, &selected_cols);
        if remaining_rows.is_empty() || remaining_cols.is_empty() {
            break;
        }

        let (mut pivot_row, mut pivot_col, mut pivot_abs) = rook_pivot(
            source,
            &remaining_rows,
            &remaining_cols,
            &selected_rows,
            &selected_cols,
        )?;
        // Checked before the stopping test: NaN compares false against every
        // threshold and must not be read as a converged residual.
        if !pivot_abs.is_finite() {
            return Err(MatrixLuciError::NaNEncountered {
                matrix: "rook residual",
            });
        }

        let stops = |value: f64, n_selected: usize| {
            (n_selected > 0 && (value < options.rel_tol * max_error || value < options.abs_tol))
                || pivot_is_undividable::<T>(value)
        };
        if pivot_abs == 0.0 {
            // `rook_pivot` returns an exact zero only after it has found
            // every remaining column to be zero, which already certifies it.
            last_error = 0.0;
            break;
        }
        if stops(pivot_abs, selected_rows.len()) {
            // INVARIANT: the factorization only ends on the exact maximum of
            // the remaining residual. A rook pivot is a local maximum of its
            // own row and column and is not a bound for the rest of the block
            // (tensor4all-rs#1256).
            let (row, col, value) = residual_argmax_abs(
                source,
                &remaining_rows,
                &remaining_cols,
                &selected_rows,
                &selected_cols,
            )?;
            if stops(value, selected_rows.len()) {
                last_error = value;
                break;
            }
            (pivot_row, pivot_col, pivot_abs) = (row, col, value);
        }

        max_error = max_error.max(pivot_abs);
        selected_rows.push(pivot_row);
        selected_cols.push(pivot_col);
        accepted.push(pivot_abs);
    }

    let rank = selected_rows.len();
    if rank >= full_rank {
        last_error = 0.0;
    } else if rank == max_bond_dim {
        // A rank ceiling reports the maximum of the remaining residual, which
        // is the dense kernel's meaning, not the last accepted pivot.
        last_error = residual_max_abs(
            source,
            &remaining_indices(nrows, &selected_rows),
            &remaining_indices(ncols, &selected_cols),
            &selected_rows,
            &selected_cols,
        )?;
    }
    accepted.push(last_error);

    Ok(PivotSelectionCore {
        row_indices: selected_rows,
        col_indices: selected_cols,
        pivot_errors: accepted,
        rank,
    })
}

macro_rules! impl_lazy_block_rook_kernel {
    ($t:ty) => {
        impl PivotKernel<$t> for LazyBlockRookKernel {
            fn factorize<S: CandidateMatrixSource<$t>>(
                &self,
                source: &S,
                options: &PivotKernelOptions,
            ) -> Result<PivotSelectionCore> {
                factorize_lazy(source, options)
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
