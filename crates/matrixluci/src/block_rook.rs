//! Lazy pivot-kernel implementations.

use crate::factors::{invert_square, load_block, matmul, subtract_inplace};
use crate::kernel::PivotKernel;
use crate::scalar::Scalar;
use crate::source::CandidateMatrixSource;
use crate::types::{DenseOwnedMatrix, PivotKernelOptions, PivotSelectionCore};
use crate::Result;
use num_complex::{Complex32, Complex64};

/// Lazy pivot kernel based on residual row/column rook search.
#[derive(Default)]
pub struct LazyBlockRookKernel;

fn residual_block<T: Scalar, S: CandidateMatrixSource<T>>(
    source: &S,
    rows: &[usize],
    cols: &[usize],
    selected_rows: &[usize],
    selected_cols: &[usize],
    pivot_inv: Option<&DenseOwnedMatrix<T>>,
) -> DenseOwnedMatrix<T> {
    let mut residual = load_block(source, rows, cols);
    if selected_rows.is_empty() {
        return residual;
    }

    let a_rj = load_block(source, rows, selected_cols);
    let a_ic = load_block(source, selected_rows, cols);
    let temp = matmul(&a_rj, pivot_inv.unwrap());
    let approx = matmul(&temp, &a_ic);
    subtract_inplace(&mut residual, &approx);
    residual
}

fn argmax_abs<T: Scalar>(matrix: &DenseOwnedMatrix<T>) -> (usize, usize, f64) {
    let mut best_row = 0usize;
    let mut best_col = 0usize;
    let mut best_abs = -1.0f64;
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            let value = matrix[[row, col]].abs_val();
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

fn rook_pivot<T: Scalar, S: CandidateMatrixSource<T>>(
    source: &S,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
    selected_rows: &[usize],
    selected_cols: &[usize],
    pivot_inv: Option<&DenseOwnedMatrix<T>>,
) -> (usize, usize, f64) {
    let mut current_col = remaining_cols[0];
    let mut current_row = remaining_rows[0];
    let max_steps = remaining_rows.len() + remaining_cols.len() + 1;

    for _ in 0..max_steps {
        let col_residual = residual_block(
            source,
            remaining_rows,
            &[current_col],
            selected_rows,
            selected_cols,
            pivot_inv,
        );
        let (best_row_pos, _, _) = argmax_abs(&col_residual);
        current_row = remaining_rows[best_row_pos];

        let row_residual = residual_block(
            source,
            &[current_row],
            remaining_cols,
            selected_rows,
            selected_cols,
            pivot_inv,
        );
        let (_, best_col_pos, best_abs) = argmax_abs(&row_residual);
        let next_col = remaining_cols[best_col_pos];

        if next_col == current_col {
            return (current_row, current_col, best_abs);
        }
        current_col = next_col;
    }

    let row_residual = residual_block(
        source,
        &[current_row],
        remaining_cols,
        selected_rows,
        selected_cols,
        pivot_inv,
    );
    let (_, best_col_pos, best_abs) = argmax_abs(&row_residual);
    (current_row, remaining_cols[best_col_pos], best_abs)
}

fn factorize_lazy<T: Scalar, S: CandidateMatrixSource<T>>(
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

    let max_rank = options.max_rank.min(full_rank);
    let mut selected_rows = Vec::with_capacity(max_rank);
    let mut selected_cols = Vec::with_capacity(max_rank);
    let mut accepted = Vec::with_capacity(max_rank + 1);
    let mut max_error = 0.0f64;
    let mut last_error = f64::NAN;

    while selected_rows.len() < max_rank {
        let remaining_rows = remaining_indices(nrows, &selected_rows);
        let remaining_cols = remaining_indices(ncols, &selected_cols);
        if remaining_rows.is_empty() || remaining_cols.is_empty() {
            break;
        }

        let pivot_inv = if selected_rows.is_empty() {
            None
        } else {
            let pivot = load_block(source, &selected_rows, &selected_cols);
            Some(invert_square(&pivot)?)
        };

        let (pivot_row, pivot_col, pivot_abs) = rook_pivot(
            source,
            &remaining_rows,
            &remaining_cols,
            &selected_rows,
            &selected_cols,
            pivot_inv.as_ref(),
        );
        last_error = pivot_abs;

        if !selected_rows.is_empty()
            && (pivot_abs < options.rel_tol * max_error || pivot_abs < options.abs_tol)
        {
            break;
        }

        if pivot_abs < T::epsilon() {
            if selected_rows.is_empty() {
                last_error = pivot_abs;
            }
            break;
        }

        max_error = max_error.max(pivot_abs);
        selected_rows.push(pivot_row);
        selected_cols.push(pivot_col);
        accepted.push(pivot_abs);
    }

    let rank = selected_rows.len();
    if rank >= full_rank {
        last_error = 0.0;
    } else if rank == max_rank && rank > 0 {
        last_error = accepted[rank - 1];
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
