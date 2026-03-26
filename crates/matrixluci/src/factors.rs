//! Optional factor reconstruction helpers.

use crate::error::MatrixLuciError;
use crate::scalar::Scalar;
use crate::source::CandidateMatrixSource;
use crate::types::{DenseOwnedMatrix, PivotSelectionCore};
use crate::Result;

/// Gather a dense column-major block from a source.
pub(crate) fn load_block<T: Scalar, S: CandidateMatrixSource<T>>(
    source: &S,
    rows: &[usize],
    cols: &[usize],
) -> DenseOwnedMatrix<T> {
    let mut data = vec![T::zero(); rows.len() * cols.len()];
    source.get_block(rows, cols, &mut data);
    DenseOwnedMatrix::from_column_major(data, rows.len(), cols.len())
}

/// Dense matrix product in column-major layout.
pub(crate) fn matmul<T: Scalar>(
    lhs: &DenseOwnedMatrix<T>,
    rhs: &DenseOwnedMatrix<T>,
) -> DenseOwnedMatrix<T> {
    assert_eq!(lhs.ncols(), rhs.nrows());
    let mut out = DenseOwnedMatrix::zeros(lhs.nrows(), rhs.ncols());
    for j in 0..rhs.ncols() {
        for k in 0..lhs.ncols() {
            let rhs_kj = rhs[[k, j]];
            for i in 0..lhs.nrows() {
                out[[i, j]] = out[[i, j]] + lhs[[i, k]] * rhs_kj;
            }
        }
    }
    out
}

/// Subtract one dense matrix from another in place.
pub(crate) fn subtract_inplace<T: Scalar>(
    lhs: &mut DenseOwnedMatrix<T>,
    rhs: &DenseOwnedMatrix<T>,
) {
    assert_eq!(lhs.nrows(), rhs.nrows());
    assert_eq!(lhs.ncols(), rhs.ncols());
    for j in 0..lhs.ncols() {
        for i in 0..lhs.nrows() {
            lhs[[i, j]] = lhs[[i, j]] - rhs[[i, j]];
        }
    }
}

fn swap_rows<T: Scalar>(matrix: &mut DenseOwnedMatrix<T>, a: usize, b: usize) {
    if a == b {
        return;
    }
    for col in 0..matrix.ncols() {
        let tmp = matrix[[a, col]];
        matrix[[a, col]] = matrix[[b, col]];
        matrix[[b, col]] = tmp;
    }
}

/// Invert a small square dense matrix with Gauss-Jordan elimination.
pub(crate) fn invert_square<T: Scalar>(
    matrix: &DenseOwnedMatrix<T>,
) -> Result<DenseOwnedMatrix<T>> {
    if matrix.nrows() != matrix.ncols() {
        return Err(MatrixLuciError::InvalidArgument {
            message: "pivot block must be square".to_string(),
        });
    }

    let n = matrix.nrows();
    let mut aug = DenseOwnedMatrix::zeros(n, 2 * n);
    for j in 0..n {
        for i in 0..n {
            aug[[i, j]] = matrix[[i, j]];
        }
        aug[[j, n + j]] = T::one();
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = 0.0f64;
        for row in k..n {
            let candidate = aug[[row, k]].abs_val();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if pivot_abs < T::epsilon() {
            return Err(MatrixLuciError::SingularPivotBlock);
        }

        swap_rows(&mut aug, k, pivot_row);

        let pivot = aug[[k, k]];
        for col in 0..(2 * n) {
            aug[[k, col]] = aug[[k, col]] / pivot;
        }

        for row in 0..n {
            if row == k {
                continue;
            }
            let factor = aug[[row, k]];
            if factor.abs_val() < T::epsilon() {
                continue;
            }
            for col in 0..(2 * n) {
                aug[[row, col]] = aug[[row, col]] - factor * aug[[k, col]];
            }
        }
    }

    let mut inv = DenseOwnedMatrix::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

/// Optional dense factors derived from a pivot selection.
#[derive(Debug, Clone)]
pub struct CrossFactors<T: Scalar> {
    /// Pivot block `A[I, J]`.
    pub pivot: DenseOwnedMatrix<T>,
    /// Columns through selected pivot columns `A[:, J]`.
    pub pivot_cols: DenseOwnedMatrix<T>,
    /// Rows through selected pivot rows `A[I, :]`.
    pub pivot_rows: DenseOwnedMatrix<T>,
}

impl<T: Scalar> CrossFactors<T> {
    /// Gather a dense block from a source.
    pub fn gather<S: CandidateMatrixSource<T>>(
        source: &S,
        rows: &[usize],
        cols: &[usize],
    ) -> DenseOwnedMatrix<T> {
        load_block(source, rows, cols)
    }

    /// Reconstruct factors from a source and pivot-only selection.
    pub fn from_source<S: CandidateMatrixSource<T>>(
        source: &S,
        selection: &PivotSelectionCore,
    ) -> Result<Self> {
        let all_rows: Vec<usize> = (0..source.nrows()).collect();
        let all_cols: Vec<usize> = (0..source.ncols()).collect();
        Ok(Self {
            pivot: load_block(source, &selection.row_indices, &selection.col_indices),
            pivot_cols: load_block(source, &all_rows, &selection.col_indices),
            pivot_rows: load_block(source, &selection.row_indices, &all_cols),
        })
    }

    /// Invert the pivot block.
    pub fn pivot_inverse(&self) -> Result<DenseOwnedMatrix<T>> {
        invert_square(&self.pivot)
    }

    /// Form `A[:, J] * A[I, J]^{-1}`.
    pub fn cols_times_pivot_inv(&self) -> Result<DenseOwnedMatrix<T>> {
        let pivot_inv = self.pivot_inverse()?;
        Ok(matmul(&self.pivot_cols, &pivot_inv))
    }

    /// Form `A[I, J]^{-1} * A[I, :]`.
    pub fn pivot_inv_times_rows(&self) -> Result<DenseOwnedMatrix<T>> {
        let pivot_inv = self.pivot_inverse()?;
        Ok(matmul(&pivot_inv, &self.pivot_rows))
    }
}

#[cfg(test)]
mod tests;
