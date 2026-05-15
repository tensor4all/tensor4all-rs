//! Cross-factor reconstruction helpers.
//!
//! [`CrossFactors`] gathers the pivot block, pivot columns, and pivot rows
//! from a [`CandidateMatrixSource`], and
//! provides methods to compute the left and right CI factors.

use crate::matrixluci::error::MatrixLuciError;
use crate::matrixluci::scalar::Scalar;
use crate::matrixluci::source::CandidateMatrixSource;
use crate::matrixluci::types::PivotSelectionCore;
use crate::matrixluci::Result;
use tensor4all_tensorbackend::{mat_mul, solve_matrix, Matrix};

/// Gather a dense column-major block from a source.
pub(crate) fn load_block<T: Scalar, S: CandidateMatrixSource<T>>(
    source: &S,
    rows: &[usize],
    cols: &[usize],
) -> Matrix<T> {
    let mut data = vec![T::zero(); rows.len() * cols.len()];
    source.get_block(rows, cols, &mut data);
    Matrix::from_col_major_vec(rows.len(), cols.len(), data)
}

/// Subtract one dense matrix from another in place.
pub(crate) fn subtract_inplace<T: Scalar>(lhs: &mut Matrix<T>, rhs: &Matrix<T>) {
    assert_eq!(lhs.nrows(), rhs.nrows());
    assert_eq!(lhs.ncols(), rhs.ncols());
    for j in 0..lhs.ncols() {
        for i in 0..lhs.nrows() {
            lhs[[i, j]] = lhs[[i, j]] - rhs[[i, j]];
        }
    }
}

/// Invert a square dense matrix via the configured tensor backend.
pub(crate) fn invert_square<T: Scalar>(matrix: &Matrix<T>) -> Result<Matrix<T>> {
    if matrix.nrows() != matrix.ncols() {
        return Err(MatrixLuciError::InvalidArgument {
            message: "pivot block must be square".to_string(),
        });
    }

    let n = matrix.nrows();
    let mut identity = Matrix::zeros(n, n);
    for i in 0..n {
        identity[[i, i]] = T::one();
    }

    solve_matrix(matrix, &identity).map_err(|err| MatrixLuciError::InvalidArgument {
        message: format!("pivot block solve failed: {err}"),
    })
}

/// Dense factors derived from a pivot selection.
///
/// Contains the pivot block `A[I, J]`, full pivot columns `A[:, J]`, and
/// full pivot rows `A[I, :]`. These are used to reconstruct the left and
/// right CI factors for the cross interpolation approximation
/// `A ~ A[:, J] * A[I, J]^{-1} * A[I, :]`.
#[derive(Debug, Clone)]
pub(crate) struct CrossFactors<T: Scalar> {
    /// Pivot block `A[I, J]`.
    pub pivot: Matrix<T>,
    /// Columns through selected pivot columns `A[:, J]`.
    pub pivot_cols: Matrix<T>,
    /// Rows through selected pivot rows `A[I, :]`.
    pub pivot_rows: Matrix<T>,
}

impl<T: Scalar> CrossFactors<T> {
    /// Gather a dense block from a source.
    #[cfg(test)]
    pub fn gather<S: CandidateMatrixSource<T>>(
        source: &S,
        rows: &[usize],
        cols: &[usize],
    ) -> Matrix<T> {
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
    pub fn pivot_inverse(&self) -> Result<Matrix<T>> {
        invert_square(&self.pivot)
    }

    /// Form `A[:, J] * A[I, J]^{-1}`.
    pub fn cols_times_pivot_inv(&self) -> Result<Matrix<T>> {
        let pivot_inv = self.pivot_inverse()?;
        mat_mul(&self.pivot_cols, &pivot_inv).map_err(|err| MatrixLuciError::InvalidArgument {
            message: format!("left factor multiplication failed: {err}"),
        })
    }

    /// Form `A[I, J]^{-1} * A[I, :]`.
    pub fn pivot_inv_times_rows(&self) -> Result<Matrix<T>> {
        let pivot_inv = self.pivot_inverse()?;
        mat_mul(&pivot_inv, &self.pivot_rows).map_err(|err| MatrixLuciError::InvalidArgument {
            message: format!("right factor multiplication failed: {err}"),
        })
    }
}

#[cfg(test)]
mod tests;
