//! Matrix LU-based Cross Interpolation (MatrixLUCI) implementation.

use crate::error::{MatrixCIError, Result};
use crate::matrix::{submatrix, zeros, Matrix};
use crate::matrixlu::RrLUOptions;
use crate::matrixluci::{
    CrossFactors, DenseFaerLuKernel, DenseMatrixSource, PivotKernel, PivotKernelOptions,
    PivotSelectionCore,
};
use crate::scalar::Scalar;
use crate::traits::AbstractMatrixCI;

/// Matrix LU-based Cross Interpolation.
///
/// This is a higher-level row-major wrapper around the lower-level `matrixluci`
/// substrate.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI, from_vec2d};
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 2.0, 3.0],
///     vec![4.0, 5.0, 6.0],
///     vec![7.0, 8.0, 9.0],
/// ]);
///
/// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
/// // The approximation must have at most rank min(nrows, ncols)
/// assert!(ci.rank() <= 3);
/// // Reconstructed matrix should match original at pivot positions
/// let row_indices = ci.row_indices().to_vec();
/// let col_indices = ci.col_indices().to_vec();
/// for (&i, &j) in row_indices.iter().zip(col_indices.iter()) {
///     let approx = ci.evaluate(i, j);
///     let exact = m[[i, j]];
///     assert!((approx - exact).abs() < 1e-10);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MatrixLUCI<T: Scalar + crate::matrixluci::Scalar> {
    nrows: usize,
    ncols: usize,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    left: Matrix<T>,
    right: Matrix<T>,
    pivot_errors: Vec<f64>,
}

pub(crate) fn map_backend_error(err: crate::matrixluci::MatrixLuciError) -> MatrixCIError {
    match err {
        crate::matrixluci::MatrixLuciError::InvalidArgument { message } => {
            MatrixCIError::InvalidArgument { message }
        }
        crate::matrixluci::MatrixLuciError::SingularPivotBlock => MatrixCIError::SingularMatrix,
    }
}

pub(crate) fn to_column_major<T: Scalar>(matrix: &Matrix<T>) -> Vec<T> {
    let mut out = Vec::with_capacity(matrix.nrows() * matrix.ncols());
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            out.push(matrix[[row, col]]);
        }
    }
    out
}

pub(crate) fn to_row_major<T: Scalar + crate::matrixluci::Scalar>(
    matrix: &crate::matrixluci::DenseOwnedMatrix<T>,
) -> Matrix<T> {
    let mut out = zeros(matrix.nrows(), matrix.ncols());
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            out[[row, col]] = matrix[[row, col]];
        }
    }
    out
}

pub(crate) fn dense_selection_from_matrix<T>(
    a: &Matrix<T>,
    options: RrLUOptions,
) -> Result<(PivotSelectionCore, CrossFactors<T>)>
where
    T: Scalar + crate::matrixluci::Scalar,
    DenseFaerLuKernel: PivotKernel<T>,
{
    let data = to_column_major(a);
    let source = DenseMatrixSource::from_column_major(&data, a.nrows(), a.ncols());
    let kernel_options = PivotKernelOptions {
        max_rank: options.max_rank,
        rel_tol: options.rel_tol,
        abs_tol: options.abs_tol,
        left_orthogonal: options.left_orthogonal,
    };

    let selection = DenseFaerLuKernel
        .factorize(&source, &kernel_options)
        .map_err(map_backend_error)?;
    let factors = CrossFactors::from_source(&source, &selection).map_err(map_backend_error)?;
    Ok((selection, factors))
}

impl<T> MatrixLUCI<T>
where
    T: Scalar + crate::matrixluci::Scalar,
    DenseFaerLuKernel: PivotKernel<T>,
{
    /// Create a MatrixLUCI from a dense row-major matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI, from_vec2d};
    ///
    /// let m = from_vec2d(vec![
    ///     vec![2.0_f64, 0.0],
    ///     vec![0.0, 3.0],
    /// ]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// assert!(ci.rank() >= 1);
    /// ```
    pub fn from_matrix(a: &Matrix<T>, options: Option<RrLUOptions>) -> Result<Self> {
        let options = options.unwrap_or_default();
        let left_orthogonal = options.left_orthogonal;
        let (selection, factors) = dense_selection_from_matrix(a, options)?;

        let left = if left_orthogonal {
            to_row_major(&factors.cols_times_pivot_inv().map_err(map_backend_error)?)
        } else {
            to_row_major(&factors.pivot_cols)
        };
        let right = if left_orthogonal {
            to_row_major(&factors.pivot_rows)
        } else {
            to_row_major(&factors.pivot_inv_times_rows().map_err(map_backend_error)?)
        };

        Ok(Self {
            nrows: a.nrows(),
            ncols: a.ncols(),
            row_indices: selection.row_indices,
            col_indices: selection.col_indices,
            left,
            right,
            pivot_errors: selection.pivot_errors,
        })
    }

    /// Left CI factor.
    pub fn left(&self) -> Matrix<T> {
        self.left.clone()
    }

    /// Right CI factor.
    pub fn right(&self) -> Matrix<T> {
        self.right.clone()
    }

    /// Pivot error history.
    pub fn pivot_errors(&self) -> Vec<f64> {
        self.pivot_errors.clone()
    }

    /// Last pivot error.
    pub fn last_pivot_error(&self) -> f64 {
        self.pivot_errors.last().copied().unwrap_or(0.0)
    }
}

impl<T> AbstractMatrixCI<T> for MatrixLUCI<T>
where
    T: Scalar + crate::matrixluci::Scalar,
{
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn rank(&self) -> usize {
        self.row_indices.len()
    }

    fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    fn evaluate(&self, i: usize, j: usize) -> T {
        let mut sum = T::zero();
        for k in 0..self.rank() {
            sum = sum + self.left[[i, k]] * self.right[[k, j]];
        }
        sum
    }

    fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Matrix<T> {
        let r = self.rank();
        let left_sub = submatrix(&self.left, rows, &(0..r).collect::<Vec<_>>());
        let right_sub = submatrix(&self.right, &(0..r).collect::<Vec<_>>(), cols);

        crate::matrix::mat_mul(&left_sub, &right_sub)
    }
}

#[cfg(test)]
mod tests;
