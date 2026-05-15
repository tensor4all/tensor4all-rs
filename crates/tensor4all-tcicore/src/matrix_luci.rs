//! Matrix LU-based Cross Interpolation (MatrixLUCI) implementation.
//!
//! [`MatrixLUCI`] provides a higher-level [`Matrix`] API over the lower-level
//! `matrixluci` substrate. It decomposes a matrix into left and right factors
//! via LU cross interpolation and implements [`AbstractMatrixCI`].

use crate::error::{MatrixCIError, Result};
use crate::matrixlu::RrLUOptions;
use crate::matrixluci::block_rook::LazyBlockRookKernel;
use crate::matrixluci::dense::DenseLuKernel;
use crate::matrixluci::factors::CrossFactors;
use crate::matrixluci::source::{DenseMatrixSource, LazyMatrixSource};
use crate::matrixluci::types::{PivotKernelOptions, PivotSelectionCore};
use crate::matrixluci::PivotKernel;
use crate::scalar::Scalar;
use crate::traits::AbstractMatrixCI;
use tensor4all_tensorbackend::Matrix;

/// Matrix LU-based Cross Interpolation.
///
/// This is a higher-level [`Matrix`] wrapper around the lower-level `matrixluci`
/// substrate.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
/// use tensor4all_tensorbackend::from_vec2d;
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
pub struct MatrixLUCI<T: Scalar + crate::MatrixLuciScalar> {
    nrows: usize,
    ncols: usize,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    left: Matrix<T>,
    right: Matrix<T>,
    pivot_errors: Vec<f64>,
}

/// High-level factors produced by MatrixLUCI.
///
/// This is the public result of the LUCI factorization facade. It exposes
/// the selected pivot metadata and the left and right factors
/// needed by higher-level tensor-network code without exposing the low-level
/// pivot kernel substrate.
///
/// Related types: [`MatrixLUCI`] is the owning CI wrapper, while
/// [`MatrixACA`](crate::MatrixACA) and [`RrLU`](crate::RrLU) are related
/// matrix factorization entry points.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::matrix_luci_factors_from_matrix;
/// use tensor4all_tensorbackend::from_vec2d;
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 2.0],
///     vec![3.0, 4.0],
/// ]);
/// let factors = matrix_luci_factors_from_matrix(&m, None).unwrap();
/// assert!(factors.rank >= 1);
/// assert_eq!(factors.row_indices.len(), factors.rank);
/// assert_eq!(factors.left.nrows(), m.nrows());
/// assert_eq!(factors.right.ncols(), m.ncols());
/// ```
#[derive(Debug, Clone)]
pub struct MatrixLuciFactors<T> {
    /// Selected row indices.
    pub row_indices: Vec<usize>,
    /// Selected column indices.
    pub col_indices: Vec<usize>,
    /// Pivot error history.
    pub pivot_errors: Vec<f64>,
    /// Selected rank.
    pub rank: usize,
    /// Left factor.
    pub left: Matrix<T>,
    /// Right factor.
    pub right: Matrix<T>,
}

pub(crate) fn map_backend_error(err: crate::matrixluci::MatrixLuciError) -> MatrixCIError {
    match err {
        crate::matrixluci::MatrixLuciError::InvalidArgument { message } => {
            MatrixCIError::InvalidArgument { message }
        }
    }
}

fn factors_to_public<T>(
    selection: PivotSelectionCore,
    factors: CrossFactors<T>,
    left_orthogonal: bool,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let left = if left_orthogonal {
        factors.cols_times_pivot_inv().map_err(map_backend_error)?
    } else {
        factors.pivot_cols.clone()
    };
    let right = if left_orthogonal {
        factors.pivot_rows.clone()
    } else {
        factors.pivot_inv_times_rows().map_err(map_backend_error)?
    };

    Ok(MatrixLuciFactors {
        row_indices: selection.row_indices,
        col_indices: selection.col_indices,
        pivot_errors: selection.pivot_errors,
        rank: selection.rank,
        left,
        right,
    })
}

pub(crate) fn dense_matrix_luci_factors_from_matrix<T>(
    a: &Matrix<T>,
    options: RrLUOptions,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
    DenseLuKernel: PivotKernel<T>,
{
    let source = DenseMatrixSource::from_column_major(a.as_col_major_slice(), a.nrows(), a.ncols());
    let kernel_options = PivotKernelOptions {
        max_rank: options.max_rank,
        rel_tol: options.rel_tol,
        abs_tol: options.abs_tol,
        left_orthogonal: options.left_orthogonal,
    };

    let selection = DenseLuKernel
        .factorize(&source, &kernel_options)
        .map_err(map_backend_error)?;
    let factors = CrossFactors::from_source(&source, &selection).map_err(map_backend_error)?;
    factors_to_public(selection, factors, options.left_orthogonal)
}

pub(crate) fn lazy_matrix_luci_factors_from_blocks<T, F>(
    nrows: usize,
    ncols: usize,
    fill_block: F,
    options: RrLUOptions,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
    F: Fn(&[usize], &[usize], &mut [T]),
    LazyBlockRookKernel: PivotKernel<T>,
{
    let source = LazyMatrixSource::new(nrows, ncols, fill_block);
    let kernel_options = PivotKernelOptions {
        max_rank: options.max_rank,
        rel_tol: options.rel_tol,
        abs_tol: options.abs_tol,
        left_orthogonal: options.left_orthogonal,
    };

    let selection = LazyBlockRookKernel
        .factorize(&source, &kernel_options)
        .map_err(map_backend_error)?;
    let factors = CrossFactors::from_source(&source, &selection).map_err(map_backend_error)?;
    factors_to_public(selection, factors, options.left_orthogonal)
}

/// Factorize a dense matrix with MatrixLUCI.
///
/// Returns the selected pivot metadata together with left and
/// right factors. This is the public facade used by higher-level crates.
///
/// # Arguments
///
/// * `a` - Dense matrix to factorize.
/// * `options` - Optional rank and tolerance controls. `None` uses the
///   default LUCI settings.
///
/// # Returns
///
/// A [`MatrixLuciFactors`] value containing the selected pivot indices,
/// error history, rank, and factors.
///
/// # Errors
///
/// Returns a [`MatrixCIError`] if the factorization fails, for example if the
/// pivot block is singular or the backend rejects the input.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::matrix_luci_factors_from_matrix;
/// use tensor4all_tensorbackend::from_vec2d;
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 0.0],
///     vec![0.0, 2.0],
/// ]);
/// let factors = matrix_luci_factors_from_matrix(&m, None).unwrap();
/// assert_eq!(factors.left.nrows(), 2);
/// assert_eq!(factors.right.ncols(), 2);
/// assert_eq!(factors.row_indices.len(), factors.rank);
/// ```
pub fn matrix_luci_factors_from_matrix<T>(
    a: &Matrix<T>,
    options: Option<RrLUOptions>,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    <T as crate::MatrixLuciScalar>::matrix_luci_factors_from_matrix(a, options.unwrap_or_default())
}

/// Factorize a lazily supplied matrix with MatrixLUCI block-rook search.
///
/// The caller provides a block-fill closure that receives row and column
/// index lists and writes the corresponding matrix block in column-major
/// order.
///
/// # Arguments
///
/// * `nrows` - Number of matrix rows.
/// * `ncols` - Number of matrix columns.
/// * `fill_block` - Closure that fills `out` with `A[rows, cols]` in
///   column-major order.
/// * `options` - Rank and tolerance controls.
///
/// # Returns
///
/// A [`MatrixLuciFactors`] value containing the pivot metadata and
/// factors.
///
/// # Errors
///
/// Returns a [`MatrixCIError`] if the lazy factorization fails or if the
/// block callback is inconsistent.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::matrix_luci_factors_from_blocks;
/// use tensor4all_tcicore::RrLUOptions;
///
/// let factors = matrix_luci_factors_from_blocks(
///     2,
///     2,
///     |rows, cols, out| {
///         for (j, &col) in cols.iter().enumerate() {
///             for (i, &row) in rows.iter().enumerate() {
///                 out[i + rows.len() * j] = if row == col { 1.0 } else { 0.0 };
///             }
///         }
///     },
///     RrLUOptions::default(),
/// ).unwrap();
/// assert_eq!(factors.rank, 2);
/// ```
pub fn matrix_luci_factors_from_blocks<T, F>(
    nrows: usize,
    ncols: usize,
    fill_block: F,
    options: RrLUOptions,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
    F: Fn(&[usize], &[usize], &mut [T]),
{
    <T as crate::MatrixLuciScalar>::matrix_luci_factors_from_blocks(
        nrows, ncols, fill_block, options,
    )
}

impl<T> MatrixLUCI<T>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    /// Create a MatrixLUCI from a dense matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![
    ///     vec![2.0_f64, 0.0],
    ///     vec![0.0, 3.0],
    /// ]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// assert!(ci.rank() >= 1);
    /// ```
    pub fn from_matrix(a: &Matrix<T>, options: Option<RrLUOptions>) -> Result<Self> {
        let factors = matrix_luci_factors_from_matrix(a, options)?;

        Ok(Self {
            nrows: a.nrows(),
            ncols: a.ncols(),
            row_indices: factors.row_indices,
            col_indices: factors.col_indices,
            left: factors.left,
            right: factors.right,
            pivot_errors: factors.pivot_errors,
        })
    }

    /// Left CI factor (shape: `nrows x rank`).
    ///
    /// The approximation is `left * right`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::{from_vec2d, mat_mul};
    ///
    /// let m = from_vec2d(vec![
    ///     vec![1.0_f64, 2.0],
    ///     vec![3.0, 4.0],
    /// ]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let reconstructed = mat_mul(&ci.left(), &ci.right()).unwrap();
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         assert!((reconstructed[[i, j]] - m[[i, j]]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    pub fn left(&self) -> Matrix<T> {
        self.left.clone()
    }

    /// Right CI factor (shape: `rank x ncols`).
    ///
    /// The approximation is `left * right`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::{from_vec2d, mat_mul};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let r = ci.right();
    /// assert_eq!(r.nrows(), ci.rank());
    /// assert_eq!(r.ncols(), ci.ncols());
    /// // left * right reconstructs the matrix
    /// let recon = mat_mul(&ci.left(), &r).unwrap();
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         assert!((recon[[i, j]] - m[[i, j]]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    pub fn right(&self) -> Matrix<T> {
        self.right.clone()
    }

    /// Pivot error history (one entry per pivot, plus a final residual estimate).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::MatrixLUCI;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let errs = ci.pivot_errors();
    /// assert!(!errs.is_empty());
    /// // All errors are non-negative
    /// for &e in &errs {
    ///     assert!(e >= 0.0);
    /// }
    /// ```
    pub fn pivot_errors(&self) -> Vec<f64> {
        self.pivot_errors.clone()
    }

    /// Last pivot error (the residual estimate after all pivots).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::MatrixLUCI;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let err = ci.last_pivot_error();
    /// assert!(err >= 0.0);
    /// ```
    pub fn last_pivot_error(&self) -> f64 {
        self.pivot_errors.last().copied().unwrap_or(0.0)
    }
}

impl<T> AbstractMatrixCI<T> for MatrixLUCI<T>
where
    T: Scalar + crate::MatrixLuciScalar,
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
        let mut result = Matrix::zeros(rows.len(), cols.len());
        for (j_out, &col) in cols.iter().enumerate() {
            for (i_out, &row) in rows.iter().enumerate() {
                let mut sum = T::zero();
                for k in 0..r {
                    sum = sum + self.left[[row, k]] * self.right[[k, col]];
                }
                result[[i_out, j_out]] = sum;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests;
