//! Dense column-major matrix type and utility functions.
//!
//! [`Matrix<T>`] is a simple dense 2D matrix in column-major layout, indexed
//! by `m[[row, col]]`. It is the shared dense matrix boundary for tensor4all
//! crates that need flat buffers and backend-backed matrix multiplication.
//!
//! # Examples
//!
//! ```
//! use tensor4all_tensorbackend::{from_vec2d, Matrix};
//!
//! let m = from_vec2d(vec![
//!     vec![1.0_f64, 2.0],
//!     vec![3.0, 4.0],
//! ]);
//! assert_eq!(m.nrows(), 2);
//! assert_eq!(m.ncols(), 2);
//! assert_eq!(m[[0, 1]], 2.0);
//! assert_eq!(m[[1, 0]], 3.0);
//! ```

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use std::ops::{Index, IndexMut};

/// A dense 2D matrix in column-major layout.
///
/// Access elements with `m[[row, col]]` syntax. Data is stored contiguously
/// in column-major order, so flat buffers use `row + nrows * col`.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::Matrix;
///
/// let mut m = Matrix::zeros(2, 3);
/// m[[0, 1]] = 5.0_f64;
/// assert_eq!(m[[0, 1]], 5.0);
/// assert_eq!(m[[0, 0]], 0.0);
/// assert_eq!(m.nrows(), 2);
/// assert_eq!(m.ncols(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    data: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T> Matrix<T> {
    /// Create a matrix from raw column-major data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != nrows * ncols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);
    /// assert_eq!(m[[0, 0]], 1.0);
    /// assert_eq!(m[[0, 1]], 2.0);
    /// assert_eq!(m[[1, 0]], 3.0);
    /// assert_eq!(m[[1, 1]], 4.0);
    /// ```
    pub fn from_col_major_vec(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), nrows * ncols);
        Self { data, nrows, ncols }
    }

    /// View the underlying column-major data as a contiguous slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_col_major_vec(2, 2, vec![1, 3, 2, 4]);
    /// assert_eq!(m.as_col_major_slice(), &[1, 3, 2, 4]);
    /// ```
    pub fn as_col_major_slice(&self) -> &[T] {
        &self.data
    }

    fn offset(&self, row: usize, col: usize) -> usize {
        row + self.nrows * col
    }

    /// Number of rows
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns
    pub fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T: Clone> Matrix<T> {
    /// Create a new matrix filled with a constant value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_elem(2, 3, 7.0);
    /// assert_eq!(m[[0, 0]], 7.0);
    /// assert_eq!(m[[1, 2]], 7.0);
    /// ```
    pub fn from_elem(nrows: usize, ncols: usize, elem: T) -> Self {
        Self {
            data: vec![elem; nrows * ncols],
            nrows,
            ncols,
        }
    }
}

impl<T: Clone + Zero> Matrix<T> {
    /// Create a zeros matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::<f64>::zeros(2, 3);
    /// assert_eq!(m.nrows(), 2);
    /// assert_eq!(m.ncols(), 3);
    /// assert_eq!(m[[0, 0]], 0.0);
    /// assert_eq!(m[[1, 2]], 0.0);
    /// ```
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![T::zero(); nrows * ncols],
            nrows,
            ncols,
        }
    }
}

impl<T> Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &Self::Output {
        &self.data[self.offset(idx[0], idx[1])]
    }
}

impl<T> IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
        let offset = self.offset(idx[0], idx[1]);
        &mut self.data[offset]
    }
}

/// Create a matrix from a 2D vector.
///
/// Each inner `Vec` is one row. The resulting matrix is stored internally in
/// column-major order.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::from_vec2d;
///
/// let m = from_vec2d(vec![
///     vec![1.0, 2.0],
///     vec![3.0, 4.0],
/// ]);
/// assert_eq!(m.nrows(), 2);
/// assert_eq!(m.ncols(), 2);
/// assert_eq!(m[[0, 1]], 2.0);
/// assert_eq!(m[[1, 0]], 3.0);
/// ```
pub fn from_vec2d<T: Clone + Zero>(data: Vec<Vec<T>>) -> Matrix<T> {
    let nrows = data.len();
    let ncols = if nrows > 0 { data[0].len() } else { 0 };
    let mut m = Matrix::zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            m[[i, j]] = data[i][j].clone();
        }
    }
    m
}

/// Get a submatrix by selecting specific rows and columns.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, submatrix};
///
/// let m = from_vec2d(vec![
///     vec![1.0, 2.0, 3.0],
///     vec![4.0, 5.0, 6.0],
///     vec![7.0, 8.0, 9.0],
/// ]);
/// let sub = submatrix(&m, &[0, 2], &[1, 2]);
/// assert_eq!(sub.nrows(), 2);
/// assert_eq!(sub.ncols(), 2);
/// assert_eq!(sub[[0, 0]], 2.0); // m[0, 1]
/// assert_eq!(sub[[1, 1]], 9.0); // m[2, 2]
/// ```
pub fn submatrix<T: Clone + Zero>(m: &Matrix<T>, rows: &[usize], cols: &[usize]) -> Matrix<T> {
    let mut result = Matrix::zeros(rows.len(), cols.len());
    for (ri, &r) in rows.iter().enumerate() {
        for (ci, &c) in cols.iter().enumerate() {
            result[[ri, ci]] = m[[r, c]].clone();
        }
    }
    result
}

/// Swap two rows in a matrix in-place.
///
/// No-op if `a == b`.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, swap_rows};
///
/// let mut m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
/// swap_rows(&mut m, 0, 1);
/// assert_eq!(m[[0, 0]], 3.0);
/// assert_eq!(m[[1, 0]], 1.0);
/// ```
pub fn swap_rows<T>(m: &mut Matrix<T>, a: usize, b: usize) {
    if a == b {
        return;
    }
    for j in 0..m.ncols {
        let idx_a = m.offset(a, j);
        let idx_b = m.offset(b, j);
        m.data.swap(idx_a, idx_b);
    }
}

/// Swap two columns in a matrix in-place.
///
/// No-op if `a == b`.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, swap_cols};
///
/// let mut m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
/// swap_cols(&mut m, 0, 1);
/// assert_eq!(m[[0, 0]], 2.0);
/// assert_eq!(m[[0, 1]], 1.0);
/// ```
pub fn swap_cols<T>(m: &mut Matrix<T>, a: usize, b: usize) {
    if a == b {
        return;
    }
    for i in 0..m.nrows {
        let idx_a = m.offset(i, a);
        let idx_b = m.offset(i, b);
        m.data.swap(idx_a, idx_b);
    }
}

/// Transpose the matrix.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, transpose};
///
/// let m = from_vec2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
/// let mt = transpose(&m);
/// assert_eq!(mt.nrows(), 3);
/// assert_eq!(mt.ncols(), 2);
/// assert_eq!(mt[[0, 0]], 1.0);
/// assert_eq!(mt[[2, 1]], 6.0);
/// ```
pub fn transpose<T: Clone + Zero>(m: &Matrix<T>) -> Matrix<T> {
    let mut result = Matrix::zeros(m.ncols, m.nrows);
    for i in 0..m.nrows {
        for j in 0..m.ncols {
            result[[j, i]] = m[[i, j]].clone();
        }
    }
    result
}

/// Find the position and value of the maximum absolute value in a submatrix.
///
/// Searches within the rectangular region defined by `rows x cols` ranges.
/// Returns `(row, col, value)` of the element with the largest `|value|^2`.
///
/// # Panics
///
/// Panics if either range is empty.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, submatrix_argmax};
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 2.0, 3.0],
///     vec![4.0, 9.0, 6.0],
///     vec![7.0, 8.0, 5.0],
/// ]);
/// let (row, col, val) = submatrix_argmax(&m, 0..3, 0..3);
/// assert_eq!(row, 1);
/// assert_eq!(col, 1);
/// assert_eq!(val, 9.0);
/// ```
pub fn submatrix_argmax<T: MatrixScalar>(
    a: &Matrix<T>,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) -> (usize, usize, T) {
    assert!(!rows.is_empty(), "rows must not be empty");
    assert!(!cols.is_empty(), "cols must not be empty");

    let mut max_val: f64 = a[[rows.start, cols.start]].matrix_abs_sq();
    let mut max_row = rows.start;
    let mut max_col = cols.start;

    for r in rows {
        for c in cols.clone() {
            let val: f64 = a[[r, c]].matrix_abs_sq();
            if val > max_val {
                max_val = val;
                max_row = r;
                max_col = c;
            }
        }
    }

    (max_row, max_col, a[[max_row, max_col]])
}

/// BLAS-backed matrix multiplication dispatch.
///
/// Implemented for all scalar types supported by tenferro einsum
/// (f64, f32, Complex64, Complex32). This trait is sealed — external
/// types cannot implement it.
pub trait BlasMul: Sized {
    #[doc(hidden)]
    fn blas_mat_mul(a: &Matrix<Self>, b: &Matrix<Self>) -> Matrix<Self>;
}

macro_rules! impl_blas_mul {
    ($($t:ty),*) => {
        $(
        impl BlasMul for $t {
            fn blas_mat_mul(a: &Matrix<Self>, b: &Matrix<Self>) -> Matrix<Self> {
                use tenferro_einsum::typed_eager_einsum;
                use tenferro_tensor::TypedTensor;
                use crate::with_default_backend;

                let m = a.nrows();
                let k = a.ncols();
                let n = b.ncols();
                assert_eq!(b.nrows(), k);

                let a_tensor = TypedTensor::<$t>::from_vec(
                    vec![m, k],
                    a.as_col_major_slice().to_vec(),
                );
                let b_tensor = TypedTensor::<$t>::from_vec(
                    vec![k, n],
                    b.as_col_major_slice().to_vec(),
                );
                let c = with_default_backend(|backend| {
                    typed_eager_einsum(backend, &[&a_tensor, &b_tensor], "ij,jk->ik")
                })
                .expect("einsum failed");
                Matrix::from_col_major_vec(m, n, c.as_slice().to_vec())
            }
        }
        )*
    };
}

impl_blas_mul!(f64, f32, num_complex::Complex64, num_complex::Complex32);

/// Scalar bound for dense backend matrix utilities.
///
/// This is the storage/linalg-layer scalar trait. Higher-level crates may
/// extend it with domain-specific methods, but matrix utilities only rely on
/// these algebraic operations and absolute-value comparisons.
pub trait MatrixScalar:
    Clone
    + Copy
    + Zero
    + One
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + Default
    + Send
    + Sync
    + BlasMul
    + 'static
{
    /// Squared absolute value as `f64`.
    fn matrix_abs_sq(self) -> f64;
}

impl MatrixScalar for f64 {
    fn matrix_abs_sq(self) -> f64 {
        self * self
    }
}

impl MatrixScalar for f32 {
    fn matrix_abs_sq(self) -> f64 {
        (self * self) as f64
    }
}

impl MatrixScalar for Complex64 {
    fn matrix_abs_sq(self) -> f64 {
        self.norm_sqr()
    }
}

impl MatrixScalar for Complex32 {
    fn matrix_abs_sq(self) -> f64 {
        self.norm_sqr() as f64
    }
}

/// Matrix multiplication: A * B.
///
/// Uses BLAS-backed einsum via tenferro for high performance.
///
/// # Panics
///
/// Panics if `a.ncols() != b.nrows()`.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, mat_mul};
///
/// let a = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
/// let b = from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
/// let c = mat_mul(&a, &b);
/// assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
/// assert!((c[[0, 1]] - 22.0).abs() < 1e-10);
/// assert!((c[[1, 0]] - 43.0).abs() < 1e-10);
/// assert!((c[[1, 1]] - 50.0).abs() < 1e-10);
/// ```
pub fn mat_mul<T: BlasMul>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    T::blas_mat_mul(a, b)
}

#[cfg(test)]
mod tests;
