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

use anyhow::{ensure, Context, Result};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use std::ops::{Index, IndexMut};
use tenferro::{Tensor, TensorBackend, TensorScalar, TypedTensor};

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

/// Error returned when converting a [`TypedTensor`] into a [`Matrix`].
///
/// Use this when accepting dynamic tensor-shaped values at a dense-matrix
/// boundary. It reports whether conversion failed because the tensor was not a
/// rank-2 matrix or because its host buffer could not be consumed.
///
/// # Examples
///
/// ```
/// use tenferro::TypedTensor;
/// use tensor4all_tensorbackend::Matrix;
///
/// let tensor = TypedTensor::from_vec_col_major(vec![2, 1, 1], vec![1.0_f64, 2.0]);
/// let err = Matrix::try_from_typed_tensor(tensor).unwrap_err();
/// assert!(err.to_string().contains("rank-2 tensor"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum MatrixTensorConversionError {
    /// The input tensor rank was not two.
    #[error("expected a rank-2 tensor, got shape {shape:?}")]
    Rank {
        /// Tensor shape that failed the rank check.
        shape: Vec<usize>,
    },
    /// The tensor did not contain an owned host buffer that can be consumed.
    #[error("failed to consume typed tensor host buffer: {message}")]
    HostBuffer {
        /// Backend conversion error reported by tenferro.
        message: String,
    },
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

    /// View the underlying column-major data as a mutable contiguous slice.
    ///
    /// The slice uses `row + nrows * col` ordering. This is useful for kernels
    /// that validate dimensions once and then operate over contiguous columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let mut m = Matrix::from_col_major_vec(2, 2, vec![1, 3, 2, 4]);
    /// m.as_col_major_mut_slice()[1] = 30;
    /// assert_eq!(m[[1, 0]], 30);
    /// ```
    pub fn as_col_major_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume the matrix and return its owned column-major buffer.
    ///
    /// The returned buffer uses `row + nrows * col` ordering. Use this when
    /// transferring matrix storage to another column-major dense container
    /// without cloning.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);
    /// let data = m.into_col_major_vec();
    /// assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn into_col_major_vec(self) -> Vec<T> {
        self.data
    }

    /// Borrow this matrix as an owned tenferro [`TypedTensor`].
    ///
    /// This clones the matrix buffer and preserves column-major layout. Use
    /// [`Matrix::into_typed_tensor`] when the matrix can be consumed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 3.0, 2.0, 4.0]);
    /// let tensor = m.to_typed_tensor();
    /// assert_eq!(tensor.shape, vec![2, 2]);
    /// assert_eq!(tensor.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
    /// assert_eq!(m.as_col_major_slice(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn to_typed_tensor(&self) -> TypedTensor<T>
    where
        T: TensorScalar,
    {
        TypedTensor::from_vec_col_major(vec![self.nrows, self.ncols], self.data.clone())
    }

    /// Consume this matrix as a tenferro [`TypedTensor`] without cloning.
    ///
    /// The tensor shape is `[nrows, ncols]`, and the owned data remains in
    /// column-major layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 3.0, 2.0, 4.0]);
    /// let tensor = m.into_typed_tensor();
    /// assert_eq!(tensor.shape, vec![2, 2]);
    /// assert_eq!(tensor.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn into_typed_tensor(self) -> TypedTensor<T>
    where
        T: TensorScalar,
    {
        TypedTensor::from_vec_col_major(vec![self.nrows, self.ncols], self.data)
    }

    /// Consume a rank-2 tenferro [`TypedTensor`] as a [`Matrix`].
    ///
    /// The input tensor must have shape `[nrows, ncols]` and an owned host
    /// buffer. The buffer is reused without cloning and interpreted as
    /// column-major matrix storage.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixTensorConversionError::Rank`] if the tensor is not
    /// rank-2, or [`MatrixTensorConversionError::HostBuffer`] if tenferro
    /// cannot export the tensor as an owned host buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::TypedTensor;
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
    /// let m = Matrix::try_from_typed_tensor(tensor).unwrap();
    /// assert_eq!(m.nrows(), 2);
    /// assert_eq!(m.ncols(), 2);
    /// assert_eq!(m[[0, 1]], 2.0);
    /// ```
    pub fn try_from_typed_tensor(
        tensor: TypedTensor<T>,
    ) -> std::result::Result<Self, MatrixTensorConversionError>
    where
        T: Clone,
    {
        let (shape, data) = tensor.try_into_vec_col_major().map_err(|source| {
            MatrixTensorConversionError::HostBuffer {
                message: source.to_string(),
            }
        })?;
        if shape.len() != 2 {
            return Err(MatrixTensorConversionError::Rank { shape });
        }
        Ok(Self::from_col_major_vec(shape[0], shape[1], data))
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
    assert!(
        rows.iter().all(|&row| row < m.nrows),
        "submatrix row index out of bounds"
    );
    assert!(
        cols.iter().all(|&col| col < m.ncols),
        "submatrix column index out of bounds"
    );

    let mut data = Vec::with_capacity(rows.len() * cols.len());
    let source = m.as_col_major_slice();
    for &col in cols {
        let col_start = col * m.nrows;
        for &row in rows {
            let offset = col_start + row;
            // SAFETY: rows and cols are range-checked above, and Matrix stores
            // exactly nrows * ncols values in column-major order.
            data.push(unsafe { source.get_unchecked(offset).clone() });
        }
    }
    Matrix::from_col_major_vec(rows.len(), cols.len(), data)
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
    assert!(a < m.nrows, "row index out of bounds");
    assert!(b < m.nrows, "row index out of bounds");
    let nrows = m.nrows;
    let ncols = m.ncols;
    let data = m.as_col_major_mut_slice();
    for j in 0..ncols {
        data.swap(a + nrows * j, b + nrows * j);
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
    assert!(a < m.ncols, "column index out of bounds");
    assert!(b < m.ncols, "column index out of bounds");
    let nrows = m.nrows;
    let start_a = nrows * a;
    let start_b = nrows * b;
    let data = m.as_col_major_mut_slice();
    if start_a < start_b {
        let (left, right) = data.split_at_mut(start_b);
        left[start_a..start_a + nrows].swap_with_slice(&mut right[..nrows]);
    } else {
        let (left, right) = data.split_at_mut(start_a);
        right[..nrows].swap_with_slice(&mut left[start_b..start_b + nrows]);
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
    for j in 0..m.ncols {
        for i in 0..m.nrows {
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
    assert!(rows.end <= a.nrows, "row range out of bounds");
    assert!(cols.end <= a.ncols, "column range out of bounds");

    let data = a.as_col_major_slice();
    let first_offset = rows.start + a.nrows * cols.start;
    // SAFETY: the non-empty ranges are checked against the matrix shape above.
    let first = unsafe { *data.get_unchecked(first_offset) };
    let mut max_val: f64 = first.matrix_abs_sq();
    let mut max_row = rows.start;
    let mut max_col = cols.start;
    let row_start = rows.start;
    let row_end = rows.end;
    let col_start = cols.start;
    let col_end = cols.end;

    for c in col_start..col_end {
        let col_start_offset = row_start + a.nrows * c;
        for (offset, r) in (col_start_offset..).zip(row_start..row_end) {
            // SAFETY: row and column loops stay within the checked ranges.
            let value = unsafe { *data.get_unchecked(offset) };
            let val: f64 = value.matrix_abs_sq();
            if val > max_val {
                max_val = val;
                max_row = r;
                max_col = c;
            }
        }
    }

    let max_offset = max_row + a.nrows * max_col;
    // SAFETY: max_row/max_col were selected from the checked ranges.
    (max_row, max_col, unsafe { *data.get_unchecked(max_offset) })
}

/// BLAS-backed matrix multiplication dispatch.
///
/// Implemented for all scalar types supported by tenferro einsum
/// (f64, f32, Complex64, Complex32). This trait is sealed — external
/// types cannot implement it.
pub trait BlasMul: Sized {
    #[doc(hidden)]
    fn blas_mat_mul(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>>;

    #[doc(hidden)]
    fn blas_mat_mul_owned(a: Matrix<Self>, b: Matrix<Self>) -> Result<Matrix<Self>>;
}

fn dot_general_matrices<T>(
    a_tensor: Tensor,
    b_tensor: Tensor,
    m: usize,
    n: usize,
) -> Result<Matrix<T>>
where
    T: TensorScalar,
{
    use crate::with_default_backend;
    use tenferro::DotGeneralConfig;

    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let c = with_default_backend(|backend| {
        backend.with_exec_session(|exec| exec.dot_general(&a_tensor, &b_tensor, &config))
    })
    .context("matrix multiplication failed")?;
    let c = T::try_into_typed(c)
        .ok_or_else(|| anyhow::anyhow!("matrix multiplication returned wrong dtype"))?;
    let result = Matrix::try_from_typed_tensor(c)?;
    ensure!(
        result.nrows() == m && result.ncols() == n,
        "matrix multiplication returned shape {}x{} for expected shape {}x{}",
        result.nrows(),
        result.ncols(),
        m,
        n
    );
    ensure!(
        result.as_col_major_slice().len() == m * n,
        "matrix multiplication returned {} values for expected shape {}x{}",
        result.as_col_major_slice().len(),
        m,
        n
    );
    Ok(result)
}

macro_rules! impl_blas_mul {
    ($($t:ty),*) => {
        $(
        impl BlasMul for $t {
            fn blas_mat_mul(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>> {
                let m = a.nrows();
                let k = a.ncols();
                let n = b.ncols();
                ensure!(
                    b.nrows() == k,
                    "matrix dimensions must agree for multiplication: left is {}x{}, right is {}x{}",
                    m,
                    k,
                    b.nrows(),
                    n
                );

                let a_tensor: Tensor = a.to_typed_tensor().into();
                let b_tensor: Tensor = b.to_typed_tensor().into();
                dot_general_matrices::<$t>(a_tensor, b_tensor, m, n)
            }

            fn blas_mat_mul_owned(a: Matrix<Self>, b: Matrix<Self>) -> Result<Matrix<Self>> {
                let m = a.nrows();
                let k = a.ncols();
                let n = b.ncols();
                ensure!(
                    b.nrows() == k,
                    "matrix dimensions must agree for multiplication: left is {}x{}, right is {}x{}",
                    m,
                    k,
                    b.nrows(),
                    n
                );

                let a_tensor: Tensor = a.into_typed_tensor().into();
                let b_tensor: Tensor = b.into_typed_tensor().into();
                dot_general_matrices::<$t>(a_tensor, b_tensor, m, n)
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
/// # Errors
///
/// Returns an error if `a.ncols() != b.nrows()` or the backend einsum fails.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, mat_mul};
///
/// let a = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
/// let b = from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
/// let c = mat_mul(&a, &b).unwrap();
/// assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
/// assert!((c[[0, 1]] - 22.0).abs() < 1e-10);
/// assert!((c[[1, 0]] - 43.0).abs() < 1e-10);
/// assert!((c[[1, 1]] - 50.0).abs() < 1e-10);
/// ```
pub fn mat_mul<T: BlasMul>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    T::blas_mat_mul(a, b)
}

/// Matrix multiplication: consume `A` and `B`, returning `A * B`.
///
/// Uses BLAS-backed einsum via tenferro. Compared with [`mat_mul`], this
/// reuses the input matrix buffers when building tenferro tensors.
///
/// # Errors
///
/// Returns an error if `a.ncols() != b.nrows()` or the backend einsum fails.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, mat_mul_owned};
///
/// let a = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
/// let b = from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
/// let c = mat_mul_owned(a, b).unwrap();
/// assert_eq!(c.as_col_major_slice(), &[19.0, 43.0, 22.0, 50.0]);
/// ```
pub fn mat_mul_owned<T: BlasMul>(a: Matrix<T>, b: Matrix<T>) -> Result<Matrix<T>> {
    T::blas_mat_mul_owned(a, b)
}

/// Batched matrix multiplication for column-major matrices with one shared shape.
///
/// Computes `C[p] = A[p] * B[p]` for `batch` matrices. Each `A[p]` is an
/// `m x k` column-major matrix and each `B[p]` is a `k x n` column-major
/// matrix. The input buffers store complete matrices consecutively, and the
/// returned buffer stores `batch` consecutive `m x n` column-major outputs.
///
/// # Errors
///
/// Returns an error if the input buffer lengths do not match the declared
/// shapes or if the backend rejects the batched GEMM.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::batched_mat_mul_same_shape;
///
/// let a = vec![1.0_f64, 3.0, 2.0, 4.0];
/// let b = vec![5.0_f64, 7.0, 6.0, 8.0];
/// let out = batched_mat_mul_same_shape(1, 2, 2, 2, &a, &b).unwrap();
/// assert_eq!(out, vec![19.0, 43.0, 22.0, 50.0]);
/// ```
pub fn batched_mat_mul_same_shape<T>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &[T],
    b: &[T],
) -> Result<Vec<T>>
where
    T: tenferro::TensorScalar + Copy,
{
    batched_mat_mul_same_shape_owned(batch, m, k, n, a.to_vec(), b.to_vec())
}

/// Batched matrix multiplication while consuming column-major input buffers.
///
/// This is the owned-buffer counterpart of [`batched_mat_mul_same_shape`].
/// It avoids cloning the two input batches when callers have just built the
/// contiguous buffers for a backend call.
///
/// # Errors
///
/// Returns an error if the input buffer lengths do not match the declared
/// shapes or if the backend rejects the batched GEMM.
pub fn batched_mat_mul_same_shape_owned<T>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a: Vec<T>,
    b: Vec<T>,
) -> Result<Vec<T>>
where
    T: tenferro::TensorScalar + Copy,
{
    use crate::with_default_backend;
    use tenferro::{DotGeneralConfig, TensorBackend};

    validate_batched_mat_mul_inputs(batch, m, k, n, a.len(), b.len())?;

    let a_tensor = T::into_tensor(vec![m, k, batch], a);
    let b_tensor = T::into_tensor(vec![k, n, batch], b);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![2],
        rhs_batch_dims: vec![2],
    };
    let c = with_default_backend(|backend| {
        backend.with_exec_session(|exec| exec.dot_general(&a_tensor, &b_tensor, &config))
    })
    .context("batched matrix multiplication failed")?;
    let c = T::try_into_typed(c)
        .ok_or_else(|| anyhow::anyhow!("batched matrix multiplication returned wrong dtype"))?;
    let (_shape, data) = c.try_into_vec_col_major()?;
    let expected_len = batch
        .checked_mul(m)
        .and_then(|value| value.checked_mul(n))
        .ok_or_else(|| anyhow::anyhow!("batched matrix multiplication output shape overflows"))?;
    ensure!(
        data.len() == expected_len,
        "batched matrix multiplication returned {} values for expected shape {}x{}x{}",
        data.len(),
        m,
        n,
        batch
    );
    Ok(data)
}

fn validate_batched_mat_mul_inputs(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a_len: usize,
    b_len: usize,
) -> Result<()> {
    let expected_a_len = batch
        .checked_mul(m)
        .and_then(|value| value.checked_mul(k))
        .ok_or_else(|| anyhow::anyhow!("batched matrix multiplication left shape overflows"))?;
    let expected_b_len = batch
        .checked_mul(k)
        .and_then(|value| value.checked_mul(n))
        .ok_or_else(|| anyhow::anyhow!("batched matrix multiplication right shape overflows"))?;
    ensure!(
        a_len == expected_a_len,
        "batched matrix multiplication left buffer has length {}, expected {}",
        a_len,
        expected_a_len
    );
    ensure!(
        b_len == expected_b_len,
        "batched matrix multiplication right buffer has length {}, expected {}",
        b_len,
        expected_b_len
    );
    Ok(())
}

#[cfg(test)]
mod tests;
