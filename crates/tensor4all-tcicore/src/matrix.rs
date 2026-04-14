//! Dense row-major matrix type and utility functions.
//!
//! [`Matrix<T>`] is a simple dense 2D matrix in row-major layout, indexed
//! by `m[[row, col]]`. It is used throughout the TCI infrastructure for
//! pivot block computations, cross interpolation factors, and dense
//! submatrix extraction.
//!
//! # Examples
//!
//! ```
//! use tensor4all_tcicore::{Matrix, from_vec2d, matrix};
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

use crate::scalar::Scalar;
use num_traits::{One, Zero};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;
use std::ops::{Index, IndexMut};

/// A dense 2D matrix in row-major layout.
///
/// Access elements with `m[[row, col]]` syntax. Data is stored contiguously
/// in row-major order.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::Matrix;
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
    /// Create a matrix from raw row-major data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != nrows * ncols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::Matrix;
    ///
    /// let m = Matrix::from_raw_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(m[[0, 0]], 1.0);
    /// assert_eq!(m[[0, 1]], 2.0);
    /// assert_eq!(m[[1, 0]], 3.0);
    /// assert_eq!(m[[1, 1]], 4.0);
    /// ```
    pub fn from_raw_vec(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), nrows * ncols);
        Self { data, nrows, ncols }
    }

    /// View the underlying row-major data as a contiguous slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::Matrix;
    ///
    /// let m = Matrix::from_raw_vec(1, 3, vec![10, 20, 30]);
    /// assert_eq!(m.as_slice(), &[10, 20, 30]);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        &self.data
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
    /// use tensor4all_tcicore::Matrix;
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
    /// use tensor4all_tcicore::Matrix;
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
        &self.data[idx[0] * self.ncols + idx[1]]
    }
}

impl<T> IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
        &mut self.data[idx[0] * self.ncols + idx[1]]
    }
}

/// Create a zeros matrix with given dimensions.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::matrix::zeros;
///
/// let m: tensor4all_tcicore::Matrix<f64> = zeros(2, 3);
/// assert_eq!(m[[0, 0]], 0.0);
/// assert_eq!(m.nrows(), 2);
/// assert_eq!(m.ncols(), 3);
/// ```
pub fn zeros<T: Clone + Zero>(nrows: usize, ncols: usize) -> Matrix<T> {
    Matrix::zeros(nrows, ncols)
}

/// Create an `n x n` identity matrix.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::matrix::eye;
///
/// let m: tensor4all_tcicore::Matrix<f64> = eye(3);
/// assert_eq!(m[[0, 0]], 1.0);
/// assert_eq!(m[[1, 1]], 1.0);
/// assert_eq!(m[[0, 1]], 0.0);
/// assert_eq!(m[[2, 0]], 0.0);
/// ```
pub fn eye<T: Clone + Zero + One>(n: usize) -> Matrix<T> {
    let mut m = zeros(n, n);
    for i in 0..n {
        m[[i, i]] = T::one();
    }
    m
}

/// Create a matrix from a 2D vector (row-major).
///
/// Each inner `Vec` is one row.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::from_vec2d;
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
    let mut m = zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            m[[i, j]] = data[i][j].clone();
        }
    }
    m
}

/// Get number of rows.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrix::nrows};
///
/// let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
/// assert_eq!(nrows(&m), 3);
/// ```
pub fn nrows<T>(m: &Matrix<T>) -> usize {
    m.nrows
}

/// Get number of columns.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrix::ncols};
///
/// let m = from_vec2d(vec![vec![1.0, 2.0, 3.0]]);
/// assert_eq!(ncols(&m), 3);
/// ```
pub fn ncols<T>(m: &Matrix<T>) -> usize {
    m.ncols
}

/// Get a row as a vector.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrix::get_row};
///
/// let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
/// assert_eq!(get_row(&m, 0), vec![1.0, 2.0]);
/// assert_eq!(get_row(&m, 1), vec![3.0, 4.0]);
/// ```
pub fn get_row<T: Clone>(m: &Matrix<T>, i: usize) -> Vec<T> {
    (0..m.ncols).map(|j| m[[i, j]].clone()).collect()
}

/// Get a column as a vector.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrix::get_col};
///
/// let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
/// assert_eq!(get_col(&m, 0), vec![1.0, 3.0]);
/// assert_eq!(get_col(&m, 1), vec![2.0, 4.0]);
/// ```
pub fn get_col<T: Clone>(m: &Matrix<T>, j: usize) -> Vec<T> {
    (0..m.nrows).map(|i| m[[i, j]].clone()).collect()
}

/// Get a submatrix by selecting specific rows and columns.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrix::submatrix};
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
    let mut result = zeros(rows.len(), cols.len());
    for (ri, &r) in rows.iter().enumerate() {
        for (ci, &c) in cols.iter().enumerate() {
            result[[ri, ci]] = m[[r, c]].clone();
        }
    }
    result
}

/// Append a column to the right of a matrix.
///
/// # Panics
///
/// Panics if `col.len() != m.nrows()`.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrix::append_col};
///
/// let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
/// let m2 = append_col(&m, &[5.0, 6.0]);
/// assert_eq!(m2.ncols(), 3);
/// assert_eq!(m2[[0, 2]], 5.0);
/// assert_eq!(m2[[1, 2]], 6.0);
/// ```
pub fn append_col<T: Clone + Zero>(m: &Matrix<T>, col: &[T]) -> Matrix<T> {
    let nr = m.nrows;
    let nc = m.ncols;
    assert_eq!(col.len(), nr);

    let mut result = zeros(nr, nc + 1);
    for i in 0..nr {
        for j in 0..nc {
            result[[i, j]] = m[[i, j]].clone();
        }
        result[[i, nc]] = col[i].clone();
    }
    result
}

/// Append a row to the bottom of a matrix.
///
/// # Panics
///
/// Panics if `row.len() != m.ncols()`.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrix::append_row};
///
/// let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
/// let m2 = append_row(&m, &[5.0, 6.0]);
/// assert_eq!(m2.nrows(), 3);
/// assert_eq!(m2[[2, 0]], 5.0);
/// assert_eq!(m2[[2, 1]], 6.0);
/// ```
pub fn append_row<T: Clone + Zero>(m: &Matrix<T>, row: &[T]) -> Matrix<T> {
    let nr = m.nrows;
    let nc = m.ncols;
    assert_eq!(row.len(), nc);

    let mut result = zeros(nr + 1, nc);
    for i in 0..nr {
        for j in 0..nc {
            result[[i, j]] = m[[i, j]].clone();
        }
    }
    for j in 0..nc {
        result[[nr, j]] = row[j].clone();
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
/// use tensor4all_tcicore::{from_vec2d, matrix::swap_rows};
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
        let idx_a = a * m.ncols + j;
        let idx_b = b * m.ncols + j;
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
/// use tensor4all_tcicore::{from_vec2d, matrix::swap_cols};
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
        let idx_a = i * m.ncols + a;
        let idx_b = i * m.ncols + b;
        m.data.swap(idx_a, idx_b);
    }
}

/// Transpose the matrix.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrix::transpose};
///
/// let m = from_vec2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
/// let mt = transpose(&m);
/// assert_eq!(mt.nrows(), 3);
/// assert_eq!(mt.ncols(), 2);
/// assert_eq!(mt[[0, 0]], 1.0);
/// assert_eq!(mt[[2, 1]], 6.0);
/// ```
pub fn transpose<T: Clone + Zero>(m: &Matrix<T>) -> Matrix<T> {
    let mut result = zeros(m.ncols, m.nrows);
    for i in 0..m.nrows {
        for j in 0..m.ncols {
            result[[j, i]] = m[[i, j]].clone();
        }
    }
    result
}

// Scalar trait is now defined in crate::scalar module

/// Calculates A * B^{-1} using Gaussian elimination.
///
/// # Panics
///
/// Panics if the number of columns of `a` does not match the dimensions of `b`,
/// or if `b` is not square.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrix::a_times_b_inv};
///
/// let a = from_vec2d(vec![vec![2.0_f64, 0.0], vec![0.0, 4.0]]);
/// let b = from_vec2d(vec![vec![1.0, 0.0], vec![0.0, 2.0]]);
/// let result = a_times_b_inv(&a, &b);
/// assert!((result[[0, 0]] - 2.0).abs() < 1e-10);
/// assert!((result[[1, 1]] - 2.0).abs() < 1e-10);
/// ```
pub fn a_times_b_inv<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    let n = ncols(a);
    assert_eq!(nrows(b), n);
    assert_eq!(ncols(b), n);

    // Solve XB = A by solving B'X' = A'
    let bt = transpose(b);
    let at = transpose(a);
    let xt = solve_linear_system(&bt, &at);
    transpose(&xt)
}

/// Calculates A^{-1} * B using Gaussian elimination.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrix::a_inv_times_b};
///
/// let a = from_vec2d(vec![vec![2.0_f64, 0.0], vec![0.0, 4.0]]);
/// let b = from_vec2d(vec![vec![6.0, 0.0], vec![0.0, 8.0]]);
/// let result = a_inv_times_b(&a, &b);
/// assert!((result[[0, 0]] - 3.0).abs() < 1e-10);
/// assert!((result[[1, 1]] - 2.0).abs() < 1e-10);
/// ```
pub fn a_inv_times_b<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    let bt = transpose(b);
    let at = transpose(a);
    let result = a_times_b_inv(&bt, &at);
    transpose(&result)
}

/// Solve linear system AX = B using Gaussian elimination with partial pivoting
#[allow(clippy::needless_range_loop)]
fn solve_linear_system<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    let n = nrows(a);
    assert_eq!(ncols(a), n);
    assert_eq!(nrows(b), n);
    let m = ncols(b);

    // Create augmented matrix [A | B]
    let mut aug: Vec<Vec<T>> = (0..n)
        .map(|i| {
            let mut row = Vec::with_capacity(n + m);
            for j in 0..n {
                row.push(a[[i, j]]);
            }
            for j in 0..m {
                row.push(b[[i, j]]);
            }
            row
        })
        .collect();

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val: f64 = aug[k][k].abs_sq();
        for i in (k + 1)..n {
            let val: f64 = aug[i][k].abs_sq();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Swap rows
        if max_idx != k {
            aug.swap(k, max_idx);
        }

        let pivot = aug[k][k];
        if pivot.abs_sq() < T::epsilon() {
            continue;
        }

        // Eliminate below
        for i in (k + 1)..n {
            let factor = aug[i][k] / pivot;
            for j in k..(n + m) {
                aug[i][j] = aug[i][j] - factor * aug[k][j];
            }
        }
    }

    // Back substitution
    let mut x: Vec<Vec<T>> = vec![vec![T::zero(); m]; n];
    for i in (0..n).rev() {
        for j in 0..m {
            let mut sum = aug[i][n + j];
            for k in (i + 1)..n {
                sum = sum - aug[i][k] * x[k][j];
            }
            let diag = aug[i][i];
            if diag.abs_sq() > T::epsilon() {
                x[i][j] = sum / diag;
            }
        }
    }

    from_vec2d(x)
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
/// use tensor4all_tcicore::{from_vec2d, matrix::submatrix_argmax};
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
pub fn submatrix_argmax<T: Scalar>(
    a: &Matrix<T>,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) -> (usize, usize, T) {
    assert!(!rows.is_empty(), "rows must not be empty");
    assert!(!cols.is_empty(), "cols must not be empty");

    let mut max_val: f64 = a[[rows.start, cols.start]].abs_sq();
    let mut max_row = rows.start;
    let mut max_col = cols.start;

    for r in rows {
        for c in cols.clone() {
            let val: f64 = a[[r, c]].abs_sq();
            if val > max_val {
                max_val = val;
                max_row = r;
                max_col = c;
            }
        }
    }

    (max_row, max_col, a[[max_row, max_col]])
}

/// Select a random subset of up to `n` elements from a slice.
///
/// If `n >= set.len()`, returns at most `set.len()` elements (a shuffled
/// subset). Returns an empty vector when the set is empty or `n` is zero.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::matrix::random_subset;
/// use rand::SeedableRng;
///
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let items = vec![10, 20, 30, 40, 50];
/// let sub = random_subset(&items, 3, &mut rng);
/// assert_eq!(sub.len(), 3);
/// // All selected elements come from the original set
/// for &x in &sub {
///     assert!(items.contains(&x));
/// }
/// // Requesting more than available returns at most set.len()
/// let all = random_subset(&items, 100, &mut rng);
/// assert_eq!(all.len(), 5);
/// ```
pub fn random_subset<T: Clone, R: Rng>(set: &[T], n: usize, rng: &mut R) -> Vec<T> {
    let n = n.min(set.len());
    if n == 0 {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..set.len()).collect();
    indices.shuffle(rng);
    indices.truncate(n);
    indices.into_iter().map(|i| set[i].clone()).collect()
}

/// Set difference: elements in `set` that are not in `exclude`.
///
/// Preserves the order of elements in `set`.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::matrix::set_diff;
///
/// let result = set_diff(&[0, 1, 2, 3, 4], &[1, 3]);
/// assert_eq!(result, vec![0, 2, 4]);
/// ```
pub fn set_diff(set: &[usize], exclude: &[usize]) -> Vec<usize> {
    let exclude_set: HashSet<usize> = exclude.iter().copied().collect();
    set.iter()
        .copied()
        .filter(|x| !exclude_set.contains(x))
        .collect()
}

/// Dot product of two vectors.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::matrix::dot;
///
/// let a = [1.0_f64, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// assert!((dot(&a, &b) - 32.0).abs() < 1e-10);
/// ```
pub fn dot<T: Scalar>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x * y)
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

fn row_major_to_col_major<T: Copy>(data: &[T], nrows: usize, ncols: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(data.len());
    for col in 0..ncols {
        for row in 0..nrows {
            out.push(data[row * ncols + col]);
        }
    }
    out
}

fn col_major_to_row_major<T: Copy>(data: &[T], nrows: usize, ncols: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(data.len());
    for row in 0..nrows {
        for col in 0..ncols {
            out.push(data[col * nrows + row]);
        }
    }
    out
}

macro_rules! impl_blas_mul {
    ($($t:ty),*) => {
        $(
        impl BlasMul for $t {
            fn blas_mat_mul(a: &Matrix<Self>, b: &Matrix<Self>) -> Matrix<Self> {
                use tenferro_einsum::typed_eager_einsum;
                use tenferro_tensor::TypedTensor;
                use tensor4all_tensorbackend::with_default_backend;

                let m = a.nrows();
                let k = a.ncols();
                let n = b.ncols();
                assert_eq!(b.nrows(), k);

                let a_tensor = TypedTensor::<$t>::from_vec(
                    vec![m, k],
                    row_major_to_col_major(a.as_slice(), m, k),
                );
                let b_tensor = TypedTensor::<$t>::from_vec(
                    vec![k, n],
                    row_major_to_col_major(b.as_slice(), k, n),
                );
                let c = with_default_backend(|backend| {
                    typed_eager_einsum(backend, &[&a_tensor, &b_tensor], "ij,jk->ik")
                })
                .expect("einsum failed");
                let c_data = col_major_to_row_major(c.as_slice(), m, n);
                Matrix::from_raw_vec(m, n, c_data)
            }
        }
        )*
    };
}

impl_blas_mul!(f64, f32, num_complex::Complex64, num_complex::Complex32);

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
/// use tensor4all_tcicore::{from_vec2d, matrix::mat_mul};
///
/// let a = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
/// let b = from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
/// let c = mat_mul(&a, &b);
/// assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
/// assert!((c[[0, 1]] - 22.0).abs() < 1e-10);
/// assert!((c[[1, 0]] - 43.0).abs() < 1e-10);
/// assert!((c[[1, 1]] - 50.0).abs() < 1e-10);
/// ```
pub fn mat_mul<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    T::blas_mat_mul(a, b)
}

#[cfg(test)]
mod tests;
