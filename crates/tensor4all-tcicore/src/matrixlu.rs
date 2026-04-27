//! Rank-Revealing LU decomposition (rrLU) implementation.
//!
//! Provides [`RrLU`], a full-pivoting LU decomposition that reveals the
//! numerical rank of a matrix. The decomposition is:
//!
//! ```text
//! P_row * A * P_col = L * U
//! ```
//!
//! where `P_row`, `P_col` are permutation matrices. The rank is determined
//! by the number of pivots exceeding the tolerance thresholds in
//! [`RrLUOptions`].
//!
//! # Examples
//!
//! ```
//! use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu};
//!
//! let m = from_vec2d(vec![
//!     vec![1.0_f64, 2.0],
//!     vec![3.0, 4.0],
//! ]);
//! let lu = rrlu(&m, None).unwrap();
//! assert_eq!(lu.npivots(), 2);
//! ```

use crate::error::{MatrixCIError, Result};
use crate::matrix::{
    ncols, nrows, submatrix_argmax, swap_cols, swap_rows, transpose, zeros, Matrix,
};
use crate::scalar::Scalar;

/// Rank-Revealing LU decomposition.
///
/// Represents a matrix `A` as `P_row * A * P_col = L * U`, where `P_row`
/// and `P_col` are permutation matrices, `L` is lower-triangular, and `U`
/// is upper-triangular. One of `L` or `U` has unit diagonal, controlled by
/// the `left_orthogonal` option.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu, matrix::mat_mul};
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 2.0, 3.0],
///     vec![4.0, 5.0, 6.0],
///     vec![7.0, 8.0, 10.0],
/// ]);
///
/// let lu = rrlu(&m, None).unwrap();
/// assert_eq!(lu.npivots(), 3);
///
/// // Verify L * U reconstructs the permuted matrix
/// let l = lu.left(false);
/// let u = lu.right(false);
/// let reconstructed = mat_mul(&l, &u);
///
/// // Check reconstruction matches the permuted matrix
/// for i in 0..3 {
///     for j in 0..3 {
///         let orig_row = lu.row_permutation()[i];
///         let orig_col = lu.col_permutation()[j];
///         assert!((reconstructed[[i, j]] - m[[orig_row, orig_col]]).abs() < 1e-10);
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RrLU<T: Scalar> {
    /// Row permutation
    row_permutation: Vec<usize>,
    /// Column permutation
    col_permutation: Vec<usize>,
    /// Lower triangular matrix L
    l: Matrix<T>,
    /// Upper triangular matrix U
    u: Matrix<T>,
    /// Whether L is left-orthogonal (L has 1s on diagonal) or U is (U has 1s on diagonal)
    left_orthogonal: bool,
    /// Number of pivots
    n_pivot: usize,
    /// Last pivot error
    error: f64,
}

impl<T: Scalar> RrLU<T> {
    /// Create an empty rrLU for a matrix of given size.
    ///
    /// Used internally. Most users should call [`rrlu`] or [`rrlu_inplace`]
    /// instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::RrLU;
    ///
    /// let lu = RrLU::<f64>::new(3, 4, true);
    /// assert_eq!(lu.nrows(), 3);
    /// assert_eq!(lu.ncols(), 4);
    /// assert_eq!(lu.npivots(), 0);
    /// assert!(lu.is_left_orthogonal());
    /// ```
    pub fn new(nr: usize, nc: usize, left_orthogonal: bool) -> Self {
        Self {
            row_permutation: (0..nr).collect(),
            col_permutation: (0..nc).collect(),
            l: zeros(nr, 0),
            u: zeros(0, nc),
            left_orthogonal,
            n_pivot: 0,
            error: f64::NAN,
        }
    }

    /// Number of rows
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// assert_eq!(lu.nrows(), 3);
    /// ```
    pub fn nrows(&self) -> usize {
        nrows(&self.l)
    }

    /// Number of columns
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// assert_eq!(lu.ncols(), 3);
    /// ```
    pub fn ncols(&self) -> usize {
        ncols(&self.u)
    }

    /// Number of pivots
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// assert_eq!(lu.npivots(), 2);
    /// ```
    pub fn npivots(&self) -> usize {
        self.n_pivot
    }

    /// Row permutation
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let perm = lu.row_permutation();
    /// assert_eq!(perm.len(), 2);
    /// // Permutation is a rearrangement of 0..nrows
    /// let mut sorted = perm.to_vec();
    /// sorted.sort();
    /// assert_eq!(sorted, vec![0, 1]);
    /// ```
    pub fn row_permutation(&self) -> &[usize] {
        &self.row_permutation
    }

    /// Column permutation
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let perm = lu.col_permutation();
    /// assert_eq!(perm.len(), 2);
    /// let mut sorted = perm.to_vec();
    /// sorted.sort();
    /// assert_eq!(sorted, vec![0, 1]);
    /// ```
    pub fn col_permutation(&self) -> &[usize] {
        &self.col_permutation
    }

    /// Get row indices (selected pivots)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu, RrLUOptions};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, Some(RrLUOptions { max_rank: 1, ..Default::default() })).unwrap();
    /// let rows = lu.row_indices();
    /// assert_eq!(rows.len(), 1);
    /// assert!(rows[0] < 2);
    /// ```
    pub fn row_indices(&self) -> Vec<usize> {
        self.row_permutation[0..self.n_pivot].to_vec()
    }

    /// Get column indices (selected pivots)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu, RrLUOptions};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, Some(RrLUOptions { max_rank: 1, ..Default::default() })).unwrap();
    /// let cols = lu.col_indices();
    /// assert_eq!(cols.len(), 1);
    /// assert!(cols[0] < 2);
    /// ```
    pub fn col_indices(&self) -> Vec<usize> {
        self.col_permutation[0..self.n_pivot].to_vec()
    }

    /// Get left matrix (optionally permuted)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu, matrix::mat_mul};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    ///
    /// // Unpermuted: L * U reconstructs the row/col-permuted matrix
    /// let l = lu.left(false);
    /// let u = lu.right(false);
    /// let prod = mat_mul(&l, &u);
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         let ri = lu.row_permutation()[i];
    ///         let cj = lu.col_permutation()[j];
    ///         assert!((prod[[i, j]] - m[[ri, cj]]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    pub fn left(&self, permute: bool) -> Matrix<T> {
        if permute {
            let mut result = zeros(nrows(&self.l), ncols(&self.l));
            for (new_i, &old_i) in self.row_permutation.iter().enumerate() {
                for j in 0..ncols(&self.l) {
                    result[[old_i, j]] = self.l[[new_i, j]];
                }
            }
            result
        } else {
            self.l.clone()
        }
    }

    /// Get right matrix (optionally permuted)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu, matrix::mat_mul};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let l = lu.left(false);
    /// let u = lu.right(false);
    /// assert_eq!(u.nrows(), lu.npivots());
    /// assert_eq!(u.ncols(), lu.ncols());
    /// // L * U reconstructs the permuted matrix
    /// let prod = mat_mul(&l, &u);
    /// assert!((prod[[0, 0]] - m[[lu.row_permutation()[0], lu.col_permutation()[0]]]).abs() < 1e-10);
    /// ```
    pub fn right(&self, permute: bool) -> Matrix<T> {
        if permute {
            let mut result = zeros(nrows(&self.u), ncols(&self.u));
            for i in 0..nrows(&self.u) {
                for (new_j, &old_j) in self.col_permutation.iter().enumerate() {
                    result[[i, old_j]] = self.u[[i, new_j]];
                }
            }
            result
        } else {
            self.u.clone()
        }
    }

    /// Get diagonal elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let d = lu.diag();
    /// assert_eq!(d.len(), lu.npivots());
    /// // Diagonal elements are non-zero for full-rank matrices
    /// for &val in &d {
    ///     assert!(val.abs() > 1e-14);
    /// }
    /// ```
    pub fn diag(&self) -> Vec<T> {
        let n = self.n_pivot;
        if self.left_orthogonal {
            (0..n).map(|i| self.u[[i, i]]).collect()
        } else {
            (0..n).map(|i| self.l[[i, i]]).collect()
        }
    }

    /// Get pivot errors
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let errs = lu.pivot_errors();
    /// // One entry per pivot plus the final residual
    /// assert_eq!(errs.len(), lu.npivots() + 1);
    /// // Errors are non-negative
    /// for &e in &errs {
    ///     assert!(e >= 0.0);
    /// }
    /// ```
    pub fn pivot_errors(&self) -> Vec<f64> {
        let mut errors: Vec<f64> = self.diag().iter().map(|d| f64::sqrt(d.abs_sq())).collect();
        errors.push(self.error);
        errors
    }

    /// Get last pivot error
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// // Full-rank decomposition has zero residual
    /// assert_eq!(lu.last_pivot_error(), 0.0);
    /// ```
    pub fn last_pivot_error(&self) -> f64 {
        self.error
    }

    /// Transpose the decomposition
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let lu_t = lu.transpose();
    /// assert_eq!(lu_t.nrows(), lu.ncols());
    /// assert_eq!(lu_t.ncols(), lu.nrows());
    /// assert_eq!(lu_t.npivots(), lu.npivots());
    /// assert_eq!(lu_t.is_left_orthogonal(), !lu.is_left_orthogonal());
    /// ```
    pub fn transpose(&self) -> RrLU<T> {
        RrLU {
            row_permutation: self.col_permutation.clone(),
            col_permutation: self.row_permutation.clone(),
            l: transpose(&self.u),
            u: transpose(&self.l),
            left_orthogonal: !self.left_orthogonal,
            n_pivot: self.n_pivot,
            error: self.error,
        }
    }

    /// Check if left-orthogonal (L has 1s on diagonal)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu, RrLUOptions};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    ///
    /// let lu = rrlu(&m, None).unwrap();
    /// assert!(lu.is_left_orthogonal()); // default
    ///
    /// let lu2 = rrlu(&m, Some(RrLUOptions {
    ///     left_orthogonal: false, ..Default::default()
    /// })).unwrap();
    /// assert!(!lu2.is_left_orthogonal());
    /// ```
    pub fn is_left_orthogonal(&self) -> bool {
        self.left_orthogonal
    }
}

/// Options for rank-revealing LU decomposition.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::RrLUOptions;
///
/// // Default: rel_tol = 1e-14, no absolute tolerance, no rank limit
/// let opts = RrLUOptions::default();
/// assert_eq!(opts.rel_tol, 1e-14);
/// assert_eq!(opts.abs_tol, 0.0);
/// assert!(opts.left_orthogonal);
///
/// // Limit rank to 5
/// let opts = RrLUOptions { max_rank: 5, ..Default::default() };
/// assert_eq!(opts.max_rank, 5);
/// ```
#[derive(Debug, Clone)]
pub struct RrLUOptions {
    /// Maximum rank
    pub max_rank: usize,
    /// Relative tolerance
    pub rel_tol: f64,
    /// Absolute tolerance
    pub abs_tol: f64,
    /// Left orthogonal (L has 1s on diagonal) or right orthogonal (U has 1s)
    pub left_orthogonal: bool,
}

impl Default for RrLUOptions {
    fn default() -> Self {
        Self {
            max_rank: usize::MAX,
            rel_tol: 1e-14,
            abs_tol: 0.0,
            left_orthogonal: true,
        }
    }
}

/// Perform in-place rank-revealing LU decomposition.
///
/// The input matrix `a` is modified in place. Use [`rrlu`] for a
/// non-destructive version.
///
/// # Errors
///
/// Returns [`MatrixCIError::NaNEncountered`]
/// if NaN values appear in the L or U factors.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu_inplace, RrLUOptions};
///
/// let mut m = from_vec2d(vec![
///     vec![1.0_f64, 2.0],
///     vec![3.0, 4.0],
/// ]);
/// let lu = rrlu_inplace(&mut m, Some(RrLUOptions { max_rank: 1, ..Default::default() })).unwrap();
/// assert_eq!(lu.npivots(), 1);
/// ```
pub fn rrlu_inplace<T: Scalar>(a: &mut Matrix<T>, options: Option<RrLUOptions>) -> Result<RrLU<T>> {
    let opts = options.unwrap_or_default();
    let nr = nrows(a);
    let nc = ncols(a);

    let mut lu = RrLU::new(nr, nc, opts.left_orthogonal);
    let max_rank = opts.max_rank.min(nr).min(nc);
    let mut max_error = 0.0f64;

    while lu.n_pivot < max_rank {
        let k = lu.n_pivot;

        if k >= nr || k >= nc {
            break;
        }

        // Find pivot with maximum absolute value in submatrix
        let (pivot_row, pivot_col, pivot_val) = submatrix_argmax(a, k..nr, k..nc);

        let pivot_abs = f64::sqrt(pivot_val.abs_sq());
        lu.error = pivot_abs;

        // Check stopping criteria (but add at least 1 pivot)
        if lu.n_pivot > 0 && (pivot_abs < opts.rel_tol * max_error || pivot_abs < opts.abs_tol) {
            break;
        }

        // Guard against tiny pivots to prevent NaN from division. A caller that
        // sets both tolerances to zero is requesting a non-truncating
        // decomposition, so only an exactly zero pivot stops the factorization.
        let min_pivot_abs = if opts.rel_tol == 0.0 && opts.abs_tol == 0.0 {
            0.0
        } else {
            f64::EPSILON
        };
        if pivot_abs <= min_pivot_abs {
            if lu.n_pivot == 0 {
                // First pivot is near-zero: the matrix is effectively zero
                lu.error = pivot_abs;
            }
            break;
        }

        max_error = max_error.max(pivot_abs);

        // Swap rows and columns
        if pivot_row != k {
            swap_rows(a, k, pivot_row);
            lu.row_permutation.swap(k, pivot_row);
        }
        if pivot_col != k {
            swap_cols(a, k, pivot_col);
            lu.col_permutation.swap(k, pivot_col);
        }

        let pivot = a[[k, k]];

        // Eliminate
        if opts.left_orthogonal {
            // Scale column below pivot
            for i in (k + 1)..nr {
                let val = a[[i, k]] / pivot;
                a[[i, k]] = val;
            }
        } else {
            // Scale row to the right of pivot
            for j in (k + 1)..nc {
                let val = a[[k, j]] / pivot;
                a[[k, j]] = val;
            }
        }

        // Update submatrix: A[k+1:, k+1:] -= A[k+1:, k] * A[k, k+1:]
        for i in (k + 1)..nr {
            for j in (k + 1)..nc {
                let x = a[[i, k]];
                let y = a[[k, j]];
                let old = a[[i, j]];
                a[[i, j]] = old - x * y;
            }
        }

        lu.n_pivot += 1;
    }

    // Extract L and U
    let n = lu.n_pivot;

    // L is lower triangular part
    let mut l = zeros(nr, n);
    for i in 0..nr {
        for j in 0..n.min(i + 1) {
            l[[i, j]] = a[[i, j]];
        }
    }

    // U is upper triangular part
    let mut u = zeros(n, nc);
    for i in 0..n {
        for j in i..nc {
            u[[i, j]] = a[[i, j]];
        }
    }

    // Set diagonal to 1 for the orthogonal factor
    if opts.left_orthogonal {
        for i in 0..n {
            l[[i, i]] = T::one();
        }
    } else {
        for i in 0..n {
            u[[i, i]] = T::one();
        }
    }

    // Check for NaNs (return error instead of panicking)
    for i in 0..nrows(&l) {
        for j in 0..ncols(&l) {
            if l[[i, j]].is_nan() {
                return Err(MatrixCIError::NaNEncountered {
                    matrix: "L".to_string(),
                });
            }
        }
    }
    for i in 0..nrows(&u) {
        for j in 0..ncols(&u) {
            if u[[i, j]].is_nan() {
                return Err(MatrixCIError::NaNEncountered {
                    matrix: "U".to_string(),
                });
            }
        }
    }

    // Set error to 0 if full rank
    if n >= nr.min(nc) {
        lu.error = 0.0;
    }

    lu.l = l;
    lu.u = u;

    Ok(lu)
}

/// Perform rank-revealing LU decomposition (non-destructive).
///
/// Clones the input matrix and calls [`rrlu_inplace`].
///
/// # Errors
///
/// Returns [`MatrixCIError::NaNEncountered`]
/// if NaN values appear in the L or U factors.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrixlu::rrlu};
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 0.0],
///     vec![0.0, 2.0],
/// ]);
/// let lu = rrlu(&m, None).unwrap();
/// assert_eq!(lu.npivots(), 2);
/// assert_eq!(lu.nrows(), 2);
/// assert_eq!(lu.ncols(), 2);
/// ```
pub fn rrlu<T: Scalar>(a: &Matrix<T>, options: Option<RrLUOptions>) -> Result<RrLU<T>> {
    let mut a_copy = a.clone();
    rrlu_inplace(&mut a_copy, options)
}

/// Convert L matrix to solve L * X = B given pivot matrix P
///
/// Modifies `c` in place so that the columns satisfy the triangular
/// system defined by `p`. The matrix `c` must have at least `nrows(p)`
/// columns.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrixlu::cols_to_l_matrix};
///
/// // Upper-triangular P (2x2)
/// let p = from_vec2d(vec![
///     vec![2.0_f64, 1.0],
///     vec![0.0, 3.0],
/// ]);
/// // c has 3 rows, 2 columns (ncols >= nrows(p))
/// let mut c = from_vec2d(vec![
///     vec![4.0_f64, 5.0],
///     vec![6.0, 9.0],
///     vec![8.0, 7.0],
/// ]);
/// cols_to_l_matrix(&mut c, &p, true);
/// // After processing: c[:,0] was divided by p[0,0]=2
/// assert!((c[[0, 0]] - 2.0).abs() < 1e-10);
/// assert!((c[[1, 0]] - 3.0).abs() < 1e-10);
/// assert!((c[[2, 0]] - 4.0).abs() < 1e-10);
/// ```
pub fn cols_to_l_matrix<T: Scalar>(c: &mut Matrix<T>, p: &Matrix<T>, _left_orthogonal: bool) {
    let n = nrows(p);

    for k in 0..n {
        let pivot = p[[k, k]];
        // c[:, k] /= pivot
        for i in 0..nrows(c) {
            let val = c[[i, k]] / pivot;
            c[[i, k]] = val;
        }

        // c[:, k+1:] -= c[:, k] * p[k, k+1:]
        for j in (k + 1)..ncols(c) {
            let p_kj = p[[k, j]];
            for i in 0..nrows(c) {
                let c_ik = c[[i, k]];
                let old = c[[i, j]];
                c[[i, j]] = old - c_ik * p_kj;
            }
        }
    }
}

/// Convert R matrix to solve X * U = B given pivot matrix P
///
/// Modifies `r` in place so that the rows satisfy the triangular
/// system defined by `p`. The matrix `r` must have at least `nrows(p)`
/// rows.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrixlu::rows_to_u_matrix};
///
/// // Lower-triangular P (2x2)
/// let p = from_vec2d(vec![
///     vec![2.0_f64, 0.0],
///     vec![1.0, 3.0],
/// ]);
/// // r has 2 rows (nrows >= nrows(p)), 3 columns
/// let mut r = from_vec2d(vec![
///     vec![4.0_f64, 6.0, 8.0],
///     vec![5.0, 9.0, 7.0],
/// ]);
/// rows_to_u_matrix(&mut r, &p, true);
/// // After processing: r[0,:] was divided by p[0,0]=2
/// assert!((r[[0, 0]] - 2.0).abs() < 1e-10);
/// assert!((r[[0, 1]] - 3.0).abs() < 1e-10);
/// assert!((r[[0, 2]] - 4.0).abs() < 1e-10);
/// ```
pub fn rows_to_u_matrix<T: Scalar>(r: &mut Matrix<T>, p: &Matrix<T>, _left_orthogonal: bool) {
    let n = nrows(p);

    for k in 0..n {
        let pivot = p[[k, k]];
        // r[k, :] /= pivot
        for j in 0..ncols(r) {
            let val = r[[k, j]] / pivot;
            r[[k, j]] = val;
        }

        // r[k+1:, :] -= p[k+1:, k] * r[k, :]
        for i in (k + 1)..nrows(r) {
            let p_ik = p[[i, k]];
            for j in 0..ncols(r) {
                let r_kj = r[[k, j]];
                let old = r[[i, j]];
                r[[i, j]] = old - p_ik * r_kj;
            }
        }
    }
}

/// Solve LU * x = b
///
/// Given factors `L` and `U` such that `A = L * U`, solves `A * x = b`
/// via forward and back substitution.
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{from_vec2d, matrixlu::{rrlu, solve_lu}};
///
/// let m = from_vec2d(vec![vec![2.0_f64, 1.0], vec![1.0, 3.0]]);
/// let lu = rrlu(&m, None).unwrap();
/// let l = lu.left(false);
/// let u = lu.right(false);
///
/// // b = [5, 10] => solve Ax = b in permuted coordinates
/// let b = from_vec2d(vec![
///     vec![m[[lu.row_permutation()[0], 0]] * 1.0 + m[[lu.row_permutation()[0], 1]] * 2.0],
///     vec![m[[lu.row_permutation()[1], 0]] * 1.0 + m[[lu.row_permutation()[1], 1]] * 2.0],
/// ]);
/// let x = solve_lu(&l, &u, &b).unwrap();
/// // Solution in permuted col order
/// assert!((x[[lu.col_permutation().iter().position(|&c| c == 0).unwrap(), 0]] - 1.0).abs() < 1e-10);
/// assert!((x[[lu.col_permutation().iter().position(|&c| c == 1).unwrap(), 0]] - 2.0).abs() < 1e-10);
/// ```
pub fn solve_lu<T: Scalar>(l: &Matrix<T>, u: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    let _n1 = nrows(l);
    let n2 = ncols(l);
    let n3 = ncols(u);
    let m = ncols(b);

    // Solve L * y = b (forward substitution)
    let mut y: Matrix<T> = zeros(n2, m);
    for i in 0..n2 {
        for k in 0..m {
            let mut sum = b[[i, k]];
            for j in 0..i {
                sum = sum - l[[i, j]] * y[[j, k]];
            }
            let diag = l[[i, i]];
            y[[i, k]] = sum / diag;
        }
    }

    // Solve U * x = y (back substitution)
    let mut x: Matrix<T> = zeros(n3, m);
    for i in (0..n3).rev() {
        for k in 0..m {
            let mut sum = y[[i, k]];
            for j in (i + 1)..n3 {
                sum = sum - u[[i, j]] * x[[j, k]];
            }
            let diag = u[[i, i]];
            x[[i, k]] = sum / diag;
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests;
