//! Folded layout of a matrix-valued fit: the two matrix legs as open legs of
//! the end cores, `L` cores instead of `L + 2`.

use super::{check_physical_point, col_major_gemm, MatrixShape, MatrixTCI2Result};
use crate::error::{Result, TCIError};
use tensor4all_core::{MatrixLuciScalar, Scalar};
use tensor4all_simplett::{TTScalar, Tensor3Ops};
use tensor4all_tensorbackend::Matrix;

/// One core, column-major with `left` fastest, then `site`, then `right`.
#[derive(Debug, Clone)]
struct FoldedCore<T> {
    left: usize,
    site: usize,
    right: usize,
    data: Vec<T>,
}

impl<T: Scalar> FoldedCore<T> {
    /// The `left x right` slab at physical coordinate `x`, column-major.
    fn slab(&self, x: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(self.left * self.right);
        for b in 0..self.right {
            let start = self.left * (x + self.site * b);
            out.extend_from_slice(&self.data[start..start + self.left]);
        }
        out
    }
}

/// Copies a simplett core into column-major order (`left` fastest).
fn core_data<T: TTScalar, C: Tensor3Ops<T>>(core: &C) -> Vec<T> {
    let mut data = Vec::with_capacity(core.left_dim() * core.site_dim() * core.right_dim());
    for r in 0..core.right_dim() {
        for s in 0..core.site_dim() {
            for l in 0..core.left_dim() {
                data.push(*core.get3(l, s, r));
            }
        }
    }
    data
}

/// Matrix-valued fit with the matrix legs folded into the end cores.
///
/// A [`MatrixTCI2Result`] stores the fit on the chain `[rows, d_1, .., d_L,
/// cols]` with unit boundary bonds, `L + 2` cores. Folding contracts the row
/// core into the first physical core and the column core into the last one, so
/// the first core is `rows x d_1 x r_1`, the last `r_{L-1} x d_L x cols`, and
/// the interior cores are unchanged: `L` cores with the matrix indices as open
/// legs. For `L == 1` both ends fold into the single core `rows x d_1 x cols`.
/// The boundary bonds are no longer of unit size, so a folded train is not a
/// `SimpleTensorTrain`.
///
/// Folding represents **the same function**. It draws no new samples, does no
/// truncation and no recompression; the only arithmetic is one contraction
/// per end, done once by [`MatrixTCI2Result::to_folded`]. Up to rounding in
/// those two contractions, [`Self::evaluate`] equals
/// [`MatrixTCI2Result::evaluate`].
///
/// # Storage
///
/// With `r_0` the bond between the row core and the first physical core, the
/// leading end changes from `rows * r_0 + r_0 * d_1 * r_1` numbers to
/// `rows * d_1 * r_1`; the trailing end likewise. Folding **may grow** storage
/// (for instance `rows` large with `r_0 = 1`, or `d_1` large), and shrinks it
/// when `r_0` is large relative to `d_1`. Compare
/// [`MatrixTCI2Result::retained_bytes`] with [`Self::retained_bytes`] for the
/// fit at hand. The `L + 2` layout stays the default everywhere; use the
/// folded one only when a caller asks for the matrix indices as open legs.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::{crossinterpolate2_matrix, MatrixShape, TCI2Options};
///
/// // F(x)[i, j] = (1 + i + 2 j) * (1 + x0 + x1), a non-symmetric 2 x 3 matrix.
/// let source = |points: &[Vec<usize>], out: &mut [f64]| {
///     for (p, x) in points.iter().enumerate() {
///         for j in 0..3 {
///             for i in 0..2 {
///                 out[p * 6 + i + 2 * j] = ((1 + i + 2 * j) * (1 + x[0] + x[1])) as f64;
///             }
///         }
///     }
/// };
/// let fit = crossinterpolate2_matrix::<f64, _>(
///     source,
///     MatrixShape { rows: 2, cols: 3 },
///     vec![3, 3],
///     vec![vec![1, 1, 1, 1]],
///     TCI2Options { tolerance: 1e-12, seed: Some(7), ..TCI2Options::default() },
/// )
/// .unwrap();
/// let folded = fit.to_folded().unwrap();
/// assert_eq!(folded.len(), 2); // L cores, not L + 2
/// assert_eq!(folded.shape(), MatrixShape { rows: 2, cols: 3 });
///
/// let m = folded.evaluate(&[2, 1]).unwrap();
/// let unfolded = fit.evaluate(&[2, 1]).unwrap();
/// for j in 0..3 {
///     for i in 0..2 {
///         assert!((m[[i, j]] - ((1 + i + 2 * j) * 4) as f64).abs() < 1e-9);
///         assert!((m[[i, j]] - unfolded[[i, j]]).abs() < 1e-9);
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FoldedMatrixTT<T: Scalar + TTScalar> {
    shape: MatrixShape,
    cores: Vec<FoldedCore<T>>,
}

impl<T: Scalar + TTScalar> FoldedMatrixTT<T> {
    /// Matrix shape of the represented function.
    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// Number of cores, equal to the number of physical sites `L`.
    pub fn len(&self) -> usize {
        self.cores.len()
    }

    /// Whether there are no cores (never true for a result of
    /// [`MatrixTCI2Result::to_folded`], which needs `L >= 1`).
    pub fn is_empty(&self) -> bool {
        self.cores.is_empty()
    }

    /// Local dimensions `d_1, .., d_L` of the physical sites.
    pub fn site_dims(&self) -> Vec<usize> {
        self.cores.iter().map(|c| c.site).collect()
    }

    /// Shape `[left, site, right]` of every core, in chain order.
    ///
    /// The first `left` is `rows` and the last `right` is `cols`.
    pub fn core_dims(&self) -> Vec<[usize; 3]> {
        self.cores
            .iter()
            .map(|c| [c.left, c.site, c.right])
            .collect()
    }

    /// Bytes of core data: the sum of core sizes times `size_of::<T>()`.
    ///
    /// Counts the numbers of the cores only (no `Vec` headers or allocator
    /// overhead). The matching figure for the `L + 2` layout is
    /// [`MatrixTCI2Result::retained_bytes`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::{crossinterpolate2_matrix, MatrixShape, TCI2Options};
    ///
    /// let fit = crossinterpolate2_matrix::<f64, _>(
    ///     |points: &[Vec<usize>], out: &mut [f64]| {
    ///         for (p, x) in points.iter().enumerate() {
    ///             out[p * 2] = 1.0 + x[0] as f64;
    ///             out[p * 2 + 1] = 2.0;
    ///         }
    ///     },
    ///     MatrixShape { rows: 2, cols: 1 },
    ///     vec![3],
    ///     vec![],
    ///     TCI2Options { seed: Some(1), ..TCI2Options::default() },
    /// )
    /// .unwrap();
    /// let folded = fit.to_folded().unwrap();
    /// let numbers: usize = folded.core_dims().iter().map(|[l, s, r]| l * s * r).sum();
    /// assert_eq!(folded.retained_bytes(), numbers * std::mem::size_of::<f64>());
    /// ```
    pub fn retained_bytes(&self) -> usize {
        self.cores.iter().map(|c| c.data.len()).sum::<usize>() * std::mem::size_of::<T>()
    }

    /// Evaluates the `rows x cols` matrix at the physical point `x`.
    ///
    /// Same contract as [`MatrixTCI2Result::evaluate`]: one whole-matrix pass of
    /// plain column-major loops, result column-major.
    ///
    /// # Errors
    ///
    /// Returns [`TCIError::DimensionMismatch`] when `x.len()` differs from
    /// `L` and [`TCIError::IndexOutOfBounds`] when a coordinate is outside its
    /// local dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::{crossinterpolate2_matrix, MatrixShape, TCI2Options, TCIError};
    ///
    /// let fit = crossinterpolate2_matrix::<f64, _>(
    ///     |points: &[Vec<usize>], out: &mut [f64]| {
    ///         for (p, x) in points.iter().enumerate() {
    ///             out[p * 2] = 1.0 + x[0] as f64;
    ///             out[p * 2 + 1] = 2.0;
    ///         }
    ///     },
    ///     MatrixShape { rows: 2, cols: 1 },
    ///     vec![3],
    ///     vec![],
    ///     TCI2Options { seed: Some(1), ..TCI2Options::default() },
    /// )
    /// .unwrap();
    /// let folded = fit.to_folded().unwrap(); // L == 1: a single core, 2 x 3 x 1
    /// assert_eq!(folded.core_dims(), vec![[2, 3, 1]]);
    /// let m = folded.evaluate(&[2]).unwrap();
    /// assert!((m[[0, 0]] - 3.0).abs() < 1e-9);
    /// assert!((m[[1, 0]] - 2.0).abs() < 1e-9);
    /// assert!(matches!(folded.evaluate(&[3]), Err(TCIError::IndexOutOfBounds { .. })));
    /// assert!(matches!(folded.evaluate(&[0, 0]), Err(TCIError::DimensionMismatch { .. })));
    /// ```
    pub fn evaluate(&self, x: &[usize]) -> Result<Matrix<T>> {
        check_physical_point(&self.site_dims(), x)?;
        let MatrixShape { rows, cols } = self.shape;
        let last = self.cores.len() - 1;
        if last == 0 {
            // L == 1: the single core is rows x d_1 x cols; its slab is the answer.
            return Ok(Matrix::from_col_major_vec(
                rows,
                cols,
                self.cores[0].slab(x[0]),
            ));
        }
        let first = &self.cores[0];
        let mut width = first.right;
        let mut current = first.slab(x[0]);
        for (core, &xk) in self.cores[1..last].iter().zip(&x[1..last]) {
            current = col_major_gemm(&current, rows, width, &core.slab(xk), core.right);
            width = core.right;
        }
        let tail = &self.cores[last];
        let out = col_major_gemm(&current, rows, width, &tail.slab(x[last]), cols);
        Ok(Matrix::from_col_major_vec(rows, cols, out))
    }
}

impl<T: Scalar + TTScalar + Default + MatrixLuciScalar> MatrixTCI2Result<T> {
    /// Bytes of core data of the `L + 2` layout: the sum of core sizes times
    /// `size_of::<T>()`, comparable with [`FoldedMatrixTT::retained_bytes`].
    ///
    /// Counts the numbers of the cores only. It is not the sampling cache:
    /// [`MatrixSamplingStats::retained_bytes`](super::MatrixSamplingStats::retained_bytes)
    /// measures the cached source points. Returns `0` when the site tensors are
    /// unavailable.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{AbstractTensorTrain, Tensor3Ops};
    /// use tensor4all_tensorci::{crossinterpolate2_matrix, MatrixShape, TCI2Options};
    ///
    /// let fit = crossinterpolate2_matrix::<f64, _>(
    ///     |points: &[Vec<usize>], out: &mut [f64]| {
    ///         for (p, x) in points.iter().enumerate() {
    ///             out[p * 2] = 1.0 + x[0] as f64;
    ///             out[p * 2 + 1] = 2.0;
    ///         }
    ///     },
    ///     MatrixShape { rows: 2, cols: 1 },
    ///     vec![3],
    ///     vec![],
    ///     TCI2Options { seed: Some(1), ..TCI2Options::default() },
    /// )
    /// .unwrap();
    /// let tt = fit.to_tensor_train().unwrap();
    /// let numbers: usize = tt
    ///     .site_tensors()
    ///     .iter()
    ///     .map(|c| c.left_dim() * c.site_dim() * c.right_dim())
    ///     .sum();
    /// assert_eq!(fit.retained_bytes(), numbers * std::mem::size_of::<f64>());
    /// ```
    pub fn retained_bytes(&self) -> usize {
        let tci = &self.result.tci;
        if !tci.is_site_tensors_available() {
            return 0;
        }
        let numbers: usize = (0..tci.local_dims().len())
            .map(|k| {
                let core = tci.site_tensor(k);
                core.left_dim() * core.site_dim() * core.right_dim()
            })
            .sum();
        numbers * std::mem::size_of::<T>()
    }

    /// Folds the matrix legs into the end cores: `L` cores instead of `L + 2`.
    ///
    /// Explicit and non-default; see [`FoldedMatrixTT`] for what folding is, for
    /// the storage effect (it may grow) and for the guarantee that the same
    /// function is represented. Costs one contraction per end, done here once.
    ///
    /// # Errors
    ///
    /// Returns [`TCIError::InvalidOperation`] when the site tensors are
    /// unavailable.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::{crossinterpolate2_matrix, MatrixShape, TCI2Options};
    ///
    /// let fit = crossinterpolate2_matrix::<f64, _>(
    ///     |points: &[Vec<usize>], out: &mut [f64]| {
    ///         for (p, x) in points.iter().enumerate() {
    ///             out[p * 2] = 1.0 + x[0] as f64;
    ///             out[p * 2 + 1] = 2.0;
    ///         }
    ///     },
    ///     MatrixShape { rows: 2, cols: 1 },
    ///     vec![3, 3],
    ///     vec![],
    ///     TCI2Options { seed: Some(1), ..TCI2Options::default() },
    /// )
    /// .unwrap();
    /// let folded = fit.to_folded().unwrap();
    /// assert_eq!(folded.len(), 2);
    /// let dims = folded.core_dims();
    /// assert_eq!(dims[0][0], 2); // rows is the open left leg
    /// assert_eq!(dims[1][2], 1); // cols is the open right leg
    /// ```
    pub fn to_folded(&self) -> Result<FoldedMatrixTT<T>> {
        let tci = &self.result.tci;
        if !tci.is_site_tensors_available() {
            return Err(TCIError::InvalidOperation {
                message: "site tensors are not available".to_string(),
            });
        }
        let MatrixShape { rows, cols } = self.shape;
        let n_physical = tci.local_dims().len() - 2;

        let row_core = tci.site_tensor(0);
        let mut row = vec![T::zero(); rows * row_core.right_dim()];
        for a in 0..row_core.right_dim() {
            for i in 0..rows {
                row[i + rows * a] = *row_core.get3(0, i, a);
            }
        }
        let col_core = tci.site_tensor(n_physical + 1);
        let mut col = vec![T::zero(); col_core.left_dim() * cols];
        for j in 0..cols {
            for b in 0..col_core.left_dim() {
                col[b + col_core.left_dim() * j] = *col_core.get3(b, j, 0);
            }
        }

        let mut cores = Vec::with_capacity(n_physical);
        for k in 1..=n_physical {
            let core = tci.site_tensor(k);
            let (left, site, right) = (core.left_dim(), core.site_dim(), core.right_dim());
            let mut folded = FoldedCore {
                left,
                site,
                right,
                data: core_data(core),
            };
            if k == 1 {
                // rows x r_0 times r_0 x (d_1 r_1): column-major (rows, d_1, r_1).
                folded.data = col_major_gemm(&row, rows, left, &folded.data, site * right);
                folded.left = rows;
            }
            if k == n_physical {
                // (l d_L) x r_L times r_L x cols: column-major (l, d_L, cols).
                folded.data = col_major_gemm(&folded.data, folded.left * site, right, &col, cols);
                folded.right = cols;
            }
            cores.push(folded);
        }
        Ok(FoldedMatrixTT {
            shape: self.shape,
            cores,
        })
    }
}
