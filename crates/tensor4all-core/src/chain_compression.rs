//! Two-sweep compression of chains of rank-3 cores.
//!
//! [`compress_chain_lu_mut`] and [`compress_chain_lu_on_session`] compress a
//! chain with pivoted LU factorizations. [`compress_chain_cores_mut`] runs the
//! same two sweeps with a caller-supplied factorization and multiplication,
//! and [`ChainCore`] adapts a storage type to the sweeps.

use std::collections::TryReserveError;

use tensor4all_tensorbackend::{
    explicit::Session, mat_mul, Matrix, MatrixMulError, MatrixShapeError,
};

use crate::col_major_array::ColMajorArrayError;
use crate::{
    rrlu_mut, ColMajorArray, ColMajorArrayRef, MatrixCIError, MatrixLUCI, MatrixLuciScalar,
    RrLUOptions,
};

/// Relative pivot threshold used by the forward sweep, and by the backward
/// sweep when [`ChainLuOptions::tolerance`] is zero.
const PIVOT_FLOOR: f64 = 1e-14;

/// Failure to validate or compress a chain of rank-3 cores.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{compress_chain_lu_mut, ChainCompressionError, ChainLuOptions, ColMajorArray};
/// let core = ColMajorArray::new(vec![1.0_f64], vec![1, 1])?;
/// let error = compress_chain_lu_mut(&mut [core], &ChainLuOptions::default()).unwrap_err();
/// assert!(matches!(error, ChainCompressionError::CoreRank { site: 0, rank: 2 }));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ChainCompressionError {
    /// A core has a rank other than three.
    #[error("core at site {site} must have three axes, got {rank}")]
    CoreRank {
        /// Zero-based site of the invalid core.
        site: usize,
        /// Number of axes of the supplied core.
        rank: usize,
    },
    /// The right bond of a site disagrees with the next site's left bond.
    #[error(
        "core at site {site} has right bond {right}, but the next core has left bond {next_left}"
    )]
    BondMismatch {
        /// Zero-based site to the left of the mismatched bond.
        site: usize,
        /// Dimension of the bond at `site`.
        right: usize,
        /// Dimension of the same bond at `site + 1`.
        next_left: usize,
    },
    /// The left boundary bond must have dimension one.
    #[error("First tensor must have left dimension 1")]
    LeftBoundary,
    /// The right boundary bond must have dimension one.
    #[error("Last tensor must have right dimension 1")]
    RightBoundary,
    /// The pivot tolerance must be finite and nonnegative.
    #[error("LU compression tolerance must be finite and nonnegative, got {0}")]
    InvalidTolerance(f64),
    /// A rank cap of zero would discard every bond.
    #[error("LU compression max_bond_dim must be positive when specified")]
    ZeroBondCap,
    /// Fusing two core axes would overflow `usize`.
    #[error("unfolding core shape {0:?} overflows usize")]
    ShapeOverflow([usize; 3]),
    /// A flat buffer does not hold a whole number of unfolding blocks.
    #[error("flat buffer of length {len} is not a whole number of {block}-element blocks")]
    BufferLength {
        /// Length of the supplied buffer.
        len: usize,
        /// Element count of one block, or zero when a block dimension is zero.
        block: usize,
    },
    /// The flat buffer could not be allocated.
    #[error("cannot allocate a compression buffer: {0}")]
    Allocation(#[from] TryReserveError),
    /// A column-major array could not be constructed.
    #[error(transparent)]
    Array(#[from] ColMajorArrayError),
    /// An unfolding or factor had incompatible matrix dimensions.
    #[error(transparent)]
    MatrixShape(#[from] MatrixShapeError),
    /// A matrix factorization failed.
    #[error(transparent)]
    Factorization(#[from] MatrixCIError),
    /// A matrix factor could not be absorbed into the adjacent core.
    #[error("{direction} compression multiply failed: {source}")]
    Multiply {
        /// Sweep direction in which the multiplication failed.
        direction: &'static str,
        /// Underlying backend multiplication failure.
        #[source]
        source: MatrixMulError,
    },
}

/// Pivoted factorization applied at each bond by [`compress_chain_lu_mut`].
///
/// # Examples
///
/// ```
/// use tensor4all_core::ChainLuFactorization;
/// assert_eq!(ChainLuFactorization::default(), ChainLuFactorization::RrLu);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChainLuFactorization {
    /// Rank-revealing LU with full pivoting ([`rrlu_mut`]).
    #[default]
    RrLu,
    /// LU-based cross interpolation ([`MatrixLUCI`]), which selects the same
    /// pivots through backend triangular solves.
    Luci,
}

/// Truncation settings for [`compress_chain_lu_mut`] and
/// [`compress_chain_lu_on_session`].
///
/// The forward sweep never truncates: it keeps every pivot above a `1e-14`
/// relative threshold. These settings apply only to the backward sweep, where
/// `tolerance` and `max_bond_dim` both limit the retained rank of each bond
/// and the smaller limit wins. `normalize_error` selects whether `tolerance`
/// is relative to the largest pivot of that bond or absolute.
///
/// | Field | Default | Meaning |
/// |-------|---------|---------|
/// | `tolerance` | `1e-12` | Pivot threshold; zero selects the `1e-14` relative floor |
/// | `max_bond_dim` | `None` | Positive cap on each internal bond, or uncapped |
/// | `normalize_error` | `true` | Relative (`true`) or absolute (`false`) `tolerance` |
/// | `factorization` | `RrLu` | Pivoted factorization at each bond |
///
/// When in doubt, keep the defaults: `1e-12` is close to lossless, values from
/// `1e-8` to `1e-6` suit most scientific work, and larger values compress more
/// aggressively.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{ChainLuFactorization, ChainLuOptions};
/// let options = ChainLuOptions {
///     tolerance: 1e-8,
///     max_bond_dim: Some(16),
///     ..ChainLuOptions::default()
/// };
/// assert!(options.normalize_error);
/// assert_eq!(options.factorization, ChainLuFactorization::RrLu);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainLuOptions {
    /// Finite nonnegative pivot threshold for the backward sweep. Zero selects
    /// the `1e-14` relative floor regardless of `normalize_error`.
    pub tolerance: f64,
    /// Positive cap on each internal bond in the backward sweep. `None` leaves
    /// the rank limited only by `tolerance`.
    pub max_bond_dim: Option<usize>,
    /// Whether a positive `tolerance` is relative to the largest pivot of each
    /// bond (`true`) or an absolute pivot threshold (`false`).
    pub normalize_error: bool,
    /// Pivoted factorization applied at each bond.
    pub factorization: ChainLuFactorization,
}

impl Default for ChainLuOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_bond_dim: None,
            normalize_error: true,
            factorization: ChainLuFactorization::RrLu,
        }
    }
}

impl ChainLuOptions {
    fn validate(&self) -> Result<(), ChainCompressionError> {
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(ChainCompressionError::InvalidTolerance(self.tolerance));
        }
        if self.max_bond_dim == Some(0) {
            return Err(ChainCompressionError::ZeroBondCap);
        }
        Ok(())
    }

    fn rrlu_options(&self, left_orthogonal: bool) -> RrLUOptions {
        if left_orthogonal {
            return RrLUOptions {
                max_bond_dim: usize::MAX,
                rel_tol: PIVOT_FLOOR,
                abs_tol: 0.0,
                left_orthogonal,
            };
        }
        let (rel_tol, abs_tol) = if self.tolerance == 0.0 {
            (PIVOT_FLOOR, 0.0)
        } else if self.normalize_error {
            (self.tolerance, 0.0)
        } else {
            (0.0, self.tolerance)
        };
        RrLUOptions {
            max_bond_dim: self.max_bond_dim.unwrap_or(usize::MAX),
            rel_tol,
            abs_tol,
            left_orthogonal,
        }
    }

    fn factorize<T: MatrixLuciScalar>(
        &self,
        mut matrix: Matrix<T>,
        left_orthogonal: bool,
    ) -> Result<(Matrix<T>, Matrix<T>), ChainCompressionError> {
        let options = self.rrlu_options(left_orthogonal);
        match self.factorization {
            ChainLuFactorization::RrLu => {
                let lu = rrlu_mut(&mut matrix, Some(options))?;
                Ok((lu.left(true), lu.right(true)))
            }
            ChainLuFactorization::Luci => {
                let luci = MatrixLUCI::from_matrix(&matrix, Some(options))?;
                Ok((luci.left(), luci.right()))
            }
        }
    }
}

/// Compress a chain of rank-3 cores in place with pivoted LU.
///
/// Each core has shape `[left_bond, site, right_bond]`. Neighboring bond
/// dimensions must match and the outer bonds must have dimension one. An
/// empty or single-core chain is validated and left unchanged. The forward
/// sweep keeps every pivot above a `1e-14` relative threshold; the backward
/// sweep truncates according to `options`. Products use the process-global
/// CPU backend; [`compress_chain_lu_on_session`] runs them on a caller-held
/// session instead.
///
/// # Errors
///
/// Returns [`ChainCompressionError::InvalidTolerance`] for a negative or
/// nonfinite tolerance and [`ChainCompressionError::ZeroBondCap`] for
/// `max_bond_dim: Some(0)`, before touching any core. Invalid core shapes and
/// factorization, multiplication or allocation failures return the other
/// [`ChainCompressionError`] variants, converted into the storage type's error.
/// If a numerical operation fails, bonds updated before the failure keep
/// their compressed values.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{compress_chain_lu_mut, ChainLuOptions, ColMajorArray};
/// let mut cores = [
///     ColMajorArray::new(vec![2.0_f64, 2.0], vec![1, 1, 2])?,
///     ColMajorArray::new(vec![3.0, 3.0], vec![2, 1, 1])?,
/// ];
/// compress_chain_lu_mut(&mut cores, &ChainLuOptions::default())?;
/// assert_eq!(cores[0].shape(), &[1, 1, 1]);
/// assert_eq!(cores[0].data()[0] * cores[1].data()[0], 12.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compress_chain_lu_mut<T: MatrixLuciScalar, C: ChainCore<T>>(
    cores: &mut [C],
    options: &ChainLuOptions,
) -> Result<(), C::Error> {
    options.validate()?;
    compress_chain_cores_mut(
        cores,
        &mut (),
        |(), matrix, left_orthogonal| Ok(options.factorize(matrix, left_orthogonal)?),
        |(), left, right| mat_mul(left, right),
    )
}

/// Compress a chain of rank-3 cores in place with pivoted LU, multiplying on
/// a caller-held session.
///
/// Identical to [`compress_chain_lu_mut`], including validation and retained
/// ranks, except that every matrix product runs on `session`. The pivoted
/// factorizations do not use the backend.
///
/// # Errors
///
/// Returns [`ChainCompressionError::InvalidTolerance`] for a negative or
/// nonfinite tolerance and [`ChainCompressionError::ZeroBondCap`] for
/// `max_bond_dim: Some(0)`, before touching any core. Returns
/// [`ChainCompressionError::Multiply`] when a product fails on `session`.
/// Invalid core shapes and factorization or allocation failures return the
/// other [`ChainCompressionError`] variants, converted into the storage
/// type's error. If a numerical operation fails, bonds updated before the
/// failure keep their compressed values.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{compress_chain_lu_on_session, ChainLuOptions, ColMajorArray};
/// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
/// let mut backend = CpuBackend::with_threads(1)?;
/// let mut session = Session::new(backend.engine_mut());
/// let mut cores = [
///     ColMajorArray::new(vec![2.0_f64, 2.0], vec![1, 1, 2])?,
///     ColMajorArray::new(vec![3.0, 3.0], vec![2, 1, 1])?,
/// ];
/// compress_chain_lu_on_session(&mut session, &mut cores, &ChainLuOptions::default())?;
/// assert_eq!(cores[1].shape(), &[1, 1, 1]);
/// assert_eq!(cores[0].data()[0] * cores[1].data()[0], 12.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compress_chain_lu_on_session<T: MatrixLuciScalar, C: ChainCore<T>>(
    session: &mut Session<'_>,
    cores: &mut [C],
    options: &ChainLuOptions,
) -> Result<(), C::Error> {
    options.validate()?;
    compress_chain_cores_mut(
        cores,
        session,
        |_, matrix, left_orthogonal| Ok(options.factorize(matrix, left_orthogonal)?),
        |session, left, right| session.mat_mul(left, right),
    )
}

/// Storage adapter that lets the chain sweeps read and rebuild rank-3 cores.
///
/// [`ColMajorArray`] implements it directly; other crates implement it for
/// their own core types so that [`compress_chain_lu_mut`] and
/// [`compress_chain_cores_mut`] can update them in place.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{ChainCore, ColMajorArray};
/// let core = <ColMajorArray<f64> as ChainCore<f64>>::from_col_major([1, 2, 1], vec![1.0, 2.0])?;
/// let view = ChainCore::<f64>::view(&core)?;
/// assert_eq!(view.shape(), &[1, 2, 1]);
/// assert_eq!(view.data(), &[1.0, 2.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait ChainCore<T>: Sized {
    /// Error returned when reading or rebuilding a core, and by the sweeps.
    type Error: From<ChainCompressionError>;

    /// Borrow the shape and flat data in column-major order.
    ///
    /// # Errors
    /// Returns the adapter's error for noncompact or device-only storage,
    /// or a shape mismatch between the declared extents and borrowed data.
    fn view(&self) -> Result<ColMajorArrayRef<'_, T>, Self::Error>;

    /// Rebuild a core from column-major data.
    ///
    /// # Errors
    /// Returns the adapter's error for a shape mismatch with the data length
    /// or a shape or stride overflow.
    fn from_col_major(shape: [usize; 3], data: Vec<T>) -> Result<Self, Self::Error>;
}

impl<T> ChainCore<T> for ColMajorArray<T> {
    type Error = ChainCompressionError;

    fn view(&self) -> Result<ColMajorArrayRef<'_, T>, Self::Error> {
        Ok(self.as_ref())
    }

    fn from_col_major(shape: [usize; 3], data: Vec<T>) -> Result<Self, Self::Error> {
        Ok(Self::new(data, shape.to_vec())?)
    }
}

/// Run the two compression sweeps with a caller-supplied factorization.
///
/// The cores are validated as in [`compress_chain_lu_mut`]. The forward sweep
/// then visits each bond from left to right, unfolds the left core to a
/// `(left_bond * site) x bond` matrix and calls `factorize(state, matrix,
/// true)`. The backward sweep visits the bonds from right to left, unfolds the
/// right core to a `bond x (site * right_bond)` matrix and calls
/// `factorize(state, matrix, false)`. `factorize` returns factors `(L, R)`
/// with `L * R` approximating the matrix; their shared dimension becomes the
/// new bond. `multiply` absorbs the factor that does not replace the visited
/// core into its neighbor. `state` is passed to both closures, for example a
/// backend session.
///
/// Unfoldings index rows as `l * site + s` and columns as `s * right + r`.
///
/// # Errors
///
/// Returns invalid core shapes, factor shapes that disagree with the
/// unfolding, multiplication and allocation failures as
/// [`ChainCompressionError`] converted into `C::Error`, and passes through
/// any error from `factorize`. If a step fails, bonds updated before the
/// failure keep their new values.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{compress_chain_cores_mut, rrlu_mut, ChainCompressionError, ColMajorArray};
/// use tensor4all_tensorbackend::mat_mul;
/// // A rank-one chain stored with bond dimension two.
/// let mut cores = [
///     ColMajorArray::new(vec![1.0_f64, 1.0, 2.0, 2.0], vec![1, 2, 2])?,
///     ColMajorArray::new(vec![1.0, 1.0, 3.0, 3.0], vec![2, 2, 1])?,
/// ];
/// compress_chain_cores_mut(
///     &mut cores,
///     &mut (),
///     |(), mut matrix, _| {
///         let lu = rrlu_mut(&mut matrix, None).map_err(ChainCompressionError::from)?;
///         Ok((lu.left(true), lu.right(true)))
///     },
///     |(), left, right| mat_mul(left, right),
/// )?;
/// assert_eq!(cores[0].shape(), &[1, 2, 1]);
/// assert_eq!(cores[1].shape(), &[1, 2, 1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compress_chain_cores_mut<T: Copy, C: ChainCore<T>, S>(
    cores: &mut [C],
    state: &mut S,
    mut factorize: impl FnMut(&mut S, Matrix<T>, bool) -> Result<(Matrix<T>, Matrix<T>), C::Error>,
    mut multiply: impl FnMut(&mut S, &Matrix<T>, &Matrix<T>) -> Result<Matrix<T>, MatrixMulError>,
) -> Result<(), C::Error> {
    let n = cores.len();
    let mut first_left = 1;
    let mut previous_right = 1;
    for (site, core) in cores.iter().enumerate() {
        let [left, _, right] = core_shape(core.view()?, site)?;
        if site == 0 {
            first_left = left;
        } else {
            check_bond(site - 1, previous_right, left)?;
        }
        previous_right = right;
    }
    if first_left != 1 {
        return Err(ChainCompressionError::LeftBoundary.into());
    }
    if previous_right != 1 {
        return Err(ChainCompressionError::RightBoundary.into());
    }

    let bonds = 0..n.saturating_sub(1);
    let forward = bonds.clone().map(|site| (site, true));
    let backward = bonds.rev().map(|site| (site, false));
    for (site, left_orthogonal) in forward.chain(backward) {
        let left = cores[site].view()?;
        let right = cores[site + 1].view()?;
        let [l, left_site, bond] = core_shape(left, site)?;
        let [next_bond, right_site, r] = core_shape(right, site + 1)?;
        check_bond(site, bond, next_bond)?;
        let rows = fused_dim(l, left_site, [l, left_site, bond])?;
        let cols = fused_dim(right_site, r, [bond, right_site, r])?;
        let left_matrix = || -> Result<_, ChainCompressionError> {
            Ok(Matrix::try_from_col_major_vec(
                rows,
                bond,
                swap_blocks(left.data(), 1, l, left_site)?,
            )?)
        };
        let right_matrix = || -> Result<_, ChainCompressionError> {
            Ok(Matrix::try_from_col_major_vec(
                bond,
                cols,
                swap_blocks(right.data(), bond, right_site, r)?,
            )?)
        };
        let matrix = if left_orthogonal {
            left_matrix()?
        } else {
            right_matrix()?
        };
        let (matrix_rows, matrix_cols) = (matrix.nrows(), matrix.ncols());
        let (left_factor, right_factor) = factorize(state, matrix, left_orthogonal)?;
        let rank = left_factor.ncols();
        matrix_data(&left_factor, matrix_rows, rank)?;
        matrix_data(&right_factor, rank, matrix_cols)?;

        let (left, right) = if left_orthogonal {
            let right = multiply(state, &right_factor, &right_matrix()?).map_err(|source| {
                ChainCompressionError::Multiply {
                    direction: "left-to-right",
                    source,
                }
            })?;
            (left_factor, right)
        } else {
            let left = multiply(state, &left_matrix()?, &left_factor).map_err(|source| {
                ChainCompressionError::Multiply {
                    direction: "right-to-left",
                    source,
                }
            })?;
            (left, right_factor)
        };
        let left = C::from_col_major(
            [l, left_site, rank],
            swap_blocks(matrix_data(&left, rows, rank)?, 1, left_site, l)?,
        )?;
        let right = C::from_col_major(
            [rank, right_site, r],
            swap_blocks(matrix_data(&right, rank, cols)?, rank, r, right_site)?,
        )?;
        cores[site] = left;
        cores[site + 1] = right;
    }
    Ok(())
}

fn core_shape<T>(
    core: ColMajorArrayRef<'_, T>,
    site: usize,
) -> Result<[usize; 3], ChainCompressionError> {
    core.shape()
        .try_into()
        .map_err(|_| ChainCompressionError::CoreRank {
            site,
            rank: core.ndim(),
        })
}

fn check_bond(site: usize, right: usize, next_left: usize) -> Result<(), ChainCompressionError> {
    if right != next_left {
        return Err(ChainCompressionError::BondMismatch {
            site,
            right,
            next_left,
        });
    }
    Ok(())
}

fn fused_dim(a: usize, b: usize, shape: [usize; 3]) -> Result<usize, ChainCompressionError> {
    a.checked_mul(b)
        .ok_or(ChainCompressionError::ShapeOverflow(shape))
}

fn matrix_data<T>(
    matrix: &Matrix<T>,
    rows: usize,
    cols: usize,
) -> Result<&[T], ChainCompressionError> {
    if matrix.nrows() != rows || matrix.ncols() != cols {
        return Err(MatrixCIError::DimensionMismatch {
            expected_rows: rows,
            expected_cols: cols,
            actual_rows: matrix.nrows(),
            actual_cols: matrix.ncols(),
        }
        .into());
    }
    Ok(matrix.as_col_major_slice())
}

// The fused row index is l * site + s and the fused column index is
// s * right + r. Exact ties in pivot magnitude are broken in this order, so it
// determines which pivots the sweeps retain; core storage is column-major, so
// each unfolding swaps the two fused axes. `data` is a sequence of blocks of
// `inner * first * second` elements; within each block, chunks of `inner`
// elements are reordered from `first`-fastest to `second`-fastest.
fn swap_blocks<T: Copy>(
    data: &[T],
    inner: usize,
    first: usize,
    second: usize,
) -> Result<Vec<T>, ChainCompressionError> {
    let mut result = Vec::new();
    result.try_reserve_exact(data.len())?;
    if data.is_empty() {
        return Ok(result);
    }
    let block = inner
        .checked_mul(first)
        .and_then(|stride| stride.checked_mul(second))
        .filter(|&block| block != 0 && data.len().is_multiple_of(block))
        .ok_or(ChainCompressionError::BufferLength {
            len: data.len(),
            block: inner.saturating_mul(first).saturating_mul(second),
        })?;
    if first == 1 || second == 1 {
        result.extend_from_slice(data);
        return Ok(result);
    }
    let stride = inner * first;
    for chunk in data.chunks_exact(block) {
        for k in 0..first {
            for j in 0..second {
                let start = k * inner + j * stride;
                result.extend_from_slice(&chunk[start..start + inner]);
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
