//! Two-sweep compression of column-major chain cores.

use std::collections::TryReserveError;

use tensor4all_tensorbackend::{mat_mul, Matrix, MatrixMulError, MatrixShapeError};

use crate::col_major_array::ColMajorArrayError;
use crate::{rrlu_mut, ColMajorArray, ColMajorArrayRef, MatrixCIError, RrLUOptions, Scalar};

/// Failure to validate or compress a chain of column-major cores.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{compress_chain_lu_mut, ChainCompressionError, ColMajorArray};
/// let core = ColMajorArray::new(vec![1.0_f64], vec![1, 1])?;
/// let error = compress_chain_lu_mut(&mut [core], 0.0, None, true).unwrap_err();
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
    /// The rank-revealing LU tolerance must be finite.
    #[error("LU compression tolerance must be finite, got {0}")]
    InvalidTolerance(f64),
    /// Fusing two core axes would overflow `usize`.
    #[error("unfolding core shape {0:?} overflows usize")]
    ShapeOverflow([usize; 3]),
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

/// Compress column-major chain cores in place with rank-revealing LU.
///
/// Each core has shape `[left_bond, site, right_bond]` with the left bond
/// varying fastest. Neighboring bond dimensions must match and the outer
/// bonds must have dimension one. An empty or single-core chain is unchanged.
///
/// The forward sweep uses a `1e-14` relative pivot threshold with unlimited
/// rank. The backward sweep uses `tolerance` relative to the largest pivot
/// when `normalize_error` is true, or as an absolute threshold otherwise.
/// A nonpositive `tolerance` selects the `1e-14` relative threshold in either
/// mode. `max_bond_dim` caps the retained rank of each internal bond. `None`
/// leaves the rank uncapped. Products use the default CPU backend.
///
/// # Errors
///
/// Returns [`ChainCompressionError`] for invalid core shapes, nonfinite
/// tolerance, or a matrix factorization, multiplication, or allocation failure.
/// If a numerical operation fails, previously updated bonds remain compressed.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{compress_chain_lu_mut, ColMajorArray};
/// let mut cores = [
///     ColMajorArray::new(vec![2.0_f64, 2.0], vec![1, 1, 2])?,
///     ColMajorArray::new(vec![3.0, 3.0], vec![2, 1, 1])?,
/// ];
/// compress_chain_lu_mut(&mut cores, 0.0, None, true)?;
/// assert_eq!(cores[0].shape(), &[1, 1, 1]);
/// assert_eq!(cores[0].data()[0] * cores[1].data()[0], 12.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compress_chain_lu_mut<T: Scalar>(
    cores: &mut [ColMajorArray<T>],
    tolerance: f64,
    max_bond_dim: Option<usize>,
    normalize_error: bool,
) -> Result<(), ChainCompressionError> {
    if !tolerance.is_finite() {
        return Err(ChainCompressionError::InvalidTolerance(tolerance));
    }
    compress_chain_lu_cores_mut(cores, tolerance, max_bond_dim, normalize_error)
}

/// Storage adapter used by SimpleTT and raw column-major cores.
#[doc(hidden)]
pub trait ChainCore<T>: Sized {
    /// Error returned when reading or rebuilding a core.
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

/// Shared LU compression for the raw-core and SimpleTT storage types.
#[doc(hidden)]
pub fn compress_chain_lu_cores_mut<T: Scalar, C: ChainCore<T>>(
    cores: &mut [C],
    tolerance: f64,
    max_bond_dim: Option<usize>,
    normalize_error: bool,
) -> Result<(), C::Error> {
    compress_chain_cores_mut(
        cores,
        &mut (),
        |(), mut matrix, left_orthogonal| {
            let options =
                chain_rrlu_options(tolerance, max_bond_dim, normalize_error, left_orthogonal);
            let lu = rrlu_mut(&mut matrix, Some(options)).map_err(ChainCompressionError::from)?;
            Ok((lu.left(true), lu.right(true)))
        },
        |(), left, right| mat_mul(left, right),
    )
}

/// Select the historical LU/CI pivot threshold and orientation.
#[doc(hidden)]
pub fn chain_rrlu_options(
    tolerance: f64,
    max_bond_dim: Option<usize>,
    normalize_error: bool,
    left_orthogonal: bool,
) -> RrLUOptions {
    if left_orthogonal {
        return RrLUOptions::default();
    }
    let (rel_tol, abs_tol) = if tolerance > 0.0 && !normalize_error {
        (0.0, tolerance)
    } else if tolerance > 0.0 {
        (tolerance, 0.0)
    } else {
        (1e-14, 0.0)
    };
    RrLUOptions {
        max_bond_dim: max_bond_dim.unwrap_or(usize::MAX),
        rel_tol,
        abs_tol,
        left_orthogonal,
    }
}

/// Run the common sweep with caller-selected factorization and multiplication.
#[doc(hidden)]
pub fn compress_chain_cores_mut<T: Scalar, C: ChainCore<T>, S>(
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

// Preserve the historical unfolding row l*site+s and column s*right+r.
// Swap the two fused axes over chunks of the unchanged inner dimensions.
fn swap_blocks<T: Copy>(
    data: &[T],
    inner: usize,
    first: usize,
    second: usize,
) -> Result<Vec<T>, ChainCompressionError> {
    let mut result = Vec::new();
    result.try_reserve_exact(data.len())?;
    if data.is_empty() || first == 1 || second == 1 {
        result.extend_from_slice(data);
        return Ok(result);
    }
    // INVARIANT: the nonempty core or unfolding has a validated element count.
    let stride = inner * first;
    for block in data.chunks_exact(stride * second) {
        for k in 0..first {
            for j in 0..second {
                let start = k * inner + j * stride;
                result.extend_from_slice(&block[start..start + inner]);
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
