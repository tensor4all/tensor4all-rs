//! Factorization methods for MPO tensors
//!
//! This module provides various factorization methods (SVD, RSVD, LU, CI)
//! for compressing and reshaping MPO tensors.

use super::error::{MPOError, Result};
use super::Matrix2;
use mdarray::DSlice;
use num_complex::ComplexFloat;
use tensor4all_tensorbackend::svd_backend;

/// Factorization method to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FactorizeMethod {
    /// Singular Value Decomposition
    #[default]
    SVD,
    /// Randomized SVD (faster for large matrices)
    RSVD,
    /// LU decomposition with rank-revealing pivoting
    LU,
    /// Cross Interpolation
    CI,
}

/// Options for factorization
#[derive(Debug, Clone)]
pub struct FactorizeOptions {
    /// Factorization method to use
    pub method: FactorizeMethod,
    /// Tolerance for truncation
    pub tolerance: f64,
    /// Maximum rank (bond dimension) after factorization
    pub max_rank: usize,
    /// Whether to return the left factor as left-orthogonal
    pub left_orthogonal: bool,
    /// Number of random projections for RSVD (parameter q)
    pub rsvd_q: usize,
    /// Oversampling parameter for RSVD (parameter p)
    pub rsvd_p: usize,
}

impl Default for FactorizeOptions {
    fn default() -> Self {
        Self {
            method: FactorizeMethod::SVD,
            tolerance: 1e-12,
            max_rank: usize::MAX,
            left_orthogonal: true,
            rsvd_q: 2,
            rsvd_p: 10,
        }
    }
}

/// Result of factorization
#[derive(Debug, Clone)]
pub struct FactorizeResult<T> {
    /// Left factor matrix (m x rank)
    pub left: Matrix2<T>,
    /// Right factor matrix (rank x n)
    pub right: Matrix2<T>,
    /// New rank (number of columns in left / rows in right)
    pub rank: usize,
    /// Discarded weight (for error estimation)
    pub discarded: f64,
}

/// Trait bounds for SVD-compatible scalars
pub trait SVDScalar:
    crate::traits::TTScalar
    + ComplexFloat
    + faer_traits::ComplexField
    + Default
    + From<<Self as ComplexFloat>::Real>
    + 'static
where
    <Self as ComplexFloat>::Real: Into<f64>,
{
}

impl<T> SVDScalar for T
where
    T: crate::traits::TTScalar
        + ComplexFloat
        + faer_traits::ComplexField
        + Default
        + From<<T as ComplexFloat>::Real>
        + 'static,
    <T as ComplexFloat>::Real: Into<f64>,
{
}

/// Factorize a matrix into left and right factors
///
/// Returns (L, R, rank, discarded) where:
/// - L: left factor matrix (rows x rank)
/// - R: right factor matrix (rank x cols)
/// - rank: the resulting rank after truncation
/// - discarded: the discarded weight (for error estimation)
///
/// The original matrix M ≈ L @ R
///
/// Note: Only SVD method is fully supported. LU and CI require additional
/// traits and should use `factorize_lu` directly.
pub fn factorize<T: SVDScalar>(
    matrix: &Matrix2<T>,
    options: &FactorizeOptions,
) -> Result<FactorizeResult<T>>
where
    <T as ComplexFloat>::Real: Into<f64>,
{
    match options.method {
        FactorizeMethod::SVD => factorize_svd(matrix, options),
        FactorizeMethod::RSVD => factorize_rsvd(matrix, options),
        FactorizeMethod::LU | FactorizeMethod::CI => {
            // For LU/CI, fall back to SVD for now
            // Full LU/CI support requires matrixci::Scalar trait
            factorize_svd(matrix, options)
        }
    }
}

// Use the shared matrix2_zeros from the parent module
use super::matrix2_zeros;

/// Factorize using SVD
fn factorize_svd<T: SVDScalar>(
    matrix: &Matrix2<T>,
    options: &FactorizeOptions,
) -> Result<FactorizeResult<T>>
where
    <T as ComplexFloat>::Real: Into<f64>,
{
    let m = matrix.dim(0);
    let n = matrix.dim(1);

    if m == 0 || n == 0 {
        return Err(MPOError::FactorizationError {
            message: "Cannot factorize empty matrix".to_string(),
        });
    }

    // Clone matrix for SVD (it may be modified)
    let mut a = matrix.clone();

    // Compute SVD using tensorbackend (handles faer/lapack switching internally)
    let a_slice: &mut DSlice<T, 2> = a.as_mut();
    let svd_result = svd_backend(a_slice).map_err(|e| MPOError::FactorizationError {
        message: format!("SVD computation failed: {:?}", e),
    })?;

    let u = svd_result.u;
    let s = svd_result.s;
    let vt = svd_result.vt;

    // Determine rank based on tolerance and max_rank
    let min_dim = m.min(n);
    let mut rank = 0;
    let mut total_weight: f64 = 0.0;

    // Sum all squared singular values for total weight
    // Singular values are stored in first row: s[[0, i]] (LAPACK-style convention)
    for i in 0..min_dim {
        let sv: f64 = s[[0, i]].abs().into();
        total_weight += sv * sv;
    }

    // Find rank by keeping singular values above tolerance
    let mut kept_weight: f64 = 0.0;
    for i in 0..min_dim {
        if rank >= options.max_rank {
            break;
        }
        let sv: f64 = s[[0, i]].abs().into();
        if sv < options.tolerance {
            break;
        }
        kept_weight += sv * sv;
        rank += 1;
    }

    // Ensure at least rank 1
    rank = rank.max(1);

    // Calculate discarded weight
    let discarded: f64 = if total_weight > 0.0 {
        1.0 - kept_weight / total_weight
    } else {
        0.0
    };

    // Build result matrices
    let mut left: Matrix2<T> = matrix2_zeros(m, rank);
    let mut right: Matrix2<T> = matrix2_zeros(rank, n);

    if options.left_orthogonal {
        // Left = U[:, :rank], Right = diag(S[:rank]) * Vt[:rank, :]
        //
        // NOTE: For complex matrices, mdarray-linalg-faer currently returns V^T
        // instead of V^H. This is a known issue and will be fixed upstream.
        // See: mdarray_linalg_faer_complex_svd_issue.md
        // Once fixed, the code below should work correctly for both real and complex.
        for i in 0..m {
            for j in 0..rank {
                left[[i, j]] = u[[i, j]];
            }
        }
        for i in 0..rank {
            // Singular values are stored in first row: s[[0, i]] (LAPACK-style convention)
            let sv = s[[0, i]];
            for j in 0..n {
                right[[i, j]] = sv * vt[[i, j]];
            }
        }
    } else {
        // Left = U[:, :rank] * diag(S[:rank]), Right = Vt[:rank, :]
        for i in 0..m {
            for j in 0..rank {
                // Singular values are stored in first row: s[[0, j]] (LAPACK-style convention)
                let sv = s[[0, j]];
                left[[i, j]] = u[[i, j]] * sv;
            }
        }
        for i in 0..rank {
            for j in 0..n {
                right[[i, j]] = vt[[i, j]];
            }
        }
    }

    Ok(FactorizeResult {
        left,
        right,
        rank,
        discarded,
    })
}

/// Factorize using randomized SVD
fn factorize_rsvd<T: SVDScalar>(
    _matrix: &Matrix2<T>,
    _options: &FactorizeOptions,
) -> Result<FactorizeResult<T>>
where
    <T as ComplexFloat>::Real: Into<f64>,
{
    // TODO: Implement RSVD-based factorization
    Err(MPOError::FactorizationError {
        message: "RSVD factorization not yet implemented".to_string(),
    })
}

/// Factorize using LU decomposition
///
/// This function requires the matrixci::Scalar trait.
/// Use this directly when you need LU-based factorization.
pub fn factorize_lu<T>(
    matrix: &Matrix2<T>,
    options: &FactorizeOptions,
) -> Result<FactorizeResult<T>>
where
    T: SVDScalar + matrixci::util::Scalar,
    <T as ComplexFloat>::Real: Into<f64>,
{
    use matrixci::{AbstractMatrixCI, MatrixLUCI, RrLUOptions};

    let m = matrix.dim(0);
    let n = matrix.dim(1);

    // Convert DTensor to matrixci::Matrix (temporary until matrixci migration)
    let mut mat_ci: matrixci::util::Matrix<T> = matrixci::util::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            mat_ci[[i, j]] = matrix[[i, j]];
        }
    }

    let lu_options = RrLUOptions {
        max_rank: options.max_rank,
        rel_tol: options.tolerance,
        abs_tol: 0.0,
        left_orthogonal: options.left_orthogonal,
    };

    let luci = MatrixLUCI::from_matrix(&mat_ci, Some(lu_options));
    let left_ci = luci.left();
    let right_ci = luci.right();
    let rank = luci.rank().max(1);

    // Convert back to DTensor
    let left_m = matrixci::util::nrows(&left_ci);
    let left_n = matrixci::util::ncols(&left_ci);
    let mut left: Matrix2<T> = matrix2_zeros(left_m, left_n);
    for i in 0..left_m {
        for j in 0..left_n {
            left[[i, j]] = left_ci[[i, j]];
        }
    }

    let right_m = matrixci::util::nrows(&right_ci);
    let right_n = matrixci::util::ncols(&right_ci);
    let mut right: Matrix2<T> = matrix2_zeros(right_m, right_n);
    for i in 0..right_m {
        for j in 0..right_n {
            right[[i, j]] = right_ci[[i, j]];
        }
    }

    Ok(FactorizeResult {
        left,
        right,
        rank,
        discarded: 0.0,
    })
}

/// Factorize using Cross Interpolation
///
/// This function requires the matrixci::Scalar trait.
/// Use this directly when you need CI-based factorization.
pub fn factorize_ci<T>(
    matrix: &Matrix2<T>,
    options: &FactorizeOptions,
) -> Result<FactorizeResult<T>>
where
    T: SVDScalar + matrixci::util::Scalar,
    <T as ComplexFloat>::Real: Into<f64>,
{
    // CI uses the same LUCI implementation as LU
    factorize_lu(matrix, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorize_svd() {
        let mut matrix: Matrix2<f64> = matrix2_zeros(4, 3);
        for i in 0..4 {
            for j in 0..3 {
                matrix[[i, j]] = (i * 3 + j + 1) as f64;
            }
        }

        let options = FactorizeOptions {
            method: FactorizeMethod::SVD,
            tolerance: 1e-12,
            max_rank: 10,
            left_orthogonal: true,
            ..Default::default()
        };

        let result = factorize(&matrix, &options).unwrap();
        assert!(result.rank >= 1);
        assert!(result.rank <= 3); // Max rank is min(4, 3) = 3

        // Verify reconstruction: L @ R ≈ original
        let m = 4;
        let n = 3;
        for i in 0..m {
            for j in 0..n {
                let mut reconstructed = 0.0;
                for k in 0..result.rank {
                    reconstructed += result.left[[i, k]] * result.right[[k, j]];
                }
                let original = matrix[[i, j]];
                assert!(
                    (reconstructed - original).abs() < 1e-10,
                    "Reconstruction failed at [{}, {}]: {} vs {}",
                    i,
                    j,
                    reconstructed,
                    original
                );
            }
        }
    }

    #[test]
    fn test_factorize_lu() {
        let mut matrix: Matrix2<f64> = matrix2_zeros(4, 3);
        for i in 0..4 {
            for j in 0..3 {
                matrix[[i, j]] = (i * 3 + j) as f64;
            }
        }

        let options = FactorizeOptions {
            method: FactorizeMethod::LU,
            tolerance: 1e-12,
            max_rank: 10,
            left_orthogonal: true,
            ..Default::default()
        };

        let result = factorize_lu(&matrix, &options).unwrap();
        assert!(result.rank >= 1);
        assert!(result.rank <= 3); // Max rank is min(4, 3) = 3
    }

    #[test]
    fn test_factorize_with_truncation() {
        // Create a rank-2 matrix
        let mut matrix: Matrix2<f64> = matrix2_zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                // Rank-2: outer product of [1,2,3,4] and [1,1,1,1] + [1,1,1,1] and [1,2,3,4]
                matrix[[i, j]] = (i + 1) as f64 + (j + 1) as f64;
            }
        }

        let options = FactorizeOptions {
            method: FactorizeMethod::SVD,
            tolerance: 1e-10,
            max_rank: 2,
            left_orthogonal: true,
            ..Default::default()
        };

        let result = factorize(&matrix, &options).unwrap();
        assert!(result.rank <= 2);
    }

    #[test]
    fn test_factorize_svd_complex64() {
        use num_complex::Complex64;

        let mut matrix: Matrix2<Complex64> = matrix2_zeros(4, 3);
        for i in 0..4 {
            for j in 0..3 {
                // Create complex values with both real and imaginary parts
                let re = (i * 3 + j + 1) as f64;
                let im = ((i + j) % 3) as f64 * 0.5;
                matrix[[i, j]] = Complex64::new(re, im);
            }
        }

        let options = FactorizeOptions {
            method: FactorizeMethod::SVD,
            tolerance: 1e-12,
            max_rank: 10,
            left_orthogonal: true,
            ..Default::default()
        };

        let result = factorize(&matrix, &options).unwrap();
        assert!(result.rank >= 1);
        assert!(result.rank <= 3); // Max rank is min(4, 3) = 3

        // Verify reconstruction: L @ R ≈ original
        let m = 4;
        let n = 3;
        let mut max_error: f64 = 0.0;
        for i in 0..m {
            for j in 0..n {
                let mut reconstructed = Complex64::new(0.0, 0.0);
                for k in 0..result.rank {
                    reconstructed += result.left[[i, k]] * result.right[[k, j]];
                }
                let original = matrix[[i, j]];
                let error = (reconstructed - original).norm();
                max_error = max_error.max(error);
            }
        }
        assert!(
            max_error < 1e-10,
            "Reconstruction error too large: {}",
            max_error
        );
    }
}
