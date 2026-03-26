//! Compression algorithms for tensor trains

use crate::error::Result;
use crate::tensortrain::TensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{tensor3_zeros, Tensor3, Tensor3Ops};
use matrixci::util::{mat_mul, ncols, nrows, zeros, Matrix};
use matrixci::Scalar;
use matrixci::{rrlu, AbstractMatrixCI, MatrixLUCI, RrLUOptions};

/// Compression method for tensor trains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionMethod {
    /// LU decomposition (rank-revealing)
    #[default]
    LU,
    /// Cross interpolation based
    CI,
    /// SVD compression (currently unimplemented and returns an error)
    SVD,
}

/// Options for compression
#[derive(Debug, Clone)]
pub struct CompressionOptions {
    /// Compression method
    pub method: CompressionMethod,
    /// Tolerance for truncation (relative)
    pub tolerance: f64,
    /// Maximum bond dimension
    pub max_bond_dim: usize,
    /// Whether to normalize the error
    pub normalize_error: bool,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            method: CompressionMethod::LU,
            tolerance: 1e-12,
            max_bond_dim: usize::MAX,
            normalize_error: true,
        }
    }
}

/// Convert Tensor3 to Matrix for factorization (left matrix view)
fn tensor3_to_left_matrix<T: TTScalar + Scalar + Default>(tensor: &Tensor3<T>) -> Matrix<T> {
    let left_dim = tensor.left_dim();
    let site_dim = tensor.site_dim();
    let right_dim = tensor.right_dim();
    let rows = left_dim * site_dim;
    let cols = right_dim;

    let mut mat = zeros(rows, cols);
    for l in 0..left_dim {
        for s in 0..site_dim {
            for r in 0..right_dim {
                mat[[l * site_dim + s, r]] = *tensor.get3(l, s, r);
            }
        }
    }
    mat
}

/// Convert Tensor3 to Matrix for factorization (right matrix view)
fn tensor3_to_right_matrix<T: TTScalar + Scalar + Default>(tensor: &Tensor3<T>) -> Matrix<T> {
    let left_dim = tensor.left_dim();
    let site_dim = tensor.site_dim();
    let right_dim = tensor.right_dim();
    let rows = left_dim;
    let cols = site_dim * right_dim;

    let mut mat = zeros(rows, cols);
    for l in 0..left_dim {
        for s in 0..site_dim {
            for r in 0..right_dim {
                mat[[l, s * right_dim + r]] = *tensor.get3(l, s, r);
            }
        }
    }
    mat
}

/// Factorize a matrix into left and right factors
fn factorize<T: TTScalar + Scalar>(
    matrix: &Matrix<T>,
    method: CompressionMethod,
    tolerance: f64,
    max_bond_dim: usize,
    left_orthogonal: bool,
) -> crate::error::Result<(Matrix<T>, Matrix<T>, usize)> {
    let reltol = if tolerance > 0.0 { tolerance } else { 1e-14 };
    let abstol = 0.0;

    let options = RrLUOptions {
        max_rank: max_bond_dim,
        rel_tol: reltol,
        abs_tol: abstol,
        left_orthogonal,
    };

    match method {
        CompressionMethod::LU => {
            let lu = rrlu(matrix, Some(options))?;
            let left = lu.left(true); // permuted
            let right = lu.right(true); // permuted
            let npivots = lu.npivots();
            Ok((left, right, npivots))
        }
        CompressionMethod::CI => {
            let luci = MatrixLUCI::from_matrix(matrix, Some(options))?;
            let left = luci.left();
            let right = luci.right();
            let npivots = luci.rank();
            Ok((left, right, npivots))
        }
        CompressionMethod::SVD => Err(crate::error::TensorTrainError::InvalidOperation {
            message: "SVD compression is not yet implemented".to_string(),
        }),
    }
}

impl<T: TTScalar + Scalar + Default> TensorTrain<T> {
    /// Compress the tensor train in-place using the specified method
    ///
    /// This performs a two-sweep compression:
    /// 1. Left-to-right sweep with left-orthogonal factorization (no truncation)
    /// 2. Right-to-left sweep with truncation
    pub fn compress(&mut self, options: &CompressionOptions) -> Result<()> {
        let n = self.len();
        if n <= 1 {
            return Ok(());
        }

        let tensors = self.site_tensors_mut();

        // Left-to-right sweep: make left-orthogonal without truncation
        for ell in 0..n - 1 {
            let left_dim = tensors[ell].left_dim();
            let site_dim = tensors[ell].site_dim();

            // Reshape to matrix: (left_dim * site_dim, right_dim)
            let mat = tensor3_to_left_matrix(&tensors[ell]);

            // Factorize without truncation
            let (left_factor, right_factor, new_bond_dim) = factorize(
                &mat,
                options.method,
                0.0,        // No truncation in left sweep
                usize::MAX, // No max bond dim in left sweep
                true,       // left orthogonal
            )?;

            // Update current tensor
            let mut new_tensor = tensor3_zeros(left_dim, site_dim, new_bond_dim);
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..new_bond_dim {
                        let row = l * site_dim + s;
                        if row < nrows(&left_factor) && r < ncols(&left_factor) {
                            new_tensor.set3(l, s, r, left_factor[[row, r]]);
                        }
                    }
                }
            }
            tensors[ell] = new_tensor;

            // Contract right_factor with next tensor
            let next_site_dim = tensors[ell + 1].site_dim();
            let next_right_dim = tensors[ell + 1].right_dim();

            // Build next tensor as matrix (old_left_dim, site_dim * right_dim)
            let next_mat = tensor3_to_right_matrix(&tensors[ell + 1]);

            // Multiply: right_factor * next_mat
            let contracted = mat_mul(&right_factor, &next_mat);

            // Update next tensor
            let mut new_next_tensor = tensor3_zeros(new_bond_dim, next_site_dim, next_right_dim);
            for l in 0..new_bond_dim {
                for s in 0..next_site_dim {
                    for r in 0..next_right_dim {
                        new_next_tensor.set3(l, s, r, contracted[[l, s * next_right_dim + r]]);
                    }
                }
            }
            tensors[ell + 1] = new_next_tensor;
        }

        // Right-to-left sweep: truncate
        for ell in (1..n).rev() {
            let site_dim = tensors[ell].site_dim();
            let right_dim = tensors[ell].right_dim();

            // Reshape to matrix: (left_dim, site_dim * right_dim)
            let mat = tensor3_to_right_matrix(&tensors[ell]);

            // Factorize with truncation
            let (left_factor, right_factor, new_bond_dim) = factorize(
                &mat,
                options.method,
                options.tolerance,
                options.max_bond_dim,
                false, // right orthogonal
            )?;

            // Update current tensor from right_factor
            let mut new_tensor = tensor3_zeros(new_bond_dim, site_dim, right_dim);
            for l in 0..new_bond_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        new_tensor.set3(l, s, r, right_factor[[l, s * right_dim + r]]);
                    }
                }
            }
            tensors[ell] = new_tensor;

            // Contract previous tensor with left_factor
            let prev_left_dim = tensors[ell - 1].left_dim();
            let prev_site_dim = tensors[ell - 1].site_dim();

            // Build prev tensor as matrix (left_dim * site_dim, old_right_dim)
            let prev_mat = tensor3_to_left_matrix(&tensors[ell - 1]);

            // Multiply: prev_mat * left_factor
            let contracted = mat_mul(&prev_mat, &left_factor);

            // Update prev tensor
            let mut new_prev_tensor = tensor3_zeros(prev_left_dim, prev_site_dim, new_bond_dim);
            for l in 0..prev_left_dim {
                for s in 0..prev_site_dim {
                    for r in 0..new_bond_dim {
                        new_prev_tensor.set3(l, s, r, contracted[[l * prev_site_dim + s, r]]);
                    }
                }
            }
            tensors[ell - 1] = new_prev_tensor;
        }

        Ok(())
    }

    /// Create a compressed copy of the tensor train
    pub fn compressed(&self, options: &CompressionOptions) -> Result<Self> {
        let mut result = self.clone();
        result.compress(options)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
