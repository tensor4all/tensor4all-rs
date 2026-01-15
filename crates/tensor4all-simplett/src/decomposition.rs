//! Shared decomposition utilities for tensor train operations
//!
//! This module provides common matrix decomposition functions used across
//! canonical forms, Vidal representations, and compression algorithms.

use crate::traits::TTScalar;
use crate::types::{Tensor3, Tensor3Ops};
use matrixci::util::{ncols, nrows, transpose, zeros, Matrix, Scalar};
use matrixci::{rrlu, RrLUOptions};

/// Compute QR decomposition using rank-revealing LU with left-orthogonal output
///
/// Returns (Q, R) where Q is left-orthogonal and A ≈ Q * R
pub fn qr_decomp<T: TTScalar + Scalar>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    let options = RrLUOptions {
        max_rank: ncols(matrix).min(nrows(matrix)),
        rel_tol: 0.0, // No truncation
        abs_tol: 0.0,
        left_orthogonal: true,
    };
    let lu = rrlu(matrix, Some(options));
    (lu.left(true), lu.right(true))
}

/// Compute LQ decomposition (transpose, QR, transpose)
///
/// Returns (L, Q) where Q is right-orthogonal and A ≈ L * Q
pub fn lq_decomp<T: TTScalar + Scalar>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    let at = transpose(matrix);
    let (qt, lt) = qr_decomp(&at);
    (transpose(&lt), transpose(&qt))
}

/// Convert Tensor3 to Matrix with left dimensions flattened
///
/// Reshapes tensor of shape (left, site, right) to matrix of shape (left * site, right)
pub fn tensor3_to_left_matrix<T: Scalar + Default + Clone>(tensor: &Tensor3<T>) -> Matrix<T> {
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

/// Convert Tensor3 to Matrix with right dimensions flattened
///
/// Reshapes tensor of shape (left, site, right) to matrix of shape (left, site * right)
pub fn tensor3_to_right_matrix<T: Scalar + Default + Clone>(tensor: &Tensor3<T>) -> Matrix<T> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tensor3_zeros;

    #[test]
    fn test_qr_decomp_identity() {
        let mat: Matrix<f64> = Matrix::from_elem((3, 3), 0.0);
        let mut identity = mat.clone();
        for i in 0..3 {
            identity[[i, i]] = 1.0;
        }
        let (q, r) = qr_decomp(&identity);
        assert_eq!(nrows(&q), 3);
        assert_eq!(ncols(&r), 3);
    }

    #[test]
    fn test_tensor3_to_left_matrix() {
        let mut tensor: Tensor3<f64> = tensor3_zeros(2, 3, 4);
        // Set some values
        for l in 0..2 {
            for s in 0..3 {
                for r in 0..4 {
                    tensor[[l, s, r]] = (l * 12 + s * 4 + r) as f64;
                }
            }
        }

        let mat = tensor3_to_left_matrix(&tensor);
        assert_eq!(nrows(&mat), 6); // 2 * 3
        assert_eq!(ncols(&mat), 4);

        // Check some values
        assert_eq!(mat[[0, 0]], 0.0); // l=0, s=0, r=0
        assert_eq!(mat[[1, 0]], 4.0); // l=0, s=1, r=0
        assert_eq!(mat[[3, 0]], 12.0); // l=1, s=0, r=0
    }

    #[test]
    fn test_tensor3_to_right_matrix() {
        let mut tensor: Tensor3<f64> = tensor3_zeros(2, 3, 4);
        for l in 0..2 {
            for s in 0..3 {
                for r in 0..4 {
                    tensor[[l, s, r]] = (l * 12 + s * 4 + r) as f64;
                }
            }
        }

        let mat = tensor3_to_right_matrix(&tensor);
        assert_eq!(nrows(&mat), 2);
        assert_eq!(ncols(&mat), 12); // 3 * 4

        // Check some values
        assert_eq!(mat[[0, 0]], 0.0); // l=0, s=0, r=0
        assert_eq!(mat[[0, 4]], 4.0); // l=0, s=1, r=0
        assert_eq!(mat[[1, 0]], 12.0); // l=1, s=0, r=0
    }
}
