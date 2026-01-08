//! Naive MPO contraction algorithm
//!
//! Contracts two MPOs by directly multiplying site tensors,
//! optionally followed by compression.

use super::contraction::ContractionOptions;
use super::environment::contract_site_tensors;
use super::error::{MPOError, Result};
use super::factorize::{factorize, FactorizeOptions, Matrix2, SVDScalar};
use super::mpo::MPO;
use super::types::{tensor4_zeros, Tensor4, Tensor4Ops};
use mdarray::DTensor;

/// Helper function to create a zero-filled 2D tensor
fn matrix2_zeros<T: Clone + Default>(rows: usize, cols: usize) -> Matrix2<T> {
    DTensor::<T, 2>::from_elem([rows, cols], T::default())
}

/// Perform naive contraction of two MPOs
///
/// This computes C = A * B where the contraction is over the shared
/// physical index (s2 of A contracts with s1 of B).
///
/// The naive algorithm:
/// 1. Contract each pair of site tensors
/// 2. Optionally compress the result
///
/// # Arguments
/// * `mpo_a` - First MPO
/// * `mpo_b` - Second MPO
/// * `options` - Optional compression options
///
/// # Returns
/// The contracted MPO C with dimensions:
/// - s1: from A
/// - s2: from B
/// - bond dimensions: product of input bond dimensions (before compression)
pub fn contract_naive<T: SVDScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    options: Option<ContractionOptions>,
) -> Result<MPO<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    if mpo_a.len() != mpo_b.len() {
        return Err(MPOError::LengthMismatch {
            expected: mpo_a.len(),
            got: mpo_b.len(),
        });
    }

    if mpo_a.is_empty() {
        return Ok(MPO::from_tensors_unchecked(Vec::new()));
    }

    let n = mpo_a.len();

    // Validate shared dimensions
    for i in 0..n {
        let (_, s2_a) = mpo_a.site_dim(i);
        let (s1_b, _) = mpo_b.site_dim(i);
        if s2_a != s1_b {
            return Err(MPOError::SharedDimensionMismatch {
                site: i,
                dim_a: s2_a,
                dim_b: s1_b,
            });
        }
    }

    // Contract each pair of site tensors
    let mut tensors: Vec<Tensor4<T>> = Vec::with_capacity(n);

    for i in 0..n {
        let a = mpo_a.site_tensor(i);
        let b = mpo_b.site_tensor(i);

        // Contract over shared index: a.s2 = b.s1
        // Result has shape:
        // (left_a * left_b, s1_a, s2_b, right_a * right_b)
        let contracted = contract_site_tensors(a, b)?;
        tensors.push(contracted);
    }

    let mut result = MPO::from_tensors_unchecked(tensors);

    // Apply compression if options are provided
    if let Some(opts) = options {
        compress_mpo(&mut result, &opts)?;
    }

    Ok(result)
}

/// Compress an MPO using the specified options
fn compress_mpo<T: SVDScalar>(mpo: &mut MPO<T>, options: &ContractionOptions) -> Result<()>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    if mpo.len() <= 1 {
        return Ok(());
    }

    let factorize_opts = FactorizeOptions {
        method: options.factorize_method,
        tolerance: options.tolerance,
        max_rank: options.max_bond_dim,
        left_orthogonal: true,
        ..Default::default()
    };

    // Sweep left to right, factorizing each bond
    for i in 0..(mpo.len() - 1) {
        let tensor = mpo.site_tensor(i);
        let left_dim = tensor.left_dim();
        let s1 = tensor.site_dim_1();
        let s2 = tensor.site_dim_2();
        let right_dim = tensor.right_dim();

        // Reshape to matrix: (left * s1 * s2, right)
        let rows = left_dim * s1 * s2;
        let cols = right_dim;
        let mut mat: Matrix2<T> = matrix2_zeros(rows, cols);
        for l in 0..left_dim {
            for i1 in 0..s1 {
                for i2 in 0..s2 {
                    let row = (l * s1 + i1) * s2 + i2;
                    for col in 0..cols {
                        mat[[row, col]] = *tensor.get4(l, i1, i2, col);
                    }
                }
            }
        }

        // Factorize
        let fact_result = factorize(&mat, &factorize_opts)?;
        let new_rank = fact_result.rank;

        // Update current tensor with left factor
        let mut new_tensor = tensor4_zeros(left_dim, s1, s2, new_rank);
        let left_rows = fact_result.left.dim(0);
        let left_cols = fact_result.left.dim(1);
        for l in 0..left_dim {
            for i1 in 0..s1 {
                for i2 in 0..s2 {
                    let row = (l * s1 + i1) * s2 + i2;
                    for r in 0..new_rank {
                        if row < left_rows && r < left_cols {
                            new_tensor.set4(l, i1, i2, r, fact_result.left[[row, r]]);
                        }
                    }
                }
            }
        }

        // Get next tensor's dimensions
        let next_tensor = mpo.site_tensor(i + 1);
        let next_left = next_tensor.left_dim();
        let next_s1 = next_tensor.site_dim_1();
        let next_s2 = next_tensor.site_dim_2();
        let next_right = next_tensor.right_dim();

        // Multiply right factor into next tensor
        // R[new_rank, right_dim] @ next[right_dim, s1, s2, next_right]
        // = new_next[new_rank, s1, s2, next_right]
        let right_rows = fact_result.right.dim(0);
        let right_cols = fact_result.right.dim(1);
        let mut new_next = tensor4_zeros(new_rank, next_s1, next_s2, next_right);
        for l in 0..new_rank {
            for i1 in 0..next_s1 {
                for i2 in 0..next_s2 {
                    for r in 0..next_right {
                        let mut sum = T::zero();
                        for k in 0..next_left.min(right_cols) {
                            if l < right_rows {
                                sum = sum
                                    + fact_result.right[[l, k]] * *next_tensor.get4(k, i1, i2, r);
                            }
                        }
                        new_next.set4(l, i1, i2, r, sum);
                    }
                }
            }
        }

        // Update tensors in place
        *mpo.site_tensor_mut(i) = new_tensor;
        *mpo.site_tensor_mut(i + 1) = new_next;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpo::factorize::FactorizeMethod;

    #[test]
    fn test_contract_naive_identity() {
        // Identity * Identity = Identity
        let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

        let result = contract_naive(&mpo_a, &mpo_b, None).unwrap();

        assert_eq!(result.len(), 2);

        // The result should be equivalent to identity
        // Check some evaluations
        assert!((result.evaluate(&[0, 0, 0, 0]).unwrap() - 1.0).abs() < 1e-10);
        assert!((result.evaluate(&[0, 1, 0, 0]).unwrap()).abs() < 1e-10);
        assert!((result.evaluate(&[1, 1, 1, 1]).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_contract_naive_constant() {
        // Constant * Constant
        let mpo_a = MPO::<f64>::constant(&[(2, 2)], 2.0);
        let mpo_b = MPO::<f64>::constant(&[(2, 2)], 3.0);

        let result = contract_naive(&mpo_a, &mpo_b, None).unwrap();

        // Each element of C = sum over k of A[i, k] * B[k, j]
        // = sum over k of 2 * 3 = 6 * (number of k values = 2) = 12
        assert_eq!(result.len(), 1);
        let val = result.evaluate(&[0, 0]).unwrap();
        assert!((val - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_contract_naive_dimension_mismatch() {
        let mpo_a = MPO::<f64>::constant(&[(2, 3)], 1.0); // s2 = 3
        let mpo_b = MPO::<f64>::constant(&[(2, 2)], 1.0); // s1 = 2 ≠ 3

        let result = contract_naive(&mpo_a, &mpo_b, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_contract_naive_with_compression() {
        let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
        let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

        let options = ContractionOptions {
            tolerance: 1e-10,
            max_bond_dim: 2,
            factorize_method: FactorizeMethod::SVD,
        };

        let result = contract_naive(&mpo_a, &mpo_b, Some(options)).unwrap();

        // Bond dimension should be compressed
        assert!(result.rank() <= 2);
    }
}
