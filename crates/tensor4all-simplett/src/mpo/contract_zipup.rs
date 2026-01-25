//! Zip-up MPO contraction algorithm
//!
//! Contracts two MPOs with on-the-fly compression at each step.
//! This is more memory-efficient than naive contraction followed by compression.

use super::contraction::ContractionOptions;
use super::error::{MPOError, Result};
use super::factorize::{factorize, FactorizeOptions, SVDScalar};
use super::mpo::MPO;
use super::types::{tensor4_zeros, Tensor4, Tensor4Ops};
use super::{matrix2_zeros, Matrix2};

/// Perform zip-up contraction of two MPOs
///
/// This computes C = A * B where the contraction is over the shared
/// physical index (s2 of A contracts with s1 of B), with on-the-fly
/// compression at each step.
///
/// The zip-up algorithm:
/// 1. Start from the left with a remainder tensor R = \[\[1\]\]
/// 2. At each site:
///    a. Contract R with A\[i\] and B\[i\]
///    b. Reshape to matrix
///    c. Factorize into left and right factors
///    d. Store left factor as result tensor
///    e. Use right factor as new remainder R
/// 3. At the last site, just store the contracted tensor
///
/// # Arguments
/// * `mpo_a` - First MPO
/// * `mpo_b` - Second MPO
/// * `options` - Contraction options (tolerance, max_bond_dim, method)
///
/// # Returns
/// The contracted and compressed MPO C
pub fn contract_zipup<T: SVDScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    options: &ContractionOptions,
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

    // Remainder tensor: R[new_link, link_a, link_b]
    // Start with 1x1x1 identity
    let mut r_left_dim = 1usize;
    let mut r_a_dim = 1usize;
    let mut r_b_dim = 1usize;
    let mut r_data: Vec<T> = vec![T::one()];

    let mut result_tensors: Vec<Tensor4<T>> = Vec::with_capacity(n);

    let factorize_opts = FactorizeOptions {
        method: options.factorize_method,
        tolerance: options.tolerance,
        max_rank: options.max_bond_dim,
        left_orthogonal: true,
        ..Default::default()
    };

    for i in 0..n {
        let a = mpo_a.site_tensor(i);
        let b = mpo_b.site_tensor(i);

        let s1_a = a.site_dim_1();
        let s2_a = a.site_dim_2(); // shared with s1_b
        let s2_b = b.site_dim_2();
        let right_a = a.right_dim();
        let right_b = b.right_dim();

        // Contract R with A and B:
        // C[new_link, s1_a, s2_b, right_a, right_b]
        // = sum_{link_a, link_b, k} R[new_link, link_a, link_b]
        //   * A[link_a, s1_a, k, right_a] * B[link_b, k, s2_b, right_b]

        let c_new_link = r_left_dim;
        let c_s1 = s1_a;
        let c_s2 = s2_b;
        let c_right_a = right_a;
        let c_right_b = right_b;

        // C as a 5D array, but we'll store it flat
        // Shape: (c_new_link * c_s1 * c_s2) x (c_right_a * c_right_b)
        let rows = c_new_link * c_s1 * c_s2;
        let cols = c_right_a * c_right_b;
        let mut c_mat: Matrix2<T> = matrix2_zeros(rows, cols);

        for ln in 0..r_left_dim {
            for la in 0..r_a_dim {
                for lb in 0..r_b_dim {
                    let r_idx = (ln * r_a_dim + la) * r_b_dim + lb;
                    let r_val = r_data[r_idx];
                    for s1 in 0..c_s1 {
                        for s2 in 0..c_s2 {
                            for k in 0..s2_a {
                                for ra in 0..c_right_a {
                                    for rb in 0..c_right_b {
                                        let row = (ln * c_s1 + s1) * c_s2 + s2;
                                        let col = ra * c_right_b + rb;
                                        c_mat[[row, col]] = c_mat[[row, col]]
                                            + r_val
                                                * *a.get4(la, s1, k, ra)
                                                * *b.get4(lb, k, s2, rb);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if i == n - 1 {
            // Last site: just reshape and store
            // Result should have right_dim = 1
            let mut result_tensor = tensor4_zeros(c_new_link, c_s1, c_s2, 1);
            for ln in 0..c_new_link {
                for s1 in 0..c_s1 {
                    for s2 in 0..c_s2 {
                        let row = (ln * c_s1 + s1) * c_s2 + s2;
                        // Sum over all right indices (should be 1x1)
                        let val = c_mat[[row, 0]];
                        result_tensor.set4(ln, s1, s2, 0, val);
                    }
                }
            }
            result_tensors.push(result_tensor);
        } else {
            // Factorize C into left and right parts
            let fact_result = factorize(&c_mat, &factorize_opts)?;
            let new_bond_dim = fact_result.rank.max(1);
            let left_rows = fact_result.left.dim(0);
            let left_cols = fact_result.left.dim(1);
            let right_rows = fact_result.right.dim(0);
            let right_cols = fact_result.right.dim(1);

            // Store left factor as site tensor: (c_new_link, c_s1, c_s2, new_bond_dim)
            let mut result_tensor = tensor4_zeros(c_new_link, c_s1, c_s2, new_bond_dim);
            for ln in 0..c_new_link {
                for s1 in 0..c_s1 {
                    for s2 in 0..c_s2 {
                        for r in 0..new_bond_dim {
                            let row = (ln * c_s1 + s1) * c_s2 + s2;
                            if row < left_rows && r < left_cols {
                                result_tensor.set4(ln, s1, s2, r, fact_result.left[[row, r]]);
                            }
                        }
                    }
                }
            }
            result_tensors.push(result_tensor);

            // Update R tensor for next iteration: R[new_bond_dim, right_a, right_b]
            r_left_dim = new_bond_dim;
            r_a_dim = c_right_a;
            r_b_dim = c_right_b;
            r_data = vec![T::zero(); r_left_dim * r_a_dim * r_b_dim];

            for l in 0..new_bond_dim {
                for ra in 0..c_right_a {
                    for rb in 0..c_right_b {
                        let col = ra * c_right_b + rb;
                        let r_idx = (l * r_a_dim + ra) * r_b_dim + rb;
                        if l < right_rows && col < right_cols {
                            r_data[r_idx] = fact_result.right[[l, col]];
                        }
                    }
                }
            }
        }
    }

    Ok(MPO::from_tensors_unchecked(result_tensors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpo::factorize::FactorizeMethod;

    #[test]
    fn test_contract_zipup_identity() {
        // Identity * Identity = Identity
        let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

        let options = ContractionOptions {
            tolerance: 1e-12,
            max_bond_dim: 10,
            factorize_method: FactorizeMethod::SVD,
        };

        let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

        assert_eq!(result.len(), 2);

        // The result should be equivalent to identity
        assert!((result.evaluate(&[0, 0, 0, 0]).unwrap() - 1.0).abs() < 1e-10);
        assert!((result.evaluate(&[0, 1, 0, 0]).unwrap()).abs() < 1e-10);
        assert!((result.evaluate(&[1, 1, 1, 1]).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_contract_zipup_constant() {
        let mpo_a = MPO::<f64>::constant(&[(2, 2)], 2.0);
        let mpo_b = MPO::<f64>::constant(&[(2, 2)], 3.0);

        let options = ContractionOptions::default();

        let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

        // Each element of C = sum over k of A[i, k] * B[k, j]
        // = sum over k of 2 * 3 = 6 * 2 = 12
        let val = result.evaluate(&[0, 0]).unwrap();
        assert!((val - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_contract_zipup_compresses() {
        // Create MPOs with higher bond dimensions
        let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
        let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

        let options = ContractionOptions {
            tolerance: 1e-12,
            max_bond_dim: 2,
            factorize_method: FactorizeMethod::SVD,
        };

        let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

        // Bond dimension should be limited
        assert!(result.rank() <= 2);
    }
}
