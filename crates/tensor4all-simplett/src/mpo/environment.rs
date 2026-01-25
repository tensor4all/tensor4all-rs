//! Environment computation for MPO contractions
//!
//! This module provides functions for computing left and right environments
//! used in variational algorithms and efficient MPO evaluation.

use super::error::{MPOError, Result};
use super::factorize::SVDScalar;
use super::mpo::MPO;
use super::types::{Tensor4, Tensor4Ops};
use mdarray::DTensor;

/// Type alias for 2D matrix using mdarray
pub type Matrix2<T> = DTensor<T, 2>;

/// Helper function to create a zero-filled 2D tensor
fn matrix2_zeros<T: Clone + Default>(rows: usize, cols: usize) -> Matrix2<T> {
    DTensor::<T, 2>::from_elem([rows, cols], T::default())
}

/// Contract two general tensors over specified indices
///
/// This is the Rust equivalent of `_contract` from Julia.
///
/// # Arguments
/// * `a` - First tensor (flattened with shape info)
/// * `a_shape` - Shape of first tensor
/// * `b` - Second tensor (flattened with shape info)
/// * `b_shape` - Shape of second tensor
/// * `idx_a` - Indices of `a` to contract over
/// * `idx_b` - Indices of `b` to contract over
///
/// # Returns
/// Contracted tensor as flat vector with shape
pub fn contract_tensors<T: SVDScalar>(
    _a: &[T],
    _a_shape: &[usize],
    _b: &[T],
    _b_shape: &[usize],
    _idx_a: &[usize],
    _idx_b: &[usize],
) -> Result<(Vec<T>, Vec<usize>)>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    // TODO: Implement general tensor contraction
    Err(MPOError::InvalidOperation {
        message: "contract_tensors not yet implemented".to_string(),
    })
}

/// Contract two 4D site tensors over their shared physical index
///
/// Given two 4D tensors:
/// - A: (left_a, s1_a, s2_a, right_a) where s2_a is the shared index
/// - B: (left_b, s1_b, s2_b, right_b) where s1_b is the shared index
///
/// Contracts over s2_a = s1_b to produce:
/// - C: (left_a * left_b, s1_a, s2_b, right_a * right_b)
///
/// This is the Rust equivalent of `_contractsitetensors` from Julia.
pub fn contract_site_tensors<T: SVDScalar>(a: &Tensor4<T>, b: &Tensor4<T>) -> Result<Tensor4<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    // Check that shared dimensions match
    if a.site_dim_2() != b.site_dim_1() {
        return Err(MPOError::SharedDimensionMismatch {
            site: 0,
            dim_a: a.site_dim_2(),
            dim_b: b.site_dim_1(),
        });
    }

    let left_a = a.left_dim();
    let s1_a = a.site_dim_1();
    let s2_a = a.site_dim_2(); // shared dimension
    let right_a = a.right_dim();

    let left_b = b.left_dim();
    let s2_b = b.site_dim_2();
    let right_b = b.right_dim();

    // Result dimensions
    let new_left = left_a * left_b;
    let new_s1 = s1_a;
    let new_s2 = s2_b;
    let new_right = right_a * right_b;

    let mut result = super::types::tensor4_zeros(new_left, new_s1, new_s2, new_right);

    // Contract over shared index
    for la in 0..left_a {
        for lb in 0..left_b {
            for s1 in 0..s1_a {
                for s2 in 0..s2_b {
                    for ra in 0..right_a {
                        for rb in 0..right_b {
                            let mut sum = T::zero();
                            for k in 0..s2_a {
                                sum = sum + *a.get4(la, s1, k, ra) * *b.get4(lb, k, s2, rb);
                            }
                            let new_l = la * left_b + lb;
                            let new_r = ra * right_b + rb;
                            let old = *result.get4(new_l, s1, s2, new_r);
                            result.set4(new_l, s1, s2, new_r, old + sum);
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Compute the left environment at site i for MPO contraction
///
/// The left environment L\[i\] represents the contraction of all sites 0..i
/// for the product of two MPOs A and B.
///
/// L\[i\] has shape (left_a_i, left_b_i) representing the accumulated
/// contraction from the left.
pub fn left_environment<T: SVDScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    site: usize,
    cache: &mut Vec<Option<Matrix2<T>>>,
) -> Result<Matrix2<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    if mpo_a.len() != mpo_b.len() {
        return Err(MPOError::LengthMismatch {
            expected: mpo_a.len(),
            got: mpo_b.len(),
        });
    }

    // Base case: left of site 0 is just [[1]]
    if site == 0 {
        let mut env: Matrix2<T> = matrix2_zeros(1, 1);
        env[[0, 0]] = T::one();
        return Ok(env);
    }

    // Check cache
    if site <= cache.len() && cache[site - 1].is_some() {
        return Ok(cache[site - 1].as_ref().unwrap().clone());
    }

    // Recursively compute from the left
    let prev_env = left_environment(mpo_a, mpo_b, site - 1, cache)?;
    let a = mpo_a.site_tensor(site - 1);
    let b = mpo_b.site_tensor(site - 1);

    // Contract: L[i-1] with A[i-1] and B[i-1]
    // L[i-1]: (left_a, left_b)
    // A[i-1]: (left_a, s1_a, s2_a, right_a)
    // B[i-1]: (left_b, s1_b, s2_b, right_b)
    //
    // Sum over left_a, left_b, s1_a (=s1_b), s2_a (=s2_b) to get:
    // L[i]: (right_a, right_b)

    // Check shared dimensions
    if a.site_dim_1() != b.site_dim_1() || a.site_dim_2() != b.site_dim_2() {
        return Err(MPOError::SharedDimensionMismatch {
            site: site - 1,
            dim_a: a.site_dim_1(),
            dim_b: b.site_dim_1(),
        });
    }

    let right_a = a.right_dim();
    let right_b = b.right_dim();
    let mut new_env: Matrix2<T> = matrix2_zeros(right_a, right_b);

    for la in 0..a.left_dim() {
        for lb in 0..b.left_dim() {
            let l_val = prev_env[[la, lb]];
            for s1 in 0..a.site_dim_1() {
                for s2 in 0..a.site_dim_2() {
                    for ra in 0..right_a {
                        for rb in 0..right_b {
                            new_env[[ra, rb]] = new_env[[ra, rb]]
                                + l_val * *a.get4(la, s1, s2, ra) * *b.get4(lb, s1, s2, rb);
                        }
                    }
                }
            }
        }
    }

    // Update cache
    while cache.len() < site {
        cache.push(None);
    }
    cache[site - 1] = Some(new_env.clone());

    Ok(new_env)
}

/// Compute the right environment at site i for MPO contraction
///
/// The right environment R\[i\] represents the contraction of all sites i+1..L
/// for the product of two MPOs A and B.
///
/// R\[i\] has shape (right_a_i, right_b_i) representing the accumulated
/// contraction from the right.
pub fn right_environment<T: SVDScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    site: usize,
    cache: &mut Vec<Option<Matrix2<T>>>,
) -> Result<Matrix2<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    if mpo_a.len() != mpo_b.len() {
        return Err(MPOError::LengthMismatch {
            expected: mpo_a.len(),
            got: mpo_b.len(),
        });
    }

    let n = mpo_a.len();

    // Base case: right of last site is just [[1]]
    if site == n - 1 {
        let mut env: Matrix2<T> = matrix2_zeros(1, 1);
        env[[0, 0]] = T::one();
        return Ok(env);
    }

    // Check cache
    let cache_idx = n - site - 2;
    if cache_idx < cache.len() && cache[cache_idx].is_some() {
        return Ok(cache[cache_idx].as_ref().unwrap().clone());
    }

    // Recursively compute from the right
    let prev_env = right_environment(mpo_a, mpo_b, site + 1, cache)?;
    let a = mpo_a.site_tensor(site + 1);
    let b = mpo_b.site_tensor(site + 1);

    // Contract: R[i+1] with A[i+1] and B[i+1]
    // R[i+1]: (right_a, right_b)
    // A[i+1]: (left_a, s1_a, s2_a, right_a)
    // B[i+1]: (left_b, s1_b, s2_b, right_b)
    //
    // Sum over right_a, right_b, s1_a (=s1_b), s2_a (=s2_b) to get:
    // R[i]: (left_a, left_b)

    // Check shared dimensions
    if a.site_dim_1() != b.site_dim_1() || a.site_dim_2() != b.site_dim_2() {
        return Err(MPOError::SharedDimensionMismatch {
            site: site + 1,
            dim_a: a.site_dim_1(),
            dim_b: b.site_dim_1(),
        });
    }

    let left_a = a.left_dim();
    let left_b = b.left_dim();
    let mut new_env: Matrix2<T> = matrix2_zeros(left_a, left_b);

    for ra in 0..a.right_dim() {
        for rb in 0..b.right_dim() {
            let r_val = prev_env[[ra, rb]];
            for s1 in 0..a.site_dim_1() {
                for s2 in 0..a.site_dim_2() {
                    for la in 0..left_a {
                        for lb in 0..left_b {
                            new_env[[la, lb]] = new_env[[la, lb]]
                                + r_val * *a.get4(la, s1, s2, ra) * *b.get4(lb, s1, s2, rb);
                        }
                    }
                }
            }
        }
    }

    // Update cache
    while cache.len() <= cache_idx {
        cache.push(None);
    }
    cache[cache_idx] = Some(new_env.clone());

    Ok(new_env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpo::types::tensor4_zeros;

    #[test]
    fn test_contract_site_tensors() {
        // Create two simple 4D tensors
        let mut a: Tensor4<f64> = tensor4_zeros(1, 2, 3, 1);
        let mut b: Tensor4<f64> = tensor4_zeros(1, 3, 2, 1);

        // Fill with some values
        for s1 in 0..2 {
            for s2 in 0..3 {
                a.set4(0, s1, s2, 0, (s1 * 3 + s2 + 1) as f64);
            }
        }
        for s1 in 0..3 {
            for s2 in 0..2 {
                b.set4(0, s1, s2, 0, (s1 * 2 + s2 + 1) as f64);
            }
        }

        let result = contract_site_tensors(&a, &b).unwrap();

        // Result should have shape (1, 2, 2, 1)
        assert_eq!(result.left_dim(), 1);
        assert_eq!(result.site_dim_1(), 2);
        assert_eq!(result.site_dim_2(), 2);
        assert_eq!(result.right_dim(), 1);
    }

    #[test]
    fn test_left_environment() {
        let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
        let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

        let mut cache = Vec::new();

        // Left environment at site 0 should be [[1]]
        let env0 = left_environment(&mpo_a, &mpo_b, 0, &mut cache).unwrap();
        assert_eq!(env0[[0, 0]], 1.0);

        // Left environment at site 1
        let env1 = left_environment(&mpo_a, &mpo_b, 1, &mut cache).unwrap();
        // Each element contributes 1*1 = 1, sum over 2*2=4 physical indices
        assert!((env1[[0, 0]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_right_environment() {
        let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
        let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

        let mut cache = Vec::new();

        // Right environment at last site should be [[1]]
        let env_last = right_environment(&mpo_a, &mpo_b, 1, &mut cache).unwrap();
        assert_eq!(env_last[[0, 0]], 1.0);

        // Right environment at site 0
        let env0 = right_environment(&mpo_a, &mpo_b, 0, &mut cache).unwrap();
        // Each element contributes 1*1 = 1, sum over 2*2=4 physical indices
        assert!((env0[[0, 0]] - 4.0).abs() < 1e-10);
    }
}
