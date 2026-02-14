//! Contraction struct with caching for efficient MPO evaluation
//!
//! The Contraction struct wraps two MPOs and provides efficient
//! evaluation with memoization of left and right environments.

use std::collections::HashMap;

use super::error::{MPOError, Result};
use super::factorize::SVDScalar;
use super::matrix2_zeros;
use super::mpo::MPO;
use super::types::Tensor4Ops;
use super::Matrix2;

/// Options for contraction operations
#[derive(Debug, Clone)]
pub struct ContractionOptions {
    /// Tolerance for truncation
    pub tolerance: f64,
    /// Maximum bond dimension after contraction
    pub max_bond_dim: usize,
    /// Factorization method for compression
    pub factorize_method: super::factorize::FactorizeMethod,
}

impl Default for ContractionOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_bond_dim: usize::MAX,
            factorize_method: super::factorize::FactorizeMethod::SVD,
        }
    }
}

/// Contraction of two MPOs with caching
///
/// This struct efficiently computes the product of two MPOs by
/// caching left and right environments for reuse.
pub struct Contraction<T: SVDScalar>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    /// First MPO
    mpo_a: MPO<T>,
    /// Second MPO
    mpo_b: MPO<T>,
    /// Cache for left environments, keyed by index sets
    left_cache: HashMap<Vec<(usize, usize)>, Matrix2<T>>,
    /// Cache for right environments, keyed by index sets
    right_cache: HashMap<Vec<(usize, usize)>, Matrix2<T>>,
    /// Optional transformation function applied to result
    transform_fn: Option<Box<dyn Fn(T) -> T + Send + Sync>>,
    /// Site dimensions for both MPOs [(s1_a, s2_a, s1_b, s2_b), ...]
    site_dims: Vec<(usize, usize, usize, usize)>,
}

impl<T: SVDScalar> Contraction<T>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    /// Create a new Contraction from two MPOs
    pub fn new(mpo_a: MPO<T>, mpo_b: MPO<T>) -> Result<Self> {
        if mpo_a.len() != mpo_b.len() {
            return Err(MPOError::LengthMismatch {
                expected: mpo_a.len(),
                got: mpo_b.len(),
            });
        }

        // Validate shared dimensions
        for i in 0..mpo_a.len() {
            let (_s1_a, s2_a) = mpo_a.site_dim(i);
            let (s1_b, _s2_b) = mpo_b.site_dim(i);
            if s2_a != s1_b {
                return Err(MPOError::SharedDimensionMismatch {
                    site: i,
                    dim_a: s2_a,
                    dim_b: s1_b,
                });
            }
        }

        let site_dims: Vec<_> = (0..mpo_a.len())
            .map(|i| {
                let (s1_a, s2_a) = mpo_a.site_dim(i);
                let (s1_b, s2_b) = mpo_b.site_dim(i);
                (s1_a, s2_a, s1_b, s2_b)
            })
            .collect();

        Ok(Self {
            mpo_a,
            mpo_b,
            left_cache: HashMap::new(),
            right_cache: HashMap::new(),
            transform_fn: None,
            site_dims,
        })
    }

    /// Create a new Contraction with a transformation function
    pub fn with_transform<F>(mpo_a: MPO<T>, mpo_b: MPO<T>, f: F) -> Result<Self>
    where
        F: Fn(T) -> T + Send + Sync + 'static,
    {
        let mut contraction = Self::new(mpo_a, mpo_b)?;
        contraction.transform_fn = Some(Box::new(f));
        Ok(contraction)
    }

    /// Get the number of sites
    pub fn len(&self) -> usize {
        self.mpo_a.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.mpo_a.is_empty()
    }

    /// Get site dimensions for the contracted result
    ///
    /// Returns (s1_result, s2_result) at each site where:
    /// - s1_result = s1_a (first physical index of A)
    /// - s2_result = s2_b (second physical index of B)
    pub fn result_site_dims(&self) -> Vec<(usize, usize)> {
        self.site_dims
            .iter()
            .map(|&(s1_a, _, _, s2_b)| (s1_a, s2_b))
            .collect()
    }

    /// Clear all cached environments
    pub fn clear_cache(&mut self) {
        self.left_cache.clear();
        self.right_cache.clear();
    }

    /// Evaluate the contraction at a specific set of indices
    ///
    /// indices should be [(i1, j1), (i2, j2), ...] where:
    /// - i_k is the index for s1 of MPO A at site k
    /// - j_k is the index for s2 of MPO B at site k
    pub fn evaluate(&mut self, indices: &[(usize, usize)]) -> Result<T> {
        if indices.len() != self.len() {
            return Err(MPOError::InvalidOperation {
                message: format!("Expected {} index pairs, got {}", self.len(), indices.len()),
            });
        }

        if self.is_empty() {
            return Err(MPOError::Empty);
        }

        // Contract from left to right
        let first_a = self.mpo_a.site_tensor(0);
        let first_b = self.mpo_b.site_tensor(0);
        let (i0, j0) = indices[0];

        // Sum over shared index k
        let mut current: Matrix2<T> = matrix2_zeros(first_a.right_dim(), first_b.right_dim());
        for k in 0..first_a.site_dim_2() {
            for ra in 0..first_a.right_dim() {
                for rb in 0..first_b.right_dim() {
                    current[[ra, rb]] = current[[ra, rb]]
                        + *first_a.get4(0, i0, k, ra) * *first_b.get4(0, k, j0, rb);
                }
            }
        }

        // Contract through remaining sites
        #[allow(clippy::needless_range_loop)]
        for site in 1..self.len() {
            let a = self.mpo_a.site_tensor(site);
            let b = self.mpo_b.site_tensor(site);
            let (i_k, j_k) = indices[site];

            let mut new_current: Matrix2<T> = matrix2_zeros(a.right_dim(), b.right_dim());

            for la in 0..a.left_dim() {
                for lb in 0..b.left_dim() {
                    let c_val = current[[la, lb]];
                    for k in 0..a.site_dim_2() {
                        for ra in 0..a.right_dim() {
                            for rb in 0..b.right_dim() {
                                new_current[[ra, rb]] = new_current[[ra, rb]]
                                    + c_val * *a.get4(la, i_k, k, ra) * *b.get4(lb, k, j_k, rb);
                            }
                        }
                    }
                }
            }

            current = new_current;
        }

        let result = current[[0, 0]];

        // Apply transformation if present
        let result = if let Some(ref f) = self.transform_fn {
            f(result)
        } else {
            result
        };

        Ok(result)
    }

    /// Evaluate the left environment up to site n (exclusive)
    ///
    /// Returns L\[n\] = product of sites 0..n
    pub fn evaluate_left(&mut self, n: usize, indices: &[(usize, usize)]) -> Result<Matrix2<T>> {
        if n > self.len() {
            return Err(MPOError::InvalidOperation {
                message: format!("Site {} is out of range [0, {}]", n, self.len()),
            });
        }

        if n == 0 {
            let mut env: Matrix2<T> = matrix2_zeros(1, 1);
            env[[0, 0]] = T::one();
            return Ok(env);
        }

        // Check cache
        let key: Vec<(usize, usize)> = indices[..n].to_vec();
        if let Some(cached) = self.left_cache.get(&key) {
            return Ok(cached.clone());
        }

        // Compute recursively
        let prev_env = self.evaluate_left(n - 1, indices)?;
        let a = self.mpo_a.site_tensor(n - 1);
        let b = self.mpo_b.site_tensor(n - 1);
        let (i_k, j_k) = indices[n - 1];

        let mut new_env: Matrix2<T> = matrix2_zeros(a.right_dim(), b.right_dim());

        for la in 0..a.left_dim() {
            for lb in 0..b.left_dim() {
                let l_val = prev_env[[la, lb]];
                for k in 0..a.site_dim_2() {
                    for ra in 0..a.right_dim() {
                        for rb in 0..b.right_dim() {
                            new_env[[ra, rb]] = new_env[[ra, rb]]
                                + l_val * *a.get4(la, i_k, k, ra) * *b.get4(lb, k, j_k, rb);
                        }
                    }
                }
            }
        }

        // Cache the result
        self.left_cache.insert(key, new_env.clone());

        Ok(new_env)
    }

    /// Evaluate the right environment from site n (exclusive) to the end
    ///
    /// Returns R\[n\] = product of sites n..L
    pub fn evaluate_right(&mut self, n: usize, indices: &[(usize, usize)]) -> Result<Matrix2<T>> {
        let len = self.len();
        if n > len {
            return Err(MPOError::InvalidOperation {
                message: format!("Site {} is out of range [0, {}]", n, len),
            });
        }

        if n == len {
            let mut env: Matrix2<T> = matrix2_zeros(1, 1);
            env[[0, 0]] = T::one();
            return Ok(env);
        }

        // Check cache
        let key: Vec<(usize, usize)> = indices[n..].to_vec();
        if let Some(cached) = self.right_cache.get(&key) {
            return Ok(cached.clone());
        }

        // Compute recursively
        let prev_env = self.evaluate_right(n + 1, indices)?;
        let a = self.mpo_a.site_tensor(n);
        let b = self.mpo_b.site_tensor(n);
        let (i_k, j_k) = indices[n];

        let mut new_env: Matrix2<T> = matrix2_zeros(a.left_dim(), b.left_dim());

        for ra in 0..a.right_dim() {
            for rb in 0..b.right_dim() {
                let r_val = prev_env[[ra, rb]];
                for k in 0..a.site_dim_2() {
                    for la in 0..a.left_dim() {
                        for lb in 0..b.left_dim() {
                            new_env[[la, lb]] = new_env[[la, lb]]
                                + r_val * *a.get4(la, i_k, k, ra) * *b.get4(lb, k, j_k, rb);
                        }
                    }
                }
            }
        }

        // Cache the result
        self.right_cache.insert(key, new_env.clone());

        Ok(new_env)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contraction_new() {
        let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

        let contraction = Contraction::new(mpo_a, mpo_b).unwrap();
        assert_eq!(contraction.len(), 2);
        assert_eq!(contraction.result_site_dims(), vec![(2, 2), (2, 2)]);
    }

    #[test]
    fn test_contraction_evaluate() {
        // Identity * Identity = Identity
        let mpo_a = MPO::<f64>::identity(&[2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2]).unwrap();

        let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();

        // C[0, 0] = sum_k I[0, k] * I[k, 0] = I[0, 0] * I[0, 0] + I[0, 1] * I[1, 0]
        //         = 1 * 1 + 0 * 0 = 1
        let val_00 = contraction.evaluate(&[(0, 0)]).unwrap();
        assert!((val_00 - 1.0).abs() < 1e-10);

        // C[0, 1] = sum_k I[0, k] * I[k, 1] = I[0, 0] * I[0, 1] + I[0, 1] * I[1, 1]
        //         = 1 * 0 + 0 * 1 = 0
        let val_01 = contraction.evaluate(&[(0, 1)]).unwrap();
        assert!(val_01.abs() < 1e-10);

        // C[1, 1] = sum_k I[1, k] * I[k, 1] = I[1, 0] * I[0, 1] + I[1, 1] * I[1, 1]
        //         = 0 * 0 + 1 * 1 = 1
        let val_11 = contraction.evaluate(&[(1, 1)]).unwrap();
        assert!((val_11 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_contraction_with_transform() {
        let mpo_a = MPO::<f64>::identity(&[2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2]).unwrap();

        let mut contraction = Contraction::with_transform(mpo_a, mpo_b, |x| x * 2.0).unwrap();

        let val = contraction.evaluate(&[(0, 0)]).unwrap();
        assert!((val - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_contraction_length_mismatch() {
        let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 2, 2]).unwrap();

        let result = Contraction::new(mpo_a, mpo_b);
        assert!(result.is_err());
        assert!(matches!(
            result.as_ref(),
            Err(MPOError::LengthMismatch {
                expected: 2,
                got: 3
            })
        ));
    }

    #[test]
    fn test_contraction_shared_dimension_mismatch() {
        // A has site_dim (2, 3), B has site_dim (2, 2) => s2_a=3 != s1_b=2
        let mpo_a = MPO::<f64>::constant(&[(2, 3)], 1.0);
        let mpo_b = MPO::<f64>::constant(&[(2, 2)], 1.0);

        let result = Contraction::new(mpo_a, mpo_b);
        assert!(result.is_err());
        assert!(matches!(
            result.as_ref(),
            Err(MPOError::SharedDimensionMismatch {
                site: 0,
                dim_a: 3,
                dim_b: 2
            })
        ));
    }

    #[test]
    fn test_contraction_is_empty() {
        let mpo_a = MPO::<f64>::identity(&[2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2]).unwrap();

        let contraction = Contraction::new(mpo_a, mpo_b).unwrap();
        assert!(!contraction.is_empty());
    }

    #[test]
    fn test_contraction_result_site_dims() {
        // A: site_dim (2, 3), B: site_dim (3, 4)
        let mpo_a = MPO::<f64>::constant(&[(2, 3)], 1.0);
        let mpo_b = MPO::<f64>::constant(&[(3, 4)], 1.0);

        let contraction = Contraction::new(mpo_a, mpo_b).unwrap();
        assert_eq!(contraction.result_site_dims(), vec![(2, 4)]);
    }

    #[test]
    fn test_contraction_evaluate_wrong_indices() {
        let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

        let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();

        let result = contraction.evaluate(&[(0, 0)]);
        assert!(result.is_err());

        let result = contraction.evaluate(&[(0, 0), (0, 0), (0, 0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_contraction_clear_cache() {
        let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

        let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();

        let _ = contraction.evaluate_left(2, &[(0, 0), (1, 1)]).unwrap();
        let _ = contraction.evaluate_right(0, &[(0, 0), (1, 1)]).unwrap();

        contraction.clear_cache();

        // After clearing, cache should be empty, but evaluation should still work
        let val = contraction.evaluate(&[(0, 0), (1, 1)]).unwrap();
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_left_boundary() {
        let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

        let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();
        let indices = vec![(0, 0), (1, 1)];

        // n=0 returns 1x1 identity
        let env0 = contraction.evaluate_left(0, &indices).unwrap();
        assert_eq!(env0.dim(0), 1);
        assert_eq!(env0.dim(1), 1);
        assert!((env0[[0, 0]] - 1.0).abs() < 1e-10);

        // n=len returns the full left environment
        let env_full = contraction.evaluate_left(2, &indices).unwrap();
        assert_eq!(env_full.dim(0), 1);
        assert_eq!(env_full.dim(1), 1);

        // Out of range
        let result = contraction.evaluate_left(3, &indices);
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluate_right_boundary() {
        let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

        let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();
        let indices = vec![(0, 0), (1, 1)];

        // n=len returns 1x1 identity
        let env_end = contraction.evaluate_right(2, &indices).unwrap();
        assert_eq!(env_end.dim(0), 1);
        assert_eq!(env_end.dim(1), 1);
        assert!((env_end[[0, 0]] - 1.0).abs() < 1e-10);

        // n=0 returns the full right environment
        let env_full = contraction.evaluate_right(0, &indices).unwrap();
        assert_eq!(env_full.dim(0), 1);
        assert_eq!(env_full.dim(1), 1);

        // Out of range
        let result = contraction.evaluate_right(3, &indices);
        assert!(result.is_err());
    }

    #[test]
    fn test_contraction_two_sites() {
        // Two-site identity contraction: I * I = I at each configuration
        let mpo_a = MPO::<f64>::identity(&[2, 3]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 3]).unwrap();

        let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();
        assert_eq!(contraction.len(), 2);
        assert_eq!(contraction.result_site_dims(), vec![(2, 2), (3, 3)]);

        // (0,0), (0,0) -> delta(0,0)*delta(0,0) = 1
        let val = contraction.evaluate(&[(0, 0), (0, 0)]).unwrap();
        assert!((val - 1.0).abs() < 1e-10);

        // (0,1), (0,0) -> delta(0,k)*delta(k,1) at site 0 = delta(0,1) = 0
        let val = contraction.evaluate(&[(0, 1), (0, 0)]).unwrap();
        assert!(val.abs() < 1e-10);

        // (1,1), (2,2) -> delta(1,1)*delta(2,2) = 1
        let val = contraction.evaluate(&[(1, 1), (2, 2)]).unwrap();
        assert!((val - 1.0).abs() < 1e-10);

        // Verify left/right environments are consistent
        let indices = vec![(1, 1), (2, 2)];
        let left = contraction.evaluate_left(1, &indices).unwrap();
        let right = contraction.evaluate_right(1, &indices).unwrap();

        // Left env after site 0 with (1,1): should be 1x1 with value 1 (diagonal)
        assert!((left[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((right[[0, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_contraction_constant_two_sites() {
        // Constant(2,2) * Constant(2,2) at each site
        // A[i,k] = 1 for all i,k; B[k,j] = 1 for all k,j
        // C[i,j] = sum_k A[i,k]*B[k,j] = sum_k 1 = 2 (shared dim is 2)
        let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
        let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

        let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();

        // At any (i,j) pair at each site, the contraction sums over the shared dim
        // Site 0: sum_k A0[0,i,k,0] * B0[0,k,j,0] = sum_k 1*1 = 2
        // Site 1: sum_k A1[0,i,k,0] * B1[0,k,j,0] = sum_k 1*1 = 2
        // Total = 2 * 2 = 4
        let val = contraction.evaluate(&[(0, 0), (0, 0)]).unwrap();
        assert!((val - 4.0).abs() < 1e-10);

        let val = contraction.evaluate(&[(1, 1), (1, 0)]).unwrap();
        assert!((val - 4.0).abs() < 1e-10);
    }
}
