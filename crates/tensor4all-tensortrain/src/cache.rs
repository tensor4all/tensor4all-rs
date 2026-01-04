//! Cached tensor train evaluation
//!
//! This module provides `TTCache`, a wrapper around tensor trains that caches
//! partial evaluations for efficient repeated evaluation.

use std::collections::HashMap;

use crate::error::{Result, TensorTrainError};
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{LocalIndex, MultiIndex, Tensor3, Tensor3Ops};

/// Cached tensor train for efficient repeated evaluation
///
/// Caches left and right partial contractions to avoid redundant computation
/// when evaluating the same tensor train at multiple index sets that share
/// common prefixes or suffixes.
#[derive(Debug, Clone)]
pub struct TTCache<T: TTScalar> {
    /// The site tensors (reshaped to 3D: left_bond x flat_site x right_bond)
    tensors: Vec<Tensor3<T>>,
    /// Cache for left partial evaluations: site -> (indices -> vector)
    cache_left: Vec<HashMap<MultiIndex, Vec<T>>>,
    /// Cache for right partial evaluations: site -> (indices -> vector)
    cache_right: Vec<HashMap<MultiIndex, Vec<T>>>,
    /// Site dimensions for each tensor (can be multi-dimensional per site)
    site_dims: Vec<Vec<usize>>,
}

impl<T: TTScalar> TTCache<T> {
    /// Create a new TTCache from a tensor train
    pub fn new<TT: AbstractTensorTrain<T>>(tt: &TT) -> Self {
        let n = tt.len();
        let tensors: Vec<Tensor3<T>> = tt.site_tensors().to_vec();
        let site_dims: Vec<Vec<usize>> = tensors
            .iter()
            .map(|t| vec![t.site_dim()])
            .collect();

        Self {
            tensors,
            cache_left: (0..n).map(|_| HashMap::new()).collect(),
            cache_right: (0..n).map(|_| HashMap::new()).collect(),
            site_dims,
        }
    }

    /// Create a new TTCache with custom site dimensions
    ///
    /// This allows treating a single tensor site as multiple logical indices.
    pub fn with_site_dims<TT: AbstractTensorTrain<T>>(tt: &TT, site_dims: Vec<Vec<usize>>) -> Result<Self> {
        let n = tt.len();
        if site_dims.len() != n {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "site_dims length {} doesn't match tensor train length {}",
                    site_dims.len(),
                    n
                ),
            });
        }

        // Validate that site_dims products match tensor site dimensions
        for (i, (tensor, dims)) in tt.site_tensors().iter().zip(site_dims.iter()).enumerate() {
            let expected: usize = dims.iter().product();
            if expected != tensor.site_dim() {
                return Err(TensorTrainError::InvalidOperation {
                    message: format!(
                        "site_dims product {} doesn't match tensor site dim {} at site {}",
                        expected,
                        tensor.site_dim(),
                        i
                    ),
                });
            }
        }

        let tensors: Vec<Tensor3<T>> = tt.site_tensors().to_vec();

        Ok(Self {
            tensors,
            cache_left: (0..n).map(|_| HashMap::new()).collect(),
            cache_right: (0..n).map(|_| HashMap::new()).collect(),
            site_dims,
        })
    }

    /// Number of sites
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get site dimensions
    pub fn site_dims(&self) -> &[Vec<usize>] {
        &self.site_dims
    }

    /// Get link dimensions
    pub fn link_dims(&self) -> Vec<usize> {
        if self.len() <= 1 {
            return Vec::new();
        }
        (1..self.len())
            .map(|i| self.tensors[i].left_dim())
            .collect()
    }

    /// Get link dimension at position i (between site i and i+1)
    pub fn link_dim(&self, i: usize) -> usize {
        self.tensors[i + 1].left_dim()
    }

    /// Clear all cached values
    pub fn clear_cache(&mut self) {
        for cache in &mut self.cache_left {
            cache.clear();
        }
        for cache in &mut self.cache_right {
            cache.clear();
        }
    }

    /// Convert multi-index to flat index for a site
    fn multi_to_flat(&self, site: usize, indices: &[LocalIndex]) -> LocalIndex {
        let dims = &self.site_dims[site];
        let mut flat = 0;
        let mut stride = 1;
        for (i, &idx) in indices.iter().rev().enumerate() {
            flat += idx * stride;
            stride *= dims[dims.len() - 1 - i];
        }
        flat
    }

    /// Evaluate from the left up to (but not including) site `end`
    ///
    /// Returns a vector of size `link_dim(end-1)` (or 1 if end == 0)
    pub fn evaluate_left(&mut self, indices: &[LocalIndex]) -> Vec<T> {
        let ell = indices.len();
        if ell == 0 {
            return vec![T::one()];
        }

        // Check cache
        let key: MultiIndex = indices.to_vec();
        if let Some(cached) = self.cache_left[ell - 1].get(&key) {
            return cached.clone();
        }

        // Compute recursively
        let result = if ell == 1 {
            // First site: just extract the slice
            let flat_idx = self.multi_to_flat(0, &indices[0..1]);
            let tensor = &self.tensors[0];
            let mut v = Vec::with_capacity(tensor.right_dim());
            for r in 0..tensor.right_dim() {
                v.push(*tensor.get3(0, flat_idx, r));
            }
            v
        } else {
            // Recursive case: left[0..ell-1] * tensor[ell-1][:, idx, :]
            let left = self.evaluate_left(&indices[0..ell - 1]);
            let flat_idx = self.multi_to_flat(ell - 1, &indices[ell - 1..ell]);
            let tensor = &self.tensors[ell - 1];

            let mut result = vec![T::zero(); tensor.right_dim()];
            for r in 0..tensor.right_dim() {
                let mut sum = T::zero();
                for l in 0..tensor.left_dim() {
                    sum = sum + left[l] * *tensor.get3(l, flat_idx, r);
                }
                result[r] = sum;
            }
            result
        };

        // Cache and return
        self.cache_left[ell - 1].insert(key, result.clone());
        result
    }

    /// Evaluate from the right starting at site `start`
    ///
    /// `indices` contains indices for sites `start` to `n-1`
    /// Returns a vector of size `link_dim(start-1)` (or 1 if start == n)
    pub fn evaluate_right(&mut self, indices: &[LocalIndex]) -> Vec<T> {
        let n = self.len();
        let ell = indices.len();
        if ell == 0 {
            return vec![T::one()];
        }

        let start = n - ell;

        // Check cache
        let key: MultiIndex = indices.to_vec();
        if let Some(cached) = self.cache_right[start].get(&key) {
            return cached.clone();
        }

        // Compute recursively
        let result = if ell == 1 {
            // Last site: just extract the slice
            let flat_idx = self.multi_to_flat(n - 1, &indices[0..1]);
            let tensor = &self.tensors[n - 1];
            let mut v = Vec::with_capacity(tensor.left_dim());
            for l in 0..tensor.left_dim() {
                v.push(*tensor.get3(l, flat_idx, 0));
            }
            v
        } else {
            // Recursive case: tensor[start][:, idx, :] * right[1..]
            let right = self.evaluate_right(&indices[1..]);
            let flat_idx = self.multi_to_flat(start, &indices[0..1]);
            let tensor = &self.tensors[start];

            let mut result = vec![T::zero(); tensor.left_dim()];
            for l in 0..tensor.left_dim() {
                let mut sum = T::zero();
                for r in 0..tensor.right_dim() {
                    sum = sum + *tensor.get3(l, flat_idx, r) * right[r];
                }
                result[l] = sum;
            }
            result
        };

        // Cache and return
        self.cache_right[start].insert(key, result.clone());
        result
    }

    /// Evaluate the tensor train at a given index set using cache
    pub fn evaluate(&mut self, indices: &[LocalIndex]) -> Result<T> {
        let n = self.len();
        if indices.len() != n {
            return Err(TensorTrainError::IndexLengthMismatch {
                expected: n,
                got: indices.len(),
            });
        }

        if n == 0 {
            return Err(TensorTrainError::Empty);
        }

        // Split at midpoint for efficiency
        let mid = n / 2;
        let left = self.evaluate_left(&indices[0..mid]);
        let right = self.evaluate_right(&indices[mid..]);

        // Contract left and right
        if left.len() != right.len() {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Left/right dimension mismatch: {} vs {}",
                    left.len(),
                    right.len()
                ),
            });
        }

        let mut result = T::zero();
        for i in 0..left.len() {
            result = result + left[i] * right[i];
        }

        Ok(result)
    }

    /// Batch evaluate the tensor train
    ///
    /// Evaluates for all combinations of left_indices and right_indices,
    /// with `n_center` free indices in the middle.
    ///
    /// Returns a tensor of shape (n_left, center_dims..., n_right)
    pub fn batch_evaluate(
        &mut self,
        left_indices: &[MultiIndex],
        right_indices: &[MultiIndex],
        n_center: usize,
    ) -> Result<(Vec<T>, Vec<usize>)> {
        let n = self.len();

        if left_indices.is_empty() || right_indices.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let n_left_sites = if left_indices.is_empty() {
            0
        } else {
            left_indices[0].len()
        };
        let n_right_sites = if right_indices.is_empty() {
            0
        } else {
            right_indices[0].len()
        };

        if n_left_sites + n_center + n_right_sites != n {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Index count mismatch: {} + {} + {} != {}",
                    n_left_sites, n_center, n_right_sites, n
                ),
            });
        }

        let n_left = left_indices.len();
        let n_right = right_indices.len();

        // Compute left environments
        let dl = if n_left_sites > 0 && n_left_sites < n {
            self.link_dim(n_left_sites - 1)
        } else {
            1
        };

        let mut left_env = vec![T::one(); n_left * dl];
        if n_left_sites > 0 {
            for (il, lindex) in left_indices.iter().enumerate() {
                let lenv = self.evaluate_left(lindex);
                for (j, &v) in lenv.iter().enumerate() {
                    left_env[il * dl + j] = v;
                }
            }
        }

        // Compute right environments
        let dr = if n_right_sites > 0 && n_left_sites + n_center < n {
            self.link_dim(n_left_sites + n_center - 1)
        } else {
            1
        };

        let mut right_env = vec![T::one(); dr * n_right];
        if n_right_sites > 0 {
            for (ir, rindex) in right_indices.iter().enumerate() {
                let renv = self.evaluate_right(rindex);
                for (j, &v) in renv.iter().enumerate() {
                    right_env[j * n_right + ir] = v;
                }
            }
        }

        // Compute center dimensions
        let mut center_dims = Vec::with_capacity(n_center);
        for i in n_left_sites..n_left_sites + n_center {
            center_dims.push(self.tensors[i].site_dim());
        }
        let center_size: usize = center_dims.iter().product();

        // Contract through center tensors
        // current shape: (n_left, current_bond_dim)
        let mut current = left_env;
        let mut current_bond = dl;

        for i in n_left_sites..n_left_sites + n_center {
            let tensor = &self.tensors[i];
            let site_dim = tensor.site_dim();
            let next_bond = tensor.right_dim();

            // new_current[il, s, r] = sum_l current[il, l] * tensor[l, s, r]
            let mut new_current = vec![T::zero(); n_left * site_dim * next_bond];

            for il in 0..n_left {
                for s in 0..site_dim {
                    for r in 0..next_bond {
                        let mut sum = T::zero();
                        for l in 0..current_bond {
                            sum = sum + current[il * current_bond + l] * *tensor.get3(l, s, r);
                        }
                        new_current[(il * site_dim + s) * next_bond + r] = sum;
                    }
                }
            }

            current = new_current;
            current_bond = next_bond;
        }

        // Contract with right environment
        // current shape: (n_left * center_size, dr)
        // right_env shape: (dr, n_right)
        // result shape: (n_left * center_size, n_right) -> (n_left, center_dims..., n_right)
        let mut result = vec![T::zero(); n_left * center_size * n_right];

        for il_c in 0..n_left * center_size {
            for ir in 0..n_right {
                let mut sum = T::zero();
                for d in 0..dr {
                    sum = sum + current[il_c * current_bond + d] * right_env[d * n_right + ir];
                }
                result[il_c * n_right + ir] = sum;
            }
        }

        // Build output shape
        let mut shape = Vec::with_capacity(2 + n_center);
        shape.push(n_left);
        shape.extend_from_slice(&center_dims);
        shape.push(n_right);

        Ok((result, shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensortrain::TensorTrain;
    use crate::types::tensor3_zeros;

    #[test]
    fn test_ttcache_evaluate() {
        let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
        let mut cache = TTCache::new(&tt);

        // Evaluate at various indices
        let val = cache.evaluate(&[0, 0, 0]).unwrap();
        assert!((val - 2.0).abs() < 1e-10);

        let val = cache.evaluate(&[1, 2, 1]).unwrap();
        assert!((val - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ttcache_caching() {
        let mut t0: Tensor3<f64> = tensor3_zeros(1, 2, 2);
        t0.set3(0, 0, 0, 1.0);
        t0.set3(0, 0, 1, 0.5);
        t0.set3(0, 1, 0, 2.0);
        t0.set3(0, 1, 1, 1.0);

        let mut t1: Tensor3<f64> = tensor3_zeros(2, 3, 1);
        for l in 0..2 {
            for s in 0..3 {
                t1.set3(l, s, 0, (l + s + 1) as f64);
            }
        }

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();
        let mut cache = TTCache::new(&tt);

        // First evaluation
        let val1 = cache.evaluate(&[0, 1]).unwrap();

        // Should be cached now
        assert!(!cache.cache_left[0].is_empty());

        // Second evaluation should use cache
        let val2 = cache.evaluate(&[0, 1]).unwrap();
        assert!((val1 - val2).abs() < 1e-10);
    }

    #[test]
    fn test_ttcache_batch_evaluate() {
        let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
        let mut cache = TTCache::new(&tt);

        let left_indices = vec![vec![0], vec![1]];
        let right_indices = vec![vec![0], vec![1]];

        let (result, shape) = cache.batch_evaluate(&left_indices, &right_indices, 1).unwrap();

        // Shape should be (2, 3, 2)
        assert_eq!(shape, vec![2, 3, 2]);
        assert_eq!(result.len(), 2 * 3 * 2);

        // All values should be 1.0
        for val in &result {
            assert!((val - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ttcache_clear() {
        let tt = TensorTrain::<f64>::constant(&[2, 3], 1.0);
        let mut cache = TTCache::new(&tt);

        // Populate cache
        let _ = cache.evaluate(&[0, 0]);
        assert!(!cache.cache_left[0].is_empty());

        // Clear cache
        cache.clear_cache();
        assert!(cache.cache_left[0].is_empty());
        assert!(cache.cache_right[0].is_empty());
    }
}
