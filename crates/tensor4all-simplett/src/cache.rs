//! Cached tensor train evaluation
//!
//! This module provides `TTCache`, a wrapper around tensor trains that caches
//! partial evaluations for efficient repeated evaluation.

use std::collections::{HashMap, HashSet};

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
        let site_dims: Vec<Vec<usize>> = tensors.iter().map(|t| vec![t.site_dim()]).collect();

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
    pub fn with_site_dims<TT: AbstractTensorTrain<T>>(
        tt: &TT,
        site_dims: Vec<Vec<usize>>,
    ) -> Result<Self> {
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
            #[allow(clippy::needless_range_loop)]
            for r in 0..tensor.right_dim() {
                let mut sum = T::zero();
                #[allow(clippy::needless_range_loop)]
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
            #[allow(clippy::needless_range_loop)]
            for l in 0..tensor.left_dim() {
                let mut sum = T::zero();
                #[allow(clippy::needless_range_loop)]
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

    /// Batch evaluate at multiple index sets
    ///
    /// This method efficiently evaluates the tensor train at multiple indices
    /// by automatically finding the optimal split position to maximize cache reuse.
    pub fn evaluate_many(&mut self, indices: &[MultiIndex]) -> Result<Vec<T>> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        let n = self.len();
        if n == 0 {
            return Err(TensorTrainError::Empty);
        }

        // Validate all indices have correct length
        for idx in indices.iter() {
            if idx.len() != n {
                return Err(TensorTrainError::IndexLengthMismatch {
                    expected: n,
                    got: idx.len(),
                });
            }
        }

        // Find optimal split position
        let split = self.find_optimal_split(indices);

        // Extract unique left and right parts with index mapping
        let mut unique_left: Vec<MultiIndex> = Vec::new();
        let mut unique_right: Vec<MultiIndex> = Vec::new();
        let mut left_map: HashMap<MultiIndex, usize> = HashMap::new();
        let mut right_map: HashMap<MultiIndex, usize> = HashMap::new();

        for idx in indices {
            let left_part: MultiIndex = idx[..split].to_vec();
            let right_part: MultiIndex = idx[split..].to_vec();

            if !left_map.contains_key(&left_part) {
                left_map.insert(left_part.clone(), unique_left.len());
                unique_left.push(left_part);
            }
            if !right_map.contains_key(&right_part) {
                right_map.insert(right_part.clone(), unique_right.len());
                unique_right.push(right_part);
            }
        }

        // Compute left environments for all unique left parts
        let left_envs: Vec<Vec<T>> = unique_left.iter().map(|l| self.evaluate_left(l)).collect();

        // Compute right environments for all unique right parts
        let right_envs: Vec<Vec<T>> = unique_right
            .iter()
            .map(|r| self.evaluate_right(r))
            .collect();

        // Compute results for each index
        let mut results = Vec::with_capacity(indices.len());
        for idx in indices {
            let left_part: MultiIndex = idx[..split].to_vec();
            let right_part: MultiIndex = idx[split..].to_vec();

            let il = left_map[&left_part];
            let ir = right_map[&right_part];

            let left_env = &left_envs[il];
            let right_env = &right_envs[ir];

            // Inner product
            let mut sum = T::zero();
            for (l, r) in left_env.iter().zip(right_env.iter()) {
                sum = sum + *l * *r;
            }
            results.push(sum);
        }

        Ok(results)
    }

    /// Find the optimal split position that minimizes the number of unique
    /// left and right parts (maximizing cache reuse)
    fn find_optimal_split(&self, indices: &[MultiIndex]) -> usize {
        let n = self.len();
        if n <= 1 {
            return n;
        }

        (1..n)
            .min_by_key(|&split| {
                let unique_left: HashSet<&[usize]> =
                    indices.iter().map(|idx| &idx[..split]).collect();
                let unique_right: HashSet<&[usize]> =
                    indices.iter().map(|idx| &idx[split..]).collect();
                unique_left.len() + unique_right.len()
            })
            .unwrap_or(n / 2)
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

    #[test]
    fn test_ttcache_evaluate_many() {
        let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
        let mut cache = TTCache::new(&tt);

        let indices = vec![vec![0, 0, 0], vec![0, 1, 0], vec![1, 2, 1], vec![0, 0, 1]];

        let results = cache.evaluate_many(&indices).unwrap();

        // All values should be 2.0 for a constant TT
        assert_eq!(results.len(), 4);
        for val in &results {
            assert!((val - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ttcache_evaluate_many_matches_single() {
        // Create a non-trivial TT
        let mut t0: Tensor3<f64> = tensor3_zeros(1, 2, 2);
        t0.set3(0, 0, 0, 1.0);
        t0.set3(0, 0, 1, 0.5);
        t0.set3(0, 1, 0, 2.0);
        t0.set3(0, 1, 1, 1.0);

        let mut t1: Tensor3<f64> = tensor3_zeros(2, 3, 2);
        for l in 0..2 {
            for s in 0..3 {
                for r in 0..2 {
                    t1.set3(l, s, r, ((l + s + r) as f64) * 0.5 + 0.1);
                }
            }
        }

        let mut t2: Tensor3<f64> = tensor3_zeros(2, 2, 1);
        for l in 0..2 {
            for s in 0..2 {
                t2.set3(l, s, 0, (l + s + 1) as f64);
            }
        }

        let tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();
        let mut cache = TTCache::new(&tt);

        // Generate all indices
        let mut indices = Vec::new();
        for i0 in 0..2 {
            for i1 in 0..3 {
                for i2 in 0..2 {
                    indices.push(vec![i0, i1, i2]);
                }
            }
        }

        // Evaluate using evaluate_many
        let batch_results = cache.evaluate_many(&indices).unwrap();

        // Compare with single evaluations
        for (idx, batch_val) in indices.iter().zip(batch_results.iter()) {
            let single_val = cache.evaluate(idx).unwrap();
            assert!(
                (batch_val - single_val).abs() < 1e-10,
                "Mismatch at {:?}: batch={}, single={}",
                idx,
                batch_val,
                single_val
            );
        }
    }

    #[test]
    fn test_ttcache_evaluate_many_cache_efficiency() {
        let tt = TensorTrain::<f64>::constant(&[2, 2, 2, 2], 1.0);
        let mut cache = TTCache::new(&tt);

        // Indices with shared prefixes/suffixes
        let indices = vec![
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 1],
            vec![0, 0, 1, 0],
            vec![0, 0, 1, 1],
            vec![1, 1, 0, 0],
            vec![1, 1, 0, 1],
        ];

        let results = cache.evaluate_many(&indices).unwrap();
        assert_eq!(results.len(), 6);

        // Check that caches are populated
        // The optimal split should create fewer unique entries than indices.len()
        let total_left_cached: usize = cache.cache_left.iter().map(|c| c.len()).sum();
        let total_right_cached: usize = cache.cache_right.iter().map(|c| c.len()).sum();

        // With shared prefixes/suffixes, we should have fewer cached entries
        // than if we computed each index independently
        assert!(total_left_cached + total_right_cached > 0);
    }

    #[test]
    fn test_ttcache_evaluate_many_empty() {
        let tt = TensorTrain::<f64>::constant(&[2, 3], 1.0);
        let mut cache = TTCache::new(&tt);

        let results = cache.evaluate_many(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_optimal_split() {
        let tt = TensorTrain::<f64>::constant(&[2, 2, 2, 2], 1.0);
        let cache = TTCache::new(&tt);

        // Indices where split=1 is optimal:
        // split=1: unique_left={[0],[1]}=2, unique_right={[0,0,0],[1,0,0]}=2, total=4
        // split=2: unique_left=4, unique_right=1, total=5
        let indices = vec![
            vec![0, 0, 0, 0],
            vec![0, 1, 0, 0],
            vec![1, 0, 0, 0],
            vec![1, 1, 0, 0],
        ];

        let split = cache.find_optimal_split(&indices);
        assert_eq!(split, 1);

        // Indices where split=3 is optimal:
        // split=1: unique_left=1, unique_right=4, total=5
        // split=2: unique_left=1, unique_right=4, total=5
        // split=3: unique_left={[0,0,0],[0,0,1]}=2, unique_right={[0],[1]}=2, total=4
        let indices2 = vec![
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 1],
            vec![0, 0, 1, 0],
            vec![0, 0, 1, 1],
        ];

        let split2 = cache.find_optimal_split(&indices2);
        assert_eq!(split2, 3);
    }
}
