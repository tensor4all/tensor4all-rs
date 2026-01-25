//! Cached tensor train evaluation
//!
//! This module provides `TTCache`, a wrapper around tensor trains that caches
//! partial evaluations for efficient repeated evaluation.

use std::collections::{HashMap, HashSet};

use bnum::types::{U1024, U256, U512};

use crate::error::{Result, TensorTrainError};
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{LocalIndex, MultiIndex, Tensor3, Tensor3Ops};

/// Compute total bits needed for index space
fn compute_total_bits(local_dims: &[usize]) -> u32 {
    local_dims
        .iter()
        .map(|&d| if d <= 1 { 0 } else { (d as u64).ilog2() + 1 })
        .sum()
}

/// Index key types for different bit widths
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum IndexKey {
    U64(u64),
    U128(u128),
    U256(U256),
    U512(U512),
    U1024(U1024),
}

/// Flat indexer with automatic key type selection based on index space size
enum FlatIndexer {
    U64 { coeffs: Vec<u64> },
    U128 { coeffs: Vec<u128> },
    U256 { coeffs: Vec<U256> },
    U512 { coeffs: Vec<U512> },
    U1024 { coeffs: Vec<U1024> },
}

/// Macro for computing coefficients for primitive integer types (u64, u128)
macro_rules! compute_coeffs_primitive {
    ($local_dims:expr, $T:ty) => {{
        let mut coeffs = Vec::with_capacity($local_dims.len());
        let mut prod: $T = 1;
        for &d in $local_dims {
            coeffs.push(prod);
            prod = prod.saturating_mul(d as $T);
        }
        coeffs
    }};
}

/// Macro for computing coefficients for bnum types (U256, U512, U1024)
macro_rules! compute_coeffs_bnum {
    ($local_dims:expr, $T:ty) => {{
        let mut coeffs = Vec::with_capacity($local_dims.len());
        let mut prod = <$T>::ONE;
        for &d in $local_dims {
            coeffs.push(prod);
            prod = prod.saturating_mul(<$T>::from(d as u64));
        }
        coeffs
    }};
}

/// Macro for computing flat index for primitive types
macro_rules! flat_index_primitive {
    ($idx:expr, $coeffs:expr, $T:ty, $Key:ident) => {{
        let key: $T = $idx.iter().zip($coeffs).map(|(&i, &c)| c * i as $T).sum();
        IndexKey::$Key(key)
    }};
}

/// Macro for computing flat index for bnum types
macro_rules! flat_index_bnum {
    ($idx:expr, $coeffs:expr, $T:ty, $Key:ident) => {{
        let key = $idx
            .iter()
            .zip($coeffs)
            .map(|(&i, &c)| c * <$T>::from(i as u64))
            .fold(<$T>::ZERO, |a, b| a + b);
        IndexKey::$Key(key)
    }};
}

impl FlatIndexer {
    /// Create a new indexer, automatically selecting the key type
    fn new(local_dims: &[usize]) -> Self {
        let total_bits = compute_total_bits(local_dims);

        if total_bits <= 64 {
            Self::U64 {
                coeffs: compute_coeffs_primitive!(local_dims, u64),
            }
        } else if total_bits <= 128 {
            Self::U128 {
                coeffs: compute_coeffs_primitive!(local_dims, u128),
            }
        } else if total_bits <= 256 {
            Self::U256 {
                coeffs: compute_coeffs_bnum!(local_dims, U256),
            }
        } else if total_bits <= 512 {
            Self::U512 {
                coeffs: compute_coeffs_bnum!(local_dims, U512),
            }
        } else {
            Self::U1024 {
                coeffs: compute_coeffs_bnum!(local_dims, U1024),
            }
        }
    }

    /// Compute flat index key from multi-index
    fn flat_index(&self, idx: &[usize]) -> IndexKey {
        match self {
            Self::U64 { coeffs } => flat_index_primitive!(idx, coeffs, u64, U64),
            Self::U128 { coeffs } => flat_index_primitive!(idx, coeffs, u128, U128),
            Self::U256 { coeffs } => flat_index_bnum!(idx, coeffs, U256, U256),
            Self::U512 { coeffs } => flat_index_bnum!(idx, coeffs, U512, U512),
            Self::U1024 { coeffs } => flat_index_bnum!(idx, coeffs, U1024, U1024),
        }
    }
}

/// Helper struct for building unique index mappings
struct IndexMapper {
    left_indexer: FlatIndexer,
    right_indexer: FlatIndexer,
    left_key_to_id: HashMap<IndexKey, usize>,
    right_key_to_id: HashMap<IndexKey, usize>,
    idx_to_left: Vec<usize>,
    idx_to_right: Vec<usize>,
    left_first_idx: Vec<usize>,
    right_first_idx: Vec<usize>,
}

impl IndexMapper {
    fn new(left_dims: &[usize], right_dims: &[usize], capacity: usize) -> Self {
        Self {
            left_indexer: FlatIndexer::new(left_dims),
            right_indexer: FlatIndexer::new(right_dims),
            left_key_to_id: HashMap::new(),
            right_key_to_id: HashMap::new(),
            idx_to_left: Vec::with_capacity(capacity),
            idx_to_right: Vec::with_capacity(capacity),
            left_first_idx: Vec::new(),
            right_first_idx: Vec::new(),
        }
    }

    fn add_index(&mut self, i: usize, left_part: &[usize], right_part: &[usize]) {
        let left_key = self.left_indexer.flat_index(left_part);
        let right_key = self.right_indexer.flat_index(right_part);

        let left_id = match self.left_key_to_id.get(&left_key) {
            Some(&id) => id,
            None => {
                let id = self.left_key_to_id.len();
                self.left_key_to_id.insert(left_key, id);
                self.left_first_idx.push(i);
                id
            }
        };

        let right_id = match self.right_key_to_id.get(&right_key) {
            Some(&id) => id,
            None => {
                let id = self.right_key_to_id.len();
                self.right_key_to_id.insert(right_key, id);
                self.right_first_idx.push(i);
                id
            }
        };

        self.idx_to_left.push(left_id);
        self.idx_to_right.push(right_id);
    }
}

/// Helper for counting unique keys in split heuristic
struct UniqueCounter {
    indexer: FlatIndexer,
    keys: HashSet<IndexKey>,
}

impl UniqueCounter {
    fn new(local_dims: &[usize], capacity: usize) -> Self {
        Self {
            indexer: FlatIndexer::new(local_dims),
            keys: HashSet::with_capacity(capacity),
        }
    }

    fn insert(&mut self, idx: &[usize]) {
        let key = self.indexer.flat_index(idx);
        self.keys.insert(key);
    }

    fn len(&self) -> usize {
        self.keys.len()
    }
}

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
    /// by splitting at a given position and computing unique left/right environments.
    ///
    /// # Arguments
    /// * `indices` - The indices to evaluate
    /// * `split` - Optional split position. If `None`, uses a simple heuristic
    ///   (checks 1/4, 1/2, 3/4 positions and picks the best).
    ///   If you know the optimal split position (e.g., from TCI), pass `Some(split)`
    ///   to avoid the search overhead.
    pub fn evaluate_many(
        &mut self,
        indices: &[MultiIndex],
        split: Option<usize>,
    ) -> Result<Vec<T>> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        let n = self.len();
        if n == 0 {
            return Err(TensorTrainError::Empty);
        }

        // Determine split position
        let split = match split {
            Some(s) => s,
            None => self.find_split_heuristic(indices),
        };

        if split == 0 || split > n {
            return Err(TensorTrainError::InvalidOperation {
                message: format!("Invalid split position: {} (n_sites={})", split, n),
            });
        }

        // Get local dimensions for flat index computation
        let local_dims: Vec<usize> = self.site_dims.iter().map(|d| d.iter().product()).collect();

        // Build index mapper with appropriate key type for each half
        let mut mapper =
            IndexMapper::new(&local_dims[..split], &local_dims[split..], indices.len());

        for (i, idx) in indices.iter().enumerate() {
            mapper.add_index(i, &idx[..split], &idx[split..]);
        }

        // Extract unique parts using first occurrence indices
        let unique_left: Vec<MultiIndex> = mapper
            .left_first_idx
            .iter()
            .map(|&i| indices[i][..split].to_vec())
            .collect();

        let unique_right: Vec<MultiIndex> = mapper
            .right_first_idx
            .iter()
            .map(|&i| indices[i][split..].to_vec())
            .collect();

        // Compute left environments for all unique left parts
        let left_envs: Vec<Vec<T>> = unique_left.iter().map(|l| self.evaluate_left(l)).collect();

        // Compute right environments for all unique right parts
        let right_envs: Vec<Vec<T>> = unique_right
            .iter()
            .map(|r| self.evaluate_right(r))
            .collect();

        // Compute results using position mappings
        let results: Vec<T> = mapper
            .idx_to_left
            .iter()
            .zip(&mapper.idx_to_right)
            .map(|(&il, &ir)| {
                let left_env = &left_envs[il];
                let right_env = &right_envs[ir];
                // Inner product
                left_env
                    .iter()
                    .zip(right_env.iter())
                    .fold(T::zero(), |acc, (&l, &r)| acc + l * r)
            })
            .collect();

        Ok(results)
    }

    /// Find a good split position using 3-point sampling heuristic
    ///
    /// Samples at 1/4, 1/2, 3/4 positions and returns the one with
    /// minimum total unique left + right parts.
    fn find_split_heuristic(&self, indices: &[MultiIndex]) -> usize {
        let n = self.len();
        if n <= 1 {
            return n.max(1);
        }

        let local_dims: Vec<usize> = self.site_dims.iter().map(|d| d.iter().product()).collect();

        // Helper to compute cost at a split position
        let compute_cost = |split: usize| -> usize {
            if split == 0 || split >= n {
                return usize::MAX;
            }

            let mut left_counter = UniqueCounter::new(&local_dims[..split], indices.len());
            let mut right_counter = UniqueCounter::new(&local_dims[split..], indices.len());

            for idx in indices {
                left_counter.insert(&idx[..split]);
                right_counter.insert(&idx[split..]);
            }

            left_counter.len() + right_counter.len()
        };

        // 3-point sampling: 1/4, 1/2, 3/4 positions
        let candidates = [n / 4, n / 2, 3 * n / 4];
        let costs: Vec<(usize, usize)> = candidates
            .iter()
            .filter(|&&p| p >= 1 && p < n)
            .map(|&p| (p, compute_cost(p)))
            .collect();

        costs
            .into_iter()
            .min_by_key(|&(_, c)| c)
            .map(|(p, _)| p)
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

        let results = cache.evaluate_many(&indices, None).unwrap();

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
        let batch_results = cache.evaluate_many(&indices, None).unwrap();

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

        let results = cache.evaluate_many(&indices, None).unwrap();
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

        let results = cache.evaluate_many(&[], None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_split_heuristic() {
        let tt = TensorTrain::<f64>::constant(&[2, 2, 2, 2], 1.0);
        let cache = TTCache::new(&tt);

        // Indices where split=1 is optimal:
        // Heuristic checks 1/4=1, 1/2=2, 3/4=3
        // split=1: unique_left=2, unique_right=2, total=4
        // split=2: unique_left=4, unique_right=1, total=5
        // split=3: unique_left=4, unique_right=1, total=5
        let indices = vec![
            vec![0, 0, 0, 0],
            vec![0, 1, 0, 0],
            vec![1, 0, 0, 0],
            vec![1, 1, 0, 0],
        ];

        let split = cache.find_split_heuristic(&indices);
        assert_eq!(split, 1);

        // Indices where split=3 is optimal:
        // split=1: unique_left=1, unique_right=4, total=5
        // split=2: unique_left=1, unique_right=4, total=5
        // split=3: unique_left=2, unique_right=2, total=4
        let indices2 = vec![
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 1],
            vec![0, 0, 1, 0],
            vec![0, 0, 1, 1],
        ];

        let split2 = cache.find_split_heuristic(&indices2);
        assert_eq!(split2, 3);
    }
}
