//! Cached tensor train evaluation
//!
//! This module provides `TTCache`, a wrapper around tensor trains that caches
//! partial evaluations for efficient repeated evaluation.

use std::collections::hash_map::Entry;

use rustc_hash::{FxHashMap, FxHashSet};
use tensor4all_core::index_key::{FlatIndexer, IndexKey, IndexKeyError};

use crate::einsum_helper::{matrix_times_col_vector, row_vector_times_matrix};
use crate::error::{Result, SimpleTensorTrainError};
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{LocalIndex, MultiIndex, Tensor3, Tensor3Ops};

fn cache_key_error(error: IndexKeyError) -> SimpleTensorTrainError {
    SimpleTensorTrainError::InvalidOperation {
        message: format!("cache index encoding failed: {error}"),
    }
}

/// Helper struct for building unique index mappings
struct IndexMapper {
    left_indexer: FlatIndexer,
    right_indexer: FlatIndexer,
    left_key_to_id: FxHashMap<IndexKey, usize>,
    right_key_to_id: FxHashMap<IndexKey, usize>,
    idx_to_left: Vec<usize>,
    idx_to_right: Vec<usize>,
    left_first_idx: Vec<usize>,
    right_first_idx: Vec<usize>,
}

impl IndexMapper {
    fn new(left_dims: &[usize], right_dims: &[usize], capacity: usize) -> Result<Self> {
        Ok(Self {
            left_indexer: FlatIndexer::try_new(left_dims).map_err(cache_key_error)?,
            right_indexer: FlatIndexer::try_new(right_dims).map_err(cache_key_error)?,
            left_key_to_id: FxHashMap::default(),
            right_key_to_id: FxHashMap::default(),
            idx_to_left: Vec::with_capacity(capacity),
            idx_to_right: Vec::with_capacity(capacity),
            left_first_idx: Vec::new(),
            right_first_idx: Vec::new(),
        })
    }

    fn add_index(&mut self, i: usize, left_part: &[usize], right_part: &[usize]) -> Result<()> {
        let left_key = self
            .left_indexer
            .encode(left_part)
            .map_err(cache_key_error)?;
        let right_key = self
            .right_indexer
            .encode(right_part)
            .map_err(cache_key_error)?;

        let next_left = self.left_key_to_id.len();
        let left_id = *match self.left_key_to_id.entry(left_key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                self.left_first_idx.push(i);
                entry.insert(next_left)
            }
        };
        let next_right = self.right_key_to_id.len();
        let right_id = *match self.right_key_to_id.entry(right_key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                self.right_first_idx.push(i);
                entry.insert(next_right)
            }
        };

        self.idx_to_left.push(left_id);
        self.idx_to_right.push(right_id);
        Ok(())
    }
}

/// Helper for counting unique keys in split heuristic
struct UniqueCounter {
    indexer: FlatIndexer,
    keys: FxHashSet<IndexKey>,
}

impl UniqueCounter {
    fn new(local_dims: &[usize], capacity: usize) -> Result<Self> {
        Ok(Self {
            indexer: FlatIndexer::try_new(local_dims).map_err(cache_key_error)?,
            keys: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
        })
    }

    fn insert(&mut self, idx: &[usize]) -> Result<()> {
        let key = self.indexer.encode(idx).map_err(cache_key_error)?;
        self.keys.insert(key);
        Ok(())
    }

    fn len(&self) -> usize {
        self.keys.len()
    }
}

/// Cached tensor train evaluator.
///
/// Wraps a tensor train and caches left and right partial contractions so
/// that repeated evaluations sharing common prefixes or suffixes reuse
/// previously computed results. This is particularly effective for
/// batch evaluation via [`evaluate_many`](Self::evaluate_many).
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain, TTCache};
///
/// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 5.0);
/// let mut cache = TTCache::new(&tt);
///
/// // Single evaluation (caches intermediate contractions)
/// let val = cache.evaluate(&[1, 2, 3]).unwrap();
/// assert!((val - 5.0).abs() < 1e-12);
///
/// // Batch evaluation reuses cached partial contractions
/// let indices = vec![vec![0, 0, 0], vec![1, 2, 3], vec![0, 1, 2]];
/// let vals = cache.evaluate_many(&indices, None).unwrap();
/// assert!(vals.iter().all(|&v| (v - 5.0).abs() < 1e-12));
/// ```
#[derive(Debug, Clone)]
pub struct TTCache<T: TTScalar> {
    /// The site tensors (reshaped to 3D: left_bond x flat_site x right_bond)
    tensors: Vec<Tensor3<T>>,
    /// Cache for left partial evaluations: site -> (indices -> vector)
    cache_left: Vec<std::collections::HashMap<MultiIndex, Vec<T>>>,
    /// Cache for right partial evaluations: site -> (indices -> vector)
    cache_right: Vec<std::collections::HashMap<MultiIndex, Vec<T>>>,
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
            cache_left: (0..n).map(|_| std::collections::HashMap::new()).collect(),
            cache_right: (0..n).map(|_| std::collections::HashMap::new()).collect(),
            site_dims,
        }
    }

    /// Create a new TTCache with custom site dimensions
    ///
    /// This allows treating a single tensor site as multiple logical indices.
    /// # Errors
    ///
    /// Returns [`SimpleTensorTrainError::InvalidOperation`] when `site_dims` has
    /// a different length than the train, when an entry is empty, when a
    /// dimension product overflows `usize`, or when a product does not match
    /// the corresponding site dimension.
    ///
    pub fn with_site_dims<TT: AbstractTensorTrain<T>>(
        tt: &TT,
        site_dims: Vec<Vec<usize>>,
    ) -> Result<Self> {
        let n = tt.len();
        if site_dims.len() != n {
            return Err(SimpleTensorTrainError::InvalidOperation {
                message: format!(
                    "site_dims length {} doesn't match tensor train length {}",
                    site_dims.len(),
                    n
                ),
            });
        }

        // Validate that site_dims products match tensor site dimensions
        for (i, (tensor, dims)) in tt.site_tensors().iter().zip(site_dims.iter()).enumerate() {
            if dims.is_empty() {
                return Err(SimpleTensorTrainError::InvalidOperation {
                    message: format!("site_dims at site {i} must contain at least one dimension"),
                });
            }
            let expected = dims
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
            let Some(expected) = expected else {
                return Err(SimpleTensorTrainError::InvalidOperation {
                    message: format!("site_dims product overflows usize at site {i}"),
                });
            };
            if expected != tensor.site_dim() {
                return Err(SimpleTensorTrainError::InvalidOperation {
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
            cache_left: (0..n).map(|_| std::collections::HashMap::new()).collect(),
            cache_right: (0..n).map(|_| std::collections::HashMap::new()).collect(),
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
    fn multi_to_flat(&self, site: usize, indices: &[LocalIndex]) -> Result<LocalIndex> {
        let dims = &self.site_dims[site];
        let mut flat = 0usize;
        let mut stride = 1usize;
        for (i, &idx) in indices.iter().rev().enumerate() {
            let term = idx.checked_mul(stride).ok_or_else(|| {
                SimpleTensorTrainError::InvalidOperation {
                    message: format!("flat index offset overflowed at site {site}"),
                }
            })?;
            flat =
                flat.checked_add(term)
                    .ok_or_else(|| SimpleTensorTrainError::InvalidOperation {
                        message: format!("flat index offset overflowed at site {site}"),
                    })?;
            stride = stride
                .checked_mul(dims[dims.len() - 1 - i])
                .ok_or_else(|| SimpleTensorTrainError::InvalidOperation {
                    message: format!("flat index stride overflowed at site {site}"),
                })?;
        }
        Ok(flat)
    }

    fn flat_site_dim(&self, site: usize) -> Result<usize> {
        self.site_dims[site]
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| SimpleTensorTrainError::InvalidOperation {
                message: format!("Site dimension product overflowed at site {site}"),
            })
    }

    fn validate_partial_indices(&self, start: usize, indices: &[LocalIndex]) -> Result<()> {
        let end = start.checked_add(indices.len()).ok_or_else(|| {
            SimpleTensorTrainError::InvalidOperation {
                message: "Partial index range overflowed usize".to_string(),
            }
        })?;
        if end > self.len() {
            return Err(SimpleTensorTrainError::IndexLengthMismatch {
                expected: self.len(),
                got: end,
            });
        }

        for (offset, &index) in indices.iter().enumerate() {
            let site = start + offset;
            let max = self.flat_site_dim(site)?;
            if index >= max {
                return Err(SimpleTensorTrainError::IndexOutOfBounds { site, index, max });
            }
        }

        Ok(())
    }

    /// Evaluate from the left up to (but not including) site `end`
    ///
    /// Returns a vector of size `link_dim(end-1)` (or 1 if end == 0)
    ///
    /// # Errors
    ///
    /// Returns an error if `indices` spans more sites than the tensor train, if
    /// any index is out of bounds for its site, or if the tenferro-backed
    /// matrix-vector contraction fails.
    pub fn evaluate_left(&mut self, indices: &[LocalIndex]) -> Result<Vec<T>> {
        let ell = indices.len();
        self.validate_partial_indices(0, indices)?;

        if ell == 0 {
            return Ok(vec![T::one()]);
        }

        // Check cache
        let key: MultiIndex = indices.to_vec();
        if let Some(cached) = self.cache_left[ell - 1].get(&key) {
            return Ok(cached.clone());
        }

        // Compute recursively
        let result = if ell == 1 {
            // First site: just extract the slice
            let flat_idx = self.multi_to_flat(0, &indices[0..1])?;
            let tensor = &self.tensors[0];
            tensor.slice_site(flat_idx)
        } else {
            // Recursive case: left[0..ell-1] * tensor[ell-1][:, idx, :]
            let left = self.evaluate_left(&indices[0..ell - 1])?;
            let flat_idx = self.multi_to_flat(ell - 1, &indices[ell - 1..ell])?;
            let tensor = &self.tensors[ell - 1];
            let slice = tensor.slice_site(flat_idx);
            row_vector_times_matrix(&left, &slice, tensor.left_dim(), tensor.right_dim()).map_err(
                |err| SimpleTensorTrainError::InvalidOperation {
                    message: format!("Failed to evaluate left environment at site {ell}: {err}"),
                },
            )?
        };

        // Cache and return
        self.cache_left[ell - 1].insert(key, result.clone());
        Ok(result)
    }

    /// Evaluate from the right starting at site `start`
    ///
    /// `indices` contains indices for sites `start` to `n-1`
    /// Returns a vector of size `link_dim(start-1)` (or 1 if start == n)
    ///
    /// # Errors
    ///
    /// Returns an error if `indices` spans more sites than the tensor train, if
    /// any index is out of bounds for its site, or if the tenferro-backed
    /// matrix-vector contraction fails.
    pub fn evaluate_right(&mut self, indices: &[LocalIndex]) -> Result<Vec<T>> {
        let n = self.len();
        let ell = indices.len();
        if ell > n {
            return Err(SimpleTensorTrainError::IndexLengthMismatch {
                expected: n,
                got: ell,
            });
        }

        if ell == 0 {
            return Ok(vec![T::one()]);
        }

        let start = n - ell;
        self.validate_partial_indices(start, indices)?;

        // Check cache
        let key: MultiIndex = indices.to_vec();
        if let Some(cached) = self.cache_right[start].get(&key) {
            return Ok(cached.clone());
        }

        // Compute recursively
        let result = if ell == 1 {
            // Last site: just extract the slice
            let flat_idx = self.multi_to_flat(n - 1, &indices[0..1])?;
            let tensor = &self.tensors[n - 1];
            tensor.slice_site(flat_idx)
        } else {
            // Recursive case: tensor[start][:, idx, :] * right[1..]
            let right = self.evaluate_right(&indices[1..])?;
            let flat_idx = self.multi_to_flat(start, &indices[0..1])?;
            let tensor = &self.tensors[start];
            let slice = tensor.slice_site(flat_idx);
            matrix_times_col_vector(&slice, tensor.left_dim(), tensor.right_dim(), &right).map_err(
                |err| SimpleTensorTrainError::InvalidOperation {
                    message: format!("Failed to evaluate right environment at site {start}: {err}"),
                },
            )?
        };

        // Cache and return
        self.cache_right[start].insert(key, result.clone());
        Ok(result)
    }

    /// Evaluate the tensor train at a given index set using cache
    /// # Errors
    ///
    /// Returns [`SimpleTensorTrainError::IndexLengthMismatch`] when `indices`
    /// does not have one entry per site, [`SimpleTensorTrainError::Empty`] for
    /// an empty train, [`SimpleTensorTrainError::IndexOutOfBounds`] when an
    /// index exceeds a site dimension, or
    /// [`SimpleTensorTrainError::InvalidOperation`] when an environment
    /// contraction fails.
    ///
    pub fn evaluate(&mut self, indices: &[LocalIndex]) -> Result<T> {
        let n = self.len();
        if indices.len() != n {
            return Err(SimpleTensorTrainError::IndexLengthMismatch {
                expected: n,
                got: indices.len(),
            });
        }

        if n == 0 {
            return Err(SimpleTensorTrainError::Empty);
        }

        // Split at midpoint for efficiency
        let mid = n / 2;
        let left = self.evaluate_left(&indices[0..mid])?;
        let right = self.evaluate_right(&indices[mid..])?;

        // Contract left and right
        if left.len() != right.len() {
            return Err(SimpleTensorTrainError::InvalidOperation {
                message: format!(
                    "Left/right shape mismatch: {} vs {}",
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
    ///
    ///   (checks 1/4, 1/2, 3/4 positions and picks the best).
    ///   If you know the optimal split position (e.g., from TCI), pass `Some(split)`
    ///   to avoid the search overhead.
    /// # Errors
    ///
    /// Returns [`SimpleTensorTrainError::IndexLengthMismatch`] when any index
    /// set does not have one entry per site, [`SimpleTensorTrainError::Empty`]
    /// for an empty train, [`SimpleTensorTrainError::IndexOutOfBounds`] when an
    /// index exceeds a site dimension, or
    /// [`SimpleTensorTrainError::InvalidOperation`] when `split` is invalid or
    /// an environment contraction fails.
    ///
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
            return Err(SimpleTensorTrainError::Empty);
        }
        for idx in indices {
            if idx.len() != n {
                return Err(SimpleTensorTrainError::IndexLengthMismatch {
                    expected: n,
                    got: idx.len(),
                });
            }
            self.validate_partial_indices(0, idx)?;
        }

        // Determine split position
        let split = match split {
            Some(s) => s,
            None => self.find_split_heuristic(indices)?,
        };

        if split == 0 || split > n {
            return Err(SimpleTensorTrainError::InvalidOperation {
                message: format!("Invalid split position: {} (n_sites={})", split, n),
            });
        }

        // Get local dimensions for flat index computation
        let local_dims: Vec<usize> = (0..n)
            .map(|site| self.flat_site_dim(site))
            .collect::<Result<Vec<_>>>()?;

        // Build index mapper with appropriate key type for each half
        let mut mapper =
            IndexMapper::new(&local_dims[..split], &local_dims[split..], indices.len())?;

        for (i, idx) in indices.iter().enumerate() {
            mapper.add_index(i, &idx[..split], &idx[split..])?;
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
        let left_envs: Vec<Vec<T>> = unique_left
            .iter()
            .map(|l| self.evaluate_left(l))
            .collect::<Result<Vec<_>>>()?;

        // Compute right environments for all unique right parts
        let right_envs: Vec<Vec<T>> = unique_right
            .iter()
            .map(|r| self.evaluate_right(r))
            .collect::<Result<Vec<_>>>()?;

        // The environments must pair up element for element: `zip` would silently truncate a malformed train's
        // final bond to its `r == 0` component, which is the projection the reductions must not return either
        // (issue #913).
        for (&il, &ir) in mapper.idx_to_left.iter().zip(&mapper.idx_to_right) {
            let (left, right) = (&left_envs[il], &right_envs[ir]);
            if left.len() != right.len() {
                return Err(SimpleTensorTrainError::InvalidOperation {
                    message: format!(
                        "Final environment has {} left and {} right elements, expected equal lengths",
                        left.len(),
                        right.len()
                    ),
                });
            }
        }

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
    fn find_split_heuristic(&self, indices: &[MultiIndex]) -> Result<usize> {
        let n = self.len();
        if n <= 1 {
            return Ok(n.max(1));
        }

        let local_dims: Vec<usize> = self
            .site_dims
            .iter()
            .map(|dims| {
                dims.iter()
                    .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| SimpleTensorTrainError::InvalidOperation {
                message: "cache heuristic site-dimension product overflowed usize".to_string(),
            })?;

        // Helper to compute cost at a split position
        let compute_cost = |split: usize| -> Result<usize> {
            if split == 0 || split >= n {
                return Ok(usize::MAX);
            }

            let mut left_counter = UniqueCounter::new(&local_dims[..split], indices.len())?;
            let mut right_counter = UniqueCounter::new(&local_dims[split..], indices.len())?;

            for idx in indices {
                left_counter.insert(&idx[..split])?;
                right_counter.insert(&idx[split..])?;
            }

            Ok(left_counter.len().saturating_add(right_counter.len()))
        };

        // 3-point sampling: 1/4, 1/2, 3/4 positions
        let candidates = [n / 4, n / 2, n.saturating_mul(3) / 4];
        let costs: Result<Vec<(usize, usize)>> = candidates
            .iter()
            .filter(|&&p| p >= 1 && p < n)
            .map(|&p| Ok((p, compute_cost(p)?)))
            .collect();

        costs?
            .into_iter()
            .min_by_key(|&(_, c)| c)
            .map(|(p, _)| p)
            .ok_or_else(|| SimpleTensorTrainError::InvalidOperation {
                message: "cache heuristic could not choose a valid split".to_string(),
            })
    }
}

#[cfg(test)]
mod tests;
