//! Cached function wrapper for expensive function evaluations
//!
//! Automatically selects the optimal internal key type based on the index space size.

use std::collections::HashMap;

/// Internal cache with automatically selected key type
enum InnerCache<V> {
    U64 {
        cache: HashMap<u64, V>,
        coeffs: Vec<u64>,
    },
    U128 {
        cache: HashMap<u128, V>,
        coeffs: Vec<u128>,
    },
}

impl<V: Clone> InnerCache<V> {
    /// Create a new cache, automatically selecting the key type
    fn new(local_dims: &[usize]) -> Self {
        let total_bits: u32 = local_dims
            .iter()
            .map(|&d| if d <= 1 { 0 } else { (d as u64).ilog2() + 1 })
            .sum();

        if total_bits <= 64 {
            let coeffs = Self::compute_coeffs_u64(local_dims);
            Self::U64 {
                cache: HashMap::new(),
                coeffs,
            }
        } else {
            let coeffs = Self::compute_coeffs_u128(local_dims);
            Self::U128 {
                cache: HashMap::new(),
                coeffs,
            }
        }
    }

    fn compute_coeffs_u64(local_dims: &[usize]) -> Vec<u64> {
        let mut coeffs = Vec::with_capacity(local_dims.len());
        let mut prod: u64 = 1;
        for &d in local_dims {
            coeffs.push(prod);
            prod = prod.saturating_mul(d as u64);
        }
        coeffs
    }

    fn compute_coeffs_u128(local_dims: &[usize]) -> Vec<u128> {
        let mut coeffs = Vec::with_capacity(local_dims.len());
        let mut prod: u128 = 1;
        for &d in local_dims {
            coeffs.push(prod);
            prod = prod.saturating_mul(d as u128);
        }
        coeffs
    }

    fn flat_index_u64(idx: &[usize], coeffs: &[u64]) -> u64 {
        idx.iter()
            .zip(coeffs)
            .map(|(&i, &c)| c * i as u64)
            .sum()
    }

    fn flat_index_u128(idx: &[usize], coeffs: &[u128]) -> u128 {
        idx.iter()
            .zip(coeffs)
            .map(|(&i, &c)| c * i as u128)
            .sum()
    }

    fn get(&self, idx: &[usize]) -> Option<&V> {
        match self {
            Self::U64 { cache, coeffs } => {
                let key = Self::flat_index_u64(idx, coeffs);
                cache.get(&key)
            }
            Self::U128 { cache, coeffs } => {
                let key = Self::flat_index_u128(idx, coeffs);
                cache.get(&key)
            }
        }
    }

    fn insert(&mut self, idx: &[usize], value: V) {
        match self {
            Self::U64 { cache, coeffs } => {
                let key = Self::flat_index_u64(idx, coeffs);
                cache.insert(key, value);
            }
            Self::U128 { cache, coeffs } => {
                let key = Self::flat_index_u128(idx, coeffs);
                cache.insert(key, value);
            }
        }
    }

    fn contains(&self, idx: &[usize]) -> bool {
        match self {
            Self::U64 { cache, coeffs } => {
                let key = Self::flat_index_u64(idx, coeffs);
                cache.contains_key(&key)
            }
            Self::U128 { cache, coeffs } => {
                let key = Self::flat_index_u128(idx, coeffs);
                cache.contains_key(&key)
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::U64 { cache, .. } => cache.len(),
            Self::U128 { cache, .. } => cache.len(),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::U64 { cache, .. } => cache.clear(),
            Self::U128 { cache, .. } => cache.clear(),
        }
    }

    fn key_type_name(&self) -> &'static str {
        match self {
            Self::U64 { .. } => "u64",
            Self::U128 { .. } => "u128",
        }
    }
}

/// A wrapper that caches function evaluations for multi-index inputs.
///
/// Automatically selects the optimal internal key type based on `local_dims`.
pub struct CachedFunction<V, F>
where
    V: Clone,
    F: Fn(&[usize]) -> V,
{
    func: F,
    cache: InnerCache<V>,
    local_dims: Vec<usize>,
    num_evals: usize,
    num_cache_hits: usize,
}

impl<V, F> CachedFunction<V, F>
where
    V: Clone,
    F: Fn(&[usize]) -> V,
{
    /// Create a new cached function wrapper.
    ///
    /// The key type is automatically selected based on `local_dims`:
    /// - `u64` if total index space fits in 64 bits
    /// - `u128` otherwise
    pub fn new(func: F, local_dims: &[usize]) -> Self {
        Self {
            func,
            cache: InnerCache::new(local_dims),
            local_dims: local_dims.to_vec(),
            num_evals: 0,
            num_cache_hits: 0,
        }
    }

    /// Evaluate the function at a given index, using cache if available.
    pub fn eval(&mut self, idx: &[usize]) -> V {
        if let Some(value) = self.cache.get(idx) {
            self.num_cache_hits += 1;
            return value.clone();
        }

        self.num_evals += 1;
        let value = (self.func)(idx);
        self.cache.insert(idx, value.clone());
        value
    }

    /// Evaluate the function at a given index, bypassing the cache.
    pub fn eval_no_cache(&self, idx: &[usize]) -> V {
        (self.func)(idx)
    }

    /// Get the local dimensions.
    pub fn local_dims(&self) -> &[usize] {
        &self.local_dims
    }

    /// Get the number of sites (length of index).
    pub fn num_sites(&self) -> usize {
        self.local_dims.len()
    }

    /// Get the number of actual function evaluations.
    pub fn num_evals(&self) -> usize {
        self.num_evals
    }

    /// Get the number of cache hits.
    pub fn num_cache_hits(&self) -> usize {
        self.num_cache_hits
    }

    /// Get the total number of calls (evals + cache hits).
    pub fn total_calls(&self) -> usize {
        self.num_evals + self.num_cache_hits
    }

    /// Get the cache hit ratio.
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.total_calls();
        if total == 0 {
            0.0
        } else {
            self.num_cache_hits as f64 / total as f64
        }
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get the number of cached entries.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Check if an index is cached.
    pub fn is_cached(&self, idx: &[usize]) -> bool {
        self.cache.contains(idx)
    }

    /// Get the internal key type name (for debugging).
    pub fn key_type(&self) -> &'static str {
        self.cache.key_type_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_function_basic() {
        let local_dims = vec![2, 3, 4];
        let mut cf = CachedFunction::new(|idx: &[usize]| idx.iter().sum::<usize>(), &local_dims);

        assert_eq!(cf.eval(&[0, 1, 2]), 3);
        assert_eq!(cf.num_evals(), 1);
        assert_eq!(cf.num_cache_hits(), 0);

        // Second call should use cache
        assert_eq!(cf.eval(&[0, 1, 2]), 3);
        assert_eq!(cf.num_evals(), 1);
        assert_eq!(cf.num_cache_hits(), 1);

        // Different index
        assert_eq!(cf.eval(&[1, 2, 3]), 6);
        assert_eq!(cf.num_evals(), 2);
        assert_eq!(cf.num_cache_hits(), 1);
    }

    #[test]
    fn test_auto_key_selection_small() {
        // Small space: should use u64
        let local_dims = vec![2; 30]; // 2^30 < 2^64
        let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims);
        assert_eq!(cf.key_type(), "u64");
    }

    #[test]
    fn test_auto_key_selection_large() {
        // Large space: should use u128
        let local_dims = vec![2; 100]; // 2^100 > 2^64
        let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims);
        assert_eq!(cf.key_type(), "u128");
    }

    #[test]
    fn test_cached_function_clear() {
        let local_dims = vec![10, 10];
        let mut cf = CachedFunction::new(|idx: &[usize]| idx[0] + idx[1], &local_dims);
        cf.eval(&[1, 2]);
        cf.eval(&[3, 4]);
        assert_eq!(cf.cache_size(), 2);

        cf.clear_cache();
        assert_eq!(cf.cache_size(), 0);
    }

    #[test]
    fn test_local_dims() {
        let local_dims = vec![2, 3, 4];
        let cf = CachedFunction::new(|_: &[usize]| 0, &local_dims);
        assert_eq!(cf.local_dims(), &[2, 3, 4]);
        assert_eq!(cf.num_sites(), 3);
    }
}
