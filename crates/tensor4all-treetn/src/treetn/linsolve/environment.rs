//! Generic environment cache for tensor network computations.
//!
//! Provides a shared infrastructure for caching environment tensors
//! used in various algorithms (linsolve, fit, etc.).

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::IndexLike;
use tensor4all_core::TensorLike;

use crate::SiteIndexNetwork;

/// Trait for network topology, used for cache invalidation traversal.
pub trait NetworkTopology<V> {
    /// Iterator over neighbors of a node.
    type Neighbors<'a>: Iterator<Item = V>
    where
        Self: 'a,
        V: 'a;

    /// Get neighbors of a node.
    fn neighbors(&self, node: &V) -> Self::Neighbors<'_>;
}

/// Simple environment cache for tensor network computations.
///
/// This struct handles:
/// - Storing computed environment tensors
/// - Cache invalidation when tensors are updated
///
/// The actual contraction logic is implemented in ProjectedOperator/ProjectedState.
#[derive(Debug, Clone)]
pub struct EnvironmentCache<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq,
{
    /// Cached environment tensors: (from, to) -> tensor
    envs: HashMap<(V, V), T>,
}

impl<T, V> EnvironmentCache<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Create a new empty environment cache.
    pub fn new() -> Self {
        Self {
            envs: HashMap::new(),
        }
    }

    /// Get a cached environment tensor if it exists.
    pub fn get(&self, from: &V, to: &V) -> Option<&T> {
        self.envs.get(&(from.clone(), to.clone()))
    }

    /// Insert an environment tensor.
    pub fn insert(&mut self, from: V, to: V, env: T) {
        self.envs.insert((from, to), env);
    }

    /// Check if environment exists for edge (from, to).
    pub fn contains(&self, from: &V, to: &V) -> bool {
        self.envs.contains_key(&(from.clone(), to.clone()))
    }

    /// Get the number of cached environments.
    pub fn len(&self) -> usize {
        self.envs.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    /// Clear all cached environments.
    pub fn clear(&mut self) {
        self.envs.clear();
    }

    /// Invalidate all caches affected by updates to tensors in region.
    ///
    /// For each `t ∈ region`:
    /// 1. Remove all `env[(t, *)]` (0th generation)
    /// 2. Recursively remove caches propagating towards leaves
    pub fn invalidate<'a, NT: NetworkTopology<V>>(
        &mut self,
        region: impl IntoIterator<Item = &'a V>,
        topology: &NT,
    ) where
        V: 'a,
    {
        for t in region {
            // Get all neighbors of t
            let neighbors: Vec<V> = topology.neighbors(t).collect();

            // Remove all env[(t, *)] and propagate recursively
            for neighbor in neighbors {
                self.invalidate_recursive(t, &neighbor, topology);
            }
        }
    }

    /// Recursively invalidate caches starting from env[(from, to)] towards leaves.
    fn invalidate_recursive<NT: NetworkTopology<V>>(&mut self, from: &V, to: &V, topology: &NT) {
        // Remove env[(from, to)] if it exists
        if self.envs.remove(&(from.clone(), to.clone())).is_some() {
            // Propagate to next generation: env[(to, x)] for all neighbors x of to, x ≠ from
            let neighbors: Vec<V> = topology.neighbors(to).filter(|n| n != from).collect();

            for neighbor in neighbors {
                self.invalidate_recursive(to, &neighbor, topology);
            }
        }
    }
}

impl<T, V> Default for EnvironmentCache<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// NetworkTopology implementations for SiteIndexNetwork
// ============================================================================

/// Implement NetworkTopology for SiteIndexNetwork.
///
/// This enables direct use of SiteIndexNetwork for cache invalidation
/// and environment computation without needing adapter types like StaticTopology.
impl<NodeName, I> NetworkTopology<NodeName>
    for SiteIndexNetwork<NodeName, I>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    I: IndexLike,
{
    type Neighbors<'a>
        = Box<dyn Iterator<Item = NodeName> + 'a>
    where
        Self: 'a,
        NodeName: 'a;

    fn neighbors(&self, node: &NodeName) -> Self::Neighbors<'_> {
        Box::new(SiteIndexNetwork::neighbors(self, node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_cache_creation() {
        // Compile-time test
    }
}
