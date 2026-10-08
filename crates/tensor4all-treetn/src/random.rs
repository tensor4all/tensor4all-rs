//! Random tensor network generation.
//!
//! Provides utilities for creating random tensor networks, useful for testing.
//!
//! Note: Currently only supports `DynId` indices (the default dynamic index type).

use crate::error::TreeTNOperationError;
use crate::site_index_network::SiteIndexNetwork;
use crate::treetn::TreeTN;
use anyhow::anyhow;
use rand::Rng;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::tensor::RandomScalar;
use tensor4all_core::{sort_indices_deterministic, IdxTensor};

/// Specification for link (bond) dimensions.
///
/// Used when creating random tensor networks to specify the dimension of each bond.
#[derive(Debug, Clone)]
pub enum LinkSpace<V> {
    /// All links have the same dimension.
    Uniform(usize),
    /// Each edge has its own dimension.
    /// The map uses ordered pairs `(min(a, b), max(a, b))` as keys for consistency.
    PerEdge(HashMap<(V, V), usize>),
}

impl<V> LinkSpace<V> {
    /// Create a uniform link space where all bonds have the same dimension.
    pub fn uniform(dim: usize) -> Self {
        Self::Uniform(dim)
    }

    /// Create a per-edge link space from a map of edge dimensions.
    pub fn per_edge(dims: HashMap<(V, V), usize>) -> Self {
        Self::PerEdge(dims)
    }
}

impl<V: Clone + Ord + Hash> LinkSpace<V> {
    /// Get the dimension for an edge between two nodes.
    ///
    /// For `PerEdge`, the key is normalized to `(min(a, b), max(a, b))`.
    pub fn get(&self, a: &V, b: &V) -> Option<usize> {
        match self {
            LinkSpace::Uniform(dim) => Some(*dim),
            LinkSpace::PerEdge(map) => {
                let key = if a < b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                };
                map.get(&key).copied()
            }
        }
    }
}

/// Type alias for the default index type used in random generation.
pub type DefaultIndex = Index<DynId, TagSet>;

/// Create a random TreeTN from a site index network (generic over scalar type).
///
/// Generates random tensors at each node with:
/// - Site indices from the `site_network`
/// - Link indices created according to `link_space`
///
/// Nodes are visited in the site network's node order, and each node tensor's
/// legs are its site indices sorted by
/// [`sort_indices_deterministic`](tensor4all_core::sort_indices_deterministic)
/// followed by its link indices in neighbor order. The same `rng` state and a
/// site network built from the same index objects therefore produce the same
/// network (link indices are fresh on every call, so compare by position).
///
/// Caveat: `sort_indices_deterministic` orders by dimension and prime level
/// first and then by the index ID, which is random for freshly created
/// indices. Two site legs of one node with equal dimension and prime level can
/// therefore be ordered differently when the site network is built from newly
/// created indices (for example in another process), and the same seed then
/// fills that node's tensor in a different leg order.
///
/// # Type Parameters
/// * `T` - Scalar type (e.g. `f64` or `Complex64`)
/// * `R` - RNG type
/// * `V` - Node name type
///
/// # Arguments
/// * `rng` - Caller-owned RNG for tensor data, including `dyn rand::RngCore`;
///   consumed directly without reseeding or creating an auxiliary generator
/// * `site_network` - Network topology and site (physical) indices
/// * `link_space` - Specification for bond dimensions
///
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// a backend failure).
///
/// # Returns
/// A network whose node tensors are filled in node order, with column-major
/// data drawn from the supplied stream.
///
/// # Example
/// ```
/// use tensor4all_treetn::{SiteIndexNetwork, random_treetn, LinkSpace};
/// use tensor4all_core::index::{Index, DynId, TagSet};
/// use tensor4all_core::tensor::RandomScalar;
/// use rand::{RngCore, SeedableRng};
/// use rand_chacha::ChaCha8Rng;
/// use std::collections::HashSet;
///
/// // Create a simple 2-node network
/// let mut site_network = SiteIndexNetwork::<String, Index<DynId, TagSet>>::new();
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// site_network.add_node("A".to_string(), HashSet::from([i.clone()])).unwrap();
/// site_network.add_node("B".to_string(), HashSet::from([j.clone()])).unwrap();
/// site_network.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
///
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
/// let mut reference = rng.clone();
/// let erased: &mut dyn RngCore = &mut rng;
/// let treetn = random_treetn::<f64, _, _>(erased, &site_network, LinkSpace::uniform(4)).unwrap();
///
/// for name in site_network.node_names() {
///     let count = if name == "A" { 8 } else { 12 };
///     let expected: Vec<f64> = (0..count).map(|_| f64::random_value(&mut reference)).collect();
///     let tensor = treetn.tensor(treetn.node_index(name).unwrap()).unwrap();
///     assert_eq!(tensor.to_vec::<f64>().unwrap(), expected);
/// }
/// assert_eq!(rng.get_word_pos(), reference.get_word_pos());
/// ```
pub fn random_treetn<T, R, V>(
    rng: &mut R,
    site_network: &SiteIndexNetwork<V, DefaultIndex>,
    link_space: LinkSpace<V>,
) -> std::result::Result<TreeTN<IdxTensor, V>, TreeTNOperationError>
where
    T: RandomScalar,
    R: Rng + ?Sized,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    // Step 1: Create link indices for each edge
    // Key: (smaller_name, larger_name), Value: link index
    let mut link_indices: HashMap<(V, V), DefaultIndex> = HashMap::new();

    // Get all edges from the site network topology
    for (a, b) in site_network.edges() {
        let key = if a < b {
            (a.clone(), b.clone())
        } else {
            (b.clone(), a.clone())
        };
        let key_clone = (key.0.clone(), key.1.clone());

        if let std::collections::hash_map::Entry::Vacant(entry) = link_indices.entry(key) {
            let dim = link_space
                .get(&key_clone.0, &key_clone.1)
                .ok_or_else(|| anyhow!("LinkSpace has no dimension for edge {:?}", key_clone))?;
            entry.insert(Index::new_dyn(dim));
        }
    }

    // Step 2: For each node, collect all indices and create random tensor
    let mut tensors = Vec::new();
    let mut node_names = Vec::new();

    for node_name in site_network.node_names() {
        let node_name = node_name.clone();

        // Collect site indices
        let site_inds = site_network
            .site_space(&node_name)
            .cloned()
            .unwrap_or_default();

        // Site legs first. A site space is an unordered set, so sort it by full
        // index identity: the leg order, and therefore which RNG draws land on
        // which tensor entry, must not depend on hash-set iteration order.
        let mut all_indices: Vec<DefaultIndex> = site_inds.into_iter().collect();
        sort_indices_deterministic(&mut all_indices);

        // Then link indices, in the network's neighbor order.
        for neighbor in site_network.neighbors(&node_name) {
            let key = if node_name < neighbor {
                (node_name.clone(), neighbor.clone())
            } else {
                (neighbor.clone(), node_name.clone())
            };

            if let Some(link_idx) = link_indices.get(&key) {
                all_indices.push(link_idx.clone());
            }
        }

        // Create random tensor
        let tensor = IdxTensor::random::<T, R>(rng, all_indices)?;

        tensors.push(tensor);
        node_names.push(node_name);
    }

    // Step 3: Create TreeTN from tensors
    TreeTN::from_tensors(tensors, node_names)
}

#[cfg(test)]
mod tests;
