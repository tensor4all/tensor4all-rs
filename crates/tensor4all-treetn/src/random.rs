//! Random tensor network generation.
//!
//! Provides utilities for creating random tensor networks, useful for testing.
//!
//! Note: Currently only supports `DynId` indices (the default dynamic index type).

use crate::site_index_network::SiteIndexNetwork;
use crate::treetn::TreeTN;
use rand::Rng;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use tensor4all_core::index::{DynId, Index, NoSymmSpace};
use tensor4all_core::TensorDynLen;

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
pub type DefaultIndex = Index<DynId, NoSymmSpace>;

/// Create a random f64 TreeTN from a site index network.
///
/// Generates random tensors at each node with:
/// - Site indices from the `site_network`
/// - Link indices created according to `link_space`
///
/// # Arguments
/// * `rng` - Random number generator for tensor data
/// * `site_network` - Network topology and site (physical) indices
/// * `link_space` - Specification for bond dimensions
///
/// # Type Parameters
/// * `R` - RNG type
/// * `V` - Node name type
///
/// # Example
/// ```
/// use tensor4all_treetn::{SiteIndexNetwork, random_treetn_f64, LinkSpace};
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use std::collections::HashSet;
///
/// // Create a simple 2-node network
/// let mut site_network = SiteIndexNetwork::<String, DynId>::new();
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// site_network.add_node("A".to_string(), HashSet::from([i.clone()])).unwrap();
/// site_network.add_node("B".to_string(), HashSet::from([j.clone()])).unwrap();
/// site_network.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
///
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
/// let treetn = random_treetn_f64(&mut rng, &site_network, LinkSpace::uniform(4));
///
/// assert_eq!(treetn.node_count(), 2);
/// ```
pub fn random_treetn_f64<R, V>(
    rng: &mut R,
    site_network: &SiteIndexNetwork<V, DynId, NoSymmSpace>,
    link_space: LinkSpace<V>,
) -> TreeTN<DynId, NoSymmSpace, V>
where
    R: Rng,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    random_treetn_impl(rng, site_network, link_space, false)
}

/// Create a random Complex64 TreeTN from a site index network.
///
/// Similar to [`random_treetn_f64`], but generates complex-valued tensors
/// where both real and imaginary parts are drawn from standard normal distribution.
///
/// # Arguments
/// * `rng` - Random number generator for tensor data
/// * `site_network` - Network topology and site (physical) indices
/// * `link_space` - Specification for bond dimensions
pub fn random_treetn_c64<R, V>(
    rng: &mut R,
    site_network: &SiteIndexNetwork<V, DynId, NoSymmSpace>,
    link_space: LinkSpace<V>,
) -> TreeTN<DynId, NoSymmSpace, V>
where
    R: Rng,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    random_treetn_impl(rng, site_network, link_space, true)
}

/// Internal implementation for creating random TreeTN.
fn random_treetn_impl<R, V>(
    rng: &mut R,
    site_network: &SiteIndexNetwork<V, DynId, NoSymmSpace>,
    link_space: LinkSpace<V>,
    is_complex: bool,
) -> TreeTN<DynId, NoSymmSpace, V>
where
    R: Rng,
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

        if !link_indices.contains_key(&key) {
            let dim = link_space.get(&key.0, &key.1)
                .expect("LinkSpace must provide dimension for all edges");
            let link_idx = Index::new_dyn(dim);
            link_indices.insert(key, link_idx);
        }
    }

    // Step 2: For each node, collect all indices and create random tensor
    let mut tensors = Vec::new();
    let mut node_names = Vec::new();

    for node_name in site_network.node_names() {
        let node_name = node_name.clone();

        // Collect site indices
        let site_inds = site_network.site_space(&node_name)
            .cloned()
            .unwrap_or_default();

        // Collect link indices from edges connected to this node
        let mut all_indices: Vec<DefaultIndex> = site_inds.into_iter().collect();

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
        let tensor = if is_complex {
            TensorDynLen::random_c64(rng, all_indices)
        } else {
            TensorDynLen::random_f64(rng, all_indices)
        };

        tensors.push(tensor);
        node_names.push(node_name);
    }

    // Step 3: Create TreeTN from tensors
    TreeTN::from_tensors(tensors, node_names)
        .expect("Failed to create TreeTN from random tensors")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashSet;

    #[test]
    fn test_random_treetn_f64_two_nodes() {
        // Use String as node name to avoid lifetime issues with &str
        let mut site_network = SiteIndexNetwork::<String, DynId>::new();
        let i = Index::new_dyn(2);
        let j = Index::new_dyn(3);
        site_network.add_node("A".to_string(), HashSet::from([i.clone()])).unwrap();
        site_network.add_node("B".to_string(), HashSet::from([j.clone()])).unwrap();
        site_network.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let treetn = random_treetn_f64(&mut rng, &site_network, LinkSpace::uniform(4));

        assert_eq!(treetn.node_count(), 2);
        assert_eq!(treetn.edge_count(), 1);
    }

    #[test]
    fn test_random_treetn_c64_chain() {
        let mut site_network = SiteIndexNetwork::<i32, DynId>::new();
        // Chain: 0 -- 1 -- 2
        for i in 0..3 {
            let idx = Index::new_dyn(2);
            site_network.add_node(i, HashSet::from([idx])).unwrap();
        }
        site_network.add_edge(&0, &1).unwrap();
        site_network.add_edge(&1, &2).unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let treetn = random_treetn_c64(&mut rng, &site_network, LinkSpace::uniform(3));

        assert_eq!(treetn.node_count(), 3);
        assert_eq!(treetn.edge_count(), 2);
    }

    #[test]
    fn test_link_space_per_edge() {
        let mut site_network = SiteIndexNetwork::<String, DynId>::new();
        site_network.add_node("A".to_string(), HashSet::new()).unwrap();
        site_network.add_node("B".to_string(), HashSet::new()).unwrap();
        site_network.add_node("C".to_string(), HashSet::new()).unwrap();
        site_network.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        site_network.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

        let mut dims = HashMap::new();
        dims.insert(("A".to_string(), "B".to_string()), 5);
        dims.insert(("B".to_string(), "C".to_string()), 10);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let treetn = random_treetn_f64(&mut rng, &site_network, LinkSpace::per_edge(dims));

        assert_eq!(treetn.node_count(), 3);
        assert_eq!(treetn.edge_count(), 2);
    }
}
