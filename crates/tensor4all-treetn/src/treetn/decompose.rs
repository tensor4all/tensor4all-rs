//! TreeTN decomposition from dense tensor.
//!
//! This module provides functions to decompose a dense tensor into a TreeTN
//! using factorization algorithms.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::{Canonical, FactorizeOptions, IndexLike, TensorLike};

use super::TreeTN;

// ============================================================================
// TreeTopology specification
// ============================================================================

/// Specification for tree topology: defines nodes and edges.
#[derive(Debug, Clone)]
pub struct TreeTopology<V> {
    /// Nodes in the tree (node name -> set of physical index positions in the input tensor)
    pub nodes: HashMap<V, Vec<usize>>,
    /// Edges in the tree: (node_a, node_b)
    pub edges: Vec<(V, V)>,
}

impl<V: Clone + Hash + Eq> TreeTopology<V> {
    /// Create a new tree topology with the given nodes and edges.
    ///
    /// # Arguments
    /// * `nodes` - Map from node name to the positions of physical indices in the input tensor
    /// * `edges` - List of edges as (node_a, node_b) pairs
    pub fn new(nodes: HashMap<V, Vec<usize>>, edges: Vec<(V, V)>) -> Self {
        Self { nodes, edges }
    }

    /// Validate that this topology describes a tree.
    pub fn validate(&self) -> Result<()> {
        let n = self.nodes.len();
        if n == 0 {
            return Err(anyhow::anyhow!("Tree topology must have at least one node"));
        }
        if n > 1 && self.edges.len() != n - 1 {
            return Err(anyhow::anyhow!(
                "Tree must have exactly n-1 edges: got {} nodes and {} edges",
                n,
                self.edges.len()
            ));
        }
        // Check all edge endpoints are valid nodes
        for (a, b) in &self.edges {
            if !self.nodes.contains_key(a) {
                return Err(anyhow::anyhow!("Edge refers to unknown node"));
            }
            if !self.nodes.contains_key(b) {
                return Err(anyhow::anyhow!("Edge refers to unknown node"));
            }
        }
        Ok(())
    }
}

// ============================================================================
// Decomposition functions
// ============================================================================

/// Decompose a dense tensor into a TreeTN using QR-based factorization.
///
/// This function takes a dense tensor and a tree topology specification, then
/// recursively decomposes the tensor using QR factorization to create a TreeTN.
///
/// # Algorithm
///
/// 1. Start from a leaf node, factorize to separate that node's physical indices
/// 2. Contract the right factor with remaining tensor, repeat for next edge
/// 3. Continue until all edges are processed
///
/// # Arguments
/// * `tensor` - The dense tensor to decompose
/// * `topology` - Tree topology specifying nodes, edges, and physical index assignments
///
/// # Returns
/// A TreeTN representing the decomposed tensor.
///
/// # Errors
/// Returns an error if:
/// - The topology is invalid
/// - Physical index positions don't match the tensor
/// - Factorization fails
pub fn factorize_tensor_to_treetn<T, V>(
    tensor: &T,
    topology: &TreeTopology<V>,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    factorize_tensor_to_treetn_with(tensor, topology, FactorizeOptions::qr())
}

/// Factorize a dense tensor into a TreeTN using specified factorization options.
///
/// This function takes a dense tensor and a tree topology specification, then
/// recursively decomposes the tensor using the specified algorithm to create a TreeTN.
///
/// # Algorithm
///
/// 1. Start from a leaf node, factorize to separate that node's physical indices
/// 2. Contract the right factor with remaining tensor, repeat for next edge
/// 3. Continue until all edges are processed
///
/// # Arguments
/// * `tensor` - The dense tensor to decompose
/// * `topology` - Tree topology specifying nodes, edges, and physical index assignments
/// * `options` - Factorization options (algorithm, max_rank, rtol, etc.)
///
/// # Returns
/// A TreeTN representing the decomposed tensor.
///
/// # Errors
/// Returns an error if:
/// - The topology is invalid
/// - Physical index positions don't match the tensor
/// - Factorization fails
pub fn factorize_tensor_to_treetn_with<T, V>(
    tensor: &T,
    topology: &TreeTopology<V>,
    options: FactorizeOptions,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    topology.validate()?;

    let tensor_indices = tensor.external_indices();

    if topology.nodes.len() == 1 {
        // Single node - just wrap the tensor
        let node_name = topology.nodes.keys().next().unwrap().clone();
        let mut tn = TreeTN::<T, V>::new();
        tn.add_tensor(node_name, tensor.clone())?;
        return Ok(tn);
    }

    // Validate that all index positions are valid
    for positions in topology.nodes.values() {
        for &pos in positions {
            if pos >= tensor_indices.len() {
                return Err(anyhow::anyhow!(
                    "Index position {} out of bounds (tensor has {} indices)",
                    pos,
                    tensor_indices.len()
                ));
            }
        }
    }

    // Build adjacency list for the tree
    let mut adj: HashMap<V, Vec<V>> = HashMap::new();
    for node in topology.nodes.keys() {
        adj.insert(node.clone(), Vec::new());
    }
    for (a, b) in &topology.edges {
        adj.get_mut(a).unwrap().push(b.clone());
        adj.get_mut(b).unwrap().push(a.clone());
    }
    // Sort each adjacency list to ensure deterministic traversal order
    for neighbors in adj.values_mut() {
        neighbors.sort();
    }

    // Find leaves (nodes with degree 1) - not currently used but kept for reference
    let _leaves: Vec<V> = adj
        .iter()
        .filter(|(_, neighbors)| neighbors.len() == 1)
        .map(|(node, _)| node.clone())
        .collect();

    // Choose root as the node with highest degree
    // Use min() to ensure deterministic selection when multiple nodes have the same degree
    let root = adj
        .iter()
        .max_by(|(node_a, neighbors_a), (node_b, neighbors_b)| {
            // First compare by degree, then by node name (ascending) for tie-breaking
            neighbors_a
                .len()
                .cmp(&neighbors_b.len())
                .then_with(|| node_b.cmp(node_a)) // Prefer smaller node name
        })
        .map(|(node, _)| node.clone())
        .ok_or_else(|| anyhow::anyhow!("Cannot find root node"))?;

    // Build traversal order using BFS from root
    let mut traversal_order: Vec<(V, Option<V>)> = Vec::new(); // (node, parent)
    let mut visited: HashSet<V> = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((root.clone(), None::<V>));

    while let Some((node, parent)) = queue.pop_front() {
        if visited.contains(&node) {
            continue;
        }
        visited.insert(node.clone());
        traversal_order.push((node.clone(), parent));

        for neighbor in adj.get(&node).unwrap() {
            if !visited.contains(neighbor) {
                queue.push_back((neighbor.clone(), Some(node.clone())));
            }
        }
    }

    // Reverse traversal order to process leaves first (post-order)
    traversal_order.reverse();

    // Store intermediate tensors as we decompose
    let mut current_tensor = tensor.clone();

    // Store the resulting node tensors
    let mut node_tensors: HashMap<V, T> = HashMap::new();

    // Store bond indices between nodes: (node_a, node_b) -> (index_on_a, index_on_b)
    let mut _bond_indices: HashMap<(V, V), (T::Index, T::Index)> = HashMap::new();

    // Use provided factorization options with Left canonical direction
    let factorize_options = FactorizeOptions {
        canonical: Canonical::Left,
        ..options
    };

    // Process nodes in post-order (leaves first)
    #[allow(clippy::needless_range_loop)]
    for i in 0..traversal_order.len() - 1 {
        let (node, parent) = &traversal_order[i];
        let parent_node = parent.as_ref().unwrap();

        // Get the physical index positions for this node
        let node_positions = topology.nodes.get(node).unwrap();

        // Find physical indices for this node in current_tensor
        let current_indices = current_tensor.external_indices();
        let left_inds: Vec<_> = node_positions
            .iter()
            .filter_map(|&pos| current_indices.get(pos).cloned())
            .collect();

        if left_inds.is_empty() && current_indices.len() > 1 {
            // No physical indices to separate - use first index
            // This happens when indices have already been separated
            continue;
        }

        // Perform factorization using TensorLike::factorize
        // left will have the node's physical indices + bond index
        // right will have bond index + remaining indices
        let factorize_result = current_tensor
            .factorize(&left_inds, &factorize_options)
            .map_err(|e| anyhow::anyhow!("Factorization failed: {:?}", e))?;

        let left = factorize_result.left;
        let right = factorize_result.right;
        let bond_index = factorize_result.bond_index;

        // Store left as the node's tensor (with physical indices + bond to parent)
        node_tensors.insert(node.clone(), left);

        // Store the bond index connecting this node to its parent
        // The bond_index is shared between left and right
        let bond_idx_node = bond_index.clone();
        let bond_idx_parent = bond_index;

        // Store in canonical order
        let key = if *node < *parent_node {
            (node.clone(), parent_node.clone())
        } else {
            (parent_node.clone(), node.clone())
        };
        if *node < *parent_node {
            _bond_indices.insert(key, (bond_idx_node, bond_idx_parent));
        } else {
            _bond_indices.insert(key, (bond_idx_parent, bond_idx_node));
        }

        // right becomes the current tensor for the next iteration
        current_tensor = right;
    }

    // The last node (root) gets the remaining tensor
    let (root_node, _) = &traversal_order.last().unwrap();
    node_tensors.insert(root_node.clone(), current_tensor);

    // Build the TreeTN using from_tensors (auto-connection by matching index IDs)
    // Since factorize() returns shared bond_index, tensors already have matching index IDs
    // IMPORTANT: Sort node_names to ensure deterministic ordering (HashMap iteration is non-deterministic)
    let mut node_names: Vec<V> = topology.nodes.keys().cloned().collect();
    node_names.sort();
    let tensors: Vec<T> = node_names
        .iter()
        .map(|name| node_tensors.get(name).cloned().unwrap())
        .collect();

    TreeTN::from_tensors(tensors, node_names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_tree_topology_validate() {
        // Test empty topology
        let empty: TreeTopology<String> = TreeTopology::new(HashMap::new(), Vec::new());
        assert!(empty.validate().is_err());

        // Test single node (valid)
        let mut nodes = HashMap::new();
        nodes.insert("node0".to_string(), vec![0]);
        let single = TreeTopology::new(nodes, Vec::new());
        assert!(single.validate().is_ok());

        // Test valid tree (2 nodes, 1 edge)
        let mut nodes = HashMap::new();
        nodes.insert("node0".to_string(), vec![0]);
        nodes.insert("node1".to_string(), vec![1]);
        let edges = vec![("node0".to_string(), "node1".to_string())];
        let tree = TreeTopology::new(nodes, edges);
        assert!(tree.validate().is_ok());

        // Test invalid: wrong number of edges
        let mut nodes = HashMap::new();
        nodes.insert("node0".to_string(), vec![0]);
        nodes.insert("node1".to_string(), vec![1]);
        let edges = vec![
            ("node0".to_string(), "node1".to_string()),
            ("node0".to_string(), "node1".to_string()),
        ];
        let invalid = TreeTopology::new(nodes, edges);
        assert!(invalid.validate().is_err());

        // Test invalid: edge references unknown node
        let mut nodes = HashMap::new();
        nodes.insert("node0".to_string(), vec![0]);
        let edges = vec![("node0".to_string(), "node2".to_string())];
        let invalid = TreeTopology::new(nodes, edges);
        assert!(invalid.validate().is_err());
    }
}
