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

/// Specification for tree topology: defines nodes and index ID assignments.
///
/// `I` is the index ID type (e.g., `DynId`). Each node maps to the IDs of its
/// physical indices in the input tensor. This ensures correct index lookup
/// regardless of tensor index ordering, which can change during factorization.
#[derive(Debug, Clone)]
pub struct TreeTopology<V, I> {
    /// Nodes in the tree (node name -> list of index IDs belonging to this node)
    pub nodes: HashMap<V, Vec<I>>,
    /// Edges in the tree: (node_a, node_b)
    pub edges: Vec<(V, V)>,
}

impl<V: Clone + Hash + Eq, I: Clone + Eq> TreeTopology<V, I> {
    /// Create a new tree topology with the given nodes and edges.
    ///
    /// # Arguments
    /// * `nodes` - Map from node name to the index IDs belonging to that node
    /// * `edges` - List of edges as (node_a, node_b) pairs
    pub fn new(nodes: HashMap<V, Vec<I>>, edges: Vec<(V, V)>) -> Self {
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
    topology: &TreeTopology<V, <T::Index as IndexLike>::Id>,
    root: &V,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    factorize_tensor_to_treetn_with(tensor, topology, FactorizeOptions::qr(), root)
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
    topology: &TreeTopology<V, <T::Index as IndexLike>::Id>,
    options: FactorizeOptions,
    root: &V,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    factorize_tensor_to_treetn_with_root_impl(tensor, topology, options, root)
}

fn factorize_tensor_to_treetn_with_root_impl<T, V>(
    tensor: &T,
    topology: &TreeTopology<V, <T::Index as IndexLike>::Id>,
    options: FactorizeOptions,
    root: &V,
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
        if &node_name != root {
            return Err(anyhow::anyhow!("Requested root node not found in topology"));
        }
        let mut tn = TreeTN::<T, V>::new();
        tn.add_tensor(node_name.clone(), tensor.clone())?;
        tn.set_canonical_region([node_name])?;
        return Ok(tn);
    }

    // Validate that all index IDs exist in the tensor
    let tensor_ids: HashSet<_> = tensor_indices.iter().map(|idx| idx.id().clone()).collect();
    for (node, ids) in &topology.nodes {
        for id in ids {
            if !tensor_ids.contains(id) {
                return Err(anyhow::anyhow!(
                    "Index ID {:?} for node {:?} not found in tensor (tensor has {} indices)",
                    id,
                    node,
                    tensor_indices.len()
                ));
            }
        }
    }

    // Validate that each physical index ID is assigned to at most one node.
    // Duplicate assignment is almost always a topology specification bug and will
    // lead to ambiguous/missing node tensors during decomposition.
    let mut assigned_ids: HashSet<<T::Index as IndexLike>::Id> = HashSet::new();
    for (node, ids) in &topology.nodes {
        for id in ids {
            if !assigned_ids.insert(id.clone()) {
                return Err(anyhow::anyhow!(
                    "Index ID {:?} is assigned to multiple nodes (at least {:?})",
                    id,
                    node
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

    // Root is required. This ensures the norm-carrying tensor (the final `S*Vh` in
    // a left-canonical decomposition) ends up on a caller-chosen node.
    if !adj.contains_key(root) {
        return Err(anyhow::anyhow!("Requested root node not found in topology"));
    }

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

    // Use provided factorization options with Left canonical direction
    let factorize_options = FactorizeOptions {
        canonical: Canonical::Left,
        ..options
    };

    // Process nodes in post-order (leaves first)
    #[allow(clippy::needless_range_loop)]
    for i in 0..traversal_order.len() - 1 {
        let (node, _parent) = &traversal_order[i];
        // Get the index IDs for this node
        let node_ids = topology.nodes.get(node).unwrap();

        // Find physical indices for this node in current_tensor by matching IDs
        let current_indices = current_tensor.external_indices();
        let left_inds: Vec<_> = node_ids
            .iter()
            .filter_map(|id| current_indices.iter().find(|idx| idx.id() == id).cloned())
            .collect();

        if left_inds.is_empty() && current_indices.len() > 1 {
            // This indicates an inconsistent topology specification for the current tensor.
            // Previously we "skipped" such nodes, but that can lead to missing tensors and
            // panics later when building the TreeTN.
            return Err(anyhow::anyhow!(
                "No physical indices found for node {:?} (requested ids={:?}) in current tensor indices={:?}",
                node,
                node_ids,
                current_indices
                    .iter()
                    .map(|idx| idx.id().clone())
                    .collect::<Vec<_>>()
            ));
        }

        // Perform factorization using TensorLike::factorize
        // left will have the node's physical indices + bond index
        // right will have bond index + remaining indices
        let factorize_result = current_tensor
            .factorize(&left_inds, &factorize_options)
            .map_err(|e| anyhow::anyhow!("Factorization failed: {:?}", e))?;

        // Store left as the node's tensor (with physical indices + bond to parent)
        node_tensors.insert(node.clone(), factorize_result.left);

        // right becomes the current tensor for the next iteration
        current_tensor = factorize_result.right;
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

    let mut tn = TreeTN::from_tensors(tensors, node_names)?;
    tn.set_canonical_region([root.clone()])?;
    Ok(tn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tensor4all_core::{DynIndex, TensorDynLen};

    #[test]
    fn test_tree_topology_validate() {
        // Test empty topology (use usize as a dummy ID type)
        let empty: TreeTopology<String, usize> = TreeTopology::new(HashMap::new(), Vec::new());
        assert!(empty.validate().is_err());

        // Test single node (valid)
        let mut nodes = HashMap::new();
        nodes.insert("node0".to_string(), vec![0usize]);
        let single = TreeTopology::new(nodes, Vec::new());
        assert!(single.validate().is_ok());

        // Test valid tree (2 nodes, 1 edge)
        let mut nodes = HashMap::new();
        nodes.insert("node0".to_string(), vec![0usize]);
        nodes.insert("node1".to_string(), vec![1usize]);
        let edges = vec![("node0".to_string(), "node1".to_string())];
        let tree = TreeTopology::new(nodes, edges);
        assert!(tree.validate().is_ok());

        // Test invalid: wrong number of edges
        let mut nodes = HashMap::new();
        nodes.insert("node0".to_string(), vec![0usize]);
        nodes.insert("node1".to_string(), vec![1usize]);
        let edges = vec![
            ("node0".to_string(), "node1".to_string()),
            ("node0".to_string(), "node1".to_string()),
        ];
        let invalid = TreeTopology::new(nodes, edges);
        assert!(invalid.validate().is_err());

        // Test invalid: edge references unknown node
        let mut nodes = HashMap::new();
        nodes.insert("node0".to_string(), vec![0usize]);
        let edges = vec![("node0".to_string(), "node2".to_string())];
        let invalid = TreeTopology::new(nodes, edges);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_factorize_tensor_to_treetn_rejects_duplicate_index_ids() {
        let i0 = DynIndex::new_dyn(2);
        let i1 = DynIndex::new_dyn(2);
        let tensor =
            TensorDynLen::from_dense_f64(vec![i0.clone(), i1.clone()], vec![1.0, 0.0, 0.0, 1.0]);

        let mut nodes: HashMap<String, Vec<<DynIndex as IndexLike>::Id>> = HashMap::new();
        nodes.insert("node0".to_string(), vec![*i0.id()]);
        nodes.insert("node1".to_string(), vec![*i0.id()]); // duplicate on purpose

        let topo = TreeTopology::new(nodes, vec![("node0".to_string(), "node1".to_string())]);

        let result = factorize_tensor_to_treetn_with(
            &tensor,
            &topo,
            FactorizeOptions::qr(),
            &"node0".to_string(),
        );
        assert!(result.is_err());
    }
}
