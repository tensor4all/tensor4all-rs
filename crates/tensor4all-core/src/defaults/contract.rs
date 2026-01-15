//! Multi-tensor contraction with optimal contraction order.
//!
//! This module provides functions to contract multiple tensors efficiently
//! by using omeco's GreedyMethod to find the optimal contraction order.
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.
//!
//! # Main Functions
//!
//! - [`contract_multi`]: Contracts tensors, handling disconnected components via outer product
//! - [`contract_connected`]: Contracts tensors that must form a connected graph

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::Result;
use omeco::{optimize_code, EinCode, GreedyMethod, NestedEinsum};
use petgraph::algo::connected_components;
use petgraph::prelude::*;

use crate::defaults::{DynId, DynIndex, Index, TensorDynLen};
use crate::index_like::IndexLike;
use crate::tensor_like::AllowedPairs;

/// Contract multiple tensors into a single tensor, handling disconnected components.
///
/// This function automatically handles disconnected tensor graphs by:
/// 1. Finding connected components based on contractable indices
/// 2. Contracting each connected component separately
/// 3. Combining results using outer product
///
/// # Arguments
/// * `tensors` - Slice of tensors to contract
/// * `allowed` - Specifies which tensor pairs can have their indices contracted
///
/// # Returns
/// The result of contracting all tensors over allowed contractable indices.
/// If tensors form disconnected components, they are combined via outer product.
///
/// # Behavior by N
/// - N=0: Error
/// - N=1: Clone of input
/// - N>=2: Contract connected components, combine with outer product
///
/// # Errors
/// - `AllowedPairs::Specified` contains a pair with no contractable indices
///
/// # Example
/// ```ignore
/// use tensor4all_core::{contract_multi, AllowedPairs};
///
/// // Connected tensors: contracts via omeco
/// let result = contract_multi(&[a, b, c], AllowedPairs::All)?;
///
/// // Disconnected tensors: contracts each component, outer product to combine
/// let result = contract_multi(&[a, b], AllowedPairs::All)?;  // a, b have no common indices
/// ```
pub fn contract_multi(tensors: &[TensorDynLen], allowed: AllowedPairs<'_>) -> Result<TensorDynLen> {
    match tensors.len() {
        0 => Err(anyhow::anyhow!("No tensors to contract")),
        1 => Ok(tensors[0].clone()),
        _ => {
            // Validate AllowedPairs::Specified pairs have contractable indices
            if let AllowedPairs::Specified(pairs) = allowed {
                for &(i, j) in pairs {
                    if !has_contractable_indices(&tensors[i], &tensors[j]) {
                        return Err(anyhow::anyhow!(
                            "Specified pair ({}, {}) has no contractable indices",
                            i,
                            j
                        ));
                    }
                }
            }

            // Find connected components
            let components = find_tensor_connected_components(tensors, allowed);

            if components.len() == 1 {
                // All tensors connected - use optimized contraction
                contract_connected(tensors, allowed)
            } else {
                // Multiple components - contract each and combine with outer product
                let mut results: Vec<TensorDynLen> = Vec::new();
                for component in &components {
                    let component_tensors: Vec<TensorDynLen> =
                        component.iter().map(|&i| tensors[i].clone()).collect();

                    // Remap AllowedPairs for the component
                    let remapped_allowed = remap_allowed_pairs(allowed, component);
                    let contracted =
                        contract_connected(&component_tensors, remapped_allowed.as_ref())?;
                    results.push(contracted);
                }

                // Combine with outer product
                let mut result = results.pop().unwrap();
                for other in results.into_iter().rev() {
                    result = result.outer_product(&other)?;
                }
                Ok(result)
            }
        }
    }
}

/// Contract multiple tensors that form a connected graph.
///
/// Uses omeco's GreedyMethod to find the optimal contraction order.
///
/// # Arguments
/// * `tensors` - Slice of tensors to contract (must form a connected graph)
/// * `allowed` - Specifies which tensor pairs can have their indices contracted
///
/// # Returns
/// The result of contracting all tensors over allowed contractable indices.
///
/// # Connectivity Requirement
/// All tensors must form a connected graph through contractable indices.
/// Two tensors are connected if they share a contractable index (same ID, dual direction).
/// If the tensors form disconnected components, this function returns an error.
///
/// Use [`contract_multi`] if you want automatic handling of disconnected components.
///
/// # Behavior by N
/// - N=0: Error
/// - N=1: Clone of input
/// - N>=2: Optimal order via omeco's GreedyMethod
///
/// # Example
/// ```ignore
/// use tensor4all_core::{contract_connected, AllowedPairs};
///
/// let tensors = vec![tensor_a, tensor_b, tensor_c];  // Must be connected
/// let result = contract_connected(&tensors, AllowedPairs::All)?;
/// ```
pub fn contract_connected(
    tensors: &[TensorDynLen],
    allowed: AllowedPairs<'_>,
) -> Result<TensorDynLen> {
    match tensors.len() {
        0 => Err(anyhow::anyhow!("No tensors to contract")),
        1 => Ok(tensors[0].clone()),
        _ => contract_connected_optimized(tensors, allowed),
    }
}

// ============================================================================
// Helper functions for connected component detection
// ============================================================================

/// Check if two tensors have any contractable indices.
fn has_contractable_indices(a: &TensorDynLen, b: &TensorDynLen) -> bool {
    a.indices
        .iter()
        .any(|idx_a| b.indices.iter().any(|idx_b| idx_a.is_contractable(idx_b)))
}

/// Find connected components of tensors based on contractable indices.
///
/// Uses petgraph for O(V+E) connected component detection.
///
/// # Arguments
/// * `tensors` - Slice of tensors
/// * `allowed` - Which tensor pairs can have their indices contracted
///
/// # Returns
/// A vector of components, where each component is a vector of tensor indices.
/// Components are sorted by their smallest tensor index for determinism.
fn find_tensor_connected_components(
    tensors: &[TensorDynLen],
    allowed: AllowedPairs<'_>,
) -> Vec<Vec<usize>> {
    let n = tensors.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![vec![0]];
    }

    // Build undirected graph
    let mut graph = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

    // Add edges based on connectivity
    match allowed {
        AllowedPairs::All => {
            // Two tensors connected if they have any contractable indices
            for i in 0..n {
                for j in (i + 1)..n {
                    if has_contractable_indices(&tensors[i], &tensors[j]) {
                        graph.add_edge(nodes[i], nodes[j], ());
                    }
                }
            }
        }
        AllowedPairs::Specified(pairs) => {
            // Only specified pairs with contractable indices are connected
            for &(i, j) in pairs {
                if has_contractable_indices(&tensors[i], &tensors[j]) {
                    graph.add_edge(nodes[i], nodes[j], ());
                }
            }
        }
    }

    // Find connected components using petgraph
    let num_components = connected_components(&graph);

    if num_components == 1 {
        // All connected - return single component with all indices
        return vec![(0..n).collect()];
    }

    // Multiple components - group by component ID
    // petgraph's connected_components returns the number, we need to compute assignments
    use petgraph::visit::Dfs;
    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for start in 0..n {
        if !visited[start] {
            let mut component = Vec::new();
            let mut dfs = Dfs::new(&graph, nodes[start]);
            while let Some(node) = dfs.next(&graph) {
                let idx = node.index();
                if !visited[idx] {
                    visited[idx] = true;
                    component.push(idx);
                }
            }
            component.sort(); // Deterministic order within component
            components.push(component);
        }
    }

    // Sort components by smallest index for determinism
    components.sort_by_key(|c| c[0]);
    components
}

/// Remap AllowedPairs for a subset of tensors.
///
/// Given original tensor indices in `component`, returns AllowedPairs
/// with indices remapped to the component's local indices.
fn remap_allowed_pairs(allowed: AllowedPairs<'_>, component: &[usize]) -> RemappedAllowedPairs {
    match allowed {
        AllowedPairs::All => RemappedAllowedPairs::All,
        AllowedPairs::Specified(pairs) => {
            // Build map from original index to local index
            let orig_to_local: HashMap<usize, usize> = component
                .iter()
                .enumerate()
                .map(|(local, &orig)| (orig, local))
                .collect();

            // Filter and remap pairs
            let remapped: Vec<(usize, usize)> = pairs
                .iter()
                .filter_map(|&(i, j)| {
                    match (orig_to_local.get(&i), orig_to_local.get(&j)) {
                        (Some(&li), Some(&lj)) => Some((li, lj)),
                        _ => None, // Pair not in this component
                    }
                })
                .collect();

            RemappedAllowedPairs::Specified(remapped)
        }
    }
}

/// Owned version of AllowedPairs for remapped components.
enum RemappedAllowedPairs {
    All,
    Specified(Vec<(usize, usize)>),
}

impl RemappedAllowedPairs {
    fn as_ref(&self) -> AllowedPairs<'_> {
        match self {
            RemappedAllowedPairs::All => AllowedPairs::All,
            RemappedAllowedPairs::Specified(pairs) => AllowedPairs::Specified(pairs),
        }
    }
}

// ============================================================================
// Core contraction implementation
// ============================================================================

/// Contract two tensors over their common indices.
/// If there are no common indices, performs outer product.
fn contract_pair(a: &TensorDynLen, b: &TensorDynLen) -> Result<TensorDynLen> {
    let common: Vec<_> = a
        .indices
        .iter()
        .filter_map(|idx_a| {
            b.indices
                .iter()
                .find(|idx_b| idx_a.is_contractable(idx_b))
                .map(|idx_b| (idx_a.clone(), idx_b.clone()))
        })
        .collect();

    if common.is_empty() {
        // No common indices - perform outer product
        a.outer_product(b)
    } else {
        a.tensordot(b, &common)
    }
}

/// Contract multiple tensors using omeco's GreedyMethod for optimal ordering.
///
/// Uses internal IDs to control which indices are contracted based on `allowed`.
fn contract_connected_optimized(
    tensors: &[TensorDynLen],
    allowed: AllowedPairs<'_>,
) -> Result<TensorDynLen> {
    // 1. Validate for Specified pairs
    if let AllowedPairs::Specified(pairs) = allowed {
        // 1a. Check that all specified pairs have contractable indices
        for &(i, j) in pairs {
            if !has_contractable_indices(&tensors[i], &tensors[j]) {
                return Err(anyhow::anyhow!(
                    "Specified pair ({}, {}) has no contractable indices",
                    i,
                    j
                ));
            }
        }
        // 1b. Check that the graph formed by pairs is connected
        validate_connected_graph(tensors.len(), pairs)?;
    }

    // 2. Build internal IDs
    //    - Allowed pairs' contractable indices get shared internal IDs
    //    - All other indices get unique internal IDs
    let (ixs, internal_id_to_original) = build_internal_ids(tensors, allowed);

    // 3. Output = count == 1 internal IDs (external indices)
    let mut idx_count: HashMap<usize, usize> = HashMap::new();
    for ix in &ixs {
        for &i in ix {
            *idx_count.entry(i).or_insert(0) += 1;
        }
    }
    let mut output: Vec<usize> = idx_count
        .iter()
        .filter(|(_, &count)| count == 1)
        .map(|(&idx, _)| idx)
        .collect();
    output.sort(); // deterministic order

    // 4. Build sizes from tensors
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        for (pos, &dim) in tensor.dims.iter().enumerate() {
            let internal_id = ixs[tensor_idx][pos];
            sizes.entry(internal_id).or_insert(dim);
        }
    }

    // 5. Optimize contraction order using GreedyMethod
    let code = EinCode::new(ixs.clone(), output.clone());
    let tree = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or_else(|| anyhow::anyhow!("Failed to optimize contraction order"))?;

    // 5a. Check if omeco returned an incomplete tree (doesn't include all tensors)
    // This happens when tensors form disconnected components in the contraction graph.
    if tree.leaf_count() != tensors.len() {
        return Err(anyhow::anyhow!(
            "Contraction graph is disconnected: only {} of {} tensors are connected. \
             All tensors must form a connected graph through contractable indices.",
            tree.leaf_count(),
            tensors.len()
        ));
    }

    // 6. Create temporary tensors with internal ID-based indices
    //    This ensures contract_pair uses internal IDs for contraction determination
    let temp_tensors: Vec<TensorDynLen> = tensors
        .iter()
        .enumerate()
        .map(|(tensor_idx, tensor)| {
            let temp_indices: Vec<DynIndex> = ixs[tensor_idx]
                .iter()
                .zip(tensor.dims.iter())
                .map(|(&internal_id, &dim)| {
                    // Create index with internal_id as the DynId
                    Index::new(DynId(internal_id as u128), dim)
                })
                .collect();
            TensorDynLen::new(temp_indices, tensor.dims.clone(), tensor.storage().clone())
        })
        .collect();

    // 7. Execute the contraction tree using temporary tensors
    let result = execute_contraction_tree(&temp_tensors, &tree)?;

    // 8. Restore original indices from internal IDs
    //    Use result's actual index order (not sorted output) to match storage layout
    let restored_indices: Vec<DynIndex> = result
        .indices
        .iter()
        .map(|temp_idx| {
            // temp_idx has DynId(internal_id as u128)
            let internal_id = temp_idx.id().0 as usize;
            let (tensor_idx, pos) = internal_id_to_original[&internal_id];
            tensors[tensor_idx].indices[pos].clone()
        })
        .collect();

    Ok(TensorDynLen::new(
        restored_indices,
        result.dims.clone(),
        result.storage().clone(),
    ))
}

/// Build internal IDs for contraction.
///
/// Internal IDs are integers that represent indices during contraction:
/// - Contractable pairs in allowed tensor pairs share the same internal ID
/// - All other indices get unique internal IDs
///
/// These internal IDs are used for:
/// 1. Determining which indices to contract (count >= 2)
/// 2. Optimizing contraction order (omeco uses these)
/// 3. Mapping back to original indices after contraction
///
/// Returns: (ixs, internal_id_to_original)
/// - ixs: Vec<Vec<usize>> - internal IDs for each tensor's indices
/// - internal_id_to_original: Maps internal_id -> (tensor_idx, index_position)
fn build_internal_ids(
    tensors: &[TensorDynLen],
    allowed: AllowedPairs<'_>,
) -> (Vec<Vec<usize>>, HashMap<usize, (usize, usize)>) {
    let mut next_id = 0usize;
    // (tensor_idx, index_position) -> internal_id
    let mut assigned: HashMap<(usize, usize), usize> = HashMap::new();
    // internal_id -> (tensor_idx, index_position) for restoring
    let mut internal_id_to_original: HashMap<usize, (usize, usize)> = HashMap::new();

    // Helper: assign shared internal ID for contractable indices between tensor pair
    // Handles hyperedges (same index appearing in 3+ tensors) by propagating IDs
    let mut assign_contractable = |ti: usize, tj: usize| {
        for (pi, idx_i) in tensors[ti].indices.iter().enumerate() {
            for (pj, idx_j) in tensors[tj].indices.iter().enumerate() {
                if idx_i.is_contractable(idx_j) {
                    let key_i = (ti, pi);
                    let key_j = (tj, pj);
                    match (assigned.get(&key_i).copied(), assigned.get(&key_j).copied()) {
                        (None, None) => {
                            // Neither assigned: create new shared ID
                            assigned.insert(key_i, next_id);
                            assigned.insert(key_j, next_id);
                            internal_id_to_original.insert(next_id, key_i);
                            next_id += 1;
                        }
                        (Some(id), None) => {
                            // key_i assigned, key_j not: share key_i's ID
                            assigned.insert(key_j, id);
                        }
                        (None, Some(id)) => {
                            // key_j assigned, key_i not: share key_j's ID
                            assigned.insert(key_i, id);
                        }
                        (Some(id_i), Some(id_j)) => {
                            // Both assigned - they should have the same ID
                            // (same original index ID means all occurrences should share internal ID)
                            debug_assert_eq!(
                                id_i, id_j,
                                "Conflicting internal IDs for contractable indices"
                            );
                        }
                    }
                }
            }
        }
    };

    match allowed {
        AllowedPairs::All => {
            // All tensor pairs are allowed to contract
            for ti in 0..tensors.len() {
                for tj in (ti + 1)..tensors.len() {
                    assign_contractable(ti, tj);
                }
            }
        }
        AllowedPairs::Specified(pairs) => {
            // Only specified pairs are allowed to contract
            for &(ti, tj) in pairs {
                assign_contractable(ti, tj);
            }
        }
    }

    // Assign unique IDs for all unassigned indices (external indices)
    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        for pos in 0..tensor.indices.len() {
            let key = (tensor_idx, pos);
            if let std::collections::hash_map::Entry::Vacant(e) = assigned.entry(key) {
                e.insert(next_id);
                internal_id_to_original.insert(next_id, key);
                next_id += 1;
            }
        }
    }

    // Build ixs: internal IDs for each tensor
    let ixs: Vec<Vec<usize>> = tensors
        .iter()
        .enumerate()
        .map(|(tensor_idx, tensor)| {
            (0..tensor.indices.len())
                .map(|pos| assigned[&(tensor_idx, pos)])
                .collect()
        })
        .collect();

    (ixs, internal_id_to_original)
}

/// Validate that the specified tensor pairs form a connected graph.
///
/// Returns an error if the graph is disconnected.
fn validate_connected_graph(num_tensors: usize, pairs: &[(usize, usize)]) -> Result<()> {
    if num_tensors == 0 {
        return Ok(());
    }
    if num_tensors == 1 {
        return Ok(());
    }
    if pairs.is_empty() {
        return Err(anyhow::anyhow!(
            "AllowedPairs::Specified with empty pairs results in disconnected graph"
        ));
    }

    // Build adjacency list
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); num_tensors];
    for &(a, b) in pairs {
        if a >= num_tensors || b >= num_tensors {
            return Err(anyhow::anyhow!(
                "Invalid tensor index in AllowedPairs: ({}, {}) but only {} tensors",
                a,
                b,
                num_tensors
            ));
        }
        adj[a].insert(b);
        adj[b].insert(a);
    }

    // BFS to check connectivity
    let mut visited = vec![false; num_tensors];
    let mut queue = VecDeque::new();
    queue.push_back(0);
    visited[0] = true;
    let mut count = 1;

    while let Some(node) = queue.pop_front() {
        for &neighbor in &adj[node] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                count += 1;
                queue.push_back(neighbor);
            }
        }
    }

    if count != num_tensors {
        return Err(anyhow::anyhow!(
            "AllowedPairs::Specified forms a disconnected graph: only {} of {} tensors are connected",
            count,
            num_tensors
        ));
    }

    Ok(())
}

/// Execute a contraction tree by recursively contracting tensors.
fn execute_contraction_tree(
    tensors: &[TensorDynLen],
    tree: &NestedEinsum<usize>,
) -> Result<TensorDynLen> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => Ok(tensors[*tensor_index].clone()),
        NestedEinsum::Node { args, .. } => {
            // Recursively evaluate children
            let children: Vec<TensorDynLen> = args
                .iter()
                .map(|arg| execute_contraction_tree(tensors, arg))
                .collect::<Result<_>>()?;

            // Contract all children (typically 2 for binary tree)
            let mut result = children[0].clone();
            for child in children.iter().skip(1) {
                result = contract_pair(&result, child)?;
            }
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::{DynId, DynIndex, Index};
    use crate::storage::{DenseStorageC64, Storage};
    use num_complex::Complex64;
    use std::sync::Arc;

    fn make_test_tensor(shape: &[usize], ids: &[u128]) -> TensorDynLen {
        let indices: Vec<DynIndex> = ids
            .iter()
            .zip(shape.iter())
            .map(|(&id, &dim)| Index::new(DynId(id), dim))
            .collect();
        let dims = shape.to_vec();
        let total_size: usize = shape.iter().product();
        let data: Vec<Complex64> = (0..total_size)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect();
        let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data)));
        TensorDynLen::new(indices, dims, storage)
    }

    #[test]
    fn test_contract_multi_empty() {
        let tensors: Vec<TensorDynLen> = vec![];
        let result = contract_multi(&tensors, AllowedPairs::All);
        assert!(result.is_err());
    }

    #[test]
    fn test_contract_multi_single() {
        let tensor = make_test_tensor(&[2, 3], &[1, 2]);
        let tensors = vec![tensor.clone()];
        let result = contract_multi(&tensors, AllowedPairs::All).unwrap();
        assert_eq!(result.dims, tensor.dims);
    }

    #[test]
    fn test_contract_multi_pair() {
        // A[i,j] * B[j,k] -> C[i,k]
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let tensors = vec![a, b];
        let result = contract_multi(&tensors, AllowedPairs::All).unwrap();
        assert_eq!(result.dims, vec![2, 4]); // i, k
    }

    #[test]
    fn test_contract_multi_three() {
        // A[i,j] * B[j,k] * C[k,l] -> D[i,l]
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let c = make_test_tensor(&[4, 5], &[3, 4]); // k=3, l=4
        let tensors = vec![a, b, c];
        let result = contract_multi(&tensors, AllowedPairs::All).unwrap();
        // Output has dimensions for i and l (order may vary due to tree structure)
        let mut sorted_dims = result.dims.clone();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![2, 5]); // i=2, l=5
    }

    #[test]
    fn test_contract_multi_four() {
        // A[i,j] * B[j,k] * C[k,l] * D[l,m] -> E[i,m]
        let a = make_test_tensor(&[2, 3], &[1, 2]);
        let b = make_test_tensor(&[3, 4], &[2, 3]);
        let c = make_test_tensor(&[4, 5], &[3, 4]);
        let d = make_test_tensor(&[5, 6], &[4, 5]);
        let tensors = vec![a, b, c, d];
        let result = contract_multi(&tensors, AllowedPairs::All).unwrap();
        // Output has dimensions for i and m (order may vary due to tree structure)
        let mut sorted_dims = result.dims.clone();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![2, 6]); // i=2, m=6
    }

    #[test]
    fn test_contract_multi_outer_product() {
        // A[i,j] * B[k,l] (no common indices) -> outer product C[i,j,k,l]
        let a = make_test_tensor(&[2, 3], &[1, 2]);
        let b = make_test_tensor(&[4, 5], &[3, 4]);
        let tensors = vec![a, b];
        let result = contract_multi(&tensors, AllowedPairs::All).unwrap();
        // Total elements should be 2*3*4*5 = 120
        let total_elements: usize = result.dims.iter().product();
        assert_eq!(total_elements, 2 * 3 * 4 * 5);
        assert_eq!(result.dims.len(), 4);
    }

    #[test]
    fn test_contract_multi_vector_outer_product() {
        // A[i] * B[j] (no common indices) -> outer product C[i,j]
        let a = make_test_tensor(&[2], &[1]); // i=1
        let b = make_test_tensor(&[3], &[2]); // j=2
        let tensors = vec![a, b];
        let result = contract_multi(&tensors, AllowedPairs::All).unwrap();
        // Total elements should be 2*3 = 6
        let total_elements: usize = result.dims.iter().product();
        assert_eq!(total_elements, 2 * 3);
        assert_eq!(result.dims.len(), 2);
    }

    #[test]
    fn test_contract_connected_disconnected_error() {
        // contract_connected should error on disconnected graphs
        let a = make_test_tensor(&[2, 3], &[1, 2]);
        let b = make_test_tensor(&[4, 5], &[3, 4]);
        let tensors = vec![a, b];
        let result = contract_connected(&tensors, AllowedPairs::All);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disconnected"));
    }

    #[test]
    fn test_contract_connected_specified_no_contractable_error() {
        // contract_connected with Specified pair that has no contractable indices
        // Should give clear error message (not just "disconnected")
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[4, 5], &[3, 4]); // k=3, l=4 (no common with a)
        let tensors = vec![a, b];
        let result = contract_connected(&tensors, AllowedPairs::Specified(&[(0, 1)]));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("no contractable indices"));
    }

    // ========================================================================
    // Tests for AllowedPairs::Specified
    // ========================================================================

    #[test]
    fn test_contract_specified_pairs() {
        // A[i,j], B[j,k], C[i,l] - tensors 0, 1, 2
        // Specified(&[(0, 1), (0, 2)]) - A-B and A-C contractions
        // j is contracted (A-B), i is contracted (A-C)
        // Result should be [k, l]
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let c = make_test_tensor(&[2, 5], &[1, 4]); // i=1, l=4
        let tensors = vec![a, b, c];
        let result = contract_multi(&tensors, AllowedPairs::Specified(&[(0, 1), (0, 2)])).unwrap();
        let mut sorted_dims = result.dims.clone();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![4, 5]); // k=4, l=5
    }

    #[test]
    fn test_contract_specified_no_contractable_indices_error() {
        // A[i,j], B[j,k], C[m,l] - tensors 0, 1, 2
        // Specified(&[(0, 1), (1, 2)]) - A-B and B-C pairs allowed
        // j is contracted (A-B), but B-C has no common indices
        // Should error because pair (1, 2) has no contractable indices
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let c = make_test_tensor(&[6, 5], &[5, 4]); // m=5, l=4 (no common with B)
        let tensors = vec![a, b, c];
        let result = contract_multi(&tensors, AllowedPairs::Specified(&[(0, 1), (1, 2)]));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no contractable indices"));
    }

    #[test]
    fn test_contract_specified_disconnected_outer_product() {
        // Specified(&[(0,1), (2,3)]) with 4 tensors
        // {A,B} and {C,D} each have contractable indices, but are disconnected
        // Should succeed via outer product of the two contracted components
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let c = make_test_tensor(&[4, 5], &[4, 5]); // m=4, n=5
        let d = make_test_tensor(&[5, 6], &[5, 6]); // n=5, p=6
        let tensors = vec![a, b, c, d];
        let result = contract_multi(&tensors, AllowedPairs::Specified(&[(0, 1), (2, 3)])).unwrap();
        // A-B contracts j: result has i, k (dims 2, 4)
        // C-D contracts n: result has m, p (dims 4, 6)
        // Outer product: result has 4 indices
        assert_eq!(result.dims.len(), 4);
        let mut sorted_dims = result.dims.clone();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![2, 4, 4, 6]);
    }

    #[test]
    fn test_validate_connected_graph() {
        // Connected
        assert!(validate_connected_graph(3, &[(0, 1), (1, 2)]).is_ok());
        assert!(validate_connected_graph(3, &[(0, 1), (0, 2)]).is_ok());
        assert!(validate_connected_graph(4, &[(0, 1), (1, 2), (2, 3)]).is_ok());

        // Disconnected
        assert!(validate_connected_graph(4, &[(0, 1), (2, 3)]).is_err());
        assert!(validate_connected_graph(3, &[(0, 1)]).is_err()); // tensor 2 not connected

        // Edge cases
        assert!(validate_connected_graph(0, &[]).is_ok());
        assert!(validate_connected_graph(1, &[]).is_ok());
        assert!(validate_connected_graph(2, &[]).is_err()); // 2 tensors, no edges
    }

    /// Test omeco's handling of hyperedges.
    ///
    /// Simulates: A(i, I) * B(j, J) * C(k, K) * delta_{IJK}
    /// where delta is a 3D superdiagonal (I==J==K).
    ///
    /// Decomposed representation:
    /// - A(i, x), B(j, x), C(k, x), delta(x)
    /// - x is a hyperedge connecting all four tensors
    #[test]
    fn test_omeco_hyperedge_delta() {
        use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod};
        use std::collections::HashMap;

        // A(i, x), B(j, x), C(k, x), delta(x)
        // x is the hyperedge (appears in all 4 tensors)
        let ixs: Vec<Vec<char>> = vec![
            vec!['i', 'x'], // A
            vec!['j', 'x'], // B
            vec!['k', 'x'], // C
            vec!['x'],      // delta (1D)
        ];
        let output = vec!['i', 'j', 'k'];

        let code = EinCode::new(ixs.clone(), output);

        let mut sizes: HashMap<char, usize> = HashMap::new();
        sizes.insert('i', 10);
        sizes.insert('j', 10);
        sizes.insert('k', 10);
        sizes.insert('x', 100); // hyperedge dimension

        let tree = optimize_code(&code, &sizes, &GreedyMethod::default())
            .expect("optimization should succeed");

        let complexity = contraction_complexity(&tree, &sizes, &ixs);

        // Verify the optimization found a solution
        // Time complexity should be reasonable (not exponentially bad)
        println!(
            "Hyperedge test - tc: 2^{:.2}, sc: 2^{:.2}",
            complexity.tc, complexity.sc
        );

        // Space complexity should be around log2(10*10*10) = ~10 for output
        // plus some intermediate tensors
        assert!(
            complexity.sc < 15.0,
            "Space complexity should be reasonable"
        );

        // The tree should have 4 leaves (one for each input tensor)
        assert_eq!(tree.leaf_count(), 4);
    }

    /// Test omeco with a simple hyperedge case: U * s * V (SVD-like)
    #[test]
    fn test_omeco_hyperedge_svd() {
        use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod};
        use std::collections::HashMap;

        // U(i, j), s(j), V(j, k)
        // j is a hyperedge connecting all 3 tensors
        let ixs: Vec<Vec<char>> = vec![
            vec!['i', 'j'], // U
            vec!['j'],      // s (1D diagonal)
            vec!['j', 'k'], // V
        ];
        let output = vec!['i', 'k'];

        let code = EinCode::new(ixs.clone(), output);

        let mut sizes: HashMap<char, usize> = HashMap::new();
        sizes.insert('i', 100);
        sizes.insert('j', 50); // bond dimension
        sizes.insert('k', 100);

        let tree = optimize_code(&code, &sizes, &GreedyMethod::default())
            .expect("optimization should succeed");

        let complexity = contraction_complexity(&tree, &sizes, &ixs);

        println!(
            "SVD hyperedge - tc: 2^{:.2}, sc: 2^{:.2}",
            complexity.tc, complexity.sc
        );
        println!("Tree structure: {:?}", tree);

        // 3 input tensors
        assert_eq!(tree.leaf_count(), 3);

        // Note: sc is high (2^18.93 ≈ 500000 = 100*50*100) because
        // intermediate tensor keeps all indices until final contraction.
        // This is a limitation of treating diagonal as just another tensor.
        //
        // With proper diagonal handling, we could:
        // 1. U(i,j) * s(j) → U'(i,j) where U'[i,j] = U[i,j] * s[j] (element-wise, O(n²))
        // 2. U'(i,j) * V(j,k) → R(i,k) (matrix mult, O(n³))
        // Instead omeco creates larger intermediate.
    }
}
