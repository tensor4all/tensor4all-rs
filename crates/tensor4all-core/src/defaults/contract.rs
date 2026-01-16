//! Multi-tensor contraction with optimal contraction order.
//!
//! This module provides functions to contract multiple tensors efficiently
//! using hyperedge-aware einsum optimization via mdarray-einsum.
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.
//!
//! # Main Functions
//!
//! - [`contract_multi`]: Contracts tensors, handling disconnected components via outer product
//! - [`contract_connected`]: Contracts tensors that must form a connected graph

use std::collections::HashMap;

use anyhow::Result;
use petgraph::algo::connected_components;
use petgraph::prelude::*;

use crate::defaults::contraction_opt::contract_multi_diag_aware;
use crate::defaults::TensorDynLen;
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
/// let result = contract_multi(&[&a, &b, &c], AllowedPairs::All)?;
///
/// // Disconnected tensors: contracts each component, outer product to combine
/// let result = contract_multi(&[&a, &b], AllowedPairs::All)?;  // a, b have no common indices
/// ```
pub fn contract_multi(
    tensors: &[&TensorDynLen],
    allowed: AllowedPairs<'_>,
) -> Result<TensorDynLen> {
    match tensors.len() {
        0 => Err(anyhow::anyhow!("No tensors to contract")),
        1 => Ok((*tensors[0]).clone()),
        _ => {
            // Validate AllowedPairs::Specified pairs have contractable indices
            if let AllowedPairs::Specified(pairs) = allowed {
                for &(i, j) in pairs {
                    if !has_contractable_indices(tensors[i], tensors[j]) {
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
                    let component_tensors: Vec<&TensorDynLen> =
                        component.iter().map(|&i| tensors[i]).collect();

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
/// Uses hyperedge-aware einsum optimization via mdarray-einsum.
/// This correctly handles Diag tensors by treating their diagonal axes as hyperedges.
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
/// - N>=2: Optimal order via hyperedge-aware greedy optimizer
///
/// # Example
/// ```ignore
/// use tensor4all_core::{contract_connected, AllowedPairs};
///
/// let tensors = vec![tensor_a, tensor_b, tensor_c];  // Must be connected
/// let tensor_refs: Vec<&_> = tensors.iter().collect();
/// let result = contract_connected(&tensor_refs, AllowedPairs::All)?;
/// ```
pub fn contract_connected(
    tensors: &[&TensorDynLen],
    allowed: AllowedPairs<'_>,
) -> Result<TensorDynLen> {
    // Delegate to Diag-aware contraction which handles both Dense and Diag tensors
    contract_multi_diag_aware(tensors, allowed)
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
/// * `tensors` - Slice of tensor references
/// * `allowed` - Which tensor pairs can have their indices contracted
///
/// # Returns
/// A vector of components, where each component is a vector of tensor indices.
/// Components are sorted by their smallest tensor index for determinism.
fn find_tensor_connected_components(
    tensors: &[&TensorDynLen],
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
                    if has_contractable_indices(tensors[i], tensors[j]) {
                        graph.add_edge(nodes[i], nodes[j], ());
                    }
                }
            }
        }
        AllowedPairs::Specified(pairs) => {
            // Only specified pairs with contractable indices are connected
            for &(i, j) in pairs {
                if has_contractable_indices(tensors[i], tensors[j]) {
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
        let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            data, &dims,
        )));
        TensorDynLen::new(indices, dims, storage)
    }

    #[test]
    fn test_contract_multi_empty() {
        let tensors: Vec<&TensorDynLen> = vec![];
        let result = contract_multi(&tensors, AllowedPairs::All);
        assert!(result.is_err());
    }

    #[test]
    fn test_contract_multi_single() {
        let tensor = make_test_tensor(&[2, 3], &[1, 2]);
        let result = contract_multi(&[&tensor], AllowedPairs::All).unwrap();
        assert_eq!(result.dims, tensor.dims);
    }

    #[test]
    fn test_contract_multi_pair() {
        // A[i,j] * B[j,k] -> C[i,k]
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let result = contract_multi(&[&a, &b], AllowedPairs::All).unwrap();
        assert_eq!(result.dims, vec![2, 4]); // i, k
    }

    #[test]
    fn test_contract_multi_three() {
        // A[i,j] * B[j,k] * C[k,l] -> D[i,l]
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let c = make_test_tensor(&[4, 5], &[3, 4]); // k=3, l=4
        let result = contract_multi(&[&a, &b, &c], AllowedPairs::All).unwrap();
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
        let result = contract_multi(&[&a, &b, &c, &d], AllowedPairs::All).unwrap();
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
        let result = contract_multi(&[&a, &b], AllowedPairs::All).unwrap();
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
        let result = contract_multi(&[&a, &b], AllowedPairs::All).unwrap();
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
        let result = contract_connected(&[&a, &b], AllowedPairs::All);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .to_lowercase()
            .contains("disconnected"));
    }

    #[test]
    fn test_contract_connected_specified_no_contractable_error() {
        // contract_connected with Specified pair that has no contractable indices
        // Should give error (disconnected since no shared indices)
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[4, 5], &[3, 4]); // k=3, l=4 (no common with a)
        let result = contract_connected(&[&a, &b], AllowedPairs::Specified(&[(0, 1)]));
        assert!(result.is_err());
        // Error message may be about disconnected (since no shared indices)
        let err_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            err_msg.contains("disconnected") || err_msg.contains("no contractable"),
            "Expected error about disconnected or no contractable indices, got: {}",
            err_msg
        );
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
        let result =
            contract_multi(&[&a, &b, &c], AllowedPairs::Specified(&[(0, 1), (0, 2)])).unwrap();
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
        let result = contract_multi(&[&a, &b, &c], AllowedPairs::Specified(&[(0, 1), (1, 2)]));
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
        let result = contract_multi(
            &[&a, &b, &c, &d],
            AllowedPairs::Specified(&[(0, 1), (2, 3)]),
        )
        .unwrap();
        // A-B contracts j: result has i, k (dims 2, 4)
        // C-D contracts n: result has m, p (dims 4, 6)
        // Outer product: result has 4 indices
        assert_eq!(result.dims.len(), 4);
        let mut sorted_dims = result.dims.clone();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![2, 4, 4, 6]);
    }

}
