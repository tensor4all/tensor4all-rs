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
//!
//! # Diag Tensor Handling
//!
//! When Diag tensors share indices, their diagonal axes are unified to create
//! hyperedges in the einsum optimizer.
//!
//! Example: `Diag(i,j) * Diag(j,k)`:
//! - Diag(i,j) has diagonal axes i and j (same index)
//! - Diag(j,k) has diagonal axes j and k (same index)
//! - After union-find: i, j, k all map to the same representative ID
//! - This creates a hyperedge that the einsum optimizer handles correctly

use std::collections::HashMap;

use anyhow::Result;
use mdarray_einsum::{einsum_optimized, Naive, TypedTensor};
use num_complex::Complex64;
use petgraph::algo::connected_components;
use petgraph::prelude::*;
use std::sync::Arc;
use tensor4all_tensorbackend::mdarray::{Dense, DynRank, Slice, Tensor};

use crate::defaults::{DynId, DynIndex, TensorComponent, TensorData, TensorDynLen};

/// Type alias for einsum input references (to satisfy clippy type_complexity).
type EinsumInput<'a, T> = (&'a [usize], &'a Slice<T, DynRank, Dense>);
use crate::index_like::IndexLike;
use crate::storage::{DenseStorageC64, DenseStorageF64, Storage};
use crate::tensor_like::AllowedPairs;

// ============================================================================
// Public API
// ============================================================================

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
                // All tensors connected - use optimized contraction (skip connectivity check)
                contract_multi_impl(tensors, allowed, true)
            } else {
                // Multiple components - contract each and combine with outer product
                let mut results: Vec<TensorDynLen> = Vec::new();
                for component in &components {
                    let component_tensors: Vec<&TensorDynLen> =
                        component.iter().map(|&i| tensors[i]).collect();

                    // Remap AllowedPairs for the component (connectivity already verified)
                    let remapped_allowed = remap_allowed_pairs(allowed, component);
                    let contracted =
                        contract_multi_impl(&component_tensors, remapped_allowed.as_ref(), true)?;
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
    match tensors.len() {
        0 => Err(anyhow::anyhow!("No tensors to contract")),
        1 => Ok((*tensors[0]).clone()),
        _ => {
            // Check connectivity first
            let components = find_tensor_connected_components(tensors, allowed);
            if components.len() > 1 {
                return Err(anyhow::anyhow!(
                    "Disconnected tensor network: {} components found",
                    components.len()
                ));
            }
            // Connectivity verified - skip check in impl
            contract_multi_impl(tensors, allowed, true)
        }
    }
}

// ============================================================================
// Union-Find for Diag axis grouping
// ============================================================================

/// Union-Find data structure for grouping axis IDs.
///
/// Used to merge diagonal axes from Diag tensors so that they share
/// the same representative ID when passed to einsum.
#[derive(Debug, Clone)]
pub struct AxisUnionFind {
    /// Maps each ID to its parent. If parent[id] == id, it's a root.
    parent: HashMap<DynId, DynId>,
    /// Rank for union by rank optimization.
    rank: HashMap<DynId, usize>,
}

impl AxisUnionFind {
    /// Create a new empty union-find structure.
    pub fn new() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
        }
    }

    /// Add an ID to the structure (as its own set).
    pub fn make_set(&mut self, id: DynId) {
        use std::collections::hash_map::Entry;
        if let Entry::Vacant(e) = self.parent.entry(id) {
            e.insert(id);
            self.rank.insert(id, 0);
        }
    }

    /// Find the representative (root) of the set containing `id`.
    /// Uses path compression for efficiency.
    pub fn find(&mut self, id: DynId) -> DynId {
        self.make_set(id);
        if self.parent[&id] != id {
            let root = self.find(self.parent[&id]);
            self.parent.insert(id, root);
        }
        self.parent[&id]
    }

    /// Union the sets containing `a` and `b`.
    /// Uses union by rank for efficiency.
    pub fn union(&mut self, a: DynId, b: DynId) {
        let root_a = self.find(a);
        let root_b = self.find(b);

        if root_a == root_b {
            return;
        }

        let rank_a = self.rank[&root_a];
        let rank_b = self.rank[&root_b];

        if rank_a < rank_b {
            self.parent.insert(root_a, root_b);
        } else if rank_a > rank_b {
            self.parent.insert(root_b, root_a);
        } else {
            self.parent.insert(root_b, root_a);
            *self.rank.get_mut(&root_a).unwrap() += 1;
        }
    }

    /// Remap an ID to its representative.
    pub fn remap(&mut self, id: DynId) -> DynId {
        self.find(id)
    }

    /// Remap a slice of IDs to their representatives.
    pub fn remap_ids(&mut self, ids: &[DynId]) -> Vec<DynId> {
        ids.iter().map(|id| self.find(*id)).collect()
    }
}

impl Default for AxisUnionFind {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Diag union-find builders
// ============================================================================

/// Build a union-find structure from TensorData components.
///
/// For each Diag component, all its indices are unified (they share the same
/// diagonal dimension). This creates hyperedges when multiple Diag components
/// share indices.
pub fn build_diag_union_from_components(components: &[&TensorComponent]) -> AxisUnionFind {
    let mut uf = AxisUnionFind::new();

    for component in components {
        // Add all indices to the union-find
        for &id in component.index_ids.iter() {
            uf.make_set(id);
        }

        // For Diag storage, union all diagonal axes
        if component.storage.is_diag() && component.index_ids.len() >= 2 {
            let first_id = component.index_ids[0];
            for &id in component.index_ids.iter().skip(1) {
                uf.union(first_id, id);
            }
        }
    }

    uf
}

/// Build a union-find structure from TensorData.
///
/// Processes all components in the TensorData, unifying diagonal axes
/// from Diag storage components.
pub fn build_diag_union_from_data(data: &TensorData) -> AxisUnionFind {
    let component_refs: Vec<&TensorComponent> = data.components.iter().collect();
    build_diag_union_from_components(&component_refs)
}

/// Build a union-find structure from a collection of tensors.
///
/// For each Diag tensor component, all its indices are unified (they share the same
/// diagonal dimension). This creates hyperedges when multiple Diag tensors
/// share indices.
pub fn build_diag_union(tensors: &[&TensorDynLen]) -> AxisUnionFind {
    let all_components: Vec<&TensorComponent> = tensors
        .iter()
        .flat_map(|t| t.tensor_data().components.iter())
        .collect();

    build_diag_union_from_components(&all_components)
}

/// Remap tensor indices using the union-find structure.
///
/// Returns a vector of remapped IDs for each tensor, suitable for passing
/// to einsum. The original tensors are not modified.
pub fn remap_tensor_ids(tensors: &[&TensorDynLen], uf: &mut AxisUnionFind) -> Vec<Vec<DynId>> {
    tensors
        .iter()
        .map(|t| t.indices.iter().map(|idx| uf.find(*idx.id())).collect())
        .collect()
}

/// Remap output IDs using the union-find structure.
pub fn remap_output_ids(output: &[DynIndex], uf: &mut AxisUnionFind) -> Vec<DynId> {
    output.iter().map(|idx| uf.find(*idx.id())).collect()
}

/// Collect dimension sizes for remapped IDs.
///
/// For unified IDs (from Diag tensors), all axes must have the same dimension,
/// so we just take the first occurrence.
pub fn collect_sizes(tensors: &[&TensorDynLen], uf: &mut AxisUnionFind) -> HashMap<DynId, usize> {
    let mut sizes = HashMap::new();

    for tensor in tensors {
        let dims = tensor.dims();
        for (idx, &dim) in tensor.indices.iter().zip(dims.iter()) {
            let rep = uf.find(*idx.id());
            sizes.entry(rep).or_insert(dim);
        }
    }

    sizes
}

// ============================================================================
// Contraction implementation
// ============================================================================

/// Internal implementation of multi-tensor contraction.
///
/// For Diag tensors, we pass them as 1D tensors (the diagonal elements) with
/// a single hyperedge ID. The einsum hyperedge optimizer will handle them correctly.
///
/// This implementation preserves storage type: if all inputs are F64, the result
/// is F64; if any input is C64, the result is C64.
///
/// # Arguments
/// * `skip_connectivity_check` - If true, assumes connectivity was already verified by caller
fn contract_multi_impl(
    tensors: &[&TensorDynLen],
    allowed: AllowedPairs<'_>,
    _skip_connectivity_check: bool,
) -> Result<TensorDynLen> {
    // 1. Build union-find from Diag tensors to unify diagonal axes
    let mut diag_uf = build_diag_union(tensors);

    // 2. Build internal IDs with Diag-awareness
    let (ixs, internal_id_to_original) = build_internal_ids(tensors, allowed, &mut diag_uf)?;

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

    // Note: Connectivity check is done by caller (contract_multi or contract_connected)
    // via find_tensor_connected_components before calling this function

    // 4. Build sizes from unique internal IDs
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        let dims = tensor.dims();
        for (pos, &dim) in dims.iter().enumerate() {
            let internal_id = ixs[tensor_idx][pos];
            sizes.entry(internal_id).or_insert(dim);
        }
    }

    // 6. Convert TensorDynLen to TypedTensor, tracking if any input is C64
    let mut typed_tensors: Vec<TypedTensor> = Vec::new();
    let mut einsum_ids: Vec<Vec<usize>> = Vec::new();
    let mut has_c64_input = false;

    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        let storage = tensor.materialize_storage()?;
        let is_diag = storage.is_diag();

        if is_diag {
            // Diag tensor: extract diagonal as 1D tensor
            let typed = match &*storage {
                Storage::DiagC64(ds) => {
                    has_c64_input = true;
                    let diag_len = ds.as_slice().len();
                    let tensor_1d = Tensor::from(ds.as_slice().to_vec())
                        .into_shape([diag_len].as_slice())
                        .into_dyn();
                    TypedTensor::C64(tensor_1d)
                }
                Storage::DiagF64(ds) => {
                    let diag_len = ds.as_slice().len();
                    let tensor_1d = Tensor::from(ds.as_slice().to_vec())
                        .into_shape([diag_len].as_slice())
                        .into_dyn();
                    TypedTensor::F64(tensor_1d)
                }
                _ => return Err(anyhow::anyhow!("Expected Diag storage")),
            };
            typed_tensors.push(typed);

            // Diag tensor with unified axes: use single hyperedge ID
            let hyperedge_id = ixs[tensor_idx][0];
            einsum_ids.push(vec![hyperedge_id]);
        } else {
            // Dense tensor
            let typed = match &*storage {
                Storage::DenseC64(ds) => {
                    has_c64_input = true;
                    TypedTensor::C64(ds.tensor().clone())
                }
                Storage::DenseF64(ds) => TypedTensor::F64(ds.tensor().clone()),
                _ => return Err(anyhow::anyhow!("Expected Dense storage")),
            };
            typed_tensors.push(typed);
            einsum_ids.push(ixs[tensor_idx].clone());
        }
    }

    // 7. Perform einsum based on input types
    let result_typed = if has_c64_input {
        // Mixed or all C64: use C64 einsum
        let c64_tensors: Vec<Tensor<Complex64, DynRank>> =
            typed_tensors.iter().map(|t| t.to_c64()).collect();
        let inputs: Vec<EinsumInput<Complex64>> = einsum_ids
            .iter()
            .zip(c64_tensors.iter())
            .map(|(ids, tensor)| (ids.as_slice(), tensor.as_ref()))
            .collect();
        let result = einsum_optimized(&Naive, &inputs, &output, &sizes);
        TypedTensor::C64(result)
    } else {
        // All F64: use F64 einsum
        let f64_tensors: Vec<&Tensor<f64, DynRank>> = typed_tensors
            .iter()
            .map(|t| t.as_f64().expect("Expected F64 tensor"))
            .collect();
        let inputs: Vec<EinsumInput<f64>> = einsum_ids
            .iter()
            .zip(f64_tensors.iter())
            .map(|(ids, tensor)| (ids.as_slice(), tensor.as_ref()))
            .collect();
        let result = einsum_optimized(&Naive, &inputs, &output, &sizes);
        TypedTensor::F64(result)
    };

    // 8. Convert result back to TensorDynLen
    let result_dims: Vec<usize> = result_typed.dims().to_vec();

    // Build result indices from output internal IDs
    let restored_indices: Vec<DynIndex> = output
        .iter()
        .map(|&internal_id| {
            let (tensor_idx, pos) = internal_id_to_original[&internal_id];
            tensors[tensor_idx].indices[pos].clone()
        })
        .collect();

    // Handle scalar output
    let (final_dims, final_indices) = if output.is_empty() && result_dims == vec![1] {
        (vec![], vec![])
    } else {
        (result_dims, restored_indices)
    };

    // Create storage from result based on type
    let storage = match result_typed {
        TypedTensor::F64(t) => {
            let data: Vec<f64> = t.into_vec();
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                data,
                &final_dims,
            )))
        }
        TypedTensor::C64(t) => {
            let data: Vec<Complex64> = t.into_vec();
            Arc::new(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
                data,
                &final_dims,
            )))
        }
    };

    Ok(TensorDynLen::new(final_indices, storage))
}

/// Build internal IDs with Diag-awareness.
///
/// Uses the union-find to ensure diagonal axes from Diag tensors share the same internal ID.
///
/// Returns: (ixs, internal_id_to_original)
#[allow(clippy::type_complexity)]
fn build_internal_ids(
    tensors: &[&TensorDynLen],
    allowed: AllowedPairs<'_>,
    diag_uf: &mut AxisUnionFind,
) -> Result<(Vec<Vec<usize>>, HashMap<usize, (usize, usize)>)> {
    let mut next_id = 0usize;
    let mut dynid_to_internal: HashMap<DynId, usize> = HashMap::new();
    let mut assigned: HashMap<(usize, usize), usize> = HashMap::new();
    let mut internal_id_to_original: HashMap<usize, (usize, usize)> = HashMap::new();

    // Process contractable pairs
    let pairs_to_process: Vec<(usize, usize)> = match allowed {
        AllowedPairs::All => {
            let mut pairs = Vec::new();
            for ti in 0..tensors.len() {
                for tj in (ti + 1)..tensors.len() {
                    pairs.push((ti, tj));
                }
            }
            pairs
        }
        AllowedPairs::Specified(pairs) => pairs.to_vec(),
    };

    for (ti, tj) in pairs_to_process {
        for (pi, idx_i) in tensors[ti].indices.iter().enumerate() {
            for (pj, idx_j) in tensors[tj].indices.iter().enumerate() {
                if idx_i.is_contractable(idx_j) {
                    let key_i = (ti, pi);
                    let key_j = (tj, pj);

                    let remapped_i = diag_uf.find(*idx_i.id());
                    let remapped_j = diag_uf.find(*idx_j.id());

                    match (assigned.get(&key_i).copied(), assigned.get(&key_j).copied()) {
                        (None, None) => {
                            let internal_id = if let Some(&id) = dynid_to_internal.get(&remapped_i)
                            {
                                id
                            } else {
                                let id = next_id;
                                next_id += 1;
                                dynid_to_internal.insert(remapped_i, id);
                                internal_id_to_original.insert(id, key_i);
                                id
                            };
                            assigned.insert(key_i, internal_id);
                            assigned.insert(key_j, internal_id);
                            if remapped_i != remapped_j {
                                dynid_to_internal.insert(remapped_j, internal_id);
                            }
                        }
                        (Some(id), None) => {
                            assigned.insert(key_j, id);
                            dynid_to_internal.insert(remapped_j, id);
                        }
                        (None, Some(id)) => {
                            assigned.insert(key_i, id);
                            dynid_to_internal.insert(remapped_i, id);
                        }
                        (Some(_id_i), Some(_id_j)) => {
                            // Both already assigned
                        }
                    }
                }
            }
        }
    }

    // Assign IDs for unassigned indices (external indices)
    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        for (pos, idx) in tensor.indices.iter().enumerate() {
            let key = (tensor_idx, pos);
            if let std::collections::hash_map::Entry::Vacant(e) = assigned.entry(key) {
                let remapped_id = diag_uf.find(*idx.id());

                let internal_id = if let Some(&id) = dynid_to_internal.get(&remapped_id) {
                    id
                } else {
                    let id = next_id;
                    next_id += 1;
                    dynid_to_internal.insert(remapped_id, id);
                    internal_id_to_original.insert(id, key);
                    id
                };
                e.insert(internal_id);
            }
        }
    }

    // Build ixs
    let ixs: Vec<Vec<usize>> = tensors
        .iter()
        .enumerate()
        .map(|(tensor_idx, tensor)| {
            (0..tensor.indices.len())
                .map(|pos| assigned[&(tensor_idx, pos)])
                .collect()
        })
        .collect();

    Ok((ixs, internal_id_to_original))
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
            for i in 0..n {
                for j in (i + 1)..n {
                    if has_contractable_indices(tensors[i], tensors[j]) {
                        graph.add_edge(nodes[i], nodes[j], ());
                    }
                }
            }
        }
        AllowedPairs::Specified(pairs) => {
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
        return vec![(0..n).collect()];
    }

    // Multiple components - group by component ID
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
            component.sort();
            components.push(component);
        }
    }

    components.sort_by_key(|c| c[0]);
    components
}

/// Remap AllowedPairs for a subset of tensors.
fn remap_allowed_pairs(allowed: AllowedPairs<'_>, component: &[usize]) -> RemappedAllowedPairs {
    match allowed {
        AllowedPairs::All => RemappedAllowedPairs::All,
        AllowedPairs::Specified(pairs) => {
            let orig_to_local: HashMap<usize, usize> = component
                .iter()
                .enumerate()
                .map(|(local, &orig)| (orig, local))
                .collect();

            let remapped: Vec<(usize, usize)> = pairs
                .iter()
                .filter_map(
                    |&(i, j)| match (orig_to_local.get(&i), orig_to_local.get(&j)) {
                        (Some(&li), Some(&lj)) => Some((li, lj)),
                        _ => None,
                    },
                )
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
    use crate::defaults::Index;
    use num_complex::Complex64;

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
        TensorDynLen::new(indices, storage)
    }

    // ========================================================================
    // contract_multi tests
    // ========================================================================

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
        assert_eq!(result.dims(), tensor.dims());
    }

    #[test]
    fn test_contract_multi_pair() {
        // A[i,j] * B[j,k] -> C[i,k]
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let result = contract_multi(&[&a, &b], AllowedPairs::All).unwrap();
        assert_eq!(result.dims(), vec![2, 4]); // i, k
    }

    #[test]
    fn test_contract_multi_three() {
        // A[i,j] * B[j,k] * C[k,l] -> D[i,l]
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let c = make_test_tensor(&[4, 5], &[3, 4]); // k=3, l=4
        let result = contract_multi(&[&a, &b, &c], AllowedPairs::All).unwrap();
        let mut sorted_dims = result.dims();
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
        let mut sorted_dims = result.dims();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![2, 6]); // i=2, m=6
    }

    #[test]
    fn test_contract_multi_outer_product() {
        // A[i,j] * B[k,l] (no common indices) -> outer product C[i,j,k,l]
        let a = make_test_tensor(&[2, 3], &[1, 2]);
        let b = make_test_tensor(&[4, 5], &[3, 4]);
        let result = contract_multi(&[&a, &b], AllowedPairs::All).unwrap();
        let result_dims = result.dims();
        let total_elements: usize = result_dims.iter().product();
        assert_eq!(total_elements, 2 * 3 * 4 * 5);
        assert_eq!(result_dims.len(), 4);
    }

    #[test]
    fn test_contract_multi_vector_outer_product() {
        // A[i] * B[j] (no common indices) -> outer product C[i,j]
        let a = make_test_tensor(&[2], &[1]); // i=1
        let b = make_test_tensor(&[3], &[2]); // j=2
        let result = contract_multi(&[&a, &b], AllowedPairs::All).unwrap();
        let result_dims = result.dims();
        let total_elements: usize = result_dims.iter().product();
        assert_eq!(total_elements, 2 * 3);
        assert_eq!(result.dims().len(), 2);
    }

    #[test]
    fn test_contract_connected_disconnected_error() {
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
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[4, 5], &[3, 4]); // k=3, l=4 (no common with a)
        let result = contract_connected(&[&a, &b], AllowedPairs::Specified(&[(0, 1)]));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            err_msg.contains("disconnected") || err_msg.contains("no contractable"),
            "Expected error about disconnected or no contractable indices, got: {}",
            err_msg
        );
    }

    // ========================================================================
    // AllowedPairs::Specified tests
    // ========================================================================

    #[test]
    fn test_contract_specified_pairs() {
        // A[i,j], B[j,k], C[i,l] - tensors 0, 1, 2
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let c = make_test_tensor(&[2, 5], &[1, 4]); // i=1, l=4
        let result =
            contract_multi(&[&a, &b, &c], AllowedPairs::Specified(&[(0, 1), (0, 2)])).unwrap();
        let mut sorted_dims = result.dims();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![4, 5]); // k=4, l=5
    }

    #[test]
    fn test_contract_specified_no_contractable_indices_error() {
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
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let c = make_test_tensor(&[4, 5], &[4, 5]); // m=4, n=5
        let d = make_test_tensor(&[5, 6], &[5, 6]); // n=5, p=6
        let result = contract_multi(
            &[&a, &b, &c, &d],
            AllowedPairs::Specified(&[(0, 1), (2, 3)]),
        )
        .unwrap();
        assert_eq!(result.dims().len(), 4);
        let mut sorted_dims = result.dims();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![2, 4, 4, 6]);
    }

    // ========================================================================
    // Union-Find tests
    // ========================================================================

    #[test]
    fn test_union_find_basic() {
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);
        let c = DynId(3);

        uf.make_set(a);
        uf.make_set(b);
        uf.make_set(c);

        assert_ne!(uf.find(a), uf.find(b));
        assert_ne!(uf.find(b), uf.find(c));

        uf.union(a, b);
        assert_eq!(uf.find(a), uf.find(b));
        assert_ne!(uf.find(a), uf.find(c));

        uf.union(b, c);
        assert_eq!(uf.find(a), uf.find(b));
        assert_eq!(uf.find(b), uf.find(c));
        assert_eq!(uf.find(a), uf.find(c));
    }

    #[test]
    fn test_union_find_chain() {
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);
        let l = DynId(4);

        uf.union(i, j);
        uf.union(j, k);
        uf.union(k, l);

        let rep = uf.find(i);
        assert_eq!(uf.find(j), rep);
        assert_eq!(uf.find(k), rep);
        assert_eq!(uf.find(l), rep);
    }

    #[test]
    fn test_remap_ids() {
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);

        uf.union(i, j);

        let ids = vec![i, j, k];
        let remapped = uf.remap_ids(&ids);

        assert_eq!(remapped[0], remapped[1]);
        assert_ne!(remapped[0], remapped[2]);
    }

    #[test]
    fn test_three_diag_chain() {
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);
        let l = DynId(4);

        uf.union(i, j);
        uf.union(j, k);
        uf.union(k, l);

        let rep = uf.find(i);
        assert_eq!(uf.find(j), rep);
        assert_eq!(uf.find(k), rep);
        assert_eq!(uf.find(l), rep);
    }

    #[test]
    fn test_three_diag_star() {
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);
        let c = DynId(3);
        let d = DynId(4);

        uf.union(a, b);
        uf.union(a, c);
        uf.union(a, d);

        let rep = uf.find(a);
        assert_eq!(uf.find(b), rep);
        assert_eq!(uf.find(c), rep);
        assert_eq!(uf.find(d), rep);
    }

    #[test]
    fn test_diag_with_three_axes() {
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);
        let l = DynId(4);

        uf.union(i, j);
        uf.union(j, k);

        let rep = uf.find(i);
        assert_eq!(uf.find(j), rep);
        assert_eq!(uf.find(k), rep);
        assert_ne!(uf.find(l), rep);
    }

    #[test]
    fn test_two_separate_diag_groups() {
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);
        let c = DynId(3);
        let d = DynId(4);

        uf.union(a, b);
        uf.union(c, d);

        assert_eq!(uf.find(a), uf.find(b));
        assert_eq!(uf.find(c), uf.find(d));
        assert_ne!(uf.find(a), uf.find(c));
    }

    #[test]
    fn test_diag_and_dense_mixed() {
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);

        uf.union(i, j);
        uf.make_set(k);

        assert_eq!(uf.find(i), uf.find(j));
        assert_ne!(uf.find(j), uf.find(k));
    }

    #[test]
    fn test_complex_network() {
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);
        let c = DynId(3);
        let d = DynId(4);
        let e = DynId(5);
        let f = DynId(6);

        uf.union(a, b);
        uf.union(b, c);
        uf.make_set(d);
        uf.union(d, e);
        uf.union(e, f);

        let rep1 = uf.find(a);
        assert_eq!(uf.find(b), rep1);
        assert_eq!(uf.find(c), rep1);

        let rep2 = uf.find(d);
        assert_eq!(uf.find(e), rep2);
        assert_eq!(uf.find(f), rep2);

        assert_ne!(rep1, rep2);
    }

    #[test]
    fn test_single_diag_tensor() {
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);

        uf.union(i, j);

        assert_eq!(uf.find(i), uf.find(j));
    }

    #[test]
    fn test_empty_union_find() {
        let mut uf = AxisUnionFind::new();

        let x = DynId(42);
        uf.make_set(x);
        assert_eq!(uf.find(x), x);
    }

    #[test]
    fn test_idempotent_union() {
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);

        uf.union(a, b);
        let rep1 = uf.find(a);

        uf.union(a, b);
        let rep2 = uf.find(a);

        uf.union(b, a);
        let rep3 = uf.find(a);

        assert_eq!(rep1, rep2);
        assert_eq!(rep2, rep3);
    }

    #[test]
    fn test_self_union() {
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        uf.union(a, a);

        assert_eq!(uf.find(a), a);
    }

    #[test]
    fn test_four_diag_tensors_chain() {
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);
        let l = DynId(4);
        let m = DynId(5);

        uf.union(i, j);
        uf.union(j, k);
        uf.union(k, l);
        uf.union(l, m);

        let rep = uf.find(i);
        assert_eq!(uf.find(j), rep);
        assert_eq!(uf.find(k), rep);
        assert_eq!(uf.find(l), rep);
        assert_eq!(uf.find(m), rep);
    }

    #[test]
    fn test_diag_tensors_merge_two_chains() {
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);
        let c = DynId(3);
        let d = DynId(4);
        let e = DynId(5);

        uf.union(a, b);
        uf.union(b, c);
        uf.union(d, e);
        uf.union(e, c);

        let rep = uf.find(a);
        assert_eq!(uf.find(b), rep);
        assert_eq!(uf.find(c), rep);
        assert_eq!(uf.find(d), rep);
        assert_eq!(uf.find(e), rep);
    }

    #[test]
    fn test_remap_preserves_order() {
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);
        let l = DynId(4);

        uf.union(i, j);
        uf.union(k, l);

        let ids = vec![i, j, k, l, i, k];
        let remapped = uf.remap_ids(&ids);

        assert_eq!(remapped.len(), 6);
        assert_eq!(remapped[0], remapped[1]);
        assert_eq!(remapped[2], remapped[3]);
        assert_ne!(remapped[0], remapped[2]);
        assert_eq!(remapped[0], remapped[4]);
        assert_eq!(remapped[2], remapped[5]);
    }

    // ========================================================================
    // contract_connected tests
    // ========================================================================

    fn make_dense_tensor(shape: &[usize], ids: &[u128]) -> TensorDynLen {
        let indices: Vec<DynIndex> = ids
            .iter()
            .zip(shape.iter())
            .map(|(&id, &dim)| Index::new(DynId(id), dim))
            .collect();
        let dims = shape.to_vec();
        let total_size: usize = shape.iter().product();
        let data: Vec<Complex64> = (0..total_size)
            .map(|i| Complex64::new((i + 1) as f64, 0.0))
            .collect();
        let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            data, &dims,
        )));
        TensorDynLen::new(indices, storage)
    }

    #[test]
    fn test_contract_connected_empty() {
        let tensors: Vec<&TensorDynLen> = vec![];
        let result = contract_connected(&tensors, AllowedPairs::All);
        assert!(result.is_err());
    }

    #[test]
    fn test_contract_connected_single() {
        let tensor = make_dense_tensor(&[2, 3], &[1, 2]);
        let result = contract_connected(&[&tensor], AllowedPairs::All).unwrap();
        assert_eq!(result.dims(), tensor.dims());
    }

    #[test]
    fn test_contract_connected_pair_dense() {
        // A[i,j] * B[j,k] -> C[i,k]
        let a = make_dense_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_dense_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let result = contract_connected(&[&a, &b], AllowedPairs::All).unwrap();
        assert_eq!(result.dims(), vec![2, 4]); // i, k
    }

    #[test]
    fn test_contract_connected_three_dense() {
        // A[i,j] * B[j,k] * C[k,l] -> D[i,l]
        let a = make_dense_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_dense_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let c = make_dense_tensor(&[4, 5], &[3, 4]); // k=3, l=4
        let result = contract_connected(&[&a, &b, &c], AllowedPairs::All).unwrap();
        let mut sorted_dims = result.dims();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![2, 5]); // i=2, l=5
    }
}
