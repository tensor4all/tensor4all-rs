//! Multi-tensor contraction with optimal contraction order.
//!
//! This module provides functions to contract multiple tensors efficiently
//! using hyperedge-aware einsum optimization via the tensorbackend
//! (tenferro-backed implementation).
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

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::time::{Duration, Instant};

use anyhow::Result;
use petgraph::algo::connected_components;
use petgraph::prelude::*;
use tensor4all_tensorbackend::einsum_native_tensors;

use crate::defaults::{DynId, DynIndex, TensorDynLen};

use crate::index_like::IndexLike;
use crate::tensor_like::AllowedPairs;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ContractOperandSignature {
    dims: Vec<usize>,
    ids: Vec<usize>,
    is_diag: bool,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ContractSignature {
    operands: Vec<ContractOperandSignature>,
    output_ids: Vec<usize>,
    output_dims: Vec<usize>,
}

#[derive(Debug, Default, Clone)]
struct ContractProfileEntry {
    calls: usize,
    total_time: Duration,
}

thread_local! {
    static CONTRACT_PROFILE_STATE: RefCell<HashMap<ContractSignature, ContractProfileEntry>> =
        RefCell::new(HashMap::new());
}

fn contract_profile_enabled() -> bool {
    env::var("T4A_PROFILE_CONTRACT").is_ok()
}

fn record_contract_profile(signature: ContractSignature, elapsed: Duration) {
    if !contract_profile_enabled() {
        return;
    }
    CONTRACT_PROFILE_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let entry = state.entry(signature).or_default();
        entry.calls += 1;
        entry.total_time += elapsed;
    });
}

/// Reset the aggregated multi-tensor contraction profile.
pub fn reset_contract_profile() {
    CONTRACT_PROFILE_STATE.with(|state| state.borrow_mut().clear());
}

/// Print and clear the aggregated multi-tensor contraction profile.
pub fn print_and_reset_contract_profile() {
    if !contract_profile_enabled() {
        return;
    }
    CONTRACT_PROFILE_STATE.with(|state| {
        let mut entries: Vec<_> = state
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        state.borrow_mut().clear();
        entries.sort_by(|(_, lhs), (_, rhs)| rhs.total_time.cmp(&lhs.total_time));

        eprintln!("=== contract_multi Profile ===");
        for (idx, (signature, entry)) in entries.into_iter().take(20).enumerate() {
            let operands = signature
                .operands
                .iter()
                .map(|operand| {
                    format!(
                        "dims={:?} ids={:?}{}",
                        operand.dims,
                        operand.ids,
                        if operand.is_diag { " diag" } else { "" }
                    )
                })
                .collect::<Vec<_>>()
                .join(" ; ");
            eprintln!(
                "#{idx:02} calls={} total={:.3}s per_call={:.3}us output_dims={:?} output_ids={:?}",
                entry.calls,
                entry.total_time.as_secs_f64(),
                entry.total_time.as_secs_f64() * 1e6 / entry.calls as f64,
                signature.output_dims,
                signature.output_ids,
            );
            eprintln!("     {operands}");
        }
    });
}

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
/// # Examples
///
/// ```
/// use tensor4all_core::{TensorDynLen, DynIndex, contract_multi, AllowedPairs};
///
/// // A[i, j] and B[j, k] share index j — contract to get C[i, k]
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let k = DynIndex::new_dyn(4);
///
/// let a = TensorDynLen::from_dense(
///     vec![i.clone(), j.clone()],
///     vec![1.0_f64; 6],
/// ).unwrap();
/// let b = TensorDynLen::from_dense(
///     vec![j.clone(), k.clone()],
///     vec![1.0_f64; 12],
/// ).unwrap();
///
/// let c = contract_multi(&[&a, &b], AllowedPairs::All).unwrap();
/// assert_eq!(c.dims(), vec![2, 4]);
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
                let mut results_iter = results.into_iter();
                let mut result = results_iter.next().unwrap();
                for other in results_iter {
                    result = result.outer_product(&other)?;
                }
                Ok(result)
            }
        }
    }
}

/// Contract multiple tensors that form a connected graph.
///
/// Uses hyperedge-aware einsum optimization via tensorbackend.
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
/// # Examples
///
/// ```
/// use tensor4all_core::{TensorDynLen, DynIndex, contract_connected, AllowedPairs};
///
/// // A[i, j] contracted with B[j, k]
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let k = DynIndex::new_dyn(4);
///
/// let a = TensorDynLen::from_dense(
///     vec![i.clone(), j.clone()],
///     vec![1.0_f64; 6],
/// ).unwrap();
/// let b = TensorDynLen::from_dense(
///     vec![j.clone(), k.clone()],
///     vec![1.0_f64; 12],
/// ).unwrap();
///
/// let c = contract_connected(&[&a, &b], AllowedPairs::All).unwrap();
/// assert_eq!(c.dims(), vec![2, 4]);
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

/// Build a union-find structure from a collection of tensors.
///
/// For each Diag tensor component, all its indices are unified (they share the same
/// diagonal dimension). This creates hyperedges when multiple Diag tensors
/// share indices.
pub fn build_diag_union(tensors: &[&TensorDynLen]) -> AxisUnionFind {
    let mut uf = AxisUnionFind::new();

    for tensor in tensors {
        for idx in tensor.indices() {
            uf.make_set(*idx.id());
        }

        if tensor.is_diag() && tensor.indices().len() >= 2 {
            let first_id = *tensor.indices()[0].id();
            for idx in tensor.indices().iter().skip(1) {
                uf.union(first_id, *idx.id());
            }
        }
    }

    uf
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

    let profile_signature = contract_profile_enabled().then(|| ContractSignature {
        operands: tensors
            .iter()
            .enumerate()
            .map(|(tensor_idx, tensor)| ContractOperandSignature {
                dims: tensor.dims().to_vec(),
                ids: ixs[tensor_idx].clone(),
                is_diag: tensor.is_diag(),
            })
            .collect(),
        output_ids: output.clone(),
        output_dims: output.iter().map(|id| sizes[id]).collect(),
    });
    let profile_started = contract_profile_enabled().then(Instant::now);

    let native_operands: Vec<_> = tensors
        .iter()
        .enumerate()
        .map(|(tensor_idx, tensor)| (tensor.as_native(), ixs[tensor_idx].as_slice()))
        .collect();

    let result_native = einsum_native_tensors(&native_operands, &output)?;
    if let (Some(signature), Some(started)) = (profile_signature, profile_started) {
        record_contract_profile(signature, started.elapsed());
    }
    let final_indices = if output.is_empty() {
        vec![]
    } else {
        output
            .iter()
            .map(|&internal_id| {
                let (tensor_idx, pos) = internal_id_to_original[&internal_id];
                tensors[tensor_idx].indices[pos].clone()
            })
            .collect()
    };
    TensorDynLen::from_native(final_indices, result_native)
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
mod tests;
