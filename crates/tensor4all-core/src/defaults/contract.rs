//! Multi-tensor contraction with optimal contraction order.
//!
//! This module provides functions to contract multiple tensors efficiently
//! using einsum optimization via the tensorbackend
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
//! Diagonal tensors are materialized as dense native operands for contraction,
//! so numeric einsum labels must keep uncontracted logical axes distinct.
//! Diagonal/structured equality metadata is propagated separately onto the
//! result when the contraction leaves equal axes behind.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::env;
use std::time::{Duration, Instant};

use anyhow::Result;
use petgraph::algo::connected_components;
use petgraph::prelude::*;
use tenferro::eager_einsum::eager_einsum_ad;
use tensor4all_tensorbackend::{einsum_native_tensors, einsum_native_tensors_owned};

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
        entries.sort_by_key(|(_, entry)| Reverse(entry.total_time));

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

/// Options for multi-tensor contraction.
///
/// Use this to choose which tensor pairs may contract and which shared indices
/// should be retained in the output instead of summed over.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{AllowedPairs, ContractionOptions, DynIndex};
///
/// let batch = DynIndex::new_dyn(2);
/// let retain = [batch.clone()];
/// let options = ContractionOptions::new(AllowedPairs::All).with_retain_indices(&retain);
///
/// assert!(matches!(options.allowed, AllowedPairs::All));
/// assert_eq!(options.retain_indices, &[batch]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ContractionOptions<'a> {
    /// Contractability policy for tensor pairs.
    pub allowed: AllowedPairs<'a>,
    /// Indices that should remain in the result even if they appear more than once.
    pub retain_indices: &'a [DynIndex],
}

impl<'a> ContractionOptions<'a> {
    /// Create contraction options with no retained indices.
    pub fn new(allowed: AllowedPairs<'a>) -> Self {
        Self {
            allowed,
            retain_indices: &[],
        }
    }

    /// Set the indices that should be retained in the output.
    pub fn with_retain_indices(mut self, retain_indices: &'a [DynIndex]) -> Self {
        self.retain_indices = retain_indices;
        self
    }
}

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
    contract_multi_with_options(tensors, ContractionOptions::new(allowed))
}

/// Contract multiple tensors into a single tensor with additional options.
///
/// This behaves like [`contract_multi`] but also allows selected shared indices
/// to be retained in the output.
///
/// # Arguments
/// * `tensors` - Slice of tensors to contract
/// * `options` - Pair-selection policy and retained indices
///
/// # Returns
/// The contracted tensor, possibly with retained shared indices in the result.
///
/// # Errors
/// Returns an error if:
/// - no tensors are provided
/// - `AllowedPairs::Specified` contains a pair with no contractable indices
/// - a retained index does not appear in the inputs
/// - a shared internal label has inconsistent dimensions
///
/// # Examples
///
/// ```
/// use tensor4all_core::{contract_multi_with_options, AllowedPairs, ContractionOptions, DynIndex, TensorDynLen};
///
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let k = DynIndex::new_dyn(4);
///
/// let a = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64; 6]).unwrap();
/// let b = TensorDynLen::from_dense(vec![j.clone(), k.clone()], vec![1.0_f64; 12]).unwrap();
/// let retain_indices = [j.clone()];
/// let options = ContractionOptions::new(AllowedPairs::All).with_retain_indices(&retain_indices);
/// let c = contract_multi_with_options(&[&a, &b], options).unwrap();
/// assert_eq!(c.dims(), vec![2, 3, 4]);
/// ```
pub fn contract_multi_with_options(
    tensors: &[&TensorDynLen],
    options: ContractionOptions<'_>,
) -> Result<TensorDynLen> {
    match tensors.len() {
        0 => Err(anyhow::anyhow!("No tensors to contract")),
        _ => {
            validate_retained_indices_exist(tensors, options.retain_indices)?;
            if tensors.len() == 1 {
                return Ok((*tensors[0]).clone());
            }

            // Validate AllowedPairs::Specified pairs have contractable indices
            if let AllowedPairs::Specified(pairs) = options.allowed {
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
            let components = find_tensor_connected_components_with_retained(
                tensors,
                options.allowed,
                options.retain_indices,
            );

            if components.len() == 1 {
                // All tensors connected - use optimized contraction (skip connectivity check)
                contract_multi_impl(tensors, options)
            } else {
                // Multiple components - contract each and combine with outer product
                let mut results: Vec<TensorDynLen> = Vec::new();
                for component in &components {
                    let component_tensors: Vec<&TensorDynLen> =
                        component.iter().map(|&i| tensors[i]).collect();
                    let component_retain_indices =
                        retained_indices_for_component(tensors, component, options.retain_indices);

                    // Remap AllowedPairs for the component (connectivity already verified)
                    let remapped_allowed = remap_allowed_pairs(options.allowed, component);
                    let component_options = ContractionOptions {
                        allowed: remapped_allowed.as_ref(),
                        retain_indices: &component_retain_indices,
                    };
                    let contracted = contract_multi_impl(&component_tensors, component_options)?;
                    results.push(contracted);
                }

                // Combine with outer product
                let mut results_iter = results.into_iter();
                let Some(mut result) = results_iter.next() else {
                    return Err(anyhow::anyhow!("No contracted components produced"));
                };
                for other in results_iter {
                    result = result.outer_product(&other)?;
                }
                Ok(result)
            }
        }
    }
}

/// Contract multiple owned tensors into a single tensor.
///
/// This is the consuming counterpart to [`contract_multi_with_options`]. It
/// preserves the same contraction semantics while allowing eligible non-AD
/// dense inputs to use tenferro's owned eager einsum executor. When any input
/// tracks gradients, or when compact structured metadata needs the borrowed
/// path, this function falls back to the shared borrowed execution so semantics
/// and reverse-mode AD remain intact.
///
/// # Arguments
/// * `tensors` - Owned tensors to contract.
/// * `options` - Pair-selection policy and retained indices.
///
/// # Returns
/// The contracted tensor, with retained shared indices preserved in the output.
///
/// # Errors
/// Returns an error for the same conditions as
/// [`contract_multi_with_options`], including empty input, invalid retained
/// indices, and incompatible contraction pairs.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{contract_multi_owned, contract_multi_with_options, AllowedPairs, ContractionOptions, DynIndex, TensorDynLen};
///
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let k = DynIndex::new_dyn(4);
/// let a = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64; 6]).unwrap();
/// let b = TensorDynLen::from_dense(vec![j.clone(), k.clone()], vec![1.0_f64; 12]).unwrap();
/// let options = ContractionOptions::new(AllowedPairs::All);
///
/// let owned = contract_multi_owned(vec![a.clone(), b.clone()], options).unwrap();
/// let borrowed = contract_multi_with_options(&[&a, &b], options).unwrap();
/// assert_eq!(owned.indices(), borrowed.indices());
/// assert_eq!(owned.to_vec::<f64>().unwrap(), borrowed.to_vec::<f64>().unwrap());
/// ```
pub fn contract_multi_owned(
    tensors: Vec<TensorDynLen>,
    options: ContractionOptions<'_>,
) -> Result<TensorDynLen> {
    match tensors.len() {
        0 => Err(anyhow::anyhow!("No tensors to contract")),
        _ => {
            let tensor_refs = tensors.iter().collect::<Vec<_>>();
            validate_retained_indices_exist(&tensor_refs, options.retain_indices)?;

            if tensors.len() == 1 {
                drop(tensor_refs);
                let Some(tensor) = tensors.into_iter().next() else {
                    return Err(anyhow::anyhow!("No tensors to contract"));
                };
                return Ok(tensor);
            }

            if let AllowedPairs::Specified(pairs) = options.allowed {
                for &(i, j) in pairs {
                    if !has_contractable_indices(tensor_refs[i], tensor_refs[j]) {
                        return Err(anyhow::anyhow!(
                            "Specified pair ({}, {}) has no contractable indices",
                            i,
                            j
                        ));
                    }
                }
            }

            let requires_borrowed_path = tensor_refs.iter().any(|tensor| tensor.tracks_grad())
                || tensor_refs
                    .iter()
                    .any(|tensor| !has_dense_axis_classes(tensor));
            if requires_borrowed_path {
                return contract_multi_with_options(&tensor_refs, options);
            }

            let components = find_tensor_connected_components_with_retained(
                &tensor_refs,
                options.allowed,
                options.retain_indices,
            );
            if components.len() > 1 {
                return contract_multi_with_options(&tensor_refs, options);
            }

            let mut diag_uf = AxisUnionFind::new();
            let plan = build_contraction_plan(&tensor_refs, options, &mut diag_uf)?;
            drop(tensor_refs);
            let native_operands = tensors
                .into_iter()
                .enumerate()
                .map(|(tensor_idx, tensor)| {
                    (
                        tensor.as_native().clone(),
                        plan.input_ids[tensor_idx].clone(),
                    )
                })
                .collect::<Vec<_>>();
            let result_native = einsum_native_tensors_owned(native_operands, &plan.output_ids)?;
            TensorDynLen::from_native_with_axis_classes(
                plan.result_indices,
                result_native,
                plan.result_axis_classes,
            )
        }
    }
}

fn has_dense_axis_classes(tensor: &TensorDynLen) -> bool {
    let storage = tensor.storage();
    storage
        .axis_classes()
        .iter()
        .copied()
        .eq(0..tensor.indices().len())
}

/// Contract multiple tensors that form a connected graph.
///
/// Uses einsum optimization via tensorbackend.
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
/// - N>=2: Optimized order via the tensorbackend einsum path
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
    contract_connected_with_options(tensors, ContractionOptions::new(allowed))
}

/// Contract a connected tensor network with additional options.
///
/// This behaves like [`contract_connected`] but also allows selected shared
/// indices to be retained in the output.
///
/// # Arguments
/// * `tensors` - Slice of tensors to contract
/// * `options` - Pair-selection policy and retained indices
///
/// # Returns
/// The contracted tensor.
///
/// # Errors
/// Returns an error if the tensors are disconnected, no tensors are provided,
/// or retained indices are invalid.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{
///     contract_connected_with_options, AllowedPairs, ContractionOptions, DynIndex, TensorDynLen,
/// };
///
/// let batch = DynIndex::new_dyn(2);
/// let i = DynIndex::new_dyn(2);
/// let k = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(2);
///
/// let a = TensorDynLen::from_dense(
///     vec![batch.clone(), i.clone(), k.clone()],
///     vec![1.0_f64; 12],
/// )
/// .unwrap();
/// let b = TensorDynLen::from_dense(
///     vec![batch.clone(), k, j.clone()],
///     vec![1.0_f64; 12],
/// )
/// .unwrap();
/// let retain = [batch.clone()];
/// let options = ContractionOptions::new(AllowedPairs::All).with_retain_indices(&retain);
///
/// let c = contract_connected_with_options(&[&a, &b], options).unwrap();
/// assert_eq!(c.indices(), &[batch, i, j]);
/// assert_eq!(c.to_vec::<f64>().unwrap(), vec![3.0; 8]);
/// ```
pub fn contract_connected_with_options(
    tensors: &[&TensorDynLen],
    options: ContractionOptions<'_>,
) -> Result<TensorDynLen> {
    match tensors.len() {
        0 => Err(anyhow::anyhow!("No tensors to contract")),
        _ => {
            validate_retained_indices_exist(tensors, options.retain_indices)?;
            if tensors.len() == 1 {
                return Ok((*tensors[0]).clone());
            }

            // Check connectivity first
            let components = find_tensor_connected_components_with_retained(
                tensors,
                options.allowed,
                options.retain_indices,
            );
            if components.len() > 1 {
                return Err(anyhow::anyhow!(
                    "Disconnected tensor network: {} components found",
                    components.len()
                ));
            }
            // Connectivity verified - skip check in impl
            contract_multi_impl(tensors, options)
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
            if let Some(rank) = self.rank.get_mut(&root_a) {
                *rank += 1;
            }
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
// Axis helper builders
// ============================================================================

/// Build a union-find structure from a collection of tensors.
///
/// This helper is kept for callers that need to group diagonal axes by index ID.
/// Numeric contraction currently keeps dense logical axes distinct and propagates
/// diagonal result metadata separately.
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
/// Diagonal tensors are passed as dense native operands for numeric contraction.
/// Their compact equality metadata is propagated separately onto the result.
///
/// This implementation preserves storage type: if all inputs are F64, the result
/// is F64; if any input is C64, the result is C64.
fn contract_multi_impl(
    tensors: &[&TensorDynLen],
    options: ContractionOptions<'_>,
) -> Result<TensorDynLen> {
    // 1. Build union-find over exact matching index IDs. Diagonal equality is
    // encoded in the dense native values and should not collapse uncontracted
    // logical axes in the numeric einsum.
    let mut diag_uf = AxisUnionFind::new();

    // 2. Build the contraction plan from internal labels.
    let plan = build_contraction_plan(tensors, options, &mut diag_uf)?;

    // Note: Connectivity check is done by caller (contract_multi or contract_connected)
    // via find_tensor_connected_components before calling this function

    // 3. Build sizes from unique internal IDs.
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        let dims = tensor.dims();
        for (pos, &dim) in dims.iter().enumerate() {
            let internal_id = plan.input_ids[tensor_idx][pos];
            match sizes.entry(internal_id) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(dim);
                }
                std::collections::hash_map::Entry::Occupied(entry) => {
                    if *entry.get() != dim {
                        return Err(anyhow::anyhow!(
                            "Internal label dimension mismatch: label {} has dimensions {} and {}",
                            internal_id,
                            entry.get(),
                            dim
                        ));
                    }
                }
            }
        }
    }

    let profile_signature = contract_profile_enabled().then(|| ContractSignature {
        operands: tensors
            .iter()
            .enumerate()
            .map(|(tensor_idx, tensor)| ContractOperandSignature {
                dims: tensor.dims().to_vec(),
                ids: plan.input_ids[tensor_idx].clone(),
                is_diag: tensor.is_diag(),
            })
            .collect(),
        output_ids: plan.output_ids.clone(),
        output_dims: plan.output_ids.iter().map(|id| sizes[id]).collect(),
    });
    let profile_started = contract_profile_enabled().then(Instant::now);

    let result = execute_contraction_plan(tensors, &plan, !options.retain_indices.is_empty())?;
    if let (Some(signature), Some(started)) = (profile_signature, profile_started) {
        record_contract_profile(signature, started.elapsed());
    }
    Ok(result)
}

fn execute_contraction_plan(
    tensors: &[&TensorDynLen],
    plan: &ContractionPlan,
    has_retained_indices: bool,
) -> Result<TensorDynLen> {
    let any_grad = tensors.iter().any(|tensor| tensor.tracks_grad());
    let first_dtype = tensors[0].as_native().dtype();
    let same_dtype = tensors
        .iter()
        .all(|tensor| tensor.as_native().dtype() == first_dtype);
    let has_non_dense_axis_classes = tensors.iter().any(|tensor| {
        tensor
            .storage()
            .axis_classes()
            .iter()
            .copied()
            .enumerate()
            .any(|(axis, class)| axis != class)
    });

    if any_grad && same_dtype && has_non_dense_axis_classes {
        if has_retained_indices {
            return Err(anyhow::anyhow!(
                "Retained AD contraction with structured storage is not yet supported"
            ));
        }

        // Structured payload AD still relies on the existing pairwise structured
        // path until structured N-ary planning is implemented.
        let mut iter = tensors.iter();
        let Some(first) = iter.next() else {
            return Err(anyhow::anyhow!("No tensors to contract"));
        };
        let mut result = (*first).clone();
        for tensor in iter {
            result = result.contract_pairwise_default(tensor);
        }
        return Ok(result);
    }

    if any_grad && same_dtype {
        let operands = tensors
            .iter()
            .map(|tensor| tensor.as_inner())
            .collect::<Vec<_>>();
        let subscripts = build_einsum_subscripts_from_usize_ids(&plan.input_ids, &plan.output_ids)?;
        let result = eager_einsum_ad(&operands, &subscripts)?;
        return TensorDynLen::from_inner_with_axis_classes(
            plan.result_indices.clone(),
            result,
            plan.result_axis_classes.clone(),
        );
    }

    let native_operands: Vec<_> = tensors
        .iter()
        .enumerate()
        .map(|(tensor_idx, tensor)| (tensor.as_native(), plan.input_ids[tensor_idx].as_slice()))
        .collect();
    let result_native = einsum_native_tensors(&native_operands, &plan.output_ids)?;
    TensorDynLen::from_native_with_axis_classes(
        plan.result_indices.clone(),
        result_native,
        plan.result_axis_classes.clone(),
    )
}

fn build_einsum_subscripts_from_usize_ids(
    input_ids: &[Vec<usize>],
    output_ids: &[usize],
) -> Result<String> {
    fn ids_to_subscript(ids: &[usize]) -> Result<String> {
        const LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let mut out = String::with_capacity(ids.len());
        for &id in ids {
            let letter = LETTERS.get(id).ok_or_else(|| {
                anyhow::anyhow!("einsum label {id} exceeds supported label range")
            })?;
            out.push(char::from(*letter));
        }
        Ok(out)
    }

    let inputs = input_ids
        .iter()
        .map(|ids| ids_to_subscript(ids))
        .collect::<Result<Vec<_>>>()?;
    Ok(format!(
        "{}->{}",
        inputs.join(","),
        ids_to_subscript(output_ids)?
    ))
}

/// A contraction plan with internal labels and result ordering.
#[derive(Debug, Clone)]
struct ContractionPlan {
    input_ids: Vec<Vec<usize>>,
    output_ids: Vec<usize>,
    result_indices: Vec<DynIndex>,
    result_axis_classes: Vec<usize>,
}

fn build_contraction_plan(
    tensors: &[&TensorDynLen],
    options: ContractionOptions<'_>,
    diag_uf: &mut AxisUnionFind,
) -> Result<ContractionPlan> {
    let retained_ids: HashSet<DynId> = options.retain_indices.iter().map(|idx| *idx.id()).collect();
    let (input_ids, internal_id_to_original) =
        build_internal_ids(tensors, options.allowed, diag_uf, &retained_ids)?;

    let mut counts: HashMap<usize, usize> = HashMap::new();
    for ids in &input_ids {
        for &internal_id in ids {
            *counts.entry(internal_id).or_insert(0) += 1;
        }
    }
    let mut output_ids = Vec::new();
    let mut seen_output = HashSet::new();
    let mut found_retained = HashSet::new();

    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        for (axis, idx) in tensor.indices.iter().enumerate() {
            let internal_id = input_ids[tensor_idx][axis];
            let should_output = counts[&internal_id] == 1 || retained_ids.contains(idx.id());
            if should_output && seen_output.insert(internal_id) {
                output_ids.push(internal_id);
            }
            if retained_ids.contains(idx.id()) {
                found_retained.insert(*idx.id());
            }
        }
    }

    for retained in retained_ids {
        if !found_retained.contains(&retained) {
            return Err(anyhow::anyhow!(
                "Retained index {:?} does not appear in the input tensors",
                retained
            ));
        }
    }

    let result_indices: Vec<DynIndex> = output_ids
        .iter()
        .map(|&internal_id| {
            let (tensor_idx, pos) = internal_id_to_original[&internal_id];
            tensors[tensor_idx].indices[pos].clone()
        })
        .collect();
    validate_unique_output_indices(&result_indices)?;
    let result_axis_classes =
        output_axis_classes(tensors, &input_ids, &output_ids, &internal_id_to_original);

    Ok(ContractionPlan {
        input_ids,
        output_ids,
        result_indices,
        result_axis_classes,
    })
}

fn validate_retained_indices_exist(
    tensors: &[&TensorDynLen],
    retain_indices: &[DynIndex],
) -> Result<()> {
    for retain in retain_indices {
        let found = tensors
            .iter()
            .any(|tensor| tensor.indices().iter().any(|idx| idx.id() == retain.id()));
        if !found {
            return Err(anyhow::anyhow!(
                "Retained index {:?} does not appear in the input tensors",
                retain
            ));
        }
    }
    Ok(())
}

fn retained_indices_for_component(
    tensors: &[&TensorDynLen],
    component: &[usize],
    retain_indices: &[DynIndex],
) -> Vec<DynIndex> {
    let mut seen = HashSet::new();
    let mut retained = Vec::new();
    for retain in retain_indices {
        if seen.insert(*retain.id())
            && component.iter().any(|&tensor_idx| {
                tensors[tensor_idx]
                    .indices()
                    .iter()
                    .any(|idx| idx.id() == retain.id())
            })
        {
            retained.push(retain.clone());
        }
    }
    retained
}

fn validate_unique_output_indices(indices: &[DynIndex]) -> Result<()> {
    let mut seen = HashSet::new();
    for idx in indices {
        if !seen.insert(*idx.id()) {
            return Err(anyhow::anyhow!(
                "Contraction result would contain duplicate output indices"
            ));
        }
    }
    Ok(())
}

fn output_axis_classes(
    tensors: &[&TensorDynLen],
    ixs: &[Vec<usize>],
    output: &[usize],
    internal_id_to_original: &HashMap<usize, (usize, usize)>,
) -> Vec<usize> {
    fn find(parent: &mut [usize], value: usize) -> usize {
        if parent[value] != value {
            parent[value] = find(parent, parent[value]);
        }
        parent[value]
    }

    fn union(parent: &mut [usize], lhs: usize, rhs: usize) {
        let lhs_root = find(parent, lhs);
        let rhs_root = find(parent, rhs);
        if lhs_root != rhs_root {
            parent[rhs_root] = lhs_root;
        }
    }

    let mut class_offsets = Vec::with_capacity(tensors.len());
    let mut next_node = 0usize;
    for tensor in tensors {
        class_offsets.push(next_node);
        let payload_rank = tensor
            .storage()
            .axis_classes()
            .iter()
            .copied()
            .max()
            .map(|value| value + 1)
            .unwrap_or(0);
        next_node += payload_rank;
    }
    let mut parent: Vec<usize> = (0..next_node).collect();
    let mut axes_by_internal_id: HashMap<usize, Vec<usize>> = HashMap::new();

    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        for (axis, &internal_id) in ixs[tensor_idx].iter().enumerate() {
            let class_id = tensor.storage().axis_classes()[axis];
            let node = class_offsets[tensor_idx] + class_id;
            axes_by_internal_id
                .entry(internal_id)
                .or_default()
                .push(node);
        }
    }

    for nodes in axes_by_internal_id.values() {
        if let Some((&first, rest)) = nodes.split_first() {
            for &node in rest {
                union(&mut parent, first, node);
            }
        }
    }

    let mut root_to_class = HashMap::new();
    let mut next_class = 0usize;
    output
        .iter()
        .map(|internal_id| {
            let (tensor_idx, axis) = internal_id_to_original[internal_id];
            let class_id = tensors[tensor_idx].storage().axis_classes()[axis];
            let node = class_offsets[tensor_idx] + class_id;
            let root = find(&mut parent, node);
            *root_to_class.entry(root).or_insert_with(|| {
                let class = next_class;
                next_class += 1;
                class
            })
        })
        .collect()
}

/// Build internal IDs for numeric contraction.
///
/// Uses the union-find to merge IDs that have already been proven equivalent by
/// the caller. Diagonal logical-axis metadata is intentionally handled outside
/// this numeric labeling step.
///
/// Returns: (ixs, internal_id_to_original)
#[allow(clippy::type_complexity)]
fn build_internal_ids(
    tensors: &[&TensorDynLen],
    allowed: AllowedPairs<'_>,
    diag_uf: &mut AxisUnionFind,
    retained_ids: &HashSet<DynId>,
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
                let internal_id = if retained_ids.contains(&remapped_id) {
                    if let Some(&id) = dynid_to_internal.get(&remapped_id) {
                        id
                    } else {
                        let id = next_id;
                        next_id += 1;
                        dynid_to_internal.insert(remapped_id, id);
                        internal_id_to_original.insert(id, key);
                        id
                    }
                } else {
                    let id = next_id;
                    next_id += 1;
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
#[allow(dead_code)]
fn find_tensor_connected_components(
    tensors: &[&TensorDynLen],
    allowed: AllowedPairs<'_>,
) -> Vec<Vec<usize>> {
    find_tensor_connected_components_with_retained(tensors, allowed, &[])
}

fn find_tensor_connected_components_with_retained(
    tensors: &[&TensorDynLen],
    allowed: AllowedPairs<'_>,
    retain_indices: &[DynIndex],
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

    if !retain_indices.is_empty() {
        for i in 0..n {
            for j in (i + 1)..n {
                if shares_retained_index(tensors[i], tensors[j], retain_indices) {
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

fn shares_retained_index(a: &TensorDynLen, b: &TensorDynLen, retain_indices: &[DynIndex]) -> bool {
    retain_indices.iter().any(|retain| {
        a.indices().iter().any(|idx_a| idx_a.id() == retain.id())
            && b.indices().iter().any(|idx_b| idx_b.id() == retain.id())
    })
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
