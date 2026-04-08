//! Apply LinearOperator to TreeTN state.
//!
//! This module provides the `apply_linear_operator` function for computing `A|x⟩`
//! where A is a LinearOperator (MPO with index mappings) and |x⟩ is a TreeTN state.
//!
//! # Algorithm
//!
//! The application works as follows:
//! 1. **Partial Site Handling**: If the operator only covers some nodes of the state,
//!    use `compose_exclusive_linear_operators` to fill gaps with identity operators.
//! 2. **Index Transformation**: Replace state's site indices with operator's input indices.
//! 3. **Contraction**: Contract the transformed state with the operator using
//!    `contract_zipup`, `contract_fit`, or `contract_naive` depending on options.
//! 4. **Output Transformation**: Replace operator's output indices with true output indices.
//!
//! # Example
//!
//! ```ignore
//! let result = apply_linear_operator(&operator, &state, ApplyOptions::default())?;
//! ```

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use anyhow::{Context, Result};

use tensor4all_core::{IndexLike, TensorIndex, TensorLike};

use super::index_mapping::IndexMapping;
use super::linear_operator::LinearOperator;
use super::Operator;
use crate::operator::compose::{
    compose_exclusive_linear_operators, compose_exclusive_linear_operators_unchecked,
};
use crate::treetn::contraction::{contract, ContractionMethod, ContractionOptions};
use crate::treetn::TreeTN;

/// Options for apply_linear_operator.
#[derive(Debug, Clone)]
pub struct ApplyOptions {
    /// Contraction method to use.
    pub method: ContractionMethod,
    /// Maximum bond dimension for truncation.
    pub max_rank: Option<usize>,
    /// Relative tolerance for truncation.
    pub rtol: Option<f64>,
    /// Number of full sweeps for Fit method.
    ///
    /// A full sweep visits each edge twice (forward and backward) using an Euler tour.
    pub nfullsweeps: usize,
    /// Convergence tolerance for Fit method.
    pub convergence_tol: Option<f64>,
}

impl Default for ApplyOptions {
    fn default() -> Self {
        Self {
            method: ContractionMethod::Zipup,
            max_rank: None,
            rtol: None,
            nfullsweeps: 1,
            convergence_tol: None,
        }
    }
}

impl ApplyOptions {
    /// Create options with ZipUp method (default).
    pub fn zipup() -> Self {
        Self::default()
    }

    /// Create options with Fit method.
    pub fn fit() -> Self {
        Self {
            method: ContractionMethod::Fit,
            ..Default::default()
        }
    }

    /// Create options with Naive method.
    pub fn naive() -> Self {
        Self {
            method: ContractionMethod::Naive,
            ..Default::default()
        }
    }

    /// Set maximum bond dimension.
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = Some(max_rank);
        self
    }

    /// Set relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = Some(rtol);
        self
    }

    /// Set number of full sweeps for Fit method.
    pub fn with_nfullsweeps(mut self, nfullsweeps: usize) -> Self {
        self.nfullsweeps = nfullsweeps;
        self
    }
}

/// Apply a LinearOperator to a TreeTN state: compute `A|x⟩`.
///
/// This function handles:
/// - Partial operators (fills gaps with identity via compose_exclusive_linear_operators)
/// - Index transformations (input/output mappings)
/// - Multiple contraction algorithms (ZipUp, Fit, Naive)
///
/// # Arguments
///
/// * `operator` - The LinearOperator to apply
/// * `state` - The input state |x⟩
/// * `options` - Options controlling the contraction algorithm
///
/// # Returns
///
/// The result `A|x⟩` as a TreeTN, or an error if application fails.
///
/// # Example
///
/// ```ignore
/// use tensor4all_treetn::operator::{apply_linear_operator, ApplyOptions};
///
/// // Apply with default options (ZipUp)
/// let result = apply_linear_operator(&operator, &state, ApplyOptions::default())?;
///
/// // Apply with truncation
/// let result = apply_linear_operator(
///     &operator,
///     &state,
///     ApplyOptions::zipup().with_max_rank(50).with_rtol(1e-10),
/// )?;
/// ```
pub fn apply_linear_operator<T, V>(
    operator: &LinearOperator<T, V>,
    state: &TreeTN<T, V>,
    options: ApplyOptions,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    // 1. Check if operator covers all state nodes
    let state_nodes: HashSet<V> = state.node_names().into_iter().collect();
    let op_nodes: HashSet<V> = operator.node_names();

    let full_operator = if op_nodes == state_nodes {
        // Operator covers all nodes - use directly
        operator.clone()
    } else if op_nodes.is_subset(&state_nodes) {
        // Partial operator - need to compose with identity on gaps
        extend_operator_to_full_space(operator, state)?
    } else {
        return Err(anyhow::anyhow!(
            "Operator nodes {:?} are not a subset of state nodes {:?}",
            op_nodes,
            state_nodes
        ));
    };

    // 2. Transform state's site indices to operator's input indices
    let transformed_state = transform_state_to_input(&full_operator, state)?;

    // 3. Contract state with operator MPO
    // Choose a center node (use first node in sorted order for determinism)
    let mut node_names: Vec<_> = state.node_names();
    node_names.sort();
    let center = node_names
        .first()
        .ok_or_else(|| anyhow::anyhow!("Empty state"))?;

    let contraction_options = ContractionOptions {
        method: options.method,
        max_rank: options.max_rank,
        rtol: options.rtol,
        nfullsweeps: options.nfullsweeps,
        convergence_tol: options.convergence_tol,
        ..Default::default()
    };

    let contracted = contract(
        &transformed_state,
        full_operator.mpo(),
        center,
        contraction_options,
    )
    .context("Failed to contract state with operator")?;

    // 4. Transform operator's output indices to true output indices
    let result = transform_output_to_true(&full_operator, contracted)?;

    Ok(result)
}

/// Extend a partial operator to cover the full state space.
///
/// Uses the operator support's Steiner tree to detect disconnected regions and
/// fills all missing nodes with identity operators.
/// For gap nodes, creates proper index mappings where:
/// - True indices = state's actual site indices
/// - Internal indices = new simulated indices for the MPO tensor
fn extend_operator_to_full_space<T, V>(
    operator: &LinearOperator<T, V>,
    state: &TreeTN<T, V>,
) -> Result<LinearOperator<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let state_network = state.site_index_network();
    let op_nodes: HashSet<V> = operator.node_names();
    let state_nodes: HashSet<V> = state.node_names().into_iter().collect();
    let mut op_node_indices: HashSet<petgraph::stable_graph::NodeIndex> = HashSet::new();
    for name in &op_nodes {
        let node_index = state_network.node_index(name).ok_or_else(|| {
            anyhow::anyhow!("Operator node {:?} is missing from the state network", name)
        })?;
        op_node_indices.insert(node_index);
    }

    let steiner_tree_nodes = state_network.steiner_tree_nodes(&op_node_indices);
    let steiner_gap_nodes: HashSet<_> = steiner_tree_nodes
        .difference(&op_node_indices)
        .copied()
        .collect();
    let gap_nodes: Vec<V> = state_nodes.difference(&op_nodes).cloned().collect();

    // Build gap site indices: for each gap node, create internal indices for the identity tensor.
    // The (input_internal, output_internal) pairs are used to build the delta tensor.
    #[allow(clippy::type_complexity)]
    let mut gap_site_indices: HashMap<V, Vec<(T::Index, T::Index)>> = HashMap::new();

    // Also track true<->internal mappings for gap nodes
    #[allow(clippy::type_complexity)]
    let mut gap_input_mappings: HashMap<V, IndexMapping<T::Index>> = HashMap::new();
    #[allow(clippy::type_complexity)]
    let mut gap_output_mappings: HashMap<V, IndexMapping<T::Index>> = HashMap::new();

    for gap_name in &gap_nodes {
        let site_space = state
            .site_space(gap_name)
            .ok_or_else(|| anyhow::anyhow!("Gap node {:?} has no site space", gap_name))?;

        // For identity at gap nodes:
        // - True indices = state's site indices (what apply_linear_operator maps from/to)
        // - Internal indices = new simulated indices for the MPO tensor
        let mut pairs: Vec<(T::Index, T::Index)> = Vec::new();

        for (i, true_idx) in site_space.iter().enumerate() {
            let input_internal = true_idx.sim();
            let output_internal = true_idx.sim();
            pairs.push((input_internal.clone(), output_internal.clone()));

            // Store mapping for the first site index of each gap node
            if i == 0 {
                gap_input_mappings.insert(
                    gap_name.clone(),
                    IndexMapping {
                        true_index: true_idx.clone(),
                        internal_index: input_internal,
                    },
                );
                gap_output_mappings.insert(
                    gap_name.clone(),
                    IndexMapping {
                        true_index: true_idx.clone(),
                        internal_index: output_internal,
                    },
                );
            }
        }

        gap_site_indices.insert(gap_name.clone(), pairs);
    }

    let mut composed = if operator.mpo.edge_count() == 0 {
        compose_exclusive_linear_operators_unchecked(state_network, &[operator], &gap_site_indices)
            .context("Failed to compose operator with identity gaps")?
    } else if steiner_gap_nodes.is_empty() {
        compose_exclusive_linear_operators(state_network, &[operator], &gap_site_indices)
            .context("Failed to compose operator with identity gaps")?
    } else {
        compose_operator_along_state_paths(
            operator,
            state_network,
            &gap_site_indices,
            gap_input_mappings.clone(),
            gap_output_mappings.clone(),
        )
        .context("Failed to compose operator along state paths")?
    };

    // Override the mappings for gap nodes to use the correct true indices
    // (compose_exclusive_linear_operators uses the internal indices as true indices for gaps)
    for (gap_name, mapping) in gap_input_mappings {
        composed.input_mapping.insert(gap_name, mapping);
    }
    for (gap_name, mapping) in gap_output_mappings {
        composed.output_mapping.insert(gap_name, mapping);
    }

    Ok(composed)
}

#[allow(clippy::type_complexity)]
fn compose_operator_along_state_paths<T, V>(
    operator: &LinearOperator<T, V>,
    state_network: &crate::site_index_network::SiteIndexNetwork<V, T::Index>,
    gap_site_indices: &HashMap<V, Vec<(T::Index, T::Index)>>,
    input_mappings: HashMap<V, IndexMapping<T::Index>>,
    output_mappings: HashMap<V, IndexMapping<T::Index>>,
) -> Result<LinearOperator<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let op_nodes: HashSet<V> = operator.node_names();
    let mut tensors_by_node: HashMap<V, T> = HashMap::new();

    let mut state_node_names: Vec<V> = state_network.node_names().into_iter().cloned().collect();
    state_node_names.sort();

    for node in &state_node_names {
        if op_nodes.contains(node) {
            let node_idx = operator.mpo.node_index(node).ok_or_else(|| {
                anyhow::anyhow!(
                    "compose_operator_along_state_paths: missing node {:?}",
                    node
                )
            })?;
            let tensor = operator.mpo.tensor(node_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "compose_operator_along_state_paths: missing tensor for {:?}",
                    node
                )
            })?;
            tensors_by_node.insert(node.clone(), tensor.clone());
        } else {
            let index_pairs = gap_site_indices.get(node).ok_or_else(|| {
                anyhow::anyhow!(
                    "compose_operator_along_state_paths: missing gap indices for {:?}",
                    node
                )
            })?;
            let input_indices: Vec<T::Index> = index_pairs.iter().map(|(i, _)| i.clone()).collect();
            let output_indices: Vec<T::Index> =
                index_pairs.iter().map(|(_, o)| o.clone()).collect();
            let tensor = if input_indices.is_empty() {
                T::delta(&[], &[]).context(
                    "compose_operator_along_state_paths: failed to build scalar identity",
                )?
            } else {
                T::delta(&input_indices, &output_indices).with_context(|| {
                    format!(
                        "compose_operator_along_state_paths: failed to build identity for gap {:?}",
                        node
                    )
                })?
            };
            tensors_by_node.insert(node.clone(), tensor);
        }
    }

    let mut op_edges: Vec<(V, V)> = operator.mpo.site_index_network().edges().collect();
    op_edges.sort();
    let mut used_state_edges: HashSet<(V, V)> = HashSet::new();

    for (node_a, node_b) in op_edges {
        let idx_a = state_network.node_index(&node_a).ok_or_else(|| {
            anyhow::anyhow!(
                "compose_operator_along_state_paths: missing state node {:?}",
                node_a
            )
        })?;
        let idx_b = state_network.node_index(&node_b).ok_or_else(|| {
            anyhow::anyhow!(
                "compose_operator_along_state_paths: missing state node {:?}",
                node_b
            )
        })?;
        let path = state_network.path_between(idx_a, idx_b).ok_or_else(|| {
            anyhow::anyhow!(
                "compose_operator_along_state_paths: no path between {:?} and {:?}",
                node_a,
                node_b
            )
        })?;
        if path.len() < 2 {
            continue;
        }

        let edge = operator.mpo.edge_between(&node_a, &node_b).ok_or_else(|| {
            anyhow::anyhow!(
                "compose_operator_along_state_paths: missing operator edge between {:?} and {:?}",
                node_a,
                node_b
            )
        })?;
        let bond = operator
            .mpo
            .bond_index(edge)
            .ok_or_else(|| {
                anyhow::anyhow!("compose_operator_along_state_paths: missing bond index")
            })?
            .clone();

        let mut chain_bonds = Vec::with_capacity(path.len() - 1);
        chain_bonds.push(bond.sim());
        for _ in 1..(path.len() - 1) {
            let next = chain_bonds[chain_bonds.len() - 1].sim();
            chain_bonds.push(next);
        }

        let start_name = state_network
            .node_name(path[0])
            .ok_or_else(|| anyhow::anyhow!("compose_operator_along_state_paths: missing start"))?
            .clone();
        let end_name = state_network
            .node_name(path[path.len() - 1])
            .ok_or_else(|| anyhow::anyhow!("compose_operator_along_state_paths: missing end"))?
            .clone();

        {
            let tensor = tensors_by_node.get_mut(&start_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "compose_operator_along_state_paths: missing tensor for {:?}",
                    start_name
                )
            })?;
            *tensor = tensor.replaceind(&bond, &chain_bonds[0]).with_context(|| {
                format!(
                    "compose_operator_along_state_paths: failed to reroute bond at {:?}",
                    start_name
                )
            })?;
        }
        {
            let tensor = tensors_by_node.get_mut(&end_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "compose_operator_along_state_paths: missing tensor for {:?}",
                    end_name
                )
            })?;
            let last_bond = &chain_bonds[chain_bonds.len() - 1];
            *tensor = tensor.replaceind(&bond, last_bond).with_context(|| {
                format!(
                    "compose_operator_along_state_paths: failed to reroute bond at {:?}",
                    end_name
                )
            })?;
        }

        for i in 1..(path.len() - 1) {
            let mid_name = state_network
                .node_name(path[i])
                .ok_or_else(|| anyhow::anyhow!("compose_operator_along_state_paths: missing mid"))?
                .clone();
            let delta = T::delta(
                std::slice::from_ref(&chain_bonds[i - 1]),
                std::slice::from_ref(&chain_bonds[i]),
            )
            .with_context(|| {
                format!(
                    "compose_operator_along_state_paths: failed to build bridge at {:?}",
                    mid_name
                )
            })?;
            let tensor = tensors_by_node.get_mut(&mid_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "compose_operator_along_state_paths: missing tensor for {:?}",
                    mid_name
                )
            })?;
            *tensor = tensor.outer_product(&delta).with_context(|| {
                format!(
                    "compose_operator_along_state_paths: failed to attach bridge at {:?}",
                    mid_name
                )
            })?;
        }

        for window in path.windows(2) {
            let a = state_network
                .node_name(window[0])
                .ok_or_else(|| {
                    anyhow::anyhow!("compose_operator_along_state_paths: missing path node")
                })?
                .clone();
            let b = state_network
                .node_name(window[1])
                .ok_or_else(|| {
                    anyhow::anyhow!("compose_operator_along_state_paths: missing path node")
                })?
                .clone();
            let edge_key = if a <= b { (a, b) } else { (b, a) };
            used_state_edges.insert(edge_key);
        }
    }

    let mut state_edges: Vec<(V, V)> = state_network.edges().collect();
    state_edges.sort();
    for (node_a, node_b) in state_edges {
        let edge_key = if node_a <= node_b {
            (node_a.clone(), node_b.clone())
        } else {
            (node_b.clone(), node_a.clone())
        };
        if used_state_edges.contains(&edge_key) {
            continue;
        }
        let (link_a, link_b) = T::Index::create_dummy_link_pair();
        let ones_a = T::ones(std::slice::from_ref(&link_a)).with_context(|| {
            format!(
                "compose_operator_along_state_paths: failed to create dummy link tensor for {:?}",
                node_a
            )
        })?;
        let ones_b = T::ones(std::slice::from_ref(&link_b)).with_context(|| {
            format!(
                "compose_operator_along_state_paths: failed to create dummy link tensor for {:?}",
                node_b
            )
        })?;

        let tensor_a = tensors_by_node.get_mut(&node_a).ok_or_else(|| {
            anyhow::anyhow!(
                "compose_operator_along_state_paths: missing tensor for {:?}",
                node_a
            )
        })?;
        *tensor_a = tensor_a.outer_product(&ones_a).with_context(|| {
            format!(
                "compose_operator_along_state_paths: failed to attach dummy link at {:?}",
                node_a
            )
        })?;

        let tensor_b = tensors_by_node.get_mut(&node_b).ok_or_else(|| {
            anyhow::anyhow!(
                "compose_operator_along_state_paths: missing tensor for {:?}",
                node_b
            )
        })?;
        *tensor_b = tensor_b.outer_product(&ones_b).with_context(|| {
            format!(
                "compose_operator_along_state_paths: failed to attach dummy link at {:?}",
                node_b
            )
        })?;
    }

    let tensors: Vec<T> = state_node_names
        .iter()
        .map(|node| {
            tensors_by_node.get(node).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "compose_operator_along_state_paths: missing tensor for {:?}",
                    node
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mpo = TreeTN::from_tensors(tensors, state_node_names.clone())
        .context("compose_operator_along_state_paths: failed to create TreeTN")?;

    Ok(LinearOperator::new(mpo, input_mappings, output_mappings))
}

/// Transform state's site indices to operator's input indices.
fn transform_state_to_input<T, V>(
    operator: &LinearOperator<T, V>,
    state: &TreeTN<T, V>,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let mut result = state.clone();

    for (node, mapping) in operator.input_mappings() {
        // Replace true_index with internal_index in the state
        result = result
            .replaceind(&mapping.true_index, &mapping.internal_index)
            .with_context(|| format!("Failed to transform input index at node {:?}", node))?;
    }

    Ok(result)
}

/// Transform operator's output indices to true output indices.
fn transform_output_to_true<T, V>(
    operator: &LinearOperator<T, V>,
    mut result: TreeTN<T, V>,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    for (node, mapping) in operator.output_mappings() {
        // Replace internal_index with true_index in the result
        result = result
            .replaceind(&mapping.internal_index, &mapping.true_index)
            .with_context(|| format!("Failed to transform output index at node {:?}", node))?;
    }

    Ok(result)
}

// ============================================================================
// TensorIndex implementation for LinearOperator
// ============================================================================

impl<T, V> TensorIndex for LinearOperator<T, V>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    type Index = T::Index;

    /// Return all external indices (true input and output indices).
    fn external_indices(&self) -> Vec<Self::Index> {
        let mut result: Vec<Self::Index> = self
            .input_mapping
            .values()
            .map(|m| m.true_index.clone())
            .collect();
        result.extend(self.output_mapping.values().map(|m| m.true_index.clone()));
        result
    }

    fn num_external_indices(&self) -> usize {
        self.input_mapping.len() + self.output_mapping.len()
    }

    /// Replace an external index (true index) in this operator.
    ///
    /// This updates the mapping but does NOT modify the internal MPO tensors.
    fn replaceind(&self, old_index: &Self::Index, new_index: &Self::Index) -> Result<Self> {
        // Validate dimension match
        if old_index.dim() != new_index.dim() {
            return Err(anyhow::anyhow!(
                "Index space mismatch: cannot replace index with dimension {} with index of dimension {}",
                old_index.dim(),
                new_index.dim()
            ));
        }

        let mut result = self.clone();

        // Check input mappings
        for (node, mapping) in &self.input_mapping {
            if mapping.true_index.same_id(old_index) {
                result.input_mapping.insert(
                    node.clone(),
                    IndexMapping {
                        true_index: new_index.clone(),
                        internal_index: mapping.internal_index.clone(),
                    },
                );
                return Ok(result);
            }
        }

        // Check output mappings
        for (node, mapping) in &self.output_mapping {
            if mapping.true_index.same_id(old_index) {
                result.output_mapping.insert(
                    node.clone(),
                    IndexMapping {
                        true_index: new_index.clone(),
                        internal_index: mapping.internal_index.clone(),
                    },
                );
                return Ok(result);
            }
        }

        Err(anyhow::anyhow!(
            "Index {:?} not found in LinearOperator mappings",
            old_index.id()
        ))
    }

    /// Replace multiple external indices.
    fn replaceinds(
        &self,
        old_indices: &[Self::Index],
        new_indices: &[Self::Index],
    ) -> Result<Self> {
        if old_indices.len() != new_indices.len() {
            return Err(anyhow::anyhow!(
                "Length mismatch: {} old indices, {} new indices",
                old_indices.len(),
                new_indices.len()
            ));
        }

        let mut result = self.clone();
        for (old, new) in old_indices.iter().zip(new_indices.iter()) {
            result = result.replaceind(old, new)?;
        }
        Ok(result)
    }
}

// ============================================================================
// Arc-based CoW wrapper for LinearOperator
// ============================================================================

/// LinearOperator with Arc-based Copy-on-Write semantics.
///
/// This wrapper uses `Arc` for the internal MPO to enable cheap cloning
/// and efficient sharing. When mutation is needed, `make_mut` performs
/// a clone only if there are other references.
#[derive(Debug, Clone)]
pub struct ArcLinearOperator<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The MPO with internal index IDs (wrapped in Arc for CoW)
    pub mpo: Arc<TreeTN<T, V>>,
    /// Input index mapping: node -> (true s_in, internal s_in_tmp)
    pub input_mapping: HashMap<V, IndexMapping<T::Index>>,
    /// Output index mapping: node -> (true s_out, internal s_out_tmp)
    pub output_mapping: HashMap<V, IndexMapping<T::Index>>,
}

impl<T, V> ArcLinearOperator<T, V>
where
    T: TensorLike,
    T::Index: IndexLike + Clone,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Create from an existing LinearOperator.
    pub fn from_linear_operator(op: LinearOperator<T, V>) -> Self {
        Self {
            mpo: Arc::new(op.mpo),
            input_mapping: op.input_mapping,
            output_mapping: op.output_mapping,
        }
    }

    /// Create a new ArcLinearOperator.
    pub fn new(
        mpo: TreeTN<T, V>,
        input_mapping: HashMap<V, IndexMapping<T::Index>>,
        output_mapping: HashMap<V, IndexMapping<T::Index>>,
    ) -> Self {
        Self {
            mpo: Arc::new(mpo),
            input_mapping,
            output_mapping,
        }
    }

    /// Get a mutable reference to the MPO, cloning if necessary.
    ///
    /// This implements Copy-on-Write semantics: if this is the only reference,
    /// no copy is made. If there are other references, the MPO is cloned first.
    pub fn mpo_mut(&mut self) -> &mut TreeTN<T, V> {
        Arc::make_mut(&mut self.mpo)
    }

    /// Get an immutable reference to the MPO.
    pub fn mpo(&self) -> &TreeTN<T, V> {
        &self.mpo
    }

    /// Convert back to a LinearOperator (unwraps Arc if possible).
    pub fn into_linear_operator(self) -> LinearOperator<T, V> {
        LinearOperator {
            mpo: Arc::try_unwrap(self.mpo).unwrap_or_else(|arc| (*arc).clone()),
            input_mapping: self.input_mapping,
            output_mapping: self.output_mapping,
        }
    }

    /// Get input mapping for a node.
    pub fn get_input_mapping(&self, node: &V) -> Option<&IndexMapping<T::Index>> {
        self.input_mapping.get(node)
    }

    /// Get output mapping for a node.
    pub fn get_output_mapping(&self, node: &V) -> Option<&IndexMapping<T::Index>> {
        self.output_mapping.get(node)
    }

    /// Get all input mappings.
    pub fn input_mappings(&self) -> &HashMap<V, IndexMapping<T::Index>> {
        &self.input_mapping
    }

    /// Get all output mappings.
    pub fn output_mappings(&self) -> &HashMap<V, IndexMapping<T::Index>> {
        &self.output_mapping
    }

    /// Get node names covered by this operator.
    pub fn node_names(&self) -> HashSet<V> {
        self.mpo
            .site_index_network()
            .node_names()
            .into_iter()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests;
