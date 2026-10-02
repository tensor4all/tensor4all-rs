//! Tree Tensor Network implementation.
//!
//! This module provides the [`TreeTN`] type, a tree-structured tensor network
//! for efficient tensor operations with canonicalization and truncation support.

// Some utility functions are WIP and not yet connected
#![allow(dead_code)]

mod addition;
mod cached_evaluator;
mod canonicalize;
pub mod contraction;
mod decompose;
#[cfg(feature = "diagnostics")]
pub mod diagnostics;
mod evaluator;
mod fit;
mod localupdate;
mod operator_impl;
mod ops;
pub mod partial_contraction;
mod restructure;
mod swap;
mod tensor_like;
mod transform;
mod truncate;

use crate::error::TreeTNOperationError;
use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{Dfs, EdgeRef};
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;

use anyhow::{Context, Result};

use crate::algorithm::CanonicalForm;
use tensor4all_core::{Canonical, FactorizeAlg, FactorizeOptions, IndexLike, TensorLike};

use crate::named_graph::NamedGraph;
use crate::site_index_network::SiteIndexNetwork;

// Re-export the decomposition functions and types
pub use decompose::{factorize_tensor_to_treetn, factorize_tensor_to_treetn_with, TreeTopology};

// Re-export cached evaluator types
pub use cached_evaluator::{
    CachedEvaluatorOptions, CachedEvaluatorPlan, CenterSearchResult, EvaluatedScalarKindMismatch,
    EvaluationHint, GreedyCenterSearch, TreeTNCachedEvaluator,
};

// Re-export evaluator types
pub use evaluator::TreeTNEvaluator;

// Re-export local update types
pub use localupdate::{
    apply_local_update_sweep, get_boundary_edges, BoundaryEdge, LocalUpdateStep,
    LocalUpdateSweepPlan, LocalUpdater, TruncateUpdater,
};

// Re-export the variational sum-fit entry point and its contraction options alias.
pub use fit::{fit_sum, FitContractionOptions as FitOptions};

// Re-export partial contraction types
pub use partial_contraction::{
    hadamard, partial_contract, partial_contract_to_site_network, sum_over_indices,
    weighted_sum_over_index_pairs, PartialContractionSpec,
};

// Re-export swap types
pub use swap::{ScheduledSwapStep, SwapOptions, SwapSchedule};

/// Legs grouped by full index, in first-occurrence order: each entry is an
/// index and the `(node, leg)` pairs that carry it (used by `from_tensors`).
type IndexGroups<I> = Vec<(I, Vec<(NodeIndex, I)>)>;

/// Factorize `tensor` into a left factor carrying `left_indices` and a right
/// factor carrying its remaining indices, where either side may be empty.
///
/// This is the single place where TreeTN splits with an empty side are
/// handled: canonicalization, `TruncateUpdater`, variational fitting, site
/// swaps and topology-preserving zip-up all route through it, so a site-free
/// node gets the same treatment everywhere.
///
/// `factorize` performs an ordinary split with both sides non-empty (for
/// example `factorize`, `factorize_in`, `factorize_full_rank` or
/// `factorize_auto`), and therefore decides the algorithm, the truncation,
/// the canonical direction and the execution context. When both sides are
/// non-empty it is called on `tensor` directly. Otherwise, a fresh
/// dimension-one axis is stacked onto each side, `factorize` splits the
/// augmented tensor, and both axes are selected away from the factors. The
/// returned bond then has dimension one and the factors keep the requested
/// canonical form: with `Canonical::Left`, an empty left side becomes a
/// unit-modulus scalar on the new bond, and an empty right side leaves the
/// left factor normalized while the right factor carries the norm.
/// `Canonical::Right` mirrors this.
///
/// The extra axes are added with `stack_along_new_index` rather than an
/// outer product with a real unit tensor: the augmented tensor keeps the
/// input's scalar type and execution context, whereas a mixed real/complex
/// outer product leaves the context and drops AD tracking.
///
/// # Errors
///
/// Returns an error when `left_indices` is not a subset of `tensor`'s
/// indices (compared by full index equality), when adding or removing the
/// singleton axes fails, or with the error of `factorize`.
fn factorize_allowing_empty_side<T, F>(
    tensor: &T,
    left_indices: &[T::Index],
    factorize: F,
) -> Result<tensor4all_core::FactorizeResult<T>>
where
    T: TensorLike,
    F: FnOnce(&T, &[T::Index]) -> Result<tensor4all_core::FactorizeResult<T>>,
{
    let tensor_indices = tensor.external_indices();
    if let Some(missing) = left_indices
        .iter()
        .find(|index| !tensor_indices.contains(index))
    {
        return Err(anyhow::anyhow!(
            "left index {missing:?} is not an index of the tensor to factorize"
        ));
    }
    let right_is_empty = tensor_indices
        .iter()
        .all(|index| left_indices.contains(index));
    if !left_indices.is_empty() && !right_is_empty {
        return factorize(tensor, left_indices);
    }

    let left_boundary = T::Index::new_link(1).context("failed to create left boundary")?;
    let right_boundary = T::Index::new_link(1).context("failed to create right boundary")?;
    let augmented = T::stack_along_new_index(&[tensor], left_boundary.clone(), -1)
        .and_then(|with_left| T::stack_along_new_index(&[&with_left], right_boundary.clone(), -1))
        .context("failed to add singleton boundaries for factorization")?;
    let mut augmented_left = left_indices.to_vec();
    augmented_left.push(left_boundary.clone());

    let result = factorize(&augmented, &augmented_left)
        .context("failed to factorize singleton-boundary tensor")?;
    let left = result
        .left
        .select_indices(std::slice::from_ref(&left_boundary), &[0])
        .context("failed to remove left singleton boundary")?;
    let right = result
        .right
        .select_indices(std::slice::from_ref(&right_boundary), &[0])
        .context("failed to remove right singleton boundary")?;

    Ok(tensor4all_core::FactorizeResult::new(
        left,
        right,
        result.bond_index,
        result.singular_values,
        result.rank,
    ))
}

/// Tree Tensor Network structure (inspired by ITensorNetworks.jl's TreeTensorNetwork).
/// Maintains a graph of tensors connected by bonds (edges).
/// Each node stores a tensor, and edges store `Connection` objects
/// that hold the bond index.
/// The structure uses SiteIndexNetwork to manage:
/// - **Topology**: Graph structure (which nodes connect to which)
/// - **Site Space**: Physical indices organized by node
/// # Type Parameters
/// - `T`: Tensor type implementing `TensorLike` (default: `IdxTensor`)
/// - `V`: Node name type for named nodes (default: NodeIndex for backward compatibility)
/// # Construction
/// - `TreeTN::new()`: Create an empty network, then use `add_tensor()` and `connect()` to build.
/// - `TreeTN::from_tensors(tensors, node_names)`: Create from tensors with auto-connection by matching index IDs.
/// # Examples
/// Build a 2-node chain manually and verify node count:
/// ```
/// use tensor4all_treetn::TreeTN;
/// use tensor4all_core::{DynIndex, IdxTensor, TensorLike};
/// // Create site and bond indices
/// let s0 = DynIndex::new_dyn(2);
/// let bond = DynIndex::new_dyn(3);
/// let s1 = DynIndex::new_dyn(2);
/// // Build tensors
/// let t0 = IdxTensor::from_dense(
///     vec![s0.clone(), bond.clone()],
///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ).unwrap();
/// let t1 = IdxTensor::from_dense(
///     vec![bond.clone(), s1.clone()],
///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ).unwrap();
/// // Use from_tensors (auto-connects nodes sharing the same index ID)
/// let tn = TreeTN::<_, String>::from_tensors(
///     vec![t0, t1],
///     vec!["A".to_string(), "B".to_string()],
/// ).unwrap();
/// assert_eq!(tn.node_count(), 2);
/// assert_eq!(tn.edge_count(), 1);
/// ```
pub struct TreeTN<T = tensor4all_core::IdxTensor, V = NodeIndex>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Named graph wrapper: provides mapping between node names (V) and NodeIndex
    /// Edges store the bond Index directly.
    pub(crate) graph: NamedGraph<V, T, T::Index>,
    /// Orthogonalization region (canonical_region).
    /// When empty, the network is not canonicalized.
    /// When non-empty, contains the node names (V) of the orthogonalization region.
    /// The region must form a connected subtree in the network.
    pub(crate) canonical_region: HashSet<V>,
    /// Canonical form used for the current canonicalization.
    /// `None` if not canonicalized (canonical_region is empty).
    /// `Some(form)` if canonicalized with the specified form.
    pub(crate) canonical_form: Option<CanonicalForm>,
    /// Site index network: manages topology and site space (physical indices).
    /// This structure enables topology and site space comparison independent of tensor data.
    pub(crate) site_index_network: SiteIndexNetwork<V, T::Index>,
    /// Link index network: manages bond/link indices with reverse lookup.
    /// Provides O(1) lookup from index ID to edge.
    pub(crate) link_index_network: crate::link_index_network::LinkIndexNetwork<T::Index>,
    /// Orthogonalization direction for each index (bond or site).
    /// Maps index to the node name (V) that the orthogonalization points towards.
    /// - For bond indices: points towards the canonical center direction
    /// - For site indices: points to the node that owns the index (always towards canonical center)
    ///
    /// Note: Uses the full index as the key (via `IndexLike: Eq + Hash`).
    pub(crate) ortho_towards: HashMap<T::Index, V>,
}

/// Internal context for sweep-to-center operations.
/// Contains precomputed information needed for both canonicalization and truncation.
#[derive(Debug)]
pub(crate) struct SweepContext {
    /// Edges to process, ordered from leaves towards center.
    /// Each edge is (src, dst) where src is the node to factorize and dst is its parent.
    pub(crate) edges: Vec<(NodeIndex, NodeIndex)>,
}

// ============================================================================
// Construction methods
// ============================================================================

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Create a new empty TreeTN.
    ///
    /// Use `add_tensor()` to add tensors and `connect()` to establish bonds manually.
    pub fn new() -> Self {
        Self {
            graph: NamedGraph::new(),
            canonical_region: HashSet::new(),
            canonical_form: None,
            site_index_network: SiteIndexNetwork::new(),
            link_index_network: crate::link_index_network::LinkIndexNetwork::new(),
            ortho_towards: HashMap::new(),
        }
    }

    /// Create a TreeTN from a list of tensors and node names using einsum rule.
    ///
    /// This function connects tensors that share common indices (by ID).
    /// The algorithm is O(n) where n is the number of tensors:
    /// 1. Add all tensors as nodes
    /// 2. Build a map from index ID to (node, index) pairs in a single pass
    /// 3. Connect nodes that share the same index ID
    ///
    /// # Arguments
    /// * `tensors` - Vector of tensors to add to the network
    /// * `node_names` - Vector of node names corresponding to each tensor
    ///
    /// # Returns
    /// A new TreeTN with tensors connected by common indices, or an error if:
    /// - The lengths of `tensors` and `node_names` don't match
    /// - An index ID appears in more than 2 tensors (TreeTN is a tree, so each bond connects exactly 2 nodes)
    /// - Connection fails (e.g., shape mismatch)
    ///
    /// # Errors
    /// Returns an error when `tensors` and `node_names` differ in length (a
    /// shape mismatch) or a tensor structure is invalid (an invalid-state
    /// failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorLike};
    ///
    /// let s0 = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(3);
    /// let s1 = DynIndex::new_dyn(2);
    ///
    /// let t0 = IdxTensor::from_dense(
    ///     vec![s0.clone(), bond.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// ).unwrap();
    /// let t1 = IdxTensor::from_dense(
    ///     vec![bond.clone(), s1.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// ).unwrap();
    ///
    /// let tn = TreeTN::<_, String>::from_tensors(
    ///     vec![t0, t1],
    ///     vec!["A".to_string(), "B".to_string()],
    /// ).unwrap();
    ///
    /// assert_eq!(tn.node_count(), 2);
    /// assert_eq!(tn.edge_count(), 1);
    /// ```
    pub fn from_tensors(
        tensors: Vec<T>,
        node_names: Vec<V>,
    ) -> std::result::Result<Self, TreeTNOperationError>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        V: Ord,
    {
        let treetn = Self::from_tensors_unchecked(tensors, node_names)?;

        // Verify structural constraints after construction
        treetn.verify_internal_consistency().context(
            "TreeTN::from_tensors: constructed TreeTN failed internal consistency check",
        )?;

        Ok(treetn)
    }

    /// Internal version of `from_tensors` that skips verification.
    /// Used by `verify_internal_consistency` to avoid infinite recursion.
    fn from_tensors_unchecked(tensors: Vec<T>, node_names: Vec<V>) -> Result<Self>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        // Validate input lengths
        if tensors.len() != node_names.len() {
            return Err(anyhow::anyhow!(
                "Length mismatch: {} tensors but {} node names",
                tensors.len(),
                node_names.len()
            ))
            .context("TreeTN::from_tensors: tensors and node_names must have the same length");
        }

        // Create empty TreeTN
        let mut treetn = Self::new();

        // Step 1: Add all tensors as nodes and collect NodeIndex mappings
        let mut node_indices = Vec::with_capacity(tensors.len());
        for (tensor, node_name) in tensors.into_iter().zip(node_names) {
            let node_idx = treetn.add_tensor_internal(node_name, tensor)?;
            node_indices.push(node_idx);
        }

        // Step 2: Group every leg by its full index metadata in O(n) time.
        // Same-id indices that differ by prime level or tags are distinct site legs.
        // Groups are kept in first-occurrence order (tensor position, then leg
        // position), and `group_of` only maps an index to its group, so edges are
        // connected in a deterministic order below. Iterating a HashMap instead
        // would make the petgraph edge insertion order, and with it every
        // neighbor-ordered traversal, depend on the hasher state.
        let mut groups: IndexGroups<T::Index> = Vec::new();
        let mut group_of: HashMap<T::Index, usize> = HashMap::new();

        for node_idx in &node_indices {
            let tensor = treetn
                .tensor(*node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_idx))?;

            for index in tensor.external_indices() {
                let group = *group_of.entry(index.clone()).or_insert_with(|| {
                    groups.push((index.clone(), Vec::new()));
                    groups.len() - 1
                });
                groups[group].1.push((*node_idx, index));
            }
        }

        // Step 3: Connect nodes that share the same full index, in first-occurrence order.
        // For TreeTN (tree structure), each bond index should appear in exactly 2 tensors.
        // Every group holds at least one leg.
        for (shared_index, nodes_with_index) in groups {
            match nodes_with_index.len() {
                1 => {
                    // Index appears in only one tensor - this is a physical index, no connection needed
                    continue;
                }
                2 => {
                    // Index appears in exactly 2 tensors - connect them
                    let (node_a, index_a) = &nodes_with_index[0];
                    let (node_b, index_b) = &nodes_with_index[1];

                    treetn
                        .connect_internal(*node_a, index_a, *node_b, index_b)
                        .with_context(|| {
                            format!(
                                "Failed to connect nodes {:?} and {:?} via index {:?}",
                                node_a, node_b, shared_index
                            )
                        })?;
                }
                n => {
                    // Index appears in more than 2 tensors - this violates tree structure
                    return Err(anyhow::anyhow!(
                        "Index {:?} appears in {} tensors, but TreeTN requires exactly 2 (tree structure)",
                        shared_index, n
                    )
                    .context("TreeTN::from_tensors: each bond index must connect exactly 2 nodes"));
                }
            }
        }

        Ok(treetn)
    }

    /// Add a tensor to the network with a node name.
    ///
    /// Returns the NodeIndex for the newly added tensor.
    ///
    /// Also updates the site_index_network with the physical indices (all indices initially,
    /// as no connections exist yet).
    /// # Errors
    /// Returns an error when the tensor's site dimensions are incompatible with
    /// its neighbors (a shape mismatch) or the node name is already in use
    /// (a duplicate operation failure).
    ///
    pub fn add_tensor(
        &mut self,
        node_name: V,
        tensor: T,
    ) -> std::result::Result<NodeIndex, TreeTNOperationError> {
        self.add_tensor_internal(node_name, tensor)
            .map_err(TreeTNOperationError::from)
    }

    /// Add a tensor to the network using NodeIndex as the node name.
    ///
    /// This method only works when `V = NodeIndex`. The `tensor` argument
    /// supplies the node data, and the generated node index is also used as
    /// the public node name.
    ///
    /// Returns the NodeIndex for the newly added tensor.
    ///
    /// # Errors
    /// Returns an error when the tensor's site dimensions are incompatible with
    /// its neighbors (a shape mismatch).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(3);
    /// let tensor = IdxTensor::from_dense(vec![i, j], vec![0.0_f64; 6]).unwrap();
    ///
    /// let mut tn = TreeTN::<IdxTensor>::new();
    /// let node = tn.add_tensor_auto_name(tensor).unwrap();
    ///
    /// assert_eq!(node.index(), 0);
    /// assert_eq!(tn.node_count(), 1);
    /// ```
    pub fn add_tensor_auto_name(
        &mut self,
        tensor: T,
    ) -> std::result::Result<NodeIndex, TreeTNOperationError>
    where
        V: From<NodeIndex> + Into<NodeIndex>,
    {
        // We need to add with a temporary name first, then get the actual NodeIndex
        let temp_idx = self.graph.graph_mut().add_node(tensor.clone());
        let node_name = V::from(temp_idx);

        // Remove the temporary node and add properly with name
        self.graph.graph_mut().remove_node(temp_idx);

        // Re-add with the correct name
        self.add_tensor_internal(node_name, tensor)
            .map_err(TreeTNOperationError::from)
    }

    /// Connect two tensors via a specified pair of indices.
    ///
    /// The indices must match as full index values.
    ///
    /// # Arguments
    /// * `node_a` - First node
    /// * `index_a` - Index on first node to use for connection
    /// * `node_b` - Second node
    /// * `index_b` - Index on second node to use for connection
    ///
    /// # Returns
    /// The EdgeIndex of the new connection, or an error if validation fails.
    /// # Errors
    /// Returns an error when the indices do not belong to the two nodes (a
    /// missing-index failure) or their dimensions differ (a dimension
    /// mismatch).
    ///
    pub fn connect(
        &mut self,
        node_a: NodeIndex,
        index_a: &T::Index,
        node_b: NodeIndex,
        index_b: &T::Index,
    ) -> std::result::Result<EdgeIndex, TreeTNOperationError> {
        self.connect_internal(node_a, index_a, node_b, index_b)
            .map_err(TreeTNOperationError::from)
    }
}

// ============================================================================
// Common implementation
// ============================================================================

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    // ------------------------------------------------------------------------
    // Internal methods (used by mode-specific methods)
    // ------------------------------------------------------------------------

    /// Internal method to add a tensor with a node name.
    pub(crate) fn add_tensor_internal(&mut self, node_name: V, tensor: T) -> Result<NodeIndex> {
        // Extract physical indices: initially all indices are physical (no connections yet).
        // The site space only records membership; ordered views of the site legs
        // (`node_site_indices`) are read from the tensor's own leg order.
        let physical_indices: HashSet<T::Index> = tensor.external_indices().into_iter().collect();

        // Add to graph
        let node_idx = self.graph.add_node(node_name.clone(), tensor)?;

        // Add to site_index_network
        self.site_index_network
            .add_node(node_name, physical_indices)?;

        Ok(node_idx)
    }

    /// Internal method to connect two tensors.
    ///
    /// In Einsum mode, `index_a` and `index_b` must be the same full index.
    pub(crate) fn connect_internal(
        &mut self,
        node_a: NodeIndex,
        index_a: &T::Index,
        node_b: NodeIndex,
        index_b: &T::Index,
    ) -> Result<EdgeIndex> {
        // Validate that indices have the same full metadata (Einsum mode requirement).
        if index_a != index_b {
            return Err(anyhow::anyhow!(
                "Indices must match in Einsum mode: {:?} != {:?}",
                index_a,
                index_b
            ))
            .context("Failed to connect tensors");
        }

        // Validate that nodes exist
        if !self.graph.contains_node(node_a) || !self.graph.contains_node(node_b) {
            return Err(anyhow::anyhow!("One or both nodes do not exist")
                .context("Failed to connect tensors"));
        }

        // Validate that indices exist in respective tensors
        let tensor_a = self
            .tensor(node_a)
            .ok_or_else(|| anyhow::anyhow!("Tensor for node_a not found"))?;
        let tensor_b = self
            .tensor(node_b)
            .ok_or_else(|| anyhow::anyhow!("Tensor for node_b not found"))?;

        // Check that indices exist in tensors
        let has_index_a = tensor_a.external_indices().iter().any(|idx| idx == index_a);
        let has_index_b = tensor_b.external_indices().iter().any(|idx| idx == index_b);

        if !has_index_a {
            return Err(anyhow::anyhow!("Index not found in tensor_a")
                .context("Failed to connect: index_a must exist in tensor_a"));
        }
        if !has_index_b {
            return Err(anyhow::anyhow!("Index not found in tensor_b")
                .context("Failed to connect: index_b must exist in tensor_b"));
        }

        // Clone the bond index.
        let bond_index = tensor_a
            .external_indices()
            .iter()
            .find(|idx| *idx == index_a)
            .ok_or_else(|| anyhow::anyhow!("Index not found in tensor_a"))?
            .clone();

        // Get node names for site_index_network (before mutable borrow)
        let node_name_a = self
            .graph
            .node_name(node_a)
            .ok_or_else(|| anyhow::anyhow!("Node name for node_a not found"))?
            .clone();
        let node_name_b = self
            .graph
            .node_name(node_b)
            .ok_or_else(|| anyhow::anyhow!("Node name for node_b not found"))?
            .clone();

        // Add edge to graph with the bond index directly
        let edge_idx = self
            .graph
            .graph_mut()
            .add_edge(node_a, node_b, bond_index.clone());

        // Add edge to site_index_network
        self.site_index_network
            .add_edge(&node_name_a, &node_name_b)?;

        // Update physical indices: remove bond index from physical indices
        // Use remove_site_index to also update the index_to_node reverse lookup
        let _ = self
            .site_index_network
            .remove_site_index(&node_name_a, &bond_index);
        let _ = self
            .site_index_network
            .remove_site_index(&node_name_b, &bond_index);

        // Register bond index in link_index_network for reverse lookup
        self.link_index_network.insert(edge_idx, &bond_index);

        Ok(edge_idx)
    }

    /// Prepare context for sweep-to-center operations.
    ///
    /// This method:
    /// 1. Validates tree structure
    /// 2. Sets canonical_region and validates connectivity
    /// 3. Computes edges from leaves towards center using edges_to_canonicalize_to_region
    ///
    /// # Arguments
    /// * `canonical_region` - The node names that will serve as centers
    /// * `context_name` - Name for error context (e.g., "canonicalize_with")
    ///
    /// # Returns
    /// A SweepContext if successful, or an error if validation fails.
    pub(crate) fn prepare_sweep_to_center(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        context_name: &str,
    ) -> Result<Option<SweepContext>> {
        // 1. Validate tree structure
        self.validate_tree()
            .with_context(|| format!("{}: graph must be a tree", context_name))?;

        // 2. Set canonical_region
        let canonical_region_v: Vec<V> = canonical_region.into_iter().collect();
        self.set_canonical_region(canonical_region_v)
            .with_context(|| format!("{}: failed to set canonical_region", context_name))?;

        if self.canonical_region.is_empty() {
            return Ok(None); // Nothing to do if no centers
        }

        // 3. Convert canonical_region names to NodeIndex set
        let center_indices: HashSet<NodeIndex> = self
            .canonical_region
            .iter()
            .filter_map(|name| self.graph.node_index(name))
            .collect();

        // 4. Validate canonical_region connectivity
        if !self.site_index_network.is_connected_subset(&center_indices) {
            return Err(anyhow::anyhow!(
                "canonical_region is not connected: {} centers but not all reachable",
                self.canonical_region.len()
            ))
            .with_context(|| {
                format!(
                    "{}: canonical_region must form a connected subtree",
                    context_name
                )
            });
        }

        // 5. Get ordered edges from leaves towards center
        let canonicalize_edges = self
            .site_index_network
            .edges_to_canonicalize_to_region(&center_indices);
        let edges: Vec<(NodeIndex, NodeIndex)> = canonicalize_edges.into_iter().collect();

        Ok(Some(SweepContext { edges }))
    }

    /// Process one edge during an exact sweep operation.
    ///
    /// Factorizes the tensor at `src` node with the requested decomposition
    /// algorithm, absorbs the right factor into `dst` (parent), and updates the
    /// edge bond and ortho_towards. No truncation controls are passed and the
    /// global rank-dropping defaults are not consulted, so the represented
    /// tensor is preserved exactly.
    ///
    /// A site-free `src` (one whose only index is the bond to `dst`) has an
    /// empty left side. Its whole tensor is then absorbed into `dst`, `src`
    /// keeps a unit-modulus scalar, and the edge is replaced by a fresh
    /// dimension-one bond. This is exact for any original bond dimension;
    /// see `factorize_allowing_empty_side`.
    ///
    /// # Arguments
    /// * `src` - The source node to factorize (further from center)
    /// * `dst` - The destination/parent node (closer to center)
    /// * `alg` - Decomposition algorithm to use
    /// * `canonical` - Which factor carries the canonical form
    /// * `context_name` - Name for error context
    ///
    /// # Returns
    /// `Ok(())` if successful, or an error if any step fails.
    pub(crate) fn sweep_edge_full_rank(
        &mut self,
        src: NodeIndex,
        dst: NodeIndex,
        alg: FactorizeAlg,
        canonical: Canonical,
        context_name: &str,
    ) -> Result<()> {
        self.sweep_edge_full_rank_scoped(src, dst, alg, canonical, context_name, None)
    }

    /// Context-scoped full-rank edge sweep.
    ///
    /// Factorizes through `factorize_full_rank_in`, so LU/CI forms return
    /// typed errors instead of running. A site-free `src` is absorbed into
    /// `dst` exactly as in [`Self::sweep_edge_full_rank`], and both factors
    /// stay in `context`.
    pub(crate) fn sweep_edge_full_rank_in(
        &mut self,
        src: NodeIndex,
        dst: NodeIndex,
        alg: FactorizeAlg,
        canonical: Canonical,
        context_name: &str,
        context: &tensor4all_tensorbackend::ExecutionContext,
    ) -> Result<()> {
        self.sweep_edge_full_rank_scoped(src, dst, alg, canonical, context_name, Some(context))
    }

    fn sweep_edge_full_rank_scoped(
        &mut self,
        src: NodeIndex,
        dst: NodeIndex,
        alg: FactorizeAlg,
        canonical: Canonical,
        context_name: &str,
        context: Option<&tensor4all_tensorbackend::ExecutionContext>,
    ) -> Result<()> {
        // Find edge between src and dst
        let edge = {
            let g = self.graph.graph();
            g.edges_connecting(src, dst)
                .next()
                .ok_or_else(|| {
                    anyhow::anyhow!("No edge found between node {:?} and {:?}", src, dst)
                })
                .with_context(|| format!("{}: edge not found", context_name))?
                .id()
        };

        // Get bond index on src-side (the index we will factorize over)
        let bond_on_src = self
            .bond_index(edge)
            .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge"))
            .with_context(|| format!("{}: failed to get bond index on src", context_name))?
            .clone();

        // Get tensor at src node
        let tensor_src = self
            .tensor(src)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", src))
            .with_context(|| format!("{}: tensor not found", context_name))?;

        // Build left_inds = all indices except dst bond
        let left_inds: Vec<T::Index> = tensor_src
            .external_indices()
            .iter()
            .filter(|idx| *idx != &bond_on_src)
            .cloned()
            .collect();

        let tensor_external_indices = tensor_src.external_indices();
        if left_inds.len() == tensor_external_indices.len() {
            return Err(anyhow::anyhow!(
                "Cannot process node {:?}: its tensor does not carry the bond to {:?}",
                src,
                dst
            ))
            .with_context(|| format!("{}: invalid tensor rank for factorization", context_name));
        }

        // Perform factorization (context-scoped when a context is supplied).
        // A site-free source has an empty left side; the shared helper then
        // leaves a unit-modulus scalar on a fresh dimension-one bond and moves
        // the whole source tensor into `dst`, which is exact for any bond
        // dimension.
        let factorize_result =
            factorize_allowing_empty_side(tensor_src, &left_inds, |tensor, left| {
                match context {
                    Some(execution) => {
                        tensor.factorize_full_rank_in(left, alg, canonical, execution)
                    }
                    None => tensor.factorize_full_rank(left, alg, canonical),
                }
                .with_context(|| format!("{alg:?} full-rank factorization failed"))
            })
            .map_err(|error| {
                // Keep the whole chain as the source, and name its root cause
                // (for example the rank-zero bond of an LU/CI split of a zero
                // tensor) in the top-level message, which is all that
                // `TreeTNOperationError`'s `Display` shows.
                let cause = error.root_cause().to_string();
                error.context(format!("{context_name}: factorization failed: {cause}"))
            })?;

        let left_tensor = factorize_result.left;
        let right_tensor = factorize_result.right;

        // Absorb right_tensor into dst
        let tensor_dst = self
            .tensor(dst)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found for dst node {:?}", dst))
            .with_context(|| format!("{}: dst tensor not found", context_name))?;

        let updated_dst_tensor = T::contract(&[tensor_dst, &right_tensor]).with_context(|| {
            format!(
                "{}: failed to absorb right factor into dst tensor",
                context_name
            )
        })?;

        // Update bond index FIRST, so replace_tensor validation matches
        let new_bond_index = factorize_result.bond_index;
        self.replace_edge_bond(edge, new_bond_index.clone())
            .with_context(|| format!("{}: failed to update edge bond index", context_name))?;

        // Update tensors
        self.replace_tensor(src, left_tensor)
            .with_context(|| format!("{}: failed to replace tensor at src node", context_name))?;
        self.replace_tensor(dst, updated_dst_tensor)
            .with_context(|| format!("{}: failed to replace tensor at dst node", context_name))?;

        // Set ortho_towards to point towards dst (canonical_region direction)
        let dst_name = self
            .graph
            .node_name(dst)
            .ok_or_else(|| anyhow::anyhow!("Dst node name not found"))?
            .clone();
        self.set_edge_ortho_towards(edge, Some(dst_name))
            .with_context(|| format!("{}: failed to set ortho_towards", context_name))?;

        Ok(())
    }

    // ------------------------------------------------------------------------
    // Public accessors
    // ------------------------------------------------------------------------

    /// Get a reference to a tensor by NodeIndex.
    pub fn tensor(&self, node: NodeIndex) -> Option<&T> {
        self.graph.graph().node_weight(node)
    }

    /// Get a mutable reference to a tensor by NodeIndex.
    pub fn tensor_mut(&mut self, node: NodeIndex) -> Option<&mut T> {
        self.graph.graph_mut().node_weight_mut(node)
    }

    pub(crate) fn remove_tensor_by_name(&mut self, node_name: &V) -> Option<T> {
        self.graph.remove_node(node_name)
    }

    /// Replace a tensor at the given node with a new tensor.
    ///
    /// Validates that the new tensor contains all indices used in connections
    /// to this node. Returns an error if any connection index is missing.
    ///
    /// Returns the old tensor if the node exists and validation passes.
    /// # Errors
    /// Returns an error when `node` is out of range (an out of bounds failure) or
    /// the new tensor has incompatible dimensions (a shape mismatch).
    ///
    pub fn replace_tensor(
        &mut self,
        node: NodeIndex,
        new_tensor: T,
    ) -> std::result::Result<Option<T>, TreeTNOperationError> {
        // Check if node exists
        if !self.graph.contains_node(node) {
            return Ok(None);
        }

        // Validate that all connection indices exist in the new tensor
        let edges = self.edges_for_node(node);
        let connection_indices: Vec<T::Index> = edges
            .iter()
            .filter_map(|(edge_idx, _neighbor)| self.bond_index(*edge_idx).cloned())
            .collect();

        // Check if all connection indices are present in the new tensor
        let new_tensor_indices = new_tensor.external_indices();
        let common = common_inds(&connection_indices, &new_tensor_indices);
        if common.len() != connection_indices.len() {
            return Err(TreeTNOperationError::from(
                anyhow::anyhow!(
                    "New tensor is missing {} connection index(es): found {} out of {} required indices",
                    connection_indices.len() - common.len(),
                    common.len(),
                    connection_indices.len()
                )
                .context("replace_tensor: new tensor must contain all indices used in connections"),
            ));
        }

        // Get node name for site_index_network update
        let node_name = self
            .graph
            .node_name(node)
            .ok_or_else(|| anyhow::anyhow!("Node name not found"))?
            .clone();

        // Calculate new physical indices: all indices minus connection indices
        let connection_indices_set: HashSet<T::Index> =
            connection_indices.iter().cloned().collect();
        let new_physical_indices: HashSet<T::Index> = new_tensor_indices
            .iter()
            .filter(|idx| !connection_indices_set.contains(idx))
            .cloned()
            .collect();

        // All validations passed, replace the tensor
        let old_tensor = self
            .graph
            .graph_mut()
            .node_weight_mut(node)
            .map(|old| std::mem::replace(old, new_tensor));

        // Update site_index_network with new physical indices
        // This properly updates both the site_space and the index_to_node mapping
        self.site_index_network
            .set_site_space(&node_name, new_physical_indices)?;

        Ok(old_tensor)
    }

    /// Get the bond index for a given edge.
    pub fn bond_index(&self, edge: EdgeIndex) -> Option<&T::Index> {
        self.graph.graph().edge_weight(edge)
    }

    /// Get a mutable reference to the bond index for a given edge.
    pub fn bond_index_mut(&mut self, edge: EdgeIndex) -> Option<&mut T::Index> {
        self.graph.graph_mut().edge_weight_mut(edge)
    }

    /// Get all edges connected to a node.
    pub fn edges_for_node(&self, node: NodeIndex) -> Vec<(EdgeIndex, NodeIndex)> {
        self.graph
            .graph()
            .edges(node)
            .map(|edge| {
                let target = edge.target();
                (edge.id(), target)
            })
            .collect()
    }

    /// Replace the bond index for an edge (e.g., after SVD creates a new bond index).
    ///
    /// Also updates site_index_network: the old bond index becomes physical again,
    /// and the new bond index is removed from physical indices.
    /// # Errors
    /// Returns an error when the edge is not found (a missing-edge failure) or
    /// the new bond has an incompatible dimension (a shape mismatch).
    ///
    pub fn replace_edge_bond(
        &mut self,
        edge: EdgeIndex,
        new_bond_index: T::Index,
    ) -> std::result::Result<(), TreeTNOperationError> {
        // Validate edge exists and get endpoints
        let (source, target) = self
            .graph
            .graph()
            .edge_endpoints(edge)
            .ok_or_else(|| anyhow::anyhow!("Edge does not exist"))?;

        // Get old bond index before updating
        let old_bond_index = self
            .bond_index(edge)
            .ok_or_else(|| anyhow::anyhow!("Bond index not found"))?
            .clone();

        // Get node names for site_index_network update
        let node_name_a = self
            .graph
            .node_name(source)
            .ok_or_else(|| anyhow::anyhow!("Node name for source not found"))?
            .clone();
        let node_name_b = self
            .graph
            .node_name(target)
            .ok_or_else(|| anyhow::anyhow!("Node name for target not found"))?
            .clone();

        // Update the bond index
        *self
            .bond_index_mut(edge)
            .ok_or_else(|| anyhow::anyhow!("Bond index not found"))? = new_bond_index.clone();

        // Update link_index_network: old id -> new id
        self.link_index_network
            .replace_index(&old_bond_index, &new_bond_index, edge)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Update ortho_towards key if present
        if let Some(dir) = self.ortho_towards.remove(&old_bond_index) {
            self.ortho_towards.insert(new_bond_index.clone(), dir);
        }

        // Update site_index_network:
        // - Old bond index becomes physical again
        // - New bond index is removed from physical
        if let Some(site_space_a) = self.site_index_network.site_space_mut(&node_name_a) {
            site_space_a.insert(old_bond_index.clone());
            site_space_a.remove(&new_bond_index);
        }
        if let Some(site_space_b) = self.site_index_network.site_space_mut(&node_name_b) {
            site_space_b.insert(old_bond_index);
            site_space_b.remove(&new_bond_index);
        }

        Ok(())
    }

    // ------------------------------------------------------------------------
    // ITensorMPS-like index relabeling helpers
    // ------------------------------------------------------------------------

    /// Return a copy with all link/bond indices replaced by fresh IDs.
    ///
    /// This is analogous to ITensorMPS.jl's `sim(link_indices, M)` / `sim!(link_indices, M)`,
    /// and is mainly useful to avoid accidental index-ID collisions when combining
    /// multiple networks.
    ///
    /// Notes:
    /// - This keeps dimensions and conjugate states, but changes identities.
    /// - This updates both endpoint tensors and internal bookkeeping.
    /// # Errors
    /// Returns an error when the link-index relabeling fails (an invalid-index
    /// failure).
    ///
    pub fn sim_link_indices(&self) -> std::result::Result<Self, TreeTNOperationError>
    where
        T::Index: IndexLike,
    {
        let mut result = self.clone();
        result.sim_link_indices_mut()?;
        Ok(result)
    }

    /// Replace all link/bond indices with fresh IDs in-place.
    ///
    /// See [`Self::sim_link_indices`] for details.
    /// # Errors
    /// Returns an error when the link-index relabeling fails (an invalid-index
    /// failure).
    ///
    pub fn sim_link_indices_mut(&mut self) -> std::result::Result<(), TreeTNOperationError>
    where
        T::Index: IndexLike,
    {
        // Snapshot edges first since replacements may touch internal maps.
        let edges: Vec<EdgeIndex> = self.graph.graph().edge_indices().collect();
        for edge in edges {
            let old_bond = self
                .bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge {:?}", edge))?
                .clone();
            let new_bond = old_bond.sim();

            // Update edge weight first so endpoint tensors can be validated against the new bond.
            *self
                .bond_index_mut(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge {:?}", edge))? =
                new_bond.clone();

            // Update endpoint tensors by matching the old bond as a full index.
            let (node_a, node_b) = self
                .graph
                .graph()
                .edge_endpoints(edge)
                .ok_or_else(|| anyhow::anyhow!("Edge {:?} not found", edge))?;
            for node in [node_a, node_b] {
                let tensor = self
                    .tensor(node)
                    .ok_or_else(|| anyhow::anyhow!("Tensor not found"))?;
                let old_in_tensor = tensor
                    .external_indices()
                    .iter()
                    .find(|idx| *idx == &old_bond)
                    .ok_or_else(|| anyhow::anyhow!("Bond index not found in endpoint tensor"))?
                    .clone();
                let new_tensor = tensor
                    .replaceind(&old_in_tensor, &new_bond)
                    .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))?;
                self.replace_tensor(node, new_tensor)?;
            }

            // Update ortho_towards key for this bond (if present).
            if let Some((key, dir)) = self
                .ortho_towards
                .iter()
                .find(|(k, _)| *k == &old_bond)
                .map(|(k, v)| (k.clone(), v.clone()))
            {
                self.ortho_towards.remove(&key);
                self.ortho_towards.insert(new_bond.clone(), dir);
            }

            // Update reverse lookup map (id -> edge).
            self.link_index_network
                .replace_index(&old_bond, &new_bond, edge)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
        }
        Ok(())
    }

    /// Set the orthogonalization direction for an index (bond or site).
    ///
    /// The direction is specified as a node name (or None to clear).
    ///
    /// # Arguments
    /// * `index` - The index to set ortho direction for
    /// * `dir` - The node name that the ortho points towards, or None to clear
    pub fn set_ortho_towards(&mut self, index: &T::Index, dir: Option<V>) {
        match dir {
            Some(node_name) => {
                self.ortho_towards.insert(index.clone(), node_name);
            }
            None => {
                self.ortho_towards.remove(index);
            }
        }
    }

    /// Get the node name that the orthogonalization points towards for an index.
    ///
    /// Returns None if ortho_towards is not set for this index.
    pub fn ortho_towards_for_index(&self, index: &T::Index) -> Option<&V> {
        self.ortho_towards.get(index)
    }

    /// Set the orthogonalization direction for an edge (by EdgeIndex).
    ///
    /// This is a convenience method that looks up the bond index and calls `set_ortho_towards`.
    ///
    /// The direction is specified as a node name (or None to clear).
    /// The node must be one of the edge's endpoints.
    /// # Errors
    /// Returns an error when the edge is not found (a missing-index failure).
    ///
    pub fn set_edge_ortho_towards(
        &mut self,
        edge: petgraph::stable_graph::EdgeIndex,
        dir: Option<V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        // Get the bond index for this edge
        let bond = self
            .bond_index(edge)
            .ok_or_else(|| anyhow::anyhow!("Edge does not exist"))?
            .clone();

        // Validate that the node (if any) is one of the edge endpoints
        if let Some(ref node_name) = dir {
            let (source, target) = self
                .graph
                .graph()
                .edge_endpoints(edge)
                .ok_or_else(|| anyhow::anyhow!("Edge does not exist"))?;

            let source_name = self.graph.node_name(source);
            let target_name = self.graph.node_name(target);

            if source_name != Some(node_name) && target_name != Some(node_name) {
                return Err(TreeTNOperationError::from(
                    anyhow::anyhow!(
                        "ortho_towards node {:?} must be one of the edge endpoints",
                        node_name
                    )
                    .context("set_edge_ortho_towards: invalid node"),
                ));
            }
        }

        self.set_ortho_towards(&bond, dir);
        Ok(())
    }

    /// Get the node name that the orthogonalization points towards for an edge.
    ///
    /// Returns None if ortho_towards is not set for this edge's bond index.
    pub fn ortho_towards_node(&self, edge: petgraph::stable_graph::EdgeIndex) -> Option<&V> {
        self.bond_index(edge)
            .and_then(|bond| self.ortho_towards.get(bond))
    }

    /// Get the NodeIndex that the orthogonalization points towards for an edge.
    ///
    /// Returns None if ortho_towards is not set for this edge's bond index.
    pub fn ortho_towards_node_index(
        &self,
        edge: petgraph::stable_graph::EdgeIndex,
    ) -> Option<NodeIndex> {
        self.ortho_towards_node(edge)
            .and_then(|name| self.graph.node_index(name))
    }

    /// Validate that the graph is a tree (or forest).
    ///
    /// Checks:
    /// - The graph is connected (all nodes reachable from the first node)
    /// - For each connected component: edges = nodes - 1 (tree condition)
    /// # Errors
    /// Returns an error when the graph is not a tree (an invalid-topology
    /// failure) or an internal invariant is violated (an invalid-state failure).
    ///
    pub fn validate_tree(&self) -> std::result::Result<(), TreeTNOperationError> {
        let g = self.graph.graph();
        if g.node_count() == 0 {
            return Ok(()); // Empty graph is trivially valid
        }

        // Check if graph is connected
        let mut visited = std::collections::HashSet::new();
        let start_node = g
            .node_indices()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Graph has no nodes"))?;

        // DFS to count reachable nodes
        // Single DFS pass (O(V + E)) to verify tree connectivity from the
        // start node; visits each reachable node exactly once.
        let mut dfs = Dfs::new(g, start_node);
        while let Some(node) = dfs.next(g) {
            visited.insert(node);
        }

        if visited.len() != g.node_count() {
            return Err(TreeTNOperationError::from(
                anyhow::anyhow!(
                    "Graph is not connected: {} nodes reachable out of {}",
                    visited.len(),
                    g.node_count()
                )
                .context("validate_tree: graph must be connected"),
            ));
        }

        // Check tree condition: edges = nodes - 1
        let node_count = g.node_count();
        let edge_count = g.edge_count();

        if edge_count != node_count - 1 {
            return Err(TreeTNOperationError::from(
                anyhow::anyhow!(
                    "Graph does not satisfy tree condition: {} edges != {} nodes - 1",
                    edge_count,
                    node_count
                )
                .context("validate_tree: tree must have edges = nodes - 1"),
            ));
        }

        Ok(())
    }

    /// Get the number of nodes in the network.
    pub fn node_count(&self) -> usize {
        self.graph.graph().node_count()
    }

    /// Get the number of edges in the network.
    pub fn edge_count(&self) -> usize {
        self.graph.graph().edge_count()
    }

    /// Get the NodeIndex for a node by name.
    pub fn node_index(&self, node_name: &V) -> Option<NodeIndex> {
        self.graph.node_index(node_name)
    }

    /// Rename an existing node while preserving topology, site space, and
    /// orthogonality metadata.
    /// # Errors
    /// Returns an error when `old_name` is not found (a missing-index failure) or
    /// `new_name` is already in use (a duplicate operation failure).
    ///
    pub fn rename_node(
        &mut self,
        old_name: &V,
        new_name: V,
    ) -> std::result::Result<(), TreeTNOperationError> {
        if old_name == &new_name {
            return Ok(());
        }

        self.graph
            .rename_node(old_name, new_name.clone())
            .context("rename_node: failed to rename graph node")?;
        self.site_index_network
            .rename_node(old_name, new_name.clone())
            .context("rename_node: failed to rename site-index node")?;

        if self.canonical_region.remove(old_name) {
            self.canonical_region.insert(new_name.clone());
        }

        for target in self.ortho_towards.values_mut() {
            if target == old_name {
                *target = new_name.clone();
            }
        }

        Ok(())
    }

    /// Get the EdgeIndex for the edge between two nodes by name.
    ///
    /// Returns `None` if either node doesn't exist or there's no edge between them.
    pub fn edge_between(&self, node_a: &V, node_b: &V) -> Option<EdgeIndex> {
        let idx_a = self.graph.node_index(node_a)?;
        let idx_b = self.graph.node_index(node_b)?;
        self.graph
            .graph()
            .find_edge(idx_a, idx_b)
            .or_else(|| self.graph.graph().find_edge(idx_b, idx_a))
    }

    /// Get all node indices in the tree tensor network.
    pub fn node_indices(&self) -> Vec<NodeIndex> {
        self.graph.graph().node_indices().collect()
    }

    /// Validate that every node tensor belongs to the supplied execution context.
    ///
    /// Generic SRC entries call this on both input trees before RNG advancement
    /// or contraction, so mixed host/CUDA inputs and foreign CUDA contexts fail
    /// at the boundary with the offending node identified.
    ///
    /// # Arguments
    /// * `context` - Caller-owned execution context both inputs must belong to.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor, TensorConstructionLike};
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tensor4all_treetn::TreeTN;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = ExecutionContext::Cpu(Arc::new(
    ///     CpuExecutionContext::from_backend(CpuBackend::new()),
    /// ));
    /// let index = DynIndex::new_dyn(2);
    /// let tensor = <IdxTensor as TensorConstructionLike>::from_dense_in(
    ///     &context,
    ///     vec![index],
    ///     vec![1.0_f64, 2.0],
    /// )?;
    /// let tree = TreeTN::from_tensors(vec![tensor], vec![0])?;
    /// tree.validate_context(&context)?;
    /// assert_eq!(tree.node_count(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`TreeTNOperationError`] identifying the offending node when any
    /// tensor does not belong to `context`.
    pub fn validate_context(
        &self,
        context: &tensor4all_tensorbackend::ExecutionContext,
    ) -> std::result::Result<(), TreeTNOperationError> {
        for node in self.graph.graph().node_indices() {
            let tensor = self.tensor(node).ok_or_else(|| {
                TreeTNOperationError::from(anyhow::anyhow!(
                    "context validation: node {node:?} has no tensor"
                ))
            })?;
            tensor.validate_context(context).map_err(|error| {
                let name = self.graph.node_name(node);
                TreeTNOperationError::from(anyhow::anyhow!(
                    "context validation: node {name:?} does not belong to the supplied execution context: {error}"
                ))
            })?;
        }
        Ok(())
    }

    /// Get all node names in the tree tensor network.
    ///
    /// Names are returned in internal `NodeIndex` order, which is the order in
    /// which nodes were added (for [`from_tensors`](Self::from_tensors), the
    /// input order) as long as no node has been removed.
    pub fn node_names(&self) -> Vec<V> {
        self.graph
            .graph()
            .node_indices()
            .filter_map(|idx| self.graph.node_name(idx).cloned())
            .collect()
    }

    /// Compute edges to canonicalize from leaves to target, returning node names.
    ///
    /// Returns `(from, to)` pairs in the order they should be processed:
    /// - `from` is the node being factorized
    /// - `to` is the parent node (towards target)
    ///
    /// This is useful for contract_zipup and similar algorithms that work with
    /// node names rather than NodeIndex.
    ///
    /// # Arguments
    /// * `target` - Target node name for the orthogonality center
    ///
    /// # Returns
    /// `None` if target node doesn't exist, otherwise a vector of `(from, to)` pairs.
    pub fn edges_to_canonicalize_by_names(&self, target: &V) -> Option<Vec<(V, V)>> {
        self.site_index_network
            .edges_to_canonicalize_by_names(target)
    }

    /// Get a reference to the orthogonalization region (using node names).
    ///
    /// When empty, the network is not canonicalized.
    pub fn canonical_region(&self) -> &HashSet<V> {
        &self.canonical_region
    }

    /// Check if the network is canonicalized.
    ///
    /// Returns `true` if `canonical_region` is non-empty, `false` otherwise.
    pub fn is_canonicalized(&self) -> bool {
        !self.canonical_region.is_empty()
    }

    /// Set the orthogonalization region (using node names).
    ///
    /// Validates that all specified nodes exist in the graph.
    /// # Errors
    /// Returns an error when a region node is not found (a missing-index failure).
    ///
    pub fn set_canonical_region(
        &mut self,
        region: impl IntoIterator<Item = V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        let region: HashSet<V> = region.into_iter().collect();

        // Validate that all nodes exist in the graph
        for node_name in &region {
            if !self.graph.has_node(node_name) {
                return Err(TreeTNOperationError::from(
                    anyhow::anyhow!("Node {:?} does not exist in the graph", node_name)
                        .context("set_canonical_region: all nodes must be valid"),
                ));
            }
        }

        self.canonical_region = region;
        Ok(())
    }

    /// Clear the orthogonalization region (mark network as not canonicalized).
    ///
    /// Also clears the canonical form.
    pub fn clear_canonical_region(&mut self) {
        self.canonical_region.clear();
        self.canonical_form = None;
    }

    /// Get the current canonical form.
    ///
    /// Returns `None` if not canonicalized.
    pub fn canonical_form(&self) -> Option<CanonicalForm> {
        self.canonical_form
    }

    /// Add a node to the orthogonalization region.
    ///
    /// Validates that the node exists in the graph.
    /// # Errors
    /// Returns an error when the node is not found (a missing-index failure).
    ///
    pub fn add_to_canonical_region(
        &mut self,
        node_name: V,
    ) -> std::result::Result<(), TreeTNOperationError> {
        if !self.graph.has_node(&node_name) {
            return Err(TreeTNOperationError::from(
                anyhow::anyhow!("Node {:?} does not exist in the graph", node_name)
                    .context("add_to_canonical_region: node must be valid"),
            ));
        }
        self.canonical_region.insert(node_name);
        Ok(())
    }

    /// Remove a node from the orthogonalization region.
    ///
    /// Returns `true` if the node was in the region, `false` otherwise.
    pub fn remove_from_canonical_region(&mut self, node_name: &V) -> bool {
        self.canonical_region.remove(node_name)
    }

    /// Get a reference to the site index network.
    ///
    /// The site index network contains both topology (graph structure) and site space (physical indices).
    pub fn site_index_network(&self) -> &SiteIndexNetwork<V, T::Index> {
        &self.site_index_network
    }

    /// Get a mutable reference to the site index network.
    pub(crate) fn site_index_network_mut(&mut self) -> &mut SiteIndexNetwork<V, T::Index> {
        &mut self.site_index_network
    }

    /// Get a reference to the site space (physical indices) for a node.
    ///
    /// The returned set is for membership tests and carries no order. Use
    /// [`node_site_indices`](Self::node_site_indices) when the order of the
    /// site legs matters.
    pub fn site_space(&self, node_name: &V) -> Option<&std::collections::HashSet<T::Index>> {
        self.site_index_network.site_space(node_name)
    }

    /// Return a node's site (physical) indices in the node tensor's leg order.
    ///
    /// The result is the node tensor's `external_indices()` with the bond
    /// legs removed, so it follows the order in which the tensor stores its
    /// legs (for tensors passed to [`from_tensors`](Self::from_tensors), the
    /// order they were constructed with). This is the per-node order used by
    /// [`TensorIndex::external_indices`](tensor4all_core::TensorIndex::external_indices),
    /// [`all_site_indices`](Self::all_site_indices) and
    /// [`contract_to_tensor`](Self::contract_to_tensor).
    ///
    /// # Arguments
    /// * `node_name` - The node whose site legs are requested.
    ///
    /// # Returns
    /// `Some(indices)` when the node exists (empty if it has no site legs),
    /// `None` when the node or its tensor is missing.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s_a = DynIndex::new_dyn(2);
    /// let s_b = DynIndex::new_dyn(3);
    /// let bond = DynIndex::new_dyn(2);
    /// let s_c = DynIndex::new_dyn(2);
    /// // Node 0 stores its legs as [s_b, bond, s_a].
    /// let t0 = IdxTensor::from_dense(
    ///     vec![s_b.clone(), bond.clone(), s_a.clone()],
    ///     vec![0.0_f64; 12],
    /// )
    /// .unwrap();
    /// let t1 = IdxTensor::from_dense(vec![bond, s_c.clone()], vec![0.0_f64; 4]).unwrap();
    /// let tn = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
    ///
    /// assert_eq!(tn.node_site_indices(&0), Some(vec![s_b, s_a]));
    /// assert_eq!(tn.node_site_indices(&1), Some(vec![s_c]));
    /// assert_eq!(tn.node_site_indices(&2), None);
    /// ```
    pub fn node_site_indices(&self, node_name: &V) -> Option<Vec<T::Index>> {
        let site_space = self.site_index_network.site_space(node_name)?;
        let node = self.graph.node_index(node_name)?;
        let tensor = self.tensor(node)?;
        Some(
            tensor
                .external_indices()
                .into_iter()
                .filter(|index| site_space.contains(index))
                .collect(),
        )
    }

    /// Validate that this network has a connected, simple linear-chain topology.
    ///
    /// A one-node network is valid; larger networks must have exactly two
    /// degree-one endpoints and all other nodes degree two. Every edge must
    /// carry one bond index and the existing TreeTN consistency checks must pass.
    ///
    /// # Errors
    /// Returns [`TreeTNOperationError`] for disconnected, branched, cyclic,
    /// multi-edge, or internally inconsistent networks.
    pub fn validate_linear_chain(&self) -> std::result::Result<(), TreeTNOperationError>
    where
        <T::Index as IndexLike>::Id: Ord,
        V: Ord,
    {
        self.verify_internal_consistency()?;

        let graph = self.graph.graph();
        let node_count = graph.node_count();
        if node_count == 0 {
            return Ok(());
        }
        let start = graph
            .node_indices()
            .next()
            .ok_or_else(|| anyhow::anyhow!("TreeTN has no start node"))?;
        let mut dfs = Dfs::new(graph, start);
        let mut visited = 0;
        while dfs.next(graph).is_some() {
            visited += 1;
        }
        if visited != node_count {
            return Err(anyhow::anyhow!("TreeTN is disconnected").into());
        }
        if graph.edge_count() != node_count - 1 {
            return Err(anyhow::anyhow!(
                "TreeTN linear chain requires {} edges, found {}",
                node_count - 1,
                graph.edge_count()
            )
            .into());
        }

        let mut endpoints = 0;
        for node in graph.node_indices() {
            match graph.neighbors_undirected(node).count() {
                1 => endpoints += 1,
                2 if node_count > 2 => {}
                0 if node_count == 1 => {}
                degree => {
                    return Err(anyhow::anyhow!(
                        "TreeTN linear chain node has invalid degree {degree}"
                    )
                    .into());
                }
            }
        }
        if node_count > 1 && endpoints != 2 {
            return Err(anyhow::anyhow!(
                "TreeTN linear chain requires two endpoints, found {endpoints}"
            )
            .into());
        }
        for edge in graph.edge_indices() {
            if self.bond_index(edge).is_none() {
                return Err(anyhow::anyhow!("TreeTN edge is missing its bond index").into());
            }
        }
        Ok(())
    }

    /// Check if two TreeTNs share equivalent site index network structure.
    ///
    /// Two TreeTNs share equivalent structure if:
    /// - Same topology (nodes and edges)
    /// - Same site space for each node
    ///
    /// This is used to verify that two TreeTNs can be added or contracted.
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to check against
    ///
    /// # Returns
    /// `true` if the networks share equivalent site index structure, `false` otherwise.
    pub fn share_equivalent_site_index_network(&self, other: &Self) -> bool
    where
        <T::Index as IndexLike>::Id: Ord,
    {
        self.site_index_network
            .share_equivalent_site_index_network(&other.site_index_network)
    }

    /// Check if two TreeTNs have the same topology (graph structure).
    ///
    /// This only checks that both networks have the same nodes and edges,
    /// not that they have the same site indices.
    ///
    /// Useful for operations like `contract_zipup` where we need networks
    /// with the same structure but possibly different site indices.
    pub fn same_topology(&self, other: &Self) -> bool {
        self.site_index_network
            .topology()
            .same_topology(other.site_index_network.topology())
    }

    /// Check if two TreeTNs have the same "appearance".
    ///
    /// Two TreeTNs have the same appearance if:
    /// 1. They have the same topology (same nodes and edges)
    /// 2. They have the same external indices (physical indices) at each node
    ///    (compared as sets, so order within a node doesn't matter)
    /// 3. They have the same orthogonalization direction (ortho_towards) on each edge
    ///
    /// This is a weaker check than `share_equivalent_site_index_network`:
    /// - `share_equivalent_site_index_network`: checks topology + site space (indices)
    /// - `same_appearance`: checks topology + site space + ortho_towards directions
    ///
    /// Note: This does NOT compare tensor data, only structural information.
    /// Note: Bond index IDs may differ between the two TreeTNs (e.g., after independent
    ///       canonicalization), so we compare ortho_towards by edge position, not by index ID.
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to compare against
    ///
    /// # Returns
    /// `true` if both TreeTNs have the same appearance, `false` otherwise.
    pub fn same_appearance(&self, other: &Self) -> bool
    where
        <T::Index as IndexLike>::Id: Ord,
        V: Ord,
    {
        // Step 1: Check topology and site space
        if !self.share_equivalent_site_index_network(other) {
            return false;
        }

        // Step 2: Check ortho_towards on each edge by position (node pair)
        // Bond index IDs may differ, so we compare by edge location (node_a, node_b)
        let mut self_bond_ortho_count = 0;
        let mut other_bond_ortho_count = 0;

        // Count bond index entries in self
        for node_name in self.node_names() {
            let self_neighbors: Vec<V> = self.site_index_network.neighbors(&node_name).collect();

            for neighbor_name in self_neighbors {
                // Only process each edge once (when node_name < neighbor_name)
                if node_name >= neighbor_name {
                    continue;
                }

                // Get edge and bond in self
                let self_edge = match self.edge_between(&node_name, &neighbor_name) {
                    Some(e) => e,
                    None => continue,
                };
                let self_bond = match self.bond_index(self_edge) {
                    Some(b) => b,
                    None => continue,
                };

                // Get edge and bond in other
                let other_edge = match other.edge_between(&node_name, &neighbor_name) {
                    Some(e) => e,
                    None => return false, // Edge exists in self but not in other
                };
                let other_bond = match other.bond_index(other_edge) {
                    Some(b) => b,
                    None => return false,
                };

                // Compare ortho_towards for this edge
                let self_ortho = self.ortho_towards.get(self_bond);
                let other_ortho = other.ortho_towards.get(other_bond);

                match (self_ortho, other_ortho) {
                    (None, None) => {} // Both have no direction - OK
                    (Some(self_dir), Some(other_dir)) => {
                        // Both have direction - must be the same
                        if self_dir != other_dir {
                            return false;
                        }
                        self_bond_ortho_count += 1;
                        other_bond_ortho_count += 1;
                    }
                    _ => return false, // One has direction, other doesn't
                }
            }
        }

        // Verify we compared all bond ortho_towards entries
        // (site index ortho_towards are not compared here as they're implied by topology)
        // Count actual bond index entries in each ortho_towards map
        let self_total_bond_entries: usize = self
            .graph
            .graph()
            .edge_indices()
            .filter_map(|e| self.bond_index(e))
            .filter(|b| self.ortho_towards.contains_key(b))
            .count();
        let other_total_bond_entries: usize = other
            .graph
            .graph()
            .edge_indices()
            .filter_map(|e| other.bond_index(e))
            .filter(|b| other.ortho_towards.contains_key(b))
            .count();

        if self_bond_ortho_count != self_total_bond_entries
            || other_bond_ortho_count != other_total_bond_entries
        {
            return false;
        }

        true
    }

    /// Perform an in-place adjacent swap on the edge (node_a, node_b).
    ///
    /// Contracts the two tensors, uses the explicitly scheduled site partition,
    /// then factorizes back in-place with `Canonical::Left` so the new center
    /// lands on `node_b`.
    pub(crate) fn swap_on_edge(
        &mut self,
        node_a_idx: NodeIndex,
        node_b_idx: NodeIndex,
        a_side_sites: &HashSet<T::Index>,
        b_side_sites: &HashSet<T::Index>,
        factorize_options: &FactorizeOptions,
    ) -> Result<()>
    where
        <T::Index as IndexLike>::Id: Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    {
        let node_b_name = self
            .graph
            .node_name(node_b_idx)
            .ok_or_else(|| anyhow::anyhow!("swap_on_edge: node_b not found"))?
            .clone();

        let edge = {
            let g = self.graph.graph();
            g.edges_connecting(node_a_idx, node_b_idx)
                .next()
                .ok_or_else(|| anyhow::anyhow!("swap_on_edge: no edge between nodes"))?
                .id()
        };
        let bond_ab = self
            .bond_index(edge)
            .ok_or_else(|| anyhow::anyhow!("swap_on_edge: bond not found"))?
            .clone();

        // Structural bonds of A and B (bonds other than bond_ab).
        let other_bonds_a: HashSet<T::Index> = self
            .edges_for_node(node_a_idx)
            .iter()
            .filter_map(|(e, _)| self.bond_index(*e).cloned())
            .filter(|b| b != &bond_ab)
            .collect();
        let other_bonds_b: HashSet<T::Index> = self
            .edges_for_node(node_b_idx)
            .iter()
            .filter_map(|(e, _)| self.bond_index(*e).cloned())
            .filter(|b| b != &bond_ab)
            .collect();

        let tensor_a = self
            .tensor(node_a_idx)
            .ok_or_else(|| anyhow::anyhow!("swap_on_edge: tensor_a not found"))?
            .clone();
        let tensor_b = self
            .tensor(node_b_idx)
            .ok_or_else(|| anyhow::anyhow!("swap_on_edge: tensor_b not found"))?
            .clone();

        // Sites currently at each node (all non-bond indices).
        let site_indices_a: HashSet<T::Index> = tensor_a
            .external_indices()
            .iter()
            .filter(|i| *i != &bond_ab && !other_bonds_a.contains(*i))
            .cloned()
            .collect();
        let site_indices_b: HashSet<T::Index> = tensor_b
            .external_indices()
            .iter()
            .filter(|i| *i != &bond_ab && !other_bonds_b.contains(*i))
            .cloned()
            .collect();
        let all_site_indices: HashSet<_> = site_indices_a.union(&site_indices_b).cloned().collect();
        let assigned_site_indices: HashSet<_> = a_side_sites.union(b_side_sites).cloned().collect();

        if !a_side_sites.is_disjoint(b_side_sites) {
            return Err(anyhow::anyhow!(
                "swap_on_edge: a_side_sites and b_side_sites overlap"
            ));
        }
        if assigned_site_indices != all_site_indices {
            return Err(anyhow::anyhow!(
                "swap_on_edge: scheduled site partition does not match current edge sites"
            ));
        }

        let tensor_ab = T::contract(&[&tensor_a, &tensor_b]).context("swap_on_edge: contract")?;

        let ab_indices = tensor_ab.external_indices();
        let left_inds: Vec<T::Index> = ab_indices
            .iter()
            .filter(|i| other_bonds_a.contains(*i) || a_side_sites.contains(*i))
            .cloned()
            .collect();

        let result = factorize_allowing_empty_side(&tensor_ab, &left_inds, |tensor, left| {
            tensor
                .factorize(left, factorize_options)
                .map_err(|e| anyhow::anyhow!("factorize: {}", e))
        })
        .context("swap_on_edge: factorize")?;

        self.replace_edge_bond(edge, result.bond_index)
            .context("swap_on_edge: replace_edge_bond")?;
        self.replace_tensor(node_a_idx, result.left)
            .context("swap_on_edge: replace tensor_a")?;
        self.replace_tensor(node_b_idx, result.right)
            .context("swap_on_edge: replace tensor_b")?;
        self.set_edge_ortho_towards(edge, Some(node_b_name))
            .context("swap_on_edge: set_edge_ortho_towards")?;

        Ok(())
    }

    /// Reorder site indices so that each full index ends up at the target node.
    ///
    /// Builds a pre-computed schedule from the topology plus current and target
    /// site assignments, canonicalizes the network to the schedule root, then
    /// executes the scheduled transport and swap steps. Partial assignment is
    /// supported: indices not listed in `target_assignment` stay on their
    /// current side of every visited edge.
    ///
    /// # Arguments
    /// * `target_assignment` - Map from full site index to target node name.
    /// * `options` - Truncation options for each SVD (default: no truncation, exact).
    ///
    /// # Errors
    /// Returns an error when the index permutation is invalid (an invalid-index
    /// or shape mismatch failure).
    ///
    pub fn swap_site_indices(
        &mut self,
        target_assignment: &HashMap<T::Index, V>,
        options: &swap::SwapOptions,
    ) -> std::result::Result<(), TreeTNOperationError>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        T::Index: Hash + Eq,
        V: Ord,
    {
        if target_assignment.is_empty() {
            return Ok(());
        }

        let current = swap::current_site_assignment(self);
        let root = self
            .node_names()
            .into_iter()
            .min()
            .ok_or_else(|| anyhow::anyhow!("swap_site_indices: empty network"))?;
        let schedule = swap::SwapSchedule::build(
            self.site_index_network().topology(),
            &current,
            target_assignment,
            &root,
        )
        .context("swap_site_indices: build schedule")?;

        if schedule.steps.is_empty() {
            return Ok(());
        }

        self.canonicalize_mut(
            std::iter::once(schedule.root.clone()),
            crate::options::CanonicalizationOptions::default(),
        )
        .context("swap_site_indices: canonicalize")?;

        // `rtol: None` must mean "no truncation", so always set an explicit
        // policy: falling back to the process-global SVD default would silently
        // drop singular values the caller never asked to drop.
        let mut swap_factorize_options = FactorizeOptions::svd()
            .with_canonical(Canonical::Left)
            .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(
                options.rtol.unwrap_or(0.0),
            ));
        if let Some(mr) = options.max_bond_dim {
            swap_factorize_options = swap_factorize_options.with_max_bond_dim(mr);
        }

        for step in &schedule.steps {
            for edge in step.transport_path.windows(2) {
                let src_name = &edge[0];
                let dst_name = &edge[1];
                let src_idx = self.node_index(src_name).ok_or_else(|| {
                    anyhow::anyhow!("swap_site_indices: transport node {:?} not found", src_name)
                })?;
                let dst_idx = self.node_index(dst_name).ok_or_else(|| {
                    anyhow::anyhow!("swap_site_indices: transport node {:?} not found", dst_name)
                })?;
                // Transporting the orthogonality center is an exact rewrite, so
                // use the full-rank sweep that canonicalization uses instead of
                // a truncating factorization.
                self.sweep_edge_full_rank(
                    src_idx,
                    dst_idx,
                    FactorizeAlg::QR,
                    Canonical::Left,
                    "swap_transport",
                )
                .context("swap_site_indices: transport")?;
            }

            let a_idx = self.node_index(&step.node_a).ok_or_else(|| {
                anyhow::anyhow!("swap_site_indices: node {:?} not found", step.node_a)
            })?;
            let b_idx = self.node_index(&step.node_b).ok_or_else(|| {
                anyhow::anyhow!("swap_site_indices: node {:?} not found", step.node_b)
            })?;
            self.swap_on_edge(
                a_idx,
                b_idx,
                &step.a_side_sites,
                &step.b_side_sites,
                &swap_factorize_options,
            )
            .context("swap_site_indices: swap_on_edge")?;
            self.set_canonical_region([step.node_b.clone()])
                .context("swap_site_indices: set_canonical_region")?;
        }

        Ok(())
    }

    /// Reorder site indices so that each index ends up at the target node.
    ///
    /// Alias for [`swap_site_indices`](Self::swap_site_indices).
    ///
    /// # Errors
    /// Returns an error when a target index is not found (a missing-index
    /// failure) or the permutation is invalid.
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// use tensor4all_core::{DynIndex, IndexLike, IdxTensor};
    /// use tensor4all_treetn::{SwapOptions, TreeTN};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let node_name_a = "A".to_string();
    /// let node_name_b = "B".to_string();
    /// let idx_a = DynIndex::new_dyn(2);
    /// let idx_b = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(1);
    /// let t0 = IdxTensor::from_dense(vec![idx_a.clone(), bond.clone()], vec![1.0, 0.0])?;
    /// let t1 = IdxTensor::from_dense(vec![bond, idx_b.clone()], vec![1.0, 0.0])?;
    /// let mut treetn = TreeTN::<IdxTensor, String>::from_tensors(
    ///     vec![t0, t1],
    ///     vec![node_name_a.clone(), node_name_b.clone()],
    /// )?;
    ///
    /// let mut target = HashMap::new();
    /// target.insert(idx_a.clone(), node_name_b.clone());
    ///
    /// treetn.swap_site_indices(&target, &SwapOptions::default())?;
    ///
    /// assert_eq!(
    ///     treetn
    ///         .site_index_network()
    ///         .find_node_by_index(&idx_a)
    ///         .map(|name| name.as_str()),
    ///     Some(node_name_b.as_str())
    /// );
    /// assert!(treetn.is_canonicalized());
    /// # Ok::<(), anyhow::Error>(())
    /// # }
    /// ```
    pub fn swap_site_indices_by_index(
        &mut self,
        target_assignment: &HashMap<T::Index, V>,
        options: &swap::SwapOptions,
    ) -> std::result::Result<(), TreeTNOperationError>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        T::Index: Hash + Eq,
        V: Ord,
    {
        self.swap_site_indices(target_assignment, options)
    }

    /// Verify internal data consistency by checking structural invariants and reconstructing the TreeTN.
    ///
    /// This function performs two categories of checks:
    ///
    /// ## Structural invariants (fail-fast checks):
    /// 0a. **Connectivity**: All tensors must form a single connected component
    /// 0b. **Index sharing**: Only edge-connected (adjacent) nodes may share index IDs.
    ///     Non-adjacent nodes sharing an index ID violates tree structure assumptions.
    ///
    /// ## Reconstruction consistency:
    /// After structural checks pass, clones all tensors and node names, reconstructs
    /// a new TreeTN using `from_tensors`, and verifies:
    /// 1. **Topology**: Same nodes and edges
    /// 2. **Site space**: Same physical indices for each node
    /// 3. **Tensors**: Same tensor data at each node
    ///
    /// This is useful for debugging and testing to ensure that the internal state
    /// of a TreeTN is consistent after complex operations.
    ///
    /// # Returns
    /// `Ok(())` if the internal data is consistent, or `Err` with details about the inconsistency.
    /// # Errors
    /// Returns an error when an internal invariant is violated (a graph
    /// consistency failure).
    ///
    pub fn verify_internal_consistency(&self) -> std::result::Result<(), TreeTNOperationError>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        // Step 0a: Verify all tensors are connected (form a single connected component)
        // Use DFS to check connectivity since StableGraph doesn't support connected_components
        let num_nodes = self.graph.graph().node_count();
        if num_nodes > 1 {
            // Start DFS from any node
            if let Some(start_node) = self.graph.graph().node_indices().next() {
                // Single DFS pass (O(V + E)) counting reachable nodes to
                // verify the tree is connected.
                let mut dfs = Dfs::new(self.graph.graph(), start_node);
                let mut visited_count = 0;
                while dfs.next(self.graph.graph()).is_some() {
                    visited_count += 1;
                }
                if visited_count != num_nodes {
                    return Err(TreeTNOperationError::from(anyhow::anyhow!(
                        "TreeTN is disconnected: DFS visited {} of {} nodes. All tensors must be connected.",
                        visited_count,
                        num_nodes
                    ).context("verify_internal_consistency: graph must be connected")));
                }
            }
        }

        // Step 0b: Verify non-adjacent tensors don't share full bond indices.
        let mut index_to_nodes: HashMap<T::Index, Vec<NodeIndex>> = HashMap::new();
        for node_idx in self.graph.graph().node_indices() {
            if let Some(tensor) = self.tensor(node_idx) {
                for index in tensor.external_indices() {
                    index_to_nodes
                        .entry(index.clone())
                        .or_default()
                        .push(node_idx);
                }
            }
        }

        // Check each index - if shared by multiple nodes, they must be adjacent.
        for (index, nodes) in &index_to_nodes {
            if nodes.len() > 2 {
                // More than 2 nodes share the same index - always invalid for tree structure.
                return Err(TreeTNOperationError::from(
                    anyhow::anyhow!(
                        "Index {:?} is shared by {} nodes, but tree structure allows at most 2",
                        index,
                        nodes.len()
                    )
                    .context("verify_internal_consistency: index shared by too many nodes"),
                ));
            }
            if nodes.len() == 2 {
                // Two nodes share the index - they must be adjacent (connected by an edge)
                let node_a = nodes[0];
                let node_b = nodes[1];
                if self.graph.graph().find_edge(node_a, node_b).is_none()
                    && self.graph.graph().find_edge(node_b, node_a).is_none()
                {
                    let name_a = self.graph.node_name(node_a);
                    let name_b = self.graph.node_name(node_b);
                    return Err(TreeTNOperationError::from(
                        anyhow::anyhow!(
                            "Non-adjacent nodes {:?} and {:?} share index {:?}. \
                        Only adjacent (edge-connected) nodes may share full bond indices.",
                            name_a,
                            name_b,
                            index
                        )
                        .context("verify_internal_consistency: non-adjacent nodes share index"),
                    ));
                }
            }
        }

        // Step 1: Clone all tensors and node names
        let node_names: Vec<V> = self.node_names();
        let tensors: Vec<T> = node_names
            .iter()
            .filter_map(|name| {
                let idx = self.graph.node_index(name)?;
                self.tensor(idx).cloned()
            })
            .collect();

        if tensors.len() != node_names.len() {
            return Err(TreeTNOperationError::from(anyhow::anyhow!(
                "Internal inconsistency: {} node names but {} tensors found",
                node_names.len(),
                tensors.len()
            )));
        }

        // Step 2: Reconstruct TreeTN from scratch using from_tensors_unchecked
        // (use unchecked version to avoid infinite recursion)
        let reconstructed = TreeTN::<T, V>::from_tensors_unchecked(tensors, node_names)
            .context("verify_internal_consistency: failed to reconstruct TreeTN")?;

        // Step 3: Verify topology matches
        if !self.same_topology(&reconstructed) {
            return Err(TreeTNOperationError::from(
                anyhow::anyhow!(
                    "Internal inconsistency: topology does not match after reconstruction"
                )
                .context("verify_internal_consistency: topology mismatch"),
            ));
        }

        // Step 4: Verify site index network matches
        if !self
            .site_index_network
            .share_equivalent_site_index_network(&reconstructed.site_index_network)
        {
            return Err(TreeTNOperationError::from(
                anyhow::anyhow!(
                "Internal inconsistency: site index network does not match after reconstruction"
            )
                .context("verify_internal_consistency: site space mismatch"),
            ));
        }

        // Step 5: Verify tensor data matches at each node
        for node_name in self.node_names() {
            let idx_self = self
                .graph
                .node_index(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in original", node_name))?;
            let idx_reconstructed =
                reconstructed.graph.node_index(&node_name).ok_or_else(|| {
                    anyhow::anyhow!("Node {:?} not found in reconstructed", node_name)
                })?;

            let tensor_self = self.tensor(idx_self).ok_or_else(|| {
                anyhow::anyhow!("Tensor not found for node {:?} in original", node_name)
            })?;
            let tensor_reconstructed =
                reconstructed.tensor(idx_reconstructed).ok_or_else(|| {
                    anyhow::anyhow!("Tensor not found for node {:?} in reconstructed", node_name)
                })?;

            // Compare tensor indices (as sets, since order may differ)
            let indices_self: HashSet<_> = tensor_self.external_indices().into_iter().collect();
            let indices_reconstructed: HashSet<_> = tensor_reconstructed
                .external_indices()
                .into_iter()
                .collect();
            if indices_self != indices_reconstructed {
                return Err(TreeTNOperationError::from(
                    anyhow::anyhow!(
                        "Internal inconsistency: tensor indices differ at node {:?}",
                        node_name
                    )
                    .context("verify_internal_consistency: tensor index mismatch"),
                ));
            }

            // Compare tensor dimensions
            if tensor_self.num_external_indices() != tensor_reconstructed.num_external_indices() {
                return Err(TreeTNOperationError::from(
                    anyhow::anyhow!(
                        "Internal inconsistency: tensor dimensions differ at node {:?}: {} vs {}",
                        node_name,
                        tensor_self.num_external_indices(),
                        tensor_reconstructed.num_external_indices()
                    )
                    .context("verify_internal_consistency: tensor shape mismatch"),
                ));
            }
        }

        Ok(())
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Find common indices between two slices of indices.
pub(crate) fn common_inds<I: IndexLike>(inds_a: &[I], inds_b: &[I]) -> Vec<I> {
    inds_a
        .iter()
        .filter(|idx| inds_b.iter().any(|other| other == *idx))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use num_complex::Complex64;
    use tenferro_cpu::CpuBackend;
    use tensor4all_core::{
        DynIndex, FactorizeResult, IdxTensor, TensorConstructionLike, TensorContractionLike,
        TensorFactorizationLike, TensorIndex,
    };
    use tensor4all_tensorbackend::{CpuExecutionContext, ExecutionContext};

    #[test]
    fn from_tensors_empty_returns_empty_network() {
        let tn = TreeTN::<IdxTensor, usize>::from_tensors(Vec::new(), Vec::new()).unwrap();

        assert_eq!(tn.node_count(), 0);
        assert_eq!(tn.edge_count(), 0);
        assert!(tn.node_names().is_empty());
    }

    // ------------------------------------------------------------------------
    // Empty-side factorization
    // ------------------------------------------------------------------------

    fn cpu_context() -> ExecutionContext {
        ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
            CpuBackend::new(),
        )))
    }

    /// Build a column-major tensor, complex when `phase` is given and owned by
    /// `context` when one is given.
    fn sample_tensor(
        indices: Vec<DynIndex>,
        data: &[f64],
        phase: Option<Complex64>,
        context: Option<&ExecutionContext>,
    ) -> IdxTensor {
        match (phase, context) {
            (None, None) => IdxTensor::from_dense(indices, data.to_vec()),
            (None, Some(context)) => <IdxTensor as TensorConstructionLike>::from_dense_in(
                context,
                indices,
                data.to_vec(),
            ),
            (Some(phase), None) => IdxTensor::from_dense(
                indices,
                data.iter().map(|value| phase * *value).collect::<Vec<_>>(),
            ),
            (Some(phase), Some(context)) => <IdxTensor as TensorConstructionLike>::from_dense_in(
                context,
                indices,
                data.iter().map(|value| phase * *value).collect::<Vec<_>>(),
            ),
        }
        .unwrap()
    }

    /// Which side of a two-site split has no indices.
    #[derive(Clone, Copy, Debug)]
    enum EmptySide {
        Left,
        Right,
        Both,
    }

    /// A tensor and the left indices whose split leaves `side` empty.
    fn empty_side_case(
        side: EmptySide,
        phase: Option<Complex64>,
        context: Option<&ExecutionContext>,
    ) -> (IdxTensor, Vec<DynIndex>) {
        let (i, j) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
        let data = [0.5, -1.0, 2.0, 0.25, 1.5, -0.75];
        match side {
            EmptySide::Left => (sample_tensor(vec![i, j], &data, phase, context), Vec::new()),
            EmptySide::Right => (
                sample_tensor(vec![i.clone(), j.clone()], &data, phase, context),
                vec![j, i],
            ),
            EmptySide::Both => (
                sample_tensor(Vec::new(), &[-2.5], phase, context),
                Vec::new(),
            ),
        }
    }

    /// Copy a tensor into the default context through its host data, so
    /// tensors owned by different execution contexts can be compared.
    fn on_host(tensor: &IdxTensor) -> IdxTensor {
        let indices = tensor.external_indices();
        if tensor.is_complex() {
            IdxTensor::from_dense(indices, tensor.to_vec::<Complex64>().unwrap())
        } else {
            IdxTensor::from_dense(indices, tensor.to_vec::<f64>().unwrap())
        }
        .unwrap()
    }

    fn sorted_indices(mut indices: Vec<DynIndex>) -> Vec<DynIndex> {
        indices.sort_by_key(|index| *index.id());
        indices
    }

    /// Check the structural and numerical contract of an empty-side split.
    fn assert_empty_side_split(
        tensor: &IdxTensor,
        left_indices: &[DynIndex],
        result: &FactorizeResult<IdxTensor>,
        alg: FactorizeAlg,
        canonical: Canonical,
        label: &str,
    ) {
        let bond = &result.bond_index;
        assert_eq!(bond.dim(), 1, "{label}: bond is not dimension one");
        assert_eq!(result.rank, 1, "{label}: rank");

        let mut expected_left = left_indices.to_vec();
        expected_left.push(bond.clone());
        let mut expected_right: Vec<DynIndex> = tensor
            .external_indices()
            .into_iter()
            .filter(|index| !left_indices.contains(index))
            .collect();
        expected_right.push(bond.clone());
        assert_eq!(
            sorted_indices(result.left.external_indices()),
            sorted_indices(expected_left),
            "{label}: left factor indices"
        );
        assert_eq!(
            sorted_indices(result.right.external_indices()),
            sorted_indices(expected_right),
            "{label}: right factor indices"
        );

        let reconstructed = on_host(&result.left)
            .contract_pair(&on_host(&result.right))
            .unwrap();
        let residual = reconstructed
            .sub(&on_host(tensor))
            .unwrap()
            .maxabs()
            .unwrap();
        let scale = tensor.maxabs().unwrap();
        assert!(
            residual <= 1e-12 * scale,
            "{label}: reconstruction residual {residual}"
        );

        // With a dimension-one bond, a unitary canonical factor has unit
        // norm and an LU/CI canonical factor has a unit pivot.
        let canonical_factor = match canonical {
            Canonical::Left => &result.left,
            Canonical::Right => &result.right,
        };
        let measure = match alg {
            FactorizeAlg::SVD | FactorizeAlg::QR => canonical_factor.norm().unwrap(),
            FactorizeAlg::LU | FactorizeAlg::CI => canonical_factor.maxabs().unwrap(),
        };
        assert!(
            (measure - 1.0).abs() <= 1e-12,
            "{label}: canonical factor is not normalized ({measure})"
        );
    }

    const EMPTY_SIDES: [EmptySide; 3] = [EmptySide::Left, EmptySide::Right, EmptySide::Both];
    const PHASES: [Option<Complex64>; 2] = [None, Some(Complex64::new(0.6, -0.8))];

    #[test]
    fn empty_side_split_honours_algorithm_and_canonical_direction() {
        let variants = [
            (FactorizeAlg::SVD, Canonical::Left),
            (FactorizeAlg::SVD, Canonical::Right),
            (FactorizeAlg::QR, Canonical::Left),
            (FactorizeAlg::LU, Canonical::Left),
            (FactorizeAlg::LU, Canonical::Right),
            (FactorizeAlg::CI, Canonical::Left),
            (FactorizeAlg::CI, Canonical::Right),
        ];
        for (alg, canonical) in variants {
            for side in EMPTY_SIDES {
                for phase in PHASES {
                    let label = format!("{alg:?}/{canonical:?}, empty {side:?}, phase {phase:?}");
                    let (tensor, left_indices) = empty_side_case(side, phase, None);
                    let result =
                        factorize_allowing_empty_side(&tensor, &left_indices, |t, left| {
                            Ok(t.factorize_full_rank(left, alg, canonical)?)
                        })
                        .unwrap();
                    assert_empty_side_split(
                        &tensor,
                        &left_indices,
                        &result,
                        alg,
                        canonical,
                        &label,
                    );
                }
            }
        }
    }

    #[test]
    fn empty_side_split_with_truncating_options_keeps_unit_bond() {
        for canonical in [Canonical::Left, Canonical::Right] {
            let options = FactorizeOptions::svd()
                .with_canonical(canonical)
                .with_max_bond_dim(1);
            for side in EMPTY_SIDES {
                let label = format!("truncating SVD/{canonical:?}, empty {side:?}");
                let (tensor, left_indices) = empty_side_case(side, None, None);
                let result = factorize_allowing_empty_side(&tensor, &left_indices, |t, left| {
                    Ok(t.factorize(left, &options)?)
                })
                .unwrap();
                assert_empty_side_split(
                    &tensor,
                    &left_indices,
                    &result,
                    FactorizeAlg::SVD,
                    canonical,
                    &label,
                );
            }
        }
    }

    #[test]
    fn empty_side_split_keeps_factors_in_explicit_context() {
        let context = cpu_context();
        let variants = [
            (FactorizeAlg::SVD, Canonical::Left),
            (FactorizeAlg::SVD, Canonical::Right),
            (FactorizeAlg::QR, Canonical::Left),
        ];
        for (alg, canonical) in variants {
            for side in EMPTY_SIDES {
                for phase in PHASES {
                    let label = format!(
                        "in-context {alg:?}/{canonical:?}, empty {side:?}, phase {phase:?}"
                    );
                    let (tensor, left_indices) = empty_side_case(side, phase, Some(&context));
                    let result =
                        factorize_allowing_empty_side(&tensor, &left_indices, |t, left| {
                            Ok(t.factorize_full_rank_in(left, alg, canonical, &context)?)
                        })
                        .unwrap_or_else(|e| panic!("{label}: {e:#}"));
                    result.left.validate_context(&context).unwrap();
                    result.right.validate_context(&context).unwrap();
                    assert_empty_side_split(
                        &tensor,
                        &left_indices,
                        &result,
                        alg,
                        canonical,
                        &label,
                    );
                }
            }
        }
    }

    #[test]
    fn non_degenerate_split_is_delegated_unchanged() {
        let (i, j) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
        let tensor = sample_tensor(vec![i.clone(), j], &[1.0, 0.0, 0.0, 2.0], None, None);
        let result = factorize_allowing_empty_side(&tensor, std::slice::from_ref(&i), |t, left| {
            Ok(t.factorize_full_rank(left, FactorizeAlg::SVD, Canonical::Left)?)
        })
        .unwrap();
        assert_eq!(result.bond_index.dim(), 2);
        assert_eq!(result.singular_values, Some(vec![2.0, 1.0]));
        let reconstructed = result.left.contract_pair(&result.right).unwrap();
        assert!(reconstructed.sub(&tensor).unwrap().maxabs().unwrap() < 1e-12);
    }

    /// A left index that the tensor does not carry is rejected before
    /// `factorize` runs, for both a non-degenerate and an empty right side.
    /// The primed copy differs from a tensor index only by prime level.
    #[test]
    fn split_rejects_left_index_outside_tensor() {
        let (i, j) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
        let foreign = DynIndex::new_dyn(2);
        let primed_i = i.prime();
        let tensor = sample_tensor(
            vec![i.clone(), j.clone()],
            &[1.0, 0.0, 0.0, 2.0],
            None,
            None,
        );
        for left_indices in [
            vec![foreign.clone()],
            vec![i.clone(), foreign],
            vec![i, j, primed_i],
        ] {
            let error = factorize_allowing_empty_side(&tensor, &left_indices, |_, _| {
                panic!("factorize must not run for an invalid left index")
            })
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("is not an index of the tensor to factorize"),
                "{error:#}"
            );
        }
    }

    // ------------------------------------------------------------------------
    // Site-free nodes in sweeps
    // ------------------------------------------------------------------------

    /// `[0] -- bond(3) -- [1]`: node 0 is a site-free leaf, node 1 carries a
    /// dimension-two site. Both tensors are multiplied by `phase`, so the
    /// represented vector is `phase^2 * [14, 32]`.
    fn site_free_leaf_tree(
        phase: Option<Complex64>,
        context: Option<&ExecutionContext>,
    ) -> TreeTN<IdxTensor, usize> {
        let bond = DynIndex::new_dyn(3);
        let site = DynIndex::new_dyn(2);
        let leaf = sample_tensor(vec![bond.clone()], &[1.0, 2.0, 3.0], phase, context);
        let parent = sample_tensor(
            vec![bond, site],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            phase,
            context,
        );
        TreeTN::from_tensors(vec![leaf, parent], vec![0, 1]).unwrap()
    }

    /// Check the leaf edge has dimension one and the values are unchanged.
    fn assert_site_free_leaf_values(tree: &TreeTN<IdxTensor, usize>, phase: Option<Complex64>) {
        let edge = tree.edge_between(&0, &1).unwrap();
        assert_eq!(tree.bond_index(edge).unwrap().dim(), 1);
        let actual = on_host(&tree.to_dense().unwrap());
        let phase_squared = phase.map(|phase| phase * phase);
        let expected = sample_tensor(
            actual.external_indices(),
            &[14.0, 32.0],
            phase_squared,
            None,
        );
        let residual = actual.sub(&expected).unwrap().maxabs().unwrap();
        assert!(residual < 1e-12, "site-free leaf residual {residual}");
    }

    /// Check every tensor of `tree` belongs to `context`.
    fn assert_in_context(tree: &TreeTN<IdxTensor, usize>, context: &ExecutionContext) {
        for node in tree.node_names() {
            let index = tree.node_index(&node).unwrap();
            tree.tensor(index)
                .unwrap()
                .validate_context(context)
                .unwrap();
        }
    }

    #[test]
    fn sweep_absorbs_site_free_leaf_with_and_without_context() {
        let context = cpu_context();
        for phase in PHASES {
            for scoped in [true, false] {
                let label = format!("phase {phase:?}, scoped {scoped}");
                let mut tree = site_free_leaf_tree(phase, scoped.then_some(&context));
                let (src, dst) = (tree.node_index(&0).unwrap(), tree.node_index(&1).unwrap());
                if scoped {
                    tree.sweep_edge_full_rank_in(
                        src,
                        dst,
                        FactorizeAlg::QR,
                        Canonical::Left,
                        "test",
                        &context,
                    )
                    .unwrap_or_else(|e| panic!("{label}: {e:#}"));
                    assert_in_context(&tree, &context);
                } else {
                    tree.sweep_edge_full_rank(src, dst, FactorizeAlg::QR, Canonical::Left, "test")
                        .unwrap_or_else(|e| panic!("{label}: {e:#}"));
                }
                assert_site_free_leaf_values(&tree, phase);
                // The swept leaf keeps a unit-modulus scalar.
                let leaf = tree.tensor(src).unwrap();
                assert!((leaf.norm().unwrap() - 1.0).abs() < 1e-12, "{label}");
            }
        }
    }

    #[test]
    fn scoped_truncation_keeps_site_free_leaf_network_in_context() {
        let context = cpu_context();
        for phase in PHASES {
            for center in [0, 1] {
                let mut tree = site_free_leaf_tree(phase, Some(&context));
                tree.truncate_impl_in([center], None, Some(4), "test", &context)
                    .unwrap_or_else(|e| panic!("phase {phase:?}, center {center}: {e:#}"));
                tree.verify_internal_consistency().unwrap();
                assert_in_context(&tree, &context);
                assert_site_free_leaf_values(&tree, phase);
            }
        }
    }

    #[test]
    fn swap_on_edge_between_two_site_free_nodes_keeps_scalar() {
        // Both sides of the split are empty: the merged tensor is a scalar.
        let bond = DynIndex::new_dyn(3);
        let mut tree = TreeTN::<IdxTensor, usize>::from_tensors(
            vec![
                sample_tensor(vec![bond.clone()], &[1.0, 2.0, 3.0], None, None),
                sample_tensor(vec![bond], &[4.0, 5.0, 6.0], None, None),
            ],
            vec![0, 1],
        )
        .unwrap();
        let (a, b) = (tree.node_index(&0).unwrap(), tree.node_index(&1).unwrap());
        let no_sites = HashSet::new();
        tree.swap_on_edge(a, b, &no_sites, &no_sites, &FactorizeOptions::svd())
            .unwrap();

        tree.verify_internal_consistency().unwrap();
        assert_eq!(tree.link_dims(), vec![1]);
        assert!((tree.tensor(a).unwrap().norm().unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(
            tree.to_dense().unwrap().to_vec::<f64>().unwrap(),
            vec![32.0]
        );
    }
}
