//! Tree Tensor Network implementation.
//!
//! This module provides the [`TreeTN`] type, a tree-structured tensor network
//! for efficient tensor operations with canonicalization and truncation support.

// Some utility functions are WIP and not yet connected
#![allow(dead_code)]

mod addition;
mod canonicalize;
pub mod contraction;
mod decompose;
mod fit;
mod localupdate;
mod operator_impl;
mod ops;
mod swap;
mod tensor_like;
mod transform;
mod truncate;

use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{Dfs, EdgeRef};
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;

use anyhow::{Context, Result};

use crate::algorithm::CanonicalForm;
use tensor4all_core::{AllowedPairs, Canonical, FactorizeOptions, IndexLike, TensorLike};

use crate::named_graph::NamedGraph;
use crate::site_index_network::SiteIndexNetwork;

// Re-export the decomposition functions and types
pub use decompose::{factorize_tensor_to_treetn, factorize_tensor_to_treetn_with, TreeTopology};

// Re-export local update types
pub use localupdate::{
    apply_local_update_sweep, get_boundary_edges, BoundaryEdge, LocalUpdateStep,
    LocalUpdateSweepPlan, LocalUpdater, TruncateUpdater,
};

// Re-export swap types
pub use swap::{SwapOptions, SwapPlan, SwapStep, SwapUpdater};

/// Tree Tensor Network structure (inspired by ITensorNetworks.jl's TreeTensorNetwork).
///
/// Maintains a graph of tensors connected by bonds (edges).
/// Each node stores a tensor, and edges store `Connection` objects
/// that hold the bond index.
///
/// The structure uses SiteIndexNetwork to manage:
/// - **Topology**: Graph structure (which nodes connect to which)
/// - **Site Space**: Physical indices organized by node
///
/// # Type Parameters
/// - `T`: Tensor type implementing `TensorLike` (default: `TensorDynLen`)
/// - `V`: Node name type for named nodes (default: NodeIndex for backward compatibility)
///
/// # Construction
///
/// - `TreeTN::new()`: Create an empty network, then use `add_tensor()` and `connect()` to build.
/// - `TreeTN::from_tensors(tensors, node_names)`: Create from tensors with auto-connection by matching index IDs.
pub struct TreeTN<T = tensor4all_core::TensorDynLen, V = NodeIndex>
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
///
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
    /// - Connection fails (e.g., dimension mismatch)
    ///
    /// # Errors
    /// Returns an error if validation fails or connection fails.
    pub fn from_tensors(tensors: Vec<T>, node_names: Vec<V>) -> Result<Self>
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
        for (tensor, node_name) in tensors.into_iter().zip(node_names.into_iter()) {
            let node_idx = treetn.add_tensor_internal(node_name, tensor)?;
            node_indices.push(node_idx);
        }

        // Step 2: Build a map from index ID to (node_index, index) pairs in O(n) time
        // Key: index ID, Value: vector of (NodeIndex, Index) pairs
        #[allow(clippy::type_complexity)]
        let mut index_map: HashMap<
            <T::Index as IndexLike>::Id,
            Vec<(NodeIndex, T::Index)>,
        > = HashMap::new();

        for node_idx in &node_indices {
            let tensor = treetn
                .tensor(*node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_idx))?;

            for index in tensor.external_indices() {
                index_map
                    .entry(index.id().clone())
                    .or_insert_with(Vec::new)
                    .push((*node_idx, index.clone()));
            }
        }

        // Step 3: Connect nodes that share the same index ID
        // For TreeTN (tree structure), each index ID should appear in exactly 2 tensors
        for (index_id, nodes_with_index) in index_map {
            match nodes_with_index.len() {
                0 => unreachable!(),
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
                                "Failed to connect nodes {:?} and {:?} via index ID {:?}",
                                node_a, node_b, index_id
                            )
                        })?;
                }
                n => {
                    // Index appears in more than 2 tensors - this violates tree structure
                    return Err(anyhow::anyhow!(
                        "Index ID {:?} appears in {} tensors, but TreeTN requires exactly 2 (tree structure)",
                        index_id, n
                    ))
                    .context("TreeTN::from_tensors: each bond index must connect exactly 2 nodes");
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
    pub fn add_tensor(&mut self, node_name: V, tensor: T) -> Result<NodeIndex> {
        self.add_tensor_internal(node_name, tensor)
    }

    /// Add a tensor to the network using NodeIndex as the node name.
    ///
    /// This method only works when `V = NodeIndex`.
    ///
    /// Returns the NodeIndex for the newly added tensor.
    pub fn add_tensor_auto_name(&mut self, tensor: T) -> NodeIndex
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
            .expect("add_tensor_internal failed for auto-named tensor")
    }

    /// Connect two tensors via a specified pair of indices.
    ///
    /// The indices must have the same ID (Einsum mode).
    ///
    /// # Arguments
    /// * `node_a` - First node
    /// * `index_a` - Index on first node to use for connection
    /// * `node_b` - Second node
    /// * `index_b` - Index on second node to use for connection
    ///
    /// # Returns
    /// The EdgeIndex of the new connection, or an error if validation fails.
    pub fn connect(
        &mut self,
        node_a: NodeIndex,
        index_a: &T::Index,
        node_b: NodeIndex,
        index_b: &T::Index,
    ) -> Result<EdgeIndex> {
        self.connect_internal(node_a, index_a, node_b, index_b)
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
        // Extract physical indices: initially all indices are physical (no connections yet)
        let physical_indices: HashSet<T::Index> = tensor.external_indices().into_iter().collect();

        // Add to graph
        let node_idx = self
            .graph
            .add_node(node_name.clone(), tensor)
            .map_err(|e| anyhow::anyhow!(e))?;

        // Add to site_index_network
        self.site_index_network
            .add_node(node_name, physical_indices)
            .map_err(|e| anyhow::anyhow!("Failed to add node to site_index_network: {}", e))?;

        Ok(node_idx)
    }

    /// Internal method to connect two tensors.
    ///
    /// In Einsum mode, `index_a` and `index_b` must have the same ID.
    pub(crate) fn connect_internal(
        &mut self,
        node_a: NodeIndex,
        index_a: &T::Index,
        node_b: NodeIndex,
        index_b: &T::Index,
    ) -> Result<EdgeIndex> {
        // Validate that indices have the same ID (Einsum mode requirement)
        if index_a.id() != index_b.id() {
            return Err(anyhow::anyhow!(
                "Index IDs must match in Einsum mode: {:?} != {:?}",
                index_a.id(),
                index_b.id()
            ))
            .context("Failed to connect tensors");
        }

        // Validate that nodes exist
        if !self.graph.contains_node(node_a) || !self.graph.contains_node(node_b) {
            return Err(anyhow::anyhow!("One or both nodes do not exist"))
                .context("Failed to connect tensors");
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
            return Err(anyhow::anyhow!("Index not found in tensor_a"))
                .context("Failed to connect: index_a must exist in tensor_a");
        }
        if !has_index_b {
            return Err(anyhow::anyhow!("Index not found in tensor_b"))
                .context("Failed to connect: index_b must exist in tensor_b");
        }

        // Clone the bond index (same ID, use index_a)
        let bond_index = tensor_a
            .external_indices()
            .iter()
            .find(|idx| idx.same_id(index_a))
            .unwrap()
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
            .add_edge(&node_name_a, &node_name_b)
            .map_err(|e| anyhow::anyhow!("Failed to add edge to site_index_network: {}", e))?;

        // Update physical indices: remove bond index from physical indices
        if let Some(site_space_a) = self.site_index_network.site_space_mut(&node_name_a) {
            site_space_a.remove(&bond_index);
        }
        if let Some(site_space_b) = self.site_index_network.site_space_mut(&node_name_b) {
            site_space_b.remove(&bond_index);
        }

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

    /// Process one edge during a sweep operation.
    ///
    /// Factorizes the tensor at `src` node, absorbs the right factor into `dst` (parent),
    /// and updates the edge bond and ortho_towards.
    ///
    /// # Arguments
    /// * `src` - The source node to factorize (further from center)
    /// * `dst` - The destination/parent node (closer to center)
    /// * `factorize_options` - Options for factorization (algorithm, rtol, max_rank)
    /// * `context_name` - Name for error context
    ///
    /// # Returns
    /// `Ok(())` if successful, or an error if any step fails.
    pub(crate) fn sweep_edge(
        &mut self,
        src: NodeIndex,
        dst: NodeIndex,
        factorize_options: &FactorizeOptions,
        context_name: &str,
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
            .filter(|idx| idx.id() != bond_on_src.id())
            .cloned()
            .collect();

        let tensor_external_indices = tensor_src.external_indices();
        if left_inds.is_empty() || left_inds.len() == tensor_external_indices.len() {
            return Err(anyhow::anyhow!(
                "Cannot process node {:?}: need at least one left index and one right index",
                src
            ))
            .with_context(|| format!("{}: invalid tensor rank for factorization", context_name));
        }

        // Perform factorization
        let factorize_result = tensor_src
            .factorize(&left_inds, factorize_options)
            .map_err(|e| anyhow::anyhow!("Factorization failed: {}", e))
            .with_context(|| format!("{}: factorization failed", context_name))?;

        let left_tensor = factorize_result.left;
        let right_tensor = factorize_result.right;

        // Absorb right_tensor into dst
        let tensor_dst = self
            .tensor(dst)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found for dst node {:?}", dst))
            .with_context(|| format!("{}: dst tensor not found", context_name))?;

        let updated_dst_tensor = T::contract(&[tensor_dst, &right_tensor], AllowedPairs::All)
            .with_context(|| {
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

    /// Replace a tensor at the given node with a new tensor.
    ///
    /// Validates that the new tensor contains all indices used in connections
    /// to this node. Returns an error if any connection index is missing.
    ///
    /// Returns the old tensor if the node exists and validation passes.
    pub fn replace_tensor(&mut self, node: NodeIndex, new_tensor: T) -> Result<Option<T>> {
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
            return Err(anyhow::anyhow!(
                "New tensor is missing {} connection index(es): found {} out of {} required indices",
                connection_indices.len() - common.len(),
                common.len(),
                connection_indices.len()
            ))
            .context("replace_tensor: new tensor must contain all indices used in connections");
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
            .set_site_space(&node_name, new_physical_indices)
            .map_err(|e| anyhow::anyhow!("Failed to update site_index_network: {}", e))?;

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
    pub fn replace_edge_bond(&mut self, edge: EdgeIndex, new_bond_index: T::Index) -> Result<()> {
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
    /// This is analogous to ITensorMPS.jl's `sim(linkinds, M)` / `sim!(linkinds, M)`,
    /// and is mainly useful to avoid accidental index-ID collisions when combining
    /// multiple networks.
    ///
    /// Notes:
    /// - This keeps dimensions and conjugate states, but changes identities.
    /// - This updates both endpoint tensors and internal bookkeeping.
    pub fn sim_linkinds(&self) -> Result<Self>
    where
        T::Index: IndexLike,
    {
        let mut result = self.clone();
        result.sim_linkinds_mut()?;
        Ok(result)
    }

    /// Replace all link/bond indices with fresh IDs in-place.
    ///
    /// See [`Self::sim_linkinds`] for details.
    pub fn sim_linkinds_mut(&mut self) -> Result<()>
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

            // Update endpoint tensors by matching the old bond by ID.
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
                    .find(|idx| idx.id() == old_bond.id())
                    .ok_or_else(|| anyhow::anyhow!("Bond index not found in endpoint tensor"))?
                    .clone();
                let new_tensor = tensor.replaceind(&old_in_tensor, &new_bond)?;
                self.replace_tensor(node, new_tensor)?;
            }

            // Update ortho_towards key for this bond (if present), matched by ID.
            if let Some((key, dir)) = self
                .ortho_towards
                .iter()
                .find(|(k, _)| k.id() == old_bond.id())
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
    pub fn set_edge_ortho_towards(
        &mut self,
        edge: petgraph::stable_graph::EdgeIndex,
        dir: Option<V>,
    ) -> Result<()> {
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
                return Err(anyhow::anyhow!(
                    "ortho_towards node {:?} must be one of the edge endpoints",
                    node_name
                ))
                .context("set_edge_ortho_towards: invalid node");
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
    pub fn validate_tree(&self) -> Result<()> {
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
        let mut dfs = Dfs::new(g, start_node);
        while let Some(node) = dfs.next(g) {
            visited.insert(node);
        }

        if visited.len() != g.node_count() {
            return Err(anyhow::anyhow!(
                "Graph is not connected: {} nodes reachable out of {}",
                visited.len(),
                g.node_count()
            ))
            .context("validate_tree: graph must be connected");
        }

        // Check tree condition: edges = nodes - 1
        let node_count = g.node_count();
        let edge_count = g.edge_count();

        if edge_count != node_count - 1 {
            return Err(anyhow::anyhow!(
                "Graph does not satisfy tree condition: {} edges != {} nodes - 1",
                edge_count,
                node_count
            ))
            .context("validate_tree: tree must have edges = nodes - 1");
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

    /// Get all node names in the tree tensor network.
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
    pub fn set_canonical_region(&mut self, region: impl IntoIterator<Item = V>) -> Result<()> {
        let region: HashSet<V> = region.into_iter().collect();

        // Validate that all nodes exist in the graph
        for node_name in &region {
            if !self.graph.has_node(node_name) {
                return Err(anyhow::anyhow!(
                    "Node {:?} does not exist in the graph",
                    node_name
                ))
                .context("set_canonical_region: all nodes must be valid");
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
    pub fn add_to_canonical_region(&mut self, node_name: V) -> Result<()> {
        if !self.graph.has_node(&node_name) {
            return Err(anyhow::anyhow!(
                "Node {:?} does not exist in the graph",
                node_name
            ))
            .context("add_to_canonical_region: node must be valid");
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
    pub fn site_index_network_mut(&mut self) -> &mut SiteIndexNetwork<V, T::Index> {
        &mut self.site_index_network
    }

    /// Get a reference to the site space (physical indices) for a node.
    pub fn site_space(&self, node_name: &V) -> Option<&std::collections::HashSet<T::Index>> {
        self.site_index_network.site_space(node_name)
    }

    /// Get a mutable reference to the site space (physical indices) for a node.
    pub fn site_space_mut(
        &mut self,
        node_name: &V,
    ) -> Option<&mut std::collections::HashSet<T::Index>> {
        self.site_index_network.site_space_mut(node_name)
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
    /// Contracts the two tensors, decides which indices go to which side
    /// using `oracle` and `target_assignment`, then factorizes back in-place.
    /// No subtree extraction — operates directly on the full TreeTN.
    ///
    /// Degenerate cases:
    /// - `left_inds` empty (Case 1): prefer a site from B as placeholder so
    ///   A's sites can migrate to B.
    /// - `left_inds` = all indices (Case 2): remove last site from left_inds
    ///   (temporarily places it on B so factorize is valid).
    pub(crate) fn swap_on_edge(
        &mut self,
        node_a_idx: NodeIndex,
        node_b_idx: NodeIndex,
        target_assignment: &HashMap<<T::Index as IndexLike>::Id, V>,
        oracle: &swap::SubtreeOracle<V>,
        factorize_options: &FactorizeOptions,
    ) -> Result<()>
    where
        <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        V: Ord,
    {
        let node_a_name = self
            .graph
            .node_name(node_a_idx)
            .ok_or_else(|| anyhow::anyhow!("swap_on_edge: node_a not found"))?
            .clone();
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

        // Structural bond ids of A and B (bonds other than bond_ab)
        let other_bond_ids_a: HashSet<<T::Index as IndexLike>::Id> = self
            .edges_for_node(node_a_idx)
            .iter()
            .filter_map(|(e, _)| self.bond_index(*e).cloned())
            .filter(|b| b.id() != bond_ab.id())
            .map(|b| b.id().to_owned())
            .collect();
        let other_bond_ids_b: HashSet<<T::Index as IndexLike>::Id> = self
            .edges_for_node(node_b_idx)
            .iter()
            .filter_map(|(e, _)| self.bond_index(*e).cloned())
            .filter(|b| b.id() != bond_ab.id())
            .map(|b| b.id().to_owned())
            .collect();

        let tensor_a = self
            .tensor(node_a_idx)
            .ok_or_else(|| anyhow::anyhow!("swap_on_edge: tensor_a not found"))?
            .clone();
        let tensor_b = self
            .tensor(node_b_idx)
            .ok_or_else(|| anyhow::anyhow!("swap_on_edge: tensor_b not found"))?
            .clone();

        // Site ids currently at each node (all non-bond indices)
        let site_ids_a: HashSet<<T::Index as IndexLike>::Id> = tensor_a
            .external_indices()
            .iter()
            .filter(|i| i.id() != bond_ab.id() && !other_bond_ids_a.contains(i.id()))
            .map(|i| i.id().to_owned())
            .collect();
        let site_ids_b: HashSet<<T::Index as IndexLike>::Id> = tensor_b
            .external_indices()
            .iter()
            .filter(|i| i.id() != bond_ab.id() && !other_bond_ids_b.contains(i.id()))
            .map(|i| i.id().to_owned())
            .collect();

        let tensor_ab = T::contract(&[&tensor_a, &tensor_b], AllowedPairs::All)
            .context("swap_on_edge: contract")?;

        let ab_indices = tensor_ab.external_indices();

        // Build left_inds with site-count preservation.
        //
        // Invariant: after factorize, A keeps exactly |site_ids_a| site indices
        // and B keeps exactly |site_ids_b| site indices.  This prevents sites from
        // accumulating on a single node across successive swap_on_edge calls.
        //
        // Priority for assigning a site to A-side (lower = preferred for A):
        //   1: originally on A, target is A-side  (keep in place — best)
        //   2: originally on B, target is A-side  (genuine swap — improves placement)
        //   3: originally on B, no target          (indifferent; can come to A, frees B-slot)
        //   4: originally on A, no target          (indifferent; stays on A)
        //   5: originally on A, target is B-side   (stays on A temporarily; blocked)
        //   6: originally on B, target is B-side   (worst; forced onto A only if unavoidable)
        let target_a_site_count = site_ids_a.len();
        let mut site_candidates: Vec<(T::Index, u8)> = Vec::new();
        for idx in &ab_indices {
            let id = idx.id();
            if site_ids_a.contains(id) {
                let priority = match target_assignment.get(id) {
                    None => 4,
                    Some(t) => {
                        if oracle.is_target_on_a_side(&node_a_name, &node_b_name, t) {
                            1
                        } else {
                            5
                        }
                    }
                };
                site_candidates.push((idx.clone(), priority));
            } else if site_ids_b.contains(id) {
                let priority = match target_assignment.get(id) {
                    None => 3,
                    Some(t) => {
                        if oracle.is_target_on_a_side(&node_a_name, &node_b_name, t) {
                            2
                        } else {
                            6
                        }
                    }
                };
                site_candidates.push((idx.clone(), priority));
            }
        }
        site_candidates.sort_by_key(|(_, p)| *p);
        let site_ids_for_a: HashSet<<T::Index as IndexLike>::Id> = site_candidates
            .iter()
            .take(target_a_site_count)
            .map(|(idx, _)| idx.id().clone())
            .collect();

        // left_inds = structural bonds of A + sites chosen for A
        let left_inds: Vec<T::Index> = ab_indices
            .iter()
            .filter(|i| other_bond_ids_a.contains(i.id()) || site_ids_for_a.contains(i.id()))
            .cloned()
            .collect();

        let result = tensor_ab
            .factorize(&left_inds, factorize_options)
            .map_err(|e| anyhow::anyhow!("swap_on_edge: factorize failed: {}", e))?;

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

    /// Reorder site indices so that each index id ends up at the target node.
    ///
    /// Uses a 2-site sweep: at each edge, indices that should move to the other node
    /// are moved via contract + SVD factorize. Partial assignment is supported:
    /// indices not in `target_assignment` stay in place.
    ///
    /// # Arguments
    /// * `target_assignment` - Map from site index id to target node name.
    /// * `options` - Truncation options for each SVD (default: no truncation, exact).
    ///
    /// # Errors
    /// Returns an error if target nodes are missing, an index id is unknown, or sweep fails.
    pub fn swap_site_indices(
        &mut self,
        target_assignment: &HashMap<<T::Index as IndexLike>::Id, V>,
        options: &swap::SwapOptions,
    ) -> Result<()>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        V: Ord,
    {
        if target_assignment.is_empty() {
            return Ok(());
        }

        // Validate: all target nodes exist and all index ids are in the network.
        let current = swap::current_site_assignment(self);
        let _plan = swap::SwapPlan::<V, T::Index>::new(
            &current,
            target_assignment,
            self.site_index_network().topology(),
        )
        .context("swap_site_indices: invalid target assignment")?;

        if !self.is_canonicalized() {
            let root = self
                .node_names()
                .into_iter()
                .min()
                .ok_or_else(|| anyhow::anyhow!("swap_site_indices: empty network"))?;
            self.canonicalize_mut(
                std::iter::once(root.clone()),
                crate::options::CanonicalizationOptions::default(),
            )
            .context("swap_site_indices: canonicalize")?;
        }

        // Build oracle and sweep plan once (topology does not change during swaps).
        let root = self
            .node_names()
            .into_iter()
            .min()
            .ok_or_else(|| anyhow::anyhow!("swap_site_indices: empty network"))?;
        let oracle = swap::SubtreeOracle::new(self.site_index_network().topology(), &root)
            .context("swap_site_indices: build oracle")?;
        let sweep_plan = LocalUpdateSweepPlan::from_treetn(self, &root, 2)
            .ok_or_else(|| anyhow::anyhow!("swap_site_indices: failed to build sweep plan"))?;

        let mut factorize_options = FactorizeOptions::svd().with_canonical(Canonical::Left);
        if let Some(mr) = options.max_rank {
            factorize_options = factorize_options.with_max_rank(mr);
        }
        if let Some(rtol) = options.rtol {
            factorize_options = factorize_options.with_rtol(rtol);
        }

        let is_satisfied = |treetn: &Self| {
            target_assignment.iter().all(|(idx_id, target_node)| {
                treetn
                    .site_space(target_node)
                    .is_some_and(|ss| ss.iter().any(|i| i.id() == idx_id))
            })
        };

        let max_passes = self.node_count().max(4);
        for _pass in 0..max_passes {
            if is_satisfied(self) {
                return Ok(());
            }
            for step in sweep_plan.iter() {
                if step.nodes.len() != 2 {
                    continue;
                }
                let node_a = &step.nodes[0];
                let node_b = &step.nodes[1];
                let a_idx = self.node_index(node_a).ok_or_else(|| {
                    anyhow::anyhow!("swap_site_indices: node {:?} not found", node_a)
                })?;
                let b_idx = self.node_index(node_b).ok_or_else(|| {
                    anyhow::anyhow!("swap_site_indices: node {:?} not found", node_b)
                })?;
                self.swap_on_edge(a_idx, b_idx, target_assignment, &oracle, &factorize_options)
                    .context("swap_site_indices: swap_on_edge")?;
                self.set_canonical_region([step.new_center.clone()])
                    .context("swap_site_indices: set_canonical_region")?;
            }
        }

        if is_satisfied(self) {
            return Ok(());
        }
        Err(anyhow::anyhow!(
            "swap_site_indices: did not converge within {} passes",
            max_passes
        ))
        .context("swap_site_indices: incomplete assignment")
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
    pub fn verify_internal_consistency(&self) -> Result<()>
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
                let mut dfs = Dfs::new(self.graph.graph(), start_node);
                let mut visited_count = 0;
                while dfs.next(self.graph.graph()).is_some() {
                    visited_count += 1;
                }
                if visited_count != num_nodes {
                    return Err(anyhow::anyhow!(
                        "TreeTN is disconnected: DFS visited {} of {} nodes. All tensors must be connected.",
                        visited_count,
                        num_nodes
                    ))
                    .context("verify_internal_consistency: graph must be connected");
                }
            }
        }

        // Step 0b: Verify non-adjacent tensors don't share index IDs
        // Build a map from index ID to nodes that have that index
        let mut index_id_to_nodes: HashMap<<T::Index as IndexLike>::Id, Vec<NodeIndex>> =
            HashMap::new();
        for node_idx in self.graph.graph().node_indices() {
            if let Some(tensor) = self.tensor(node_idx) {
                for index in tensor.external_indices() {
                    index_id_to_nodes
                        .entry(index.id().clone())
                        .or_default()
                        .push(node_idx);
                }
            }
        }

        // Check each index ID - if shared by multiple nodes, they must be adjacent
        for (index_id, nodes) in &index_id_to_nodes {
            if nodes.len() > 2 {
                // More than 2 nodes share the same index ID - always invalid for tree structure
                return Err(anyhow::anyhow!(
                    "Index ID {:?} is shared by {} nodes, but tree structure allows at most 2",
                    index_id,
                    nodes.len()
                ))
                .context("verify_internal_consistency: index ID shared by too many nodes");
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
                    return Err(anyhow::anyhow!(
                        "Non-adjacent nodes {:?} and {:?} share index ID {:?}. \
                        Only adjacent (edge-connected) nodes may share index IDs.",
                        name_a,
                        name_b,
                        index_id
                    ))
                    .context("verify_internal_consistency: non-adjacent nodes share index ID");
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
            return Err(anyhow::anyhow!(
                "Internal inconsistency: {} node names but {} tensors found",
                node_names.len(),
                tensors.len()
            ));
        }

        // Step 2: Reconstruct TreeTN from scratch using from_tensors_unchecked
        // (use unchecked version to avoid infinite recursion)
        let reconstructed = TreeTN::<T, V>::from_tensors_unchecked(tensors, node_names)
            .context("verify_internal_consistency: failed to reconstruct TreeTN")?;

        // Step 3: Verify topology matches
        if !self.same_topology(&reconstructed) {
            return Err(anyhow::anyhow!(
                "Internal inconsistency: topology does not match after reconstruction"
            ))
            .context("verify_internal_consistency: topology mismatch");
        }

        // Step 4: Verify site index network matches
        if !self
            .site_index_network
            .share_equivalent_site_index_network(&reconstructed.site_index_network)
        {
            return Err(anyhow::anyhow!(
                "Internal inconsistency: site index network does not match after reconstruction"
            ))
            .context("verify_internal_consistency: site space mismatch");
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
                return Err(anyhow::anyhow!(
                    "Internal inconsistency: tensor indices differ at node {:?}",
                    node_name
                ))
                .context("verify_internal_consistency: tensor index mismatch");
            }

            // Compare tensor dimensions
            if tensor_self.num_external_indices() != tensor_reconstructed.num_external_indices() {
                return Err(anyhow::anyhow!(
                    "Internal inconsistency: tensor dimensions differ at node {:?}: {} vs {}",
                    node_name,
                    tensor_self.num_external_indices(),
                    tensor_reconstructed.num_external_indices()
                ))
                .context("verify_internal_consistency: tensor dimension mismatch");
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
    let set_b: HashSet<_> = inds_b.iter().map(|idx| idx.id()).collect();
    inds_a
        .iter()
        .filter(|idx| set_b.contains(idx.id()))
        .cloned()
        .collect()
}

/// Compute strides for row-major (C-order) indexing.
pub(crate) fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Convert linear index to multi-index.
pub(crate) fn linear_to_multi_index(
    mut linear: usize,
    strides: &[usize],
    rank: usize,
) -> Vec<usize> {
    let mut multi = vec![0; rank];
    for i in 0..rank {
        multi[i] = linear / strides[i];
        linear %= strides[i];
    }
    multi
}

/// Convert multi-index to linear index.
pub(crate) fn multi_to_linear_index(multi: &[usize], strides: &[usize]) -> usize {
    multi.iter().zip(strides.iter()).map(|(&m, &s)| m * s).sum()
}
