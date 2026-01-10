//! Site Index Network (inspired by ITensorNetworks.jl's IndsNetwork)
//!
//! Provides a structure combining:
//! - **NodeNameNetwork**: Graph topology (node connections)
//! - **Site space map**: Physical indices at each node (`HashMap<NodeName, HashSet<I>>`)
//!
//! This design separates the index structure from tensor data,
//! enabling topology and site space comparison independent of tensor values.

use crate::node_name_network::{CanonicalizeEdges, NodeNameNetwork};
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableGraph};
use petgraph::Undirected;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use tensor4all_core::IndexLike;

// Re-export CanonicalizeEdges for convenience
pub use crate::node_name_network::CanonicalizeEdges as CanonicalizeEdgesType;

/// Site Index Network (inspired by ITensorNetworks.jl's IndsNetwork)
///
/// Represents the index structure of a tensor network:
/// - **Topology**: Graph structure via `NodeNameNetwork`
/// - **Site space**: Physical indices at each node via `HashMap`
///
/// This structure enables:
/// - Comparing topologies and site spaces independently of tensor data
/// - Extracting index information without accessing tensor values
/// - Validating network structure consistency
///
/// # Type Parameters
/// - `NodeName`: Node name type (must be Clone, Hash, Eq, Send, Sync, Debug)
/// - `I`: Index type (must implement `IndexLike`)
#[derive(Debug, Clone)]
pub struct SiteIndexNetwork<NodeName, I>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    I: IndexLike,
{
    /// Graph topology (node names and connections only).
    topology: NodeNameNetwork<NodeName>,
    /// Site space (physical indices) for each node.
    site_spaces: HashMap<NodeName, HashSet<I>>,
}

impl<NodeName, I> SiteIndexNetwork<NodeName, I>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    I: IndexLike,
{
    /// Create a new empty SiteIndexNetwork.
    pub fn new() -> Self {
        Self {
            topology: NodeNameNetwork::new(),
            site_spaces: HashMap::new(),
        }
    }

    /// Create a new SiteIndexNetwork with initial capacity.
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            topology: NodeNameNetwork::with_capacity(nodes, edges),
            site_spaces: HashMap::with_capacity(nodes),
        }
    }

    /// Add a node with site space (physical indices).
    ///
    /// # Arguments
    /// * `node_name` - The name of the node
    /// * `site_space` - The physical indices at this node (order doesn't matter)
    ///
    /// Returns an error if the node already exists.
    pub fn add_node(
        &mut self,
        node_name: NodeName,
        site_space: impl Into<HashSet<I>>,
    ) -> Result<NodeIndex, String> {
        let node_idx = self.topology.add_node(node_name.clone())?;
        self.site_spaces.insert(node_name, site_space.into());
        Ok(node_idx)
    }

    /// Check if a node exists.
    pub fn has_node(&self, node_name: &NodeName) -> bool {
        self.topology.has_node(node_name)
    }

    /// Get the site space (physical indices) for a node.
    pub fn site_space(&self, node_name: &NodeName) -> Option<&HashSet<I>> {
        self.site_spaces.get(node_name)
    }

    /// Get a mutable reference to the site space for a node.
    pub fn site_space_mut(&mut self, node_name: &NodeName) -> Option<&mut HashSet<I>> {
        self.site_spaces.get_mut(node_name)
    }

    /// Get the site space by NodeIndex.
    pub fn site_space_by_index(&self, node: NodeIndex) -> Option<&HashSet<I>> {
        let name = self.topology.node_name(node)?;
        self.site_spaces.get(name)
    }

    /// Add an edge between two nodes.
    ///
    /// Returns an error if either node doesn't exist.
    pub fn add_edge(&mut self, n1: &NodeName, n2: &NodeName) -> Result<EdgeIndex, String> {
        self.topology.add_edge(n1, n2)
    }

    /// Get the NodeIndex for a node name.
    pub fn node_index(&self, node_name: &NodeName) -> Option<NodeIndex> {
        self.topology.node_index(node_name)
    }

    /// Get the node name for a NodeIndex.
    pub fn node_name(&self, node: NodeIndex) -> Option<&NodeName> {
        self.topology.node_name(node)
    }

    /// Get all node names.
    pub fn node_names(&self) -> Vec<&NodeName> {
        self.topology.node_names()
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.topology.node_count()
    }

    /// Get the number of edges.
    pub fn edge_count(&self) -> usize {
        self.topology.edge_count()
    }

    /// Get a reference to the underlying topology (NodeNameNetwork).
    pub fn topology(&self) -> &NodeNameNetwork<NodeName> {
        &self.topology
    }

    /// Get all edges as pairs of node names.
    ///
    /// Returns an iterator of `(NodeName, NodeName)` pairs.
    pub fn edges(&self) -> impl Iterator<Item = (NodeName, NodeName)> + '_ {
        let graph = self.topology.graph();
        graph.edge_indices().filter_map(move |edge| {
            let (a, b) = graph.edge_endpoints(edge)?;
            let name_a = self.topology.node_name(a)?.clone();
            let name_b = self.topology.node_name(b)?.clone();
            Some((name_a, name_b))
        })
    }

    /// Get all neighbors of a node.
    ///
    /// Returns an iterator of neighbor node names.
    pub fn neighbors(&self, node_name: &NodeName) -> impl Iterator<Item = NodeName> + '_ {
        let node_idx = self.topology.node_index(node_name);
        let graph = self.topology.graph();
        let topology = &self.topology;

        node_idx
            .into_iter()
            .flat_map(move |idx| graph.neighbors(idx))
            .filter_map(move |n| topology.node_name(n).cloned())
    }

    /// Get a reference to the internal graph.
    pub fn graph(&self) -> &StableGraph<(), (), Undirected> {
        self.topology.graph()
    }

    /// Get a mutable reference to the internal graph.
    ///
    /// **Warning**: Directly modifying the internal graph can break consistency.
    pub fn graph_mut(&mut self) -> &mut StableGraph<(), (), Undirected> {
        self.topology.graph_mut()
    }

    /// Check if two SiteIndexNetworks share equivalent site index structure.
    ///
    /// Two networks are equivalent if:
    /// - Same topology (nodes and edges)
    /// - Same site space for each node
    ///
    /// This is used to verify that two TreeTNs can be added or contracted.
    pub fn share_equivalent_site_index_network(&self, other: &Self) -> bool {
        // Check topology
        if !self.topology.same_topology(&other.topology) {
            return false;
        }

        // Check site spaces
        for name in self.node_names() {
            match (self.site_space(name), other.site_space(name)) {
                (Some(self_indices), Some(other_indices)) => {
                    if self_indices != other_indices {
                        return false;
                    }
                }
                (None, None) => continue,
                _ => return false,
            }
        }

        true
    }

    // =========================================================================
    // Delegated graph algorithms (from NodeNameNetwork)
    // =========================================================================

    /// Perform a post-order DFS traversal starting from the given root node.
    pub fn post_order_dfs(&self, root: &NodeName) -> Option<Vec<NodeName>> {
        self.topology.post_order_dfs(root)
    }

    /// Perform a post-order DFS traversal starting from the given root NodeIndex.
    pub fn post_order_dfs_by_index(&self, root: NodeIndex) -> Vec<NodeIndex> {
        self.topology.post_order_dfs_by_index(root)
    }

    /// Find the shortest path between two nodes.
    pub fn path_between(&self, from: NodeIndex, to: NodeIndex) -> Option<Vec<NodeIndex>> {
        self.topology.path_between(from, to)
    }

    /// Check if a subset of nodes forms a connected subgraph.
    pub fn is_connected_subset(&self, nodes: &HashSet<NodeIndex>) -> bool {
        self.topology.is_connected_subset(nodes)
    }

    /// Compute edges to canonicalize from current state to target.
    pub fn edges_to_canonicalize(
        &self,
        current_region: Option<&HashSet<NodeIndex>>,
        target: NodeIndex,
    ) -> CanonicalizeEdges {
        self.topology.edges_to_canonicalize(current_region, target)
    }

    /// Compute edges to canonicalize from leaves to target, returning node names.
    ///
    /// This is similar to `edges_to_canonicalize(None, target)` but returns
    /// `(from_name, to_name)` pairs instead of `(NodeIndex, NodeIndex)`.
    ///
    /// See [`NodeNameNetwork::edges_to_canonicalize_by_names`] for details.
    pub fn edges_to_canonicalize_by_names(
        &self,
        target: &NodeName,
    ) -> Option<Vec<(NodeName, NodeName)>> {
        self.topology.edges_to_canonicalize_by_names(target)
    }

    /// Compute edges to canonicalize from leaves towards a connected region (multiple centers).
    ///
    /// See [`NodeNameNetwork::edges_to_canonicalize_to_region`] for details.
    pub fn edges_to_canonicalize_to_region(
        &self,
        target_region: &HashSet<NodeIndex>,
    ) -> CanonicalizeEdges {
        self.topology.edges_to_canonicalize_to_region(target_region)
    }

    /// Compute edges to canonicalize towards a region, returning node names.
    ///
    /// See [`NodeNameNetwork::edges_to_canonicalize_to_region_by_names`] for details.
    pub fn edges_to_canonicalize_to_region_by_names(
        &self,
        target_region: &HashSet<NodeName>,
    ) -> Option<Vec<(NodeName, NodeName)>> {
        self.topology
            .edges_to_canonicalize_to_region_by_names(target_region)
    }

    // =========================================================================
    // Operator/state compatibility checking
    // =========================================================================

    /// Check if an operator can act on this state (as a ket).
    ///
    /// Returns `Ok(result_network)` if the operator can act on self,
    /// where `result_network` is the SiteIndexNetwork of the output state.
    ///
    /// For an operator to act on a state:
    /// - They must have the same topology (same nodes and edges)
    /// - The operator must have indices that can contract with the state's site indices
    ///
    /// # Arguments
    /// * `operator` - The operator's SiteIndexNetwork
    ///
    /// # Returns
    /// - `Ok(SiteIndexNetwork)` - The resulting state's site index network after the operator acts
    /// - `Err(String)` - Error message if the operator cannot act on this state
    ///
    /// # Note
    /// This is a simplified version that assumes the operator's output indices
    /// have the same structure as the input (i.e., the result has the same
    /// site index structure as the original state). For more complex operators
    /// with different input/output dimensions, a more sophisticated approach
    /// would be needed.
    pub fn apply_operator_topology(&self, operator: &Self) -> Result<Self, String> {
        // Check topology match
        if !self.topology.same_topology(&operator.topology) {
            return Err(format!(
                "Operator and state have different topologies. State nodes: {:?}, Operator nodes: {:?}",
                self.node_names(),
                operator.node_names()
            ));
        }

        // For now, assume the operator preserves the site index structure
        // (output has same site indices as input). This is the common case
        // for Hamiltonians and other operators where H|ψ⟩ has the same
        // index structure as |ψ⟩.
        //
        // A more complete implementation would:
        // 1. Check that operator has compatible input indices
        // 2. Return the actual output index structure
        Ok(self.clone())
    }

    /// Check if this network has compatible site dimensions with another.
    ///
    /// Two networks have compatible site dimensions if:
    /// - Same topology (nodes and edges)
    /// - Each node has the same number of site indices
    /// - Site index dimensions match (after sorting, order doesn't matter)
    ///
    /// This is useful for checking if two states can be added or if
    /// a state matches the expected output of an operator.
    pub fn compatible_site_dimensions(&self, other: &Self) -> bool {
        // Check topology
        if !self.topology.same_topology(&other.topology) {
            return false;
        }

        // Check site dimensions for each node
        for name in self.node_names() {
            match (self.site_space(name), other.site_space(name)) {
                (Some(self_indices), Some(other_indices)) => {
                    // Check same number of indices
                    if self_indices.len() != other_indices.len() {
                        return false;
                    }

                    // Get dimensions and sort for comparison
                    // Use IndexLike::dim() to get the dimension
                    let mut self_dims: Vec<_> = self_indices.iter().map(|idx| idx.dim()).collect();
                    let mut other_dims: Vec<_> =
                        other_indices.iter().map(|idx| idx.dim()).collect();
                    self_dims.sort();
                    other_dims.sort();

                    if self_dims != other_dims {
                        return false;
                    }
                }
                (None, None) => continue,
                _ => return false,
            }
        }

        true
    }
}

impl<NodeName, I> Default for SiteIndexNetwork<NodeName, I>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    I: IndexLike,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Type alias for backwards compatibility
// ============================================================================

use tensor4all_core::index::{DynId, Index, NoSymmSpace};
use tensor4all_core::DefaultTagSet;

/// Type alias for the default SiteIndexNetwork using DynId indices.
///
/// This preserves backwards compatibility with existing code that uses
/// `SiteIndexNetwork<NodeName, Id, Symm, Tags>`.
pub type DefaultSiteIndexNetwork<NodeName> =
    SiteIndexNetwork<NodeName, Index<DynId, NoSymmSpace, DefaultTagSet>>;

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::DynIndex;

    #[test]
    fn test_site_index_network_basic() {
        let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

        let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
        let site2: HashSet<_> = [DynIndex::new_dyn(3)].into();
        net.add_node("A".to_string(), site1).unwrap();
        net.add_node("B".to_string(), site2).unwrap();

        assert_eq!(net.node_count(), 2);
        assert!(net.has_node(&"A".to_string()));

        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        assert_eq!(net.edge_count(), 1);
    }

    #[test]
    fn test_post_order_dfs_chain() {
        let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

        let empty: HashSet<DynIndex> = HashSet::new();
        net.add_node("A".to_string(), empty.clone()).unwrap();
        net.add_node("B".to_string(), empty.clone()).unwrap();
        net.add_node("C".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

        let result = net.post_order_dfs(&"B".to_string()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.last().unwrap(), "B");
        assert!(result.contains(&"A".to_string()));
        assert!(result.contains(&"C".to_string()));
    }

    #[test]
    fn test_path_between_chain() {
        let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

        let empty: HashSet<DynIndex> = HashSet::new();
        let a = net.add_node("A".to_string(), empty.clone()).unwrap();
        let b = net.add_node("B".to_string(), empty.clone()).unwrap();
        let c = net.add_node("C".to_string(), empty.clone()).unwrap();
        let d = net.add_node("D".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

        let path = net.path_between(a, d).unwrap();
        assert_eq!(path, vec![a, b, c, d]);
    }

    #[test]
    fn test_edges_to_canonicalize_full() {
        let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

        let empty: HashSet<DynIndex> = HashSet::new();
        let a = net.add_node("A".to_string(), empty.clone()).unwrap();
        let b = net.add_node("B".to_string(), empty.clone()).unwrap();
        let c = net.add_node("C".to_string(), empty.clone()).unwrap();
        let d = net.add_node("D".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

        let edges = net.edges_to_canonicalize(None, d);
        assert_eq!(edges.len(), 3);
        let edge_vec: Vec<_> = edges.iter().cloned().collect();
        assert_eq!(edge_vec, vec![(a, b), (b, c), (c, d)]);
    }

    #[test]
    fn test_is_connected_subset() {
        let mut net: SiteIndexNetwork<String, DynIndex> =
            SiteIndexNetwork::new();

        let empty: HashSet<DynIndex> = HashSet::new();
        let a = net.add_node("A".to_string(), empty.clone()).unwrap();
        let b = net.add_node("B".to_string(), empty.clone()).unwrap();
        let c = net.add_node("C".to_string(), empty.clone()).unwrap();
        let _d = net.add_node("D".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

        assert!(net.is_connected_subset(&HashSet::new()));
        assert!(net.is_connected_subset(&[a].into()));
        assert!(net.is_connected_subset(&[b, c].into()));
        assert!(!net.is_connected_subset(&[a, c].into()));
    }

    #[test]
    fn test_share_equivalent_site_index_network() {
        let mut net1: SiteIndexNetwork<String, DynIndex> =
            SiteIndexNetwork::new();
        let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
        net1.add_node("A".to_string(), site1.clone()).unwrap();
        net1.add_node("B".to_string(), HashSet::new()).unwrap();
        net1.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

        let mut net2: SiteIndexNetwork<String, DynIndex> =
            SiteIndexNetwork::new();
        net2.add_node("A".to_string(), site1.clone()).unwrap();
        net2.add_node("B".to_string(), HashSet::new()).unwrap();
        net2.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

        assert!(net1.share_equivalent_site_index_network(&net2));

        // Different site space
        let mut net3: SiteIndexNetwork<String, DynIndex> =
            SiteIndexNetwork::new();
        let site3: HashSet<_> = [DynIndex::new_dyn(5)].into();
        net3.add_node("A".to_string(), site3).unwrap();
        net3.add_node("B".to_string(), HashSet::new()).unwrap();
        net3.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

        assert!(!net1.share_equivalent_site_index_network(&net3));
    }

    #[test]
    fn test_apply_operator_topology() {
        // Create state network
        let mut state: SiteIndexNetwork<String, DynIndex> =
            SiteIndexNetwork::new();
        let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
        let site2: HashSet<_> = [DynIndex::new_dyn(3)].into();
        state.add_node("A".to_string(), site1.clone()).unwrap();
        state.add_node("B".to_string(), site2.clone()).unwrap();
        state.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

        // Create operator with same topology
        let mut operator: SiteIndexNetwork<String, DynIndex> =
            SiteIndexNetwork::new();
        // Operator has different index IDs but same dimensions
        let op_site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
        let op_site2: HashSet<_> = [DynIndex::new_dyn(3)].into();
        operator.add_node("A".to_string(), op_site1).unwrap();
        operator.add_node("B".to_string(), op_site2).unwrap();
        operator
            .add_edge(&"A".to_string(), &"B".to_string())
            .unwrap();

        // Should succeed
        let result = state.apply_operator_topology(&operator);
        assert!(result.is_ok());

        // Create operator with different topology
        let mut bad_operator: SiteIndexNetwork<String, DynIndex> =
            SiteIndexNetwork::new();
        bad_operator
            .add_node("A".to_string(), HashSet::new())
            .unwrap();
        bad_operator
            .add_node("C".to_string(), HashSet::new())
            .unwrap(); // Different node name
        bad_operator
            .add_edge(&"A".to_string(), &"C".to_string())
            .unwrap();

        // Should fail
        let result = state.apply_operator_topology(&bad_operator);
        assert!(result.is_err());
    }

    #[test]
    fn test_compatible_site_dimensions() {
        // Create two networks with same dimensions
        let mut net1: SiteIndexNetwork<String, DynIndex> =
            SiteIndexNetwork::new();
        let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
        net1.add_node("A".to_string(), site1).unwrap();
        net1.add_node("B".to_string(), HashSet::new()).unwrap();
        net1.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

        let mut net2: SiteIndexNetwork<String, DynIndex> =
            SiteIndexNetwork::new();
        // Different ID but same dimension
        let site2: HashSet<_> = [DynIndex::new_dyn(2)].into();
        net2.add_node("A".to_string(), site2).unwrap();
        net2.add_node("B".to_string(), HashSet::new()).unwrap();
        net2.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

        assert!(net1.compatible_site_dimensions(&net2));

        // Create network with different dimension
        let mut net3: SiteIndexNetwork<String, DynIndex> =
            SiteIndexNetwork::new();
        let site3: HashSet<_> = [DynIndex::new_dyn(5)].into(); // Different dim
        net3.add_node("A".to_string(), site3).unwrap();
        net3.add_node("B".to_string(), HashSet::new()).unwrap();
        net3.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

        assert!(!net1.compatible_site_dimensions(&net3));
    }
}
