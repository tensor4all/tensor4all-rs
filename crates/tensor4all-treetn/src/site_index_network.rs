//! Site Index Network (inspired by ITensorNetworks.jl's IndsNetwork)
//!
//! Provides a graph structure where:
//! - Nodes store site space (physical indices): `HashSet<Index>`
//! - Edges represent connections (no data stored on edges)
//!
//! This design separates the index structure from tensor data,
//! enabling topology and site space comparison independent of tensor values.
//!
//! Note: Link space (bond indices) is not stored here, as it can be derived
//! from tensor data or is already stored in Connection objects in TreeTN.

use crate::named_graph::NamedGraph;
use petgraph::algo::astar;
use petgraph::stable_graph::{StableGraph, NodeIndex, EdgeIndex};
use petgraph::visit::DfsPostOrder;
use petgraph::Undirected;
use tensor4all::index::{Index, NoSymmSpace, Symmetry};
use tensor4all::DefaultTagSet;
use std::collections::HashSet;
use std::hash::Hash;
use std::fmt::Debug;

/// Ordered sequence of directed edges for canonicalization.
///
/// Each edge is `(from, to)` where:
/// - `from` is the node being orthogonalized away from
/// - `to` is the direction towards the orthogonality center
///
/// # Note on ordering
/// - For path-based canonicalization (moving ortho center), edges are connected:
///   each edge's `to` equals the next edge's `from`.
/// - For full canonicalization (from scratch), edges represent parent edges in
///   post-order DFS traversal, which may not be connected as a path but
///   guarantees correct processing order (children before parents).
///
/// # Example
/// For a chain A - B - C - D (canonizing towards D):
/// ```text
/// edges = [(A, B), (B, C), (C, D)]
/// ```
///
/// For a star with center C (canonizing towards C):
/// ```text
///     A
///     |
/// B - C - D
///     |
///     E
///
/// edges = [(A, C), (B, C), (D, C), (E, C)]  (order depends on DFS)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizeEdges {
    edges: Vec<(NodeIndex, NodeIndex)>,
}

impl CanonicalizeEdges {
    /// Create an empty edge sequence (no-op canonicalization).
    pub fn empty() -> Self {
        Self { edges: Vec::new() }
    }

    /// Create from a list of edges.
    ///
    /// Note: For path-based canonicalization, edges should be connected (each edge's `to`
    /// equals next edge's `from`). For full canonicalization, edges may not be connected
    /// but must be in correct processing order.
    pub fn from_edges(edges: Vec<(NodeIndex, NodeIndex)>) -> Self {
        Self { edges }
    }

    /// Check if empty (already at target, no work needed).
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Number of edges to process.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Iterate over edges in order.
    pub fn iter(&self) -> impl Iterator<Item = &(NodeIndex, NodeIndex)> {
        self.edges.iter()
    }

    /// Get the final target node (orthogonality center).
    ///
    /// Returns `None` if empty.
    pub fn target(&self) -> Option<NodeIndex> {
        self.edges.last().map(|(_, to)| *to)
    }

    /// Get the starting node (first node to be factorized).
    ///
    /// Returns `None` if empty.
    pub fn start(&self) -> Option<NodeIndex> {
        self.edges.first().map(|(from, _)| *from)
    }
}

impl IntoIterator for CanonicalizeEdges {
    type Item = (NodeIndex, NodeIndex);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.edges.into_iter()
    }
}

impl<'a> IntoIterator for &'a CanonicalizeEdges {
    type Item = &'a (NodeIndex, NodeIndex);
    type IntoIter = std::slice::Iter<'a, (NodeIndex, NodeIndex)>;

    fn into_iter(self) -> Self::IntoIter {
        self.edges.iter()
    }
}

/// Site Index Network (inspired by ITensorNetworks.jl's IndsNetwork)
///
/// Represents the index structure of a tensor network:
/// - **Site space** (node data): Physical indices at each node
/// - **Edges**: Graph structure only (no edge data)
///
/// This structure enables:
/// - Comparing topologies and site spaces independently of tensor data
/// - Extracting index information without accessing tensor values
/// - Validating network structure consistency
///
/// # Type Parameters
/// - `NodeName`: Node name type (must be Clone, Hash, Eq, Send, Sync, Debug)
/// - `Id`: Index ID type
/// - `Symm`: Symmetry type
/// - `Tags`: Tag type (default: DefaultTagSet)
#[derive(Debug, Clone)]
pub struct SiteIndexNetwork<NodeName, Id, Symm = NoSymmSpace, Tags = DefaultTagSet>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    Id: Clone + Hash + Eq,
    Symm: Clone + Symmetry,
    Tags: Clone,
{
    /// Named graph: maps node names to site space.
    ///
    /// Type parameters:
    /// - 1st (NodeName): Node name type
    /// - 2nd (NodeData): Site space (physical indices) - `HashSet<Index>` stored at each node
    /// - 3rd (EdgeData): `()` - No edge data (edges represent connections only)
    graph: NamedGraph<NodeName, HashSet<Index<Id, Symm, Tags>>, ()>,
}

impl<NodeName, Id, Symm, Tags> SiteIndexNetwork<NodeName, Id, Symm, Tags>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    Id: Clone + Hash + Eq,
    Symm: Clone + Symmetry,
    Tags: Clone,
{
    /// Create a new empty SiteIndexNetwork.
    pub fn new() -> Self {
        Self {
            graph: NamedGraph::new(),
        }
    }

    /// Create a new SiteIndexNetwork with initial capacity.
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: NamedGraph::with_capacity(nodes, edges),
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
        site_space: impl Into<HashSet<Index<Id, Symm, Tags>>>,
    ) -> Result<NodeIndex, String> {
        self.graph.add_node(node_name, site_space.into())
    }

    /// Check if a node exists.
    pub fn has_node(&self, node_name: &NodeName) -> bool {
        self.graph.has_node(node_name)
    }

    /// Get the site space (physical indices) for a node.
    pub fn site_space(&self, node_name: &NodeName) -> Option<&HashSet<Index<Id, Symm, Tags>>> {
        self.graph.node_data(node_name)
    }

    /// Get a mutable reference to the site space for a node.
    pub fn site_space_mut(&mut self, node_name: &NodeName) -> Option<&mut HashSet<Index<Id, Symm, Tags>>> {
        self.graph.node_data_mut(node_name)
    }

    /// Get the site space by NodeIndex.
    pub fn site_space_by_index(&self, node: NodeIndex) -> Option<&HashSet<Index<Id, Symm, Tags>>> {
        self.graph.node_weight(node)
    }

    /// Add an edge between two nodes (no edge data).
    ///
    /// # Arguments
    /// * `n1` - First node name
    /// * `n2` - Second node name
    ///
    /// Returns an error if either node doesn't exist.
    pub fn add_edge(&mut self, n1: &NodeName, n2: &NodeName) -> Result<EdgeIndex, String> {
        self.graph.add_edge(n1, n2, ())
    }

    /// Get the NodeIndex for a node name.
    pub fn node_index(&self, node_name: &NodeName) -> Option<NodeIndex> {
        self.graph.node_index(node_name)
    }

    /// Get the node name for a NodeIndex.
    pub fn node_name(&self, node: NodeIndex) -> Option<&NodeName> {
        self.graph.node_name(node)
    }

    /// Get all node names.
    pub fn node_names(&self) -> Vec<&NodeName> {
        self.graph.node_names()
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get a reference to the internal graph.
    ///
    /// This allows direct access to petgraph algorithms.
    pub fn graph(&self) -> &StableGraph<HashSet<Index<Id, Symm, Tags>>, (), Undirected> {
        self.graph.graph()
    }

    /// Get a mutable reference to the internal graph.
    ///
    /// **Warning**: Directly modifying the internal graph can break the node-name-to-index mapping.
    pub fn graph_mut(&mut self) -> &mut StableGraph<HashSet<Index<Id, Symm, Tags>>, (), Undirected> {
        self.graph.graph_mut()
    }

    /// Check if two SiteIndexNetworks have compatible topology and site space.
    ///
    /// Two networks are compatible if:
    /// - Same number of nodes and edges
    /// - Same node names
    /// - Same site space for each node (HashSet equality, order doesn't matter)
    pub fn is_compatible(&self, other: &Self) -> bool {
        // Check node count
        if self.node_count() != other.node_count() {
            return false;
        }

        // Check edge count
        if self.edge_count() != other.edge_count() {
            return false;
        }

        // Check all nodes have same site space (HashSet comparison, order-independent)
        for node_name in self.node_names() {
            let self_site = self.site_space(node_name);
            let other_site = other.site_space(node_name);

            match (self_site, other_site) {
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

    /// Perform a post-order DFS traversal starting from the given root node.
    ///
    /// Returns node names in post-order (children before parents, leaves first).
    /// Uses petgraph's `DfsPostOrder` internally.
    ///
    /// # Arguments
    /// * `root` - The node name to start traversal from
    ///
    /// # Returns
    /// `Some(Vec<NodeName>)` with nodes in post-order, or `None` if root doesn't exist.
    ///
    /// # Example
    /// For a tree: A - B - C (where B is root)
    /// Post-order from B: [A, C, B] or [C, A, B] (neighbors visited in arbitrary order)
    pub fn post_order_dfs(&self, root: &NodeName) -> Option<Vec<NodeName>> {
        let root_idx = self.graph.node_index(root)?;
        let g = self.graph.graph();

        let mut dfs = DfsPostOrder::new(g, root_idx);
        let mut result = Vec::new();

        while let Some(node_idx) = dfs.next(g) {
            if let Some(name) = self.graph.node_name(node_idx) {
                result.push(name.clone());
            }
        }

        Some(result)
    }

    /// Perform a post-order DFS traversal starting from the given root NodeIndex.
    ///
    /// Returns NodeIndex in post-order (children before parents, leaves first).
    /// Uses petgraph's `DfsPostOrder` internally.
    ///
    /// # Arguments
    /// * `root` - The NodeIndex to start traversal from
    ///
    /// # Returns
    /// Vector of NodeIndex in post-order.
    pub fn post_order_dfs_by_index(&self, root: NodeIndex) -> Vec<NodeIndex> {
        let g = self.graph.graph();
        let mut dfs = DfsPostOrder::new(g, root);
        let mut result = Vec::new();

        while let Some(node_idx) = dfs.next(g) {
            result.push(node_idx);
        }

        result
    }

    /// Find the shortest path between two nodes using petgraph's A* algorithm.
    ///
    /// Since this is an unweighted graph (tree), we use unit edge weights.
    /// Uses petgraph's `astar` with a trivial heuristic (behaves like BFS/Dijkstra).
    ///
    /// # Arguments
    /// * `from` - Starting node
    /// * `to` - Target node
    ///
    /// # Returns
    /// `Some(Vec<NodeIndex>)` containing the path from `from` to `to` (inclusive),
    /// or `None` if no path exists (nodes are disconnected or don't exist).
    pub fn path_between(&self, from: NodeIndex, to: NodeIndex) -> Option<Vec<NodeIndex>> {
        let g = self.graph.graph();

        // Check if both nodes exist
        if g.node_weight(from).is_none() || g.node_weight(to).is_none() {
            return None;
        }

        // Same node case
        if from == to {
            return Some(vec![from]);
        }

        // Use A* with trivial heuristic (unit edge cost, zero estimate)
        astar(
            g,
            from,
            |n| n == to,
            |_| 1usize,  // Unit edge cost
            |_| 0usize,  // No heuristic (behaves like Dijkstra/BFS)
        )
        .map(|(_, path)| path)
    }

    /// Convert a node sequence to an edge sequence.
    ///
    /// For `[n1, n2, n3, ...]`, returns `[(n1, n2), (n2, n3), ...]`.
    fn nodes_to_edges(nodes: &[NodeIndex]) -> CanonicalizeEdges {
        if nodes.len() < 2 {
            return CanonicalizeEdges::empty();
        }
        let edges: Vec<_> = nodes.windows(2)
            .map(|w| (w[0], w[1]))
            .collect();
        CanonicalizeEdges::from_edges(edges)
    }

    /// Compute edges to canonicalize from current state to target.
    ///
    /// This method determines which edges need to be processed (factorized)
    /// to move the orthogonality center from the current region to the target.
    ///
    /// # Arguments
    /// * `current_region` - Current ortho region (`None` = not canonicalized)
    /// * `target` - Target node for the orthogonality center
    ///
    /// # Returns
    /// Ordered `CanonicalizeEdges` to process for canonicalization.
    ///
    /// # Cases
    /// - **Not canonicalized** (`current_region = None`): Full canonicalization using post-order DFS
    /// - **Already at target**: Returns empty (no-op)
    /// - **Move required**: Returns path from current to target
    ///
    /// # Example
    /// ```text
    /// Chain: A - B - C - D
    ///
    /// Full canonicalize to D:
    ///   edges_to_canonicalize(None, D) → [(A,B), (B,C), (C,D)]
    ///
    /// Move from B to D:
    ///   edges_to_canonicalize(Some({B}), D) → [(B,C), (C,D)]
    ///
    /// Already at D:
    ///   edges_to_canonicalize(Some({D}), D) → []
    ///
    /// Star:     A
    ///           |
    ///       B - C - D
    ///           |
    ///           E
    ///
    /// Full canonicalize to C:
    ///   edges_to_canonicalize(None, C) → [(A,C), (B,C), (D,C), (E,C)]
    ///   (order of leaves may vary, but all edges point to center C)
    /// ```
    pub fn edges_to_canonicalize(
        &self,
        current_region: Option<&HashSet<NodeIndex>>,
        target: NodeIndex,
    ) -> CanonicalizeEdges {
        match current_region {
            None => {
                // Not canonicalized: compute parent edges for each node in post-order.
                // Post-order DFS guarantees children are processed before parents,
                // so we get edges from leaves towards root in correct order.
                let post_order = self.post_order_dfs_by_index(target);
                self.compute_parent_edges(&post_order, target)
            }
            Some(current) if current.contains(&target) => {
                // Already at target: no-op
                CanonicalizeEdges::empty()
            }
            Some(current) => {
                // Move from current to target: find path
                // Use any node in current as starting point
                if let Some(&start) = current.iter().next() {
                    if let Some(path) = self.path_between(start, target) {
                        Self::nodes_to_edges(&path)
                    } else {
                        // No path found (shouldn't happen in a connected tree)
                        CanonicalizeEdges::empty()
                    }
                } else {
                    // Empty current region (shouldn't happen if is_canonicalized())
                    CanonicalizeEdges::empty()
                }
            }
        }
    }

    /// Compute parent edges for each node in the given order.
    ///
    /// For each node (except the root), finds the edge towards the root.
    /// This is used for full canonicalization where we need to process
    /// edges from leaves towards root.
    ///
    /// # Arguments
    /// * `nodes` - Nodes in processing order (typically post-order DFS)
    /// * `root` - The root node (target of canonicalization)
    ///
    /// # Returns
    /// Edges `(from, parent)` for each non-root node.
    fn compute_parent_edges(&self, nodes: &[NodeIndex], root: NodeIndex) -> CanonicalizeEdges {
        let g = self.graph.graph();
        let mut edges = Vec::with_capacity(nodes.len().saturating_sub(1));

        // Build parent map using BFS from root
        let mut parent: std::collections::HashMap<NodeIndex, NodeIndex> = std::collections::HashMap::new();
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(root);
        visited.insert(root);

        while let Some(node) = queue.pop_front() {
            for neighbor in g.neighbors(node) {
                if visited.insert(neighbor) {
                    parent.insert(neighbor, node);
                    queue.push_back(neighbor);
                }
            }
        }

        // For each node in order, add edge to its parent
        for &node in nodes {
            if node != root {
                if let Some(&p) = parent.get(&node) {
                    edges.push((node, p));
                }
            }
        }

        CanonicalizeEdges::from_edges(edges)
    }

    /// Check if a subset of nodes forms a connected subgraph.
    ///
    /// Uses DFS to verify that all nodes in the subset are reachable from each other
    /// within the induced subgraph (only following edges where both endpoints are in the subset).
    ///
    /// # Arguments
    /// * `nodes` - Set of NodeIndex to check for connectivity
    ///
    /// # Returns
    /// `true` if the subset is connected (or empty), `false` otherwise.
    ///
    /// # Example
    /// ```text
    /// Tree: A - B - C - D
    ///
    /// is_connected_subset({B, C}) → true  (adjacent)
    /// is_connected_subset({A, C}) → false (not adjacent, B not in subset)
    /// is_connected_subset({A, B, C}) → true (connected chain)
    /// ```
    pub fn is_connected_subset(&self, nodes: &HashSet<NodeIndex>) -> bool {
        if nodes.is_empty() || nodes.len() == 1 {
            return true;
        }

        let g = self.graph.graph();

        // Start DFS from any node in the subset
        let start = *nodes.iter().next().unwrap();
        let mut seen = HashSet::new();
        let mut stack = vec![start];
        seen.insert(start);

        while let Some(v) = stack.pop() {
            for nb in g.neighbors(v) {
                // Only follow edges within the subset
                if nodes.contains(&nb) && seen.insert(nb) {
                    stack.push(nb);
                }
            }
        }

        // Connected if we reached all nodes
        seen.len() == nodes.len()
    }
}

impl<NodeName, Id, Symm, Tags> Default for SiteIndexNetwork<NodeName, Id, Symm, Tags>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    Id: Clone + Hash + Eq,
    Symm: Clone + Symmetry,
    Tags: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all::index::Index;

    #[test]
    fn test_site_index_network_basic() {
        let mut net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();

        // Add nodes with site space (order doesn't matter with HashSet)
        let site1: HashSet<_> = [Index::new(1u128, NoSymmSpace::new(2))].into();
        let site2: HashSet<_> = [Index::new(2u128, NoSymmSpace::new(3))].into();
        net.add_node("A".to_string(), site1).unwrap();
        net.add_node("B".to_string(), site2).unwrap();

        assert_eq!(net.node_count(), 2);
        assert!(net.has_node(&"A".to_string()));

        // Add edge (no edge data)
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

        assert_eq!(net.edge_count(), 1);
    }

    #[test]
    fn test_post_order_dfs_chain() {
        // Create a chain: A - B - C
        let mut net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();

        let empty: HashSet<Index<u128, NoSymmSpace, DefaultTagSet>> = HashSet::new();
        net.add_node("A".to_string(), empty.clone()).unwrap();
        net.add_node("B".to_string(), empty.clone()).unwrap();
        net.add_node("C".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

        // Post-order from B: A and C are leaves, B is last
        let result = net.post_order_dfs(&"B".to_string()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.last().unwrap(), "B"); // Root is last in post-order
        // A and C come before B (order between them is not guaranteed)
        assert!(result.contains(&"A".to_string()));
        assert!(result.contains(&"C".to_string()));
    }

    #[test]
    fn test_post_order_dfs_star() {
        // Create a star: A, B, C all connected to D (center)
        let mut net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();

        let empty: HashSet<Index<u128, NoSymmSpace, DefaultTagSet>> = HashSet::new();
        net.add_node("A".to_string(), empty.clone()).unwrap();
        net.add_node("B".to_string(), empty.clone()).unwrap();
        net.add_node("C".to_string(), empty.clone()).unwrap();
        net.add_node("D".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"D".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"D".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

        // Post-order from D: A, B, C are leaves (in some order), D is last
        let result = net.post_order_dfs(&"D".to_string()).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.last().unwrap(), "D"); // Root is last
    }

    #[test]
    fn test_post_order_dfs_nonexistent_root() {
        let net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();
        assert!(net.post_order_dfs(&"X".to_string()).is_none());
    }

    #[test]
    fn test_path_between_chain() {
        // Create a chain: A - B - C - D
        let mut net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();

        let empty: HashSet<Index<u128, NoSymmSpace, DefaultTagSet>> = HashSet::new();
        let a = net.add_node("A".to_string(), empty.clone()).unwrap();
        let b = net.add_node("B".to_string(), empty.clone()).unwrap();
        let c = net.add_node("C".to_string(), empty.clone()).unwrap();
        let d = net.add_node("D".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

        // Path from A to D
        let path = net.path_between(a, d).unwrap();
        assert_eq!(path, vec![a, b, c, d]);

        // Path from D to A (reverse)
        let path_rev = net.path_between(d, a).unwrap();
        assert_eq!(path_rev, vec![d, c, b, a]);

        // Same node
        let path_same = net.path_between(b, b).unwrap();
        assert_eq!(path_same, vec![b]);
    }

    #[test]
    fn test_edges_to_canonicalize_full() {
        // Create a chain: A - B - C - D
        let mut net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();

        let empty: HashSet<Index<u128, NoSymmSpace, DefaultTagSet>> = HashSet::new();
        let a = net.add_node("A".to_string(), empty.clone()).unwrap();
        let b = net.add_node("B".to_string(), empty.clone()).unwrap();
        let c = net.add_node("C".to_string(), empty.clone()).unwrap();
        let d = net.add_node("D".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

        // Full canonicalize to D (current = None)
        let edges = net.edges_to_canonicalize(None, d);
        assert_eq!(edges.len(), 3);
        // Post-order: A, B, C, D → edges: (A,B), (B,C), (C,D)
        let edge_vec: Vec<_> = edges.iter().cloned().collect();
        assert_eq!(edge_vec, vec![(a, b), (b, c), (c, d)]);
        assert_eq!(edges.target(), Some(d));
    }

    #[test]
    fn test_edges_to_canonicalize_move() {
        // Create a chain: A - B - C - D
        let mut net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();

        let empty: HashSet<Index<u128, NoSymmSpace, DefaultTagSet>> = HashSet::new();
        let _a = net.add_node("A".to_string(), empty.clone()).unwrap();
        let b = net.add_node("B".to_string(), empty.clone()).unwrap();
        let c = net.add_node("C".to_string(), empty.clone()).unwrap();
        let d = net.add_node("D".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

        // Move from B to D
        let current: HashSet<NodeIndex> = [b].into();
        let edges = net.edges_to_canonicalize(Some(&current), d);
        assert_eq!(edges.len(), 2);
        let edge_vec: Vec<_> = edges.iter().cloned().collect();
        assert_eq!(edge_vec, vec![(b, c), (c, d)]);
    }

    #[test]
    fn test_edges_to_canonicalize_already_at_target() {
        // Create a chain: A - B - C
        let mut net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();

        let empty: HashSet<Index<u128, NoSymmSpace, DefaultTagSet>> = HashSet::new();
        let _a = net.add_node("A".to_string(), empty.clone()).unwrap();
        let b = net.add_node("B".to_string(), empty.clone()).unwrap();
        let _c = net.add_node("C".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

        // Already at B
        let current: HashSet<NodeIndex> = [b].into();
        let edges = net.edges_to_canonicalize(Some(&current), b);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_canonicalize_edges_struct() {
        let edges = CanonicalizeEdges::empty();
        assert!(edges.is_empty());
        assert_eq!(edges.len(), 0);
        assert!(edges.target().is_none());
        assert!(edges.start().is_none());

        let n1 = NodeIndex::new(0);
        let n2 = NodeIndex::new(1);
        let n3 = NodeIndex::new(2);
        let edges = CanonicalizeEdges::from_edges(vec![(n1, n2), (n2, n3)]);
        assert!(!edges.is_empty());
        assert_eq!(edges.len(), 2);
        assert_eq!(edges.start(), Some(n1));
        assert_eq!(edges.target(), Some(n3));

        // Test iteration
        let collected: Vec<_> = edges.iter().cloned().collect();
        assert_eq!(collected, vec![(n1, n2), (n2, n3)]);
    }

    #[test]
    fn test_is_connected_subset() {
        // Create a chain: A - B - C - D
        let mut net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();

        let empty: HashSet<Index<u128, NoSymmSpace, DefaultTagSet>> = HashSet::new();
        let a = net.add_node("A".to_string(), empty.clone()).unwrap();
        let b = net.add_node("B".to_string(), empty.clone()).unwrap();
        let c = net.add_node("C".to_string(), empty.clone()).unwrap();
        let d = net.add_node("D".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

        // Empty set is connected
        assert!(net.is_connected_subset(&HashSet::new()));

        // Single node is connected
        assert!(net.is_connected_subset(&[a].into()));

        // Adjacent nodes are connected
        assert!(net.is_connected_subset(&[b, c].into()));

        // Chain is connected
        assert!(net.is_connected_subset(&[a, b, c].into()));
        assert!(net.is_connected_subset(&[a, b, c, d].into()));

        // Non-adjacent nodes are NOT connected (gap in the middle)
        assert!(!net.is_connected_subset(&[a, c].into()));
        assert!(!net.is_connected_subset(&[a, d].into()));
        assert!(!net.is_connected_subset(&[a, c, d].into()));
    }

    #[test]
    fn test_edges_to_canonicalize_star() {
        // Create a star: A, B, D, E all connected to C (center)
        //     A
        //     |
        // B - C - D
        //     |
        //     E
        let mut net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();

        let empty: HashSet<Index<u128, NoSymmSpace, DefaultTagSet>> = HashSet::new();
        let a = net.add_node("A".to_string(), empty.clone()).unwrap();
        let b = net.add_node("B".to_string(), empty.clone()).unwrap();
        let c = net.add_node("C".to_string(), empty.clone()).unwrap();
        let d = net.add_node("D".to_string(), empty.clone()).unwrap();
        let e = net.add_node("E".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"E".to_string()).unwrap();

        // Full canonicalize to C (current = None)
        let edges = net.edges_to_canonicalize(None, c);

        // Should have 4 edges (one from each leaf to center)
        assert_eq!(edges.len(), 4);

        // Each edge should go TO center C
        let edge_vec: Vec<_> = edges.iter().cloned().collect();
        for (from, to) in &edge_vec {
            assert_eq!(*to, c, "All edges should point to center C");
            assert_ne!(*from, c, "No edge should start from center C");
        }

        // All leaves should be present as 'from' nodes
        let from_nodes: HashSet<_> = edge_vec.iter().map(|(from, _)| *from).collect();
        assert!(from_nodes.contains(&a), "Edge from A should exist");
        assert!(from_nodes.contains(&b), "Edge from B should exist");
        assert!(from_nodes.contains(&d), "Edge from D should exist");
        assert!(from_nodes.contains(&e), "Edge from E should exist");
    }

    #[test]
    fn test_edges_to_canonicalize_y_shaped() {
        // Create a Y-shaped tree:
        //     A
        //     |
        //     B
        //    / \
        //   C   D
        let mut net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();

        let empty: HashSet<Index<u128, NoSymmSpace, DefaultTagSet>> = HashSet::new();
        let a = net.add_node("A".to_string(), empty.clone()).unwrap();
        let b = net.add_node("B".to_string(), empty.clone()).unwrap();
        let c = net.add_node("C".to_string(), empty.clone()).unwrap();
        let d = net.add_node("D".to_string(), empty.clone()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"D".to_string()).unwrap();

        // Full canonicalize to A (current = None)
        let edges = net.edges_to_canonicalize(None, a);

        // Should have 3 edges
        assert_eq!(edges.len(), 3);

        // Verify structure: C->B, D->B, B->A (order of C,D may vary)
        let edge_vec: Vec<_> = edges.iter().cloned().collect();

        // B->A should be last (B is processed after its children C and D)
        assert_eq!(edge_vec.last(), Some(&(b, a)), "B->A should be last edge");

        // C and D should both point to B
        let first_two: HashSet<_> = edge_vec[..2].iter().cloned().collect();
        assert!(first_two.contains(&(c, b)), "C->B should be in first two edges");
        assert!(first_two.contains(&(d, b)), "D->B should be in first two edges");
    }
}

