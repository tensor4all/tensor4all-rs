//! Dynamic Tree Tensor Network with heterogeneous node types.
//!
//! This module provides `DynTreeTN`, a tree tensor network that can hold
//! heterogeneous `TensorLike` objects as node payloads. Unlike `TreeTN` which
//! stores homogeneous `TensorDynLen` tensors, `DynTreeTN` allows mixing different
//! tensor representations (dense tensors, other TTNs, custom types, etc.).
//!
//! # Type Parameters
//!
//! The associated types are fixed to canonical values for simplicity:
//! - `Id = DynId` (runtime identity)
//! - `Symm = NoSymmSpace` (no symmetry)
//! - `Tags = TagSet` (Arc-wrapped tag set)
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_treetn::DynTreeTN;
//! use tensor4all::{TensorDynLen, TensorLike};
//!
//! let mut tn = DynTreeTN::<String>::new();
//!
//! // Add a dense tensor
//! let tensor = TensorDynLen::from_indices(...);
//! tn.add_node("A".to_string(), tensor)?;
//!
//! // Add another TreeTN as a node (heterogeneous!)
//! let sub_network = TreeTN::new(tensors, node_names)?;
//! tn.add_node("B".to_string(), sub_network)?;
//! ```

use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use std::collections::HashSet;
use std::hash::Hash;
use std::fmt::Debug;

use std::collections::HashMap;
use tensor4all::index::{DynId, Index, NoSymmSpace, TagSet};
use tensor4all::{TensorDynLen, TensorLike};

use crate::named_graph::NamedGraph;
use crate::site_index_network::SiteIndexNetwork;

use anyhow::{Context, Result};

/// Type alias for the canonical Index type used in DynTreeTN.
pub type DynIndex = Index<DynId, NoSymmSpace, TagSet>;

/// Type alias for boxed TensorLike trait objects with canonical type parameters.
///
/// This is the node payload type for `DynTreeTN`.
pub type BoxedTensorLike = Box<dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = TagSet>>;

/// Dynamic Tree Tensor Network that can hold heterogeneous TensorLike objects.
///
/// Unlike `TreeTN<Id, Symm, V>` which stores homogeneous `TensorDynLen<Id, Symm>` tensors,
/// `DynTreeTN<V>` can store any object implementing `TensorLike` as node payloads.
/// This enables mixing different tensor representations within the same network.
///
/// # Type Parameters
///
/// - `V`: Node name type (default: `NodeIndex` for backward compatibility)
///
/// # Fixed Associated Types
///
/// The `TensorLike` associated types are fixed to canonical values:
/// - `Id = DynId` (runtime identity)
/// - `Symm = NoSymmSpace` (no symmetry)
/// - `Tags = TagSet` (Arc-wrapped tag set)
///
/// This ensures all nodes in the network have compatible index types.
///
/// # Heterogeneous Nodes
///
/// Each node can hold a different concrete type implementing `TensorLike`:
/// - `TensorDynLen<DynId, NoSymmSpace>` - dense tensors
/// - `TreeTN<DynId, NoSymmSpace, _>` - sub-networks
/// - Custom types implementing `TensorLike`
///
/// # Example
///
/// ```ignore
/// let mut tn = DynTreeTN::<String>::new();
///
/// // Mix different tensor types
/// tn.add_node("dense".to_string(), some_tensor)?;
/// tn.add_node("network".to_string(), some_treetn)?;
/// ```
pub struct DynTreeTN<V = NodeIndex>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Named graph: maps node names to boxed TensorLike trait objects.
    /// Edges store bond indices directly (Index<DynId, NoSymmSpace>).
    graph: NamedGraph<V, BoxedTensorLike, Index<DynId, NoSymmSpace>>,
    /// Orthogonalization region (node names).
    canonical_center: HashSet<V>,
    /// Site index network: manages topology and site space (physical indices).
    site_index_network: SiteIndexNetwork<V, DynId, NoSymmSpace, TagSet>,
    /// Orthogonalization direction for each edge (node name that ortho points towards).
    ortho_towards: HashMap<EdgeIndex, V>,
}

impl<V> DynTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Create a new empty DynTreeTN.
    pub fn new() -> Self {
        Self {
            graph: NamedGraph::new(),
            canonical_center: HashSet::new(),
            site_index_network: SiteIndexNetwork::new(),
            ortho_towards: HashMap::new(),
        }
    }

    /// Add a TensorLike object as a node in the network.
    ///
    /// The object is boxed and stored as a trait object. Its external indices
    /// become the site/physical indices for this node.
    ///
    /// # Arguments
    /// * `node_name` - Name for the node
    /// * `tensor_like` - Any object implementing `TensorLike` with the canonical type parameters
    ///
    /// # Returns
    /// The `NodeIndex` of the newly added node, or an error if the node name already exists.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tn = DynTreeTN::<String>::new();
    /// let tensor = TensorDynLen::from_indices(indices, storage);
    /// tn.add_node("A".to_string(), tensor)?;
    /// ```
    pub fn add_node<T>(&mut self, node_name: V, tensor_like: T) -> Result<NodeIndex>
    where
        T: TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = TagSet> + 'static,
    {
        // Get external indices before boxing
        let external_indices: HashSet<DynIndex> = tensor_like.external_indices().into_iter().collect();

        // Box the tensor_like object
        let boxed: BoxedTensorLike = Box::new(tensor_like);

        // Add to graph
        let node_idx = self.graph.add_node(node_name.clone(), boxed)
            .map_err(|e: String| anyhow::anyhow!(e))
            .context("Failed to add node to graph")?;

        // Add to site_index_network
        self.site_index_network.add_node(node_name, external_indices)
            .map_err(|e| anyhow::anyhow!("Failed to add node to site_index_network: {}", e))?;

        Ok(node_idx)
    }

    /// Add a pre-boxed TensorLike object as a node.
    ///
    /// This is useful when you already have a `BoxedTensorLike` or when working
    /// with trait objects directly.
    ///
    /// # Arguments
    /// * `node_name` - Name for the node
    /// * `boxed` - Boxed TensorLike trait object
    ///
    /// # Returns
    /// The `NodeIndex` of the newly added node.
    pub fn add_boxed_node(&mut self, node_name: V, boxed: BoxedTensorLike) -> Result<NodeIndex> {
        // Get external indices from the boxed object
        let external_indices: HashSet<DynIndex> = boxed.external_indices().into_iter().collect();

        // Add to graph
        let node_idx = self.graph.add_node(node_name.clone(), boxed)
            .map_err(|e: String| anyhow::anyhow!(e))
            .context("Failed to add node to graph")?;

        // Add to site_index_network
        self.site_index_network.add_node(node_name, external_indices)
            .map_err(|e| anyhow::anyhow!("Failed to add node to site_index_network: {}", e))?;

        Ok(node_idx)
    }

    /// Get a reference to a node's TensorLike object by NodeIndex.
    pub fn node(&self, node: NodeIndex) -> Option<&dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = TagSet>> {
        self.graph.graph().node_weight(node).map(|b| b.as_ref())
    }

    /// Get a reference to a node's TensorLike object by node name.
    pub fn node_by_name(&self, name: &V) -> Option<&dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = TagSet>> {
        self.graph.node_index(name)
            .and_then(|idx| self.node(idx))
    }

    /// Connect two nodes with a bond.
    ///
    /// The indices must exist in the respective nodes' external indices and have matching dimensions.
    ///
    /// # Arguments
    /// * `node_a` - First node
    /// * `index_a` - Index from node_a to use for the bond
    /// * `node_b` - Second node
    /// * `index_b` - Index from node_b to use for the bond
    ///
    /// # Returns
    /// The `EdgeIndex` of the created connection.
    pub fn connect(
        &mut self,
        node_a: NodeIndex,
        index_a: &DynIndex,
        node_b: NodeIndex,
        index_b: &DynIndex,
    ) -> Result<EdgeIndex> {
        // Validate that nodes exist
        if !self.graph.contains_node(node_a) || !self.graph.contains_node(node_b) {
            return Err(anyhow::anyhow!("One or both nodes do not exist"))
                .context("Failed to connect nodes");
        }

        // Validate that indices exist in respective nodes' external indices
        let node_a_obj = self.graph.graph().node_weight(node_a)
            .ok_or_else(|| anyhow::anyhow!("Node A not found"))?;
        let node_b_obj = self.graph.graph().node_weight(node_b)
            .ok_or_else(|| anyhow::anyhow!("Node B not found"))?;

        let ext_a = node_a_obj.external_indices();
        let ext_b = node_b_obj.external_indices();

        let has_index_a = ext_a.iter().any(|idx| index_a.id == idx.id);
        let has_index_b = ext_b.iter().any(|idx| index_b.id == idx.id);

        if !has_index_a {
            return Err(anyhow::anyhow!("Index not found in node A's external indices"))
                .context("Failed to connect: index_a must be an external index of node A");
        }
        if !has_index_b {
            return Err(anyhow::anyhow!("Index not found in node B's external indices"))
                .context("Failed to connect: index_b must be an external index of node B");
        }

        // Get node names for site_index_network (before mutable borrow)
        let node_name_a = self.graph.node_name(node_a)
            .ok_or_else(|| anyhow::anyhow!("Node name for node_a not found"))?
            .clone();
        let node_name_b = self.graph.node_name(node_b)
            .ok_or_else(|| anyhow::anyhow!("Node name for node_b not found"))?
            .clone();

        // In Einsum mode, both indices share the same ID, so store just one bond index
        // Validate that index_a and index_b have matching IDs and dimensions
        if index_a.id != index_b.id {
            return Err(anyhow::anyhow!(
                "In Einsum mode, indices must have the same ID: {:?} != {:?}",
                index_a.id, index_b.id
            )).context("Failed to connect: index mismatch");
        }
        if index_a.size() != index_b.size() {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: {} != {}",
                index_a.size(), index_b.size()
            )).context("Failed to connect: dimension mismatch");
        }

        // Store the bond index directly on the edge (converts to Index<DynId, NoSymmSpace> by dropping tags)
        let bond_index = Index::new(index_a.id, index_a.symm.clone());

        // Add edge to graph
        let edge_idx = self.graph.graph_mut().add_edge(node_a, node_b, bond_index);

        // Add edge to site_index_network
        self.site_index_network.add_edge(&node_name_a, &node_name_b)
            .map_err(|e| anyhow::anyhow!("Failed to add edge to site_index_network: {}", e))?;

        // Update physical indices: remove connection indices from site space
        if let Some(site_space_a) = self.site_index_network.site_space_mut(&node_name_a) {
            site_space_a.remove(index_a);
        }
        if let Some(site_space_b) = self.site_index_network.site_space_mut(&node_name_b) {
            site_space_b.remove(index_b);
        }

        Ok(edge_idx)
    }

    /// Get a reference to a bond index by EdgeIndex.
    pub fn bond_index(&self, edge: EdgeIndex) -> Option<&Index<DynId, NoSymmSpace>> {
        self.graph.graph().edge_weight(edge)
    }

    /// Get a mutable reference to a bond index by EdgeIndex.
    pub fn bond_index_mut(&mut self, edge: EdgeIndex) -> Option<&mut Index<DynId, NoSymmSpace>> {
        self.graph.graph_mut().edge_weight_mut(edge)
    }

    /// Get the number of nodes in the network.
    pub fn node_count(&self) -> usize {
        self.graph.graph().node_count()
    }

    /// Get the number of edges in the network.
    pub fn edge_count(&self) -> usize {
        self.graph.graph().edge_count()
    }

    /// Get all node names in the network.
    pub fn node_names(&self) -> Vec<V> {
        self.graph.graph().node_indices()
            .filter_map(|idx| self.graph.node_name(idx).cloned())
            .collect()
    }

    /// Get all node indices in the network.
    pub fn node_indices(&self) -> Vec<NodeIndex> {
        self.graph.graph().node_indices().collect()
    }

    /// Get the NodeIndex for a node name.
    pub fn node_index(&self, name: &V) -> Option<NodeIndex> {
        self.graph.node_index(name)
    }

    /// Get the node name for a NodeIndex.
    pub fn node_name(&self, node: NodeIndex) -> Option<&V> {
        self.graph.node_name(node)
    }

    /// Get a reference to the site index network.
    pub fn site_index_network(&self) -> &SiteIndexNetwork<V, DynId, NoSymmSpace, TagSet> {
        &self.site_index_network
    }

    /// Get a reference to the orthogonalization region.
    pub fn canonical_center(&self) -> &HashSet<V> {
        &self.canonical_center
    }

    /// Contract the entire network to a single tensor.
    ///
    /// Each node's `to_tensor()` is called to convert it to a `TensorDynLen`,
    /// then all tensors are contracted along their connected indices.
    ///
    /// # Returns
    /// A `TensorDynLen` representing the contracted network.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The network is empty
    /// - Any node's `to_tensor()` fails
    /// - The graph is not a tree (has cycles or is disconnected)
    pub fn contract_to_tensor(&self) -> Result<TensorDynLen<DynId, NoSymmSpace>> {
        let node_count = self.node_count();

        if node_count == 0 {
            return Err(anyhow::anyhow!("Cannot contract empty network"));
        }

        if node_count == 1 {
            // Single node: just convert to tensor
            let node = self.graph.graph().node_indices().next().unwrap();
            let obj = self.graph.graph().node_weight(node).unwrap();
            return obj.to_tensor();
        }

        // For multi-node networks, we contract pairwise
        // First, collect all tensors
        let mut tensors: Vec<TensorDynLen<DynId, NoSymmSpace>> = Vec::new();
        for node in self.graph.graph().node_indices() {
            let obj = self.graph.graph().node_weight(node).unwrap();
            let tensor = obj.to_tensor()
                .with_context(|| format!("Failed to convert node {:?} to tensor", self.graph.node_name(node)))?;
            tensors.push(tensor);
        }

        // Contract all tensors together using common indices
        let mut result = tensors.remove(0);
        for tensor in tensors {
            result = result.contract_einsum(&tensor);
        }

        Ok(result)
    }
}

impl<V> Default for DynTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<V> Clone for DynTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    fn clone(&self) -> Self {
        // Clone each boxed tensor using dyn-clone
        let mut new_graph = NamedGraph::new();

        for node in self.graph.graph().node_indices() {
            if let (Some(name), Some(boxed)) = (
                self.graph.node_name(node),
                self.graph.graph().node_weight(node),
            ) {
                let cloned_box: BoxedTensorLike = dyn_clone::clone_box(&**boxed);
                let _ = new_graph.add_node(name.clone(), cloned_box);
            }
        }

        // Clone edges
        for edge in self.graph.graph().edge_indices() {
            if let Some((src, tgt)) = self.graph.graph().edge_endpoints(edge) {
                if let Some(conn) = self.graph.graph().edge_weight(edge) {
                    new_graph.graph_mut().add_edge(src, tgt, conn.clone());
                }
            }
        }

        Self {
            graph: new_graph,
            canonical_center: self.canonical_center.clone(),
            site_index_network: self.site_index_network.clone(),
            ortho_towards: self.ortho_towards.clone(),
        }
    }
}

impl<V> Debug for DynTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynTreeTN")
            .field("node_count", &self.node_count())
            .field("edge_count", &self.edge_count())
            .field("canonical_center", &self.canonical_center)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all::storage::DenseStorageF64;
    use tensor4all::Storage;
    use std::sync::Arc;

    fn make_tensor(indices: Vec<DynIndex>) -> TensorDynLen<DynId, NoSymmSpace> {
        let total_size: usize = indices.iter().map(|idx| idx.size()).product();
        let data: Vec<f64> = (0..total_size).map(|i| i as f64).collect();
        let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));
        let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();
        // Convert DynIndex to Index<DynId, NoSymmSpace>
        let indices_no_tags: Vec<Index<DynId, NoSymmSpace>> = indices
            .iter()
            .map(|idx| Index::new(idx.id, idx.symm.clone()))
            .collect();
        TensorDynLen::new(indices_no_tags, dims, storage)
    }

    #[test]
    fn test_dyn_treetn_new() {
        let tn = DynTreeTN::<String>::new();
        assert_eq!(tn.node_count(), 0);
        assert_eq!(tn.edge_count(), 0);
    }

    #[test]
    fn test_dyn_treetn_add_node() {
        let mut tn = DynTreeTN::<String>::new();

        let i: DynIndex = Index::new_dyn(2);
        let j: DynIndex = Index::new_dyn(3);
        let tensor = make_tensor(vec![i.clone(), j.clone()]);

        let node_idx = tn.add_node("A".to_string(), tensor).unwrap();

        assert_eq!(tn.node_count(), 1);
        assert!(tn.node(node_idx).is_some());
        assert!(tn.node_by_name(&"A".to_string()).is_some());
    }

    #[test]
    fn test_dyn_treetn_heterogeneous_nodes() {
        let mut tn = DynTreeTN::<String>::new();

        // Add a TensorDynLen
        let i: DynIndex = Index::new_dyn(2);
        let tensor = make_tensor(vec![i.clone()]);
        tn.add_node("tensor".to_string(), tensor).unwrap();

        // Add another TensorDynLen with different dimensions
        let j: DynIndex = Index::new_dyn(3);
        let tensor2 = make_tensor(vec![j.clone()]);
        tn.add_node("tensor2".to_string(), tensor2).unwrap();

        assert_eq!(tn.node_count(), 2);

        // Verify we can access both nodes
        let node1 = tn.node_by_name(&"tensor".to_string()).unwrap();
        let node2 = tn.node_by_name(&"tensor2".to_string()).unwrap();

        assert_eq!(node1.num_external_indices(), 1);
        assert_eq!(node2.num_external_indices(), 1);
    }

    #[test]
    fn test_dyn_treetn_connect() {
        let mut tn = DynTreeTN::<String>::new();

        let i: DynIndex = Index::new_dyn(2);
        let bond: DynIndex = Index::new_dyn(4);
        let j: DynIndex = Index::new_dyn(3);

        let tensor_a = make_tensor(vec![i.clone(), bond.clone()]);
        let tensor_b = make_tensor(vec![bond.clone(), j.clone()]);

        let node_a = tn.add_node("A".to_string(), tensor_a).unwrap();
        let node_b = tn.add_node("B".to_string(), tensor_b).unwrap();

        // Connect using the shared bond index
        let edge = tn.connect(node_a, &bond, node_b, &bond).unwrap();

        assert_eq!(tn.edge_count(), 1);
        assert!(tn.bond_index(edge).is_some());
    }

    #[test]
    fn test_dyn_treetn_contract_single_node() {
        let mut tn = DynTreeTN::<String>::new();

        let i: DynIndex = Index::new_dyn(2);
        let j: DynIndex = Index::new_dyn(3);
        let tensor = make_tensor(vec![i.clone(), j.clone()]);

        tn.add_node("A".to_string(), tensor).unwrap();

        let result = tn.contract_to_tensor().unwrap();
        assert_eq!(result.indices.len(), 2);
    }

    #[test]
    fn test_dyn_treetn_contract_two_nodes() {
        let mut tn = DynTreeTN::<String>::new();

        let i: DynIndex = Index::new_dyn(2);
        let bond: DynIndex = Index::new_dyn(4);
        let j: DynIndex = Index::new_dyn(3);

        let tensor_a = make_tensor(vec![i.clone(), bond.clone()]);
        let tensor_b = make_tensor(vec![bond.clone(), j.clone()]);

        let node_a = tn.add_node("A".to_string(), tensor_a).unwrap();
        let node_b = tn.add_node("B".to_string(), tensor_b).unwrap();

        tn.connect(node_a, &bond, node_b, &bond).unwrap();

        let result = tn.contract_to_tensor().unwrap();
        // Result should have indices i and j (bond contracted)
        assert_eq!(result.indices.len(), 2);
    }

    #[test]
    fn test_dyn_treetn_clone() {
        let mut tn = DynTreeTN::<String>::new();

        let i: DynIndex = Index::new_dyn(2);
        let tensor = make_tensor(vec![i.clone()]);
        tn.add_node("A".to_string(), tensor).unwrap();

        let cloned = tn.clone();
        assert_eq!(cloned.node_count(), 1);
        assert!(cloned.node_by_name(&"A".to_string()).is_some());
    }

    #[test]
    fn test_dyn_treetn_mix_tensor_and_treetn() {
        use crate::TreeTN;

        let mut dyn_tn = DynTreeTN::<String>::new();

        // Create a TensorDynLen node
        let i: DynIndex = Index::new_dyn(2);
        let tensor = make_tensor(vec![i.clone()]);
        dyn_tn.add_node("dense_tensor".to_string(), tensor).unwrap();

        // Create a TreeTN (sub-network) with its own structure
        let j: DynIndex = Index::new_dyn(3);
        let sub_tensor = make_tensor(vec![j.clone()]);
        let sub_tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
            vec![sub_tensor],
            vec!["sub_node".to_string()],
        ).unwrap();

        // Add the TreeTN as a node in DynTreeTN - heterogeneous!
        dyn_tn.add_node("sub_network".to_string(), sub_tn).unwrap();

        assert_eq!(dyn_tn.node_count(), 2);

        // Verify we can access both nodes through the TensorLike interface
        let dense_node = dyn_tn.node_by_name(&"dense_tensor".to_string()).unwrap();
        let network_node = dyn_tn.node_by_name(&"sub_network".to_string()).unwrap();

        assert_eq!(dense_node.num_external_indices(), 1);
        assert_eq!(network_node.num_external_indices(), 1);

        // Both nodes can be converted to tensors through the TensorLike interface
        let dense_tensor = dense_node.to_tensor().unwrap();
        let network_tensor = network_node.to_tensor().unwrap();

        assert_eq!(dense_tensor.indices.len(), 1);
        assert_eq!(network_tensor.indices.len(), 1);
    }

    #[test]
    fn test_dyn_treetn_nested_networks() {
        use crate::TreeTN;

        // Create a TreeTN with two connected nodes
        let i: DynIndex = Index::new_dyn(2);
        let bond: DynIndex = Index::new_dyn(4);
        let j: DynIndex = Index::new_dyn(3);

        let tensor_a = make_tensor(vec![i.clone(), bond.clone()]);
        let tensor_b = make_tensor(vec![bond.clone(), j.clone()]);

        let inner_tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
            vec![tensor_a, tensor_b],
            vec!["A".to_string(), "B".to_string()],
        ).unwrap();

        // The inner TreeTN has external indices i and j (bond is internal)
        assert_eq!(inner_tn.num_external_indices(), 2);

        // Add to DynTreeTN
        let mut dyn_tn = DynTreeTN::<String>::new();
        dyn_tn.add_node("inner_network".to_string(), inner_tn).unwrap();

        // The DynTreeTN node should have the same external indices
        let node = dyn_tn.node_by_name(&"inner_network".to_string()).unwrap();
        assert_eq!(node.num_external_indices(), 2);

        // Contract the DynTreeTN - this will call to_tensor() on the inner TreeTN
        let result = dyn_tn.contract_to_tensor().unwrap();
        assert_eq!(result.indices.len(), 2);
    }
}
