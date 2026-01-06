//! Contraction and addition operations for TreeTN.
//!
//! This module provides methods for:
//! - Replacing internal indices with fresh IDs (`sim_internal_inds`)
//! - Adding two TreeTNs (`add`)
//! - Contracting TreeTN to tensor (`contract_to_tensor`)
//! - Zip-up contraction (`contract_zipup`)
//! - Naive contraction (`contract_naive`)
//! - Validation (`validate_ortho_consistency`)

use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::{Context, Result};

use tensor4all::index::{DynId, NoSymmSpace, Symmetry};
use tensor4all::{factorize, Canonical, CanonicalForm, FactorizeAlg, FactorizeOptions};
use tensor4all::TensorDynLen;

use super::addition::direct_sum_tensors;
use super::{common_inds, TreeTN};
use crate::named_graph::NamedGraph;
use crate::site_index_network::SiteIndexNetwork;

impl<Id, Symm, V> TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Create a copy with all internal (link/bond) indices replaced by fresh IDs.
    ///
    /// External (site/physical) indices remain unchanged. This is useful when
    /// contracting two TreeTNs that might have overlapping internal index IDs.
    ///
    /// # Returns
    /// A new TreeTN with all bond indices replaced by `sim` indices (same dimension,
    /// new unique ID).
    ///
    /// # Example
    /// ```ignore
    /// let tn1 = create_random_treetn();
    /// let tn2 = create_random_treetn();
    ///
    /// // Before contraction, ensure no overlapping internal indices
    /// let tn2_sim = tn2.sim_internal_inds();
    /// let result = contract_zipup(&tn1, &tn2_sim, ...);
    /// ```
    pub fn sim_internal_inds(&self) -> Self
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry,
    {
        use tensor4all::index_ops::sim;

        // Clone the structure
        let mut result = self.clone();

        // For each edge, create a sim index and update both the edge and tensors
        let edges: Vec<EdgeIndex> = result.graph.graph().edge_indices().collect();

        for edge in edges {
            // Get the current bond index
            let old_bond_idx = match result.bond_index(edge) {
                Some(idx) => idx.clone(),
                None => continue,
            };

            // Create a new sim index (same dimension, new ID)
            let new_bond_idx = sim(&old_bond_idx);

            // Get the endpoint nodes
            let (node_a, node_b) = match result.graph.graph().edge_endpoints(edge) {
                Some(endpoints) => endpoints,
                None => continue,
            };

            // Update the edge weight
            if let Some(edge_weight) = result.graph.graph_mut().edge_weight_mut(edge) {
                *edge_weight = new_bond_idx.clone();
            }

            // Update tensor at node_a
            if let Some(tensor_a) = result.graph.graph_mut().node_weight_mut(node_a) {
                *tensor_a = tensor_a.replaceind(&old_bond_idx, &new_bond_idx);
            }

            // Update tensor at node_b
            if let Some(tensor_b) = result.graph.graph_mut().node_weight_mut(node_b) {
                *tensor_b = tensor_b.replaceind(&old_bond_idx, &new_bond_idx);
            }
        }

        result
    }

    /// Add two TreeTN together using direct-sum (block) construction.
    ///
    /// This method constructs a new TTN whose bond indices are the **direct sums**
    /// of the original bond indices, so that the resulting network represents the
    /// exact sum of the two input networks.
    ///
    /// # Algorithm
    ///
    /// 1. Compute merged bond indices from both networks (using `compute_merged_bond_indices`)
    /// 2. For each node, compute the direct sum of tensors (using `direct_sum_tensors`)
    /// 3. Build the result TreeTN with connections using the merged bond indices
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to add
    ///
    /// # Returns
    /// A new TreeTN representing `self + other`, or an error if the networks are incompatible.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Topologies are not compatible
    /// - Site spaces are not equal
    /// - Storage types are not dense (only DenseF64 and DenseC64 are supported)
    ///
    /// # Notes
    /// - The result is not canonicalized; `canonical_center` is cleared.
    /// - Bond dimensions increase: new_dim = dim_A + dim_B.
    /// - Only dense storage (DenseF64/DenseC64) is currently supported.
    pub fn add(self, other: Self) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: Ord,
    {
        // Validate site index network equivalence
        if !self.share_equivalent_site_index_network(&other) {
            return Err(anyhow::anyhow!(
                "Cannot add TreeTN: site index networks are not equivalent"
            ));
        }

        if self.node_count() == 0 {
            return Ok(other);
        }
        if other.node_count() == 0 {
            return Ok(self);
        }

        // Step 1: Compute merged bond indices
        let merged_bonds = self.compute_merged_bond_indices(&other)?;

        // Build node names list (sorted)
        let node_names: Vec<V> = {
            let mut names: Vec<V> = self.graph.graph().node_indices()
                .filter_map(|idx| self.graph.node_name(idx).cloned())
                .collect();
            names.sort();
            names
        };

        // Create new TreeTN (empty)
        let mut result = Self {
            graph: NamedGraph::new(),
            canonical_center: HashSet::new(),
            canonical_form: None,
            site_index_network: SiteIndexNetwork::new(),
            ortho_towards: HashMap::new(),
        };

        // Step 2: Process each node using direct_sum_tensors
        for node_name in &node_names {
            let node_idx_a = self.graph.node_index(node_name)
                .ok_or_else(|| anyhow::anyhow!("Node not found in self"))?;
            let node_idx_b = other.graph.node_index(node_name)
                .ok_or_else(|| anyhow::anyhow!("Node not found in other"))?;

            let tensor_a = self.tensor(node_idx_a)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in self"))?;
            let tensor_b = other.tensor(node_idx_b)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in other"))?;

            // Get site indices for this node
            let site_indices: HashSet<Id> = self.site_space(node_name)
                .map(|s| s.iter().map(|idx| idx.id.clone()).collect())
                .unwrap_or_default();

            // Build neighbor_names maps for tensor_a and tensor_b
            let mut neighbor_names_a: HashMap<Id, V> = HashMap::new();
            for edge in self.edges_for_node(node_idx_a) {
                if let Some(bond_idx) = self.bond_index(edge.0) {
                    let neighbor_name = self.graph.node_name(edge.1)
                        .ok_or_else(|| anyhow::anyhow!("Neighbor name not found"))?
                        .clone();
                    neighbor_names_a.insert(bond_idx.id.clone(), neighbor_name);
                }
            }

            let mut neighbor_names_b: HashMap<Id, V> = HashMap::new();
            for edge in other.edges_for_node(node_idx_b) {
                if let Some(bond_idx) = other.bond_index(edge.0) {
                    let neighbor_name = other.graph.node_name(edge.1)
                        .ok_or_else(|| anyhow::anyhow!("Neighbor name not found"))?
                        .clone();
                    neighbor_names_b.insert(bond_idx.id.clone(), neighbor_name);
                }
            }

            // Build bond_info_by_neighbor for this node
            let mut bond_info_by_neighbor: HashMap<V, _> = HashMap::new();
            for (neighbor, info) in merged_bonds.iter()
                .filter_map(|((a, b), info)| {
                    if a == node_name {
                        Some((b, info))
                    } else if b == node_name {
                        Some((a, info))
                    } else {
                        None
                    }
                })
            {
                bond_info_by_neighbor.insert(neighbor.clone(), info);
            }

            // Compute direct sum
            let new_tensor = direct_sum_tensors(
                tensor_a,
                tensor_b,
                &site_indices,
                &bond_info_by_neighbor,
                &neighbor_names_a,
                &neighbor_names_b,
            )?;

            result.add_tensor_internal(node_name.clone(), new_tensor)?;
        }

        // Step 3: Add connections using the merged bond indices
        for ((src_name, tgt_name), info) in &merged_bonds {
            let src_node = result.graph.node_index(src_name)
                .ok_or_else(|| anyhow::anyhow!("Source node not found in result"))?;
            let tgt_node = result.graph.node_index(tgt_name)
                .ok_or_else(|| anyhow::anyhow!("Target node not found in result"))?;
            // Use the same merged_index for both endpoints
            result.connect_internal(src_node, &info.merged_index, tgt_node, &info.merged_index)?;
        }

        // Clear canonical_center (sum is not canonicalized)
        result.clear_canonical_center();

        Ok(result)
    }

    /// Contract the TreeTN to a single dense tensor.
    ///
    /// This method contracts all tensors in the network into a single tensor
    /// containing all physical indices. The contraction is performed using
    /// an edge-based order (post-order DFS edges towards root), processing
    /// each edge in sequence and using Connection information to identify
    /// which indices to contract.
    ///
    /// # Returns
    /// A single tensor representing the full contraction of the network.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The network is empty
    /// - The graph is not a valid tree
    /// - Tensor contraction fails
    pub fn contract_to_tensor(&self) -> Result<TensorDynLen<Id, Symm>>
    where
        V: Ord,
    {
        if self.node_count() == 0 {
            return Err(anyhow::anyhow!("Cannot contract empty TreeTN"));
        }

        if self.node_count() == 1 {
            // Single node - just return a clone of its tensor
            let node = self.graph.graph().node_indices().next()
                .ok_or_else(|| anyhow::anyhow!("No nodes found"))?;
            return self.tensor(node)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Tensor not found"));
        }

        // Validate tree structure
        self.validate_tree()
            .context("contract_to_tensor: graph must be a tree")?;

        // Choose a deterministic root (minimum node name)
        let root_name = self.graph.graph().node_indices()
            .filter_map(|idx| self.graph.node_name(idx).cloned())
            .min()
            .ok_or_else(|| anyhow::anyhow!("No nodes found"))?;
        let root = self.graph.node_index(&root_name)
            .ok_or_else(|| anyhow::anyhow!("Root node not found"))?;

        // Get edges to process (post-order DFS edges towards root)
        let edges = self.site_index_network.edges_to_canonicalize(None, root);

        // Initialize with original tensors
        let mut tensors: HashMap<NodeIndex, TensorDynLen<Id, Symm>> = self.graph.graph()
            .node_indices()
            .filter_map(|n| self.tensor(n).cloned().map(|t| (n, t)))
            .collect();

        // Process each edge: contract tensor at `from` into tensor at `to`
        for (from, to) in edges {
            let from_tensor = tensors.remove(&from)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", from))?;
            let to_tensor = tensors.remove(&to)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", to))?;

            // Find the edge and get bond index
            let edge = self.graph.graph().find_edge(from, to)
                .or_else(|| self.graph.graph().find_edge(to, from))
                .ok_or_else(|| anyhow::anyhow!("Edge not found between {:?} and {:?}", from, to))?;

            // In Einsum mode, both endpoints share the same bond index
            let bond_idx = self.bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge"))?
                .clone();

            // Contract and store result at `to`
            let contracted = to_tensor.tensordot(&from_tensor, &[(bond_idx.clone(), bond_idx)])
                .context("Failed to contract along edge")?;
            tensors.insert(to, contracted);
        }

        // The root's tensor is the final result
        tensors.remove(&root)
            .ok_or_else(|| anyhow::anyhow!("Contraction produced no result"))
    }

    /// Contract two TreeTNs with the same topology using the zip-up algorithm.
    ///
    /// The zip-up algorithm traverses from leaves towards the center, contracting
    /// corresponding nodes from both networks and optionally truncating at each step.
    ///
    /// # Algorithm
    /// 1. Replace internal (bond) indices of both networks with fresh IDs to avoid collision
    /// 2. Traverse from leaves towards center
    /// 3. At each edge (child → parent):
    ///    - Contract the child tensors from both networks (along their shared site indices)
    ///    - Factorize, keeping site indices + parent bond on left (canonical form)
    ///    - Store left factor as child tensor in result
    ///    - Contract right factor into parent tensor
    /// 4. Contract the final center tensors
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to contract with (must have same topology)
    /// * `center` - The center node name towards which to contract
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    ///
    /// # Returns
    /// The contracted TreeTN result, or an error if topologies don't match or contraction fails.
    ///
    /// # Notes
    /// - Both TreeTNs must have the same topology (checked via `same_topology`)
    /// - Site (external) indices are contracted between corresponding nodes
    /// - Bond (internal) indices are replaced with fresh IDs before contraction
    pub fn contract_zipup(
        &self,
        other: &Self,
        center: &V,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        self.contract_zipup_with(other, center, CanonicalForm::Unitary, rtol, max_rank)
    }

    /// Contract two TreeTNs with the same topology using the zip-up algorithm with a specified form.
    ///
    /// See [`contract_zipup`](Self::contract_zipup) for details.
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to contract with (must have same topology)
    /// * `center` - The center node name towards which to contract
    /// * `form` - The canonical form / algorithm to use for factorization
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    pub fn contract_zipup_with(
        &self,
        other: &Self,
        center: &V,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        // 1. Verify topologies are compatible (same graph structure)
        if !self.same_topology(other) {
            return Err(anyhow::anyhow!("contract_zipup: networks have incompatible topologies"));
        }

        // 2. Replace internal indices with fresh IDs to avoid collision
        let tn1 = self.sim_internal_inds();
        let tn2 = other.sim_internal_inds();

        // 3. Get traversal edges from leaves to center
        let edges = tn1.edges_to_canonicalize_by_names(center)
            .ok_or_else(|| anyhow::anyhow!("contract_zipup: center node {:?} not found", center))?;

        if edges.is_empty() && self.node_count() == 1 {
            // Single node case: just contract the two tensors
            let node_idx = tn1.graph.graph().node_indices().next()
                .ok_or_else(|| anyhow::anyhow!("contract_zipup: no nodes found"))?;
            let t1 = tn1.tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("contract_zipup: tensor not found in tn1"))?;
            let t2 = tn2.tensor(tn2.graph.graph().node_indices().next()
                .ok_or_else(|| anyhow::anyhow!("contract_zipup: tensor not found in tn2"))?)
                .ok_or_else(|| anyhow::anyhow!("contract_zipup: tensor not found"))?;

            // Contract along common indices (site indices)
            let common = common_inds(&t1.indices, &t2.indices);
            let result = if common.is_empty() {
                // Outer product when no common site indices
                t1.outer_product(t2)?
            } else {
                let pairs: Vec<_> = common.iter().map(|idx| (idx.clone(), idx.clone())).collect();
                t1.tensordot(t2, &pairs)?
            };
            let mut result_tn = Self::new();
            let node_name = tn1.graph.node_name(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Node name not found"))?.clone();
            result_tn.add_tensor(node_name, result)?;
            return Ok(result_tn);
        }

        // 4. Initialize result tensors (start with contracted node tensors)
        // For each node, contract the tensors from tn1 and tn2 along their common indices
        let mut result_tensors: HashMap<V, TensorDynLen<Id, Symm>> = HashMap::new();

        for node_name in tn1.node_names() {
            let node1 = tn1.node_index(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tn1", node_name))?;
            let node2 = tn2.node_index(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tn2", node_name))?;

            let t1 = tn1.tensor(node1)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?} in tn1", node_name))?;
            let t2 = tn2.tensor(node2)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?} in tn2", node_name))?;

            // Contract along site indices (external indices)
            let site_inds1: HashSet<_> = tn1.site_index_network.site_space(&node_name)
                .map(|s| s.iter().map(|i| i.id.clone()).collect())
                .unwrap_or_default();

            let common: Vec<_> = common_inds(&t1.indices, &t2.indices)
                .into_iter()
                .filter(|idx| site_inds1.contains(&idx.id))
                .collect();

            let contracted = if common.is_empty() {
                // Outer product when no common site indices
                t1.outer_product(t2)?
            } else {
                let pairs: Vec<_> = common.iter().map(|idx| (idx.clone(), idx.clone())).collect();
                t1.tensordot(t2, &pairs)?
            };

            result_tensors.insert(node_name, contracted);
        }

        // 5. Process edges from leaves to center
        let alg = match form {
            CanonicalForm::Unitary => FactorizeAlg::SVD,
            CanonicalForm::LU => FactorizeAlg::LU,
            CanonicalForm::CI => FactorizeAlg::CI,
        };

        for (child_name, parent_name) in &edges {
            let child_tensor = result_tensors.remove(child_name)
                .ok_or_else(|| anyhow::anyhow!("Child tensor {:?} not found", child_name))?;

            // Get the bond index between child and parent from tn1 (bond1) and tn2 (bond2)
            // After sim_internal_inds, these are fresh indices
            let edge1 = tn1.edge_between(child_name, parent_name)
                .ok_or_else(|| anyhow::anyhow!("Edge not found between {:?} and {:?} in tn1", child_name, parent_name))?;
            let edge2 = tn2.edge_between(child_name, parent_name)
                .ok_or_else(|| anyhow::anyhow!("Edge not found between {:?} and {:?} in tn2", child_name, parent_name))?;

            let bond1 = tn1.bond_index(edge1)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge in tn1"))?
                .clone();
            let bond2 = tn2.bond_index(edge2)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge in tn2"))?
                .clone();

            // Factorize: left_inds = all indices except bond1 and bond2 (towards parent)
            let left_inds: Vec<_> = child_tensor.indices.iter()
                .filter(|idx| idx.id != bond1.id && idx.id != bond2.id)
                .cloned()
                .collect();

            if left_inds.is_empty() {
                // All indices are bond indices - just absorb into parent
                // (This happens when site indices are fully contracted, i.e., inner product case)
                let parent_tensor = result_tensors.remove(parent_name)
                    .ok_or_else(|| anyhow::anyhow!("Parent tensor {:?} not found", parent_name))?;

                let contracted = parent_tensor.tensordot(&child_tensor, &[(bond1.clone(), bond1), (bond2.clone(), bond2)])?;
                result_tensors.insert(parent_name.clone(), contracted);
                // Don't re-insert child - it's been absorbed
                // The node will be omitted from the result TreeTN
                continue;
            }

            let factorize_options = FactorizeOptions {
                alg,
                canonical: Canonical::Left,
                rtol,
                max_rank,
            };

            let factorize_result = factorize(&child_tensor, &left_inds, &factorize_options)
                .map_err(|e| anyhow::anyhow!("Factorization failed at node {:?}: {}", child_name, e))?;

            // Store left factor as child
            result_tensors.insert(child_name.clone(), factorize_result.left);

            // Contract right factor into parent
            let parent_tensor = result_tensors.remove(parent_name)
                .ok_or_else(|| anyhow::anyhow!("Parent tensor {:?} not found", parent_name))?;

            // Right factor has: new_bond (from factorize), bond1, bond2
            // Parent has: bond1, bond2 (among other indices)
            // Contract along bond1 and bond2
            let contracted = parent_tensor.tensordot(&factorize_result.right, &[(bond1.clone(), bond1), (bond2.clone(), bond2)])?;
            result_tensors.insert(parent_name.clone(), contracted);
        }

        // 6. Build result TreeTN
        let mut result = Self::new();

        // Add tensors for nodes that still have tensors
        // (Some nodes may have been absorbed during the zip-up process in inner product cases)
        let remaining_nodes: Vec<_> = tn1.node_names()
            .into_iter()
            .filter(|name| result_tensors.contains_key(name))
            .collect();

        for node_name in &remaining_nodes {
            let tensor = result_tensors.remove(node_name)
                .ok_or_else(|| anyhow::anyhow!("Result tensor not found for node {:?}", node_name))?;
            result.add_tensor(node_name.clone(), tensor)?;
        }

        // Connect nodes based on topology using matching index IDs
        // Only connect edges where both endpoints exist in the result
        for (a, b) in tn1.site_index_network.edges() {
            let node_a = match result.node_index(&a) {
                Some(idx) => idx,
                None => continue, // Node was absorbed
            };
            let node_b = match result.node_index(&b) {
                Some(idx) => idx,
                None => continue, // Node was absorbed
            };

            let tensor_a = result.tensor(node_a).unwrap();
            let tensor_b = result.tensor(node_b).unwrap();

            // Find the common index (should be exactly one new bond index)
            let common = common_inds(&tensor_a.indices, &tensor_b.indices);
            if let Some(bond_idx) = common.first() {
                result.connect_internal(node_a, bond_idx, node_b, bond_idx)?;
            }
        }

        // Set canonical center (only if it exists in result)
        if result.node_index(center).is_some() {
            result.set_canonical_center(std::iter::once(center.clone()))?;
        }

        Ok(result)
    }

    /// Contract two TreeTNs using naive full contraction.
    ///
    /// This is a reference implementation that:
    /// 1. Replaces internal indices with fresh IDs (sim_internal_inds)
    /// 2. Converts both TreeTNs to full tensors
    /// 3. Contracts along common site indices
    ///
    /// The result is a single tensor, not a TreeTN. This is useful for:
    /// - Testing correctness of more sophisticated algorithms like `contract_zipup`
    /// - Computing exact results for small networks
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to contract with (must have same topology)
    ///
    /// # Returns
    /// A tensor representing the contracted result.
    ///
    /// # Note
    /// This method is O(exp(n)) in both time and memory where n is the number of nodes.
    /// Use `contract_zipup` for efficient contraction of large networks.
    pub fn contract_naive(
        &self,
        other: &Self,
    ) -> Result<TensorDynLen<Id, Symm>>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + From<DynId> + std::fmt::Debug,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: Ord,
    {
        // 1. Verify topologies are compatible
        if !self.same_topology(other) {
            return Err(anyhow::anyhow!("contract_naive: networks have incompatible topologies"));
        }

        // 2. Replace internal indices with fresh IDs to avoid collision
        let tn1 = self.sim_internal_inds();
        let tn2 = other.sim_internal_inds();

        // 3. Convert both networks to full tensors
        let tensor1 = tn1.contract_to_tensor()
            .map_err(|e| anyhow::anyhow!("contract_naive: failed to contract tn1: {}", e))?;
        let tensor2 = tn2.contract_to_tensor()
            .map_err(|e| anyhow::anyhow!("contract_naive: failed to contract tn2: {}", e))?;

        // 4. Find common indices (site indices) to contract
        let common = common_inds(&tensor1.indices, &tensor2.indices);

        // 5. Contract along common indices
        if common.is_empty() {
            // Outer product when no common indices
            tensor1.outer_product(&tensor2)
        } else {
            let pairs: Vec<_> = common.iter().map(|idx| (idx.clone(), idx.clone())).collect();
            tensor1.tensordot(&tensor2, &pairs)
        }
    }

    /// Validate that `canonical_center` and edge `ortho_towards` are consistent.
    ///
    /// Rules:
    /// - If `canonical_center` is empty (not canonicalized), all indices must have `ortho_towards == None`.
    /// - If `canonical_center` is non-empty:
    ///   - It must form a connected subtree
    ///   - All edges from outside the center region must have `ortho_towards` pointing towards the center
    ///   - Edges entirely inside the center region may have `ortho_towards == None`
    pub fn validate_ortho_consistency(&self) -> Result<()> {
        // If not canonicalized, require no ortho_towards at all
        if self.canonical_center.is_empty() {
            if !self.ortho_towards.is_empty() {
                return Err(anyhow::anyhow!(
                    "Found {} ortho_towards entries but canonical_center is empty",
                    self.ortho_towards.len()
                ))
                .context("validate_ortho_consistency: canonical_center empty implies no ortho_towards");
            }
            return Ok(());
        }

        // Validate all canonical_center nodes exist and convert to NodeIndex
        let mut center_indices = HashSet::new();
        for c in &self.canonical_center {
            let idx = self.graph.node_index(c)
                .ok_or_else(|| anyhow::anyhow!("canonical_center node {:?} does not exist", c))?;
            center_indices.insert(idx);
        }

        // Check canonical_center connectivity
        if !self.site_index_network.is_connected_subset(&center_indices) {
            return Err(anyhow::anyhow!("canonical_center is not connected"))
                .context("validate_ortho_consistency: canonical_center must form a connected subtree");
        }

        // Get expected edges from edges_to_canonicalize_to_region
        // These are edges (src, dst) where src is outside the center and dst is towards the center
        let expected_edges = self.site_index_network.edges_to_canonicalize_to_region(&center_indices);

        // Build a set of expected (bond_id, expected_direction) pairs
        let mut expected_directions: HashMap<Id, V> = HashMap::new();
        for (src, dst) in expected_edges.iter() {
            // Find the edge between src and dst
            let edge = self.graph.graph().find_edge(*src, *dst)
                .or_else(|| self.graph.graph().find_edge(*dst, *src))
                .ok_or_else(|| anyhow::anyhow!("Edge not found between {:?} and {:?}", src, dst))?;

            let bond_id = self.bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge"))?
                .id
                .clone();

            // The expected ortho_towards direction is dst (towards center)
            let dst_name = self.graph.node_name(*dst)
                .ok_or_else(|| anyhow::anyhow!("Node name not found for {:?}", dst))?
                .clone();

            expected_directions.insert(bond_id, dst_name);
        }

        // Verify all expected directions are present in ortho_towards
        for (bond_id, expected_dir) in &expected_directions {
            match self.ortho_towards.get(bond_id) {
                Some(actual_dir) => {
                    if actual_dir != expected_dir {
                        return Err(anyhow::anyhow!(
                            "ortho_towards for bond {:?} points to {:?} but expected {:?}",
                            bond_id, actual_dir, expected_dir
                        ))
                        .context("validate_ortho_consistency: wrong direction");
                    }
                }
                None => {
                    return Err(anyhow::anyhow!(
                        "ortho_towards for bond {:?} is missing, expected to point to {:?}",
                        bond_id, expected_dir
                    ))
                    .context("validate_ortho_consistency: missing ortho_towards");
                }
            }
        }

        // Verify no unexpected bond ortho_towards entries
        // (site index ortho_towards are allowed even if not in expected_directions)
        let bond_ids: HashSet<Id> = self.graph.graph().edge_indices()
            .filter_map(|e| self.bond_index(e))
            .map(|b| b.id.clone())
            .collect();

        for (id, _) in &self.ortho_towards {
            if bond_ids.contains(id) && !expected_directions.contains_key(id) {
                // This is a bond inside the canonical_center - should not have ortho_towards
                return Err(anyhow::anyhow!(
                    "Unexpected ortho_towards for bond {:?} (inside canonical_center)",
                    id
                ))
                .context("validate_ortho_consistency: bonds inside center should not have ortho_towards");
            }
        }

        Ok(())
    }
}
