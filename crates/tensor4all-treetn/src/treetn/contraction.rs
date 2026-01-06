//! Contraction and addition operations for TreeTN.
//!
//! This module provides methods for:
//! - Replacing internal indices with fresh IDs (`sim_internal_inds`)
//! - Adding two TreeTNs (`add`)
//! - Contracting TreeTN to tensor (`contract_to_tensor`)
//! - Zip-up contraction (`contract_zipup`)
//! - Naive contraction (`contract_naive`)
//! - Validation (`validate_ortho_consistency`)

use num_complex::Complex64;
use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

use anyhow::{Context, Result};

use tensor4all::index::{DynId, Index, NoSymmSpace, Symmetry, TagSet};
use tensor4all::storage::{DenseStorageC64, DenseStorageF64, Storage};
use tensor4all::{factorize, Canonical, CanonicalForm, FactorizeAlg, FactorizeOptions};
use tensor4all::TensorDynLen;

use super::{common_inds, compute_strides, linear_to_multi_index, multi_to_linear_index, TreeTN};
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
    /// For each edge `e`, create a new bond index with dimension `dA(e) + dB(e)`.
    /// For each node `v`, create a new tensor that embeds:
    /// - `T_A(v)` in the "A block" (first part of each bond dimension)
    /// - `T_B(v)` in the "B block" (second part of each bond dimension)
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

        // Build a mapping from V -> NodeIndex for both networks
        let node_names: Vec<V> = {
            let mut names: Vec<V> = self.graph.graph().node_indices()
                .filter_map(|idx| self.graph.node_name(idx).cloned())
                .collect();
            names.sort();
            names
        };

        // Build edge info: for each edge in self, store (src_name, tgt_name, bond_dim_A, bond_dim_B, shared_index)
        // Use a SINGLE shared index for both endpoints of each edge (fix for issue #2)
        let mut edge_info: HashMap<(V, V), (usize, usize, Index<Id, Symm>)> = HashMap::new();

        for edge in self.graph.graph().edge_indices() {
            let (src, tgt) = self.graph.graph().edge_endpoints(edge)
                .ok_or_else(|| anyhow::anyhow!("Edge has no endpoints"))?;
            let bond_index_a = self.bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found in self"))?;
            let bond_dim_a = bond_index_a.size();

            let src_name = self.graph.node_name(src)
                .ok_or_else(|| anyhow::anyhow!("Source node name not found"))?
                .clone();
            let tgt_name = self.graph.node_name(tgt)
                .ok_or_else(|| anyhow::anyhow!("Target node name not found"))?
                .clone();

            // Find corresponding edge in other
            let src_idx_other = other.graph.node_index(&src_name)
                .ok_or_else(|| anyhow::anyhow!("Source node not found in other"))?;
            let tgt_idx_other = other.graph.node_index(&tgt_name)
                .ok_or_else(|| anyhow::anyhow!("Target node not found in other"))?;

            // Find edge between these nodes in other
            let edge_other = other.graph.graph().edges_connecting(src_idx_other, tgt_idx_other)
                .next()
                .or_else(|| other.graph.graph().edges_connecting(tgt_idx_other, src_idx_other).next())
                .ok_or_else(|| anyhow::anyhow!("Edge not found in other"))?;
            let bond_index_b = other.bond_index(edge_other.id())
                .ok_or_else(|| anyhow::anyhow!("Bond index not found in other"))?;
            let bond_dim_b = bond_index_b.size();

            // Create ONE shared bond index for both endpoints (same ID)
            let new_dim = bond_dim_a + bond_dim_b;
            // Create Link index using DynId, then convert to Id, Symm
            let dyn_bond_index = Index::new_link(new_dim)
                .map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))?;
            let shared_index: Index<Id, Symm, TagSet> = Index {
                id: dyn_bond_index.id.into(),
                symm: dyn_bond_index.symm.into(),
                tags: dyn_bond_index.tags,
            };

            // Store in canonical order (smaller name first)
            let key = if src_name < tgt_name {
                (src_name.clone(), tgt_name.clone())
            } else {
                (tgt_name.clone(), src_name.clone())
            };
            edge_info.insert(key, (bond_dim_a, bond_dim_b, shared_index));
        }

        // Create new TreeTN (empty)
        let mut result = Self {
            graph: NamedGraph::new(),
            canonical_center: HashSet::new(),
            canonical_form: None,
            site_index_network: SiteIndexNetwork::new(),
            ortho_towards: HashMap::new(),
        };

        // Process each node
        for node_name in &node_names {
            let node_idx_a = self.graph.node_index(node_name)
                .ok_or_else(|| anyhow::anyhow!("Node not found in self"))?;
            let node_idx_b = other.graph.node_index(node_name)
                .ok_or_else(|| anyhow::anyhow!("Node not found in other"))?;

            let tensor_a = self.tensor(node_idx_a)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in self"))?;
            let tensor_b = other.tensor(node_idx_b)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in other"))?;

            // Get physical indices for this node (from site_index_network)
            let physical_indices_a: HashSet<&Index<Id, Symm>> = self.site_space(node_name)
                .map(|s| s.iter().collect())
                .unwrap_or_default();

            // Separate physical and bond indices for tensor A, preserving order
            let mut physical_inds_a: Vec<Index<Id, Symm>> = Vec::new();
            let mut bond_inds_a: Vec<(Index<Id, Symm>, V, usize)> = Vec::new(); // (index, neighbor_name, original_position)

            for (pos, idx) in tensor_a.indices.iter().enumerate() {
                if physical_indices_a.contains(idx) {
                    physical_inds_a.push(idx.clone());
                } else {
                    // This is a bond index - find which neighbor it connects to
                    for edge in self.edges_for_node(node_idx_a) {
                        #[allow(deprecated)]
                        if let Ok(edge_idx) = self.edge_index_for_node(edge.0, node_idx_a) {
                            if edge_idx.id == idx.id {
                                let neighbor_name = self.graph.node_name(edge.1)
                                    .ok_or_else(|| anyhow::anyhow!("Neighbor name not found"))?
                                    .clone();
                                bond_inds_a.push((idx.clone(), neighbor_name, pos));
                                break;
                            }
                        }
                    }
                }
            }

            // Build the canonical index order: physical indices first (in original order), then bonds (sorted by neighbor)
            let mut canonical_indices: Vec<Index<Id, Symm>> = physical_inds_a.clone();
            let mut canonical_dims: Vec<usize> = physical_inds_a.iter()
                .map(|idx| idx.size())
                .collect();

            // Sort bonds by neighbor name for deterministic ordering
            let mut sorted_bonds = bond_inds_a.clone();
            sorted_bonds.sort_by(|a, b| a.1.cmp(&b.1));

            // Build axis mapping: for each axis in canonical order, what's the corresponding new dim?
            // And track bond positions for embedding
            let mut bond_axis_info: Vec<(usize, usize, usize)> = Vec::new(); // (canonical_axis, dim_a, dim_b)

            for (_old_idx, neighbor_name, _orig_pos) in &sorted_bonds {
                let key = if *node_name < *neighbor_name {
                    (node_name.clone(), neighbor_name.clone())
                } else {
                    (neighbor_name.clone(), node_name.clone())
                };
                let (dim_a, dim_b, shared_idx) = edge_info.get(&key)
                    .ok_or_else(|| anyhow::anyhow!("Edge info not found"))?;

                bond_axis_info.push((canonical_indices.len(), *dim_a, *dim_b));
                canonical_indices.push(shared_idx.clone());
                canonical_dims.push(dim_a + dim_b);
            }

            // Now we need to permute tensor_a and tensor_b to match canonical order
            // Build permutation for tensor_a
            let mut perm_a: Vec<usize> = Vec::new();
            // First, add physical indices in the order they appear in physical_inds_a
            for phys_idx in &physical_inds_a {
                let pos = tensor_a.indices.iter().position(|i| i.id == phys_idx.id)
                    .ok_or_else(|| anyhow::anyhow!("Physical index not found in tensor_a"))?;
                perm_a.push(pos);
            }
            // Then add bond indices in sorted order
            for (_old_idx, neighbor_name, _) in &sorted_bonds {
                // Find the original position of this bond in tensor_a
                for (_orig_bond_idx, orig_neighbor, orig_pos) in &bond_inds_a {
                    if orig_neighbor == neighbor_name {
                        perm_a.push(*orig_pos);
                        break;
                    }
                }
            }

            // Create permuted version of tensor_a storage
            let permuted_dims_a: Vec<usize> = perm_a.iter().map(|&i| tensor_a.dims[i]).collect();

            // Do the same for tensor_b
            let physical_indices_b: HashSet<&Index<Id, Symm>> = other.site_space(node_name)
                .map(|s| s.iter().collect())
                .unwrap_or_default();

            let mut bond_inds_b: Vec<(Index<Id, Symm>, V, usize)> = Vec::new();
            for (pos, idx) in tensor_b.indices.iter().enumerate() {
                if !physical_indices_b.contains(idx) {
                    for edge in other.edges_for_node(node_idx_b) {
                        #[allow(deprecated)]
                        if let Ok(edge_idx) = other.edge_index_for_node(edge.0, node_idx_b) {
                            if edge_idx.id == idx.id {
                                let neighbor_name = other.graph.node_name(edge.1)
                                    .ok_or_else(|| anyhow::anyhow!("Neighbor name not found"))?
                                    .clone();
                                bond_inds_b.push((idx.clone(), neighbor_name, pos));
                                break;
                            }
                        }
                    }
                }
            }

            let mut perm_b: Vec<usize> = Vec::new();
            // Physical indices in same order as physical_inds_a (matched by ID in site_space)
            for phys_idx in &physical_inds_a {
                // Find matching physical index in tensor_b by ID
                let pos = tensor_b.indices.iter().position(|i| {
                    physical_indices_b.contains(i) &&
                    self.site_space(node_name).map_or(false, |s| s.iter().any(|si| si.id == i.id && si.id == phys_idx.id))
                }).or_else(|| {
                    // Fallback: match by position in physical index set
                    tensor_b.indices.iter().position(|i| physical_indices_b.contains(i))
                }).ok_or_else(|| anyhow::anyhow!("Physical index not found in tensor_b"))?;
                if !perm_b.contains(&pos) {
                    perm_b.push(pos);
                }
            }
            // Bond indices in sorted order (by neighbor name)
            for (_old_idx, neighbor_name, _) in &sorted_bonds {
                for (_, orig_neighbor, orig_pos) in &bond_inds_b {
                    if orig_neighbor == neighbor_name {
                        perm_b.push(*orig_pos);
                        break;
                    }
                }
            }

            let permuted_dims_b: Vec<usize> = perm_b.iter().map(|&i| tensor_b.dims[i]).collect();

            // Create the new tensor storage
            let total_size: usize = canonical_dims.iter().product();
            let is_complex = matches!(tensor_a.storage.as_ref(), Storage::DenseC64(_))
                || matches!(tensor_b.storage.as_ref(), Storage::DenseC64(_));

            // If there are no bonds for this node, add tensors element-wise.
            // Block embedding only makes sense when there are bond dimensions to separate the blocks.
            let has_bonds = !bond_axis_info.is_empty();

            let new_storage = if is_complex {
                let data_a = match tensor_a.storage.as_ref() {
                    Storage::DenseF64(d) => d.as_slice().iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>(),
                    Storage::DenseC64(d) => d.as_slice().to_vec(),
                    _ => return Err(anyhow::anyhow!("Only dense storage is supported for TTN addition")),
                };
                let data_b = match tensor_b.storage.as_ref() {
                    Storage::DenseF64(d) => d.as_slice().iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>(),
                    Storage::DenseC64(d) => d.as_slice().to_vec(),
                    _ => return Err(anyhow::anyhow!("Only dense storage is supported for TTN addition")),
                };

                // Permute data before embedding/adding
                let permuted_data_a = permute_data_c64(&data_a, &tensor_a.dims, &perm_a);
                let permuted_data_b = permute_data_c64(&data_b, &tensor_b.dims, &perm_b);

                if has_bonds {
                    // Use block embedding for nodes with bonds
                    let mut result_data = vec![Complex64::new(0.0, 0.0); total_size];
                    embed_block_c64(&mut result_data, &canonical_dims, &permuted_data_a, &permuted_dims_a, &bond_axis_info, true)?;
                    embed_block_c64(&mut result_data, &canonical_dims, &permuted_data_b, &permuted_dims_b, &bond_axis_info, false)?;
                    Storage::DenseC64(DenseStorageC64::from_vec(result_data))
                } else {
                    // No bonds: add tensors element-wise
                    let result_data: Vec<Complex64> = permuted_data_a.iter()
                        .zip(permuted_data_b.iter())
                        .map(|(&a, &b)| a + b)
                        .collect();
                    Storage::DenseC64(DenseStorageC64::from_vec(result_data))
                }
            } else {
                let data_a = match tensor_a.storage.as_ref() {
                    Storage::DenseF64(d) => d.as_slice().to_vec(),
                    _ => return Err(anyhow::anyhow!("Only dense storage is supported for TTN addition")),
                };
                let data_b = match tensor_b.storage.as_ref() {
                    Storage::DenseF64(d) => d.as_slice().to_vec(),
                    _ => return Err(anyhow::anyhow!("Only dense storage is supported for TTN addition")),
                };

                // Permute data before embedding/adding
                let permuted_data_a = permute_data_f64(&data_a, &tensor_a.dims, &perm_a);
                let permuted_data_b = permute_data_f64(&data_b, &tensor_b.dims, &perm_b);

                if has_bonds {
                    // Use block embedding for nodes with bonds
                    let mut result_data = vec![0.0_f64; total_size];
                    embed_block_f64(&mut result_data, &canonical_dims, &permuted_data_a, &permuted_dims_a, &bond_axis_info, true)?;
                    embed_block_f64(&mut result_data, &canonical_dims, &permuted_data_b, &permuted_dims_b, &bond_axis_info, false)?;
                    Storage::DenseF64(DenseStorageF64::from_vec(result_data))
                } else {
                    // No bonds: add tensors element-wise
                    let result_data: Vec<f64> = permuted_data_a.iter()
                        .zip(permuted_data_b.iter())
                        .map(|(&a, &b)| a + b)
                        .collect();
                    Storage::DenseF64(DenseStorageF64::from_vec(result_data))
                }
            };

            let new_tensor = TensorDynLen::new(canonical_indices, canonical_dims, Arc::new(new_storage));
            result.add_tensor_internal(node_name.clone(), new_tensor)?;
        }

        // Add connections using the SAME shared index on both endpoints
        for ((src_name, tgt_name), (_, _, shared_idx)) in &edge_info {
            let src_node = result.graph.node_index(src_name)
                .ok_or_else(|| anyhow::anyhow!("Source node not found in result"))?;
            let tgt_node = result.graph.node_index(tgt_name)
                .ok_or_else(|| anyhow::anyhow!("Target node not found in result"))?;
            // Use the same shared_idx for both endpoints - this ensures contraction works
            result.connect_internal(src_node, shared_idx, tgt_node, shared_idx)?;
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

            // Find the edge and get bond indices
            let edge = self.graph.graph().find_edge(from, to)
                .or_else(|| self.graph.graph().find_edge(to, from))
                .ok_or_else(|| anyhow::anyhow!("Edge not found between {:?} and {:?}", from, to))?;

            #[allow(deprecated)]
            let idx_from = self.edge_index_for_node(edge, from)?.clone();
            #[allow(deprecated)]
            let idx_to = self.edge_index_for_node(edge, to)?.clone();

            // Contract and store result at `to`
            let contracted = to_tensor.tensordot(&from_tensor, &[(idx_to, idx_from)])
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

// ============================================================================
// Helper functions for TreeTN addition (direct-sum block embedding)
// ============================================================================

/// Embed a source tensor block into a larger destination tensor for f64 data.
///
/// This function places the source data into the appropriate block of the destination
/// tensor, where bond indices are expanded according to `bond_positions`.
///
/// # Arguments
/// * `dest` - Destination tensor data (mutable, will be modified)
/// * `dest_dims` - Dimensions of the destination tensor
/// * `src` - Source tensor data
/// * `src_dims` - Dimensions of the source tensor
/// * `bond_positions` - For each bond axis: (position_in_dest, dim_a, dim_b)
/// * `is_a_block` - If true, embed in the "A" block (0..dim_a); else in "B" block (dim_a..dim_a+dim_b)
fn embed_block_f64(
    dest: &mut [f64],
    dest_dims: &[usize],
    src: &[f64],
    src_dims: &[usize],
    bond_positions: &[(usize, usize, usize)],
    is_a_block: bool,
) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }

    // For each element in src, compute its position in dest
    let src_total: usize = src_dims.iter().product();
    if src.len() != src_total {
        return Err(anyhow::anyhow!(
            "Source data length {} doesn't match dims product {}",
            src.len(),
            src_total
        ));
    }

    // Build stride arrays for both tensors
    let src_strides = compute_strides(src_dims);
    let dest_strides = compute_strides(dest_dims);

    // Create a map from dest axis position to bond info
    let bond_map: HashMap<usize, (usize, usize)> = bond_positions.iter()
        .map(|&(pos, dim_a, dim_b)| (pos, (dim_a, dim_b)))
        .collect();

    // Iterate over all elements in source
    for src_linear in 0..src_total {
        // Convert to multi-index
        let src_multi = linear_to_multi_index(src_linear, &src_strides, src_dims.len());

        // Compute destination multi-index
        let mut dest_multi = Vec::with_capacity(dest_dims.len());
        let mut src_idx = 0;

        for (dest_axis, _dest_dim) in dest_dims.iter().enumerate() {
            if let Some(&(dim_a, _dim_b)) = bond_map.get(&dest_axis) {
                // This is a bond axis
                let src_bond_idx = src_multi[src_idx];
                let dest_bond_idx = if is_a_block {
                    src_bond_idx // A block: 0..dim_a
                } else {
                    src_bond_idx + dim_a // B block: dim_a..dim_a+dim_b
                };
                dest_multi.push(dest_bond_idx);
                src_idx += 1;
            } else {
                // Physical axis - direct copy
                dest_multi.push(src_multi[src_idx]);
                src_idx += 1;
            }
        }

        // Convert dest multi-index to linear
        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);
        dest[dest_linear] = src[src_linear];
    }

    Ok(())
}

/// Embed a source tensor block into a larger destination tensor for Complex64 data.
fn embed_block_c64(
    dest: &mut [Complex64],
    dest_dims: &[usize],
    src: &[Complex64],
    src_dims: &[usize],
    bond_positions: &[(usize, usize, usize)],
    is_a_block: bool,
) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }

    let src_total: usize = src_dims.iter().product();
    if src.len() != src_total {
        return Err(anyhow::anyhow!(
            "Source data length {} doesn't match dims product {}",
            src.len(),
            src_total
        ));
    }

    let src_strides = compute_strides(src_dims);
    let dest_strides = compute_strides(dest_dims);

    let bond_map: HashMap<usize, (usize, usize)> = bond_positions.iter()
        .map(|&(pos, dim_a, dim_b)| (pos, (dim_a, dim_b)))
        .collect();

    for src_linear in 0..src_total {
        let src_multi = linear_to_multi_index(src_linear, &src_strides, src_dims.len());

        let mut dest_multi = Vec::with_capacity(dest_dims.len());
        let mut src_idx = 0;

        for (dest_axis, _dest_dim) in dest_dims.iter().enumerate() {
            if let Some(&(dim_a, _dim_b)) = bond_map.get(&dest_axis) {
                let src_bond_idx = src_multi[src_idx];
                let dest_bond_idx = if is_a_block {
                    src_bond_idx
                } else {
                    src_bond_idx + dim_a
                };
                dest_multi.push(dest_bond_idx);
                src_idx += 1;
            } else {
                dest_multi.push(src_multi[src_idx]);
                src_idx += 1;
            }
        }

        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);
        dest[dest_linear] = src[src_linear];
    }

    Ok(())
}

/// Permute tensor data according to axis permutation.
///
/// Given tensor data in row-major order with shape `dims`, rearrange the data
/// so that axis `i` in the output corresponds to axis `perm[i]` in the input.
///
/// # Arguments
/// * `data` - Input tensor data in row-major order
/// * `dims` - Dimensions of the input tensor
/// * `perm` - Permutation: perm[i] = j means output axis i comes from input axis j
///
/// # Returns
/// Permuted data with shape `[dims[perm[0]], dims[perm[1]], ...]`
fn permute_data_f64(data: &[f64], dims: &[usize], perm: &[usize]) -> Vec<f64> {
    if perm.is_empty() || data.is_empty() {
        return data.to_vec();
    }

    // Check if permutation is identity
    let is_identity = perm.iter().enumerate().all(|(i, &p)| i == p);
    if is_identity {
        return data.to_vec();
    }

    let rank = dims.len();
    let new_dims: Vec<usize> = perm.iter().map(|&i| dims[i]).collect();
    let total: usize = dims.iter().product();

    let src_strides = compute_strides(dims);
    let dest_strides = compute_strides(&new_dims);

    let mut result = vec![0.0_f64; total];

    for src_linear in 0..total {
        let src_multi = linear_to_multi_index(src_linear, &src_strides, rank);

        // Map source multi-index to destination multi-index
        let dest_multi: Vec<usize> = (0..rank).map(|i| src_multi[perm[i]]).collect();
        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);

        result[dest_linear] = data[src_linear];
    }

    result
}

/// Permute tensor data according to axis permutation (Complex64 version).
fn permute_data_c64(data: &[Complex64], dims: &[usize], perm: &[usize]) -> Vec<Complex64> {
    if perm.is_empty() || data.is_empty() {
        return data.to_vec();
    }

    // Check if permutation is identity
    let is_identity = perm.iter().enumerate().all(|(i, &p)| i == p);
    if is_identity {
        return data.to_vec();
    }

    let rank = dims.len();
    let new_dims: Vec<usize> = perm.iter().map(|&i| dims[i]).collect();
    let total: usize = dims.iter().product();

    let src_strides = compute_strides(dims);
    let dest_strides = compute_strides(&new_dims);

    let mut result = vec![Complex64::new(0.0, 0.0); total];

    for src_linear in 0..total {
        let src_multi = linear_to_multi_index(src_linear, &src_strides, rank);

        // Map source multi-index to destination multi-index
        let dest_multi: Vec<usize> = (0..rank).map(|i| src_multi[perm[i]]).collect();
        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);

        result[dest_linear] = data[src_linear];
    }

    result
}
