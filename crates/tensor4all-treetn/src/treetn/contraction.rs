//! Contraction and operations for TreeTN.
//!
//! This module provides methods for:
//! - Replacing internal indices with fresh IDs (`sim_internal_inds`)
//! - Contracting TreeTN to tensor (`contract_to_tensor`)
//! - Zip-up contraction (`contract_zipup`)
//! - Naive contraction (`contract_naive`)
//! - Validation (`validate_ortho_consistency`)

use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::algorithm::CanonicalForm;
use tensor4all_core::{
    AllowedPairs, Canonical, FactorizeAlg, FactorizeOptions, IndexLike, TensorLike,
};

use super::TreeTN;

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
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
    pub fn sim_internal_inds(&self) -> Self {
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
            let new_bond_idx = old_bond_idx.sim();

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
                if let Ok(new_tensor) = tensor_a.replaceind(&old_bond_idx, &new_bond_idx) {
                    *tensor_a = new_tensor;
                }
            }

            // Update tensor at node_b
            if let Some(tensor_b) = result.graph.graph_mut().node_weight_mut(node_b) {
                if let Ok(new_tensor) = tensor_b.replaceind(&old_bond_idx, &new_bond_idx) {
                    *tensor_b = new_tensor;
                }
            }
        }

        result
    }

    /// Contract the TreeTN to a single tensor.
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
    pub fn contract_to_tensor(&self) -> Result<T>
    where
        V: Ord,
    {
        if self.node_count() == 0 {
            return Err(anyhow::anyhow!("Cannot contract empty TreeTN"));
        }

        if self.node_count() == 1 {
            // Single node - just return a clone of its tensor
            let node = self
                .graph
                .graph()
                .node_indices()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No nodes found"))?;
            return self
                .tensor(node)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Tensor not found"));
        }

        // Validate tree structure
        self.validate_tree()
            .context("contract_to_tensor: graph must be a tree")?;

        // Choose a deterministic root (minimum node name)
        let root_name = self
            .graph
            .graph()
            .node_indices()
            .filter_map(|idx| self.graph.node_name(idx).cloned())
            .min()
            .ok_or_else(|| anyhow::anyhow!("No nodes found"))?;
        let root = self
            .graph
            .node_index(&root_name)
            .ok_or_else(|| anyhow::anyhow!("Root node not found"))?;

        // Get edges to process (post-order DFS edges towards root)
        let edges = self.site_index_network.edges_to_canonicalize(None, root);

        // Initialize with original tensors
        let mut tensors: HashMap<NodeIndex, T> = self
            .graph
            .graph()
            .node_indices()
            .filter_map(|n| self.tensor(n).cloned().map(|t| (n, t)))
            .collect();

        // Process each edge: contract tensor at `from` into tensor at `to`
        for (from, to) in edges {
            let from_tensor = tensors
                .remove(&from)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", from))?;
            let to_tensor = tensors
                .remove(&to)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", to))?;

            // Contract and store result at `to`
            // (bond indices are auto-detected via is_contractable)
            let contracted = T::contract(&[&to_tensor, &from_tensor], AllowedPairs::All)
                .context("Failed to contract along edge")?;
            tensors.insert(to, contracted);
        }

        // The root's tensor is the final result
        let result = tensors
            .remove(&root)
            .ok_or_else(|| anyhow::anyhow!("Contraction produced no result"))?;

        // Permute result indices to match canonical site index order:
        // node names sorted, then site indices per node in consistent order
        let mut expected_indices: Vec<T::Index> = Vec::new();
        let mut node_names: Vec<V> = self.node_names();
        node_names.sort();
        for node_name in &node_names {
            if let Some(site_space) = self.site_space(node_name) {
                // Use site indices in insertion order (deterministic from site_space)
                expected_indices.extend(site_space.iter().cloned());
            }
        }

        // Get current index order from result tensor
        let current_indices = result.external_indices();

        // Check if permutation is needed
        if current_indices.len() != expected_indices.len() {
            // This shouldn't happen, but return as-is if sizes don't match
            return Ok(result);
        }

        // Check if already in correct order
        let already_ordered = current_indices
            .iter()
            .zip(expected_indices.iter())
            .all(|(c, e)| c == e);

        if already_ordered {
            return Ok(result);
        }

        // Build permutation: for each expected index, find its position in current indices
        // Then use replaceind to reorder (permuting indices)
        result.permuteinds(&expected_indices)
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
    pub fn contract_zipup(
        &self,
        other: &Self,
        center: &V,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        V: Ord,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        self.contract_zipup_with(other, center, CanonicalForm::Unitary, rtol, max_rank)
    }

    /// Contract two TreeTNs with the same topology using the zip-up algorithm with a specified form.
    ///
    /// See [`contract_zipup`](Self::contract_zipup) for details.
    pub fn contract_zipup_with(
        &self,
        other: &Self,
        center: &V,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        V: Ord,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        let total_start = Instant::now();
        let mut timings: Vec<(&str, std::time::Duration)> = Vec::new();
        let enable_profiling = std::env::var("T4A_PROFILE_CONTRACTION").is_ok();

        // 1. Verify topologies are compatible (same graph structure)
        let t0 = Instant::now();
        if !self.same_topology(other) {
            return Err(anyhow::anyhow!(
                "contract_zipup: networks have incompatible topologies"
            ));
        }
        if enable_profiling {
            timings.push(("Topology verification", t0.elapsed()));
        }

        // 2. Replace internal indices with fresh IDs to avoid collision
        let t0 = Instant::now();
        let tn1 = self.sim_internal_inds();
        let tn2 = other.sim_internal_inds();
        if enable_profiling {
            timings.push(("Sim internal indices", t0.elapsed()));
        }

        // 3. Get traversal edges from leaves to center
        let t0 = Instant::now();
        let edges = tn1
            .edges_to_canonicalize_by_names(center)
            .ok_or_else(|| anyhow::anyhow!("contract_zipup: center node {:?} not found", center))?;
        if enable_profiling {
            timings.push(("Get edges", t0.elapsed()));
        }

        if edges.is_empty() && self.node_count() == 1 {
            // Single node case: just contract the two tensors
            let node_idx = tn1
                .graph
                .graph()
                .node_indices()
                .next()
                .ok_or_else(|| anyhow::anyhow!("contract_zipup: no nodes found"))?;
            let t1 = tn1
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("contract_zipup: tensor not found in tn1"))?;
            let t2 = tn2
                .tensor(
                    tn2.graph.graph().node_indices().next().ok_or_else(|| {
                        anyhow::anyhow!("contract_zipup: tensor not found in tn2")
                    })?,
                )
                .ok_or_else(|| anyhow::anyhow!("contract_zipup: tensor not found"))?;

            // Contract t1 and t2 - common indices are auto-detected via is_contractable
            let result = T::contract(&[t1, t2], AllowedPairs::All)?;
            let mut result_tn = Self::new();
            let node_name = tn1
                .graph
                .node_name(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Node name not found"))?
                .clone();
            result_tn.add_tensor(node_name, result)?;
            return Ok(result_tn);
        }

        // 4. Initialize result tensors (contract corresponding node tensors)
        let t0 = Instant::now();
        let mut result_tensors: HashMap<V, T> = HashMap::new();
        let mut site_contraction_time = std::time::Duration::ZERO;

        for node_name in tn1.node_names() {
            let node1 = tn1
                .node_index(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tn1", node_name))?;
            let node2 = tn2
                .node_index(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tn2", node_name))?;

            let t1 = tn1.tensor(node1).ok_or_else(|| {
                anyhow::anyhow!("Tensor not found for node {:?} in tn1", node_name)
            })?;
            let t2 = tn2.tensor(node2).ok_or_else(|| {
                anyhow::anyhow!("Tensor not found for node {:?} in tn2", node_name)
            })?;

            // T::contract auto-contracts all is_contractable pairs
            let t_contract = Instant::now();
            let contracted = T::contract(&[t1, t2], AllowedPairs::All)?;
            site_contraction_time += t_contract.elapsed();

            result_tensors.insert(node_name, contracted);
        }
        if enable_profiling {
            timings.push(("Initial site contraction", t0.elapsed()));
            timings.push(("  - Site contraction (sum)", site_contraction_time));
        }

        // 5. Process edges from leaves towards center (zip-up)
        let t0 = Instant::now();
        let alg = match form {
            CanonicalForm::Unitary => FactorizeAlg::SVD,
            CanonicalForm::LU => FactorizeAlg::LU,
            CanonicalForm::CI => FactorizeAlg::CI,
        };

        let mut factorize_time = std::time::Duration::ZERO;
        let mut edge_contraction_time = std::time::Duration::ZERO;
        let mut edge_count: usize = 0;

        for (child_name, parent_name) in &edges {
            edge_count += 1;

            let child_tensor = result_tensors
                .remove(child_name)
                .ok_or_else(|| anyhow::anyhow!("Child tensor {:?} not found", child_name))?;

            // Get the bond indices between child and parent from tn1 (bond1) and tn2 (bond2)
            let edge1 = tn1.edge_between(child_name, parent_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "Edge not found between {:?} and {:?} in tn1",
                    child_name,
                    parent_name
                )
            })?;
            let edge2 = tn2.edge_between(child_name, parent_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "Edge not found between {:?} and {:?} in tn2",
                    child_name,
                    parent_name
                )
            })?;

            let bond1 = tn1
                .bond_index(edge1)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge in tn1"))?
                .clone();
            let bond2 = tn2
                .bond_index(edge2)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge in tn2"))?
                .clone();

            // Keep everything on the child except the two bond indices to the parent.
            // This includes site indices and any new bonds to children created earlier.
            let left_inds: Vec<_> = child_tensor
                .external_indices()
                .into_iter()
                .filter(|idx| *idx.id() != *bond1.id() && *idx.id() != *bond2.id())
                .collect();

            if left_inds.is_empty() {
                // Inner-product-like case: child has only bond indices, absorb it directly.
                let parent_tensor = result_tensors
                    .remove(parent_name)
                    .ok_or_else(|| anyhow::anyhow!("Parent tensor {:?} not found", parent_name))?;

                let contracted = T::contract(&[&parent_tensor, &child_tensor], AllowedPairs::All)?;
                result_tensors.insert(parent_name.clone(), contracted);
                continue;
            }

            let factorize_options = FactorizeOptions {
                alg,
                canonical: Canonical::Left,
                rtol,
                max_rank,
            };

            let t_factorize = Instant::now();
            let factorize_result = child_tensor
                .factorize(&left_inds, &factorize_options)
                .map_err(|e| {
                    anyhow::anyhow!("Factorization failed at node {:?}: {}", child_name, e)
                })?;
            factorize_time += t_factorize.elapsed();

            // Store left factor as the updated child tensor
            result_tensors.insert(child_name.clone(), factorize_result.left);

            // Absorb right factor into parent
            let parent_tensor = result_tensors
                .remove(parent_name)
                .ok_or_else(|| anyhow::anyhow!("Parent tensor {:?} not found", parent_name))?;

            let t_contract = Instant::now();
            let contracted = T::contract(
                &[&parent_tensor, &factorize_result.right],
                AllowedPairs::All,
            )?;
            edge_contraction_time += t_contract.elapsed();
            result_tensors.insert(parent_name.clone(), contracted);
        }

        if enable_profiling {
            timings.push(("Edge processing (total)", t0.elapsed()));
            timings.push(("  - Factorize (sum)", factorize_time));
            timings.push(("  - Edge contraction (sum)", edge_contraction_time));
            eprintln!("  - Number of edges processed: {}", edge_count);
        }

        // 6. Build result TreeTN
        let t0 = Instant::now();
        let mut result = Self::new();

        // Add tensors for nodes that still have tensors
        // (Some nodes may have been absorbed during the zip-up process in inner product cases)
        let remaining_nodes: Vec<_> = tn1
            .node_names()
            .into_iter()
            .filter(|name| result_tensors.contains_key(name))
            .collect();

        for node_name in &remaining_nodes {
            let tensor = result_tensors.remove(node_name).ok_or_else(|| {
                anyhow::anyhow!("Result tensor not found for node {:?}", node_name)
            })?;
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
            let common = find_common_indices(tensor_a, tensor_b);
            if let Some(bond_idx) = common.first() {
                result.connect_internal(node_a, bond_idx, node_b, bond_idx)?;
            }
        }

        // Set canonical center (only if it exists in result)
        if result.node_index(center).is_some() {
            result.set_canonical_center(std::iter::once(center.clone()))?;
        }
        if enable_profiling {
            timings.push(("Build result TreeTN", t0.elapsed()));
        }

        // Print profiling results if enabled
        if enable_profiling {
            let total_time = total_start.elapsed();
            eprintln!("=== contract_zipup Profiling ===");
            for (name, duration) in &timings {
                if duration.as_secs() > 0 || duration.as_millis() > 0 {
                    let percentage = (duration.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
                    eprintln!("  {}: {:?} ({:.1}%)", name, duration, percentage);
                } else {
                    eprintln!("  {}: {:?}", name, duration);
                }
            }
            eprintln!("  Total: {:?}", total_time);
            eprintln!("=================================");
        }

        Ok(result)
    }

    /// Contract two TreeTNs using zip-up algorithm with accumulated intermediate tensors.
    ///
    /// This is an improved version of zip-up contraction that maintains intermediate tensors
    /// (environment tensors) as it processes from leaves towards the root, similar to
    /// ITensors.jl's MPO zip-up algorithm.
    ///
    /// # Algorithm
    /// 1. Process leaves: contract A[leaf] * B[leaf], factorize, store R at parent
    /// 2. Process internal nodes: contract [R_accumulated..., A[node], B[node]], factorize, store R_new at parent
    /// 3. Process root: contract [R_list..., A[root], B[root]], store as final result
    /// 4. Set canonical center
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to contract with (must have same topology)
    /// * `center` - The center node name towards which to contract
    /// * `form` - Canonical form (Unitary/LU/CI)
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    ///
    /// # Returns
    /// The contracted TreeTN result, or an error if topologies don't match or contraction fails.
    pub fn contract_zipup_tree_accumulated(
        &self,
        other: &Self,
        center: &V,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        V: Ord,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        // 1. Verify topologies are compatible
        if !self.same_topology(other) {
            return Err(anyhow::anyhow!(
                "contract_zipup_tree_accumulated: networks have incompatible topologies"
            ));
        }

        // 2. Replace internal indices with fresh IDs to avoid collision
        let tn_a = self.sim_internal_inds();
        let tn_b = other.sim_internal_inds();

        // 3. Get traversal edges from leaves to center (post-order DFS)
        let edges = tn_a.edges_to_canonicalize_by_names(center).ok_or_else(|| {
            anyhow::anyhow!(
                "contract_zipup_tree_accumulated: center node {:?} not found",
                center
            )
        })?;

        // 4. Handle single node case
        if edges.is_empty() && self.node_count() == 1 {
            let node_idx = tn_a.graph.graph().node_indices().next().ok_or_else(|| {
                anyhow::anyhow!("contract_zipup_tree_accumulated: no nodes found")
            })?;
            let t_a = tn_a.tensor(node_idx).ok_or_else(|| {
                anyhow::anyhow!("contract_zipup_tree_accumulated: tensor not found in tn_a")
            })?;
            let t_b = tn_b
                .tensor(tn_b.graph.graph().node_indices().next().ok_or_else(|| {
                    anyhow::anyhow!("contract_zipup_tree_accumulated: tensor not found in tn_b")
                })?)
                .ok_or_else(|| {
                    anyhow::anyhow!("contract_zipup_tree_accumulated: tensor not found in tn_b")
                })?;

            let contracted = T::contract(&[t_a, t_b], AllowedPairs::All)?;
            let node_name = tn_a.graph.node_name(node_idx).ok_or_else(|| {
                anyhow::anyhow!("contract_zipup_tree_accumulated: node name not found")
            })?;

            let mut result = TreeTN::new();
            result.add_tensor(node_name.clone(), contracted)?;
            result.set_canonical_center(std::iter::once(center.clone()))?;
            return Ok(result);
        }

        // 5. Initialize intermediate tensors storage: HashMap<node_name, Vec<intermediate_tensor>>
        let mut intermediate_tensors: HashMap<V, Vec<T>> = HashMap::new();

        // 6. Initialize result tensors: HashMap<node_name, tensor>
        let mut result_tensors: HashMap<V, T> = HashMap::new();

        // 7. Determine which nodes are leaves (for processing logic)
        let root_name = center.clone();

        // Helper: Get bond index between two nodes
        let get_bond_index = |tn: &TreeTN<T, V>, node_a: &V, node_b: &V| -> Result<T::Index> {
            let edge = tn.edge_between(node_a, node_b).ok_or_else(|| {
                anyhow::anyhow!("Edge not found between {:?} and {:?}", node_a, node_b)
            })?;
            tn.bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge"))
                .cloned()
        };

        // 8. Set up factorization options based on form
        let alg = match form {
            CanonicalForm::Unitary => FactorizeAlg::SVD,
            CanonicalForm::LU => FactorizeAlg::LU,
            CanonicalForm::CI => FactorizeAlg::CI,
        };

        let factorize_options = FactorizeOptions {
            alg,
            canonical: Canonical::Left,
            rtol,
            max_rank,
        };

        // 9. Process edges from leaves towards root
        for (source_name, destination_name) in &edges {
            // Get tensors from both networks
            let node_a_idx = tn_a
                .node_index(source_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tn_a", source_name))?;
            let node_b_idx = tn_b
                .node_index(source_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tn_b", source_name))?;

            let tensor_a = tn_a
                .tensor(node_a_idx)
                .ok_or_else(|| {
                    anyhow::anyhow!("Tensor not found for node {:?} in tn_a", source_name)
                })?
                .clone();
            let tensor_b = tn_b
                .tensor(node_b_idx)
                .ok_or_else(|| {
                    anyhow::anyhow!("Tensor not found for node {:?} in tn_b", source_name)
                })?
                .clone();

            // Check if this is a leaf node (no intermediate tensors accumulated yet)
            let is_leaf = !intermediate_tensors.contains_key(source_name)
                || intermediate_tensors
                    .get(source_name)
                    .map(|v| v.is_empty())
                    .unwrap_or(true);

            let c_temp = if is_leaf {
                // Leaf node: contract A[source] * B[source]
                T::contract(&[&tensor_a, &tensor_b], AllowedPairs::All)
                    .context("Failed to contract leaf tensors")?
            } else {
                // Internal node: contract [R_accumulated..., A[source], B[source]]
                let mut tensor_list = Vec::new();
                if let Some(r_list) = intermediate_tensors.remove(source_name) {
                    tensor_list.extend(r_list);
                }
                tensor_list.push(tensor_a);
                tensor_list.push(tensor_b);
                let tensor_refs: Vec<&T> = tensor_list.iter().collect();
                T::contract(&tensor_refs, AllowedPairs::All)
                    .context("Failed to contract internal node tensors")?
            };

            // Factorize child tensor and pass the right factor to destination (even if destination is root)
            let bond_to_dest_a = get_bond_index(&tn_a, source_name, destination_name)
                .context("Failed to get bond index to destination in tn_a")?;
            let bond_to_dest_b = get_bond_index(&tn_b, source_name, destination_name)
                .context("Failed to get bond index to destination in tn_b")?;

            // left_inds = all indices except the two parent bonds (keep site + child bonds)
            let left_inds: Vec<_> = c_temp
                .external_indices()
                .into_iter()
                .filter(|idx| {
                    *idx.id() != *bond_to_dest_a.id() && *idx.id() != *bond_to_dest_b.id()
                })
                .collect();

            if left_inds.is_empty() {
                // If no left indices remain, pass the tensor directly to destination
                intermediate_tensors
                    .entry(destination_name.clone())
                    .or_default()
                    .push(c_temp);
                continue;
            }

            let factorize_result = c_temp
                .factorize(&left_inds, &factorize_options)
                .context("Failed to factorize")?;

            // Store left factor as result tensor for source node
            result_tensors.insert(source_name.clone(), factorize_result.left);

            // Store right factor (intermediate tensor R) at destination
            intermediate_tensors
                .entry(destination_name.clone())
                .or_default()
                .push(factorize_result.right);

            // Note: bond index update will be handled when building the result TreeTN
        }

        // 9.5. Process root node (if it has intermediate tensors accumulated)
        if let Some(r_list) = intermediate_tensors.remove(&root_name) {
            // Get root tensors from both networks
            let root_a_idx = tn_a
                .node_index(&root_name)
                .ok_or_else(|| anyhow::anyhow!("Root node {:?} not found in tn_a", root_name))?;
            let root_b_idx = tn_b
                .node_index(&root_name)
                .ok_or_else(|| anyhow::anyhow!("Root node {:?} not found in tn_b", root_name))?;

            let root_tensor_a = tn_a
                .tensor(root_a_idx)
                .ok_or_else(|| anyhow::anyhow!("Root tensor not found in tn_a"))?
                .clone();
            let root_tensor_b = tn_b
                .tensor(root_b_idx)
                .ok_or_else(|| anyhow::anyhow!("Root tensor not found in tn_b"))?
                .clone();

            // Contract [R_list..., A[root], B[root]]
            let mut tensor_list = r_list;
            tensor_list.push(root_tensor_a);
            tensor_list.push(root_tensor_b);
            let tensor_refs: Vec<&T> = tensor_list.iter().collect();
            let root_result = T::contract(&tensor_refs, AllowedPairs::All)
                .context("Failed to contract root node tensors")?;

            // Store root result (no factorization needed)
            result_tensors.insert(root_name.clone(), root_result);
        } else {
            // No intermediate tensors: root is a single node or already processed
            // Check if root tensors need to be contracted
            if !result_tensors.contains_key(&root_name) {
                let root_a_idx = tn_a.node_index(&root_name).ok_or_else(|| {
                    anyhow::anyhow!("Root node {:?} not found in tn_a", root_name)
                })?;
                let root_b_idx = tn_b.node_index(&root_name).ok_or_else(|| {
                    anyhow::anyhow!("Root node {:?} not found in tn_b", root_name)
                })?;

                let root_tensor_a = tn_a
                    .tensor(root_a_idx)
                    .ok_or_else(|| anyhow::anyhow!("Root tensor not found in tn_a"))?;
                let root_tensor_b = tn_b
                    .tensor(root_b_idx)
                    .ok_or_else(|| anyhow::anyhow!("Root tensor not found in tn_b"))?;

                let root_result = T::contract(&[root_tensor_a, root_tensor_b], AllowedPairs::All)
                    .context("Failed to contract root node tensors")?;

                result_tensors.insert(root_name.clone(), root_result);
            }
        }

        // 10. Build result TreeTN
        let mut result = TreeTN::new();

        // Add all result tensors
        for (node_name, tensor) in result_tensors {
            result.add_tensor(node_name, tensor)?;
        }

        // Connect nodes based on original topology
        for (source_name, destination_name) in &edges {
            if let (Some(node_a_idx), Some(node_b_idx)) = (
                result.node_index(source_name),
                result.node_index(destination_name),
            ) {
                let tensor_a = result.tensor(node_a_idx).unwrap();
                let tensor_b = result.tensor(node_b_idx).unwrap();

                // Find the common index (should be the bond index)
                use tensor4all_core::index_ops::common_inds;
                let indices_a = tensor_a.external_indices();
                let indices_b = tensor_b.external_indices();
                let common = common_inds::<T::Index>(&indices_a, &indices_b);
                if let Some(bond_idx) = common.first() {
                    result.connect_internal(node_a_idx, bond_idx, node_b_idx, bond_idx)?;
                }
            }
        }

        // 11. Set canonical center
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
    pub fn contract_naive(&self, other: &Self) -> Result<T>
    where
        V: Ord,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        // 1. Verify topologies are compatible
        if !self.same_topology(other) {
            return Err(anyhow::anyhow!(
                "contract_naive: networks have incompatible topologies"
            ));
        }

        // 2. Replace internal indices with fresh IDs to avoid collision
        let tn1 = self.sim_internal_inds();
        let tn2 = other.sim_internal_inds();

        // 3. Convert both networks to full tensors
        let tensor1 = tn1
            .contract_to_tensor()
            .map_err(|e| anyhow::anyhow!("contract_naive: failed to contract tn1: {}", e))?;
        let tensor2 = tn2
            .contract_to_tensor()
            .map_err(|e| anyhow::anyhow!("contract_naive: failed to contract tn2: {}", e))?;

        // 4. Contract along common indices
        // T::contract auto-contracts all is_contractable pairs
        T::contract(&[&tensor1, &tensor2], AllowedPairs::All)
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
                .context(
                    "validate_ortho_consistency: canonical_center empty implies no ortho_towards",
                );
            }
            return Ok(());
        }

        // Validate all canonical_center nodes exist and convert to NodeIndex
        let mut center_indices = HashSet::new();
        for c in &self.canonical_center {
            let idx = self
                .graph
                .node_index(c)
                .ok_or_else(|| anyhow::anyhow!("canonical_center node {:?} does not exist", c))?;
            center_indices.insert(idx);
        }

        // Check canonical_center connectivity
        if !self.site_index_network.is_connected_subset(&center_indices) {
            return Err(anyhow::anyhow!("canonical_center is not connected")).context(
                "validate_ortho_consistency: canonical_center must form a connected subtree",
            );
        }

        // Get expected edges from edges_to_canonicalize_to_region
        let expected_edges = self
            .site_index_network
            .edges_to_canonicalize_to_region(&center_indices);

        // Build a set of expected (bond, expected_direction) pairs
        let mut expected_directions: HashMap<T::Index, V> = HashMap::new();
        for (src, dst) in expected_edges.iter() {
            // Find the edge between src and dst
            let edge = self
                .graph
                .graph()
                .find_edge(*src, *dst)
                .or_else(|| self.graph.graph().find_edge(*dst, *src))
                .ok_or_else(|| anyhow::anyhow!("Edge not found between {:?} and {:?}", src, dst))?;

            let bond = self
                .bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge"))?
                .clone();

            // The expected ortho_towards direction is dst (towards center)
            let dst_name = self
                .graph
                .node_name(*dst)
                .ok_or_else(|| anyhow::anyhow!("Node name not found for {:?}", dst))?
                .clone();

            expected_directions.insert(bond, dst_name);
        }

        // Verify all expected directions are present in ortho_towards
        for (bond, expected_dir) in &expected_directions {
            match self.ortho_towards.get(bond) {
                Some(actual_dir) => {
                    if actual_dir != expected_dir {
                        return Err(anyhow::anyhow!(
                            "ortho_towards for bond {:?} points to {:?} but expected {:?}",
                            bond,
                            actual_dir,
                            expected_dir
                        ))
                        .context("validate_ortho_consistency: wrong direction");
                    }
                }
                None => {
                    return Err(anyhow::anyhow!(
                        "ortho_towards for bond {:?} is missing, expected to point to {:?}",
                        bond,
                        expected_dir
                    ))
                    .context("validate_ortho_consistency: missing ortho_towards");
                }
            }
        }

        // Verify no unexpected bond ortho_towards entries
        // (site index ortho_towards are allowed even if not in expected_directions)
        let bond_indices: HashSet<T::Index> = self
            .graph
            .graph()
            .edge_indices()
            .filter_map(|e| self.bond_index(e))
            .cloned()
            .collect();

        for idx in self.ortho_towards.keys() {
            if bond_indices.contains(idx) && !expected_directions.contains_key(idx) {
                // This is a bond inside the canonical_center - should not have ortho_towards
                return Err(anyhow::anyhow!(
                    "Unexpected ortho_towards for bond {:?} (inside canonical_center)",
                    idx
                ))
                .context(
                    "validate_ortho_consistency: bonds inside center should not have ortho_towards",
                );
            }
        }

        Ok(())
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Find common indices between two tensors (by ID).
fn find_common_indices<T: TensorLike>(a: &T, b: &T) -> Vec<T::Index>
where
    <T::Index as IndexLike>::Id: Eq + std::hash::Hash,
{
    let a_ids: HashSet<_> = a
        .external_indices()
        .iter()
        .map(|i| i.id().clone())
        .collect();
    b.external_indices()
        .into_iter()
        .filter(|i| a_ids.contains(i.id()))
        .collect()
}

// ============================================================================
// Contraction Method Dispatcher
// ============================================================================

use super::fit::FitContractionOptions;

/// Contraction method for TreeTN operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContractionMethod {
    /// Zip-up contraction (faster, one-pass).
    #[default]
    Zipup,
    /// Fit/variational contraction (iterative optimization).
    Fit,
    /// Naive contraction: contract to full tensor, then decompose back to TreeTN.
    /// Useful for debugging and testing, but O(exp(n)) in memory.
    Naive,
}

/// Options for the generic contract function.
#[derive(Debug, Clone)]
pub struct ContractionOptions {
    /// Contraction method to use.
    pub method: ContractionMethod,
    /// Maximum bond dimension (optional).
    pub max_rank: Option<usize>,
    /// Relative tolerance for truncation (optional).
    pub rtol: Option<f64>,
    /// Number of full sweeps for Fit method.
    ///
    /// A full sweep visits each edge twice (forward and backward) using an Euler tour.
    pub nfullsweeps: usize,
    /// Convergence tolerance for Fit method (None = fixed sweeps).
    pub convergence_tol: Option<f64>,
    /// Factorization algorithm for Fit method.
    pub factorize_alg: FactorizeAlg,
}

impl Default for ContractionOptions {
    fn default() -> Self {
        Self {
            method: ContractionMethod::default(),
            max_rank: None,
            rtol: None,
            nfullsweeps: 1,
            convergence_tol: None,
            factorize_alg: FactorizeAlg::default(),
        }
    }
}

impl ContractionOptions {
    /// Create options with specified method.
    pub fn new(method: ContractionMethod) -> Self {
        Self {
            method,
            ..Default::default()
        }
    }

    /// Create options for zipup contraction.
    pub fn zipup() -> Self {
        Self::new(ContractionMethod::Zipup)
    }

    /// Create options for fit contraction.
    pub fn fit() -> Self {
        Self::new(ContractionMethod::Fit)
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

    /// Set convergence tolerance for Fit method.
    pub fn with_convergence_tol(mut self, tol: f64) -> Self {
        self.convergence_tol = Some(tol);
        self
    }

    /// Set factorization algorithm for Fit method.
    pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self {
        self.factorize_alg = alg;
        self
    }
}

/// Contract two TreeTNs using the specified method.
///
/// This is the main entry point for TreeTN contraction. It dispatches to the
/// appropriate algorithm based on the options.
pub fn contract<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: ContractionOptions,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    match options.method {
        ContractionMethod::Zipup => {
            tn_a.contract_zipup(tn_b, center, options.rtol, options.max_rank)
        }
        ContractionMethod::Fit => {
            let fit_options = FitContractionOptions::new(options.nfullsweeps)
                .with_factorize_alg(options.factorize_alg);
            let fit_options = if let Some(max_rank) = options.max_rank {
                fit_options.with_max_rank(max_rank)
            } else {
                fit_options
            };
            let fit_options = if let Some(rtol) = options.rtol {
                fit_options.with_rtol(rtol)
            } else {
                fit_options
            };
            let fit_options = if let Some(tol) = options.convergence_tol {
                fit_options.with_convergence_tol(tol)
            } else {
                fit_options
            };
            super::fit::contract_fit(tn_a, tn_b, center, fit_options)
        }
        ContractionMethod::Naive => {
            contract_naive_to_treetn(tn_a, tn_b, center, options.max_rank, options.rtol)
        }
    }
}

/// Contract two TreeTNs using naive contraction, then decompose back to TreeTN.
///
/// This method:
/// 1. Contracts both networks to full tensors
/// 2. Contracts the tensors along common (site) indices
/// 3. Decomposes the result back to a TreeTN using the original topology
///
/// This is O(exp(n)) in memory and is primarily useful for debugging and testing.
pub fn contract_naive_to_treetn<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    _max_rank: Option<usize>,
    _rtol: Option<f64>,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    // 1. Contract to full tensor using existing contract_naive
    let contracted_tensor = tn_a.contract_naive(tn_b)?;

    // 2. Build topology from tn_a's structure and decompose
    use super::decompose::factorize_tensor_to_treetn_with;

    // Build topology
    let mut nodes: HashMap<V, Vec<usize>> = HashMap::new();
    let mut idx_position = 0usize;

    // Collect node names in sorted order for deterministic index assignment
    let mut node_names: Vec<_> = tn_a.node_names();
    node_names.sort();

    for node_name in &node_names {
        let site_space = tn_a
            .site_index_network
            .site_space(node_name)
            .ok_or_else(|| anyhow::anyhow!("Site space not found for node {:?}", node_name))?;

        // Each site index becomes a position in the contracted tensor
        let positions: Vec<usize> = (idx_position..idx_position + site_space.len()).collect();
        idx_position += site_space.len();
        nodes.insert(node_name.clone(), positions);
    }

    // Get edges from the graph
    let edges: Vec<(V, V)> = tn_a
        .graph
        .graph()
        .edge_indices()
        .filter_map(|e| {
            let (src, dst) = tn_a.graph.graph().edge_endpoints(e)?;
            let src_name = tn_a.graph.node_name(src)?;
            let dst_name = tn_a.graph.node_name(dst)?;
            Some((src_name.clone(), dst_name.clone()))
        })
        .collect();

    let topology = super::decompose::TreeTopology::new(nodes, edges);

    // 3. Decompose back to TreeTN
    let mut result =
        factorize_tensor_to_treetn_with(&contracted_tensor, &topology, FactorizeAlg::SVD)?;

    // Set canonical center
    if result.node_index(center).is_some() {
        result.set_canonical_center(std::iter::once(center.clone()))?;
    }

    Ok(result)
}
