//! Contraction and operations for TreeTN.
//!
//! This module provides methods for:
//! - Replacing internal indices with fresh IDs (`sim_internal_inds`)
//! - Contracting TreeTN to tensor (`contract_to_tensor`)
//! - Zip-up contraction (`contract_zipup`)
//! - Validation (`validate_ortho_consistency`)

use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::{Context, Result};

use tensor4all_core::{Canonical, FactorizeAlg, FactorizeOptions, IndexLike, TensorLike};
use crate::algorithm::CanonicalForm;

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

            // Find the edge and get bond index
            let edge = self
                .graph
                .graph()
                .find_edge(from, to)
                .or_else(|| self.graph.graph().find_edge(to, from))
                .ok_or_else(|| anyhow::anyhow!("Edge not found between {:?} and {:?}", from, to))?;

            // Both endpoints share the same bond index
            let bond_idx = self
                .bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge"))?
                .clone();

            // Contract and store result at `to`
            let contracted = to_tensor
                .tensordot(&from_tensor, &[(bond_idx.clone(), bond_idx)])
                .context("Failed to contract along edge")?;
            tensors.insert(to, contracted);
        }

        // The root's tensor is the final result
        tensors
            .remove(&root)
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
    pub fn contract_zipup(
        &self,
        other: &Self,
        center: &V,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
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
        <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        // 1. Verify topologies are compatible (same graph structure)
        if !self.same_topology(other) {
            return Err(anyhow::anyhow!(
                "contract_zipup: networks have incompatible topologies"
            ));
        }

        // 2. Replace internal indices with fresh IDs to avoid collision
        let tn1 = self.sim_internal_inds();
        let tn2 = other.sim_internal_inds();

        // 3. Get traversal edges from leaves to center
        let edges = tn1
            .edges_to_canonicalize_by_names(center)
            .ok_or_else(|| anyhow::anyhow!("contract_zipup: center node {:?} not found", center))?;

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

            // Contract along common indices (site indices)
            let common = find_common_indices(t1, t2);
            let result = if common.is_empty() {
                // TODO: Outer product - need to add to TensorLike
                return Err(anyhow::anyhow!("Outer product not yet supported in TensorLike"));
            } else {
                let pairs: Vec<_> = common
                    .iter()
                    .map(|idx| (idx.clone(), idx.clone()))
                    .collect();
                t1.tensordot(t2, &pairs)?
            };
            let mut result_tn = Self::new();
            let node_name = tn1
                .graph
                .node_name(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Node name not found"))?
                .clone();
            result_tn.add_tensor(node_name, result)?;
            return Ok(result_tn);
        }

        // 4. Initialize result tensors (start with contracted node tensors)
        let mut result_tensors: HashMap<V, T> = HashMap::new();

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

            // Contract along site indices (external indices)
            let site_ids: HashSet<_> = tn1
                .site_index_network
                .site_space(&node_name)
                .map(|s| s.iter().map(|i| i.id().clone()).collect())
                .unwrap_or_default();

            let common: Vec<_> = find_common_indices(t1, t2)
                .into_iter()
                .filter(|idx| site_ids.contains(idx.id()))
                .collect();

            let contracted = if common.is_empty() {
                // TODO: Outer product - need to add to TensorLike
                return Err(anyhow::anyhow!("Outer product not yet supported in TensorLike"));
            } else {
                let pairs: Vec<_> = common
                    .iter()
                    .map(|idx| (idx.clone(), idx.clone()))
                    .collect();
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
            let child_tensor = result_tensors
                .remove(child_name)
                .ok_or_else(|| anyhow::anyhow!("Child tensor {:?} not found", child_name))?;

            // Get the bond index between child and parent from tn1 (bond1) and tn2 (bond2)
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

            // Factorize: left_inds = all indices except bond1 and bond2 (towards parent)
            let left_inds: Vec<_> = child_tensor
                .external_indices()
                .into_iter()
                .filter(|idx| *idx.id() != *bond1.id() && *idx.id() != *bond2.id())
                .collect();

            if left_inds.is_empty() {
                // All indices are bond indices - just absorb into parent
                let parent_tensor = result_tensors
                    .remove(parent_name)
                    .ok_or_else(|| anyhow::anyhow!("Parent tensor {:?} not found", parent_name))?;

                let contracted = parent_tensor.tensordot(
                    &child_tensor,
                    &[(bond1.clone(), bond1), (bond2.clone(), bond2)],
                )?;
                result_tensors.insert(parent_name.clone(), contracted);
                continue;
            }

            let factorize_options = FactorizeOptions {
                alg,
                canonical: Canonical::Left,
                rtol,
                max_rank,
            };

            let factorize_result = child_tensor.factorize(&left_inds, &factorize_options)
                .map_err(|e| {
                    anyhow::anyhow!("Factorization failed at node {:?}: {}", child_name, e)
                })?;

            // Store left factor as child
            result_tensors.insert(child_name.clone(), factorize_result.left);

            // Contract right factor into parent
            let parent_tensor = result_tensors
                .remove(parent_name)
                .ok_or_else(|| anyhow::anyhow!("Parent tensor {:?} not found", parent_name))?;

            let contracted = parent_tensor.tensordot(
                &factorize_result.right,
                &[(bond1.clone(), bond1), (bond2.clone(), bond2)],
            )?;
            result_tensors.insert(parent_name.clone(), contracted);
        }

        // 6. Build result TreeTN
        let mut result = Self::new();

        // Add tensors for nodes that still have tensors
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
        for (a, b) in tn1.site_index_network.edges() {
            let node_a = match result.node_index(&a) {
                Some(idx) => idx,
                None => continue,
            };
            let node_b = match result.node_index(&b) {
                Some(idx) => idx,
                None => continue,
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

        Ok(result)
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
        let bond_indices: HashSet<T::Index> = self
            .graph
            .graph()
            .edge_indices()
            .filter_map(|e| self.bond_index(e))
            .cloned()
            .collect();

        for (idx, _) in &self.ortho_towards {
            if bond_indices.contains(idx) && !expected_directions.contains_key(idx) {
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
    let a_ids: HashSet<_> = a.external_indices().iter().map(|i| i.id().clone()).collect();
    b.external_indices()
        .into_iter()
        .filter(|i| a_ids.contains(i.id()))
        .collect()
}

// ============================================================================
// Contraction Method Dispatcher
// ============================================================================

// TODO: Re-enable these after fit module is fully working
// use super::fit::{contract_fit, FitContractionOptions};

/// Contraction method for TreeTN operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContractionMethod {
    /// Zip-up contraction (faster, one-pass).
    #[default]
    Zipup,
    /// Fit/variational contraction (iterative optimization).
    Fit,
    /// Naive contraction: contract to full tensor, then decompose back to TreeTN.
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
    /// Number of sweeps for Fit method.
    pub nsweeps: usize,
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
            nsweeps: 2,
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

    /// Set number of sweeps for Fit method.
    pub fn with_nsweeps(mut self, nsweeps: usize) -> Self {
        self.nsweeps = nsweeps;
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
///
/// Currently only Zipup method is supported.
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
            // TODO: Re-enable after fit module is fully working
            Err(anyhow::anyhow!("Fit contraction not yet implemented for TensorLike"))
        }
        ContractionMethod::Naive => {
            // TODO: Implement naive contraction for TensorLike
            Err(anyhow::anyhow!("Naive contraction not yet implemented for TensorLike"))
        }
    }
}
