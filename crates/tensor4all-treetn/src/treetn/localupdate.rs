//! Local update operations for TreeTN.
//!
//! This module provides APIs for:
//! - Extracting a sub-tree from a TreeTN (creating a new TreeTN object)
//! - Replacing a sub-tree with another TreeTN of the same appearance
//! - Generating sweep plans for local update algorithms (truncation, fitting)
//!
//! These operations are fundamental for local update algorithms in tensor networks.

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{Context, Result};

use tensor4all_core::{AllowedPairs, IndexLike, TensorLike};

use super::TreeTN;
use crate::node_name_network::NodeNameNetwork;

// ============================================================================
// Local Update Sweep Plan
// ============================================================================

/// A single step in a local update sweep.
///
/// Each step specifies:
/// - Which nodes to extract for local update
/// - Where the canonical center should be after the update
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalUpdateStep<V> {
    /// Nodes to extract for local update.
    /// For nsite=1: single node
    /// For nsite=2: two adjacent nodes (an edge)
    pub nodes: Vec<V>,

    /// The canonical center after this update step.
    /// For nsite=1: same as `nodes[0]`.
    /// For nsite=2: the node in the direction of the next step.
    pub new_center: V,
}

/// A complete sweep plan for local updates.
///
/// Generated from an Euler tour, this plan specifies the sequence of
/// local update operations for algorithms like truncation and fitting.
///
/// # Sweep Direction
/// The sweep follows an Euler tour starting from the root, visiting each
/// edge twice (forward and backward). This ensures all bonds are updated
/// in both directions, which is essential for algorithms like DMRG/TEBD.
///
/// # nsite Parameter
/// - `nsite=1`: Single-site updates. Each step extracts one node.
/// - `nsite=2`: Two-site updates. Each step extracts two adjacent nodes (an edge).
///
/// Two-site updates are more expensive but can change bond dimensions and
/// are necessary for algorithms like TDVP-2 or two-site DMRG.
#[derive(Debug, Clone)]
pub struct LocalUpdateSweepPlan<V> {
    /// The sequence of update steps.
    pub steps: Vec<LocalUpdateStep<V>>,

    /// Number of sites per update (1 or 2).
    pub nsite: usize,
}

impl<V> LocalUpdateSweepPlan<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Generate a sweep plan from a TreeTN's topology.
    ///
    /// Convenience method that extracts the NodeNameNetwork topology from a TreeTN.
    pub fn from_treetn<T>(treetn: &TreeTN<T, V>, root: &V, nsite: usize) -> Option<Self>
    where
        T: TensorLike,
    {
        Self::new(treetn.site_index_network().topology(), root, nsite)
    }

    /// Generate a sweep plan from a NodeNameNetwork.
    ///
    /// Uses Euler tour traversal to visit all edges in both directions.
    ///
    /// # Arguments
    /// * `network` - The network topology
    /// * `root` - The starting node for the sweep
    /// * `nsite` - Number of sites per update (1 or 2)
    ///
    /// # Returns
    /// A sweep plan, or `None` if the root doesn't exist or nsite is invalid.
    ///
    /// # Example
    /// For nsite=1 on chain A-B-C with root B:
    /// - Euler tour vertices: [B, A, B, C, B]
    /// - Steps: [(B, B), (A, A), (B, B), (C, C)] (each vertex except last)
    ///
    /// For nsite=2 on chain A-B-C with root B:
    /// - Euler tour edges: [(B,A), (A,B), (B,C), (C,B)]
    /// - Steps: [({B,A}, A), ({A,B}, B), ({B,C}, C), ({C,B}, B)]
    pub fn new(network: &NodeNameNetwork<V>, root: &V, nsite: usize) -> Option<Self> {
        if nsite != 1 && nsite != 2 {
            return None;
        }

        let root_idx = network.node_index(root)?;

        match nsite {
            1 => {
                // nsite=1: Use vertex sequence from Euler tour
                let vertices = network.euler_tour_vertices_by_index(root_idx);
                if vertices.is_empty() {
                    return Some(Self::empty(nsite));
                }

                // Each vertex (except the last return to root) is a step
                // The new_center is the current vertex itself
                let steps: Vec<_> = vertices
                    .iter()
                    .take(vertices.len().saturating_sub(1))
                    .filter_map(|&v| {
                        let name = network.node_name(v)?.clone();
                        Some(LocalUpdateStep {
                            nodes: vec![name.clone()],
                            new_center: name,
                        })
                    })
                    .collect();

                Some(Self { steps, nsite })
            }
            2 => {
                // nsite=2: Use edge sequence from Euler tour
                let edges = network.euler_tour_edges_by_index(root_idx);
                if edges.is_empty() {
                    // Single node: no edges to update
                    return Some(Self::empty(nsite));
                }

                // Each edge (u, v) becomes a step with nodes [u, v]
                // The new_center is v (the direction we're moving)
                let steps: Vec<_> = edges
                    .iter()
                    .filter_map(|&(u, v)| {
                        let u_name = network.node_name(u)?.clone();
                        let v_name = network.node_name(v)?.clone();
                        Some(LocalUpdateStep {
                            nodes: vec![u_name, v_name.clone()],
                            new_center: v_name,
                        })
                    })
                    .collect();

                Some(Self { steps, nsite })
            }
            _ => None,
        }
    }

    /// Create an empty sweep plan.
    pub fn empty(nsite: usize) -> Self {
        Self {
            steps: Vec::new(),
            nsite,
        }
    }

    /// Check if the plan is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Number of update steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Iterate over the steps.
    pub fn iter(&self) -> impl Iterator<Item = &LocalUpdateStep<V>> {
        self.steps.iter()
    }
}

// ============================================================================
// Boundary edge/bond utilities
// ============================================================================

/// Boundary edge information: (node_in_region, neighbor_outside, bond_index).
///
/// Represents an edge connecting a node inside the region to a neighbor outside the region.
#[derive(Debug, Clone)]
pub struct BoundaryEdge<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq,
{
    /// Node inside the region
    pub node_in_region: V,
    /// Neighbor outside the region
    pub neighbor_outside: V,
    /// Bond index connecting node_in_region to neighbor_outside
    pub bond_index: T::Index,
}

/// Get all boundary edges for a given region in a TreeTN.
///
/// Returns edges connecting nodes inside the region to neighbors outside the region.
/// This is useful for maintaining stable bond IDs across updates (e.g., for environment cache consistency).
///
/// # Arguments
/// * `treetn` - The TreeTN to analyze
/// * `region` - Nodes that are inside the region
///
/// # Returns
/// Vector of boundary edges, each containing the node in region, neighbor outside, and bond index.
pub fn get_boundary_edges<T, V>(
    treetn: &TreeTN<T, V>,
    region: &[V],
) -> Result<Vec<BoundaryEdge<T, V>>>
where
    T: TensorLike,
    T::Index: IndexLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let mut boundary_edges = Vec::new();
    let region_set: HashSet<&V> = region.iter().collect();

    for node in region {
        for neighbor in treetn.site_index_network().neighbors(node) {
            if !region_set.contains(&neighbor) {
                // This is a boundary edge: node is in region, neighbor is outside
                if let Some(edge) = treetn.edge_between(node, &neighbor) {
                    if let Some(bond) = treetn.bond_index(edge) {
                        boundary_edges.push(BoundaryEdge {
                            node_in_region: node.clone(),
                            neighbor_outside: neighbor.clone(),
                            bond_index: bond.clone(),
                        });
                    }
                }
            }
        }
    }

    Ok(boundary_edges)
}

// ============================================================================
// LocalUpdater trait
// ============================================================================

/// Trait for local update operations during a sweep.
///
/// Implementors of this trait provide the actual update logic that transforms
/// a local subtree into an updated version. This allows different algorithms
/// (truncation, fitting, DMRG, TDVP) to share the same sweep infrastructure.
///
/// # Type Parameters
/// - `T`: Tensor type implementing TensorLike
/// - `V`: Node name type
///
/// # Workflow
/// During `apply_local_update_sweep`:
/// 1. For each step in the sweep plan:
///    a. Extract the subtree containing `step.nodes`
///    b. Call `update()` with the extracted subtree and step info
///    c. Replace the subtree in the original TreeTN with the updated one
///    d. Update the canonical center to `step.new_center`
pub trait LocalUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Optional hook called before performing an update step.
    ///
    /// This is called with the full TreeTN state *before* the update is applied.
    /// Implementors can use it to validate assumptions or prefetch/update caches.
    fn before_step(
        &mut self,
        _step: &LocalUpdateStep<V>,
        _full_treetn_before: &TreeTN<T, V>,
    ) -> Result<()> {
        Ok(())
    }

    /// Update a local subtree.
    ///
    /// # Arguments
    /// * `subtree` - The extracted subtree to update
    /// * `step` - The current step information (nodes and new_center)
    /// * `full_treetn` - Reference to the full (global) TreeTN. This provides global context
    ///   (e.g., topology, neighbor relations, and index/bond metadata) that some update
    ///   algorithms may need. It may be unused by simple updaters.
    ///
    /// # Returns
    /// The updated subtree, which must have the same "appearance" as the input
    /// (same nodes, same external indices, same ortho_towards structure).
    ///
    /// # Errors
    /// Returns an error if the update fails (e.g., SVD doesn't converge).
    fn update(
        &mut self,
        subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<T, V>,
    ) -> Result<TreeTN<T, V>>;

    /// Optional hook called after an update step has been applied to the full TreeTN.
    ///
    /// This is called after:
    /// - The updated subtree has been inserted back into the full TreeTN
    /// - The canonical center has been moved to `step.new_center`
    ///
    /// Implementors can use this to update caches that must see the post-update state.
    fn after_step(
        &mut self,
        _step: &LocalUpdateStep<V>,
        _full_treetn_after: &TreeTN<T, V>,
    ) -> Result<()> {
        Ok(())
    }
}

/// Apply a local update sweep to a TreeTN.
///
/// This function orchestrates the sweep by:
/// 1. Iterating through the sweep plan
/// 2. For each step:
///    a. Validate that the canonical center is a single node within the extracted subtree
///    b. Extract the local subtree
///    c. Call the updater to transform it
///    d. Replace the subtree back into the TreeTN
///
/// # Arguments
/// * `treetn` - The TreeTN to update (modified in place)
/// * `plan` - The sweep plan specifying the update order
/// * `updater` - The local updater implementation
///
/// # Preconditions
/// - The TreeTN must be canonicalized with a single-node canonical center
/// - The canonical center must be within the first step's nodes
///
/// # Returns
/// `Ok(())` if the sweep completes successfully.
///
/// # Errors
/// Returns an error if:
/// - TreeTN is not canonicalized (canonical_center is empty)
/// - canonical_center is not a single node
/// - canonical_center is not within the extracted subtree
/// - Subtree extraction fails
/// - The updater returns an error
/// - Subtree replacement fails
pub fn apply_local_update_sweep<T, V, U>(
    treetn: &mut TreeTN<T, V>,
    plan: &LocalUpdateSweepPlan<V>,
    updater: &mut U,
) -> Result<()>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    U: LocalUpdater<T, V>,
{
    for step in plan.iter() {
        // Validate: canonical_center must be a single node within the step's nodes
        let canonical_center = treetn.canonical_center();
        if canonical_center.is_empty() {
            return Err(anyhow::anyhow!(
                "TreeTN is not canonicalized: canonical_center is empty"
            ))
            .context("apply_local_update_sweep: TreeTN must be canonicalized before sweep");
        }
        if canonical_center.len() != 1 {
            return Err(anyhow::anyhow!(
                "canonical_center must be a single node, got {} nodes",
                canonical_center.len()
            ))
            .context("apply_local_update_sweep: canonical_center must be a single node");
        }
        let center_node = canonical_center.iter().next().unwrap();
        let step_nodes_set: HashSet<V> = step.nodes.iter().cloned().collect();
        if !step_nodes_set.contains(center_node) {
            return Err(anyhow::anyhow!(
                "canonical_center {:?} is not within the extracted subtree {:?}",
                center_node,
                step.nodes
            ))
            .context(
                "apply_local_update_sweep: canonical_center must be within extracted subtree",
            );
        }

        updater
            .before_step(step, treetn)
            .context("apply_local_update_sweep: LocalUpdater::before_step failed")?;

        // Extract subtree for the nodes in this step
        let subtree = treetn.extract_subtree(&step.nodes)?;

        // Apply the update
        let updated_subtree = updater.update(subtree, step, treetn)?;

        // Replace the subtree back
        treetn.replace_subtree(&step.nodes, &updated_subtree)?;

        // Update canonical center
        treetn.set_canonical_center([step.new_center.clone()])?;

        updater
            .after_step(step, treetn)
            .context("apply_local_update_sweep: LocalUpdater::after_step failed")?;
    }

    Ok(())
}

// ============================================================================
// TruncateUpdater - LocalUpdater implementation for truncation
// ============================================================================

use tensor4all_core::{Canonical, FactorizeOptions};

/// Truncation updater for nsite=2 sweeps.
///
/// This updater performs SVD-based truncation on two-site subtrees,
/// compressing bond dimensions while preserving the tensor train structure.
///
/// # Algorithm
/// For each step with nodes [A, B] where B is the new center:
/// 1. Contract tensors A and B into a single tensor AB
/// 2. Factorize AB using SVD with truncation (left indices = A's external + bond to A's other neighbors)
/// 3. The left tensor becomes the new A, the right tensor becomes the new B
/// 4. B is the orthogonality center (isometry pointing towards B)
///
/// # Usage
/// ```ignore
/// let mut updater = TruncateUpdater::new(max_rank, rtol);
/// apply_local_update_sweep(&mut treetn, &plan, &mut updater)?;
/// ```
#[derive(Debug, Clone)]
pub struct TruncateUpdater {
    /// Maximum bond dimension after truncation.
    pub max_rank: Option<usize>,
    /// Relative tolerance for truncation (singular values below rtol * max_sv are dropped).
    pub rtol: Option<f64>,
}

impl TruncateUpdater {
    /// Create a new truncation updater.
    ///
    /// # Arguments
    /// * `max_rank` - Maximum bond dimension (None for no limit)
    /// * `rtol` - Relative tolerance for truncation (None for no tolerance-based truncation)
    pub fn new(max_rank: Option<usize>, rtol: Option<f64>) -> Self {
        Self { max_rank, rtol }
    }
}

impl<T, V> LocalUpdater<T, V> for TruncateUpdater
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    fn update(
        &mut self,
        mut subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        _full_treetn: &TreeTN<T, V>,
    ) -> Result<TreeTN<T, V>> {
        // TruncateUpdater is designed for nsite=2
        if step.nodes.len() != 2 {
            return Err(anyhow::anyhow!(
                "TruncateUpdater requires exactly 2 nodes, got {}",
                step.nodes.len()
            ));
        }

        let node_a = &step.nodes[0];
        let node_b = &step.nodes[1];

        // Get node indices
        let idx_a = subtree
            .node_index(node_a)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", node_a))?;
        let idx_b = subtree
            .node_index(node_b)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", node_b))?;

        // Get the bond between A and B
        let edge_ab = subtree
            .edge_between(node_a, node_b)
            .ok_or_else(|| anyhow::anyhow!("No edge between {:?} and {:?}", node_a, node_b))?;
        let bond_ab = subtree
            .bond_index(edge_ab)
            .ok_or_else(|| anyhow::anyhow!("Bond index not found"))?
            .clone();

        // Contract A and B
        let tensor_a = subtree.tensor(idx_a).unwrap();
        let tensor_b = subtree.tensor(idx_b).unwrap();
        let tensor_ab = T::contract(&[tensor_a, tensor_b], AllowedPairs::All)
            .context("Failed to contract A and B")?;

        // Determine left indices (indices that will remain on A after factorization)
        // These are: all indices of A except the bond to B
        let left_inds: Vec<_> = tensor_a
            .external_indices()
            .iter()
            .filter(|idx| idx.id() != bond_ab.id())
            .cloned()
            .collect();

        // Set up factorization options
        let mut options = FactorizeOptions::svd().with_canonical(Canonical::Left); // Left canonical: A is isometry, B has the norm

        if let Some(max_rank) = self.max_rank {
            options = options.with_max_rank(max_rank);
        }
        if let Some(rtol) = self.rtol {
            options = options.with_rtol(rtol);
        }

        // Factorize
        let factorize_result = tensor_ab
            .factorize(&left_inds, &options)
            .map_err(|e| anyhow::anyhow!("Factorization failed: {}", e))?;

        let new_tensor_a = factorize_result.left;
        let new_tensor_b = factorize_result.right;
        let new_bond = factorize_result.bond_index;

        // Update the subtree - first update the edge bond, then the tensors
        // The factorize result creates a new bond index, so we update the edge to use it
        subtree.replace_edge_bond(edge_ab, new_bond.clone())?;
        subtree.replace_tensor(idx_a, new_tensor_a)?;
        subtree.replace_tensor(idx_b, new_tensor_b)?;

        // Set ortho_towards: bond points towards new_center (B)
        subtree.set_ortho_towards(&new_bond, Some(step.new_center.clone()));

        // Set canonical center to the new center
        subtree.set_canonical_center([step.new_center.clone()])?;

        Ok(subtree)
    }
}

// ============================================================================
// Sub-tree extraction
// ============================================================================

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Extract a sub-tree from this TreeTN.
    ///
    /// Creates a new TreeTN containing only the specified nodes and their
    /// connecting edges. Tensors are cloned into the new TreeTN.
    ///
    /// # Arguments
    /// * `node_names` - The names of nodes to include in the sub-tree
    ///
    /// # Returns
    /// A new TreeTN containing the specified sub-tree, or an error if:
    /// - Any specified node doesn't exist
    /// - The specified nodes don't form a connected subtree
    ///
    /// # Notes
    /// - Bond indices between included nodes are preserved
    /// - Bond indices to excluded nodes become external (site) indices in the sub-tree
    /// - ortho_towards directions are copied for edges within the sub-tree
    /// - canonical_center is intersected with the extracted nodes
    pub fn extract_subtree(&self, node_names: &[V]) -> Result<Self>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        V: Ord,
    {
        if node_names.is_empty() {
            return Err(anyhow::anyhow!("Cannot extract empty subtree"));
        }

        // Validate all nodes exist
        for name in node_names {
            if self.graph.node_index(name).is_none() {
                return Err(anyhow::anyhow!("Node {:?} does not exist", name))
                    .context("extract_subtree: invalid node name");
            }
        }

        // Check connectivity: the specified nodes must form a connected subtree
        let node_indices: HashSet<_> = node_names
            .iter()
            .filter_map(|n| self.graph.node_index(n))
            .collect();

        if !self.site_index_network.is_connected_subset(&node_indices) {
            return Err(anyhow::anyhow!(
                "Specified nodes do not form a connected subtree"
            ))
            .context("extract_subtree: nodes must be connected");
        }

        let node_name_set: HashSet<V> = node_names.iter().cloned().collect();

        // Create new TreeTN with extracted tensors
        let mut subtree = TreeTN::<T, V>::new();

        // Step 1: Add all nodes with their tensors
        for name in node_names {
            let node_idx = self.graph.node_index(name).unwrap();
            let tensor = self
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", name))?
                .clone();

            subtree
                .add_tensor(name.clone(), tensor)
                .context("extract_subtree: failed to add tensor")?;
        }

        // Step 2: Add edges between nodes in the subtree
        // Track which edges we've already added to avoid duplicates
        let mut added_edges: HashSet<(V, V)> = HashSet::new();

        for name in node_names {
            let neighbors: Vec<V> = self.site_index_network.neighbors(name).collect();

            for neighbor in neighbors {
                // Only add edge if neighbor is also in the subtree
                if !node_name_set.contains(&neighbor) {
                    continue;
                }

                // Avoid adding the same edge twice (undirected)
                let edge_key = if *name < neighbor {
                    (name.clone(), neighbor.clone())
                } else {
                    (neighbor.clone(), name.clone())
                };

                if added_edges.contains(&edge_key) {
                    continue;
                }
                added_edges.insert(edge_key);

                // Get bond index from original TreeTN
                let orig_edge = self.edge_between(name, &neighbor).ok_or_else(|| {
                    anyhow::anyhow!("Edge not found between {:?} and {:?}", name, neighbor)
                })?;

                let bond_index = self
                    .bond_index(orig_edge)
                    .ok_or_else(|| anyhow::anyhow!("Bond index not found"))?
                    .clone();

                // Get node indices in new subtree
                let subtree_node_a = subtree
                    .graph
                    .node_index(name)
                    .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", name))?;
                let subtree_node_b = subtree
                    .graph
                    .node_index(&neighbor)
                    .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", neighbor))?;

                // Connect in subtree
                subtree
                    .connect(subtree_node_a, &bond_index, subtree_node_b, &bond_index)
                    .context("extract_subtree: failed to connect nodes")?;

                // Copy ortho_towards if it exists (keyed by full bond index)
                if let Some(ortho_dir) = self.ortho_towards.get(&bond_index) {
                    // Only copy if the direction node is in the subtree
                    if node_name_set.contains(ortho_dir) {
                        subtree
                            .ortho_towards
                            .insert(bond_index.clone(), ortho_dir.clone());
                    }
                }
            }
        }

        // Step 3: Set canonical_center to intersection with extracted nodes
        let new_center: HashSet<V> = self
            .canonical_center
            .intersection(&node_name_set)
            .cloned()
            .collect();
        subtree.canonical_center = new_center;

        // Copy canonical_form if any center nodes were included
        if !subtree.canonical_center.is_empty() {
            subtree.canonical_form = self.canonical_form;
        }

        Ok(subtree)
    }
}

// ============================================================================
// Sub-tree replacement
// ============================================================================

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Replace a sub-tree with another TreeTN of the same topology.
    ///
    /// This method replaces the tensors and ortho_towards directions for a subset
    /// of nodes with those from another TreeTN. The replacement TreeTN must have
    /// the same topology (nodes and edges) as the sub-tree being replaced.
    ///
    /// # Arguments
    /// * `node_names` - The names of nodes to replace
    /// * `replacement` - The TreeTN to use as replacement
    ///
    /// # Returns
    /// `Ok(())` if the replacement succeeds, or an error if:
    /// - Any specified node doesn't exist
    /// - The replacement doesn't have the same topology as the extracted sub-tree
    /// - Tensor replacement fails
    ///
    /// # Notes
    /// - The replacement TreeTN must have the same nodes, edges, and site indices
    /// - Bond dimensions may differ (this is the typical use case for truncation)
    /// - ortho_towards may differ (will be copied from replacement)
    /// - The original TreeTN is modified in-place
    pub fn replace_subtree(&mut self, node_names: &[V], replacement: &Self) -> Result<()>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        if node_names.is_empty() {
            return Ok(()); // Nothing to replace
        }

        // Extract current subtree for comparison
        let current_subtree = self.extract_subtree(node_names)?;

        // Verify that replacement has the same topology (nodes and edges)
        // Note: site index network may differ due to bond dimension changes in truncation
        if !current_subtree.same_topology(replacement) {
            return Err(anyhow::anyhow!(
                "Replacement TreeTN does not have the same topology as the current subtree"
            ))
            .context("replace_subtree: topology mismatch");
        }

        let node_name_set: HashSet<V> = node_names.iter().cloned().collect();
        let mut processed_edges: HashSet<(V, V)> = HashSet::new();

        // Step 1: Update edge bond indices FIRST (before replacing tensors)
        // This is crucial because replace_tensor validates that tensors contain connection indices
        for name in node_names {
            let neighbors: Vec<V> = self.site_index_network.neighbors(name).collect();

            for neighbor in neighbors {
                // Only process edges within the subtree
                if !node_name_set.contains(&neighbor) {
                    continue;
                }

                let edge_key = if *name < neighbor {
                    (name.clone(), neighbor.clone())
                } else {
                    (neighbor.clone(), name.clone())
                };

                if processed_edges.contains(&edge_key) {
                    continue;
                }
                processed_edges.insert(edge_key.clone());

                // Get edges in both self and replacement
                let self_edge = self
                    .edge_between(name, &neighbor)
                    .ok_or_else(|| anyhow::anyhow!("Edge not found in self"))?;
                let replacement_edge = replacement
                    .edge_between(name, &neighbor)
                    .ok_or_else(|| anyhow::anyhow!("Edge not found in replacement"))?;

                // Get new bond index from replacement
                let new_bond = replacement
                    .bond_index(replacement_edge)
                    .ok_or_else(|| anyhow::anyhow!("Bond index not found in replacement"))?
                    .clone();

                // Update bond index in self
                self.replace_edge_bond(self_edge, new_bond.clone())
                    .with_context(|| {
                        format!(
                            "replace_subtree: failed to update bond between {:?} and {:?}",
                            name, neighbor
                        )
                    })?;

                // Copy ortho_towards from replacement (using the new bond)
                match replacement.ortho_towards.get(&new_bond) {
                    Some(dir) => {
                        self.ortho_towards.insert(new_bond, dir.clone());
                    }
                    None => {
                        self.ortho_towards.remove(&new_bond);
                    }
                }
            }
        }

        // Step 2: Replace tensors (now bond indices match)
        for name in node_names {
            let self_node_idx = self
                .graph
                .node_index(name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", name))?;
            let replacement_node_idx = replacement
                .graph
                .node_index(name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in replacement", name))?;

            let new_tensor = replacement
                .tensor(replacement_node_idx)
                .ok_or_else(|| {
                    anyhow::anyhow!("Tensor not found for node {:?} in replacement", name)
                })?
                .clone();

            self.replace_tensor(self_node_idx, new_tensor)
                .with_context(|| {
                    format!(
                        "replace_subtree: failed to replace tensor at node {:?}",
                        name
                    )
                })?;
        }

        // Update canonical_center: remove old nodes, add from replacement
        for name in node_names {
            self.canonical_center.remove(name);
        }
        for name in &replacement.canonical_center {
            if node_name_set.contains(name) {
                self.canonical_center.insert(name.clone());
            }
        }

        // Update canonical_form if replacement has one
        if replacement.canonical_form.is_some() {
            self.canonical_form = replacement.canonical_form;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::{DynIndex, TensorDynLen, TensorIndex};

    /// Create a 4-node Y-shape TreeTN:
    ///     A
    ///     |
    ///     B
    ///    / \
    ///   C   D
    fn create_y_shape_treetn() -> (
        TreeTN<TensorDynLen, String>,
        DynIndex,
        DynIndex,
        DynIndex,
        DynIndex,
    ) {
        let mut tn = TreeTN::<TensorDynLen, String>::new();

        let site_a = DynIndex::new_dyn(2);
        let site_c = DynIndex::new_dyn(2);
        let site_d = DynIndex::new_dyn(2);
        let bond_ab = DynIndex::new_dyn(3);
        let bond_bc = DynIndex::new_dyn(3);
        let bond_bd = DynIndex::new_dyn(3);

        // Tensor A: [site_a, bond_ab]
        let tensor_a =
            TensorDynLen::from_dense_f64(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6]);
        tn.add_tensor("A".to_string(), tensor_a).unwrap();

        // Tensor B: [bond_ab, bond_bc, bond_bd]
        let tensor_b = TensorDynLen::from_dense_f64(
            vec![bond_ab.clone(), bond_bc.clone(), bond_bd.clone()],
            vec![1.0; 27],
        );
        tn.add_tensor("B".to_string(), tensor_b).unwrap();

        // Tensor C: [bond_bc, site_c]
        let tensor_c =
            TensorDynLen::from_dense_f64(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6]);
        tn.add_tensor("C".to_string(), tensor_c).unwrap();

        // Tensor D: [bond_bd, site_d]
        let tensor_d =
            TensorDynLen::from_dense_f64(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6]);
        tn.add_tensor("D".to_string(), tensor_d).unwrap();

        // Connect
        let n_a = tn.node_index(&"A".to_string()).unwrap();
        let n_b = tn.node_index(&"B".to_string()).unwrap();
        let n_c = tn.node_index(&"C".to_string()).unwrap();
        let n_d = tn.node_index(&"D".to_string()).unwrap();

        tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
        tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();
        tn.connect(n_b, &bond_bd, n_d, &bond_bd).unwrap();

        (tn, site_a, site_c, site_d, bond_ab)
    }

    #[test]
    fn test_extract_subtree_single_node() {
        let (tn, _site_a, _, _, _) = create_y_shape_treetn();

        // Extract just node A
        let subtree = tn.extract_subtree(&["A".to_string()]).unwrap();

        assert_eq!(subtree.node_count(), 1);
        assert_eq!(subtree.edge_count(), 0);

        // Should have site_a as external index plus bond_ab (which becomes external)
        let n_a = subtree.node_index(&"A".to_string()).unwrap();
        let tensor_a = subtree.tensor(n_a).unwrap();
        assert_eq!(tensor_a.num_external_indices(), 2);

        // Verify consistency after extraction
        subtree.verify_internal_consistency().unwrap();
    }

    #[test]
    fn test_extract_subtree_two_nodes() {
        let (tn, _, _, _, _) = create_y_shape_treetn();

        // Extract A-B subtree
        let subtree = tn
            .extract_subtree(&["A".to_string(), "B".to_string()])
            .unwrap();

        assert_eq!(subtree.node_count(), 2);
        assert_eq!(subtree.edge_count(), 1);

        // Verify connectivity
        let _n_a = subtree.node_index(&"A".to_string()).unwrap();
        let _n_b = subtree.node_index(&"B".to_string()).unwrap();
        assert!(subtree
            .edge_between(&"A".to_string(), &"B".to_string())
            .is_some());

        // Verify consistency after extraction
        subtree.verify_internal_consistency().unwrap();
    }

    #[test]
    fn test_extract_subtree_disconnected_fails() {
        let (tn, _, _, _, _) = create_y_shape_treetn();

        // Try to extract A and C (not connected)
        let result = tn.extract_subtree(&["A".to_string(), "C".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_subtree_preserves_consistency() {
        let (tn, _, _, _, _) = create_y_shape_treetn();

        // Extract B-C-D subtree
        let subtree = tn
            .extract_subtree(&["B".to_string(), "C".to_string(), "D".to_string()])
            .unwrap();

        // Verify consistency
        subtree.verify_internal_consistency().unwrap();
    }

    #[test]
    fn test_replace_subtree_same_appearance() {
        let (mut tn, _, _, _, _) = create_y_shape_treetn();

        // Extract subtree, modify tensor data (but keep same structure), replace
        let subtree = tn.extract_subtree(&["C".to_string()]).unwrap();

        // Replace with itself (should work)
        tn.replace_subtree(&["C".to_string()], &subtree).unwrap();

        // Verify consistency
        tn.verify_internal_consistency().unwrap();
    }

    #[test]
    fn test_replace_subtree_two_nodes() {
        let (mut tn, _, _, _, _) = create_y_shape_treetn();

        // Extract C-D subtree (through B)... wait, C and D are not connected.
        // Let's use B-C subtree instead.
        let subtree = tn
            .extract_subtree(&["B".to_string(), "C".to_string()])
            .unwrap();

        // Replace with itself
        tn.replace_subtree(&["B".to_string(), "C".to_string()], &subtree)
            .unwrap();

        // Verify consistency
        tn.verify_internal_consistency().unwrap();
    }

    // ========================================================================
    // LocalUpdateSweepPlan tests
    // ========================================================================

    /// Create a chain network: A - B - C
    fn create_chain_network() -> NodeNameNetwork<String> {
        let mut net = NodeNameNetwork::new();
        net.add_node("A".to_string()).unwrap();
        net.add_node("B".to_string()).unwrap();
        net.add_node("C".to_string()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net
    }

    /// Create a Y-shape network:
    ///     A
    ///     |
    ///     B
    ///    / \
    ///   C   D
    fn create_y_network() -> NodeNameNetwork<String> {
        let mut net = NodeNameNetwork::new();
        net.add_node("A".to_string()).unwrap();
        net.add_node("B".to_string()).unwrap();
        net.add_node("C".to_string()).unwrap();
        net.add_node("D".to_string()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"D".to_string()).unwrap();
        net
    }

    #[test]
    fn test_sweep_plan_nsite1_chain() {
        let net = create_chain_network();

        // Sweep from B (middle)
        let plan = LocalUpdateSweepPlan::new(&net, &"B".to_string(), 1).unwrap();

        // nsite=1: Euler tour vertices from B: [B, A, B, C, B]
        // Steps: all vertices except the last (return to root)
        // So we have 4 steps: B, A, B, C
        assert_eq!(plan.nsite, 1);
        assert_eq!(plan.len(), 4);

        // Each step should have exactly one node
        for step in plan.iter() {
            assert_eq!(step.nodes.len(), 1);
            // For nsite=1, new_center == nodes[0]
            assert_eq!(&step.new_center, &step.nodes[0]);
        }

        // First step should be B (starting point)
        assert_eq!(&plan.steps[0].nodes[0], "B");
    }

    #[test]
    fn test_sweep_plan_nsite2_chain() {
        let net = create_chain_network();

        // Sweep from B (middle)
        let plan = LocalUpdateSweepPlan::new(&net, &"B".to_string(), 2).unwrap();

        // nsite=2: Euler tour edges from B visits 2 edges × 2 directions = 4 edges
        // The exact order depends on neighbor ordering in petgraph
        assert_eq!(plan.nsite, 2);
        assert_eq!(plan.len(), 4);

        // Each step should have exactly two nodes
        for step in plan.iter() {
            assert_eq!(step.nodes.len(), 2);
        }

        // Each edge (A-B and B-C) should be visited twice (in both directions)
        let mut ab_count = 0;
        let mut bc_count = 0;
        for step in plan.iter() {
            let has_a = step.nodes.contains(&"A".to_string());
            let has_b = step.nodes.contains(&"B".to_string());
            let has_c = step.nodes.contains(&"C".to_string());

            if has_a && has_b {
                ab_count += 1;
            }
            if has_b && has_c {
                bc_count += 1;
            }
        }
        assert_eq!(ab_count, 2); // Edge A-B visited twice
        assert_eq!(bc_count, 2); // Edge B-C visited twice

        // First step should start from B
        assert!(plan.steps[0].nodes.contains(&"B".to_string()));
    }

    #[test]
    fn test_sweep_plan_nsite1_y_shape() {
        let net = create_y_network();

        // Sweep from B (center of Y)
        let plan = LocalUpdateSweepPlan::new(&net, &"B".to_string(), 1).unwrap();

        assert_eq!(plan.nsite, 1);
        // Y-shape has 3 edges, so Euler tour visits each twice = 6 edges
        // Vertices sequence: 7 vertices (starting node + 6 edge traversals)
        // Steps: 6 (all except last return to B)
        assert_eq!(plan.len(), 6);

        // All 4 nodes should be visited
        let visited: HashSet<_> = plan.iter().map(|s| s.nodes[0].clone()).collect();
        assert!(visited.contains("A"));
        assert!(visited.contains("B"));
        assert!(visited.contains("C"));
        assert!(visited.contains("D"));
    }

    #[test]
    fn test_sweep_plan_nsite2_y_shape() {
        let net = create_y_network();

        // Sweep from B (center of Y)
        let plan = LocalUpdateSweepPlan::new(&net, &"B".to_string(), 2).unwrap();

        assert_eq!(plan.nsite, 2);
        // Y-shape has 3 edges, each visited twice = 6 edge traversals
        assert_eq!(plan.len(), 6);

        // Each edge should appear in both directions
        let mut edge_pairs: HashSet<(String, String)> = HashSet::new();
        for step in plan.iter() {
            let mut nodes = step.nodes.clone();
            nodes.sort();
            edge_pairs.insert((nodes[0].clone(), nodes[1].clone()));
        }
        // 3 unique edges: {A,B}, {B,C}, {B,D}
        assert_eq!(edge_pairs.len(), 3);
        assert!(edge_pairs.contains(&("A".to_string(), "B".to_string())));
        assert!(edge_pairs.contains(&("B".to_string(), "C".to_string())));
        assert!(edge_pairs.contains(&("B".to_string(), "D".to_string())));
    }

    #[test]
    fn test_sweep_plan_single_node() {
        let mut net = NodeNameNetwork::new();
        net.add_node("A".to_string()).unwrap();

        // Single node: no edges, empty plan
        let plan_nsite1 = LocalUpdateSweepPlan::new(&net, &"A".to_string(), 1).unwrap();
        // Single node has no edges, so Euler tour returns just [A]
        // Steps: all except last = 0 steps
        assert!(plan_nsite1.is_empty());

        let plan_nsite2 = LocalUpdateSweepPlan::new(&net, &"A".to_string(), 2).unwrap();
        assert!(plan_nsite2.is_empty());
    }

    #[test]
    fn test_sweep_plan_invalid_nsite() {
        let net = create_chain_network();

        // nsite=0 is invalid
        assert!(LocalUpdateSweepPlan::new(&net, &"B".to_string(), 0).is_none());

        // nsite=3 is invalid
        assert!(LocalUpdateSweepPlan::new(&net, &"B".to_string(), 3).is_none());
    }

    #[test]
    fn test_sweep_plan_nonexistent_root() {
        let net = create_chain_network();

        // Root "X" doesn't exist
        assert!(LocalUpdateSweepPlan::new(&net, &"X".to_string(), 1).is_none());
    }

    // ========================================================================
    // TruncateUpdater and apply_local_update_sweep tests
    // ========================================================================

    /// Create a chain TreeTN: A - B - C
    /// Each node has a site index of dim 2, bonds of dim 4
    fn create_chain_treetn() -> TreeTN<TensorDynLen, String> {
        let mut tn = TreeTN::<TensorDynLen, String>::new();

        let site_a = DynIndex::new_dyn(2);
        let site_b = DynIndex::new_dyn(2);
        let site_c = DynIndex::new_dyn(2);
        let bond_ab = DynIndex::new_dyn(4);
        let bond_bc = DynIndex::new_dyn(4);

        // Tensor A: [site_a, bond_ab] dim 2x4
        let tensor_a =
            TensorDynLen::from_dense_f64(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 8]);
        tn.add_tensor("A".to_string(), tensor_a).unwrap();

        // Tensor B: [bond_ab, site_b, bond_bc] dim 4x2x4
        let tensor_b = TensorDynLen::from_dense_f64(
            vec![bond_ab.clone(), site_b.clone(), bond_bc.clone()],
            vec![1.0; 32],
        );
        tn.add_tensor("B".to_string(), tensor_b).unwrap();

        // Tensor C: [bond_bc, site_c] dim 4x2
        let tensor_c =
            TensorDynLen::from_dense_f64(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 8]);
        tn.add_tensor("C".to_string(), tensor_c).unwrap();

        // Connect
        let n_a = tn.node_index(&"A".to_string()).unwrap();
        let n_b = tn.node_index(&"B".to_string()).unwrap();
        let n_c = tn.node_index(&"C".to_string()).unwrap();

        tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
        tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();

        tn
    }

    #[test]
    fn test_truncate_updater_basic() {
        use crate::CanonicalizationOptions;

        let tn = create_chain_treetn();

        // Canonicalize towards B (the root of the sweep)
        // This is required before using TruncateUpdater
        let mut tn = tn
            .canonicalize(["B".to_string()], CanonicalizationOptions::default())
            .expect("Failed to canonicalize");

        // Create sweep plan with nsite=2 from B
        let plan = LocalUpdateSweepPlan::from_treetn(&tn, &"B".to_string(), 2).unwrap();
        assert_eq!(plan.len(), 4); // 2 edges × 2 directions

        // Create truncate updater with max_rank=2
        let mut updater = TruncateUpdater::new(Some(2), None);

        // Apply sweep
        apply_local_update_sweep(&mut tn, &plan, &mut updater).unwrap();

        // Verify consistency after sweep
        tn.verify_internal_consistency().unwrap();

        // Check that bond dimensions are reduced
        // After truncation with max_rank=2, all bonds should have dim <= 2
        for node_name in tn.node_names() {
            let node_idx = tn.node_index(&node_name).unwrap();
            let tensor = tn.tensor(node_idx).unwrap();
            for dim in tensor.external_indices().iter().map(|i| i.dim()) {
                // Site dims are 2, truncated bonds should be <= 2
                assert!(dim <= 4); // Original max was 4
            }
        }
    }

    #[test]
    fn test_apply_local_update_sweep_preserves_structure() {
        use crate::CanonicalizationOptions;

        let tn = create_chain_treetn();
        let original_node_count = tn.node_count();
        let original_edge_count = tn.edge_count();

        // Canonicalize towards B (the root of the sweep)
        // This is required before using TruncateUpdater
        let mut tn = tn
            .canonicalize(["B".to_string()], CanonicalizationOptions::default())
            .expect("Failed to canonicalize");

        // Create sweep plan with nsite=2 from B
        let plan = LocalUpdateSweepPlan::from_treetn(&tn, &"B".to_string(), 2).unwrap();

        // Apply with no truncation (max_rank=None)
        let mut updater = TruncateUpdater::new(None, None);
        apply_local_update_sweep(&mut tn, &plan, &mut updater).unwrap();

        // Structure should be preserved
        assert_eq!(tn.node_count(), original_node_count);
        assert_eq!(tn.edge_count(), original_edge_count);

        // Verify consistency
        tn.verify_internal_consistency().unwrap();
    }

    #[test]
    fn test_apply_local_update_sweep_requires_canonicalization() {
        // Test that apply_local_update_sweep fails when TreeTN is not canonicalized
        let mut tn = create_chain_treetn();

        // Create sweep plan with nsite=2 from B
        let plan = LocalUpdateSweepPlan::from_treetn(&tn, &"B".to_string(), 2).unwrap();

        // Create truncate updater
        let mut updater = TruncateUpdater::new(Some(2), None);

        // Apply sweep should fail because TreeTN is not canonicalized
        let result = apply_local_update_sweep(&mut tn, &plan, &mut updater);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("not canonicalized") || err_msg.contains("canonical_center is empty"),
            "Unexpected error message: {}",
            err_msg
        );
    }

    #[test]
    fn test_sweep_plan_from_treetn() {
        let tn = create_chain_treetn();

        // from_treetn should work the same as new with topology
        let plan1 = LocalUpdateSweepPlan::from_treetn(&tn, &"B".to_string(), 2).unwrap();
        let plan2 =
            LocalUpdateSweepPlan::new(tn.site_index_network().topology(), &"B".to_string(), 2)
                .unwrap();

        assert_eq!(plan1.len(), plan2.len());
        assert_eq!(plan1.nsite, plan2.nsite);
    }
}
