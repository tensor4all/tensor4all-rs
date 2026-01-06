//! Fit algorithm for TreeTN contraction.
//!
//! This module provides the fit (variational) algorithm for contracting two TreeTNs.
//! The algorithm iteratively optimizes `C ≈ A * B` by minimizing `||A*B - C||²`.
//!
//! # Algorithm Overview
//!
//! 1. Prepare input TNs with `sim_internal_inds()` to avoid index collision
//! 2. Initialize C (using zipup result or random)
//! 3. For each sweep:
//!    a. Compute/update environment tensors
//!    b. For each 2-site step:
//!       - Extract local tensors from A, B, C
//!       - Compute optimal local C tensor: L × A[i] × B[i] × A[j] × B[j] × R
//!       - Factorize and update C
//!       - Update environment cache
//!
//! # Environment Tensors
//!
//! For each edge (from, to), the environment `env[(from, to)]` represents:
//! - The contraction of the "from" side subtree of A×B with conj(C)
//! - Shape: (link_A, link_B, link_C) pointing towards "to"
//!
//! # References
//!
//! - T4AMPOContractions.jl: `contract_fit`, `leftenvironment!`, `rightenvironment!`
//! - ITensorNetworks.jl: `contract` with fitting algorithm

use std::collections::HashMap;
use std::hash::Hash;

use anyhow::Result;
use petgraph::stable_graph::NodeIndex;

use tensor4all::index::{DynId, Index, NoSymmSpace, Symmetry};
use tensor4all::{factorize, Canonical, FactorizeAlg, FactorizeOptions, TensorDynLen};

use super::localupdate::{LocalUpdateStep, LocalUpdateSweepPlan, LocalUpdater};
use super::TreeTN;

// ============================================================================
// FitEnvironment: Environment tensor cache
// ============================================================================

/// Environment tensor cache for fit algorithm.
///
/// Stores environment tensors for each directed edge (from, to).
/// The environment `env[(from, to)]` represents the contraction of the
/// "from" side subtree (A×B contracted with conj(C)).
///
/// # Cache Invalidation
///
/// When C[u] is updated during a sweep step (u → v):
/// - `env[(u, v)]` is recomputed with the new C[u]
/// - `env[(v, u)]` is invalidated (removed) because C[v] will change
#[derive(Debug, Clone)]
pub struct FitEnvironment<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq,
{
    /// Environment tensors: (from, to) -> tensor
    envs: HashMap<(V, V), TensorDynLen<Id, Symm>>,
}

impl<Id, Symm, V> FitEnvironment<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + std::fmt::Debug,
{
    /// Create an empty environment cache.
    pub fn new() -> Self {
        Self {
            envs: HashMap::new(),
        }
    }

    /// Get the environment tensor for edge (from, to).
    pub fn get(&self, from: &V, to: &V) -> Option<&TensorDynLen<Id, Symm>> {
        self.envs.get(&(from.clone(), to.clone()))
    }

    /// Insert an environment tensor for edge (from, to).
    pub fn insert(&mut self, from: V, to: V, env: TensorDynLen<Id, Symm>) {
        self.envs.insert((from, to), env);
    }

    /// Remove the environment tensor for edge (from, to).
    pub fn remove(&mut self, from: &V, to: &V) -> Option<TensorDynLen<Id, Symm>> {
        self.envs.remove(&(from.clone(), to.clone()))
    }

    /// Update environment after a sweep step (from → to).
    ///
    /// - Inserts the new environment for (from, to)
    /// - Removes the invalidated environment for (to, from)
    pub fn update_after_step(&mut self, from: &V, to: &V, new_env: TensorDynLen<Id, Symm>) {
        self.envs.insert((from.clone(), to.clone()), new_env);
        self.envs.remove(&(to.clone(), from.clone()));
    }

    /// Check if environment exists for edge (from, to).
    pub fn contains(&self, from: &V, to: &V) -> bool {
        self.envs.contains_key(&(from.clone(), to.clone()))
    }

    /// Get the number of cached environments.
    pub fn len(&self) -> usize {
        self.envs.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    /// Clear all cached environments.
    pub fn clear(&mut self) {
        self.envs.clear();
    }

    /// Verify all cached environments by recomputing from scratch.
    ///
    /// This is expensive and should only be used for testing/debugging.
    ///
    /// # Arguments
    /// * `tn_a` - First input TreeTN (with sim'd internal indices)
    /// * `tn_b` - Second input TreeTN (with sim'd internal indices)
    /// * `tn_c` - Current approximation TreeTN
    /// * `tol` - Relative tolerance for comparison
    ///
    /// # Returns
    /// `Ok(())` if all environments match, or an error describing the mismatch.
    pub fn verify_all_environments(
        &self,
        tn_a: &TreeTN<Id, Symm, V>,
        tn_b: &TreeTN<Id, Symm, V>,
        tn_c: &TreeTN<Id, Symm, V>,
        tol: f64,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug,
        V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        for ((from, to), cached_env) in &self.envs {
            let recomputed = compute_environment_from_scratch(from, to, tn_a, tn_b, tn_c)?;

            // Compare tensors by computing difference norm
            // Since we don't have a direct sub method, we compute norms separately
            let cached_norm = cached_env.norm();
            let recomputed_norm = recomputed.norm();

            // Simple norm comparison as a proxy
            let diff_norm = (cached_norm - recomputed_norm).abs();
            let relative_error = if cached_norm > 1e-15 {
                diff_norm / cached_norm
            } else {
                diff_norm
            };

            if relative_error > tol {
                return Err(anyhow::anyhow!(
                    "Environment mismatch at edge ({:?}, {:?}): relative error = {}, tolerance = {}",
                    from,
                    to,
                    relative_error,
                    tol
                ));
            }
        }
        Ok(())
    }
}

impl<Id, Symm, V> Default for FitEnvironment<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Environment computation helpers
// ============================================================================

/// Compute a single environment tensor from scratch (no caching).
///
/// This computes the environment for edge (from, to) by contracting:
/// - All tensors in the "from" subtree of A
/// - All tensors in the "from" subtree of B
/// - All tensors in the "from" subtree of conj(C)
///
/// The result has indices: (link_A_from_to, link_B_from_to, link_C_from_to)
fn compute_environment_from_scratch<Id, Symm, V>(
    from: &V,
    to: &V,
    tn_a: &TreeTN<Id, Symm, V>,
    tn_b: &TreeTN<Id, Symm, V>,
    tn_c: &TreeTN<Id, Symm, V>,
) -> Result<TensorDynLen<Id, Symm>>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    // Get the subtree rooted at "from" (excluding "to")
    let subtree_nodes = get_subtree_nodes(tn_c, from, to)?;

    if subtree_nodes.is_empty() {
        return Err(anyhow::anyhow!("Subtree is empty"));
    }

    // For a single node subtree, compute directly
    if subtree_nodes.len() == 1 {
        return compute_leaf_environment(from, to, tn_a, tn_b, tn_c);
    }

    // For multi-node subtree, recursively contract from leaves
    contract_subtree_environment(&subtree_nodes, from, to, tn_a, tn_b, tn_c)
}

/// Get all nodes in the subtree rooted at `root`, excluding the subtree containing `exclude`.
fn get_subtree_nodes<Id, Symm, V>(
    tn: &TreeTN<Id, Symm, V>,
    root: &V,
    exclude: &V,
) -> Result<Vec<V>>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let root_idx = tn
        .node_index(root)
        .ok_or_else(|| anyhow::anyhow!("Root node {:?} not found", root))?;
    let exclude_idx = tn
        .node_index(exclude)
        .ok_or_else(|| anyhow::anyhow!("Exclude node {:?} not found", exclude))?;

    let graph = tn.site_index_network().topology().graph();

    // DFS from root, but don't cross through exclude
    let mut result = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![root_idx];

    while let Some(node) = stack.pop() {
        if visited.contains(&node) {
            continue;
        }
        visited.insert(node);

        let node_name = tn
            .site_index_network()
            .topology()
            .node_name(node)
            .ok_or_else(|| anyhow::anyhow!("Node name not found for {:?}", node))?;
        result.push(node_name.clone());

        // Add neighbors except exclude
        for neighbor in graph.neighbors(node) {
            if neighbor != exclude_idx && !visited.contains(&neighbor) {
                stack.push(neighbor);
            }
        }
    }

    Ok(result)
}

/// Compute environment for a leaf node (no children in subtree).
fn compute_leaf_environment<Id, Symm, V>(
    node: &V,
    _towards: &V,
    tn_a: &TreeTN<Id, Symm, V>,
    tn_b: &TreeTN<Id, Symm, V>,
    tn_c: &TreeTN<Id, Symm, V>,
) -> Result<TensorDynLen<Id, Symm>>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    // Get tensors
    let node_idx_a = tn_a
        .node_index(node)
        .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tn_a", node))?;
    let node_idx_b = tn_b
        .node_index(node)
        .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tn_b", node))?;
    let node_idx_c = tn_c
        .node_index(node)
        .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tn_c", node))?;

    let tensor_a = tn_a
        .tensor(node_idx_a)
        .ok_or_else(|| anyhow::anyhow!("Tensor not found in tn_a"))?;
    let tensor_b = tn_b
        .tensor(node_idx_b)
        .ok_or_else(|| anyhow::anyhow!("Tensor not found in tn_b"))?;
    let tensor_c = tn_c
        .tensor(node_idx_c)
        .ok_or_else(|| anyhow::anyhow!("Tensor not found in tn_c"))?;

    // Get site indices (physical indices) for this node
    let site_space_a = tn_a.site_space(node).cloned().unwrap_or_default();
    let site_space_b = tn_b.site_space(node).cloned().unwrap_or_default();

    // Find common indices between A's site and B's site
    let common_site_pairs: Vec<_> = site_space_a
        .iter()
        .filter_map(|idx_a| {
            site_space_b
                .iter()
                .find(|idx_b| idx_a.id == idx_b.id)
                .map(|idx_b| (idx_a.clone(), idx_b.clone()))
        })
        .collect();

    // Contract A * B on site indices
    let ab = if common_site_pairs.is_empty() {
        // No common site indices - outer product
        tensor_a.tensordot(tensor_b, &[])?
    } else {
        tensor_a.tensordot(tensor_b, &common_site_pairs)?
    };

    // Contract with conj(C) on remaining site indices
    let c_conj = tensor_c.conj();

    // Find common indices between AB and conj(C)
    let site_space_c = tn_c.site_space(node).cloned().unwrap_or_default();
    let ab_c_common: Vec<_> = ab
        .indices
        .iter()
        .filter_map(|idx_ab| {
            site_space_c
                .iter()
                .find(|idx_c| idx_ab.id == idx_c.id)
                .map(|idx_c| (idx_ab.clone(), idx_c.clone()))
        })
        .collect();

    let env = ab.tensordot(&c_conj, &ab_c_common)?;

    Ok(env)
}

/// Contract a multi-node subtree to compute environment.
fn contract_subtree_environment<Id, Symm, V>(
    subtree_nodes: &[V],
    root: &V,
    towards: &V,
    tn_a: &TreeTN<Id, Symm, V>,
    tn_b: &TreeTN<Id, Symm, V>,
    tn_c: &TreeTN<Id, Symm, V>,
) -> Result<TensorDynLen<Id, Symm>>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let subtree_set: std::collections::HashSet<V> = subtree_nodes.iter().cloned().collect();

    // Process nodes from leaves towards root using topology order
    let root_idx = tn_c.node_index(root).unwrap();
    let towards_idx = tn_c.node_index(towards).unwrap();

    let graph = tn_c.site_index_network().topology().graph();

    // Collect postorder sequence, excluding 'towards' subtree
    fn collect_postorder<V: Clone + Hash + Eq>(
        graph: &petgraph::stable_graph::StableUnGraph<V, ()>,
        node: NodeIndex,
        exclude: NodeIndex,
        visited: &mut std::collections::HashSet<NodeIndex>,
        result: &mut Vec<NodeIndex>,
    ) {
        if visited.contains(&node) || node == exclude {
            return;
        }
        visited.insert(node);

        for neighbor in graph.neighbors(node) {
            if neighbor != exclude && !visited.contains(&neighbor) {
                collect_postorder(graph, neighbor, exclude, visited, result);
            }
        }
        result.push(node);
    }

    let mut visited = std::collections::HashSet::new();
    let mut postorder_nodes = Vec::new();
    collect_postorder(graph, root_idx, towards_idx, &mut visited, &mut postorder_nodes);

    // Store computed environments for each node
    let mut node_envs: HashMap<V, TensorDynLen<Id, Symm>> = HashMap::new();

    // Process nodes in postorder
    for node_idx in postorder_nodes {
        let node_name = tn_c
            .site_index_network()
            .topology()
            .node_name(node_idx)
            .unwrap()
            .clone();

        // Collect environments from children (neighbors in subtree, excluding towards)
        let neighbors: Vec<_> = tn_c.site_index_network().neighbors(&node_name).collect();
        let child_envs: Vec<_> = neighbors
            .iter()
            .filter(|n| subtree_set.contains(*n) && **n != *towards)
            .filter_map(|n| node_envs.get(n).cloned())
            .collect();

        // Get local tensors
        let node_idx_a = tn_a.node_index(&node_name).unwrap();
        let node_idx_b = tn_b.node_index(&node_name).unwrap();
        let node_idx_c = tn_c.node_index(&node_name).unwrap();

        let tensor_a = tn_a.tensor(node_idx_a).unwrap();
        let tensor_b = tn_b.tensor(node_idx_b).unwrap();
        let tensor_c = tn_c.tensor(node_idx_c).unwrap();

        // Contract: child_envs × A × B × conj(C)
        let env = if child_envs.is_empty() {
            // Leaf node: compute from tensors directly
            compute_leaf_environment(&node_name, towards, tn_a, tn_b, tn_c)?
        } else {
            // Non-leaf: contract with child environments
            let site_space_a = tn_a.site_space(&node_name).cloned().unwrap_or_default();
            let site_space_b = tn_b.site_space(&node_name).cloned().unwrap_or_default();

            // Contract A and B on site indices
            let common_site_pairs: Vec<_> = site_space_a
                .iter()
                .filter_map(|idx_a| {
                    site_space_b
                        .iter()
                        .find(|idx_b| idx_a.id == idx_b.id)
                        .map(|idx_b| (idx_a.clone(), idx_b.clone()))
                })
                .collect();

            let ab = if common_site_pairs.is_empty() {
                tensor_a.tensordot(tensor_b, &[])?
            } else {
                tensor_a.tensordot(tensor_b, &common_site_pairs)?
            };

            // Contract with conj(C)
            let c_conj = tensor_c.conj();
            let site_space_c = tn_c.site_space(&node_name).cloned().unwrap_or_default();
            let ab_c_common: Vec<_> = ab
                .indices
                .iter()
                .filter_map(|idx_ab| {
                    site_space_c
                        .iter()
                        .find(|idx_c| idx_ab.id == idx_c.id)
                        .map(|idx_c| (idx_ab.clone(), idx_c.clone()))
                })
                .collect();

            let mut result = ab.tensordot(&c_conj, &ab_c_common)?;

            // Contract with child environments
            for child_env in child_envs {
                // Find common indices between result and child_env
                let common: Vec<_> = result
                    .indices
                    .iter()
                    .filter_map(|idx_r| {
                        child_env
                            .indices
                            .iter()
                            .find(|idx_e| idx_r.id == idx_e.id)
                            .map(|idx_e| (idx_r.clone(), idx_e.clone()))
                    })
                    .collect();

                result = result.tensordot(&child_env, &common)?;
            }

            result
        };

        node_envs.insert(node_name, env);
    }

    // Return the environment at root
    node_envs
        .remove(root)
        .ok_or_else(|| anyhow::anyhow!("Failed to compute environment at root"))
}

// ============================================================================
// FitUpdater: LocalUpdater implementation for fit algorithm
// ============================================================================

/// Fit updater for variational contraction.
///
/// Implements the `LocalUpdater` trait to perform 2-site updates
/// that optimize `C ≈ A * B`.
#[derive(Debug)]
pub struct FitUpdater<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// First input TreeTN (with sim'd internal indices)
    pub tn_a: TreeTN<Id, Symm, V>,
    /// Second input TreeTN (with sim'd internal indices)
    pub tn_b: TreeTN<Id, Symm, V>,
    /// Environment cache
    pub envs: FitEnvironment<Id, Symm, V>,
    /// Maximum bond dimension
    pub max_rank: Option<usize>,
    /// Relative tolerance for truncation
    pub rtol: Option<f64>,
    /// Factorization algorithm
    pub factorize_alg: FactorizeAlg,
}

impl<Id, Symm, V> FitUpdater<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug + From<DynId>,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Create a new FitUpdater.
    ///
    /// # Arguments
    /// * `tn_a` - First input TreeTN (will be sim'd internally)
    /// * `tn_b` - Second input TreeTN (will be sim'd internally)
    /// * `max_rank` - Maximum bond dimension for truncation
    /// * `rtol` - Relative tolerance for truncation
    pub fn new(
        tn_a: TreeTN<Id, Symm, V>,
        tn_b: TreeTN<Id, Symm, V>,
        max_rank: Option<usize>,
        rtol: Option<f64>,
    ) -> Self {
        // Apply sim to avoid index collision
        let tn_a = tn_a.sim_internal_inds();
        let tn_b = tn_b.sim_internal_inds();

        Self {
            tn_a,
            tn_b,
            envs: FitEnvironment::new(),
            max_rank,
            rtol,
            factorize_alg: FactorizeAlg::SVD,
        }
    }

    /// Set the factorization algorithm.
    pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self {
        self.factorize_alg = alg;
        self
    }

    /// Initialize all environments towards a center node.
    ///
    /// Computes environments from all leaves towards the center.
    /// Should be called before the first sweep.
    pub fn initialize_environments(&mut self, center: &V, tn_c: &TreeTN<Id, Symm, V>) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug,
        V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        self.envs.clear();

        // Collect edges to process (from leaves towards center)
        let edges_to_center = tn_c
            .edges_to_canonicalize_by_names(center)
            .ok_or_else(|| anyhow::anyhow!("Failed to get edges to center"))?;

        // Process each edge
        for (from, to) in edges_to_center {
            let env = compute_environment_from_scratch(&from, &to, &self.tn_a, &self.tn_b, tn_c)?;
            self.envs.insert(from, to, env);
        }

        Ok(())
    }
}

impl<Id, Symm, V> LocalUpdater<Id, Symm, V> for FitUpdater<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    fn update(
        &mut self,
        mut subtree: TreeTN<Id, Symm, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<Id, Symm, V>,
    ) -> Result<TreeTN<Id, Symm, V>> {
        // FitUpdater is designed for nsite=2
        if step.nodes.len() != 2 {
            return Err(anyhow::anyhow!(
                "FitUpdater requires exactly 2 nodes, got {}",
                step.nodes.len()
            ));
        }

        let node_u = &step.nodes[0];
        let node_v = &step.nodes[1];

        // Get node indices in various TreeTNs
        let idx_u_a = self.tn_a.node_index(node_u).unwrap();
        let idx_v_a = self.tn_a.node_index(node_v).unwrap();
        let idx_u_b = self.tn_b.node_index(node_u).unwrap();
        let idx_v_b = self.tn_b.node_index(node_v).unwrap();

        // Get tensors
        let a_u = self.tn_a.tensor(idx_u_a).unwrap();
        let a_v = self.tn_a.tensor(idx_v_a).unwrap();
        let b_u = self.tn_b.tensor(idx_u_b).unwrap();
        let b_v = self.tn_b.tensor(idx_v_b).unwrap();

        // Collect environments from neighbors (excluding the edge between u and v)
        let mut env_tensors: Vec<TensorDynLen<Id, Symm>> = Vec::new();

        // Environments from u's neighbors (except v)
        for neighbor in full_treetn.site_index_network().neighbors(node_u) {
            if neighbor == *node_v {
                continue;
            }
            if let Some(env) = self.envs.get(&neighbor, node_u) {
                env_tensors.push(env.clone());
            }
        }

        // Environments from v's neighbors (except u)
        for neighbor in full_treetn.site_index_network().neighbors(node_v) {
            if neighbor == *node_u {
                continue;
            }
            if let Some(env) = self.envs.get(&neighbor, node_v) {
                env_tensors.push(env.clone());
            }
        }

        // Compute optimal 2-site tensor: env × A[u] × B[u] × A[v] × B[v] × env
        // Contract A[u] and B[u] on site indices
        let site_u_a = self.tn_a.site_space(node_u).cloned().unwrap_or_default();
        let site_u_b = self.tn_b.site_space(node_u).cloned().unwrap_or_default();
        let common_u: Vec<_> = site_u_a
            .iter()
            .filter_map(|idx_a| {
                site_u_b
                    .iter()
                    .find(|idx_b| idx_a.id == idx_b.id)
                    .map(|idx_b| (idx_a.clone(), idx_b.clone()))
            })
            .collect();

        let ab_u = a_u.tensordot(b_u, &common_u)?;

        // Contract A[v] and B[v] on site indices
        let site_v_a = self.tn_a.site_space(node_v).cloned().unwrap_or_default();
        let site_v_b = self.tn_b.site_space(node_v).cloned().unwrap_or_default();
        let common_v: Vec<_> = site_v_a
            .iter()
            .filter_map(|idx_a| {
                site_v_b
                    .iter()
                    .find(|idx_b| idx_a.id == idx_b.id)
                    .map(|idx_b| (idx_a.clone(), idx_b.clone()))
            })
            .collect();

        let ab_v = a_v.tensordot(b_v, &common_v)?;

        // Contract ab_u and ab_v on internal bonds
        let common_bonds: Vec<_> = ab_u
            .indices
            .iter()
            .filter_map(|idx_u| {
                ab_v.indices
                    .iter()
                    .find(|idx_v| idx_u.id == idx_v.id)
                    .map(|idx_v| (idx_u.clone(), idx_v.clone()))
            })
            .collect();

        let mut ab_uv = if common_bonds.is_empty() {
            ab_u.tensordot(&ab_v, &[])?
        } else {
            ab_u.tensordot(&ab_v, &common_bonds)?
        };

        // Contract with environment tensors
        for env in env_tensors {
            let common: Vec<_> = ab_uv
                .indices
                .iter()
                .filter_map(|idx| {
                    env.indices
                        .iter()
                        .find(|idx_e| idx.id == idx_e.id)
                        .map(|idx_e| (idx.clone(), idx_e.clone()))
                })
                .collect();
            ab_uv = ab_uv.tensordot(&env, &common)?;
        }

        // The result ab_uv is the optimal 2-site tensor
        // Factorize to get new C[u] and C[v]

        // Determine left indices (indices that will remain on u after factorization)
        let site_c_u = full_treetn.site_space(node_u).cloned().unwrap_or_default();
        let left_inds: Vec<_> = ab_uv
            .indices
            .iter()
            .filter(|idx| {
                // Keep site indices of u and link indices to u's other neighbors
                site_c_u.iter().any(|s| s.id == idx.id)
                    || full_treetn
                        .site_index_network()
                        .neighbors(node_u)
                        .filter(|n| n != node_v)
                        .any(|neighbor| {
                            full_treetn
                                .edge_between(node_u, &neighbor)
                                .and_then(|e| full_treetn.bond_index(e))
                                .map(|b| b.id == idx.id)
                                .unwrap_or(false)
                        })
            })
            .cloned()
            .collect();

        // Set up factorization options
        let mut options = match self.factorize_alg {
            FactorizeAlg::SVD => FactorizeOptions::svd(),
            FactorizeAlg::QR => FactorizeOptions::qr(),
            FactorizeAlg::LU => FactorizeOptions::lu(),
            FactorizeAlg::CI => FactorizeOptions::ci(),
        };
        options = options.with_canonical(Canonical::Left);

        if let Some(max_rank) = self.max_rank {
            options = options.with_max_rank(max_rank);
        }
        if let Some(rtol) = self.rtol {
            options = options.with_rtol(rtol);
        }

        // Factorize
        let factorize_result = factorize(&ab_uv, &left_inds, &options)
            .map_err(|e| anyhow::anyhow!("Factorization failed: {}", e))?;

        let new_tensor_u = factorize_result.left;
        let new_tensor_v = factorize_result.right;
        let new_bond = factorize_result.bond_index;

        // Get original bond index ID to preserve it
        let edge_uv = subtree.edge_between(node_u, node_v).unwrap();
        let old_bond = subtree.bond_index(edge_uv).unwrap().clone();

        // Replace bond index in new tensors with original ID
        let preserved_bond = Index::new_with_tags(
            old_bond.id.clone(),
            new_bond.symm.clone(),
            old_bond.tags.clone(),
        );

        let new_tensor_u = new_tensor_u.replaceind(&new_bond, &preserved_bond);
        let new_tensor_v = new_tensor_v.replaceind(&new_bond, &preserved_bond);

        // Update subtree
        let idx_u_sub = subtree.node_index(node_u).unwrap();
        let idx_v_sub = subtree.node_index(node_v).unwrap();

        subtree.replace_edge_bond(edge_uv, preserved_bond.clone())?;
        subtree.replace_tensor(idx_u_sub, new_tensor_u)?;
        subtree.replace_tensor(idx_v_sub, new_tensor_v)?;

        // Set ortho_towards
        subtree.set_ortho_towards(&preserved_bond.id, Some(step.new_center.clone()));
        subtree.set_canonical_center([step.new_center.clone()])?;

        // Update environment cache
        // Compute new environment for (u, v) direction
        let new_env = compute_environment_from_scratch(
            node_u,
            node_v,
            &self.tn_a,
            &self.tn_b,
            full_treetn, // Use full_treetn which will be updated after this
        )?;
        self.envs.update_after_step(node_u, node_v, new_env);

        Ok(subtree)
    }
}

// ============================================================================
// High-level API: contract_fit
// ============================================================================

/// Options for fit contraction.
#[derive(Debug, Clone)]
pub struct FitContractionOptions {
    /// Number of sweeps to perform.
    pub nsweeps: usize,
    /// Maximum bond dimension.
    pub max_rank: Option<usize>,
    /// Relative tolerance for truncation.
    pub rtol: Option<f64>,
    /// Factorization algorithm.
    pub factorize_alg: FactorizeAlg,
    /// Tolerance for early termination based on relative change.
    /// If `None`, run exactly `nsweeps` sweeps.
    /// If `Some(tol)`, stop early if `||C_{i+1} - C_i|| / ||C_i|| < tol`.
    pub convergence_tol: Option<f64>,
}

impl Default for FitContractionOptions {
    fn default() -> Self {
        Self {
            nsweeps: 2,
            max_rank: None,
            rtol: None,
            factorize_alg: FactorizeAlg::SVD,
            convergence_tol: None,
        }
    }
}

impl FitContractionOptions {
    /// Create new options with specified number of sweeps.
    pub fn new(nsweeps: usize) -> Self {
        Self {
            nsweeps,
            ..Default::default()
        }
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

    /// Set factorization algorithm.
    pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self {
        self.factorize_alg = alg;
        self
    }

    /// Set convergence tolerance for early termination.
    pub fn with_convergence_tol(mut self, tol: f64) -> Self {
        self.convergence_tol = Some(tol);
        self
    }
}

/// Contract two TreeTNs using the fit (variational) algorithm.
///
/// # Arguments
/// * `tn_a` - First TreeTN
/// * `tn_b` - Second TreeTN (must have same topology as `tn_a`)
/// * `center` - Node to use as sweep center
/// * `options` - Contraction options
///
/// # Returns
/// The contracted TreeTN `C ≈ A * B`, or an error if contraction fails.
///
/// # Algorithm
/// 1. Initialize C using zipup contraction
/// 2. Apply fit algorithm with sweeps
/// 3. Return optimized C
pub fn contract_fit<Id, Symm, V>(
    tn_a: &TreeTN<Id, Symm, V>,
    tn_b: &TreeTN<Id, Symm, V>,
    center: &V,
    options: FitContractionOptions,
) -> Result<TreeTN<Id, Symm, V>>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    use super::localupdate::apply_local_update_sweep;
    use crate::CanonicalizationOptions;

    // Validate topologies match
    if !tn_a.same_topology(tn_b) {
        return Err(anyhow::anyhow!(
            "TreeTNs must have the same topology for fit contraction"
        ));
    }

    // Initialize C using zipup (arguments: rtol, max_rank)
    let mut tn_c = tn_a.contract_zipup(tn_b, center, options.rtol, options.max_rank)?;

    // Canonicalize towards center
    tn_c = tn_c.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    // Create FitUpdater
    let mut updater = FitUpdater::new(
        tn_a.clone(),
        tn_b.clone(),
        options.max_rank,
        options.rtol,
    )
    .with_factorize_alg(options.factorize_alg);

    // Initialize environments
    updater.initialize_environments(center, &tn_c)?;

    // Create sweep plan
    let plan = LocalUpdateSweepPlan::from_treetn(&tn_c, center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create sweep plan"))?;

    // Perform sweeps
    for _sweep in 0..options.nsweeps {
        let log_norm_before = if options.convergence_tol.is_some() {
            Some(tn_c.log_norm()?)
        } else {
            None
        };

        apply_local_update_sweep(&mut tn_c, &plan, &mut updater)?;

        // Check convergence using log_norm
        if let (Some(tol), Some(log_norm_before)) = (options.convergence_tol, log_norm_before) {
            let log_norm_after = tn_c.log_norm()?;
            // relative change in norm: |exp(log_after) - exp(log_before)| / exp(log_before)
            // = |exp(log_after - log_before) - 1|
            let relative_change = (f64::exp(log_norm_after - log_norm_before) - 1.0).abs();
            if relative_change < tol {
                break;
            }
        }
    }

    Ok(tn_c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tensor4all::index::DefaultIndex;
    use tensor4all::storage::{DenseStorageF64, Storage};
    use tensor4all::NoSymmSpace;

    type TestIndex = DefaultIndex<DynId>;

    #[test]
    fn test_fit_environment_new() {
        let env: FitEnvironment<DynId, NoSymmSpace, String> = FitEnvironment::new();
        assert!(env.is_empty());
        assert_eq!(env.len(), 0);
    }

    #[test]
    fn test_fit_environment_insert_get() {
        let mut env: FitEnvironment<DynId, NoSymmSpace, String> = FitEnvironment::new();

        let idx = TestIndex::new_dyn(2);
        let tensor = TensorDynLen::new(
            vec![idx],
            vec![2],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 2.0]))),
        );

        env.insert("A".to_string(), "B".to_string(), tensor.clone());

        assert_eq!(env.len(), 1);
        assert!(env.contains(&"A".to_string(), &"B".to_string()));
        assert!(!env.contains(&"B".to_string(), &"A".to_string()));

        let retrieved = env.get(&"A".to_string(), &"B".to_string());
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_fit_environment_update_after_step() {
        let mut env: FitEnvironment<DynId, NoSymmSpace, String> = FitEnvironment::new();

        let idx = TestIndex::new_dyn(2);
        let tensor1 = TensorDynLen::new(
            vec![idx.clone()],
            vec![2],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 2.0]))),
        );
        let tensor2 = TensorDynLen::new(
            vec![idx.clone()],
            vec![2],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![3.0, 4.0]))),
        );

        // Insert initial environment
        env.insert("B".to_string(), "A".to_string(), tensor1);
        assert!(env.contains(&"B".to_string(), &"A".to_string()));

        // Update after step A -> B
        env.update_after_step(&"A".to_string(), &"B".to_string(), tensor2);

        // (A, B) should exist, (B, A) should be removed
        assert!(env.contains(&"A".to_string(), &"B".to_string()));
        assert!(!env.contains(&"B".to_string(), &"A".to_string()));
    }

    #[test]
    fn test_fit_contraction_options_default() {
        let options = FitContractionOptions::default();
        assert_eq!(options.nsweeps, 2);
        assert!(options.max_rank.is_none());
        assert!(options.rtol.is_none());
        assert!(options.convergence_tol.is_none());
    }

    #[test]
    fn test_fit_contraction_options_builder() {
        let options = FitContractionOptions::new(5)
            .with_max_rank(10)
            .with_rtol(1e-6)
            .with_convergence_tol(1e-8);

        assert_eq!(options.nsweeps, 5);
        assert_eq!(options.max_rank, Some(10));
        assert_eq!(options.rtol, Some(1e-6));
        assert_eq!(options.convergence_tol, Some(1e-8));
    }
}
