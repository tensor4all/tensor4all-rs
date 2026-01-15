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

use tensor4all_core::{
    AllowedPairs, Canonical, FactorizeAlg, FactorizeOptions, IndexLike, TensorLike,
};

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
/// # Lazy Evaluation
///
/// The cache starts empty. When an environment is requested via `get_or_compute`,
/// it is computed recursively from the leaves and cached for future use.
///
/// # Cache Invalidation
///
/// When tensors in a region T are updated, all caches containing those tensors
/// must be invalidated. The invalidation propagates recursively from T towards
/// the leaves of the tree.
#[derive(Debug, Clone)]
pub struct FitEnvironment<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq,
{
    /// Environment tensors: (from, to) -> tensor
    envs: HashMap<(V, V), T>,
}

impl<T, V> FitEnvironment<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Create an empty environment cache.
    pub fn new() -> Self {
        Self {
            envs: HashMap::new(),
        }
    }

    /// Get the environment tensor for edge (from, to) if it exists.
    pub fn get(&self, from: &V, to: &V) -> Option<&T> {
        self.envs.get(&(from.clone(), to.clone()))
    }

    /// Insert an environment tensor for edge (from, to).
    /// This is mainly for testing; normally use `get_or_compute` for lazy evaluation.
    #[allow(dead_code)]
    pub(crate) fn insert(&mut self, from: V, to: V, env: T) {
        self.envs.insert((from, to), env);
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

    /// Get or compute the environment tensor for edge (from, to).
    ///
    /// If the environment is cached, returns it directly.
    /// Otherwise, recursively computes it from child environments (towards leaves)
    /// and caches the result.
    ///
    /// # Arguments
    /// * `from` - The node whose subtree we're computing
    /// * `to` - The direction we're looking towards
    /// * `tn_a` - First input TreeTN
    /// * `tn_b` - Second input TreeTN
    /// * `tn_c` - Current approximation TreeTN
    pub fn get_or_compute(
        &mut self,
        from: &V,
        to: &V,
        tn_a: &TreeTN<T, V>,
        tn_b: &TreeTN<T, V>,
        tn_c: &TreeTN<T, V>,
    ) -> Result<T>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        // If already cached, return a clone
        if let Some(env) = self.envs.get(&(from.clone(), to.clone())) {
            return Ok(env.clone());
        }

        // Get neighbors of `from` excluding `to`
        let child_neighbors: Vec<V> = tn_c
            .site_index_network()
            .neighbors(from)
            .filter(|n| n != to)
            .collect();

        // Recursively get or compute child environments
        let child_envs: Vec<T> = child_neighbors
            .iter()
            .map(|child| self.get_or_compute(child, from, tn_a, tn_b, tn_c))
            .collect::<Result<Vec<_>>>()?;

        // Compute the environment for (from, to) using child environments
        let env = compute_single_node_environment(from, to, tn_a, tn_b, tn_c, &child_envs)?;

        // Cache and return
        self.envs.insert((from.clone(), to.clone()), env.clone());
        Ok(env)
    }

    /// Invalidate all caches affected by updates to tensors in region T.
    ///
    /// For each `t ∈ T`:
    /// 1. Remove all `env[(t, *)]` (0th generation)
    /// 2. Recursively remove caches propagating towards leaves
    ///
    /// # Arguments
    /// * `region` - The set of nodes whose tensors were updated
    /// * `tn_c` - The TreeTN (for topology information)
    pub fn invalidate<'a>(&mut self, region: impl IntoIterator<Item = &'a V>, tn_c: &TreeTN<T, V>)
    where
        V: 'a + Send + Sync,
    {
        for t in region {
            // Get all neighbors of t
            let neighbors: Vec<V> = tn_c.site_index_network().neighbors(t).collect();

            // Remove all env[(t, *)] and propagate recursively
            for neighbor in neighbors {
                self.invalidate_recursive(t, &neighbor, tn_c);
            }
        }
    }

    /// Recursively invalidate caches starting from env[(from, to)] towards leaves.
    ///
    /// If env[(from, to)] exists, remove it and propagate to env[(to, x)] for all x ≠ from.
    fn invalidate_recursive(&mut self, from: &V, to: &V, tn_c: &TreeTN<T, V>) {
        // Remove env[(from, to)] if it exists
        if self.envs.remove(&(from.clone(), to.clone())).is_some() {
            // Propagate to next generation: env[(to, x)] for all neighbors x of to, x ≠ from
            let neighbors: Vec<V> = tn_c
                .site_index_network()
                .neighbors(to)
                .filter(|n| n != from)
                .collect();

            for neighbor in neighbors {
                self.invalidate_recursive(to, &neighbor, tn_c);
            }
        }
    }

    /// Verify cache structural consistency.
    ///
    /// For any `env[(x, x1)]` where `x` is not a leaf (has neighbors other than `x1`),
    /// all child environments `env[(y, x)]` for neighbors `y ≠ x1` must exist.
    ///
    /// # Arguments
    /// * `tn_c` - The TreeTN (for topology information)
    ///
    /// # Returns
    /// `Ok(())` if consistent, or an error describing the inconsistency.
    pub fn verify_structural_consistency(&self, tn_c: &TreeTN<T, V>) -> Result<()>
    where
        V: Clone + Hash + Eq + std::fmt::Debug,
    {
        for (from, to) in self.envs.keys() {
            // Get neighbors of `from` excluding `to`
            let child_neighbors: Vec<V> = tn_c
                .site_index_network()
                .neighbors(from)
                .filter(|n| n != to)
                .collect();

            // If `from` is not a leaf, all child environments must exist
            for child in &child_neighbors {
                if !self.envs.contains_key(&(child.clone(), from.clone())) {
                    return Err(anyhow::anyhow!(
                        "Structural inconsistency: env[({:?}, {:?})] exists but child env[({:?}, {:?})] is missing",
                        from, to, child, from
                    ));
                }
            }
        }
        Ok(())
    }
}

impl<T, V> Default for FitEnvironment<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Environment computation helpers
// ============================================================================

/// Compute environment for a leaf node (no children in subtree).
fn compute_leaf_environment<T, V>(
    node: &V,
    _towards: &V,
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    tn_c: &TreeTN<T, V>,
) -> Result<T>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
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

    // Contract A × B × conj(C) - collect all tensors and let contract() find the optimal contraction order
    let c_conj = tensor_c.conj();
    let env = T::contract(
        &[tensor_a.clone(), tensor_b.clone(), c_conj],
        AllowedPairs::All,
    )
    .map_err(|e| anyhow::anyhow!("contract failed: {}", e))?;

    Ok(env)
}

/// Compute environment for a single node using child environments.
///
/// This computes: child_envs × A[node] × B[node] × conj(C[node])
/// leaving open only the indices connecting to `towards`.
fn compute_single_node_environment<T, V>(
    node: &V,
    towards: &V,
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    tn_c: &TreeTN<T, V>,
    child_envs: &[T],
) -> Result<T>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    // Get local tensors
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

    if child_envs.is_empty() {
        // Leaf node: compute from tensors directly
        return compute_leaf_environment(node, towards, tn_a, tn_b, tn_c);
    }

    // Non-leaf: contract A × B × conj(C) × child_envs
    // Collect all tensors and let contract() find the optimal contraction order
    let c_conj = tensor_c.conj();
    let mut tensor_list = vec![tensor_a.clone(), tensor_b.clone(), c_conj];
    tensor_list.extend(child_envs.iter().cloned());
    let result = T::contract(&tensor_list, AllowedPairs::All)
        .map_err(|e| anyhow::anyhow!("contract failed: {}", e))?;

    Ok(result)
}

// ============================================================================
// FitUpdater: LocalUpdater implementation for fit algorithm
// ============================================================================

/// Fit updater for variational contraction.
///
/// Implements the `LocalUpdater` trait to perform 2-site updates
/// that optimize `C ≈ A * B`.
#[derive(Debug)]
pub struct FitUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// First input TreeTN (with sim'd internal indices)
    pub tn_a: TreeTN<T, V>,
    /// Second input TreeTN (with sim'd internal indices)
    pub tn_b: TreeTN<T, V>,
    /// Environment cache
    pub envs: FitEnvironment<T, V>,
    /// Maximum bond dimension
    pub max_rank: Option<usize>,
    /// Relative tolerance for truncation
    pub rtol: Option<f64>,
    /// Factorization algorithm
    pub factorize_alg: FactorizeAlg,
}

impl<T, V> FitUpdater<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Create a new FitUpdater.
    ///
    /// # Arguments
    /// * `tn_a` - First input TreeTN
    /// * `tn_b` - Second input TreeTN
    /// * `max_rank` - Maximum bond dimension for truncation
    /// * `rtol` - Relative tolerance for truncation
    ///
    /// Note: sim_internal_inds() should be called on tn_a and tn_b before passing
    /// if index collision is a concern. This is not done here because contraction
    /// module (which provides sim_internal_inds) is currently disabled.
    pub fn new(
        tn_a: TreeTN<T, V>,
        tn_b: TreeTN<T, V>,
        max_rank: Option<usize>,
        rtol: Option<f64>,
    ) -> Self {
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
}

impl<T, V> LocalUpdater<T, V> for FitUpdater<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    fn update(
        &mut self,
        mut subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<T, V>,
    ) -> Result<TreeTN<T, V>> {
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
        // Uses lazy evaluation: computes and caches if not already present
        let mut env_tensors: Vec<T> = Vec::new();

        // Environments from u's neighbors (except v)
        for neighbor in full_treetn.site_index_network().neighbors(node_u) {
            if neighbor == *node_v {
                continue;
            }
            let env =
                self.envs
                    .get_or_compute(&neighbor, node_u, &self.tn_a, &self.tn_b, full_treetn)?;
            env_tensors.push(env);
        }

        // Environments from v's neighbors (except u)
        for neighbor in full_treetn.site_index_network().neighbors(node_v) {
            if neighbor == *node_u {
                continue;
            }
            let env =
                self.envs
                    .get_or_compute(&neighbor, node_v, &self.tn_a, &self.tn_b, full_treetn)?;
            env_tensors.push(env);
        }

        // Compute optimal 2-site tensor: env × A[u] × B[u] × A[v] × B[v] × env
        // Collect all tensors and let contract() find the optimal contraction order
        let mut tensor_list = vec![a_u.clone(), b_u.clone(), a_v.clone(), b_v.clone()];
        tensor_list.extend(env_tensors);
        let ab_uv = T::contract(&tensor_list, AllowedPairs::All)
            .map_err(|e| anyhow::anyhow!("contract failed: {}", e))?;

        // The result ab_uv is the optimal 2-site tensor
        // Factorize to get new C[u] and C[v]

        // Determine left indices (indices that will remain on u after factorization)
        let site_c_u = full_treetn.site_space(node_u).cloned().unwrap_or_default();
        let left_inds: Vec<_> = ab_uv
            .external_indices()
            .iter()
            .filter(|idx| {
                // Keep site indices of u and link indices to u's other neighbors
                site_c_u.iter().any(|s| s.same_id(*idx))
                    || full_treetn
                        .site_index_network()
                        .neighbors(node_u)
                        .filter(|n| n != node_v)
                        .any(|neighbor| {
                            full_treetn
                                .edge_between(node_u, &neighbor)
                                .and_then(|e| full_treetn.bond_index(e))
                                .map(|b| b.same_id(*idx))
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

        // Factorize using TensorLike::factorize
        let factorize_result = ab_uv
            .factorize(&left_inds, &options)
            .map_err(|e| anyhow::anyhow!("Factorization failed: {}", e))?;

        let new_tensor_u = factorize_result.left;
        let new_tensor_v = factorize_result.right;
        let new_bond = factorize_result.bond_index;

        // Get edge between u and v
        let edge_uv = subtree.edge_between(node_u, node_v).unwrap();

        // Update subtree with new bond and tensors
        let idx_u_sub = subtree.node_index(node_u).unwrap();
        let idx_v_sub = subtree.node_index(node_v).unwrap();

        subtree.replace_edge_bond(edge_uv, new_bond.clone())?;
        subtree.replace_tensor(idx_u_sub, new_tensor_u)?;
        subtree.replace_tensor(idx_v_sub, new_tensor_v)?;

        // Set ortho_towards
        subtree.set_ortho_towards(&new_bond, Some(step.new_center.clone()));
        subtree.set_canonical_center([step.new_center.clone()])?;

        Ok(subtree)
    }

    fn after_step(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_after: &TreeTN<T, V>,
    ) -> Result<()> {
        // Invalidate all caches affected by the updated region
        self.envs.invalidate(&step.nodes, full_treetn_after);
        Ok(())
    }
}

// ============================================================================
// High-level API: contract_fit
// ============================================================================

/// Options for fit contraction.
#[derive(Debug, Clone)]
pub struct FitContractionOptions {
    /// Number of full sweeps to perform.
    ///
    /// A full sweep visits each edge twice (forward and backward) using an Euler tour.
    pub nfullsweeps: usize,
    /// Maximum bond dimension.
    pub max_rank: Option<usize>,
    /// Relative tolerance for truncation.
    pub rtol: Option<f64>,
    /// Factorization algorithm.
    pub factorize_alg: FactorizeAlg,
    /// Tolerance for early termination based on relative change.
    /// If `None`, run exactly `nfullsweeps` sweeps.
    /// If `Some(tol)`, stop early if `||C_{i+1} - C_i|| / ||C_i|| < tol`.
    pub convergence_tol: Option<f64>,
}

impl Default for FitContractionOptions {
    fn default() -> Self {
        Self {
            nfullsweeps: 1,
            max_rank: None,
            rtol: None,
            factorize_alg: FactorizeAlg::SVD,
            convergence_tol: None,
        }
    }
}

impl FitContractionOptions {
    /// Create new options with specified number of full sweeps.
    pub fn new(nfullsweeps: usize) -> Self {
        Self {
            nfullsweeps,
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
/// This algorithm minimizes `||A*B - C||²` iteratively by optimizing
/// each local tensor of C while keeping others fixed.
///
/// # Arguments
/// * `tn_a` - First TreeTN
/// * `tn_b` - Second TreeTN
/// * `center` - Node to use as canonical center
/// * `options` - Fit algorithm options
///
/// # Returns
/// A new TreeTN representing the contracted result.
pub fn contract_fit<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: FitContractionOptions,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
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

    // Create FitUpdater (environments are computed lazily)
    let mut updater = FitUpdater::new(tn_a.clone(), tn_b.clone(), options.max_rank, options.rtol)
        .with_factorize_alg(options.factorize_alg);

    // Create sweep plan
    let plan = LocalUpdateSweepPlan::from_treetn(&tn_c, center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create sweep plan"))?;

    // Perform sweeps
    for _sweep in 0..options.nfullsweeps {
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
