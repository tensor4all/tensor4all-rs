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
use std::time::{Duration, Instant};

use anyhow::Result;

use tensor4all_core::{
    print_and_reset_contract_profile, print_and_reset_native_einsum_profile,
    reset_contract_profile, reset_native_einsum_profile, AllowedPairs, Canonical, FactorizeAlg,
    FactorizeOptions, IndexLike, SvdTruncationPolicy, TensorLike,
};

use super::localupdate::{LocalUpdateStep, LocalUpdateSweepPlan, LocalUpdater};
use super::TreeTN;

#[derive(Debug, Default, Clone)]
struct FitProfile {
    zipup_init_time: Duration,
    canonicalize_time: Duration,
    sweep_time: Duration,
    env_get_time: Duration,
    env_leaf_time: Duration,
    env_internal_time: Duration,
    left_inds_time: Duration,
    two_site_contract_time: Duration,
    factorize_time: Duration,
    replace_time: Duration,
    invalidate_time: Duration,
    env_requests: usize,
    env_hits: usize,
    env_misses: usize,
    step_count: usize,
    sweep_count: usize,
}

thread_local! {
    static FIT_PROFILE_STATE: std::cell::RefCell<Option<FitProfile>> =
        const { std::cell::RefCell::new(None) };
}

fn fit_profile_enabled() -> bool {
    std::env::var("T4A_PROFILE_FIT").is_ok()
}

fn fit_profile_reset() {
    if fit_profile_enabled() {
        FIT_PROFILE_STATE.with(|state| {
            *state.borrow_mut() = Some(FitProfile::default());
        });
    }
}

fn with_fit_profile(f: impl FnOnce(&mut FitProfile)) {
    if fit_profile_enabled() {
        FIT_PROFILE_STATE.with(|state| {
            if let Some(profile) = state.borrow_mut().as_mut() {
                f(profile);
            }
        });
    }
}

fn take_fit_profile() -> Option<FitProfile> {
    FIT_PROFILE_STATE.with(|state| state.borrow_mut().take())
}

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
        let started = fit_profile_enabled().then(Instant::now);
        with_fit_profile(|profile| {
            profile.env_requests += 1;
        });

        // If already cached, return a clone
        if let Some(env) = self.envs.get(&(from.clone(), to.clone())) {
            if let Some(started) = started {
                with_fit_profile(|profile| {
                    profile.env_hits += 1;
                    profile.env_get_time += started.elapsed();
                });
            }
            return Ok(env.clone());
        }

        with_fit_profile(|profile| {
            profile.env_misses += 1;
        });

        // Get neighbors of `from` excluding `to`
        let mut child_neighbors: Vec<V> = tn_c
            .site_index_network()
            .neighbors(from)
            .filter(|n| n != to)
            .collect();
        child_neighbors.sort_by_key(|node| {
            tn_c.node_index(node)
                .expect("neighbor must exist in site index network")
                .index()
        });

        // Recursively get or compute child environments
        let child_envs: Vec<T> = child_neighbors
            .iter()
            .map(|child| self.get_or_compute(child, from, tn_a, tn_b, tn_c))
            .collect::<Result<Vec<_>>>()?;

        // Compute the environment for (from, to) using child environments
        let env = compute_single_node_environment(from, to, tn_a, tn_b, tn_c, &child_envs)?;

        // Cache and return
        self.envs.insert((from.clone(), to.clone()), env.clone());
        if let Some(started) = started {
            with_fit_profile(|profile| {
                profile.env_get_time += started.elapsed();
            });
        }
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
            let mut neighbors: Vec<V> = tn_c.site_index_network().neighbors(t).collect();
            neighbors.sort_by_key(|node| {
                tn_c.node_index(node)
                    .expect("neighbor must exist in site index network")
                    .index()
            });

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
            let mut neighbors: Vec<V> = tn_c
                .site_index_network()
                .neighbors(to)
                .filter(|n| n != from)
                .collect();
            neighbors.sort_by_key(|node| {
                tn_c.node_index(node)
                    .expect("neighbor must exist in site index network")
                    .index()
            });

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
            let mut child_neighbors: Vec<V> = tn_c
                .site_index_network()
                .neighbors(from)
                .filter(|n| n != to)
                .collect();
            child_neighbors.sort_by_key(|node| {
                tn_c.node_index(node)
                    .expect("neighbor must exist in site index network")
                    .index()
            });

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
    let started = fit_profile_enabled().then(Instant::now);

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

    // Contract A × B × conj(C) with a single multi-tensor call.
    let c_conj = tensor_c.conj();
    let env = T::contract(&[tensor_a, tensor_b, &c_conj], AllowedPairs::All)
        .map_err(|e| anyhow::anyhow!("contract failed: {}", e))?;

    if let Some(started) = started {
        with_fit_profile(|profile| {
            profile.env_leaf_time += started.elapsed();
        });
    }

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
    let started = fit_profile_enabled().then(Instant::now);

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

    // Non-leaf: contract A × B × conj(C) × child_envs in one multi-tensor call.
    let c_conj = tensor_c.conj();
    let mut tensor_refs: Vec<&T> = vec![tensor_a, tensor_b, &c_conj];
    tensor_refs.extend(child_envs.iter());
    let result = T::contract(&tensor_refs, AllowedPairs::All)
        .map_err(|e| anyhow::anyhow!("contract failed: {}", e))?;

    if let Some(started) = started {
        with_fit_profile(|profile| {
            profile.env_internal_time += started.elapsed();
        });
    }

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
    /// Legacy relative tolerance retained for same-crate tests and call chains.
    pub(crate) rtol: Option<f64>,
    /// Explicit SVD truncation policy
    pub svd_policy: Option<SvdTruncationPolicy>,
    /// QR-specific relative tolerance
    pub qr_rtol: Option<f64>,
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
    /// * `svd_policy` - Explicit SVD truncation policy
    /// * `qr_rtol` - QR-specific relative tolerance
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
            svd_policy: rtol.map(SvdTruncationPolicy::new),
            qr_rtol: None,
            factorize_alg: FactorizeAlg::SVD,
        }
    }

    /// Set the factorization algorithm.
    pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self {
        self.factorize_alg = alg;
        self
    }

    /// Set the SVD truncation policy used by fit sweeps.
    pub(crate) fn with_svd_policy(mut self, policy: Option<SvdTruncationPolicy>) -> Self {
        self.rtol = policy.map(|value| value.threshold);
        self.svd_policy = policy;
        self
    }

    /// Set the QR-specific relative tolerance used by fit sweeps.
    pub(crate) fn with_qr_rtol(mut self, qr_rtol: Option<f64>) -> Self {
        self.qr_rtol = qr_rtol;
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
        with_fit_profile(|profile| {
            profile.step_count += 1;
        });

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
        let mut u_neighbors: Vec<V> = full_treetn.site_index_network().neighbors(node_u).collect();
        u_neighbors.sort_by_key(|node| {
            full_treetn
                .node_index(node)
                .expect("neighbor must exist in site index network")
                .index()
        });
        for neighbor in u_neighbors {
            if neighbor == *node_v {
                continue;
            }
            let env =
                self.envs
                    .get_or_compute(&neighbor, node_u, &self.tn_a, &self.tn_b, full_treetn)?;
            env_tensors.push(env);
        }

        // Environments from v's neighbors (except u)
        let mut v_neighbors: Vec<V> = full_treetn.site_index_network().neighbors(node_v).collect();
        v_neighbors.sort_by_key(|node| {
            full_treetn
                .node_index(node)
                .expect("neighbor must exist in site index network")
                .index()
        });
        for neighbor in v_neighbors {
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
        let contract_started = fit_profile_enabled().then(Instant::now);
        let mut tensor_refs: Vec<&T> = vec![a_u, b_u, a_v, b_v];
        tensor_refs.extend(env_tensors.iter());
        let ab_uv = T::contract(&tensor_refs, AllowedPairs::All)
            .map_err(|e| anyhow::anyhow!("contract failed: {}", e))?;
        if let Some(contract_started) = contract_started {
            with_fit_profile(|profile| {
                profile.two_site_contract_time += contract_started.elapsed();
            });
        }

        // The result ab_uv is the optimal 2-site tensor
        // Factorize to get new C[u] and C[v]

        // Determine left indices (indices that will remain on u after factorization)
        let left_inds_started = fit_profile_enabled().then(Instant::now);
        let site_c_u = full_treetn.site_space(node_u).cloned().unwrap_or_default();
        let mut left_inds: Vec<_> = ab_uv
            .external_indices()
            .iter()
            .filter(|idx| {
                // Keep site indices of u and link indices to u's other neighbors
                site_c_u.contains(*idx)
                    || full_treetn
                        .site_index_network()
                        .neighbors(node_u)
                        .filter(|n| n != node_v)
                        .any(|neighbor| {
                            full_treetn
                                .edge_between(node_u, &neighbor)
                                .and_then(|e| full_treetn.bond_index(e))
                                .map(|b| b == *idx)
                                .unwrap_or(false)
                        })
            })
            .cloned()
            .collect();
        left_inds.sort_by(|a, b| a.id().cmp(b.id()));
        if let Some(left_inds_started) = left_inds_started {
            with_fit_profile(|profile| {
                profile.left_inds_time += left_inds_started.elapsed();
            });
        }

        // Set up factorization options
        let mut options = match self.factorize_alg {
            FactorizeAlg::SVD => FactorizeOptions::svd(),
            FactorizeAlg::QR => FactorizeOptions::qr(),
            FactorizeAlg::LU => FactorizeOptions::lu(),
            FactorizeAlg::CI => FactorizeOptions::ci(),
        };
        options = options.with_canonical(Canonical::Left);

        if let Some(policy) = self.svd_policy {
            options = options.with_svd_policy(policy);
        }
        if let Some(qr_rtol) = self.qr_rtol {
            options = options.with_qr_rtol(qr_rtol);
        }
        options
            .validate()
            .map_err(|e| anyhow::anyhow!("invalid fit factorization options: {e}"))?;

        // Determine bond dimension cap for this factorization step.
        // - If max_rank is explicitly specified, use it.
        // - If an algorithm-specific tolerance is specified (but max_rank is not),
        //   allow bonds to grow freely and let the factorization policy decide.
        // - Otherwise, cap at the existing bond dimension to preserve the zipup
        //   initialization size.
        let bond_cap = if self.max_rank.is_some() {
            self.max_rank
        } else if self.svd_policy.is_some() || self.qr_rtol.is_some() {
            None
        } else {
            subtree
                .edge_between(node_u, node_v)
                .and_then(|e| subtree.bond_index(e))
                .map(|b| b.dim())
        };
        if let Some(cap) = bond_cap {
            options = options.with_max_rank(cap);
        }

        // Factorize using TensorLike::factorize
        let factorize_started = fit_profile_enabled().then(Instant::now);
        let factorize_result = ab_uv
            .factorize(&left_inds, &options)
            .map_err(|e| anyhow::anyhow!("Factorization failed: {}", e))?;
        if let Some(factorize_started) = factorize_started {
            with_fit_profile(|profile| {
                profile.factorize_time += factorize_started.elapsed();
            });
        }

        let new_tensor_u = factorize_result.left;
        let new_tensor_v = factorize_result.right;
        let new_bond = factorize_result.bond_index;

        // Get edge between u and v
        let edge_uv = subtree.edge_between(node_u, node_v).unwrap();

        // Update subtree with new bond and tensors
        let idx_u_sub = subtree.node_index(node_u).unwrap();
        let idx_v_sub = subtree.node_index(node_v).unwrap();

        let replace_started = fit_profile_enabled().then(Instant::now);
        subtree.replace_edge_bond(edge_uv, new_bond.clone())?;
        subtree.replace_tensor(idx_u_sub, new_tensor_u)?;
        subtree.replace_tensor(idx_v_sub, new_tensor_v)?;

        // Set ortho_towards
        subtree.set_ortho_towards(&new_bond, Some(step.new_center.clone()));
        subtree.set_canonical_region([step.new_center.clone()])?;
        if let Some(replace_started) = replace_started {
            with_fit_profile(|profile| {
                profile.replace_time += replace_started.elapsed();
            });
        }

        Ok(subtree)
    }

    fn after_step(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_after: &TreeTN<T, V>,
    ) -> Result<()> {
        // Invalidate all caches affected by the updated region
        let started = fit_profile_enabled().then(Instant::now);
        self.envs.invalidate(&step.nodes, full_treetn_after);
        if let Some(started) = started {
            with_fit_profile(|profile| {
                profile.invalidate_time += started.elapsed();
            });
        }
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
    /// Legacy relative tolerance retained for same-crate tests and call chains.
    pub(crate) rtol: Option<f64>,
    /// Explicit SVD truncation policy.
    pub svd_policy: Option<SvdTruncationPolicy>,
    /// QR-specific relative tolerance.
    pub qr_rtol: Option<f64>,
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
            svd_policy: None,
            qr_rtol: None,
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

    /// Set the SVD truncation policy.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.rtol = Some(policy.threshold);
        self.svd_policy = Some(policy);
        self
    }

    /// Set the QR-specific relative tolerance.
    pub fn with_qr_rtol(mut self, rtol: f64) -> Self {
        self.qr_rtol = Some(rtol);
        self
    }

    /// Set relative tolerance as a per-value relative SVD policy.
    pub(crate) fn with_rtol(self, rtol: f64) -> Self {
        self.with_svd_policy(SvdTruncationPolicy::new(rtol))
    }

    /// Get the legacy SVD threshold value when represented as an rtol.
    pub(crate) fn rtol(&self) -> Option<f64> {
        self.svd_policy.map(|policy| policy.threshold)
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
    use crate::CanonicalForm;
    let profile_enabled = fit_profile_enabled();
    if profile_enabled {
        fit_profile_reset();
    }
    reset_contract_profile();
    reset_native_einsum_profile();

    // Validate topologies match
    if !tn_a.same_topology(tn_b) {
        return Err(anyhow::anyhow!(
            "TreeTNs must have the same topology for fit contraction"
        ));
    }

    // Initialize C using the SVD-based zipup contraction.
    let zipup_started = profile_enabled.then(Instant::now);
    let mut tn_c = tn_a.contract_zipup_tree_accumulated(
        tn_b,
        center,
        CanonicalForm::Unitary,
        options.svd_policy,
        options.max_rank,
    )?;
    if let Some(zipup_started) = zipup_started {
        with_fit_profile(|profile| {
            profile.zipup_init_time += zipup_started.elapsed();
        });
    }

    // The zip-up initializer already returns a network centered at `center`.

    // With neither max_rank nor an algorithm-specific truncation override, fit
    // sweeps cannot change the bond cap and only introduce numerical drift
    // relative to the zip-up initializer.
    if options.max_rank.is_none() && options.svd_policy.is_none() && options.qr_rtol.is_none() {
        return Ok(tn_c);
    }

    // Create FitUpdater (environments are computed lazily)
    let mut updater = FitUpdater::new(tn_a.clone(), tn_b.clone(), options.max_rank, None)
        .with_svd_policy(options.svd_policy)
        .with_qr_rtol(options.qr_rtol)
        .with_factorize_alg(options.factorize_alg);

    // Create sweep plan
    let plan = LocalUpdateSweepPlan::from_treetn(&tn_c, center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create sweep plan"))?;

    // Perform sweeps
    for _sweep in 0..options.nfullsweeps {
        with_fit_profile(|profile| {
            profile.sweep_count += 1;
        });
        let log_norm_before = if options.convergence_tol.is_some() {
            Some(tn_c.log_norm()?)
        } else {
            None
        };

        let sweep_started = profile_enabled.then(Instant::now);
        apply_local_update_sweep(&mut tn_c, &plan, &mut updater)?;
        if let Some(sweep_started) = sweep_started {
            with_fit_profile(|profile| {
                profile.sweep_time += sweep_started.elapsed();
            });
        }

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

    if let Some(profile) = take_fit_profile() {
        eprintln!("=== contract_fit Profiling ===");
        eprintln!("zipup init:        {:?}", profile.zipup_init_time);
        eprintln!("canonicalize:      {:?}", profile.canonicalize_time);
        eprintln!("sweeps total:      {:?}", profile.sweep_time);
        eprintln!("steps:             {}", profile.step_count);
        eprintln!("sweeps:            {}", profile.sweep_count);
        eprintln!(
            "env get:           {:?} (requests={}, hits={}, misses={})",
            profile.env_get_time, profile.env_requests, profile.env_hits, profile.env_misses
        );
        eprintln!("env leaf compute:  {:?}", profile.env_leaf_time);
        eprintln!("env node compute:  {:?}", profile.env_internal_time);
        eprintln!("2-site contract:   {:?}", profile.two_site_contract_time);
        eprintln!("left_inds:         {:?}", profile.left_inds_time);
        eprintln!("factorize:         {:?}", profile.factorize_time);
        eprintln!("replace/update:    {:?}", profile.replace_time);
        eprintln!("invalidate:        {:?}", profile.invalidate_time);
    }
    // These are tensor4all-owned profiling hooks, intentionally kept during the
    // migration so fit profiling still works without the removed tenferro
    // runtime APIs.
    print_and_reset_contract_profile();
    print_and_reset_native_einsum_profile();

    Ok(tn_c)
}

#[cfg(test)]
mod tests;
