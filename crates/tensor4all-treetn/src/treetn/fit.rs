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

use crate::error::TreeTNOperationError;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;

use tensor4all_core::{
    print_and_reset_contract_profile, print_and_reset_native_einsum_profile,
    reset_contract_profile, reset_native_einsum_profile, sort_indices_deterministic, AnyScalar,
    Canonical, DynIndex, FactorizeAlg, FactorizeOptions, IdxTensor, IndexLike, SvdTruncationPolicy,
    TensorLike,
};

use num_complex::Complex64;

use super::contraction::{GlobalLeaves, LeafOps, SessionLeaves};
use super::localupdate::{LocalUpdateStep, LocalUpdateSweepPlan, LocalUpdater};
use super::TreeTN;

#[cfg(test)]
use std::cell::Cell;

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

#[cfg(test)]
thread_local! {
    static FORCE_FIT_PROFILE: Cell<bool> = const { Cell::new(false) };
}

fn fit_profile_enabled() -> bool {
    #[cfg(test)]
    if FORCE_FIT_PROFILE.with(Cell::get) {
        return true;
    }
    std::env::var("T4A_PROFILE_FIT").is_ok()
}

#[cfg(test)]
fn set_fit_profile_enabled_for_tests(enabled: bool) {
    FORCE_FIT_PROFILE.with(|slot| slot.set(enabled));
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

fn sorted_neighbors_by_node_index<T, V>(
    tn: &TreeTN<T, V>,
    node: &V,
    excluded: Option<&V>,
    context: &str,
) -> Result<Vec<V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let mut neighbors = Vec::new();
    for neighbor in tn.site_index_network().neighbors(node) {
        if excluded == Some(&neighbor) {
            continue;
        }
        let node_idx = tn.node_index(&neighbor).ok_or_else(|| {
            anyhow::anyhow!(
                "{context}: neighbor {:?} of node {:?} is missing from TreeTN node map",
                neighbor,
                node
            )
        })?;
        neighbors.push((node_idx.index(), neighbor));
    }
    neighbors.sort_by_key(|(index, _node)| *index);
    Ok(neighbors.into_iter().map(|(_index, node)| node).collect())
}

/// Like [`sorted_neighbors_by_node_index`], but caches the sorted neighbor list
/// of `node` in `cache` on first use.
///
/// The order depends only on the network topology, which is fixed for a whole
/// fit run, so the per-node sort and `node_index` lookups happen once per
/// (node, topology) instead of on every environment access.
fn cached_sorted_neighbors<T, V>(
    cache: &mut HashMap<V, Vec<V>>,
    tn: &TreeTN<T, V>,
    node: &V,
    context: &str,
) -> Result<Vec<V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    if let Some(order) = cache.get(node) {
        return Ok(order.clone());
    }
    let order = sorted_neighbors_by_node_index(tn, node, None, context)?;
    cache.insert(node.clone(), order.clone());
    Ok(order)
}

fn tensor_at_node<'a, T, V>(tn: &'a TreeTN<T, V>, node: &V, tree_name: &str) -> Result<&'a T>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let node_idx = tn
        .node_index(node)
        .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in {}", node, tree_name))?;
    tn.tensor(node_idx)
        .ok_or_else(|| anyhow::anyhow!("Tensor for node {:?} not found in {}", node, tree_name))
}

fn tensors_share_contractable_index<T>(left: &T, right: &T) -> bool
where
    T: TensorLike,
{
    let left_indices = left.external_indices();
    let right_indices = right.external_indices();
    left_indices.iter().any(|left_index| {
        right_indices
            .iter()
            .any(|right_index| left_index.is_contractable(right_index))
    })
}

fn tensor_connected_components<T>(tensors: &[&T]) -> Vec<Vec<usize>>
where
    T: TensorLike,
{
    let mut visited = vec![false; tensors.len()];
    let mut components = Vec::new();

    for start in 0..tensors.len() {
        if visited[start] {
            continue;
        }

        let mut component = Vec::new();
        let mut stack = vec![start];
        visited[start] = true;

        while let Some(current) = stack.pop() {
            component.push(current);
            for candidate in 0..tensors.len() {
                if visited[candidate] {
                    continue;
                }
                if tensors_share_contractable_index(tensors[current], tensors[candidate]) {
                    visited[candidate] = true;
                    stack.push(candidate);
                }
            }
        }

        component.sort_unstable();
        components.push(component);
    }

    components
}

fn contract_fit_tensor_refs<T>(tensors: &[&T], leaves: &dyn LeafOps<T>) -> Result<T>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    if tensors.is_empty() {
        return Err(anyhow::anyhow!(
            "fit contraction requires at least one tensor"
        ));
    }

    let components = tensor_connected_components(tensors);
    if components.len() == 1 {
        return leaves.contract_refs(tensors);
    }

    let mut contracted_components = Vec::with_capacity(components.len());
    for component in components {
        let component_refs = component
            .iter()
            .map(|&tensor_index| tensors[tensor_index])
            .collect::<Vec<_>>();
        let contracted = if component_refs.len() == 1 {
            component_refs[0].clone()
        } else {
            leaves
                .contract_refs(&component_refs)
                .map_err(|e| anyhow::anyhow!("component contract failed: {}", e))?
        };
        contracted_components.push(contracted);
    }

    let mut iter = contracted_components.into_iter();
    let mut result = iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("fit contraction produced no components"))?;
    for component in iter {
        result = leaves
            .outer_product(&result, &component)
            .map_err(|e| anyhow::anyhow!("component outer product failed: {}", e))?;
    }

    Ok(result)
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
    /// Environment tensors: from -> to -> tensor.
    envs: HashMap<V, HashMap<V, T>>,
    /// Sorted neighbor lists (by `TreeTN` node index), cached per node.
    neighbor_order: HashMap<V, Vec<V>>,
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
            neighbor_order: HashMap::new(),
        }
    }

    /// Get the environment tensor for edge (from, to) if it exists.
    pub fn get(&self, from: &V, to: &V) -> Option<&T> {
        self.envs.get(from).and_then(|to_envs| to_envs.get(to))
    }

    /// Insert an environment tensor for edge (from, to).
    /// This is mainly for testing; normally use `get_or_compute` for lazy evaluation.
    #[allow(dead_code)]
    pub(crate) fn insert(&mut self, from: V, to: V, env: T) {
        self.envs.entry(from).or_default().insert(to, env);
    }

    /// Check if environment exists for edge (from, to).
    pub fn contains(&self, from: &V, to: &V) -> bool {
        self.get(from, to).is_some()
    }

    /// Get the number of cached environments.
    pub fn len(&self) -> usize {
        self.envs.values().map(HashMap::len).sum()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    /// Clear all cached environments.
    pub fn clear(&mut self) {
        self.envs.clear();
        self.neighbor_order.clear();
    }

    /// Sorted neighbors of `node`, computing and caching the order on first use.
    fn sorted_neighbors(&mut self, tn: &TreeTN<T, V>, node: &V, context: &str) -> Result<Vec<V>> {
        cached_sorted_neighbors(&mut self.neighbor_order, tn, node, context)
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
    /// # Errors
    ///
    /// Returns an error when `from` is not a node of `tn_c`, or when a node
    /// environment cannot be computed because a contraction fails or an index
    /// does not match.
    ///
    pub fn get_or_compute(
        &mut self,
        from: &V,
        to: &V,
        tn_a: &TreeTN<T, V>,
        tn_b: &TreeTN<T, V>,
        tn_c: &TreeTN<T, V>,
        leaves: &dyn LeafOps<T>,
    ) -> std::result::Result<T, TreeTNOperationError>
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
        if let Some(env) = self.get(from, to) {
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
        let mut child_neighbors =
            self.sorted_neighbors(tn_c, from, "FitEnvironment::get_or_compute")?;
        child_neighbors.retain(|neighbor| neighbor != to);

        // Recursively get or compute child environments
        let child_envs: Vec<T> = child_neighbors
            .iter()
            .map(|child| self.get_or_compute(child, from, tn_a, tn_b, tn_c, leaves))
            .collect::<std::result::Result<Vec<_>, TreeTNOperationError>>()?;

        // Compute the environment for (from, to) using child environments
        let env = compute_single_node_environment(from, to, tn_a, tn_b, tn_c, &child_envs, leaves)?;

        // Cache and return
        self.insert(from.clone(), to.clone(), env.clone());
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
        let _ = self.try_invalidate(region, tn_c);
    }

    fn try_invalidate<'a>(
        &mut self,
        region: impl IntoIterator<Item = &'a V>,
        tn_c: &TreeTN<T, V>,
    ) -> Result<()>
    where
        V: 'a + Send + Sync,
    {
        for t in region {
            // Get all neighbors of t
            let neighbors = self.sorted_neighbors(tn_c, t, "FitEnvironment::invalidate")?;

            // Remove all env[(t, *)] and propagate recursively
            for neighbor in neighbors {
                self.invalidate_recursive(t, &neighbor, tn_c)?;
            }
        }
        Ok(())
    }

    /// Recursively invalidate caches starting from env[(from, to)] towards leaves.
    ///
    /// If env[(from, to)] exists, remove it and propagate to env[(to, x)] for all x ≠ from.
    fn invalidate_recursive(&mut self, from: &V, to: &V, tn_c: &TreeTN<T, V>) -> Result<()> {
        // Remove env[(from, to)] if it exists
        let removed = self
            .envs
            .get_mut(from)
            .and_then(|to_envs| to_envs.remove(to))
            .is_some();
        if self.envs.get(from).is_some_and(HashMap::is_empty) {
            self.envs.remove(from);
        }
        if removed {
            // Propagate to next generation: env[(to, x)] for all neighbors x of to, x ≠ from
            let mut neighbors =
                self.sorted_neighbors(tn_c, to, "FitEnvironment::invalidate_recursive")?;
            neighbors.retain(|neighbor| neighbor != from);

            for neighbor in neighbors {
                self.invalidate_recursive(to, &neighbor, tn_c)?;
            }
        }
        Ok(())
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
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when a cached environment's node is not in `tn_c`, or when
    /// a required child environment is missing.
    ///
    pub fn verify_structural_consistency(
        &self,
        tn_c: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError>
    where
        V: Clone + Hash + Eq + std::fmt::Debug,
    {
        for (from, to_envs) in self.envs.iter() {
            for to in to_envs.keys() {
                // Get neighbors of `from` excluding `to`
                let child_neighbors = sorted_neighbors_by_node_index(
                    tn_c,
                    from,
                    Some(to),
                    "FitEnvironment::verify_structural_consistency",
                )?;

                // If `from` is not a leaf, all child environments must exist
                for child in &child_neighbors {
                    if self.get(child, from).is_none() {
                        return Err(anyhow::anyhow!(
                            "Structural inconsistency: env[({:?}, {:?})] exists but child env[({:?}, {:?})] is missing",
                            from, to, child, from
                        ).into());
                    }
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

/// Environment cache for one sum target and the current variational state.
#[derive(Debug, Clone)]
struct SumFitEnvironment<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq,
{
    envs: HashMap<V, HashMap<V, T>>,
    neighbor_order: HashMap<V, Vec<V>>,
}

impl<T, V> SumFitEnvironment<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    fn new() -> Self {
        Self {
            envs: HashMap::new(),
            neighbor_order: HashMap::new(),
        }
    }

    /// Sorted neighbors of `node`, computing and caching the order on first use.
    fn sorted_neighbors(&mut self, tn: &TreeTN<T, V>, node: &V, context: &str) -> Result<Vec<V>> {
        cached_sorted_neighbors(&mut self.neighbor_order, tn, node, context)
    }

    fn get_or_compute(
        &mut self,
        from: &V,
        to: &V,
        target: &TreeTN<T, V>,
        psi: &TreeTN<T, V>,
        leaves: &dyn LeafOps<T>,
    ) -> std::result::Result<T, TreeTNOperationError> {
        if let Some(env) = self.envs.get(from).and_then(|to_envs| to_envs.get(to)) {
            return Ok(env.clone());
        }

        let mut child_neighbors =
            self.sorted_neighbors(psi, from, "SumFitEnvironment::get_or_compute")?;
        child_neighbors.retain(|neighbor| neighbor != to);
        let child_envs = child_neighbors
            .iter()
            .map(|child| self.get_or_compute(child, from, target, psi, leaves))
            .collect::<std::result::Result<Vec<_>, TreeTNOperationError>>()?;

        let target_tensor = tensor_at_node(target, from, "sum target")?;
        let psi_conj = tensor_at_node(psi, from, "fit state")?.conj();
        let mut tensor_refs = vec![target_tensor, &psi_conj];
        tensor_refs.extend(child_envs.iter());
        let env = contract_fit_tensor_refs(&tensor_refs, leaves)
            .map_err(|e| anyhow::anyhow!("sum target environment contraction failed: {e}"))?;
        self.envs
            .entry(from.clone())
            .or_default()
            .insert(to.clone(), env.clone());
        Ok(env)
    }

    fn prepare(
        &mut self,
        target: &TreeTN<T, V>,
        psi: &TreeTN<T, V>,
        leaves: &dyn LeafOps<T>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        for from in psi.node_names() {
            let neighbors: Vec<_> = psi.site_index_network().neighbors(&from).collect();
            for to in neighbors {
                self.get_or_compute(&from, &to, target, psi, leaves)?;
            }
        }
        Ok(())
    }

    fn invalidate(
        &mut self,
        region: &[V],
        psi: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        for node in region {
            let neighbors = self.sorted_neighbors(psi, node, "SumFitEnvironment::invalidate")?;
            for neighbor in neighbors {
                self.invalidate_recursive(node, &neighbor, psi)?;
            }
        }
        Ok(())
    }

    fn invalidate_recursive(
        &mut self,
        from: &V,
        to: &V,
        psi: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        let removed = self
            .envs
            .get_mut(from)
            .and_then(|to_envs| to_envs.remove(to))
            .is_some();
        if self.envs.get(from).is_some_and(HashMap::is_empty) {
            self.envs.remove(from);
        }
        if removed {
            let mut neighbors =
                self.sorted_neighbors(psi, to, "SumFitEnvironment::invalidate_recursive")?;
            neighbors.retain(|neighbor| neighbor != from);
            for neighbor in neighbors {
                self.invalidate_recursive(to, &neighbor, psi)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct FitFactorizeConfig {
    max_bond_dim: Option<usize>,
    svd_policy: Option<SvdTruncationPolicy>,
    qr_rtol: Option<f64>,
    factorize_alg: FactorizeAlg,
}

fn fit_factorize_options(config: FitFactorizeConfig) -> Result<FactorizeOptions> {
    let mut options = match config.factorize_alg {
        FactorizeAlg::SVD => FactorizeOptions::svd(),
        FactorizeAlg::QR => FactorizeOptions::qr(),
        FactorizeAlg::LU => FactorizeOptions::lu(),
        FactorizeAlg::CI => FactorizeOptions::ci(),
    }
    .with_canonical(Canonical::Left);
    if let Some(max_bond_dim) = config.max_bond_dim {
        options = options.with_max_bond_dim(max_bond_dim);
    }
    if let Some(policy) = config.svd_policy {
        options = options.with_svd_policy(policy);
    }
    if let Some(qr_rtol) = config.qr_rtol {
        options = options.with_qr_rtol(qr_rtol);
    }
    options
        .validate()
        .map_err(|e| anyhow::anyhow!("invalid fit factorization options: {e}"))?;
    Ok(options)
}

fn fit_two_site_update<T, V>(
    mut subtree: TreeTN<T, V>,
    step: &LocalUpdateStep<V>,
    full_treetn: &TreeTN<T, V>,
    local_optimum: T,
    left_bond_indices: &[T::Index],
    config: FitFactorizeConfig,
    leaves: &dyn LeafOps<T>,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let node_u = step
        .nodes
        .first()
        .ok_or_else(|| anyhow::anyhow!("fit two-site update requires a first node"))?;
    let node_v = step
        .nodes
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("fit two-site update requires a second node"))?;
    let site_c_u = full_treetn.site_space(node_u).cloned().ok_or_else(|| {
        anyhow::anyhow!(
            "fit two-site update: site space for node {:?} not found in full TreeTN",
            node_u
        )
    })?;

    let local_indices = local_optimum.external_indices();
    let left_inds_started = fit_profile_enabled().then(Instant::now);
    let mut left_inds: Vec<_> = local_indices
        .iter()
        .filter(|idx| site_c_u.contains(*idx) || left_bond_indices.iter().any(|bond| bond == *idx))
        .cloned()
        .collect();
    sort_indices_deterministic(&mut left_inds);
    if let Some(left_inds_started) = left_inds_started {
        with_fit_profile(|profile| {
            profile.left_inds_time += left_inds_started.elapsed();
        });
    }

    let mut options = fit_factorize_options(config)?;
    let bond_cap = if config.max_bond_dim.is_some() {
        config.max_bond_dim
    } else if config.svd_policy.is_some() || config.qr_rtol.is_some() {
        None
    } else {
        subtree
            .edge_between(node_u, node_v)
            .and_then(|edge| subtree.bond_index(edge))
            .map(|bond| bond.dim())
    };
    if let Some(bond_cap) = bond_cap {
        options = options.with_max_bond_dim(bond_cap);
    }

    let factorize_started = fit_profile_enabled().then(Instant::now);
    // A site-free endpoint leaves one side of the split empty; the shared
    // helper keeps the requested left-canonical form in that case, so the
    // norm ends up on `node_v` together with the canonical center.
    let factorize_result =
        super::factorize_allowing_empty_side(&local_optimum, &left_inds, |tensor, left| {
            leaves
                .factorize(tensor, left, &options)
                .map_err(|e| anyhow::anyhow!("factorization failed: {e}"))
        })?;
    if let Some(factorize_started) = factorize_started {
        with_fit_profile(|profile| {
            profile.factorize_time += factorize_started.elapsed();
        });
    }

    let edge_uv = subtree.edge_between(node_u, node_v).ok_or_else(|| {
        anyhow::anyhow!(
            "fit two-site update: subtree is missing edge between {:?} and {:?}",
            node_u,
            node_v
        )
    })?;
    let idx_u_sub = subtree
        .node_index(node_u)
        .ok_or_else(|| anyhow::anyhow!("fit two-site update: node {:?} not found", node_u))?;
    let idx_v_sub = subtree
        .node_index(node_v)
        .ok_or_else(|| anyhow::anyhow!("fit two-site update: node {:?} not found", node_v))?;

    let replace_started = fit_profile_enabled().then(Instant::now);
    subtree.replace_edge_bond(edge_uv, factorize_result.bond_index.clone())?;
    subtree
        .replace_tensor(idx_u_sub, factorize_result.left)?
        .ok_or_else(|| anyhow::anyhow!("fit two-site update: first node disappeared"))?;
    subtree
        .replace_tensor(idx_v_sub, factorize_result.right)?
        .ok_or_else(|| anyhow::anyhow!("fit two-site update: second node disappeared"))?;
    subtree.set_ortho_towards(&factorize_result.bond_index, Some(step.new_center.clone()));
    subtree.set_canonical_region([step.new_center.clone()])?;
    if let Some(replace_started) = replace_started {
        with_fit_profile(|profile| {
            profile.replace_time += replace_started.elapsed();
        });
    }
    Ok(subtree)
}

/// Compute environment for a leaf node (no children in subtree).
fn compute_leaf_environment<T, V>(
    node: &V,
    _towards: &V,
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    tn_c: &TreeTN<T, V>,
    leaves: &dyn LeafOps<T>,
) -> Result<T>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let started = fit_profile_enabled().then(Instant::now);

    // Get tensors
    let tensor_a = tensor_at_node(tn_a, node, "tn_a")?;
    let tensor_b = tensor_at_node(tn_b, node, "tn_b")?;
    let tensor_c = tensor_at_node(tn_c, node, "tn_c")?;

    // A, B, and C must form one connected local environment.
    let c_conj = tensor_c.conj();
    let env = contract_fit_tensor_refs(&[tensor_a, tensor_b, &c_conj], leaves)?;

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
    leaves: &dyn LeafOps<T>,
) -> Result<T>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let started = fit_profile_enabled().then(Instant::now);

    // Get local tensors
    let tensor_a = tensor_at_node(tn_a, node, "tn_a")?;
    let tensor_b = tensor_at_node(tn_b, node, "tn_b")?;
    let tensor_c = tensor_at_node(tn_c, node, "tn_c")?;

    if child_envs.is_empty() {
        // Leaf node: compute from tensors directly
        return compute_leaf_environment(node, towards, tn_a, tn_b, tn_c, leaves);
    }

    // Non-leaf: all local tensors and child environments must form one
    // connected contraction graph.
    let c_conj = tensor_c.conj();
    let mut tensor_refs: Vec<&T> = vec![tensor_a, tensor_b, &c_conj];
    tensor_refs.extend(child_envs.iter());
    let result = contract_fit_tensor_refs(&tensor_refs, leaves)?;

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
    pub max_bond_dim: Option<usize>,
    /// Legacy relative tolerance retained for same-crate tests and call chains.
    pub(crate) rtol: Option<f64>,
    /// Explicit SVD truncation policy
    pub svd_policy: Option<SvdTruncationPolicy>,
    /// QR-specific relative tolerance
    pub qr_rtol: Option<f64>,
    /// Factorization algorithm
    pub factorize_alg: FactorizeAlg,
    /// Leaf execution strategy for this updater's numeric leaves
    /// (`GlobalLeaves` by default; the caller-owned context route installs
    /// `SessionLeaves` through [`Self::with_leaves`]).
    pub(crate) leaves: Box<dyn LeafOps<T>>,
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
    /// * `max_bond_dim` - Maximum bond dimension for truncation
    /// * `svd_policy` - Explicit SVD truncation policy
    /// * `qr_rtol` - QR-specific relative tolerance
    ///
    /// Note: sim_internal_inds() should be called on tn_a and tn_b before passing
    /// if index collision is a concern. This is not done here because contraction
    /// module (which provides sim_internal_inds) is currently disabled.
    /// Install a leaf execution strategy (issue #859 F7).
    ///
    /// Used by the caller-owned context entry (`contract_fit_in`) to route the
    /// updater's contractions and factorizations through a context-scoped
    /// strategy instead of the process-global route.
    pub(crate) fn with_leaves(mut self, leaves: Box<dyn LeafOps<T>>) -> Self {
        self.leaves = leaves;
        self
    }

    pub fn new(
        tn_a: TreeTN<T, V>,
        tn_b: TreeTN<T, V>,
        max_bond_dim: Option<usize>,
        rtol: Option<f64>,
    ) -> Self {
        Self {
            tn_a,
            tn_b,
            envs: FitEnvironment::new(),
            max_bond_dim,
            rtol,
            svd_policy: rtol.map(SvdTruncationPolicy::new),
            qr_rtol: None,
            factorize_alg: FactorizeAlg::SVD,
            leaves: Box::new(GlobalLeaves),
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
        subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<T, V>,
    ) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError> {
        if step.nodes.len() != 2 {
            return Err(anyhow::anyhow!(
                "FitUpdater requires exactly 2 nodes, got {}",
                step.nodes.len()
            )
            .into());
        }

        let node_u = &step.nodes[0];
        let node_v = &step.nodes[1];
        with_fit_profile(|profile| {
            profile.step_count += 1;
        });

        if full_treetn.node_index(node_u).is_none() {
            return Err(anyhow::anyhow!("Node {:?} not found in full TreeTN", node_u).into());
        }
        if full_treetn.node_index(node_v).is_none() {
            return Err(anyhow::anyhow!("Node {:?} not found in full TreeTN", node_v).into());
        }
        if full_treetn.edge_between(node_u, node_v).is_none() {
            return Err(anyhow::anyhow!(
                "FitUpdater update step nodes {:?} and {:?} are not adjacent in full TreeTN",
                node_u,
                node_v
            )
            .into());
        }

        let a_u = tensor_at_node(&self.tn_a, node_u, "tn_a")?;
        let a_v = tensor_at_node(&self.tn_a, node_v, "tn_a")?;
        let b_u = tensor_at_node(&self.tn_b, node_u, "tn_b")?;
        let b_v = tensor_at_node(&self.tn_b, node_v, "tn_b")?;

        let mut env_tensors = Vec::new();
        let u_neighbors = self
            .envs
            .sorted_neighbors(full_treetn, node_u, "FitUpdater::update")?;
        let mut left_bond_indices = Vec::new();
        for neighbor in &u_neighbors {
            if neighbor == node_v {
                continue;
            }
            let edge = full_treetn.edge_between(node_u, neighbor).ok_or_else(|| {
                anyhow::anyhow!(
                    "FitUpdater: missing edge between {:?} and {:?} in full TreeTN",
                    node_u,
                    neighbor
                )
            })?;
            let bond = full_treetn.bond_index(edge).ok_or_else(|| {
                anyhow::anyhow!(
                    "FitUpdater: missing bond index for edge between {:?} and {:?}",
                    node_u,
                    neighbor
                )
            })?;
            left_bond_indices.push(bond.clone());
            env_tensors.push(self.envs.get_or_compute(
                neighbor,
                node_u,
                &self.tn_a,
                &self.tn_b,
                full_treetn,
                self.leaves.as_ref(),
            )?);
        }

        let v_neighbors = self
            .envs
            .sorted_neighbors(full_treetn, node_v, "FitUpdater::update")?;
        for neighbor in &v_neighbors {
            if neighbor != node_u {
                env_tensors.push(self.envs.get_or_compute(
                    neighbor,
                    node_v,
                    &self.tn_a,
                    &self.tn_b,
                    full_treetn,
                    self.leaves.as_ref(),
                )?);
            }
        }

        let contract_started = fit_profile_enabled().then(Instant::now);
        let mut tensor_refs = vec![a_u, b_u, a_v, b_v];
        tensor_refs.extend(env_tensors.iter());
        let local_optimum = contract_fit_tensor_refs(&tensor_refs, self.leaves.as_ref())?;
        if let Some(contract_started) = contract_started {
            with_fit_profile(|profile| {
                profile.two_site_contract_time += contract_started.elapsed();
            });
        }

        fit_two_site_update(
            subtree,
            step,
            full_treetn,
            local_optimum,
            &left_bond_indices,
            FitFactorizeConfig {
                max_bond_dim: self.max_bond_dim,
                svd_policy: self.svd_policy,
                qr_rtol: self.qr_rtol,
                factorize_alg: self.factorize_alg,
            },
            self.leaves.as_ref(),
        )
    }

    fn after_step(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_after: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        // Invalidate all caches affected by the updated region
        let started = fit_profile_enabled().then(Instant::now);
        self.envs.try_invalidate(&step.nodes, full_treetn_after)?;
        if let Some(started) = started {
            with_fit_profile(|profile| {
                profile.invalidate_time += started.elapsed();
            });
        }
        Ok(())
    }
}

// ============================================================================
// Shared Euler-tour sweep loop
// ============================================================================
fn run_fit_sweeps<T, V, U>(
    treetn: &mut TreeTN<T, V>,
    plan: &LocalUpdateSweepPlan<V>,
    updater: &mut U,
    nfullsweeps: usize,
    convergence_tol: Option<f64>,
) -> Result<()>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    U: LocalUpdater<T, V>,
{
    use super::localupdate::apply_local_update_sweep;

    for _sweep in 0..nfullsweeps {
        with_fit_profile(|profile| {
            profile.sweep_count += 1;
        });
        let log_norm_before = convergence_tol.map(|_| treetn.log_norm()).transpose()?;
        let sweep_started = fit_profile_enabled().then(Instant::now);
        apply_local_update_sweep(treetn, plan, updater)?;
        if let Some(sweep_started) = sweep_started {
            with_fit_profile(|profile| {
                profile.sweep_time += sweep_started.elapsed();
            });
        }

        if let (Some(log_norm_before), Some(tol)) = (log_norm_before, convergence_tol) {
            let log_norm_after = treetn.log_norm()?;
            let relative_change = (f64::exp(log_norm_after - log_norm_before) - 1.0).abs();
            if relative_change < tol {
                break;
            }
        }
    }
    Ok(())
}

fn finish_fit_profile(entry_point: &str) {
    if let Some(profile) = take_fit_profile() {
        eprintln!("=== {entry_point} Profiling ===");
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
    print_and_reset_contract_profile();
    print_and_reset_native_einsum_profile();
}

#[derive(Debug)]
struct SumFitUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    targets: Vec<TreeTN<T, V>>,
    envs: Vec<SumFitEnvironment<T, V>>,
    factorize_config: FitFactorizeConfig,
}

impl<T, V> SumFitUpdater<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    fn new(
        targets: Vec<TreeTN<T, V>>,
        max_bond_dim: Option<usize>,
        svd_policy: Option<SvdTruncationPolicy>,
        qr_rtol: Option<f64>,
        factorize_alg: FactorizeAlg,
    ) -> Self {
        let envs = targets.iter().map(|_| SumFitEnvironment::new()).collect();
        Self {
            targets,
            envs,
            factorize_config: FitFactorizeConfig {
                max_bond_dim,
                svd_policy,
                qr_rtol,
                factorize_alg,
            },
        }
    }

    fn prepare(&mut self, psi: &TreeTN<T, V>) -> std::result::Result<(), TreeTNOperationError> {
        for (target_index, (target, env)) in
            self.targets.iter().zip(self.envs.iter_mut()).enumerate()
        {
            // `SumFitUpdater` has no leaf strategy of its own: `fit_sum`'s
            // public entry has no caller-owned context route, so this driver
            // always runs on the process-global route (unlike `FitUpdater`,
            // which carries `self.leaves`).
            env.prepare(target, psi, &GlobalLeaves).map_err(|error| {
                anyhow::anyhow!(
                    "fit_sum: target {target_index} initial environment construction failed: {error}"
                )
            })?;
        }
        Ok(())
    }
}

impl<T, V> LocalUpdater<T, V> for SumFitUpdater<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    fn update(
        &mut self,
        subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<T, V>,
    ) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError> {
        if step.nodes.len() != 2 {
            return Err(anyhow::anyhow!(
                "fit_sum updater requires exactly 2 nodes, got {}",
                step.nodes.len()
            )
            .into());
        }
        let node_u = &step.nodes[0];
        let node_v = &step.nodes[1];
        if full_treetn.node_index(node_u).is_none() || full_treetn.node_index(node_v).is_none() {
            return Err(anyhow::anyhow!("fit_sum updater step references a missing node").into());
        }
        if full_treetn.edge_between(node_u, node_v).is_none() {
            return Err(anyhow::anyhow!("fit_sum updater step nodes are not adjacent").into());
        }
        with_fit_profile(|profile| {
            profile.step_count += 1;
        });

        let (u_neighbors, v_neighbors) = match self.envs.first_mut() {
            Some(env) => (
                env.sorted_neighbors(full_treetn, node_u, "fit_sum updater")?,
                env.sorted_neighbors(full_treetn, node_v, "fit_sum updater")?,
            ),
            None => (Vec::new(), Vec::new()),
        };
        let mut left_bond_indices = Vec::new();
        for neighbor in &u_neighbors {
            if neighbor == node_v {
                continue;
            }
            let edge = full_treetn.edge_between(node_u, neighbor).ok_or_else(|| {
                anyhow::anyhow!("fit_sum updater: missing edge to a neighboring node")
            })?;
            let bond = full_treetn
                .bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("fit_sum updater: missing neighboring bond"))?;
            left_bond_indices.push(bond.clone());
        }

        let mut local_sum: Option<T> = None;
        for (target_index, (target, env)) in
            self.targets.iter().zip(self.envs.iter_mut()).enumerate()
        {
            let target_u = tensor_at_node(target, node_u, "sum target")
                .map_err(|error| anyhow::anyhow!("fit_sum: target {target_index}: {error}"))?;
            let target_v = tensor_at_node(target, node_v, "sum target")
                .map_err(|error| anyhow::anyhow!("fit_sum: target {target_index}: {error}"))?;
            let mut env_tensors = Vec::new();
            for neighbor in &u_neighbors {
                if neighbor != node_v {
                    env_tensors.push(
                        env.get_or_compute(neighbor, node_u, target, full_treetn, &GlobalLeaves)
                            .map_err(|error| {
                                anyhow::anyhow!(
                                    "fit_sum: target {target_index} environment contraction failed: {error}"
                                )
                            })?,
                    );
                }
            }
            for neighbor in &v_neighbors {
                if neighbor != node_u {
                    env_tensors.push(
                        env.get_or_compute(neighbor, node_v, target, full_treetn, &GlobalLeaves)
                            .map_err(|error| {
                                anyhow::anyhow!(
                                    "fit_sum: target {target_index} environment contraction failed: {error}"
                                )
                            })?,
                    );
                }
            }

            let contract_started = fit_profile_enabled().then(Instant::now);
            let mut tensor_refs = vec![target_u, target_v];
            tensor_refs.extend(env_tensors.iter());
            let contribution =
                contract_fit_tensor_refs(&tensor_refs, &GlobalLeaves).map_err(|error| {
                    anyhow::anyhow!(
                    "fit_sum: target {target_index} local contribution contraction failed: {error}"
                )
                })?;
            if let Some(contract_started) = contract_started {
                with_fit_profile(|profile| {
                    profile.two_site_contract_time += contract_started.elapsed();
                });
            }

            local_sum = Some(match local_sum {
                None => contribution,
                Some(accumulated) => accumulated
                    .axpby(
                        AnyScalar::new_real(1.0),
                        &contribution,
                        AnyScalar::new_real(1.0),
                    )
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "fit_sum: target {target_index} tensor accumulation failed: {error}"
                        )
                    })?,
            });
        }
        let local_sum =
            local_sum.ok_or_else(|| anyhow::anyhow!("fit_sum: no targets to accumulate"))?;
        fit_two_site_update(
            subtree,
            step,
            full_treetn,
            local_sum,
            &left_bond_indices,
            self.factorize_config,
            &GlobalLeaves,
        )
        .map_err(|error| anyhow::anyhow!("fit_sum: two-site update failed: {error}"))
        .map_err(TreeTNOperationError::from)
    }

    fn after_step(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_after: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        let started = fit_profile_enabled().then(Instant::now);
        for (target_index, env) in self.envs.iter_mut().enumerate() {
            env.invalidate(&step.nodes, full_treetn_after)
                .map_err(|error| {
                    anyhow::anyhow!(
                        "fit_sum: target {target_index} environment invalidation failed: {error}"
                    )
                })?;
        }
        if let Some(started) = started {
            with_fit_profile(|profile| {
                profile.invalidate_time += started.elapsed();
            });
        }
        Ok(())
    }
}

// High-level API: contract_fit
// ============================================================================

/// Options for [`fit_sum`] and `contract_fit`.
///
/// The crate-root [`crate::FitOptions`] alias exposes this type for sum fitting.
/// A zero sweep count is valid and returns the validated initial state for
/// [`fit_sum`]. Invalid dimensions or tolerances are rejected before that
/// short-circuit.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::FitOptions;
///
/// let options = FitOptions::new(0).with_max_bond_dim(2);
/// assert_eq!(options.nfullsweeps, 0);
/// assert_eq!(options.max_bond_dim, Some(2));
/// ```
#[derive(Debug, Clone)]
pub struct FitContractionOptions {
    /// Number of full sweeps to perform.
    ///
    /// A full sweep visits each edge twice (forward and backward) using an Euler tour.
    /// `0` skips fitting after validation; positive values perform that many sweeps
    /// unless `convergence_tol` stops earlier.
    pub nfullsweeps: usize,
    /// Optional maximum output bond dimension. `None` preserves the existing
    /// bond-space cap when no truncation policy overrides it; `Some(0)` is invalid.
    pub max_bond_dim: Option<usize>,
    /// Legacy relative tolerance retained for same-crate tests and call chains.
    pub(crate) rtol: Option<f64>,
    /// Explicit SVD truncation policy. Its settings are validated before fitting,
    /// including when `nfullsweeps == 0`.
    pub svd_policy: Option<SvdTruncationPolicy>,
    /// QR-specific relative tolerance. It must be finite and non-negative and
    /// must be accepted by the selected factorization algorithm.
    pub qr_rtol: Option<f64>,
    /// Factorization algorithm used for local two-site updates.
    pub factorize_alg: FactorizeAlg,
    /// Tolerance for early termination based on relative change.
    /// If `None`, run exactly `nfullsweeps` sweeps. If `Some(tol)`, `tol` must
    /// be finite and non-negative; fitting stops when the relative change in
    /// the network norm is below `tol`.
    pub convergence_tol: Option<f64>,
}

impl Default for FitContractionOptions {
    fn default() -> Self {
        Self {
            nfullsweeps: 1,
            max_bond_dim: None,
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
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
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

fn validate_fit_sum_options(options: &FitContractionOptions) -> Result<()> {
    if let Some(tol) = options.convergence_tol {
        if !tol.is_finite() || tol < 0.0 {
            return Err(anyhow::anyhow!(
                "convergence tolerance must be finite and non-negative"
            ));
        }
    }
    if let Some(qr_rtol) = options.qr_rtol {
        if !qr_rtol.is_finite() || qr_rtol < 0.0 {
            return Err(anyhow::anyhow!(
                "QR relative tolerance must be finite and non-negative"
            ));
        }
    }
    fit_factorize_options(FitFactorizeConfig {
        max_bond_dim: options.max_bond_dim,
        svd_policy: options.svd_policy,
        qr_rtol: options.qr_rtol,
        factorize_alg: options.factorize_alg,
    })?;
    Ok(())
}

/// Fit a TreeTN to the sum of one or more target TreeTNs.
///
/// The `initial` network supplies the output topology, site-index identities,
/// and starting bond spaces. Each target is reindexed to that site space before
/// the variational sweeps; target bonds remain private to their term.
///
/// # Arguments
/// * `targets` - Non-empty target networks with the same named topology and
///   per-node site dimensions as `initial`.
/// * `initial` - The required initial approximation and output topology.
/// * `center` - The node at which canonicalization and the Euler-tour sweeps
///   start.
/// * `options` - Fit sweep, factorization, truncation, and convergence options.
///   `nfullsweeps == 0` returns an unchanged clone of `initial` after validation.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] when validation, site reindexing,
/// environment construction, tensor contraction, accumulation, or
/// factorization fails.
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::{fit_sum, FitOptions, TreeTN};
///
/// let site = DynIndex::new_dyn(2);
/// let make = |values| {
///     TreeTN::<IdxTensor, usize>::from_tensors(
///         vec![IdxTensor::from_dense(vec![site.clone()], values).unwrap()],
///         vec![0],
///     )
///     .unwrap()
/// };
/// let targets = [make(vec![1.0, 2.0]), make(vec![3.0, 4.0])];
/// let initial = make(vec![0.0, 0.0]);
/// let fitted = fit_sum(&targets, &initial, &0, FitOptions::new(1)).unwrap();
/// assert_eq!(
///     fitted
///         .to_dense()
///         .unwrap()
///         .to_vec::<f64>()
///         .unwrap(),
///     vec![4.0, 6.0]
/// );
/// ```
pub fn fit_sum<T, V>(
    targets: &[TreeTN<T, V>],
    initial: &TreeTN<T, V>,
    center: &V,
    options: FitContractionOptions,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let profile_enabled = fit_profile_enabled();
    if profile_enabled {
        fit_profile_reset();
    }
    reset_contract_profile();
    reset_native_einsum_profile();

    validate_fit_sum_options(&options).context("fit_sum: invalid options")?;
    if targets.is_empty() {
        return Err(anyhow::anyhow!("fit_sum: targets must not be empty").into());
    }
    if initial.node_count() == 0 {
        return Err(anyhow::anyhow!("fit_sum: initial TreeTN must not be empty").into());
    }
    initial
        .verify_internal_consistency()
        .context("fit_sum: initial TreeTN validation failed")?;
    if initial.node_index(center).is_none() {
        return Err(anyhow::anyhow!("fit_sum: center node is not in initial TreeTN").into());
    }

    let mut aligned_targets = Vec::with_capacity(targets.len());
    for (target_index, target) in targets.iter().enumerate() {
        if target.node_count() == 0 {
            return Err(
                anyhow::anyhow!("fit_sum: target {target_index} TreeTN must not be empty").into(),
            );
        }
        target
            .verify_internal_consistency()
            .with_context(|| format!("fit_sum: target {target_index} TreeTN validation failed"))?;
        if !target.same_topology(initial) {
            return Err(anyhow::anyhow!(
                "fit_sum: target {target_index} has an incompatible named topology"
            )
            .into());
        }
        let aligned = target.reindex_site_space_like(initial).with_context(|| {
            format!("fit_sum: target {target_index} site-space validation failed")
        })?;
        aligned_targets.push(aligned.sim_internal_inds());
    }

    if options.nfullsweeps == 0 {
        finish_fit_profile("fit_sum");
        return Ok(initial.clone());
    }

    if initial.node_count() == 1 {
        let node = initial.node_names().into_iter().next().ok_or_else(|| {
            anyhow::anyhow!("fit_sum: initial node disappeared during validation")
        })?;
        let mut local_sum: Option<T> = None;
        for (target_index, target) in aligned_targets.iter().enumerate() {
            let tensor = tensor_at_node(target, &node, "sum target")
                .with_context(|| format!("fit_sum: target {target_index} tensor lookup failed"))?;
            local_sum = Some(match local_sum {
                None => tensor.clone(),
                Some(accumulated) => accumulated
                    .axpby(AnyScalar::new_real(1.0), tensor, AnyScalar::new_real(1.0))
                    .with_context(|| {
                        format!("fit_sum: target {target_index} tensor accumulation failed")
                    })?,
            });
        }
        let local_sum = local_sum.ok_or_else(|| anyhow::anyhow!("fit_sum: no targets to sum"))?;
        let node_index = initial
            .node_index(&node)
            .ok_or_else(|| anyhow::anyhow!("fit_sum: initial node lookup failed"))?;
        let mut result = initial.clone();
        result
            .replace_tensor(node_index, local_sum)
            .context("fit_sum: failed to replace one-node result tensor")?
            .ok_or_else(|| anyhow::anyhow!("fit_sum: one-node result disappeared"))?;
        finish_fit_profile("fit_sum");
        return Ok(result);
    }

    let canonicalize_started = profile_enabled.then(Instant::now);
    let mut psi = initial.clone();
    psi.canonicalize_mut(
        std::iter::once(center.clone()),
        crate::options::CanonicalizationOptions::forced(),
    )
    .context("fit_sum: failed to canonicalize initial TreeTN")?;
    if let Some(canonicalize_started) = canonicalize_started {
        with_fit_profile(|profile| {
            profile.canonicalize_time += canonicalize_started.elapsed();
        });
    }

    let mut updater = SumFitUpdater::new(
        aligned_targets,
        options.max_bond_dim,
        options.svd_policy,
        options.qr_rtol,
        options.factorize_alg,
    );
    updater
        .prepare(&psi)
        .context("fit_sum: failed to build initial target environments")?;
    let plan = LocalUpdateSweepPlan::from_treetn(&psi, center, 2)
        .ok_or_else(|| anyhow::anyhow!("fit_sum: failed to create two-site sweep plan"))?;
    run_fit_sweeps(
        &mut psi,
        &plan,
        &mut updater,
        options.nfullsweeps,
        options.convergence_tol,
    )?;
    finish_fit_profile("fit_sum");
    Ok(psi)
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
/// # Errors
///
/// Returns [`TreeTNOperationError`] when the two inputs have different topologies, when the
/// initial zip-up contraction fails, when a canonical center cannot be set, or
/// when creating the sweep plan or running the variational sweeps fails.
///
pub fn contract_fit<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: FitContractionOptions,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    contract_fit_with_leaves(tn_a, tn_b, center, options, Box::new(GlobalLeaves))
}

/// Fit contraction whose numerical work runs in a caller-owned execution
/// context (issue #859 F7).
///
/// The same algorithm, arguments and validation as [`contract`](crate::contraction::contract); what
/// differs is the resource the initial zip-up, the environment contractions
/// and the per-sweep factorizations run in. They go through the
/// context-scoped core entries (`contract_pair_on_session`, `factorize_in`)
/// instead of the process-global route. Inputs must be host-only or belong to
/// the supplied context (`TreeTN::validate_context`); `LU`/`CI` factorization
/// algorithms and the `factorize_auto` Gram shortcut have no context-scoped
/// twin, so the context route factorizes directly with the same options and
/// agrees with the plain route to tight tolerance.
///
/// # Refusal contract
///
/// The caller-owned route refuses, with a typed factorization error, shapes
/// the plain route happens to factorize: in particular an operand site index
/// that is the *prime* of the other operand's index (same id, different prime
/// level) leaves an unpaired primed site whose context-scoped factorization is
/// refused rather than conflated with its unprimed twin. Refusing is the
/// explicit contract here; the divergence is recorded in
/// `docs/worklogs/2026-10-12-f7-treetn-fit-context.md`.
///
/// # Arguments
/// * `tn_a` - First input TreeTN
/// * `tn_b` - Second input TreeTN
/// * `center` - Center node name
/// * `options` - Fit contraction options
/// * `context` - The caller-owned execution context the numerics run in
///
/// # Returns
/// The contracted TreeTN.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] when the topology or options are invalid,
/// when an operand does not validate against `context`, when the
/// context-scoped factorization refuses an operand (see the refusal contract
/// above), or when the caller-owned context refuses a contraction or
/// factorization.
pub fn contract_fit_in<V>(
    tn_a: &TreeTN<tensor4all_core::IdxTensor, V>,
    tn_b: &TreeTN<tensor4all_core::IdxTensor, V>,
    center: &V,
    options: FitContractionOptions,
    context: &tensor4all_tensorbackend::ExecutionContext,
) -> std::result::Result<TreeTN<tensor4all_core::IdxTensor, V>, TreeTNOperationError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <tensor4all_core::DynIndex as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    tn_a.validate_context(context)?;
    tn_b.validate_context(context)?;
    contract_fit_with_leaves(
        tn_a,
        tn_b,
        center,
        options,
        Box::new(SessionLeaves(context.clone())),
    )
}

#[allow(clippy::too_many_arguments)]
fn contract_fit_with_leaves<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: FitContractionOptions,
    leaves: Box<dyn LeafOps<T>>,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    use crate::CanonicalForm;
    let profile_enabled = fit_profile_enabled();
    if profile_enabled {
        fit_profile_reset();
    }
    reset_contract_profile();
    reset_native_einsum_profile();

    // Validate topologies match
    if !tn_a.same_topology(tn_b) {
        return Err(
            anyhow::anyhow!("TreeTNs must have the same topology for fit contraction").into(),
        );
    }

    // Initialize C using the SVD-based zipup contraction while preserving
    // the input topology required by variational sweeps.
    let zipup_started = profile_enabled.then(Instant::now);
    let tn_c = tn_a.contract_zipup_preserving_topology_with_leaves(
        tn_b,
        center,
        CanonicalForm::Unitary,
        options.svd_policy,
        options.max_bond_dim,
        leaves.as_ref(),
    )?;
    if let Some(zipup_started) = zipup_started {
        with_fit_profile(|profile| {
            profile.zipup_init_time += zipup_started.elapsed();
        });
    }

    // The zip-up initializer already returns a network centered at `center`.
    run_fit_from_state_with_leaves(tn_a, tn_b, center, options, tn_c, leaves)
}

/// Contract two TreeTNs from a caller-supplied initial state, with the contraction leaves owned here.
///
/// The same sweep engine as [`contract_fit_with_leaves`]; what differs is that `C₀` is supplied rather than built
/// from the zip-up contraction. That is what lets a caller start from a small-rank network instead of the exact
/// product bond - see [`contract_fit_from_initial`] and [`low_rank_initializer_tree_tn`].
fn contract_fit_from_initial_with_leaves<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: FitContractionOptions,
    tn_c: TreeTN<T, V>,
    leaves: Box<dyn LeafOps<T>>,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    if fit_profile_enabled() {
        fit_profile_reset();
    }
    reset_contract_profile();
    reset_native_einsum_profile();

    if !tn_a.same_topology(tn_b) {
        return Err(
            anyhow::anyhow!("TreeTNs must have the same topology for fit contraction").into(),
        );
    }
    if !tn_a.same_topology(&tn_c) {
        return Err(anyhow::anyhow!(
            "fit initial TreeTN must have the same topology as the input TreeTNs"
        )
        .into());
    }

    run_fit_from_state_with_leaves(tn_a, tn_b, center, options, tn_c, leaves)
}

/// Run the variational sweeps on an already-initialized `C₀`.
///
/// One sweep engine for both entries: the initialization is the only difference between them, so the sweeps, the
/// zero-sweep shortcut, the convergence handling and the profile bookkeeping stay in one place.
fn run_fit_from_state_with_leaves<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: FitContractionOptions,
    mut tn_c: TreeTN<T, V>,
    leaves: Box<dyn LeafOps<T>>,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    // The sweep machinery requires the network to carry a single canonical center at the requested node.
    tn_c.set_canonical_region([center.clone()])?;

    // Zero sweeps means "use the initial state as-is". Positive sweep counts are honored even when no truncation
    // override is provided, so callers can explicitly exercise the variational update path.
    if options.nfullsweeps == 0 {
        finish_fit_profile("contract_fit");
        return Ok(tn_c);
    }

    // Create FitUpdater (environments are computed lazily)
    let mut updater = FitUpdater::new(tn_a.clone(), tn_b.clone(), options.max_bond_dim, None)
        .with_leaves(leaves)
        .with_svd_policy(options.svd_policy)
        .with_qr_rtol(options.qr_rtol)
        .with_factorize_alg(options.factorize_alg);

    // Create sweep plan
    let plan = LocalUpdateSweepPlan::from_treetn(&tn_c, center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create sweep plan"))?;

    run_fit_sweeps(
        &mut tn_c,
        &plan,
        &mut updater,
        options.nfullsweeps,
        options.convergence_tol,
    )?;
    finish_fit_profile("contract_fit");

    Ok(tn_c)
}

/// Contract two TreeTNs using the fit (variational) algorithm, starting from a caller-supplied `C₀`.
///
/// [`contract_fit`] builds its starting state with the SVD-based zip-up contraction, which materializes the full
/// product bond (up to `χ_A·χ_B` per edge) when no truncation policy is supplied. A caller that does not want
/// that can build a small-rank starting state with [`low_rank_initializer_tree_tn`] - or any other network with
/// the input topology - and pass it here.
///
/// # Arguments
/// * `tn_a` - First input TreeTN
/// * `tn_b` - Second input TreeTN
/// * `center` - Node to use as canonical center
/// * `options` - Fit algorithm options
/// * `tn_c` - The starting state; it must have the same topology as the inputs, and it is used as-is when
///   `options.nfullsweeps` is zero
///
/// # Returns
/// A new TreeTN representing the contracted result.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] when `tn_a`, `tn_b`, and `tn_c` do not share the same
/// topology, or when setting the canonical center or running the variational
/// sweeps fails.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::{contract_fit_from_initial, low_rank_initializer_tree_tn, FitOptions, TreeTN};
///
/// let s0 = DynIndex::new_dyn(2);
/// let s1 = DynIndex::new_dyn(2);
/// let a0 = DynIndex::new_dyn(2);
/// let a1 = DynIndex::new_dyn(2);
/// let b0 = DynIndex::new_dyn(2);
/// let b1 = DynIndex::new_dyn(2);
/// let ba = DynIndex::new_dyn(2);
/// let bb = DynIndex::new_dyn(2);
/// let ta = IdxTensor::from_dense(vec![s0.clone(), a0.clone(), ba.clone()], (1..=8).map(|v| v as f64 / 8.0).collect()).unwrap();
/// let ta2 = IdxTensor::from_dense(vec![ba, s1.clone(), a1], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5]).unwrap();
/// let tb = IdxTensor::from_dense(vec![s0, b0.clone(), bb.clone()], (1..=8).map(|v| (v as f64 - 1.0) / 9.0).collect()).unwrap();
/// let tb2 = IdxTensor::from_dense(vec![bb, s1, b1], vec![1.0, 1.0, 0.0, 0.0, 0.25, 0.5, 0.25, 0.25]).unwrap();
/// let tn_a = TreeTN::from_tensors(vec![ta, ta2], vec![0usize, 1]).unwrap();
/// let tn_b = TreeTN::from_tensors(vec![tb, tb2], vec![0usize, 1]).unwrap();
/// let center = 1usize;
/// let init = low_rank_initializer_tree_tn(&tn_a, &tn_b, 1, 7).unwrap();
/// let c = contract_fit_from_initial(&tn_a, &tn_b, &center, FitOptions::new(2), init).unwrap();
/// assert_eq!(c.node_count(), 2);
/// ```
pub fn contract_fit_from_initial<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: FitContractionOptions,
    tn_c: TreeTN<T, V>,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    contract_fit_from_initial_with_leaves(tn_a, tn_b, center, options, tn_c, Box::new(GlobalLeaves))
}

/// Build a deterministic small-rank starting state for [`contract_fit_from_initial`].
///
/// Every surviving output site index is carried on its node and every edge gets a fresh bond of `bond_dim`, so no
/// exact-product bond of dimension `χ_A·χ_B` is ever formed. `seed` fixes the random fill, and the dtype follows
/// the inputs (a complex operand makes the fill complex).
///
/// The canonical center is not part of this: the network this builds is not canonical, and the fit entry that
/// consumes it (`contract_fit_from_initial`) records and sweeps from the center it is given.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] for an invalid rank request (`bond_dim == 0`), and when the two inputs' index
/// topology does not match - the same input validation [`contract_fit_from_initial`] applies.
pub fn low_rank_initializer_tree_tn<V>(
    tn_a: &TreeTN<IdxTensor, V>,
    tn_b: &TreeTN<IdxTensor, V>,
    bond_dim: usize,
    seed: u64,
) -> std::result::Result<TreeTN<IdxTensor, V>, TreeTNOperationError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    if bond_dim == 0 {
        return Err(anyhow::anyhow!("fit low-rank initializer requires bond_dim >= 1").into());
    }
    if !tn_a.same_topology(tn_b) {
        return Err(
            anyhow::anyhow!("TreeTNs must have the same topology for fit contraction").into(),
        );
    }

    // Dtype of the random fill follows the input networks.
    let complex = network_tensors_complex(tn_a) || network_tensors_complex(tn_b);
    let mut rng = StdRng::seed_from_u64(seed);

    // 1. Fresh bond indices for each edge, with the requested dimension.
    let mut edges: Vec<(V, V)> = tn_a.site_index_network().edges().collect();
    edges.sort();
    let mut edge_bond: HashMap<(V, V), DynIndex> = HashMap::with_capacity(edges.len());
    for (a, b) in &edges {
        let key = if a < b {
            (a.clone(), b.clone())
        } else {
            (b.clone(), a.clone())
        };
        edge_bond.insert(key, DynIndex::new_dyn(bond_dim));
    }

    // 2. Per-node tensor: surviving site indices (symmetric difference of the
    //    per-node site spaces of A and B) plus the incident bonds.
    let mut node_names: Vec<V> = tn_a
        .site_index_network()
        .node_names()
        .into_iter()
        .cloned()
        .collect();
    node_names.sort();

    let mut tensors: Vec<IdxTensor> = Vec::with_capacity(node_names.len());
    for node in &node_names {
        let a_sites: HashSet<DynIndex> = tn_a
            .site_index_network()
            .site_space(node)
            .cloned()
            .unwrap_or_default();
        let b_sites: HashSet<DynIndex> = tn_b
            .site_index_network()
            .site_space(node)
            .cloned()
            .unwrap_or_default();
        let shared: HashSet<DynIndex> = a_sites.intersection(&b_sites).cloned().collect();

        let mut indices: Vec<DynIndex> = Vec::new();
        indices.extend(a_sites.iter().filter(|s| !shared.contains(*s)).cloned());
        indices.extend(b_sites.iter().filter(|s| !shared.contains(*s)).cloned());
        sort_indices_deterministic(&mut indices);

        let mut neighbors: Vec<V> = tn_a.site_index_network().neighbors(node).collect();
        neighbors.sort();
        for neighbor in &neighbors {
            let key = if node < neighbor {
                (node.clone(), neighbor.clone())
            } else {
                (neighbor.clone(), node.clone())
            };
            if let Some(bond) = edge_bond.get(&key) {
                indices.push(bond.clone());
            }
        }

        let tensor = if complex {
            IdxTensor::random::<Complex64, _>(&mut rng, indices.clone())
                .map_err(TreeTNOperationError::from)?
        } else {
            IdxTensor::random::<f64, _>(&mut rng, indices.clone())
                .map_err(TreeTNOperationError::from)?
        };
        tensors.push(tensor);
    }

    let tn_c = TreeTN::from_tensors(tensors, node_names)?;
    Ok(tn_c)
}

fn network_tensors_complex<V>(tn: &TreeTN<IdxTensor, V>) -> bool
where
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let mut complex = false;
    for node in tn.site_index_network().node_names() {
        let node = node.clone();
        if let Some(idx) = tn.node_index(&node) {
            if let Some(t) = tn.tensor(idx) {
                if t.is_complex() {
                    complex = true;
                    break;
                }
            }
        }
    }
    complex
}

#[cfg(test)]
mod tests;
