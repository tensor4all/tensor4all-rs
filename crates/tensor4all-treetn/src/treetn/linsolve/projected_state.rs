//! ProjectedState: 2-chain environment for RHS computation.
//!
//! Computes `<b|x_local>` efficiently for Tree Tensor Networks.

use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::{AllowedPairs, IndexLike, TensorLike};

use super::environment::{EnvironmentCache, NetworkTopology};
use crate::treetn::TreeTN;

/// ProjectedState: Manages 2-chain environments for RHS computation.
///
/// This computes `<b|x_local>` for each local region during the sweep.
///
/// For Tree Tensor Networks, the environment is computed by contracting
/// all tensors outside the "open region" into environment tensors.
/// The open region consists of nodes being updated in the current sweep step.
///
/// # Structure
///
/// For each edge (from, to) pointing towards the open region, we cache:
/// ```text
/// env[(from, to)] = contraction of:
///   - bra tensor at `from` (conjugated RHS)
///   - ket tensor at `from` (current solution)
///   - all child environments (edges pointing away from `to`)
/// ```
///
/// This forms a "2-chain" overlap: `<b|x>` contracted over
/// all nodes except the open region.
pub struct ProjectedState<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The RHS state |b⟩
    pub rhs: TreeTN<T, V>,
    /// Environment cache
    pub envs: EnvironmentCache<T, V>,
}

impl<T, V> ProjectedState<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Create a new ProjectedState.
    pub fn new(rhs: TreeTN<T, V>) -> Self {
        Self {
            rhs,
            envs: EnvironmentCache::new(),
        }
    }

    /// Compute the local constant term `<b|_local` for the given region.
    ///
    /// This returns the local RHS tensors contracted with environments.
    ///
    /// **Note:** Only V_in = V_out is currently supported. The ket_state is used
    /// for both bra and ket in environment computations.
    ///
    /// # Arguments
    /// * `region` - The nodes in the local update region
    /// * `ket_state` - The current solution state
    /// * `topology` - The network topology
    pub fn local_constant_term<NT: NetworkTopology<V>>(
        &mut self,
        region: &[V],
        ket_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<T> {
        // Debug: Log which state is used
        static FIRST_LOCAL_TERM: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(true);
        if FIRST_LOCAL_TERM.swap(false, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "  [ProjectedState::local_constant_term] region={:?}, using ket_state as bra_state",
                region
            );
        }

        // Use ket_state directly as bra_state (V_in = V_out)
        self.local_constant_term_impl(region, ket_state, ket_state, topology)
    }

    /// Internal implementation for computing local constant term with explicit bra state.
    fn local_constant_term_impl<NT: NetworkTopology<V>>(
        &mut self,
        region: &[V],
        _ket_state: &TreeTN<T, V>,
        bra_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<T> {
        // Ensure environments are computed
        self.ensure_environments(region, bra_state, topology)?;

        // Collect all tensors to contract: local RHS tensors + environments
        let mut all_tensors: Vec<T> = Vec::new();

        // Collect local RHS tensors (conjugated)
        for node in region {
            let node_idx = self
                .rhs
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in RHS", node))?;
            let tensor = self
                .rhs
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in RHS"))?
                .conj();
            all_tensors.push(tensor);
        }

        // Collect environments from neighbors outside the region
        let mut env_tensors: Vec<(V, V, T)> = Vec::new();
        for node in region {
            for neighbor in topology.neighbors(node) {
                if !region.contains(&neighbor) {
                    if let Some(env) = self.envs.get(&neighbor, node) {
                        env_tensors.push((neighbor.clone(), node.clone(), env.clone()));
                        all_tensors.push(env.clone());
                    }
                }
            }
        }

        // Debug: Log which environments are used
        static FIRST_LOCAL_TERM_ENV: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(true);
        if FIRST_LOCAL_TERM_ENV.swap(false, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "  [ProjectedState::local_constant_term_impl] region={:?}, using {} environments",
                region,
                env_tensors.len()
            );
            for (from, to, env) in &env_tensors {
                let env_norm = env.norm();
                let env_shape: Vec<usize> =
                    env.external_indices().iter().map(|i| i.dim()).collect();
                eprintln!(
                    "    env[({:?}, {:?})]: shape={:?}, norm={:.6e}",
                    from, to, env_shape, env_norm
                );
            }
        }

        // Use T::contract for optimal contraction ordering
        let tensor_refs: Vec<&T> = all_tensors.iter().collect();
        let result = T::contract(&tensor_refs, AllowedPairs::All)?;

        // Debug: Log final local constant term
        static FIRST_LOCAL_TERM_RESULT: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(true);
        if FIRST_LOCAL_TERM_RESULT.swap(false, std::sync::atomic::Ordering::Relaxed) {
            let result_norm = result.norm();
            let result_shape: Vec<usize> =
                result.external_indices().iter().map(|i| i.dim()).collect();
            eprintln!(
                "  [ProjectedState::local_constant_term_impl] result: shape={:?}, norm={:.6e}",
                result_shape, result_norm
            );
        }

        Ok(result)
    }

    /// Ensure environments are computed for neighbors of the region.
    ///
    /// # Arguments
    /// * `bra_state` - Reference state in V_out for environment computation
    fn ensure_environments<NT: NetworkTopology<V>>(
        &mut self,
        region: &[V],
        bra_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<()> {
        for node in region {
            for neighbor in topology.neighbors(node) {
                if !region.contains(&neighbor) && !self.envs.contains(&neighbor, node) {
                    let env = self.compute_environment(&neighbor, node, bra_state, topology)?;
                    self.envs.insert(neighbor.clone(), node.clone(), env);
                }
            }
        }
        Ok(())
    }

    /// Recursively compute environment for edge (from, to).
    ///
    /// Computes `<b|ref_out>` partial contraction at node `from`.
    ///
    /// # Arguments
    /// * `bra_state` - Reference state in V_out for environment computation
    fn compute_environment<NT: NetworkTopology<V>>(
        &mut self,
        from: &V,
        to: &V,
        bra_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<T> {
        // First, ensure child environments are computed
        let child_neighbors: Vec<V> = topology.neighbors(from).filter(|n| n != to).collect();

        for child in &child_neighbors {
            if !self.envs.contains(child, from) {
                let child_env = self.compute_environment(child, from, bra_state, topology)?;
                self.envs.insert(child.clone(), from.clone(), child_env);
            }
        }

        // Collect child environments
        let child_envs: Vec<T> = child_neighbors
            .iter()
            .filter_map(|child| self.envs.get(child, from).cloned())
            .collect();

        // Contract bra (RHS) with ket (bra_state as reference) at this node
        let node_idx_bra = self
            .rhs
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in RHS", from))?;
        let node_idx_ket = bra_state
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in bra_state", from))?;

        let tensor_bra = self
            .rhs
            .tensor(node_idx_bra)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in RHS"))?;
        let tensor_ket = bra_state
            .tensor(node_idx_ket)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in bra_state"))?;

        // Debug: Log environment computation (only for first step to avoid spam)
        static FIRST_ENV_COMPUTE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(true);
        if FIRST_ENV_COMPUTE.swap(false, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "  [ProjectedState::compute_environment] from={:?}, to={:?}, computing <b|bra_state>",
                from, to
            );
            eprintln!(
                "  [ProjectedState::compute_environment] bra_state is used as ket in <b|ket>"
            );
        }

        let bra_conj = tensor_bra.conj();

        // Contract bra and ket - T::contract auto-detects contractable pairs
        let bra_ket = T::contract(&[&bra_conj, tensor_ket], AllowedPairs::All)?;

        // Contract bra*ket with child environments using T::contract
        let result = if child_envs.is_empty() {
            bra_ket
        } else {
            let mut all_tensors: Vec<&T> = vec![&bra_ket];
            all_tensors.extend(child_envs.iter());
            T::contract(&all_tensors, AllowedPairs::All)?
        };

        // Debug: Log environment value for first step (site0, site1 region)
        static ENV_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let count = ENV_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count < 5 {
            let env_norm = result.norm();
            let env_shape: Vec<usize> = result.external_indices().iter().map(|i| i.dim()).collect();
            eprintln!(
                "  [ProjectedState::compute_environment] env[({:?}, {:?})]: shape={:?}, norm={:.6e}",
                from, to, env_shape, env_norm
            );
        }

        Ok(result)
    }

    /// Invalidate caches affected by updates to the given region.
    pub fn invalidate<NT: NetworkTopology<V>>(&mut self, region: &[V], topology: &NT) {
        self.envs.invalidate(region, topology);
    }
}
