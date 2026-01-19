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
    /// # Arguments
    /// * `region` - The nodes in the local update region
    /// * `ket_state` - The current solution state (in V_in)
    /// * `topology` - The network topology
    ///
    /// For V_in ≠ V_out case, use `local_constant_term_with_bra` instead.
    pub fn local_constant_term<NT: NetworkTopology<V>>(
        &mut self,
        region: &[V],
        ket_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<T> {
        // For V_in = V_out case, use a copied bra_state with fresh link IDs.
        // This avoids accidental link-index collisions between bra/ket networks.
        let bra_state = ket_state.sim_linkinds()?;
        self.local_constant_term_with_bra(region, ket_state, &bra_state, topology)
    }

    /// Compute the local constant term `<b|_local` for the given region with explicit bra state.
    ///
    /// For V_in ≠ V_out case, provides a reference state in V_out for environment computation.
    ///
    /// # Arguments
    /// * `region` - The nodes in the local update region
    /// * `ket_state` - The current solution state (in V_in)
    /// * `bra_state` - Reference state for bra in environment computation (in V_out)
    /// * `topology` - The network topology
    pub fn local_constant_term_with_bra<NT: NetworkTopology<V>>(
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
        for node in region {
            for neighbor in topology.neighbors(node) {
                if !region.contains(&neighbor) {
                    if let Some(env) = self.envs.get(&neighbor, node) {
                        all_tensors.push(env.clone());
                    }
                }
            }
        }

        // Use T::contract for optimal contraction ordering
        let tensor_refs: Vec<&T> = all_tensors.iter().collect();
        T::contract(&tensor_refs, AllowedPairs::All)
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

        let bra_conj = tensor_bra.conj();

        // Contract bra and ket - T::contract auto-detects contractable pairs
        let bra_ket = T::contract(&[&bra_conj, tensor_ket], AllowedPairs::All)?;

        // Contract bra*ket with child environments using T::contract
        if child_envs.is_empty() {
            Ok(bra_ket)
        } else {
            let mut all_tensors: Vec<&T> = vec![&bra_ket];
            all_tensors.extend(child_envs.iter());
            T::contract(&all_tensors, AllowedPairs::All)
        }
    }

    /// Invalidate caches affected by updates to the given region.
    pub fn invalidate<NT: NetworkTopology<V>>(&mut self, region: &[V], topology: &NT) {
        self.envs.invalidate(region, topology);
    }
}
