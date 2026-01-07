//! ProjectedState: 2-chain environment for RHS computation.
//!
//! Computes `<b|x_local>` efficiently for Tree Tensor Networks.

use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::index::{DynId, NoSymmSpace, Symmetry};
use tensor4all_core::{contract_multi, TensorDynLen};

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
pub struct ProjectedState<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The RHS state |b⟩
    pub rhs: TreeTN<Id, Symm, V>,
    /// Environment cache
    pub envs: EnvironmentCache<Id, Symm, V>,
}

impl<Id, Symm, V> ProjectedState<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId> + Send + Sync + 'static,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Create a new ProjectedState.
    pub fn new(rhs: TreeTN<Id, Symm, V>) -> Self {
        Self {
            rhs,
            envs: EnvironmentCache::new(),
        }
    }

    /// Compute the local constant term `<b|_local` for the given region.
    ///
    /// This returns the local RHS tensors contracted with environments.
    pub fn local_constant_term<T: NetworkTopology<V>>(
        &mut self,
        region: &[V],
        solution: &TreeTN<Id, Symm, V>,
        topology: &T,
    ) -> Result<TensorDynLen<Id, Symm>> {
        // Ensure environments are computed
        self.ensure_environments(region, solution, topology)?;

        // Collect all tensors to contract: local RHS tensors + environments
        let mut all_tensors: Vec<TensorDynLen<Id, Symm>> = Vec::new();

        // Collect local RHS tensors (conjugated)
        for node in region {
            let node_idx = self.rhs
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in RHS", node))?;
            let tensor = self.rhs
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

        // Use contract_multi for optimal contraction ordering
        contract_multi(&all_tensors)
    }

    /// Ensure environments are computed for neighbors of the region.
    fn ensure_environments<T: NetworkTopology<V>>(
        &mut self,
        region: &[V],
        solution: &TreeTN<Id, Symm, V>,
        topology: &T,
    ) -> Result<()> {
        for node in region {
            for neighbor in topology.neighbors(node) {
                if !region.contains(&neighbor) && !self.envs.contains(&neighbor, node) {
                    let env = self.compute_environment(&neighbor, node, solution, topology)?;
                    self.envs.insert(neighbor.clone(), node.clone(), env);
                }
            }
        }
        Ok(())
    }

    /// Recursively compute environment for edge (from, to).
    fn compute_environment<T: NetworkTopology<V>>(
        &mut self,
        from: &V,
        to: &V,
        solution: &TreeTN<Id, Symm, V>,
        topology: &T,
    ) -> Result<TensorDynLen<Id, Symm>> {
        // First, ensure child environments are computed
        let child_neighbors: Vec<V> = topology
            .neighbors(from)
            .filter(|n| n != to)
            .collect();

        for child in &child_neighbors {
            if !self.envs.contains(child, from) {
                let child_env = self.compute_environment(child, from, solution, topology)?;
                self.envs.insert(child.clone(), from.clone(), child_env);
            }
        }

        // Collect child environments
        let child_envs: Vec<TensorDynLen<Id, Symm>> = child_neighbors
            .iter()
            .filter_map(|child| self.envs.get(child, from).cloned())
            .collect();

        // Contract bra (RHS) with ket (solution) at this node
        let node_idx_bra = self.rhs
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in RHS", from))?;
        let node_idx_ket = solution
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in solution", from))?;

        let tensor_bra = self.rhs
            .tensor(node_idx_bra)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in RHS"))?;
        let tensor_ket = solution
            .tensor(node_idx_ket)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in solution"))?;

        let bra_conj = tensor_bra.conj();

        // Find common site indices (contract only over site indices, not bond indices)
        let site_space_bra = self.rhs.site_space(from).cloned().unwrap_or_default();
        let site_space_ket = solution.site_space(from).cloned().unwrap_or_default();

        let common_site_pairs: Vec<_> = site_space_bra
            .iter()
            .filter_map(|idx_bra| {
                site_space_ket
                    .iter()
                    .find(|idx_ket| idx_bra.id == idx_ket.id)
                    .map(|idx_ket| (idx_bra.clone(), idx_ket.clone()))
            })
            .collect();

        let bra_ket = if common_site_pairs.is_empty() {
            bra_conj.tensordot(tensor_ket, &[])?
        } else {
            bra_conj.tensordot(tensor_ket, &common_site_pairs)?
        };

        // Contract bra*ket with child environments using contract_multi
        if child_envs.is_empty() {
            Ok(bra_ket)
        } else {
            let mut all_tensors = vec![bra_ket];
            all_tensors.extend(child_envs);
            contract_multi(&all_tensors)
        }
    }

    /// Invalidate caches affected by updates to the given region.
    pub fn invalidate<T: NetworkTopology<V>>(&mut self, region: &[V], topology: &T) {
        self.envs.invalidate(region, topology);
    }
}
