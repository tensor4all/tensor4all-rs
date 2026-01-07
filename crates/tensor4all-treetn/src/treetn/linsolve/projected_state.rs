//! ProjectedState: 2-chain environment for RHS computation.
//!
//! Computes `<b|x_local>` efficiently for Tree Tensor Networks.

use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::index::{DynId, NoSymmSpace, Symmetry};
use tensor4all_core::TensorDynLen;

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
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug,
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

        // Collect environments from neighbors outside the region
        let mut env_tensors: Vec<TensorDynLen<Id, Symm>> = Vec::new();

        for node in region {
            for neighbor in topology.neighbors(node) {
                if !region.contains(&neighbor) {
                    if let Some(env) = self.envs.get(&neighbor, node) {
                        env_tensors.push(env.clone());
                    }
                }
            }
        }

        // Contract local RHS tensors
        let mut result: Option<TensorDynLen<Id, Symm>> = None;

        for node in region {
            let node_idx = self.rhs
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in RHS", node))?;
            let tensor = self.rhs
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in RHS"))?
                .conj();

            result = Some(match result {
                None => tensor,
                Some(r) => {
                    let common: Vec<_> = r
                        .indices
                        .iter()
                        .filter_map(|idx_r| {
                            tensor
                                .indices
                                .iter()
                                .find(|idx_t| idx_r.id == idx_t.id)
                                .map(|idx_t| (idx_r.clone(), idx_t.clone()))
                        })
                        .collect();
                    r.tensordot(&tensor, &common)?
                }
            });
        }

        let mut result = result.ok_or_else(|| anyhow::anyhow!("Empty region"))?;

        // Contract with environments
        for env in env_tensors {
            let common: Vec<_> = result
                .indices
                .iter()
                .filter_map(|idx_r| {
                    env.indices
                        .iter()
                        .find(|idx_e| idx_r.id == idx_e.id)
                        .map(|idx_e| (idx_r.clone(), idx_e.clone()))
                })
                .collect();
            result = result.tensordot(&env, &common)?;
        }

        Ok(result)
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

        // Find common site indices
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

        let mut result = if common_site_pairs.is_empty() {
            bra_conj.tensordot(tensor_ket, &[])?
        } else {
            bra_conj.tensordot(tensor_ket, &common_site_pairs)?
        };

        // Contract with child environments
        for child_env in child_envs {
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

        Ok(result)
    }

    /// Invalidate caches affected by updates to the given region.
    pub fn invalidate<T: NetworkTopology<V>>(&mut self, region: &[V], topology: &T) {
        self.envs.invalidate(region, topology);
    }
}
