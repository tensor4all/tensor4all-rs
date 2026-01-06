//! ProjectedOperator: 3-chain environment for operator application.
//!
//! Computes `<psi|H|v>` efficiently at each site.

use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::index::{DynId, NoSymmSpace, Symmetry};
use tensor4all_core::TensorDynLen;

use super::environment::{EnvironmentCache, NetworkTopology};
use crate::treetn::TreeTN;

/// ProjectedOperator: Manages 3-chain environments for operator application.
///
/// This computes `<psi|H|v>` at each site during the sweep.
///
/// # Diagram
///
/// ```text
/// o--o--o-      -o--o--o--o--o--o <psi|   ← bra (conjugate of state vector)
/// |  |  |  |  |  |  |  |  |  |  |
/// o--o--o--o--o--o--o--o--o--o--o H       ← MPO (operator A)
/// |  |  |  |  |  |  |  |  |  |  |
/// o--o--o-      -o--o--o--o--o--o |psi>   ← ket (state vector)
///        ↑      ↑
///      lpos    rpos (unprojected sites)
/// ```
pub struct ProjectedOperator<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The operator H
    pub operator: TreeTN<Id, Symm, V>,
    /// Environment cache
    pub envs: EnvironmentCache<Id, Symm, V>,
}

impl<Id, Symm, V> ProjectedOperator<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Create a new ProjectedOperator.
    pub fn new(operator: TreeTN<Id, Symm, V>) -> Self {
        Self {
            operator,
            envs: EnvironmentCache::new(),
        }
    }

    /// Apply the operator to a local tensor: compute `H|v⟩` at the current position.
    ///
    /// # Arguments
    /// * `v` - The local tensor to apply the operator to
    /// * `region` - The nodes in the open region
    /// * `state` - The current state |psi⟩ (used for environment computation)
    /// * `topology` - Network topology for traversal
    ///
    /// # Returns
    /// The result of applying H to v: `H|v⟩`
    pub fn apply<T: NetworkTopology<V>>(
        &mut self,
        v: &TensorDynLen<Id, Symm>,
        region: &[V],
        state: &TreeTN<Id, Symm, V>,
        topology: &T,
    ) -> Result<TensorDynLen<Id, Symm>> {
        // Ensure environments are computed
        self.ensure_environments(region, state, topology)?;

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

        // Contract local operator tensors
        let mut op_tensor: Option<TensorDynLen<Id, Symm>> = None;

        for node in region {
            let node_idx = self.operator
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in operator", node))?;
            let tensor = self.operator
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in operator"))?
                .clone();

            op_tensor = Some(match op_tensor {
                None => tensor,
                Some(t) => {
                    let common: Vec<_> = t
                        .indices
                        .iter()
                        .filter_map(|idx_t| {
                            tensor
                                .indices
                                .iter()
                                .find(|idx| idx_t.id == idx.id)
                                .map(|idx| (idx_t.clone(), idx.clone()))
                        })
                        .collect();
                    t.tensordot(&tensor, &common)?
                }
            });
        }

        let op_tensor = op_tensor.ok_or_else(|| anyhow::anyhow!("Empty region"))?;

        // Contract environments with operator
        let mut result = op_tensor;
        for env in &env_tensors {
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
            result = result.tensordot(env, &common)?;
        }

        // Contract with input vector v
        let common: Vec<_> = result
            .indices
            .iter()
            .filter_map(|idx_r| {
                v.indices
                    .iter()
                    .find(|idx_v| idx_r.id == idx_v.id)
                    .map(|idx_v| (idx_r.clone(), idx_v.clone()))
            })
            .collect();

        result = result.tensordot(v, &common)?;

        Ok(result)
    }

    /// Ensure environments are computed for neighbors of the region.
    fn ensure_environments<T: NetworkTopology<V>>(
        &mut self,
        region: &[V],
        state: &TreeTN<Id, Symm, V>,
        topology: &T,
    ) -> Result<()> {
        for node in region {
            for neighbor in topology.neighbors(node) {
                if !region.contains(&neighbor) && !self.envs.contains(&neighbor, node) {
                    let env = self.compute_environment(&neighbor, node, state, topology)?;
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
        state: &TreeTN<Id, Symm, V>,
        topology: &T,
    ) -> Result<TensorDynLen<Id, Symm>> {
        // First, ensure child environments are computed
        let child_neighbors: Vec<V> = topology
            .neighbors(from)
            .filter(|n| n != to)
            .collect();

        for child in &child_neighbors {
            if !self.envs.contains(child, from) {
                let child_env = self.compute_environment(child, from, state, topology)?;
                self.envs.insert(child.clone(), from.clone(), child_env);
            }
        }

        // Collect child environments
        let child_envs: Vec<TensorDynLen<Id, Symm>> = child_neighbors
            .iter()
            .filter_map(|child| self.envs.get(child, from).cloned())
            .collect();

        // Get tensors from bra, operator, and ket at this node
        let node_idx_bra = state
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in state", from))?;
        let node_idx_op = self.operator
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in operator", from))?;
        let node_idx_ket = state
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in state", from))?;

        let tensor_bra = state
            .tensor(node_idx_bra)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in state"))?;
        let tensor_op = self.operator
            .tensor(node_idx_op)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in operator"))?;
        let tensor_ket = state
            .tensor(node_idx_ket)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in state"))?;

        // Contract ket with operator on site indices
        let site_space_ket = state.site_space(from).cloned().unwrap_or_default();
        let site_space_op = self.operator.site_space(from).cloned().unwrap_or_default();

        let ket_op_common: Vec<_> = site_space_ket
            .iter()
            .filter_map(|idx_ket| {
                site_space_op
                    .iter()
                    .find(|idx_op| idx_ket.id == idx_op.id)
                    .map(|idx_op| (idx_ket.clone(), idx_op.clone()))
            })
            .collect();

        let ket_op = if ket_op_common.is_empty() {
            tensor_ket.tensordot(tensor_op, &[])?
        } else {
            tensor_ket.tensordot(tensor_op, &ket_op_common)?
        };

        // Contract with bra (conjugated) on output site indices
        let bra_conj = tensor_bra.conj();
        let site_space_bra = state.site_space(from).cloned().unwrap_or_default();

        let ket_op_bra_common: Vec<_> = ket_op
            .indices
            .iter()
            .filter_map(|idx| {
                site_space_bra
                    .iter()
                    .find(|idx_bra| idx.id == idx_bra.id)
                    .map(|idx_bra| (idx.clone(), idx_bra.clone()))
            })
            .collect();

        let mut result = ket_op.tensordot(&bra_conj, &ket_op_bra_common)?;

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

    /// Compute the local dimension (size of the local Hilbert space).
    pub fn local_dimension(&self, region: &[V]) -> usize {
        let mut dim = 1;
        for node in region {
            if let Some(site_space) = self.operator.site_space(node) {
                for idx in site_space {
                    dim *= idx.symm.total_dim();
                }
            }
        }
        dim
    }

    /// Invalidate caches affected by updates to the given region.
    pub fn invalidate<T: NetworkTopology<V>>(&mut self, region: &[V], topology: &T) {
        self.envs.invalidate(region, topology);
    }
}
