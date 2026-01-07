//! ProjectedOperator: 3-chain environment for operator application.
//!
//! Computes `<psi|H|v>` efficiently for Tree Tensor Networks.

use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::index::{DynId, NoSymmSpace, Symmetry};
use tensor4all_core::{contract_multi, TensorDynLen};

use super::environment::{EnvironmentCache, NetworkTopology};
use crate::treetn::TreeTN;

/// ProjectedOperator: Manages 3-chain environments for operator application.
///
/// This computes `<psi|H|v>` for each local region during the sweep.
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
///   - bra tensor at `from` (conjugated)
///   - operator tensor at `from`
///   - ket tensor at `from`
///   - all child environments (edges pointing away from `to`)
/// ```
///
/// This forms a "3-chain" sandwich: `<bra| H |ket>` contracted over
/// all nodes except the open region.
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
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId> + Send + Sync + 'static,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug + Send + Sync,
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

        // Collect all tensors to contract: operator tensors + environments + input v
        let mut all_tensors: Vec<TensorDynLen<Id, Symm>> = Vec::new();

        // Collect local operator tensors
        for node in region {
            let node_idx = self.operator
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in operator", node))?;
            let tensor = self.operator
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in operator"))?
                .clone();
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

        // Add input vector v
        all_tensors.push(v.clone());

        // Use contract_multi for optimal contraction ordering
        contract_multi(&all_tensors)
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

        // Environment contraction for 3-chain: <bra| H |ket>
        //
        // Index convention:
        // - state's site index: s (same ID for both bra and ket)
        // - operator's input index: s_in (same ID as state's s)
        // - operator's output index: s_out (different ID)
        //
        // Contraction order: (bra * operator) * ket
        // 1. bra(s) * operator(s_out, s_in) contracts s with s_out? No!
        //    We need s_out to match bra's s, but they have different IDs.
        //
        // The issue: s_out has a NEW ID, not matching bra's s.
        // For proper contraction, operator output indices should match bra's site indices.
        //
        // Current workaround: Contract over ALL common indices between tensors,
        // not just site space indices. The operator has bond indices that connect
        // to environments, and site indices that connect to bra/ket.

        let bra_conj = tensor_bra.conj();
        let site_space_ket = state.site_space(from).cloned().unwrap_or_default();

        // Contract ket with operator on INPUT site indices (same ID by design)
        let ket_op_common: Vec<_> = tensor_ket
            .indices
            .iter()
            .filter_map(|idx_ket| {
                // Find operator index with same ID (this is s_in)
                tensor_op
                    .indices
                    .iter()
                    .find(|idx_op| idx_ket.id == idx_op.id)
                    .map(|idx_op| (idx_ket.clone(), idx_op.clone()))
            })
            .filter(|(idx_ket, _)| {
                // Only contract site indices, not bond indices
                site_space_ket.iter().any(|s| s.id == idx_ket.id)
            })
            .collect();

        let ket_op = if ket_op_common.is_empty() {
            tensor_ket.tensordot(tensor_op, &[])?
        } else {
            tensor_ket.tensordot(tensor_op, &ket_op_common)?
        };

        // Now contract with bra on OUTPUT site indices
        // The operator's s_out should contract with bra's s.
        // But they have DIFFERENT IDs! This is the fundamental issue.
        //
        // Solution: Find operator indices that are NOT in state's site space
        // (these are s_out), and contract them with bra's site indices by DIMENSION matching.
        //
        // Better solution: The operator's site_space should only contain OUTPUT indices,
        // and we explicitly pass which indices are input vs output.
        //
        // For now, contract all remaining common indices between ket_op and bra_conj.
        // After ket*op contraction, ket_op has:
        //   - bond indices (from ket and operator)
        //   - operator's output indices (s_out)
        // bra_conj has:
        //   - bond indices (from bra = state)
        //   - bra's site indices (s)
        //
        // Bond indices should match, site indices (s_out vs s) don't match.

        let ket_op_bra_common: Vec<_> = ket_op
            .indices
            .iter()
            .filter_map(|idx| {
                bra_conj
                    .indices
                    .iter()
                    .find(|idx_bra| idx.id == idx_bra.id)
                    .map(|idx_bra| (idx.clone(), idx_bra.clone()))
            })
            .collect();

        let ket_op_bra = ket_op.tensordot(&bra_conj, &ket_op_bra_common)?;

        // Contract ket_op_bra with child environments using contract_multi
        if child_envs.is_empty() {
            Ok(ket_op_bra)
        } else {
            let mut all_tensors = vec![ket_op_bra];
            all_tensors.extend(child_envs);
            contract_multi(&all_tensors)
        }
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
