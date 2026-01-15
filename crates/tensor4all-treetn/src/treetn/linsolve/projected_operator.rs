//! ProjectedOperator: 3-chain environment for operator application.
//!
//! Computes `<psi|H|v>` efficiently for Tree Tensor Networks.
//!
//! # Index Mapping
//!
//! For MPOs where input/output site indices have different IDs from the state's
//! site indices (required because a tensor cannot have two indices with the same ID),
//! use `with_index_mappings` to define the correspondence.

use std::collections::HashMap;
use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::{AllowedPairs, IndexLike, TensorLike};

use super::environment::{EnvironmentCache, NetworkTopology};
use crate::treetn::TreeTN;

/// Mapping between true site indices and internal MPO indices.
///
/// In the equation `A * x = b`:
/// - The state `x` has site indices with certain IDs
/// - The MPO `A` internally uses different IDs (`s_in_tmp`, `s_out_tmp`)
/// - This mapping defines the correspondence
#[derive(Debug, Clone)]
pub struct IndexMapping<I>
where
    I: IndexLike,
{
    /// True site index (from state x or b)
    pub true_index: I,
    /// Internal MPO index (s_in_tmp or s_out_tmp)
    pub internal_index: I,
}

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
pub struct ProjectedOperator<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The operator H
    pub operator: TreeTN<T, V>,
    /// Environment cache
    pub envs: EnvironmentCache<T, V>,
    /// Input index mapping (true site index -> MPO's internal input index)
    /// Used when MPO has internal indices different from state's site indices.
    pub input_mapping: Option<HashMap<V, IndexMapping<T::Index>>>,
    /// Output index mapping (true site index -> MPO's internal output index)
    pub output_mapping: Option<HashMap<V, IndexMapping<T::Index>>>,
}

impl<T, V> ProjectedOperator<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Create a new ProjectedOperator.
    pub fn new(operator: TreeTN<T, V>) -> Self {
        Self {
            operator,
            envs: EnvironmentCache::new(),
            input_mapping: None,
            output_mapping: None,
        }
    }

    /// Create a new ProjectedOperator with index mappings from a LinearOperator.
    ///
    /// The mappings define how state's site indices relate to MPO's internal indices.
    /// This is required when the MPO uses internal indices (s_in_tmp, s_out_tmp)
    /// that differ from the state's site indices.
    pub fn with_index_mappings(
        operator: TreeTN<T, V>,
        input_mapping: HashMap<V, IndexMapping<T::Index>>,
        output_mapping: HashMap<V, IndexMapping<T::Index>>,
    ) -> Self {
        Self {
            operator,
            envs: EnvironmentCache::new(),
            input_mapping: Some(input_mapping),
            output_mapping: Some(output_mapping),
        }
    }

    /// Apply the operator to a local tensor: compute `H|v⟩` at the current position.
    ///
    /// If index mappings are set (via `with_index_mappings`), this method:
    /// 1. Transforms input `v`'s site indices to MPO's internal input indices
    /// 2. Contracts with MPO tensors and environment tensors
    /// 3. Transforms result's internal output indices back to true site indices
    ///
    /// # Arguments
    /// * `v` - The local tensor to apply the operator to
    /// * `region` - The nodes in the open region
    /// * `ket_state` - The current state |ket⟩ (used for ket in environment computation)
    /// * `bra_state` - The reference state ⟨bra| (used for bra in environment computation)
    ///   For V_in = V_out, this is the same as ket_state.
    ///   For V_in ≠ V_out, this should be a state in V_out.
    /// * `topology` - Network topology for traversal
    ///
    /// # Returns
    /// The result of applying H to v: `H|v⟩`
    pub fn apply<NT: NetworkTopology<V>>(
        &mut self,
        v: &T,
        region: &[V],
        ket_state: &TreeTN<T, V>,
        bra_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<T> {
        // Ensure environments are computed
        self.ensure_environments(region, ket_state, bra_state, topology)?;

        // Collect all tensors to contract: operator tensors + environments + input v
        let mut all_tensors: Vec<T> = Vec::new();

        // Step 1: Transform input tensor - replace true site indices with internal indices
        let transformed_v = if let Some(ref input_mapping) = self.input_mapping {
            let mut result = v.clone();
            for node in region {
                if let Some(mapping) = input_mapping.get(node) {
                    result = result.replaceind(&mapping.true_index, &mapping.internal_index)?;
                }
            }
            result
        } else {
            v.clone()
        };
        all_tensors.push(transformed_v);

        // Step 2: Collect local operator tensors
        for node in region {
            let node_idx = self
                .operator
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in operator", node))?;
            let tensor = self
                .operator
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in operator"))?
                .clone();
            all_tensors.push(tensor);
        }

        // Step 3: Collect environments from neighbors outside the region
        for node in region {
            for neighbor in topology.neighbors(node) {
                if !region.contains(&neighbor) {
                    if let Some(env) = self.envs.get(&neighbor, node) {
                        all_tensors.push(env.clone());
                    }
                }
            }
        }

        // Contract all tensors
        let contracted = T::contract(&all_tensors, AllowedPairs::All)?;

        // Step 4: Transform output - replace internal output indices with true indices
        if let Some(ref output_mapping) = self.output_mapping {
            let mut result = contracted;
            for node in region {
                if let Some(mapping) = output_mapping.get(node) {
                    result = result.replaceind(&mapping.internal_index, &mapping.true_index)?;
                }
            }
            Ok(result)
        } else {
            Ok(contracted)
        }
    }

    /// Ensure environments are computed for neighbors of the region.
    fn ensure_environments<NT: NetworkTopology<V>>(
        &mut self,
        region: &[V],
        ket_state: &TreeTN<T, V>,
        bra_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<()> {
        for node in region {
            for neighbor in topology.neighbors(node) {
                if !region.contains(&neighbor) && !self.envs.contains(&neighbor, node) {
                    let env =
                        self.compute_environment(&neighbor, node, ket_state, bra_state, topology)?;
                    self.envs.insert(neighbor.clone(), node.clone(), env);
                }
            }
        }
        Ok(())
    }

    /// Recursively compute environment for edge (from, to).
    ///
    /// # Arguments
    /// * `from` - Source node of the edge
    /// * `to` - Destination node of the edge
    /// * `ket_state` - State for ket tensors (input space, V_in)
    /// * `bra_state` - State for bra tensors (output space, V_out)
    /// * `topology` - Network topology
    fn compute_environment<NT: NetworkTopology<V>>(
        &mut self,
        from: &V,
        to: &V,
        ket_state: &TreeTN<T, V>,
        bra_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<T> {
        // First, ensure child environments are computed
        let child_neighbors: Vec<V> = topology.neighbors(from).filter(|n| n != to).collect();

        for child in &child_neighbors {
            if !self.envs.contains(child, from) {
                let child_env =
                    self.compute_environment(child, from, ket_state, bra_state, topology)?;
                self.envs.insert(child.clone(), from.clone(), child_env);
            }
        }

        // Collect child environments
        let child_envs: Vec<T> = child_neighbors
            .iter()
            .filter_map(|child| self.envs.get(child, from).cloned())
            .collect();

        // Get tensors from bra (V_out), operator, and ket (V_in) at this node
        let node_idx_bra = bra_state
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in bra_state", from))?;
        let node_idx_op = self
            .operator
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in operator", from))?;
        let node_idx_ket = ket_state
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in ket_state", from))?;

        let tensor_bra = bra_state
            .tensor(node_idx_bra)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in bra_state"))?;
        let tensor_op = self
            .operator
            .tensor(node_idx_op)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in operator"))?;
        let tensor_ket = ket_state
            .tensor(node_idx_ket)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in ket_state"))?;

        // Environment contraction for 3-chain: <bra| H |ket>
        //
        // When using index mappings (from LinearOperator):
        // - ket's site index (s) needs to be replaced with MPO's input index (s_in_tmp) for contraction
        // - bra's site index (s) needs to be replaced with MPO's output index (s_out_tmp) for contraction
        //
        // Without mappings: indices are assumed to match directly (same ID).

        let bra_conj = tensor_bra.conj();

        // Transform ket tensor for contraction with operator
        let transformed_ket = if let Some(ref input_mapping) = self.input_mapping {
            if let Some(mapping) = input_mapping.get(from) {
                tensor_ket.replaceind(&mapping.true_index, &mapping.internal_index)?
            } else {
                tensor_ket.clone()
            }
        } else {
            tensor_ket.clone()
        };

        // Transform bra_conj tensor for contraction with operator
        let transformed_bra_conj = if let Some(ref output_mapping) = self.output_mapping {
            if let Some(mapping) = output_mapping.get(from) {
                bra_conj.replaceind(&mapping.true_index, &mapping.internal_index)?
            } else {
                bra_conj.clone()
            }
        } else {
            bra_conj.clone()
        };

        // Contract ket with op - T::contract auto-detects contractable pairs
        let ket_op = T::contract(&[transformed_ket, tensor_op.clone()], AllowedPairs::All)?;

        // Contract ket_op with transformed_bra_conj
        let ket_op_bra = T::contract(&[ket_op, transformed_bra_conj], AllowedPairs::All)?;

        // Contract ket_op_bra with child environments using T::contract
        if child_envs.is_empty() {
            Ok(ket_op_bra)
        } else {
            let mut all_tensors: Vec<T> = vec![ket_op_bra];
            all_tensors.extend(child_envs);
            T::contract(&all_tensors, AllowedPairs::All)
        }
    }

    /// Compute the local dimension (size of the local Hilbert space).
    pub fn local_dimension(&self, region: &[V]) -> usize {
        let mut dim = 1;
        for node in region {
            if let Some(site_space) = self.operator.site_space(node) {
                for idx in site_space {
                    dim *= idx.dim();
                }
            }
        }
        dim
    }

    /// Invalidate caches affected by updates to the given region.
    pub fn invalidate<NT: NetworkTopology<V>>(&mut self, region: &[V], topology: &NT) {
        self.envs.invalidate(region, topology);
    }
}
