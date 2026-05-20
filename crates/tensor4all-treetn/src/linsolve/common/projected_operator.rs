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

use tensor4all_core::{IndexLike, TensorLike};

use super::environment::{EnvironmentCache, NetworkTopology};
use crate::operator::IndexMapping;
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

struct LocalIndexMapping<I> {
    true_in: I,
    internal_in: I,
    temp_in: I,
    true_out: I,
    internal_out: I,
    temp_out: I,
}

enum ContractOperand<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<'a, T> ContractOperand<'a, T> {
    fn as_ref(&'a self) -> &'a T {
        match self {
            Self::Borrowed(tensor) => tensor,
            Self::Owned(tensor) => tensor,
        }
    }
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
    /// 1. Transforms input `v`'s site indices using **unique** temp indices (avoids duplicate IDs)
    /// 2. Contracts with MPO tensors and environment tensors
    /// 3. Transforms result's temp output indices back to true site indices
    /// 4. Replaces bra-side boundary bonds with ket-side so output lives in same space as `v`
    /// 5. Permutes result to `v`'s index order so output structure matches input
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
    /// The result of applying H to v: `H|v⟩`, with same index set and order as `v`.
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

        let mut all_tensors = Vec::new();
        let mut temp_out_to_true: Vec<(T::Index, T::Index)> = Vec::new();

        if let (Some(ref input_mapping), Some(ref output_mapping)) =
            (&self.input_mapping, &self.output_mapping)
        {
            // MPO-with-mappings path: use unique temp indices to avoid duplicate IDs.
            // Replace true_index -> temp_in on v (never use internal_index on v).
            // Use same temp_in/temp_out on op tensors so they contract with v.
            let mut per_node: Vec<Option<LocalIndexMapping<T::Index>>> = Vec::new();
            for node in region {
                match (input_mapping.get(node), output_mapping.get(node)) {
                    (Some(im), Some(om)) => {
                        let temp_in = im.internal_index.sim();
                        let temp_out = om.internal_index.sim();
                        per_node.push(Some(LocalIndexMapping {
                            true_in: im.true_index.clone(),
                            internal_in: im.internal_index.clone(),
                            temp_in,
                            true_out: om.true_index.clone(),
                            internal_out: om.internal_index.clone(),
                            temp_out,
                        }));
                    }
                    (None, None) => {
                        if self
                            .operator
                            .site_space(node)
                            .is_some_and(|site_space| !site_space.is_empty())
                        {
                            return Err(anyhow::anyhow!(
                                "Missing index mappings for operator node {:?} with non-empty site space",
                                node
                            ));
                        }
                        per_node.push(None);
                    }
                    (None, Some(_)) => {
                        return Err(anyhow::anyhow!("Missing input_mapping for node {:?}", node));
                    }
                    (Some(_), None) => {
                        return Err(anyhow::anyhow!(
                            "Missing output_mapping for node {:?}",
                            node
                        ));
                    }
                }
            }

            if per_node.iter().any(Option::is_some) {
                let mut transformed_v = v.clone();
                for mapping in &per_node {
                    if let Some(mapping) = mapping {
                        transformed_v =
                            transformed_v.replaceind(&mapping.true_in, &mapping.temp_in)?;
                    }
                }
                all_tensors.push(ContractOperand::Owned(transformed_v));
            } else {
                all_tensors.push(ContractOperand::Borrowed(v));
            }

            for (node, mapping) in region.iter().zip(per_node.iter()) {
                let node_idx = self
                    .operator
                    .node_index(node)
                    .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in operator", node))?;
                let tensor = self
                    .operator
                    .tensor(node_idx)
                    .ok_or_else(|| anyhow::anyhow!("Tensor not found in operator"))?;
                if let Some(mapping) = mapping {
                    let mut t = tensor.clone();
                    t = t.replaceind(&mapping.internal_in, &mapping.temp_in)?;
                    t = t.replaceind(&mapping.internal_out, &mapping.temp_out)?;
                    temp_out_to_true.push((mapping.temp_out.clone(), mapping.true_out.clone()));
                    all_tensors.push(ContractOperand::Owned(t));
                } else {
                    all_tensors.push(ContractOperand::Borrowed(tensor));
                }
            }
        } else {
            // No mappings: plain path
            all_tensors.push(ContractOperand::Borrowed(v));
            for node in region {
                let node_idx = self
                    .operator
                    .node_index(node)
                    .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in operator", node))?;
                let tensor = self
                    .operator
                    .tensor(node_idx)
                    .ok_or_else(|| anyhow::anyhow!("Tensor not found in operator"))?;
                all_tensors.push(ContractOperand::Borrowed(tensor));
            }
        }

        // Collect environments from neighbors outside the region
        for node in region {
            for neighbor in topology.neighbors(node) {
                if region.contains(&neighbor) {
                    continue;
                }
                if let Some(env) = self.envs.get(&neighbor, node) {
                    all_tensors.push(ContractOperand::Borrowed(env));
                }
            }
        }

        let tensor_refs: Vec<&T> = all_tensors.iter().map(ContractOperand::as_ref).collect();
        let mut contracted = T::contract(&tensor_refs)?;

        // Replace temp_out -> true_index
        for (temp_out, true_idx) in &temp_out_to_true {
            contracted = contracted.replaceind(temp_out, true_idx)?;
        }

        // Bra -> ket boundary bonds in result so output lives in same space as v (ket bonds).
        for node in region {
            for neighbor in topology.neighbors(node) {
                if region.contains(&neighbor) {
                    continue;
                }
                let ket_edge = match ket_state.edge_between(node, &neighbor) {
                    Some(e) => e,
                    None => continue,
                };
                let bra_edge = match bra_state.edge_between(node, &neighbor) {
                    Some(e) => e,
                    None => continue,
                };
                let ket_bond = match ket_state.bond_index(ket_edge) {
                    Some(b) => b.clone(),
                    None => continue,
                };
                let bra_bond = match bra_state.bond_index(bra_edge) {
                    Some(b) => b.clone(),
                    None => continue,
                };
                if contracted.external_indices().iter().any(|i| i == &bra_bond) {
                    contracted = contracted.replaceind(&bra_bond, &ket_bond)?;
                }
            }
        }

        // Align result to v's index order
        let v_inds = v.external_indices();
        let res_inds = contracted.external_indices();
        let v_index_keys: std::collections::HashSet<_> =
            v_inds.iter().map(|i| (i.clone(), i.dim())).collect();
        let res_index_keys: std::collections::HashSet<_> =
            res_inds.iter().map(|i| (i.clone(), i.dim())).collect();
        if v_index_keys == res_index_keys && v_inds.len() == res_inds.len() {
            contracted = contracted.permuteinds(&v_inds)?;
        }

        Ok(contracted)
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
        let child_envs: Vec<&T> = child_neighbors
            .iter()
            .filter_map(|child| self.envs.get(child, from))
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

        let input_unmapped = self
            .input_mapping
            .as_ref()
            .is_none_or(|mapping| !mapping.contains_key(from));
        let output_unmapped = self
            .output_mapping
            .as_ref()
            .is_none_or(|mapping| !mapping.contains_key(from));
        let operator_site_is_empty = self
            .operator
            .site_space(from)
            .is_some_and(|site_space| site_space.is_empty());
        let no_site_spectator = input_unmapped && output_unmapped && operator_site_is_empty;

        let mut tensor_refs = Vec::with_capacity(3 + child_envs.len());

        // Transform ket tensor for contraction with operator
        if let Some(ref input_mapping) = self.input_mapping {
            if let Some(mapping) = input_mapping.get(from) {
                tensor_refs.push(ContractOperand::Owned(
                    tensor_ket.replaceind(&mapping.true_index, &mapping.internal_index)?,
                ));
            } else {
                tensor_refs.push(ContractOperand::Borrowed(tensor_ket));
            }
        } else {
            tensor_refs.push(ContractOperand::Borrowed(tensor_ket));
        }

        if !no_site_spectator {
            tensor_refs.push(ContractOperand::Borrowed(tensor_op));
        }

        // Transform bra_conj tensor for contraction with operator
        if let Some(ref output_mapping) = self.output_mapping {
            if let Some(mapping) = output_mapping.get(from) {
                tensor_refs.push(ContractOperand::Owned(
                    bra_conj.replaceind(&mapping.true_index, &mapping.internal_index)?,
                ));
            } else {
                tensor_refs.push(ContractOperand::Owned(bra_conj));
            }
        } else {
            tensor_refs.push(ContractOperand::Owned(bra_conj));
        }

        // Contract ket, op, bra, and child environments together
        // Let contract() find the optimal contraction order
        tensor_refs.extend(child_envs.into_iter().map(ContractOperand::Borrowed));
        let tensor_refs = tensor_refs
            .iter()
            .map(ContractOperand::as_ref)
            .collect::<Vec<_>>();
        let contracted = T::contract(&tensor_refs)?;
        if no_site_spectator {
            contracted.contract_pair(tensor_op)
        } else {
            Ok(contracted)
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
