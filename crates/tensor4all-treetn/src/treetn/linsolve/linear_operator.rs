//! LinearOperator: Wrapper for MPO with index mapping.
//!
//! This module provides a `LinearOperator` that wraps an MPO (Matrix Product Operator)
//! and handles the index ID mapping between true site indices and internal MPO indices.
//!
//! # Problem
//!
//! In the equation `A * x = b`:
//! - `A.s_in` should match `x`'s site indices
//! - `A.s_out` should match `b`'s site indices
//!
//! However, a tensor cannot have two indices with the same ID. So the MPO internally
//! uses `s_in_tmp` and `s_out_tmp` with independent IDs.
//!
//! # Solution
//!
//! `LinearOperator` stores:
//! - The MPO with internal index IDs (`s_in_tmp`, `s_out_tmp`)
//! - Mapping from true `s_in`/`s_out` to internal `s_in_tmp`/`s_out_tmp`
//!
//! When applying to `x`, it automatically handles the index transformations.

use std::collections::HashMap;
use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::IndexLike;
use tensor4all_core::TensorLike;

use super::projected_operator::IndexMapping;
use crate::treetn::TreeTN;

/// LinearOperator: Wraps an MPO with index mapping for automatic transformations.
///
/// # Deprecated
///
/// This type is deprecated. Use `LinsolveUpdater::with_index_mappings()` instead,
/// which directly accepts index mappings without the intermediate `LinearOperator` wrapper.
/// The index mappings are now handled internally by `ProjectedOperator`.
///
/// # Type Parameters
///
/// * `T` - Tensor type implementing `TensorLike`
/// * `V` - Node name type
#[derive(Debug, Clone)]
pub struct LinearOperator<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The MPO with internal index IDs
    pub mpo: TreeTN<T, V>,
    /// Input index mapping: node -> (true s_in, internal s_in_tmp)
    pub input_mapping: HashMap<V, IndexMapping<T::Index>>,
    /// Output index mapping: node -> (true s_out, internal s_out_tmp)
    pub output_mapping: HashMap<V, IndexMapping<T::Index>>,
}

impl<T, V> LinearOperator<T, V>
where
    T: TensorLike,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Create a new LinearOperator from an MPO and index mappings.
    ///
    /// # Arguments
    ///
    /// * `mpo` - The MPO with internal index IDs
    /// * `input_mapping` - Mapping from true input indices to internal indices
    /// * `output_mapping` - Mapping from true output indices to internal indices
    pub fn new(
        mpo: TreeTN<T, V>,
        input_mapping: HashMap<V, IndexMapping<T::Index>>,
        output_mapping: HashMap<V, IndexMapping<T::Index>>,
    ) -> Self {
        Self {
            mpo,
            input_mapping,
            output_mapping,
        }
    }

    /// Create a LinearOperator from an MPO and a reference state.
    ///
    /// This assumes:
    /// - The MPO has site indices that we need to map
    /// - The state's site indices define the true input space
    /// - For `A * x = b` with `space(x) = space(b)`, the output space equals input space
    ///
    /// # Arguments
    ///
    /// * `mpo` - The MPO (operator A)
    /// * `state` - Reference state (defines the true site index space)
    ///
    /// # Returns
    ///
    /// A LinearOperator with proper index mappings, or an error if structure is incompatible.
    pub fn from_mpo_and_state(
        mpo: TreeTN<T, V>,
        state: &TreeTN<T, V>,
    ) -> Result<Self> {
        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        for node in mpo.site_index_network().node_names() {
            // Get state's site indices for this node
            let state_site = state.site_space(&node);

            // Get MPO's site indices for this node
            let mpo_site = mpo.site_space(&node);

            match (state_site, mpo_site) {
                (Some(state_indices), Some(mpo_indices)) => {
                    // MPO should have exactly 2 site indices per state site index:
                    // one for input (s_in_tmp) and one for output (s_out_tmp)
                    // Both should have the same dimension as the state's site index.

                    if state_indices.len() * 2 != mpo_indices.len() {
                        return Err(anyhow::anyhow!(
                            "Node {:?}: MPO should have 2x site indices. State has {}, MPO has {}",
                            node,
                            state_indices.len(),
                            mpo_indices.len()
                        ));
                    }

                    // For each state site index, find matching MPO indices by dimension
                    for state_idx in state_indices {
                        let dim = state_idx.dim();

                        // Find MPO indices with matching dimension
                        let matching_mpo: Vec<_> = mpo_indices
                            .iter()
                            .filter(|idx| idx.dim() == dim)
                            .collect();

                        if matching_mpo.len() < 2 {
                            return Err(anyhow::anyhow!(
                                "Node {:?}: Not enough MPO indices with dimension {}. Found {}",
                                node,
                                dim,
                                matching_mpo.len()
                            ));
                        }

                        // Convention: first matching is s_in_tmp, second is s_out_tmp
                        // (This depends on how the MPO was constructed)
                        input_mapping.insert(
                            node.clone(),
                            IndexMapping {
                                true_index: state_idx.clone(),
                                internal_index: matching_mpo[0].clone(),
                            },
                        );

                        // For output, use the same true index (space(x) = space(b))
                        output_mapping.insert(
                            node.clone(),
                            IndexMapping {
                                true_index: state_idx.clone(),
                                internal_index: matching_mpo[1].clone(),
                            },
                        );
                    }
                }
                (None, None) => {
                    // No site indices for this node, OK
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Node {:?}: Mismatched site space presence between state and MPO",
                        node
                    ));
                }
            }
        }

        Ok(Self {
            mpo,
            input_mapping,
            output_mapping,
        })
    }

    /// Apply the operator to a state: compute `A|x⟩`.
    ///
    /// This handles index transformations automatically:
    /// 1. Replace state's site indices with MPO's input indices (s_in_tmp)
    /// 2. Contract state with MPO
    /// 3. Replace MPO's output indices (s_out_tmp) with true output indices
    ///
    /// # Arguments
    ///
    /// * `state` - The input state |x⟩
    ///
    /// # Returns
    ///
    /// The result `A|x⟩` with correct output indices.
    pub fn apply(&self, state: &TreeTN<T, V>) -> Result<TreeTN<T, V>> {
        // For now, implement a simple node-by-node application
        // This is a placeholder - full implementation requires proper TTN contraction

        let mut result_tensors: HashMap<V, T> = HashMap::new();

        for node in state.site_index_network().node_names() {
            let node_idx = state
                .node_index(&node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in state", node))?;
            let state_tensor = state
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node))?;

            // Get operator tensor
            let op_node_idx = self
                .mpo
                .node_index(&node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in MPO", node))?;
            let op_tensor = self
                .mpo
                .tensor(op_node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?} in MPO", node))?;

            // Step 1: Replace state's site indices with MPO's input indices
            let transformed_state = if let Some(mapping) = self.input_mapping.get(&node) {
                state_tensor.replaceind(&mapping.true_index, &mapping.internal_index)?
            } else {
                state_tensor.clone()
            };

            // Step 2: Contract with operator
            // Find common indices between transformed state and operator
            let state_indices = transformed_state.external_indices();
            let op_indices = op_tensor.external_indices();
            let common: Vec<_> = state_indices
                .iter()
                .filter_map(|idx_s| {
                    op_indices
                        .iter()
                        .find(|idx_o| idx_s.same_id(idx_o))
                        .map(|idx_o| (idx_s.clone(), idx_o.clone()))
                })
                .collect();

            let contracted = transformed_state.tensordot(op_tensor, &common)?;

            // Step 3: Replace MPO's output indices with true output indices
            let result_tensor = if let Some(mapping) = self.output_mapping.get(&node) {
                contracted.replaceind(&mapping.internal_index, &mapping.true_index)?
            } else {
                contracted
            };

            result_tensors.insert(node.clone(), result_tensor);
        }

        // Build result TreeTN from tensors
        // For now, return a simple error - we need to implement proper TreeTN construction
        Err(anyhow::anyhow!(
            "LinearOperator::apply: Full TTN contraction not yet implemented. \
             Use apply_local for local tensor application."
        ))
    }

    /// Apply the operator to a local tensor at a specific region.
    ///
    /// This is used during the sweep for local updates.
    ///
    /// # Arguments
    ///
    /// * `local_tensor` - The local tensor (merged tensors from the region)
    /// * `region` - The nodes in the current region
    ///
    /// # Returns
    ///
    /// The result of applying the operator to the local tensor.
    pub fn apply_local(&self, local_tensor: &T, region: &[V]) -> Result<T> {
        // Step 1: Replace input indices in local_tensor with internal indices
        let mut transformed = local_tensor.clone();
        for node in region {
            if let Some(mapping) = self.input_mapping.get(node) {
                transformed = transformed.replaceind(&mapping.true_index, &mapping.internal_index)?;
            }
        }

        // Step 2: Contract with local operator tensors
        let mut op_tensor: Option<T> = None;
        for node in region {
            let node_idx = self
                .mpo
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in MPO", node))?;
            let tensor = self
                .mpo
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in MPO for node {:?}", node))?
                .clone();

            op_tensor = Some(match op_tensor {
                None => tensor,
                Some(t) => {
                    let t_indices = t.external_indices();
                    let tensor_indices = tensor.external_indices();
                    let common: Vec<_> = t_indices
                        .iter()
                        .filter_map(|idx_t| {
                            tensor_indices
                                .iter()
                                .find(|idx| idx_t.same_id(idx))
                                .map(|idx| (idx_t.clone(), idx.clone()))
                        })
                        .collect();
                    t.tensordot(&tensor, &common)?
                }
            });
        }

        let op_tensor = op_tensor.ok_or_else(|| anyhow::anyhow!("Empty region"))?;

        // Contract transformed tensor with operator
        let transformed_indices = transformed.external_indices();
        let op_indices = op_tensor.external_indices();
        let common: Vec<_> = transformed_indices
            .iter()
            .filter_map(|idx_t| {
                op_indices
                    .iter()
                    .find(|idx_o| idx_t.same_id(idx_o))
                    .map(|idx_o| (idx_t.clone(), idx_o.clone()))
            })
            .collect();

        let contracted = transformed.tensordot(&op_tensor, &common)?;

        // Step 3: Replace output indices back to true indices
        let mut result = contracted;
        for node in region {
            if let Some(mapping) = self.output_mapping.get(node) {
                result = result.replaceind(&mapping.internal_index, &mapping.true_index)?;
            }
        }

        Ok(result)
    }

    /// Get the internal MPO.
    pub fn mpo(&self) -> &TreeTN<T, V> {
        &self.mpo
    }

    /// Get input mapping for a node.
    pub fn get_input_mapping(&self, node: &V) -> Option<&IndexMapping<T::Index>> {
        self.input_mapping.get(node)
    }

    /// Get output mapping for a node.
    pub fn get_output_mapping(&self, node: &V) -> Option<&IndexMapping<T::Index>> {
        self.output_mapping.get(node)
    }
}

// ============================================================================
// Helper methods
// ============================================================================

use std::collections::HashSet;

impl<T, V> LinearOperator<T, V>
where
    T: TensorLike,
    T::Index: IndexLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Get all input site indices (true indices from state space).
    pub fn input_site_indices(&self) -> HashSet<T::Index> {
        self.input_mapping
            .values()
            .map(|m| m.true_index.clone())
            .collect()
    }

    /// Get all output site indices (true indices from result space).
    pub fn output_site_indices(&self) -> HashSet<T::Index> {
        self.output_mapping
            .values()
            .map(|m| m.true_index.clone())
            .collect()
    }

    /// Get all input mappings.
    pub fn input_mappings(&self) -> &HashMap<V, IndexMapping<T::Index>> {
        &self.input_mapping
    }

    /// Get all output mappings.
    pub fn output_mappings(&self) -> &HashMap<V, IndexMapping<T::Index>> {
        &self.output_mapping
    }
}

#[cfg(test)]
mod tests {
    // Tests will be added when we integrate with the rest of the system
}
