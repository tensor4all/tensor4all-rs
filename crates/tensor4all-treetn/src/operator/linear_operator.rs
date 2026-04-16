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

use tensor4all_core::AllowedPairs;
use tensor4all_core::IndexLike;
use tensor4all_core::TensorLike;

use super::index_mapping::IndexMapping;
use crate::treetn::TreeTN;

/// LinearOperator: Wraps an MPO with index mapping for automatic transformations.
///
/// # Type Parameters
///
/// * `T` - Tensor type implementing `TensorLike`
/// * `V` - Node name type
///
/// # Examples
///
/// A `LinearOperator` is typically obtained from a constructor function rather
/// than built directly. The `mpo` field contains the underlying TreeTN:
///
/// ```
/// use tensor4all_treetn::LinearOperator;
/// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
/// use tensor4all_treetn::TreeTN;
/// use std::collections::HashMap;
///
/// // Build a trivial single-node LinearOperator wrapping a 2x2 identity
/// let s_in = DynIndex::new_dyn(2);
/// let s_out = DynIndex::new_dyn(2);
/// let t = TensorDynLen::from_dense(
///     vec![s_in.clone(), s_out.clone()],
///     vec![1.0_f64, 0.0, 0.0, 1.0],
/// ).unwrap();
///
/// let mpo = TreeTN::<_, usize>::from_tensors(vec![t], vec![0]).unwrap();
/// assert_eq!(mpo.node_count(), 1);
/// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// // Build a 2x2 identity operator (single node)
    /// let site = DynIndex::new_dyn(2);
    /// let s_in = DynIndex::new_dyn(2);
    /// let s_out = DynIndex::new_dyn(2);
    /// let mpo_tensor = TensorDynLen::from_dense(
    ///     vec![s_in.clone(), s_out.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// ).unwrap();
    /// let mpo = TreeTN::<_, usize>::from_tensors(vec![mpo_tensor], vec![0]).unwrap();
    ///
    /// let mut input_mapping = HashMap::new();
    /// input_mapping.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: s_in });
    /// let mut output_mapping = HashMap::new();
    /// output_mapping.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: s_out });
    ///
    /// let op = LinearOperator::new(mpo, input_mapping, output_mapping);
    /// assert_eq!(op.mpo().node_count(), 1);
    /// ```
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

    fn ordered_site_indices(network: &TreeTN<T, V>, node: &V) -> Result<Vec<T::Index>> {
        let Some(site_space) = network.site_space(node) else {
            return Ok(Vec::new());
        };
        let node_idx = network
            .node_index(node)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tensor network", node))?;
        let tensor = network
            .tensor(node_idx)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node))?;

        Ok(tensor
            .external_indices()
            .into_iter()
            .filter(|index| site_space.contains(index))
            .collect())
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
    pub fn from_mpo_and_state(mpo: TreeTN<T, V>, state: &TreeTN<T, V>) -> Result<Self> {
        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        for node in mpo.site_index_network().node_names() {
            let state_site = state.site_space(node);
            let mpo_site = mpo.site_space(node);

            match (state_site, mpo_site) {
                (Some(_state_indices), Some(_)) => {
                    let state_indices = Self::ordered_site_indices(state, node)?;
                    let mut mpo_indices = Self::ordered_site_indices(&mpo, node)?;

                    // MPO should have exactly 2 site indices per state site index:
                    // one for input and one for output. We infer them from tensor index
                    // order using the MPO convention `[... , s_out, s_in, ...]`.
                    if state_indices.len() * 2 != mpo_indices.len() {
                        return Err(anyhow::anyhow!(
                            "Node {:?}: MPO should have 2x site indices. State has {}, MPO has {}",
                            node,
                            state_site.map_or(0, |indices| indices.len()),
                            mpo_site.map_or(0, |indices| indices.len())
                        ));
                    }

                    for state_idx in state_indices {
                        let matching_positions: Vec<_> = mpo_indices
                            .iter()
                            .enumerate()
                            .filter_map(|(position, index)| {
                                (index.dim() == state_idx.dim()).then_some(position)
                            })
                            .take(2)
                            .collect();

                        if matching_positions.len() < 2 {
                            return Err(anyhow::anyhow!(
                                "Node {:?}: Not enough MPO indices with dimension {}. Found {}",
                                node,
                                state_idx.dim(),
                                matching_positions.len()
                            ));
                        }

                        let input_position = matching_positions
                            .iter()
                            .copied()
                            .find(|position| mpo_indices[*position].same_id(&state_idx))
                            .unwrap_or(matching_positions[1]);
                        let output_position = matching_positions
                            .iter()
                            .copied()
                            .find(|position| *position != input_position)
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Node {:?}: Could not disambiguate input/output site indices",
                                    node
                                )
                            })?;

                        let output_internal = mpo_indices[output_position].clone();
                        let input_internal = mpo_indices[input_position].clone();

                        let mut removal_positions = [output_position, input_position];
                        removal_positions.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
                        for position in removal_positions {
                            mpo_indices.remove(position);
                        }

                        output_mapping.insert(
                            node.clone(),
                            IndexMapping {
                                true_index: state_idx.clone(),
                                internal_index: output_internal,
                            },
                        );
                        input_mapping.insert(
                            node.clone(),
                            IndexMapping {
                                true_index: state_idx.clone(),
                                internal_index: input_internal,
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
                transformed =
                    transformed.replaceind(&mapping.true_index, &mapping.internal_index)?;
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
                Some(t) => T::contract(&[&t, &tensor], AllowedPairs::All)?,
            });
        }

        let op_tensor = op_tensor.ok_or_else(|| anyhow::anyhow!("Empty region"))?;

        // Contract transformed tensor with operator
        let contracted = T::contract(&[&transformed, &op_tensor], AllowedPairs::All)?;

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

    fn single_site_index_from_state(state: &TreeTN<T, V>, node: &V) -> Result<T::Index> {
        let site_space = state
            .site_space(node)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in state site space", node))?;

        if site_space.len() != 1 {
            return Err(anyhow::anyhow!(
                "Node {:?}: expected exactly 1 site index in state, found {}",
                node,
                site_space.len()
            ));
        }

        site_space
            .iter()
            .next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Node {:?}: missing site index", node))
    }

    /// Reset true input indices to match the given state's site space.
    ///
    /// This only rewrites the external mapping. The internal MPO indices are unchanged.
    pub fn set_input_space_from_state(&mut self, state: &TreeTN<T, V>) -> Result<()> {
        let nodes: Vec<V> = self.input_mapping.keys().cloned().collect();
        for node in nodes {
            let new_true_index = Self::single_site_index_from_state(state, &node)?;
            let mapping = self
                .input_mapping
                .get_mut(&node)
                .ok_or_else(|| anyhow::anyhow!("Input mapping missing for node {:?}", node))?;
            if mapping.internal_index.dim() != new_true_index.dim() {
                return Err(anyhow::anyhow!(
                    "Node {:?}: input mapping dimension {} does not match state site dimension {}",
                    node,
                    mapping.internal_index.dim(),
                    new_true_index.dim()
                ));
            }
            mapping.true_index = new_true_index;
        }
        Ok(())
    }

    /// Reset true output indices to match the given state's site space.
    ///
    /// This only rewrites the external mapping. The internal MPO indices are unchanged.
    pub fn set_output_space_from_state(&mut self, state: &TreeTN<T, V>) -> Result<()> {
        let nodes: Vec<V> = self.output_mapping.keys().cloned().collect();
        for node in nodes {
            let new_true_index = Self::single_site_index_from_state(state, &node)?;
            let mapping = self
                .output_mapping
                .get_mut(&node)
                .ok_or_else(|| anyhow::anyhow!("Output mapping missing for node {:?}", node))?;
            if mapping.internal_index.dim() != new_true_index.dim() {
                return Err(anyhow::anyhow!(
                    "Node {:?}: output mapping dimension {} does not match state site dimension {}",
                    node,
                    mapping.internal_index.dim(),
                    new_true_index.dim()
                ));
            }
            mapping.true_index = new_true_index;
        }
        Ok(())
    }

    /// Align this operator's input and output site index mappings to match a target state.
    ///
    /// For each node in the operator's input and output mappings, the `true_index` is
    /// updated to the corresponding site index from the target state. The internal MPO
    /// indices remain unchanged.
    ///
    /// This is useful when an operator (e.g., from `shift_operator` or `affine_operator`)
    /// was constructed with its own site indices, but needs to be applied to a state that
    /// has different site index IDs. After calling `align_to_state`, the operator's
    /// `true_index` fields will reference the state's site indices, enabling correct
    /// index contraction during `apply_local`.
    ///
    /// # Arguments
    ///
    /// * `state` - The target state whose site indices define the true index space.
    ///   Each node in the operator's mappings must exist in the state with exactly one
    ///   site index of matching dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A node in the operator's mapping is not found in the state
    /// - A node in the state has more than one site index
    /// - The dimension of the state's site index does not match the operator's internal index
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// use tensor4all_core::{DynIndex, IndexLike, TensorDynLen};
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// let state_index = DynIndex::new_dyn(2);
    /// let state_tensor = TensorDynLen::from_dense(vec![state_index.clone()], vec![1.0, 2.0]).unwrap();
    /// let state = TreeTN::<TensorDynLen, usize>::from_tensors(vec![state_tensor], vec![0]).unwrap();
    ///
    /// let input_internal = DynIndex::new_dyn(2);
    /// let output_internal = DynIndex::new_dyn(2);
    /// let mpo_tensor = TensorDynLen::from_dense(
    ///     vec![input_internal.clone(), output_internal.clone()],
    ///     vec![1.0, 0.0, 0.0, 1.0],
    /// ).unwrap();
    /// let mpo = TreeTN::<TensorDynLen, usize>::from_tensors(vec![mpo_tensor], vec![0]).unwrap();
    ///
    /// let mut input_mapping = HashMap::new();
    /// input_mapping.insert(
    ///     0usize,
    ///     IndexMapping {
    ///         true_index: DynIndex::new_dyn(2),
    ///         internal_index: input_internal,
    ///     },
    /// );
    /// let mut output_mapping = HashMap::new();
    /// output_mapping.insert(
    ///     0usize,
    ///     IndexMapping {
    ///         true_index: DynIndex::new_dyn(2),
    ///         internal_index: output_internal,
    ///     },
    /// );
    ///
    /// let mut op = LinearOperator::new(mpo, input_mapping, output_mapping);
    /// op.align_to_state(&state).unwrap();
    ///
    /// assert!(op.input_mappings()[&0].true_index.same_id(&state_index));
    /// assert!(op.output_mappings()[&0].true_index.same_id(&state_index));
    /// ```
    pub fn align_to_state(&mut self, state: &TreeTN<T, V>) -> Result<()> {
        self.set_input_space_from_state(state)?;
        self.set_output_space_from_state(state)?;
        Ok(())
    }
}

// ============================================================================
// Helper methods
// ============================================================================

use std::collections::HashSet;

use crate::operator::Operator;
use crate::SiteIndexNetwork;

// Implement Operator trait for LinearOperator
impl<T, V> Operator<T, V> for LinearOperator<T, V>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn site_indices(&self) -> HashSet<T::Index> {
        // Return union of input and output true indices
        let mut result: HashSet<T::Index> = self
            .input_mapping
            .values()
            .map(|m| m.true_index.clone())
            .collect();
        result.extend(self.output_mapping.values().map(|m| m.true_index.clone()));
        result
    }

    fn site_index_network(&self) -> &SiteIndexNetwork<V, T::Index> {
        self.mpo.site_index_network()
    }

    fn node_names(&self) -> HashSet<V> {
        self.mpo
            .site_index_network()
            .node_names()
            .into_iter()
            .cloned()
            .collect()
    }
}

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
mod tests;
