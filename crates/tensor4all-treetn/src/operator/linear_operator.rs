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
    pub fn from_mpo_and_state(mpo: TreeTN<T, V>, state: &TreeTN<T, V>) -> Result<Self> {
        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        for node in mpo.site_index_network().node_names() {
            // Get state's site indices for this node
            let state_site = state.site_space(node);

            // Get MPO's site indices for this node
            let mpo_site = mpo.site_space(node);

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
                        let matching_mpo: Vec<_> =
                            mpo_indices.iter().filter(|idx| idx.dim() == dim).collect();

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
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tensor4all_core::{DynIndex, IndexLike, TensorDynLen};

    use crate::operator::index_mapping::IndexMapping;
    use crate::operator::Operator;
    use crate::treetn::TreeTN;

    /// Create a simple "MPO" TreeTN with two site indices per node (input + output).
    /// Structure: single node "A" with indices (s_in_tmp, s_out_tmp)
    /// Also returns a "state" TreeTN with a single site index s.
    fn make_simple_mpo_and_state() -> (
        TreeTN<TensorDynLen, String>,
        TreeTN<TensorDynLen, String>,
        DynIndex, // s (true site index)
        DynIndex, // s_in_tmp
        DynIndex, // s_out_tmp
    ) {
        let s = DynIndex::new_dyn(2); // true site index
        let s_in_tmp = DynIndex::new_dyn(2); // MPO input index
        let s_out_tmp = DynIndex::new_dyn(2); // MPO output index

        // MPO tensor: identity operator (2x2 -> 2x2)
        // s_in_tmp x s_out_tmp with identity values
        let mpo_data = vec![1.0, 0.0, 0.0, 1.0]; // identity matrix
        let mpo_tensor =
            TensorDynLen::from_dense_f64(vec![s_in_tmp.clone(), s_out_tmp.clone()], mpo_data);
        let mpo =
            TreeTN::<TensorDynLen, String>::from_tensors(vec![mpo_tensor], vec!["A".to_string()])
                .unwrap();

        // State tensor
        let state_data = vec![1.0, 0.0]; // |0> state
        let state_tensor = TensorDynLen::from_dense_f64(vec![s.clone()], state_data);
        let state =
            TreeTN::<TensorDynLen, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
                .unwrap();

        (mpo, state, s, s_in_tmp, s_out_tmp)
    }

    #[test]
    fn test_linear_operator_new() {
        let (mpo, _state, s, s_in_tmp, s_out_tmp) = make_simple_mpo_and_state();

        let mut input_mapping = HashMap::new();
        input_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_in_tmp.clone(),
            },
        );

        let mut output_mapping = HashMap::new();
        output_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_out_tmp.clone(),
            },
        );

        let op = LinearOperator::new(mpo, input_mapping, output_mapping);

        assert!(op.get_input_mapping(&"A".to_string()).is_some());
        assert!(op.get_output_mapping(&"A".to_string()).is_some());
        assert!(op.get_input_mapping(&"B".to_string()).is_none());
        assert!(op.get_output_mapping(&"B".to_string()).is_none());
    }

    #[test]
    fn test_linear_operator_mpo_accessor() {
        let (mpo, _state, s, s_in_tmp, s_out_tmp) = make_simple_mpo_and_state();

        let mut input_mapping = HashMap::new();
        input_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_in_tmp.clone(),
            },
        );

        let mut output_mapping = HashMap::new();
        output_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_out_tmp.clone(),
            },
        );

        let op = LinearOperator::new(mpo, input_mapping, output_mapping);

        // Test mpo() accessor
        assert_eq!(op.mpo().node_count(), 1);
    }

    #[test]
    fn test_linear_operator_input_output_site_indices() {
        let (mpo, _state, s, s_in_tmp, s_out_tmp) = make_simple_mpo_and_state();

        let mut input_mapping = HashMap::new();
        input_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_in_tmp.clone(),
            },
        );

        let mut output_mapping = HashMap::new();
        output_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_out_tmp.clone(),
            },
        );

        let op = LinearOperator::new(mpo, input_mapping, output_mapping);

        let input_indices = op.input_site_indices();
        assert_eq!(input_indices.len(), 1);
        assert!(input_indices.iter().any(|i| i.id() == s.id()));

        let output_indices = op.output_site_indices();
        assert_eq!(output_indices.len(), 1);
        assert!(output_indices.iter().any(|i| i.id() == s.id()));
    }

    #[test]
    fn test_linear_operator_input_output_mappings() {
        let (mpo, _state, s, s_in_tmp, s_out_tmp) = make_simple_mpo_and_state();

        let mut input_mapping = HashMap::new();
        input_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_in_tmp.clone(),
            },
        );

        let mut output_mapping = HashMap::new();
        output_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_out_tmp.clone(),
            },
        );

        let op = LinearOperator::new(mpo, input_mapping, output_mapping);

        assert_eq!(op.input_mappings().len(), 1);
        assert_eq!(op.output_mappings().len(), 1);
    }

    #[test]
    fn test_linear_operator_operator_trait_site_indices() {
        let (mpo, _state, s, s_in_tmp, s_out_tmp) = make_simple_mpo_and_state();

        let mut input_mapping = HashMap::new();
        input_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_in_tmp.clone(),
            },
        );

        let mut output_mapping = HashMap::new();
        output_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_out_tmp.clone(),
            },
        );

        let op = LinearOperator::new(mpo, input_mapping, output_mapping);

        // Operator trait: site_indices returns union of true input and output indices
        let site_indices = Operator::site_indices(&op);
        // Since input and output use the same true index s, should be just 1
        assert_eq!(site_indices.len(), 1);
    }

    #[test]
    fn test_linear_operator_operator_trait_node_names() {
        let (mpo, _state, s, s_in_tmp, s_out_tmp) = make_simple_mpo_and_state();

        let mut input_mapping = HashMap::new();
        input_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_in_tmp.clone(),
            },
        );

        let mut output_mapping = HashMap::new();
        output_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_out_tmp.clone(),
            },
        );

        let op = LinearOperator::new(mpo, input_mapping, output_mapping);

        let names = Operator::node_names(&op);
        assert_eq!(names.len(), 1);
        assert!(names.contains("A"));
    }

    #[test]
    fn test_linear_operator_apply_local_identity() {
        let (mpo, _state, s, s_in_tmp, s_out_tmp) = make_simple_mpo_and_state();

        let mut input_mapping = HashMap::new();
        input_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_in_tmp.clone(),
            },
        );

        let mut output_mapping = HashMap::new();
        output_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_out_tmp.clone(),
            },
        );

        let op = LinearOperator::new(mpo, input_mapping, output_mapping);

        // Create a local tensor to apply operator to: |0> = [1, 0]
        let local_tensor = TensorDynLen::from_dense_f64(vec![s.clone()], vec![1.0, 0.0]);

        let result = op.apply_local(&local_tensor, &["A".to_string()]).unwrap();

        // Identity operator should preserve the state
        let result_data = result.to_vec_f64().unwrap();
        assert_eq!(result_data.len(), 2);
        // Values should be approximately [1, 0]
        assert!((result_data[0] - 1.0).abs() < 1e-10);
        assert!((result_data[1]).abs() < 1e-10);
    }

    #[test]
    fn test_linear_operator_operator_trait_site_index_network() {
        let (mpo, _state, s, s_in_tmp, s_out_tmp) = make_simple_mpo_and_state();

        let mut input_mapping = HashMap::new();
        input_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_in_tmp.clone(),
            },
        );

        let mut output_mapping = HashMap::new();
        output_mapping.insert(
            "A".to_string(),
            IndexMapping {
                true_index: s.clone(),
                internal_index: s_out_tmp.clone(),
            },
        );

        let op = LinearOperator::new(mpo, input_mapping, output_mapping);

        let sin = Operator::site_index_network(&op);
        assert_eq!(sin.node_names().len(), 1);
    }
}
