
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
        TensorDynLen::from_dense(vec![s_in_tmp.clone(), s_out_tmp.clone()], mpo_data).unwrap();
    let mpo = TreeTN::<TensorDynLen, String>::from_tensors(vec![mpo_tensor], vec!["A".to_string()])
        .unwrap();

    // State tensor
    let state_data = vec![1.0, 0.0]; // |0> state
    let state_tensor = TensorDynLen::from_dense(vec![s.clone()], state_data).unwrap();
    let state =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
            .unwrap();

    (mpo, state, s, s_in_tmp, s_out_tmp)
}

/// Create a simple LinearOperator for testing.
/// Returns (operator, s, s_in_tmp, s_out_tmp)
fn make_linear_operator() -> (
    LinearOperator<TensorDynLen, String>,
    DynIndex, // s (true site index)
    DynIndex, // s_in_tmp
    DynIndex, // s_out_tmp
) {
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
    (op, s, s_in_tmp, s_out_tmp)
}

#[test]
fn test_linear_operator_new() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    assert!(op.get_input_mapping(&"A".to_string()).is_some());
    assert!(op.get_output_mapping(&"A".to_string()).is_some());
    assert!(op.get_input_mapping(&"B".to_string()).is_none());
    assert!(op.get_output_mapping(&"B".to_string()).is_none());
}

#[test]
fn test_linear_operator_mpo_accessor() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    // Test mpo() accessor
    assert_eq!(op.mpo().node_count(), 1);
}

#[test]
fn test_linear_operator_input_output_site_indices() {
    let (op, s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    let input_indices = op.input_site_indices();
    assert_eq!(input_indices.len(), 1);
    assert!(input_indices.iter().any(|i| i.id() == s.id()));

    let output_indices = op.output_site_indices();
    assert_eq!(output_indices.len(), 1);
    assert!(output_indices.iter().any(|i| i.id() == s.id()));
}

#[test]
fn test_linear_operator_input_output_mappings() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    assert_eq!(op.input_mappings().len(), 1);
    assert_eq!(op.output_mappings().len(), 1);
}

#[test]
fn test_linear_operator_operator_trait_site_indices() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    // Operator trait: site_indices returns union of true input and output indices
    let site_indices = Operator::site_indices(&op);
    // Since input and output use the same true index s, should be just 1
    assert_eq!(site_indices.len(), 1);
}

#[test]
fn test_linear_operator_operator_trait_node_names() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    let names = Operator::node_names(&op);
    assert_eq!(names.len(), 1);
    assert!(names.contains("A"));
}

#[test]
fn test_linear_operator_apply_local_identity() {
    let (op, s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    // Create a local tensor to apply operator to: |0> = [1, 0]
    let local_tensor = TensorDynLen::from_dense(vec![s.clone()], vec![1.0, 0.0]).unwrap();

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
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    let sin = Operator::site_index_network(&op);
    assert_eq!(sin.node_names().len(), 1);
}
