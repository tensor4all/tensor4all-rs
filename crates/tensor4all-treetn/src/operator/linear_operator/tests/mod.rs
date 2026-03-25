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
    let result_data = result.to_vec::<f64>().unwrap();
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

#[test]
fn test_from_mpo_and_state() {
    let (mpo, state, _s, _s_in_tmp, _s_out_tmp) = make_simple_mpo_and_state();
    let op = LinearOperator::from_mpo_and_state(mpo, &state).unwrap();

    // Should have input and output mappings for node "A"
    assert!(op.get_input_mapping(&"A".to_string()).is_some());
    assert!(op.get_output_mapping(&"A".to_string()).is_some());
    assert_eq!(op.input_mappings().len(), 1);
    assert_eq!(op.output_mappings().len(), 1);
}

#[test]
fn test_from_mpo_and_state_mismatched_site_count() {
    // MPO with 3 site indices but state with 1 -> 2*1 != 3
    let s = DynIndex::new_dyn(2);
    let s_in_tmp = DynIndex::new_dyn(2);
    let s_out_tmp = DynIndex::new_dyn(2);
    let s_extra = DynIndex::new_dyn(2);

    let mpo_tensor = TensorDynLen::from_dense(
        vec![s_in_tmp.clone(), s_out_tmp.clone(), s_extra.clone()],
        vec![0.0; 8],
    )
    .unwrap();
    let mpo = TreeTN::<TensorDynLen, String>::from_tensors(vec![mpo_tensor], vec!["A".to_string()])
        .unwrap();

    let state_tensor = TensorDynLen::from_dense(vec![s.clone()], vec![1.0, 0.0]).unwrap();
    let state =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
            .unwrap();

    let result = LinearOperator::from_mpo_and_state(mpo, &state);
    assert!(result.is_err());
}

#[test]
fn test_from_mpo_and_state_no_site_indices() {
    // Both MPO and state with no site indices at a node
    // This is the (None, None) branch
    let bond = DynIndex::new_dyn(2);
    let s = DynIndex::new_dyn(2);
    let s_in = DynIndex::new_dyn(2);
    let s_out = DynIndex::new_dyn(2);

    // State: A has site index, B has no site index (only bond)
    let t_a = TensorDynLen::from_dense(vec![s.clone(), bond.clone()], vec![1.0; 4]).unwrap();
    let t_b = TensorDynLen::from_dense(vec![bond.clone()], vec![1.0; 2]).unwrap();
    let mut state = TreeTN::<TensorDynLen, String>::new();
    state.add_tensor("A".to_string(), t_a).unwrap();
    state.add_tensor("B".to_string(), t_b).unwrap();
    let a = state.node_index(&"A".to_string()).unwrap();
    let b = state.node_index(&"B".to_string()).unwrap();
    state.connect(a, &bond, b, &bond).unwrap();

    // MPO: A has two site indices, B has no site index (only bond)
    let bond2 = DynIndex::new_dyn(2);
    let t_a_mpo = TensorDynLen::from_dense(
        vec![s_in.clone(), s_out.clone(), bond2.clone()],
        vec![0.0; 8],
    )
    .unwrap();
    let t_b_mpo = TensorDynLen::from_dense(vec![bond2.clone()], vec![0.0; 2]).unwrap();
    let mut mpo = TreeTN::<TensorDynLen, String>::new();
    mpo.add_tensor("A".to_string(), t_a_mpo).unwrap();
    mpo.add_tensor("B".to_string(), t_b_mpo).unwrap();
    let a2 = mpo.node_index(&"A".to_string()).unwrap();
    let b2 = mpo.node_index(&"B".to_string()).unwrap();
    mpo.connect(a2, &bond2, b2, &bond2).unwrap();

    // Should succeed - the (None, None) branch for node B
    let result = LinearOperator::from_mpo_and_state(mpo, &state);
    assert!(result.is_ok());
}

#[test]
fn test_from_mpo_and_state_mismatched_presence() {
    // State has site indices at A, but MPO has none at A -> (Some, None) branch
    let s = DynIndex::new_dyn(2);

    let state_tensor = TensorDynLen::from_dense(vec![s.clone()], vec![1.0, 0.0]).unwrap();
    let state =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
            .unwrap();

    // MPO with no site indices
    let mpo_idx = DynIndex::new_dyn(1);
    let mpo_tensor = TensorDynLen::from_dense(vec![mpo_idx.clone()], vec![1.0]).unwrap();
    let mpo = TreeTN::<TensorDynLen, String>::from_tensors(vec![mpo_tensor], vec!["A".to_string()])
        .unwrap();

    // The state has site indices but the mpo's site indices are different,
    // so this should trigger a mismatch error
    let result = LinearOperator::from_mpo_and_state(mpo, &state);
    // This hits the (Some, None) or (None, Some) branch
    assert!(result.is_err());
}

#[test]
fn test_from_mpo_and_state_not_enough_matching_dims() {
    // MPO has 2 site indices but with different dimension from state
    let s = DynIndex::new_dyn(2);
    let s_in_tmp = DynIndex::new_dyn(3); // dim 3 != dim 2
    let s_out_tmp = DynIndex::new_dyn(3);

    let mpo_tensor =
        TensorDynLen::from_dense(vec![s_in_tmp.clone(), s_out_tmp.clone()], vec![0.0; 9]).unwrap();
    let mpo = TreeTN::<TensorDynLen, String>::from_tensors(vec![mpo_tensor], vec!["A".to_string()])
        .unwrap();

    let state_tensor = TensorDynLen::from_dense(vec![s.clone()], vec![1.0, 0.0]).unwrap();
    let state =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
            .unwrap();

    let result = LinearOperator::from_mpo_and_state(mpo, &state);
    assert!(result.is_err());
}

/// Create a 2-node MPO and state for multi-node apply_local test.
fn make_two_node_mpo_and_operator() -> (
    LinearOperator<TensorDynLen, String>,
    DynIndex, // s0 (true site index for A)
    DynIndex, // s1 (true site index for B)
) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s0_in = DynIndex::new_dyn(2);
    let s0_out = DynIndex::new_dyn(2);
    let s1_in = DynIndex::new_dyn(2);
    let s1_out = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);

    // Identity MPO: two nodes with bond dim 1
    // A tensor: s0_in x s0_out x bond
    let t_a = TensorDynLen::from_dense(
        vec![s0_in.clone(), s0_out.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 1.0], // identity for each bond=0 slice
    )
    .unwrap();
    // B tensor: bond x s1_in x s1_out
    let t_b = TensorDynLen::from_dense(
        vec![bond.clone(), s1_in.clone(), s1_out.clone()],
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();

    let mut mpo = TreeTN::<TensorDynLen, String>::new();
    mpo.add_tensor("A".to_string(), t_a).unwrap();
    mpo.add_tensor("B".to_string(), t_b).unwrap();
    let a = mpo.node_index(&"A".to_string()).unwrap();
    let b = mpo.node_index(&"B".to_string()).unwrap();
    mpo.connect(a, &bond, b, &bond).unwrap();

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        "B".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_in.clone(),
        },
    );

    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        "B".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_out.clone(),
        },
    );

    let op = LinearOperator::new(mpo, input_mapping, output_mapping);
    (op, s0, s1)
}

#[test]
fn test_apply_local_multi_node_region() {
    let (op, s0, s1) = make_two_node_mpo_and_operator();

    // Create a local tensor spanning both nodes: |00> = [1, 0, 0, 0]
    let local_tensor =
        TensorDynLen::from_dense(vec![s0.clone(), s1.clone()], vec![1.0, 0.0, 0.0, 0.0]).unwrap();

    let result = op
        .apply_local(&local_tensor, &["A".to_string(), "B".to_string()])
        .unwrap();

    // Identity operator should preserve the state
    let result_data = result.to_vec::<f64>().unwrap();
    assert_eq!(result_data.len(), 4);
    assert!((result_data[0] - 1.0).abs() < 1e-10);
    for &v in &result_data[1..] {
        assert!(v.abs() < 1e-10);
    }
}
