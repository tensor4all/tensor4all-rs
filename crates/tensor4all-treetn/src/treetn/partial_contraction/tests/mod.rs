use super::*;
use crate::treetn::contraction::{ContractionMethod, ContractionOptions};
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorIndex};

fn make_partial_contraction_inputs() -> (
    TreeTN<TensorDynLen, String>,
    TreeTN<TensorDynLen, String>,
    DynIndex,
    DynIndex,
    DynIndex,
    DynIndex,
    DynIndex,
    DynIndex,
) {
    let s_contract_a = DynIndex::new_dyn(2);
    let s_multiply_a = DynIndex::new_dyn(2);
    let s_a_only = DynIndex::new_dyn(3);
    let bond_a = DynIndex::new_dyn(2);

    let s_contract_b = DynIndex::new_dyn(2);
    let s_multiply_b = DynIndex::new_dyn(2);
    let s_b_only = DynIndex::new_dyn(4);
    let bond_b = DynIndex::new_dyn(2);

    let t_a0 = TensorDynLen::from_dense(
        vec![s_contract_a.clone(), s_multiply_a.clone(), bond_a.clone()],
        vec![1.0; 8],
    )
    .unwrap();
    let t_a1 =
        TensorDynLen::from_dense(vec![bond_a.clone(), s_a_only.clone()], vec![1.0; 6]).unwrap();

    let t_b0 = TensorDynLen::from_dense(
        vec![s_contract_b.clone(), s_multiply_b.clone(), bond_b.clone()],
        vec![2.0; 8],
    )
    .unwrap();
    let t_b1 =
        TensorDynLen::from_dense(vec![bond_b.clone(), s_b_only.clone()], vec![2.0; 8]).unwrap();

    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_a0, t_a1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_b0, t_b1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    (
        tn_a,
        tn_b,
        s_contract_a,
        s_contract_b,
        s_multiply_a,
        s_multiply_b,
        s_a_only,
        s_b_only,
    )
}

#[test]
fn test_partial_contraction_spec_creation() {
    let idx_a = DynIndex::new_dyn(4);
    let idx_b = DynIndex::new_dyn(4);
    let spec: PartialContractionSpec<DynIndex> = PartialContractionSpec {
        contract_pairs: vec![(idx_a.clone(), idx_b.clone())],
        multiply_pairs: vec![],
    };
    assert_eq!(spec.contract_pairs.len(), 1);
    assert!(spec.multiply_pairs.is_empty());
}

#[test]
fn test_partial_contract_rejects_same_node_contract_and_multiply() {
    // contract_pairs and multiply_pairs target the same node "A" → must error
    let (tn_a, tn_b, s_contract_a, s_contract_b, s_multiply_a, s_multiply_b, _s_a_only, _s_b_only) =
        make_partial_contraction_inputs();

    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_contract_a, s_contract_b)],
        multiply_pairs: vec![(s_multiply_a, s_multiply_b)],
    };

    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"B".to_string(),
        ContractionOptions::new(ContractionMethod::Naive),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("same node"));
}

#[test]
fn test_partial_contract_rejects_duplicate_pair_usage() {
    let (tn_a, tn_b, s_contract_a, s_contract_b, s_multiply_a, s_multiply_b, _s_a_only, _s_b_only) =
        make_partial_contraction_inputs();

    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_contract_a.clone(), s_contract_b.clone())],
        multiply_pairs: vec![(s_contract_a, s_multiply_b), (s_multiply_a, s_contract_b)],
    };

    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"B".to_string(),
        ContractionOptions::default(),
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("appears in multiple pairs"));
}
