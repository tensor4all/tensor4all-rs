use super::*;
use crate::treetn::contraction::{ContractionMethod, ContractionOptions};
use tensor4all_core::{DynIndex, TensorDynLen};

struct PartialContractionInputs {
    tn_a: TreeTN<TensorDynLen, String>,
    tn_b: TreeTN<TensorDynLen, String>,
    s_contract_a: DynIndex,
    s_contract_b: DynIndex,
    s_multiply_a: DynIndex,
    s_multiply_b: DynIndex,
    s_a_only: DynIndex,
    s_b_only: DynIndex,
}

fn make_partial_contraction_inputs() -> PartialContractionInputs {
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

    PartialContractionInputs {
        tn_a,
        tn_b,
        s_contract_a,
        s_contract_b,
        s_multiply_a,
        s_multiply_b,
        s_a_only,
        s_b_only,
    }
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
    let PartialContractionInputs {
        tn_a,
        tn_b,
        s_contract_a,
        s_contract_b,
        s_multiply_a,
        s_multiply_b,
        ..
    } = make_partial_contraction_inputs();

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
    let PartialContractionInputs {
        tn_a,
        tn_b,
        s_contract_a,
        s_contract_b,
        s_multiply_a,
        s_multiply_b,
        ..
    } = make_partial_contraction_inputs();

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

#[test]
fn test_partial_contract_rejects_dimension_mismatch() {
    let PartialContractionInputs {
        tn_a,
        tn_b,
        s_contract_a,
        s_b_only,
        ..
    } = make_partial_contraction_inputs();

    // s_contract_a has dim 2, s_b_only has dim 4 → mismatch
    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_contract_a, s_b_only)],
        multiply_pairs: vec![],
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("dimension mismatch"));
}

#[test]
fn test_partial_contract_rejects_index_not_in_network() {
    let PartialContractionInputs { tn_a, tn_b, .. } = make_partial_contraction_inputs();

    let unknown = DynIndex::new_dyn(2);
    let spec = PartialContractionSpec {
        contract_pairs: vec![(unknown, DynIndex::new_dyn(2))],
        multiply_pairs: vec![],
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn test_partial_contract_contract_only() {
    // Contract s_contract pair, no multiply. Sites on different nodes.
    let s_a = DynIndex::new_dyn(3);
    let s_b = DynIndex::new_dyn(3);
    let extra_a = DynIndex::new_dyn(2);
    let extra_b = DynIndex::new_dyn(2);

    let t_a = TensorDynLen::from_dense(vec![s_a.clone(), extra_a.clone()], vec![1.0; 6]).unwrap();
    let t_b = TensorDynLen::from_dense(vec![s_b.clone(), extra_b.clone()], vec![2.0; 6]).unwrap();

    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
    let tn_b =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_a, s_b)],
        multiply_pairs: vec![],
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::new(ContractionMethod::Naive),
    )
    .unwrap();

    // s_a/s_b contracted away; extra_a and extra_b remain
    assert_eq!(result.external_indices().len(), 2);
}

#[test]
fn test_partial_contract_empty_spec() {
    // Empty spec: no contract, no multiply → full outer product
    let s_a = DynIndex::new_dyn(2);
    let s_b = DynIndex::new_dyn(3);

    let t_a = TensorDynLen::from_dense(vec![s_a.clone()], vec![1.0; 2]).unwrap();
    let t_b = TensorDynLen::from_dense(vec![s_b.clone()], vec![2.0; 3]).unwrap();

    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
    let tn_b =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

    let spec: PartialContractionSpec<DynIndex> = PartialContractionSpec {
        contract_pairs: vec![],
        multiply_pairs: vec![],
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::new(ContractionMethod::Naive),
    )
    .unwrap();

    // Both s_a and s_b remain as external
    assert_eq!(result.external_indices().len(), 2);
}

#[test]
fn test_partial_contract_rejects_same_node_in_second_network() {
    // Same as test_partial_contract_rejects_same_node_contract_and_multiply
    // but verifies the tn_b side check fires too.
    // tn_b has contract and multiply indices on the same node "A".
    let s_a1 = DynIndex::new_dyn(2);
    let s_a2 = DynIndex::new_dyn(2);
    let bond_a = DynIndex::new_dyn(2);

    // tn_a: node "A" has s_a1 only, node "B" has s_a2 only
    let t_a0 = TensorDynLen::from_dense(vec![s_a1.clone(), bond_a.clone()], vec![1.0; 4]).unwrap();
    let t_a1 = TensorDynLen::from_dense(vec![bond_a.clone(), s_a2.clone()], vec![1.0; 4]).unwrap();
    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_a0, t_a1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // tn_b: node "A" has s_b_contract AND s_b_multiply on same node
    let s_b_contract = DynIndex::new_dyn(2);
    let s_b_multiply = DynIndex::new_dyn(2);
    let bond_b = DynIndex::new_dyn(2);
    let s_b_only = DynIndex::new_dyn(3);

    let t_b0 = TensorDynLen::from_dense(
        vec![s_b_contract.clone(), s_b_multiply.clone(), bond_b.clone()],
        vec![2.0; 8],
    )
    .unwrap();
    let t_b1 =
        TensorDynLen::from_dense(vec![bond_b.clone(), s_b_only.clone()], vec![2.0; 6]).unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_b0, t_b1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // contract s_a1 ↔ s_b_contract (node "A" in both)
    // multiply s_a2 ↔ s_b_multiply (node "B" in tn_a, node "A" in tn_b) → tn_b same node!
    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_a1, s_b_contract)],
        multiply_pairs: vec![(s_a2, s_b_multiply)],
    };

    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("second network"));
}

#[test]
fn test_partial_contract_rejects_topology_mismatch() {
    // tn_a has 1 node, tn_b has 2 nodes → topology mismatch
    let s_a = DynIndex::new_dyn(2);
    let s_b = DynIndex::new_dyn(2);
    let bond_b = DynIndex::new_dyn(2);
    let s_b2 = DynIndex::new_dyn(3);

    let t_a = TensorDynLen::from_dense(vec![s_a.clone()], vec![1.0; 2]).unwrap();
    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();

    let t_b0 = TensorDynLen::from_dense(vec![s_b.clone(), bond_b.clone()], vec![2.0; 4]).unwrap();
    let t_b1 = TensorDynLen::from_dense(vec![bond_b.clone(), s_b2.clone()], vec![2.0; 6]).unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_b0, t_b1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    let spec: PartialContractionSpec<DynIndex> = PartialContractionSpec {
        contract_pairs: vec![],
        multiply_pairs: vec![],
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("incompatible topologies"));
}
