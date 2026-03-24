use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorIndex, TensorLike};

use crate::treetn::TreeTN;

/// Create two TreeTNs with the same topology for testing addition.
/// Both are 2-node chains: A -- bond -- B
fn make_two_matching_treetns() -> (
    TreeTN<TensorDynLen, String>,
    TreeTN<TensorDynLen, String>,
    DynIndex, // s0
    DynIndex, // s1
) {
    let s0 = DynIndex::new_dyn(2);
    let bond_a = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(2);

    let t0_a = TensorDynLen::from_dense(
        vec![s0.clone(), bond_a.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let t1_a = TensorDynLen::from_dense(
        vec![bond_a.clone(), s1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    .unwrap();

    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0_a, t1_a],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    let bond_b = DynIndex::new_dyn(2);

    let t0_b = TensorDynLen::from_dense(vec![s0.clone(), bond_b.clone()], vec![0.0, 1.0, 1.0, 0.0])
        .unwrap();
    let t1_b = TensorDynLen::from_dense(vec![bond_b.clone(), s1.clone()], vec![1.0, 0.0, 0.0, 1.0])
        .unwrap();

    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0_b, t1_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    (tn_a, tn_b, s0, s1)
}

#[test]
fn test_compute_merged_bond_indices() {
    let (tn_a, tn_b, _s0, _s1) = make_two_matching_treetns();

    let merged = tn_a.compute_merged_bond_indices(&tn_b).unwrap();
    assert_eq!(merged.len(), 1);

    // The key should be ("A", "B") in canonical order
    let key = ("A".to_string(), "B".to_string());
    let info = merged.get(&key).unwrap();
    assert_eq!(info.dim_a, 3); // tn_a bond dim
    assert_eq!(info.dim_b, 2); // tn_b bond dim
}

#[test]
fn test_add_basic() {
    let (tn_a, tn_b, s0, s1) = make_two_matching_treetns();

    let result = tn_a.add(&tn_b).unwrap();

    // Result should have same number of nodes
    assert_eq!(result.node_count(), 2);
    assert_eq!(result.edge_count(), 1);

    // Result should have the same site indices
    let ext_ids: Vec<_> = result.external_indices().iter().map(|i| *i.id()).collect();
    assert_eq!(ext_ids.len(), 2);
    assert!(ext_ids.contains(s0.id()));
    assert!(ext_ids.contains(s1.id()));

    // Bond dimension should be sum of original dimensions
    let edge = result.graph.graph().edge_indices().next().unwrap();
    let merged_bond = result.bond_index(edge).unwrap();
    assert_eq!(merged_bond.dim(), 3 + 2); // dim_a + dim_b
}

#[test]
fn test_add_verifies_with_contraction() {
    let (tn_a, tn_b, _s0, _s1) = make_two_matching_treetns();

    let result = tn_a.add(&tn_b).unwrap();

    // Contract all three and verify: contract(result) ≈ contract(tn_a) + contract(tn_b)
    let tensor_a = tn_a.contract_to_tensor().unwrap();
    let tensor_b = tn_b.contract_to_tensor().unwrap();
    let tensor_sum = result.contract_to_tensor().unwrap();

    // expected = tensor_a + tensor_b
    let expected = tensor_a
        .axpby(
            tensor4all_core::AnyScalar::new_real(1.0),
            &tensor_b,
            tensor4all_core::AnyScalar::new_real(1.0),
        )
        .unwrap();
    assert!(
        tensor_sum.isapprox(&expected, 1e-10, 0.0),
        "contract(result) != contract(tn_a) + contract(tn_b): maxabs diff = {}",
        (&tensor_sum - &expected).maxabs()
    );
}

#[test]
fn test_add_topology_mismatch() {
    let s0 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), bond.clone()], vec![1.0; 6]).unwrap();
    let t1 = TensorDynLen::from_dense(vec![bond.clone(), s1.clone()], vec![1.0; 6]).unwrap();
    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0, t1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Single-node TN (different topology)
    let t_single = TensorDynLen::from_dense(vec![s0.clone()], vec![1.0, 2.0]).unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(vec![t_single], vec!["X".to_string()])
        .unwrap();

    let result = tn_a.add(&tn_b);
    assert!(result.is_err());
}
