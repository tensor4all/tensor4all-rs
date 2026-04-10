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

fn make_two_matching_treetns_different_site_ids(
) -> (TreeTN<TensorDynLen, String>, TreeTN<TensorDynLen, String>) {
    let s0_a = DynIndex::new_dyn(2);
    let bond_a = DynIndex::new_dyn(3);
    let s1_a = DynIndex::new_dyn(2);

    let t0_a = TensorDynLen::from_dense(
        vec![s0_a.clone(), bond_a.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let t1_a = TensorDynLen::from_dense(
        vec![bond_a.clone(), s1_a.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    .unwrap();
    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0_a, t1_a],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    let s0_b = DynIndex::new_dyn(2);
    let bond_b = DynIndex::new_dyn(2);
    let s1_b = DynIndex::new_dyn(2);

    let t0_b =
        TensorDynLen::from_dense(vec![s0_b.clone(), bond_b.clone()], vec![0.0, 1.0, 1.0, 0.0])
            .unwrap();
    let t1_b =
        TensorDynLen::from_dense(vec![bond_b.clone(), s1_b.clone()], vec![1.0, 0.0, 0.0, 1.0])
            .unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0_b, t1_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    (tn_a, tn_b)
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

/// Regression test for #353: add() failed for single-site TTs (no bond indices).
#[test]
fn test_add_single_site() {
    let s = DynIndex::new_dyn(3);

    let t_a = TensorDynLen::from_dense(vec![s.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    let t_b = TensorDynLen::from_dense(vec![s.clone()], vec![10.0, 20.0, 30.0]).unwrap();

    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
    let tn_b =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

    let result = tn_a.add(&tn_b).unwrap();
    assert_eq!(result.node_count(), 1);

    let dense = result.contract_to_tensor().unwrap();
    let expected = TensorDynLen::from_dense(vec![s.clone()], vec![11.0, 22.0, 33.0]).unwrap();
    assert!(
        dense.isapprox(&expected, 1e-10, 0.0),
        "single-site add failed: maxabs diff = {}",
        (&dense - &expected).maxabs()
    );
}

#[test]
fn test_reindex_site_space_like_matches_template_ids() {
    let (tn_a, tn_b) = make_two_matching_treetns_different_site_ids();

    let reindexed = tn_b.reindex_site_space_like(&tn_a).unwrap();
    assert!(reindexed.share_equivalent_site_index_network(&tn_a));
}

#[test]
fn test_add_aligned_accepts_equivalent_site_space_with_different_ids() {
    let (tn_a, tn_b) = make_two_matching_treetns_different_site_ids();
    let tn_b_aligned = tn_b.reindex_site_space_like(&tn_a).unwrap();

    assert!(!tn_a.share_equivalent_site_index_network(&tn_b));

    let sum = tn_a.add_aligned(&tn_b).unwrap();
    assert!(sum.share_equivalent_site_index_network(&tn_a));

    let dense_sum = sum.contract_to_tensor().unwrap();
    let dense_expected = tn_a
        .contract_to_tensor()
        .unwrap()
        .axpby(
            tensor4all_core::AnyScalar::new_real(1.0),
            &tn_b_aligned.contract_to_tensor().unwrap(),
            tensor4all_core::AnyScalar::new_real(1.0),
        )
        .unwrap();
    assert!(
        dense_sum.isapprox(&dense_expected, 1e-10, 0.0),
        "add_aligned failed: maxabs diff = {}",
        (&dense_sum - &dense_expected).maxabs()
    );
}
