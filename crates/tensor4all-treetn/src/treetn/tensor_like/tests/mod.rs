
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorIndex};

use crate::treetn::TreeTN;

/// Helper to create a simple 2-node TreeTN: A -- bond -- B
fn make_two_node_treetn() -> (
    TreeTN<TensorDynLen, String>,
    DynIndex, // s0
    DynIndex, // bond
    DynIndex, // s1
) {
    let s0 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(2);

    let t0 = TensorDynLen::from_dense(
        vec![s0.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![bond.clone(), s1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    .unwrap();

    let tn = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0, t1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    (tn, s0, bond, s1)
}

#[test]
fn test_external_indices() {
    let (tn, s0, bond, s1) = make_two_node_treetn();
    let ext = tn.external_indices();
    assert_eq!(ext.len(), 2);

    let ext_ids: Vec<_> = ext.iter().map(|i| *i.id()).collect();
    assert!(ext_ids.contains(s0.id()));
    assert!(ext_ids.contains(s1.id()));
    // Bond should NOT be in external indices
    assert!(!ext_ids.contains(bond.id()));
}

#[test]
fn test_num_external_indices() {
    let (tn, _s0, _bond, _s1) = make_two_node_treetn();
    assert_eq!(tn.num_external_indices(), 2);
}

#[test]
fn test_num_external_indices_single_node() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(4);
    let t = TensorDynLen::from_dense(vec![i.clone(), j.clone(), k.clone()], vec![0.0; 24]).unwrap();
    let tn = TreeTN::<TensorDynLen, String>::from_tensors(vec![t], vec!["A".to_string()]).unwrap();
    assert_eq!(tn.num_external_indices(), 3);
}

#[test]
fn test_replaceind_site_index() {
    let (tn, s0, _bond, s1) = make_two_node_treetn();

    let s0_new = DynIndex::new_dyn(2);
    let tn2 = tn.replaceind(&s0, &s0_new).unwrap();

    let ext_ids: Vec<_> = tn2.external_indices().iter().map(|i| *i.id()).collect();
    assert!(!ext_ids.contains(s0.id()));
    assert!(ext_ids.contains(s0_new.id()));
    assert!(ext_ids.contains(s1.id()));
}

#[test]
fn test_replaceind_link_index_via_sim_linkinds() {
    let (tn, s0, bond, s1) = make_two_node_treetn();

    // sim_linkinds replaces all link indices with fresh IDs
    let tn2 = tn.sim_linkinds().unwrap();

    // Site indices should remain unchanged
    let ext_ids: Vec<_> = tn2.external_indices().iter().map(|i| *i.id()).collect();
    assert!(ext_ids.contains(s0.id()));
    assert!(ext_ids.contains(s1.id()));

    // The bond index should have a different ID from the original
    let edge = tn2.graph.graph().edge_indices().next().unwrap();
    let new_bond = tn2.bond_index(edge).unwrap();
    assert_ne!(*new_bond.id(), *bond.id());
    // But same dimension
    assert_eq!(new_bond.dim(), bond.dim());
}

#[test]
fn test_replaceind_dimension_mismatch() {
    let (tn, s0, _bond, _s1) = make_two_node_treetn();

    let wrong_dim = DynIndex::new_dyn(5); // s0 has dim 2
    let result = tn.replaceind(&s0, &wrong_dim);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Index space mismatch"));
}

#[test]
fn test_replaceind_not_found() {
    let (tn, _s0, _bond, _s1) = make_two_node_treetn();

    let unknown = DynIndex::new_dyn(7);
    let new_idx = DynIndex::new_dyn(7);
    let result = tn.replaceind(&unknown, &new_idx);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn test_replaceinds_multiple() {
    let (tn, s0, _bond, s1) = make_two_node_treetn();

    let s0_new = DynIndex::new_dyn(2);
    let s1_new = DynIndex::new_dyn(2);

    let tn2 = tn
        .replaceinds(&[s0.clone(), s1.clone()], &[s0_new.clone(), s1_new.clone()])
        .unwrap();

    let ext_ids: Vec<_> = tn2.external_indices().iter().map(|i| *i.id()).collect();
    assert!(!ext_ids.contains(s0.id()));
    assert!(!ext_ids.contains(s1.id()));
    assert!(ext_ids.contains(s0_new.id()));
    assert!(ext_ids.contains(s1_new.id()));
}

#[test]
fn test_replaceinds_length_mismatch() {
    let (tn, s0, _bond, _s1) = make_two_node_treetn();

    let s0_new = DynIndex::new_dyn(2);
    let s1_new = DynIndex::new_dyn(2);

    let result = tn.replaceinds(std::slice::from_ref(&s0), &[s0_new.clone(), s1_new.clone()]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Length mismatch"));
}
