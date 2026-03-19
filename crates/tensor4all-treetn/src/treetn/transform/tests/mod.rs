
use super::*;
use crate::site_index_network::SiteIndexNetwork;
use std::collections::HashSet;
use tensor4all_core::{DynIndex, TensorDynLen};

fn make_two_node_chain() -> (TreeTN<TensorDynLen, String>, DynIndex, DynIndex) {
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(4);

    // Make a simple chain A --bond-- B
    let t0 = TensorDynLen::from_dense(vec![s0.clone(), bond.clone()], vec![1.0; 2 * 4]).unwrap();
    let t1 = TensorDynLen::from_dense(vec![bond.clone(), s1.clone()], vec![1.0; 4 * 3]).unwrap();

    tn.add_tensor("A".to_string(), t0).unwrap();
    tn.add_tensor("B".to_string(), t1).unwrap();

    let a = tn.node_index(&"A".to_string()).unwrap();
    let b = tn.node_index(&"B".to_string()).unwrap();
    tn.connect(a, &bond, b, &bond).unwrap();

    (tn, s0, s1)
}

#[test]
fn test_fuse_to_two_nodes_into_one() {
    let (tn, s0, s1) = make_two_node_chain();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("AB".to_string(), HashSet::from([s0.clone(), s1.clone()]))
        .unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 1);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(
        orig_full.distance(&fused_full) < 1e-12,
        "fused tensor should match original contraction"
    );
}

#[test]
fn test_split_to_one_node_into_two() {
    let (tn, s0, s1) = make_two_node_chain();

    // First fuse into a single node AB.
    let mut fuse_target = SiteIndexNetwork::<String, DynIndex>::new();
    fuse_target
        .add_node("AB".to_string(), HashSet::from([s0.clone(), s1.clone()]))
        .unwrap();
    let fused = tn.fuse_to(&fuse_target).unwrap();

    // Now split AB into A -- B.
    let mut split_target = SiteIndexNetwork::<String, DynIndex>::new();
    split_target
        .add_node("A".to_string(), HashSet::from([s0.clone()]))
        .unwrap();
    split_target
        .add_node("B".to_string(), HashSet::from([s1.clone()]))
        .unwrap();
    split_target
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    let split = fused
        .split_to(&split_target, &SplitOptions::default())
        .unwrap();
    assert_eq!(split.node_count(), 2);

    let fused_full = fused.contract_to_tensor().unwrap();
    let split_full = split.contract_to_tensor().unwrap();
    assert!(
        fused_full.distance(&split_full) < 1e-12,
        "split tensor should match fused tensor"
    );
}

#[test]
fn test_fuse_to_errors_for_unknown_site_index() {
    let (tn, _s0, _s1) = make_two_node_chain();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("X".to_string(), HashSet::from([DynIndex::new_dyn(2)]))
        .unwrap();

    assert!(tn.fuse_to(&target).is_err());
}
