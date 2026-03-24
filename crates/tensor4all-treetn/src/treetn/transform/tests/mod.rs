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

/// Create a 3-node chain: A -- B -- C
fn make_three_node_chain() -> (
    TreeTN<TensorDynLen, String>,
    DynIndex, // s0
    DynIndex, // s1
    DynIndex, // s2
) {
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(3);
    let bond_ab = DynIndex::new_dyn(2);
    let bond_bc = DynIndex::new_dyn(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), bond_ab.clone()], vec![1.0; 2 * 2]).unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![bond_ab.clone(), s1.clone(), bond_bc.clone()],
        vec![1.0; 2 * 2 * 2],
    )
    .unwrap();
    let t2 = TensorDynLen::from_dense(vec![bond_bc.clone(), s2.clone()], vec![1.0; 2 * 3]).unwrap();

    tn.add_tensor("A".to_string(), t0).unwrap();
    tn.add_tensor("B".to_string(), t1).unwrap();
    tn.add_tensor("C".to_string(), t2).unwrap();

    let a = tn.node_index(&"A".to_string()).unwrap();
    let b = tn.node_index(&"B".to_string()).unwrap();
    let c = tn.node_index(&"C".to_string()).unwrap();
    tn.connect(a, &bond_ab, b, &bond_ab).unwrap();
    tn.connect(b, &bond_bc, c, &bond_bc).unwrap();

    (tn, s0, s1, s2)
}

#[test]
fn test_fuse_to_three_nodes_pairwise() {
    let (tn, s0, s1, s2) = make_three_node_chain();

    // Fuse A+B into AB, keep C
    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("AB".to_string(), HashSet::from([s0.clone(), s1.clone()]))
        .unwrap();
    target
        .add_node("C".to_string(), HashSet::from([s2.clone()]))
        .unwrap();
    target
        .add_edge(&"AB".to_string(), &"C".to_string())
        .unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 2);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(orig_full.distance(&fused_full) < 1e-12);
}

#[test]
fn test_fuse_to_three_nodes_into_one() {
    let (tn, s0, s1, s2) = make_three_node_chain();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node(
            "ABC".to_string(),
            HashSet::from([s0.clone(), s1.clone(), s2.clone()]),
        )
        .unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 1);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(orig_full.distance(&fused_full) < 1e-12);
}

#[test]
fn test_fuse_to_missing_target_for_current_node() {
    let (tn, s0, _s1, _s2) = make_three_node_chain();

    // Target only accounts for node A, missing B and C
    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("X".to_string(), HashSet::from([s0.clone()]))
        .unwrap();

    let result = tn.fuse_to(&target);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("missing target") || msg.contains("no corresponding target"));
}

#[test]
fn test_split_to_with_actual_splitting() {
    let (tn, s0, s1, s2) = make_three_node_chain();

    // Fuse all into one node
    let mut fuse_target = SiteIndexNetwork::<String, DynIndex>::new();
    fuse_target
        .add_node(
            "ABC".to_string(),
            HashSet::from([s0.clone(), s1.clone(), s2.clone()]),
        )
        .unwrap();
    let fused = tn.fuse_to(&fuse_target).unwrap();
    assert_eq!(fused.node_count(), 1);

    // Split back into 3 nodes
    let mut split_target = SiteIndexNetwork::<String, DynIndex>::new();
    split_target
        .add_node("A".to_string(), HashSet::from([s0.clone()]))
        .unwrap();
    split_target
        .add_node("B".to_string(), HashSet::from([s1.clone()]))
        .unwrap();
    split_target
        .add_node("C".to_string(), HashSet::from([s2.clone()]))
        .unwrap();
    split_target
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();
    split_target
        .add_edge(&"B".to_string(), &"C".to_string())
        .unwrap();

    let split = fused
        .split_to(&split_target, &SplitOptions::default())
        .unwrap();
    assert_eq!(split.node_count(), 3);

    let fused_full = fused.contract_to_tensor().unwrap();
    let split_full = split.contract_to_tensor().unwrap();
    assert!(fused_full.distance(&split_full) < 1e-12);
}

#[test]
fn test_split_to_with_final_sweep() {
    let (tn, s0, s1) = make_two_node_chain();

    // Fuse into one node
    let mut fuse_target = SiteIndexNetwork::<String, DynIndex>::new();
    fuse_target
        .add_node("AB".to_string(), HashSet::from([s0.clone(), s1.clone()]))
        .unwrap();
    let fused = tn.fuse_to(&fuse_target).unwrap();

    // Split back with final_sweep enabled
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

    let options = SplitOptions {
        final_sweep: true,
        ..SplitOptions::default()
    };

    let split = fused.split_to(&split_target, &options).unwrap();
    assert_eq!(split.node_count(), 2);

    let fused_full = fused.contract_to_tensor().unwrap();
    let split_full = split.contract_to_tensor().unwrap();
    assert!(fused_full.distance(&split_full) < 1e-10);
}

#[test]
fn test_split_to_incompatible_target() {
    let (tn, _s0, _s1) = make_two_node_chain();

    // Target has a site index not in current TN
    let mut split_target = SiteIndexNetwork::<String, DynIndex>::new();
    split_target
        .add_node("X".to_string(), HashSet::from([DynIndex::new_dyn(7)]))
        .unwrap();
    split_target
        .add_node("Y".to_string(), HashSet::from([DynIndex::new_dyn(8)]))
        .unwrap();
    split_target
        .add_edge(&"X".to_string(), &"Y".to_string())
        .unwrap();

    let result = tn.split_to(&split_target, &SplitOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_fuse_identity_mapping() {
    // Fuse where each current node maps to exactly one target (identity mapping)
    let (tn, s0, s1) = make_two_node_chain();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("A".to_string(), HashSet::from([s0.clone()]))
        .unwrap();
    target
        .add_node("B".to_string(), HashSet::from([s1.clone()]))
        .unwrap();
    target.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 2);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(orig_full.distance(&fused_full) < 1e-12);
}
