//! Integration tests for site index swap (swap_site_indices).

use std::collections::HashMap;

use num_complex::Complex64;
use tensor4all_core::{IndexLike, StorageScalar, TensorDynLen, TensorLike};
use tensor4all_treetn::{SwapOptions, TreeTN};

// ============================================================================
// Generic helper functions
// ============================================================================

/// 2-node chain: A -- B with one site index each. Returns (tn, site_at_A, site_at_B).
fn two_node_chain<T: StorageScalar + From<f64>>() -> (
    TreeTN<TensorDynLen, String>,
    tensor4all_core::DynIndex,
    tensor4all_core::DynIndex,
) {
    let mut tn = TreeTN::<TensorDynLen, String>::new();
    let s0 = tensor4all_core::DynIndex::new_dyn(2);
    let s1 = tensor4all_core::DynIndex::new_dyn(2);
    let bond = tensor4all_core::DynIndex::new_dyn(3);
    let t0 = TensorDynLen::from_dense_data(
        vec![s0.clone(), bond.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(T::from)
            .collect(),
    );
    let t1 = TensorDynLen::from_dense_data(
        vec![bond.clone(), s1.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(T::from)
            .collect(),
    );
    tn.add_tensor("A".to_string(), t0).unwrap();
    tn.add_tensor("B".to_string(), t1).unwrap();
    let na = tn.node_index(&"A".to_string()).unwrap();
    let nb = tn.node_index(&"B".to_string()).unwrap();
    tn.connect(na, &bond, nb, &bond).unwrap();
    (tn, s0, s1)
}

/// 3-node chain: "0" -- "1" -- "2" with one site index each.
fn three_node_chain<T: StorageScalar + From<f64>>() -> (
    TreeTN<TensorDynLen, String>,
    tensor4all_core::DynIndex,
    tensor4all_core::DynIndex,
    tensor4all_core::DynIndex,
) {
    let mut tn = TreeTN::<TensorDynLen, String>::new();
    let s0 = tensor4all_core::DynIndex::new_dyn(2);
    let s1 = tensor4all_core::DynIndex::new_dyn(2);
    let s2 = tensor4all_core::DynIndex::new_dyn(2);
    let b01 = tensor4all_core::DynIndex::new_dyn(3);
    let b12 = tensor4all_core::DynIndex::new_dyn(3);
    let t0 = TensorDynLen::from_dense_data(
        vec![s0.clone(), b01.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(T::from)
            .collect(),
    );
    let t1 = TensorDynLen::from_dense_data(
        vec![b01.clone(), s1.clone(), b12.clone()],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0,
        ]
        .into_iter()
        .map(T::from)
        .collect(),
    );
    let t2 = TensorDynLen::from_dense_data(
        vec![b12.clone(), s2.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(T::from)
            .collect(),
    );
    tn.add_tensor("0".to_string(), t0).unwrap();
    tn.add_tensor("1".to_string(), t1).unwrap();
    tn.add_tensor("2".to_string(), t2).unwrap();
    let n0 = tn.node_index(&"0".to_string()).unwrap();
    let n1 = tn.node_index(&"1".to_string()).unwrap();
    let n2 = tn.node_index(&"2".to_string()).unwrap();
    tn.connect(n0, &b01, n1, &b01).unwrap();
    tn.connect(n1, &b12, n2, &b12).unwrap();
    (tn, s0, s1, s2)
}

/// 4-node chain: "0" -- "1" -- "2" -- "3" with one site index each.
fn four_node_chain<T: StorageScalar + From<f64>>() -> (
    TreeTN<TensorDynLen, String>,
    tensor4all_core::DynIndex,
    tensor4all_core::DynIndex,
    tensor4all_core::DynIndex,
    tensor4all_core::DynIndex,
) {
    let mut tn = TreeTN::<TensorDynLen, String>::new();
    let s0 = tensor4all_core::DynIndex::new_dyn(2);
    let s1 = tensor4all_core::DynIndex::new_dyn(2);
    let s2 = tensor4all_core::DynIndex::new_dyn(2);
    let s3 = tensor4all_core::DynIndex::new_dyn(2);
    let b01 = tensor4all_core::DynIndex::new_dyn(2);
    let b12 = tensor4all_core::DynIndex::new_dyn(2);
    let b23 = tensor4all_core::DynIndex::new_dyn(2);
    let t0 = TensorDynLen::from_dense_data(
        vec![s0.clone(), b01.clone()],
        vec![1.0, 2.0, 3.0, 4.0].into_iter().map(T::from).collect(),
    );
    let t1 = TensorDynLen::from_dense_data(
        vec![b01.clone(), s1.clone(), b12.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(T::from)
            .collect(),
    );
    let t2 = TensorDynLen::from_dense_data(
        vec![b12.clone(), s2.clone(), b23.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(T::from)
            .collect(),
    );
    let t3 = TensorDynLen::from_dense_data(
        vec![b23.clone(), s3.clone()],
        vec![1.0, 2.0, 3.0, 4.0].into_iter().map(T::from).collect(),
    );
    tn.add_tensor("0".to_string(), t0).unwrap();
    tn.add_tensor("1".to_string(), t1).unwrap();
    tn.add_tensor("2".to_string(), t2).unwrap();
    tn.add_tensor("3".to_string(), t3).unwrap();
    let n0 = tn.node_index(&"0".to_string()).unwrap();
    let n1 = tn.node_index(&"1".to_string()).unwrap();
    let n2 = tn.node_index(&"2".to_string()).unwrap();
    let n3 = tn.node_index(&"3".to_string()).unwrap();
    tn.connect(n0, &b01, n1, &b01).unwrap();
    tn.connect(n1, &b12, n2, &b12).unwrap();
    tn.connect(n2, &b23, n3, &b23).unwrap();
    (tn, s0, s1, s2, s3)
}

/// 4-node chain with sites x0,x1 at nodes "0","1" and y0,y1 at nodes "2","3". R=2.
fn chain_2r_interleave<T: StorageScalar + From<f64>>() -> (
    TreeTN<TensorDynLen, String>,
    tensor4all_core::DynIndex, // x0
    tensor4all_core::DynIndex, // x1
    tensor4all_core::DynIndex, // y0
    tensor4all_core::DynIndex, // y1
) {
    let mut tn = TreeTN::<TensorDynLen, String>::new();
    let x0 = tensor4all_core::DynIndex::new_dyn(2);
    let x1 = tensor4all_core::DynIndex::new_dyn(2);
    let y0 = tensor4all_core::DynIndex::new_dyn(2);
    let y1 = tensor4all_core::DynIndex::new_dyn(2);
    let b01 = tensor4all_core::DynIndex::new_dyn(2);
    let b12 = tensor4all_core::DynIndex::new_dyn(2);
    let b23 = tensor4all_core::DynIndex::new_dyn(2);
    let t0 = TensorDynLen::from_dense_data(
        vec![x0.clone(), b01.clone()],
        vec![1.0, 0.0, 0.0, 1.0].into_iter().map(T::from).collect(),
    );
    let t1 = TensorDynLen::from_dense_data(
        vec![b01.clone(), x1.clone(), b12.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            .into_iter()
            .map(T::from)
            .collect(),
    );
    let t2 = TensorDynLen::from_dense_data(
        vec![b12.clone(), y0.clone(), b23.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            .into_iter()
            .map(T::from)
            .collect(),
    );
    let t3 = TensorDynLen::from_dense_data(
        vec![b23.clone(), y1.clone()],
        vec![1.0, 0.0, 0.0, 1.0].into_iter().map(T::from).collect(),
    );
    tn.add_tensor("0".to_string(), t0).unwrap();
    tn.add_tensor("1".to_string(), t1).unwrap();
    tn.add_tensor("2".to_string(), t2).unwrap();
    tn.add_tensor("3".to_string(), t3).unwrap();
    let n0 = tn.node_index(&"0".to_string()).unwrap();
    let n1 = tn.node_index(&"1".to_string()).unwrap();
    let n2 = tn.node_index(&"2".to_string()).unwrap();
    let n3 = tn.node_index(&"3".to_string()).unwrap();
    tn.connect(n0, &b01, n1, &b01).unwrap();
    tn.connect(n1, &b12, n2, &b12).unwrap();
    tn.connect(n2, &b23, n3, &b23).unwrap();
    (tn, x0, x1, y0, y1)
}

/// Y-shape: center "C", leaves "L0", "L1", "L2". Each leaf has one site index.
fn y_shape_tree<T: StorageScalar + From<f64>>() -> (
    TreeTN<TensorDynLen, String>,
    tensor4all_core::DynIndex,
    tensor4all_core::DynIndex,
    tensor4all_core::DynIndex,
) {
    let mut tn = TreeTN::<TensorDynLen, String>::new();
    let s0 = tensor4all_core::DynIndex::new_dyn(2);
    let s1 = tensor4all_core::DynIndex::new_dyn(2);
    let s2 = tensor4all_core::DynIndex::new_dyn(2);
    let bc0 = tensor4all_core::DynIndex::new_dyn(2);
    let bc1 = tensor4all_core::DynIndex::new_dyn(2);
    let bc2 = tensor4all_core::DynIndex::new_dyn(2);
    let t_center = TensorDynLen::from_dense_data(
        vec![bc0.clone(), bc1.clone(), bc2.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            .into_iter()
            .map(T::from)
            .collect(),
    );
    let t0 = TensorDynLen::from_dense_data(
        vec![bc0.clone(), s0.clone()],
        vec![1.0, 0.0, 0.0, 1.0].into_iter().map(T::from).collect(),
    );
    let t1 = TensorDynLen::from_dense_data(
        vec![bc1.clone(), s1.clone()],
        vec![1.0, 0.0, 0.0, 1.0].into_iter().map(T::from).collect(),
    );
    let t2 = TensorDynLen::from_dense_data(
        vec![bc2.clone(), s2.clone()],
        vec![1.0, 0.0, 0.0, 1.0].into_iter().map(T::from).collect(),
    );
    tn.add_tensor("C".to_string(), t_center).unwrap();
    tn.add_tensor("L0".to_string(), t0).unwrap();
    tn.add_tensor("L1".to_string(), t1).unwrap();
    tn.add_tensor("L2".to_string(), t2).unwrap();
    let nc = tn.node_index(&"C".to_string()).unwrap();
    let n0 = tn.node_index(&"L0".to_string()).unwrap();
    let n1 = tn.node_index(&"L1".to_string()).unwrap();
    let n2 = tn.node_index(&"L2".to_string()).unwrap();
    tn.connect(nc, &bc0, n0, &bc0).unwrap();
    tn.connect(nc, &bc1, n1, &bc1).unwrap();
    tn.connect(nc, &bc2, n2, &bc2).unwrap();
    (tn, s0, s1, s2)
}

// ============================================================================
// Generic test functions
// ============================================================================

fn test_swap_two_node_chain_generic<T: StorageScalar + From<f64>>() {
    let (mut tn, s0, s1) = two_node_chain::<T>();
    let net = tn.site_index_network();
    assert_eq!(
        net.find_node_by_index_id(s0.id()).map(|n| n.as_str()),
        Some("A")
    );
    assert_eq!(
        net.find_node_by_index_id(s1.id()).map(|n| n.as_str()),
        Some("B")
    );

    let mut target = HashMap::new();
    target.insert(s0.id().to_owned(), "B".to_string());
    target.insert(s1.id().to_owned(), "A".to_string());

    tn.swap_site_indices(&target, &SwapOptions::default())
        .unwrap();

    let net = tn.site_index_network();
    assert_eq!(
        net.find_node_by_index_id(s0.id()).map(|n| n.as_str()),
        Some("B")
    );
    assert_eq!(
        net.find_node_by_index_id(s1.id()).map(|n| n.as_str()),
        Some("A")
    );
}

fn test_swap_three_node_chain_generic<T: StorageScalar + From<f64>>() {
    let (mut tn, s0, s1, s2) = three_node_chain::<T>();
    let net = tn.site_index_network();
    assert_eq!(
        net.find_node_by_index_id(s0.id()).map(|n| n.as_str()),
        Some("0")
    );
    assert_eq!(
        net.find_node_by_index_id(s1.id()).map(|n| n.as_str()),
        Some("1")
    );
    assert_eq!(
        net.find_node_by_index_id(s2.id()).map(|n| n.as_str()),
        Some("2")
    );

    // Swap so that s0->"2", s1->"0", s2->"1"
    let mut target = HashMap::new();
    target.insert(s0.id().to_owned(), "2".to_string());
    target.insert(s1.id().to_owned(), "0".to_string());
    target.insert(s2.id().to_owned(), "1".to_string());

    tn.swap_site_indices(&target, &SwapOptions::default())
        .unwrap();

    let net = tn.site_index_network();
    assert_eq!(
        net.find_node_by_index_id(s0.id()).map(|n| n.as_str()),
        Some("2")
    );
    assert_eq!(
        net.find_node_by_index_id(s1.id()).map(|n| n.as_str()),
        Some("0")
    );
    assert_eq!(
        net.find_node_by_index_id(s2.id()).map(|n| n.as_str()),
        Some("1")
    );
}

fn test_swap_four_node_chain_generic<T: StorageScalar + From<f64>>() {
    let (mut tn, s0, s1, s2, s3) = four_node_chain::<T>();
    // Swap adjacent pairs: (0,1) and (2,3) so s0<->s1, s2<->s3
    let mut target = HashMap::new();
    target.insert(s0.id().to_owned(), "1".to_string());
    target.insert(s1.id().to_owned(), "0".to_string());
    target.insert(s2.id().to_owned(), "3".to_string());
    target.insert(s3.id().to_owned(), "2".to_string());

    tn.swap_site_indices(&target, &SwapOptions::default())
        .unwrap();

    let net = tn.site_index_network();
    assert_eq!(
        net.find_node_by_index_id(s0.id()).map(|n| n.as_str()),
        Some("1")
    );
    assert_eq!(
        net.find_node_by_index_id(s1.id()).map(|n| n.as_str()),
        Some("0")
    );
    assert_eq!(
        net.find_node_by_index_id(s2.id()).map(|n| n.as_str()),
        Some("3")
    );
    assert_eq!(
        net.find_node_by_index_id(s3.id()).map(|n| n.as_str()),
        Some("2")
    );
}

fn test_swap_2r_interleave_generic<T: StorageScalar + From<f64>>() {
    let (mut tn, x0, x1, y0, y1) = chain_2r_interleave::<T>();
    let before = tn.contract_to_tensor().unwrap();

    // Target: x0 at "0", y0 at "1", x1 at "2", y1 at "3" (interleaved)
    let mut target = HashMap::new();
    target.insert(x0.id().to_owned(), "0".to_string());
    target.insert(y0.id().to_owned(), "1".to_string());
    target.insert(x1.id().to_owned(), "2".to_string());
    target.insert(y1.id().to_owned(), "3".to_string());

    tn.swap_site_indices(&target, &SwapOptions::default())
        .unwrap();

    let net = tn.site_index_network();
    assert_eq!(
        net.find_node_by_index_id(x0.id()).map(|n| n.as_str()),
        Some("0")
    );
    assert_eq!(
        net.find_node_by_index_id(y0.id()).map(|n| n.as_str()),
        Some("1")
    );
    assert_eq!(
        net.find_node_by_index_id(x1.id()).map(|n| n.as_str()),
        Some("2")
    );
    assert_eq!(
        net.find_node_by_index_id(y1.id()).map(|n| n.as_str()),
        Some("3")
    );

    let after = tn.contract_to_tensor().unwrap();
    assert!(
        before.isapprox(&after, 1e-10, 1e-10),
        "contract_to_tensor must match before and after swap"
    );
}

fn test_swap_y_shape_generic<T: StorageScalar + From<f64>>() {
    let (mut tn, s0, s1, s2) = y_shape_tree::<T>();
    // Swap s0 and s1 between L0 and L1 (path crosses center)
    let mut target = HashMap::new();
    target.insert(s0.id().to_owned(), "L1".to_string());
    target.insert(s1.id().to_owned(), "L0".to_string());

    tn.swap_site_indices(&target, &SwapOptions::default())
        .unwrap();

    let net = tn.site_index_network();
    assert_eq!(
        net.find_node_by_index_id(s0.id()).map(|n| n.as_str()),
        Some("L1")
    );
    assert_eq!(
        net.find_node_by_index_id(s1.id()).map(|n| n.as_str()),
        Some("L0")
    );
    assert_eq!(
        net.find_node_by_index_id(s2.id()).map(|n| n.as_str()),
        Some("L2")
    );
}

fn test_swap_correctness_contract_generic<T: StorageScalar + From<f64>>() {
    let (mut tn, s0, s1) = two_node_chain::<T>();
    let before = tn.contract_to_tensor().unwrap();

    let mut target = HashMap::new();
    target.insert(s0.id().to_owned(), "B".to_string());
    target.insert(s1.id().to_owned(), "A".to_string());
    tn.swap_site_indices(&target, &SwapOptions::default())
        .unwrap();

    let after = tn.contract_to_tensor().unwrap();
    assert!(
        before.isapprox(&after, 1e-10, 1e-10),
        "contract_to_tensor must match before and after swap"
    );
}

// ============================================================================
// f64 tests
// ============================================================================

#[test]
fn test_swap_two_node_chain() {
    test_swap_two_node_chain_generic::<f64>();
}

#[test]
fn test_swap_empty_target_no_op() {
    let (mut tn, _s0, _s1) = two_node_chain::<f64>();
    let target: HashMap<_, _> = HashMap::new();
    tn.swap_site_indices(&target, &SwapOptions::default())
        .unwrap();
}

#[test]
fn test_swap_partial() {
    let (mut tn, s0, s1) = two_node_chain::<f64>();
    // Only move s0 to B; s1 not in target. On a 2-node chain one factorize can only 1-1 split.
    let mut target = HashMap::new();
    target.insert(s0.id().to_owned(), "B".to_string());
    tn.swap_site_indices(&target, &SwapOptions::default())
        .unwrap();
    let net = tn.site_index_network();
    assert_eq!(
        net.find_node_by_index_id(s0.id()).map(|n| n.as_str()),
        Some("B")
    );
    assert_eq!(
        net.find_node_by_index_id(s1.id()).map(|n| n.as_str()),
        Some("A")
    );
}

#[test]
fn test_swap_three_node_chain() {
    test_swap_three_node_chain_generic::<f64>();
}

#[test]
fn test_swap_four_node_chain() {
    test_swap_four_node_chain_generic::<f64>();
}

#[test]
fn test_swap_2r_interleave() {
    test_swap_2r_interleave_generic::<f64>();
}

#[test]
#[ignore = "Y-shape swap requires separate implementation (see plan/swap-site-indices-perf.md)"]
fn test_swap_y_shape() {
    test_swap_y_shape_generic::<f64>();
}

#[test]
fn test_swap_correctness_contract() {
    test_swap_correctness_contract_generic::<f64>();
}

// ============================================================================
// Complex64 tests
// ============================================================================

#[test]
fn test_swap_two_node_chain_c64() {
    test_swap_two_node_chain_generic::<Complex64>();
}

#[test]
fn test_swap_three_node_chain_c64() {
    test_swap_three_node_chain_generic::<Complex64>();
}

#[test]
fn test_swap_four_node_chain_c64() {
    test_swap_four_node_chain_generic::<Complex64>();
}

#[test]
fn test_swap_2r_interleave_c64() {
    test_swap_2r_interleave_generic::<Complex64>();
}

#[test]
#[ignore = "Y-shape swap requires separate implementation (see plan/swap-site-indices-perf.md)"]
fn test_swap_y_shape_c64() {
    test_swap_y_shape_generic::<Complex64>();
}

#[test]
fn test_swap_correctness_contract_c64() {
    test_swap_correctness_contract_generic::<Complex64>();
}

// ============================================================================
// Invalid target: swap_site_indices returns Err
// ============================================================================

#[test]
fn test_swap_invalid_target_nonexistent_node() {
    let (mut tn, s0, _s1) = two_node_chain::<f64>();
    let mut target = HashMap::new();
    target.insert(s0.id().to_owned(), "Z".to_string()); // "Z" does not exist
    let result = tn.swap_site_indices(&target, &SwapOptions::default());
    assert!(
        result.is_err(),
        "swap_site_indices must fail for unknown node"
    );
}

#[test]
fn test_swap_invalid_target_unknown_index_id() {
    let (mut tn, _s0, _s1) = two_node_chain::<f64>();
    let unknown_id = tensor4all_core::DynIndex::new_dyn(2).id().to_owned();
    let mut target = HashMap::new();
    target.insert(unknown_id, "A".to_string());
    let result = tn.swap_site_indices(&target, &SwapOptions::default());
    assert!(
        result.is_err(),
        "swap_site_indices must fail for unknown index id"
    );
}
