//! Basic tests for TreeTN
//!
//! In Einsum mode, tensors that share a bond must use the SAME Index (same ID).
//! When connecting node_a to node_b, both tensors must contain the same bond index.

use tensor4all_treetn::{TreeTN, TreeTopology};
use tensor4all::index::{DefaultIndex as Index, DynId};
use tensor4all::{TensorDynLen, Storage};
use tensor4all::NoSymmSpace;
use tensor4all::storage::{DenseStorageF64, DenseStorageC64};
use std::sync::Arc;
use std::collections::HashMap;
use petgraph::graph::NodeIndex;
use num_complex::Complex64;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a simple 2-node TreeTN with shared bond index.
/// Returns (tn, node1, node2, edge, physical1, bond, physical2)
fn create_two_node_treetn() -> (
    TreeTN<DynId, NoSymmSpace, NodeIndex>,
    NodeIndex, NodeIndex,
    petgraph::graph::EdgeIndex,
    Index<DynId>,
    Index<DynId>,
    Index<DynId>,
) {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let phys1 = Index::new_dyn(2);
    let bond = Index::new_dyn(3);
    let phys2 = Index::new_dyn(4);

    let tensor1 = TensorDynLen::new(
        vec![phys1.clone(), bond.clone()],
        vec![2, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6]))),
    );
    let node1 = tn.add_tensor_auto_name(tensor1);

    let tensor2 = TensorDynLen::new(
        vec![bond.clone(), phys2.clone()],
        vec![3, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12]))),
    );
    let node2 = tn.add_tensor_auto_name(tensor2);

    let edge = tn.connect(node1, &bond, node2, &bond).unwrap();

    (tn, node1, node2, edge, phys1, bond, phys2)
}

/// Create a 3-node chain: n1 -- n2 -- n3
fn create_three_node_chain() -> (
    TreeTN<DynId, NoSymmSpace, NodeIndex>,
    NodeIndex, NodeIndex, NodeIndex,
    petgraph::graph::EdgeIndex, petgraph::graph::EdgeIndex,
    Index<DynId>, // bond12
    Index<DynId>, // bond23
) {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let phys1 = Index::new_dyn(2);
    let bond12 = Index::new_dyn(3);
    let bond23 = Index::new_dyn(4);
    let phys3 = Index::new_dyn(5);

    let tensor1 = TensorDynLen::new(
        vec![phys1.clone(), bond12.clone()],
        vec![2, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6]))),
    );
    let node1 = tn.add_tensor_auto_name(tensor1);

    let tensor2 = TensorDynLen::new(
        vec![bond12.clone(), bond23.clone()],
        vec![3, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12]))),
    );
    let node2 = tn.add_tensor_auto_name(tensor2);

    let tensor3 = TensorDynLen::new(
        vec![bond23.clone(), phys3.clone()],
        vec![4, 5],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 20]))),
    );
    let node3 = tn.add_tensor_auto_name(tensor3);

    let edge12 = tn.connect(node1, &bond12, node2, &bond12).unwrap();
    let edge23 = tn.connect(node2, &bond23, node3, &bond23).unwrap();

    (tn, node1, node2, node3, edge12, edge23, bond12, bond23)
}

// ============================================================================
// Basic TreeTN Tests
// ============================================================================

#[test]
fn test_treetn_add_tensor() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::new(
        vec![i, j],
        vec![2, 3],
        Arc::new(Storage::new_dense_f64(6)),
    );

    let node = tn.add_tensor_auto_name(tensor);
    assert_eq!(tn.node_count(), 1);

    let retrieved = tn.tensor(node);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().dims, vec![2, 3]);
}

#[test]
fn test_treetn_replace_tensor() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::new(
        vec![i.clone(), j.clone()],
        vec![2, 3],
        Arc::new(Storage::new_dense_f64(6)),
    );
    let node = tn.add_tensor_auto_name(tensor);

    // Replace with a tensor that has the same indices
    let new_tensor = TensorDynLen::new(
        vec![i.clone(), j.clone()],
        vec![2, 3],
        Arc::new(Storage::new_dense_f64(6)),
    );
    let result = tn.replace_tensor(node, new_tensor);
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());
}

#[test]
fn test_treetn_replace_tensor_nonexistent() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::new(
        vec![i, j],
        vec![2, 3],
        Arc::new(Storage::new_dense_f64(6)),
    );

    let invalid_node = NodeIndex::new(999);
    let result = tn.replace_tensor(invalid_node, tensor);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_treetn_replace_tensor_missing_bond_index() {
    let (mut tn, node1, _node2, _edge, _phys1, bond, _phys2) = create_two_node_treetn();

    // Try to replace node1 with a tensor that doesn't have the bond index
    let new_i = Index::new_dyn(5);
    let new_j = Index::new_dyn(6);
    let new_tensor = TensorDynLen::new(
        vec![new_i, new_j], // bond is missing!
        vec![5, 6],
        Arc::new(Storage::new_dense_f64(30)),
    );

    let result = tn.replace_tensor(node1, new_tensor);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("missing") || err_msg.contains("index") || err_msg.contains("indices"));
}

#[test]
fn test_treetn_replace_tensor_with_bond() {
    let (mut tn, node1, _node2, _edge, _phys1, bond, _phys2) = create_two_node_treetn();

    // Replace node1 with a tensor that has the same bond index
    let new_i = Index::new_dyn(5);
    let new_tensor = TensorDynLen::new(
        vec![new_i, bond.clone()],
        vec![5, 3],
        Arc::new(Storage::new_dense_f64(15)),
    );

    let result = tn.replace_tensor(node1, new_tensor);
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());
}

// ============================================================================
// Connection Tests
// ============================================================================

#[test]
fn test_treetn_connect() {
    let (tn, _node1, _node2, edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    assert_eq!(tn.edge_count(), 1);
    let bond_idx = tn.bond_index(edge);
    assert!(bond_idx.is_some());
    assert_eq!(bond_idx.unwrap().size(), 3);
}

#[test]
fn test_treetn_connect_id_mismatch() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let tensor1 = TensorDynLen::new(
        vec![i1.clone(), j1.clone()],
        vec![2, 3],
        Arc::new(Storage::new_dense_f64(6)),
    );
    let node1 = tn.add_tensor_auto_name(tensor1);

    let i2 = Index::new_dyn(3); // Different ID than j1
    let k2 = Index::new_dyn(4);
    let tensor2 = TensorDynLen::new(
        vec![i2.clone(), k2.clone()],
        vec![3, 4],
        Arc::new(Storage::new_dense_f64(12)),
    );
    let node2 = tn.add_tensor_auto_name(tensor2);

    // Try to connect with different index IDs - should fail in Einsum mode
    let result = tn.connect(node1, &j1, node2, &i2);
    assert!(result.is_err());
    // Use Debug format to get full error chain, or check root cause
    let err = result.unwrap_err();
    let err_chain = format!("{:?}", err);
    assert!(
        err_chain.contains("ID") || err_chain.contains("Einsum") || err_chain.contains("match"),
        "Error chain should mention ID mismatch: {}", err_chain
    );
}

#[test]
fn test_treetn_connect_invalid_node() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::new(
        vec![i.clone(), j.clone()],
        vec![2, 3],
        Arc::new(Storage::new_dense_f64(6)),
    );
    let node1 = tn.add_tensor_auto_name(tensor);

    let invalid_node = NodeIndex::new(999);
    let result = tn.connect(node1, &j, invalid_node, &j);
    assert!(result.is_err());
}

#[test]
fn test_treetn_connect_index_not_in_tensor() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let tensor1 = TensorDynLen::new(
        vec![i1.clone(), j1.clone()],
        vec![2, 3],
        Arc::new(Storage::new_dense_f64(6)),
    );
    let node1 = tn.add_tensor_auto_name(tensor1);

    let bond = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let tensor2 = TensorDynLen::new(
        vec![bond.clone(), k2.clone()],
        vec![3, 4],
        Arc::new(Storage::new_dense_f64(12)),
    );
    let node2 = tn.add_tensor_auto_name(tensor2);

    // Try to connect with an index that doesn't exist in tensor1
    let fake_index = Index::new_dyn(3);
    let result = tn.connect(node1, &fake_index, node2, &bond);
    assert!(result.is_err());
}

// ============================================================================
// Bond Index Tests
// ============================================================================

#[test]
fn test_treetn_bond_index() {
    let (tn, _node1, _node2, edge, _phys1, bond, _phys2) = create_two_node_treetn();

    let bond_idx = tn.bond_index(edge).unwrap();
    assert_eq!(bond_idx.id, bond.id);
    assert_eq!(bond_idx.size(), 3);
}

#[test]
fn test_treetn_replace_edge_bond() {
    let (mut tn, _node1, _node2, edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    assert_eq!(tn.bond_index(edge).unwrap().size(), 3);

    // Replace with new bond index
    let new_idx = Index::new_dyn(5);
    let result = tn.replace_edge_bond(edge, new_idx);
    assert!(result.is_ok());
    assert_eq!(tn.bond_index(edge).unwrap().size(), 5);
}

// ============================================================================
// Ortho Towards Tests
// ============================================================================

#[test]
fn test_treetn_set_edge_ortho_towards() {
    let (mut tn, node1, node2, edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    // Set ortho direction to node1
    let result = tn.set_edge_ortho_towards(edge, Some(node1));
    assert!(result.is_ok());
    assert_eq!(tn.ortho_towards_node(edge), Some(&node1));

    // Set ortho direction to node2
    let result = tn.set_edge_ortho_towards(edge, Some(node2));
    assert!(result.is_ok());
    assert_eq!(tn.ortho_towards_node(edge), Some(&node2));

    // Clear ortho direction
    let result = tn.set_edge_ortho_towards(edge, None);
    assert!(result.is_ok());
    assert!(tn.ortho_towards_node(edge).is_none());
}

#[test]
fn test_treetn_set_edge_ortho_towards_invalid() {
    let (mut tn, node1, node2, edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    // Create a third node that is not connected to this edge
    let i3 = Index::new_dyn(5);
    let k3 = Index::new_dyn(6);
    let tensor3 = TensorDynLen::new(
        vec![i3, k3],
        vec![5, 6],
        Arc::new(Storage::new_dense_f64(30)),
    );
    let node3 = tn.add_tensor_auto_name(tensor3);

    // Try to set ortho direction to a node that's not an endpoint
    let result = tn.set_edge_ortho_towards(edge, Some(node3));
    assert!(result.is_err());
}

// ============================================================================
// Tree Validation Tests
// ============================================================================

#[test]
fn test_treetn_validate_tree_simple() {
    let (tn, _node1, _node2, _edge, _phys1, _bond, _phys2) = create_two_node_treetn();
    assert!(tn.validate_tree().is_ok());
}

#[test]
fn test_treetn_validate_tree_three_nodes() {
    let (tn, _n1, _n2, _n3, _e12, _e23, _b12, _b23) = create_three_node_chain();
    assert!(tn.validate_tree().is_ok());
}

#[test]
fn test_treetn_validate_tree_cycle() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    // Create a cycle: n1 -- n2 -- n3 -- n1
    let bond12 = Index::new_dyn(3);
    let bond23 = Index::new_dyn(4);
    let bond31 = Index::new_dyn(5);

    let tensor1 = TensorDynLen::new(
        vec![bond12.clone(), bond31.clone()],
        vec![3, 5],
        Arc::new(Storage::new_dense_f64(15)),
    );
    let node1 = tn.add_tensor_auto_name(tensor1);

    let tensor2 = TensorDynLen::new(
        vec![bond12.clone(), bond23.clone()],
        vec![3, 4],
        Arc::new(Storage::new_dense_f64(12)),
    );
    let node2 = tn.add_tensor_auto_name(tensor2);

    let tensor3 = TensorDynLen::new(
        vec![bond23.clone(), bond31.clone()],
        vec![4, 5],
        Arc::new(Storage::new_dense_f64(20)),
    );
    let node3 = tn.add_tensor_auto_name(tensor3);

    tn.connect(node1, &bond12, node2, &bond12).unwrap();
    tn.connect(node2, &bond23, node3, &bond23).unwrap();
    tn.connect(node3, &bond31, node1, &bond31).unwrap();

    // Should fail: 3 nodes, 3 edges (cycle)
    assert!(tn.validate_tree().is_err());
}

#[test]
fn test_treetn_validate_tree_disconnected() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    // Create two disconnected nodes
    let i1 = Index::new_dyn(2);
    let tensor1 = TensorDynLen::new(
        vec![i1],
        vec![2],
        Arc::new(Storage::new_dense_f64(2)),
    );
    let _node1 = tn.add_tensor_auto_name(tensor1);

    let i2 = Index::new_dyn(3);
    let tensor2 = TensorDynLen::new(
        vec![i2],
        vec![3],
        Arc::new(Storage::new_dense_f64(3)),
    );
    let _node2 = tn.add_tensor_auto_name(tensor2);

    // Should fail: not connected
    assert!(tn.validate_tree().is_err());
}

#[test]
fn test_treetn_validate_tree_empty() {
    let tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    assert!(tn.validate_tree().is_ok());
}

// ============================================================================
// Ortho Region Tests
// ============================================================================

#[test]
fn test_canonical_center_empty() {
    let tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    assert!(!tn.is_canonicalized());
    assert!(tn.canonical_center().is_empty());
}

#[test]
fn test_set_canonical_center() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let tensor = TensorDynLen::new(
        vec![i],
        vec![2],
        Arc::new(Storage::new_dense_f64(2)),
    );
    let node = tn.add_tensor_auto_name(tensor);

    assert!(!tn.is_canonicalized());

    let result = tn.set_canonical_center(vec![node]);
    assert!(result.is_ok());
    assert!(tn.is_canonicalized());
    assert!(tn.canonical_center().contains(&node));
}

#[test]
fn test_set_canonical_center_invalid_node() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let invalid_node = NodeIndex::new(999);
    let result = tn.set_canonical_center(vec![invalid_node]);
    assert!(result.is_err());
}

#[test]
fn test_clear_canonical_center() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let tensor = TensorDynLen::new(
        vec![i],
        vec![2],
        Arc::new(Storage::new_dense_f64(2)),
    );
    let node = tn.add_tensor_auto_name(tensor);

    tn.set_canonical_center(vec![node]).unwrap();
    assert!(tn.is_canonicalized());

    tn.clear_canonical_center();
    assert!(!tn.is_canonicalized());
    assert!(tn.canonical_center().is_empty());
}

// ============================================================================
// Validate Ortho Consistency Tests
// ============================================================================

#[test]
fn test_validate_ortho_consistency_disconnected_centers() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    // Create two disconnected nodes
    let i1 = Index::new_dyn(2);
    let tensor1 = TensorDynLen::new(vec![i1], vec![2], Arc::new(Storage::new_dense_f64(2)));
    let n1 = tn.add_tensor_auto_name(tensor1);

    let i2 = Index::new_dyn(3);
    let tensor2 = TensorDynLen::new(vec![i2], vec![3], Arc::new(Storage::new_dense_f64(3)));
    let n2 = tn.add_tensor_auto_name(tensor2);

    // Two centers that are not connected should fail
    tn.set_canonical_center(vec![n1, n2]).unwrap();
    assert!(tn.validate_ortho_consistency().is_err());
}

#[test]
fn test_validate_ortho_consistency_none_only_inside_centers() {
    let (mut tn, _node1, node2, edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    // Only node2 is center
    tn.set_canonical_center(vec![node2]).unwrap();

    // Clear ortho_towards on a boundary edge (should be forbidden)
    tn.set_edge_ortho_towards(edge, None).unwrap();
    assert!(tn.validate_ortho_consistency().is_err());
}

#[test]
fn test_validate_ortho_consistency_chain_pointing_towards_center() {
    let (mut tn, _n1, n2, _n3, e12, e23, _b12, _b23) = create_three_node_chain();

    // n2 is center
    tn.set_canonical_center(vec![n2]).unwrap();

    // Both boundary edges must point into center
    tn.set_edge_ortho_towards(e12, Some(n2)).unwrap();
    tn.set_edge_ortho_towards(e23, Some(n2)).unwrap();

    assert!(tn.validate_ortho_consistency().is_ok());
}

// ============================================================================
// Canonicalization Tests
// ============================================================================

#[test]
fn test_canonicalize_simple() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let phys1 = Index::new_dyn(2);
    let bond = Index::new_dyn(3);
    let phys2 = Index::new_dyn(4);

    let data1: Vec<f64> = vec![1.0; 6];
    let tensor1 = TensorDynLen::new(
        vec![phys1.clone(), bond.clone()],
        vec![2, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data1))),
    );
    let n1 = tn.add_tensor_auto_name(tensor1);

    let data2: Vec<f64> = vec![1.0; 12];
    let tensor2 = TensorDynLen::new(
        vec![bond.clone(), phys2.clone()],
        vec![3, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data2))),
    );
    let n2 = tn.add_tensor_auto_name(tensor2);

    tn.connect(n1, &bond, n2, &bond).unwrap();

    // Canonicalize towards n2
    let tn_canon = tn.canonicalize(vec![n2]).unwrap();

    assert!(tn_canon.is_canonicalized());
    assert!(tn_canon.canonical_center().contains(&n2));
    assert!(tn_canon.validate_ortho_consistency().is_ok());
}

// ============================================================================
// Contraction Tests
// ============================================================================

#[test]
fn test_contract_two_nodes() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let bond = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let data1: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let tensor1 = TensorDynLen::new(
        vec![i.clone(), bond.clone()],
        vec![2, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data1))),
    );
    let n1 = tn.add_tensor_auto_name(tensor1);

    let data2: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor2 = TensorDynLen::new(
        vec![bond.clone(), k.clone()],
        vec![3, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data2))),
    );
    let n2 = tn.add_tensor_auto_name(tensor2);

    tn.connect(n1, &bond, n2, &bond).unwrap();

    let result = tn.contract_to_tensor().unwrap();

    // Result should have physical indices [i, k] with dims [2, 4]
    assert_eq!(result.dims, vec![2, 4]);
    assert_eq!(result.indices.len(), 2);
}

#[test]
fn test_contract_chain() {
    let (tn, _n1, _n2, _n3, _e12, _e23, _b12, _b23) = create_three_node_chain();

    let result = tn.contract_to_tensor().unwrap();

    // Result should have physical indices from n1 and n3
    assert_eq!(result.dims, vec![2, 5]);
}

// ============================================================================
// Log Norm Tests
// ============================================================================

#[test]
fn test_log_norm_simple() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let bond = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let data1: Vec<f64> = vec![1.0; 6];
    let tensor1 = TensorDynLen::new(
        vec![i.clone(), bond.clone()],
        vec![2, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data1))),
    );
    let n1 = tn.add_tensor_auto_name(tensor1);

    let data2: Vec<f64> = vec![1.0; 12];
    let tensor2 = TensorDynLen::new(
        vec![bond.clone(), k.clone()],
        vec![3, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data2))),
    );
    let n2 = tn.add_tensor_auto_name(tensor2);

    tn.connect(n1, &bond, n2, &bond).unwrap();

    let log_norm = tn.log_norm().unwrap();
    assert!(log_norm.is_finite());
}

// ============================================================================
// TreeTN Add Tests
// ============================================================================

#[test]
fn test_treetn_add_single_node() {
    let mut tn_a = TreeTN::<DynId, NoSymmSpace, usize>::new();
    let mut tn_b = TreeTN::<DynId, NoSymmSpace, usize>::new();

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let data_a: Vec<f64> = (1..=6).map(|x| x as f64).collect();
    let tensor_a = TensorDynLen::new(
        vec![i.clone(), j.clone()],
        vec![2, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a))),
    );
    tn_a.add_tensor(0, tensor_a).unwrap();

    let data_b: Vec<f64> = (1..=6).map(|x| (x * 10) as f64).collect();
    let tensor_b = TensorDynLen::new(
        vec![i.clone(), j.clone()],
        vec![2, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b))),
    );
    tn_b.add_tensor(0, tensor_b).unwrap();

    let tn_sum = tn_a.add(tn_b).unwrap();
    assert_eq!(tn_sum.node_count(), 1);
}

#[test]
fn test_treetn_add_two_nodes() {
    let mut tn_a = TreeTN::<DynId, NoSymmSpace, usize>::new();
    let mut tn_b = TreeTN::<DynId, NoSymmSpace, usize>::new();

    let i = Index::new_dyn(2);
    let bond = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    // Network A
    let tensor_a0 = TensorDynLen::new(
        vec![i.clone(), bond.clone()],
        vec![2, 3],
        Arc::new(Storage::new_dense_f64(6)),
    );
    let tensor_a1 = TensorDynLen::new(
        vec![bond.clone(), k.clone()],
        vec![3, 4],
        Arc::new(Storage::new_dense_f64(12)),
    );
    tn_a.add_tensor(0, tensor_a0).unwrap();
    tn_a.add_tensor(1, tensor_a1).unwrap();
    let node_a0 = tn_a.node_index(&0).unwrap();
    let node_a1 = tn_a.node_index(&1).unwrap();
    tn_a.connect(node_a0, &bond, node_a1, &bond).unwrap();

    // Network B with bond dimension 5
    let bond_b = Index::new_dyn(5);
    let tensor_b0 = TensorDynLen::new(
        vec![i.clone(), bond_b.clone()],
        vec![2, 5],
        Arc::new(Storage::new_dense_f64(10)),
    );
    let tensor_b1 = TensorDynLen::new(
        vec![bond_b.clone(), k.clone()],
        vec![5, 4],
        Arc::new(Storage::new_dense_f64(20)),
    );
    tn_b.add_tensor(0, tensor_b0).unwrap();
    tn_b.add_tensor(1, tensor_b1).unwrap();
    let node_b0 = tn_b.node_index(&0).unwrap();
    let node_b1 = tn_b.node_index(&1).unwrap();
    tn_b.connect(node_b0, &bond_b, node_b1, &bond_b).unwrap();

    let tn_sum = tn_a.add(tn_b).unwrap();
    assert_eq!(tn_sum.node_count(), 2);
    assert_eq!(tn_sum.edge_count(), 1);

    // Bond dimension should be 3 + 5 = 8
    let edge = tn_sum.edges_for_node(tn_sum.node_indices()[0])[0].0;
    assert_eq!(tn_sum.bond_index(edge).unwrap().size(), 8);
}

// ============================================================================
// Truncation Tests
// ============================================================================

#[test]
fn test_truncate_simple() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let bond = Index::new_dyn(10); // Large bond
    let k = Index::new_dyn(4);

    // Create tensors with rank-deficient data (effectively rank 2)
    // tensor1[i, bond] = data[i * 10 + bond], only bond=0,1 are nonzero
    let mut data1 = vec![0.0f64; 20];
    for idx in 0..2 {
        data1[idx * 10] = 1.0;
        data1[idx * 10 + 1] = 0.5;
    }
    let tensor1 = TensorDynLen::new(
        vec![i.clone(), bond.clone()],
        vec![2, 10],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data1))),
    );
    let n1 = tn.add_tensor_auto_name(tensor1);

    // tensor2[bond, k] similar structure
    let mut data2 = vec![0.0f64; 40];
    for k_idx in 0..4 {
        data2[0 * 4 + k_idx] = 1.0;
        data2[1 * 4 + k_idx] = 0.3;
    }
    let tensor2 = TensorDynLen::new(
        vec![bond.clone(), k.clone()],
        vec![10, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data2))),
    );
    let n2 = tn.add_tensor_auto_name(tensor2);

    let edge = tn.connect(n1, &bond, n2, &bond).unwrap();

    // Original bond dimension is 10
    assert_eq!(tn.bond_index(edge).unwrap().size(), 10);

    // Truncate with max_rank = 3
    let truncated = tn.truncate(std::iter::once(n2), None, Some(3)).unwrap();

    // Bond should now be at most 3
    let new_edge = truncated.edges_for_node(n1)[0].0;
    assert!(truncated.bond_index(new_edge).unwrap().size() <= 3);
}

#[test]
fn test_truncate_mut_simple() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let bond = Index::new_dyn(8);
    let k = Index::new_dyn(4);

    let data1: Vec<f64> = (0..16).map(|x| x as f64).collect();
    let tensor1 = TensorDynLen::new(
        vec![i.clone(), bond.clone()],
        vec![2, 8],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data1))),
    );
    let n1 = tn.add_tensor_auto_name(tensor1);

    let data2: Vec<f64> = (0..32).map(|x| x as f64).collect();
    let tensor2 = TensorDynLen::new(
        vec![bond.clone(), k.clone()],
        vec![8, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data2))),
    );
    let n2 = tn.add_tensor_auto_name(tensor2);

    let edge = tn.connect(n1, &bond, n2, &bond).unwrap();

    // Original bond dimension
    assert_eq!(tn.bond_index(edge).unwrap().size(), 8);

    // Truncate in-place with max_rank = 4
    tn.truncate_mut(std::iter::once(n2), None, Some(4)).unwrap();

    // Bond should now be at most 4
    let new_edge = tn.edges_for_node(n1)[0].0;
    assert!(tn.bond_index(new_edge).unwrap().size() <= 4);
}

#[test]
fn test_truncate_three_node_chain() {
    let (tn, n1, _n2, n3, e12, e23, _b12, _b23) = create_three_node_chain();

    // Original bond dimensions
    assert_eq!(tn.bond_index(e12).unwrap().size(), 3);
    assert_eq!(tn.bond_index(e23).unwrap().size(), 4);

    // Truncate towards center (n2) with max_rank = 2
    let truncated = tn.truncate(std::iter::once(n1), None, Some(2)).unwrap();

    // All bonds should now be at most 2
    for (edge, _) in truncated.edges_for_node(n1) {
        assert!(truncated.bond_index(edge).unwrap().size() <= 2);
    }
    for (edge, _) in truncated.edges_for_node(n3) {
        assert!(truncated.bond_index(edge).unwrap().size() <= 2);
    }
}

#[test]
fn test_truncate_with_rtol() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let bond = Index::new_dyn(5);
    let k = Index::new_dyn(3);

    // Create a low-rank tensor (rank ~1)
    let mut data1 = vec![0.0f64; 10];
    data1[0] = 1.0;
    data1[5] = 1.0;
    let tensor1 = TensorDynLen::new(
        vec![i.clone(), bond.clone()],
        vec![2, 5],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data1))),
    );
    let n1 = tn.add_tensor_auto_name(tensor1);

    let mut data2 = vec![0.0f64; 15];
    data2[0] = 1.0;
    data2[1] = 1.0;
    data2[2] = 1.0;
    let tensor2 = TensorDynLen::new(
        vec![bond.clone(), k.clone()],
        vec![5, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data2))),
    );
    let n2 = tn.add_tensor_auto_name(tensor2);

    tn.connect(n1, &bond, n2, &bond).unwrap();

    // Truncate with rtol - should reduce rank significantly for low-rank data
    let truncated = tn.truncate(std::iter::once(n2), Some(1e-10), None).unwrap();

    // Bond dimension should be reduced
    let new_edge = truncated.edges_for_node(n1)[0].0;
    let new_bond_size = truncated.bond_index(new_edge).unwrap().size();
    // The effective rank should be 1 or 2
    assert!(new_bond_size <= 2, "Expected bond_size <= 2, got {}", new_bond_size);
}

// ============================================================================
// sim_internal_inds Tests
// ============================================================================

#[test]
fn test_sim_internal_inds() {
    let (tn, node1, node2, _edge, phys1, bond, phys2) = create_two_node_treetn();

    // Get original bond ID
    let original_bond_id = bond.id;
    let original_bond_size = bond.size();

    // Create a copy with simulated internal indices
    let tn_sim = tn.sim_internal_inds();

    // Check that structure is preserved
    assert_eq!(tn_sim.node_count(), 2);
    assert_eq!(tn_sim.edge_count(), 1);

    // Check that bond index has new ID but same size
    let new_edge = tn_sim.edges_for_node(node1)[0].0;
    let new_bond = tn_sim.bond_index(new_edge).unwrap();
    assert_eq!(new_bond.size(), original_bond_size);
    assert_ne!(new_bond.id, original_bond_id, "Bond ID should be different after sim_internal_inds");

    // Check that physical indices are unchanged
    let tensor1 = tn_sim.tensor(node1).unwrap();
    let tensor2 = tn_sim.tensor(node2).unwrap();

    // Find physical indices (those with size 2 and 4)
    let has_phys1 = tensor1.indices.iter().any(|idx| idx.id == phys1.id);
    let has_phys2 = tensor2.indices.iter().any(|idx| idx.id == phys2.id);
    assert!(has_phys1, "Physical index 1 should be preserved");
    assert!(has_phys2, "Physical index 2 should be preserved");

    // Check that tensors contain the new bond index
    let has_new_bond_in_t1 = tensor1.indices.iter().any(|idx| idx.id == new_bond.id);
    let has_new_bond_in_t2 = tensor2.indices.iter().any(|idx| idx.id == new_bond.id);
    assert!(has_new_bond_in_t1, "Tensor 1 should have new bond index");
    assert!(has_new_bond_in_t2, "Tensor 2 should have new bond index");

    // Original tensors should still have old bond
    let orig_tensor1 = tn.tensor(node1).unwrap();
    let has_old_bond = orig_tensor1.indices.iter().any(|idx| idx.id == original_bond_id);
    assert!(has_old_bond, "Original tensor should still have original bond");
}

// ============================================================================
// DynTreeTN Tests (in dyn_treetn module)
// ============================================================================

// DynTreeTN tests are in the dyn_treetn.rs file as unit tests.

// ============================================================================
// contract_zipup tests
// ============================================================================

// Note: contract_zipup is a complex algorithm that needs more testing.
// The current implementation has known limitations with cases where
// all site indices are contracted (inner product case).
// These tests verify basic functionality.

#[test]
fn test_contract_zipup_basic_api() {
    use tensor4all_treetn::{SiteIndexNetwork, random_treetn_f64, LinkSpace};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashSet;

    // Create two networks with the same topology but different site indices
    let mut site_network1 = SiteIndexNetwork::<String, DynId>::new();
    let site_a1 = Index::new_dyn(2);
    let site_b1 = Index::new_dyn(3);
    site_network1.add_node("A".to_string(), HashSet::from([site_a1.clone()])).unwrap();
    site_network1.add_node("B".to_string(), HashSet::from([site_b1.clone()])).unwrap();
    site_network1.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    let mut site_network2 = SiteIndexNetwork::<String, DynId>::new();
    let site_a2 = Index::new_dyn(2);
    let site_b2 = Index::new_dyn(3);
    site_network2.add_node("A".to_string(), HashSet::from([site_a2.clone()])).unwrap();
    site_network2.add_node("B".to_string(), HashSet::from([site_b2.clone()])).unwrap();
    site_network2.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    let mut rng1 = ChaCha8Rng::seed_from_u64(42);
    let mut rng2 = ChaCha8Rng::seed_from_u64(123);
    let tn1 = random_treetn_f64(&mut rng1, &site_network1, LinkSpace::uniform(4));
    let tn2 = random_treetn_f64(&mut rng2, &site_network2, LinkSpace::uniform(4));

    // Verify same_topology works
    assert!(tn1.same_topology(&tn2));

    // Note: contract_zipup with non-shared site indices currently has issues
    // because tensordot requires at least one contraction pair.
    // This test verifies the API exists and type checks correctly.
}

/// Test contract_zipup with shared site indices (inner product case).
/// This is the typical use case: computing overlap <ψ|φ>.
#[test]
fn test_contract_zipup_shared_site_indices() {
    use std::collections::HashSet;

    // Create shared site indices (same Index objects for both networks)
    let site_a = Index::new_dyn(2);  // Physical index at node A
    let site_b = Index::new_dyn(3);  // Physical index at node B

    // Create bond indices (different for each network)
    let bond1 = Index::new_dyn(4);  // Bond for network 1
    let bond2 = Index::new_dyn(4);  // Bond for network 2

    // Create tensor for node A in network 1: A1[site_a, bond1]
    let tensor_a1 = TensorDynLen::new(
        vec![site_a.clone(), bond1.clone()],
        vec![2, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 8]))),
    );

    // Create tensor for node B in network 1: B1[bond1, site_b]
    let tensor_b1 = TensorDynLen::new(
        vec![bond1.clone(), site_b.clone()],
        vec![4, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12]))),
    );

    // Create TreeTN 1
    let tn1: TreeTN<DynId, NoSymmSpace, String> = TreeTN::from_tensors(
        vec![tensor_a1, tensor_b1],
        vec!["A".to_string(), "B".to_string()],
    ).unwrap();

    // Create tensor for node A in network 2: A2[site_a, bond2]
    // Uses the SAME site_a index
    let tensor_a2 = TensorDynLen::new(
        vec![site_a.clone(), bond2.clone()],
        vec![2, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![2.0; 8]))),
    );

    // Create tensor for node B in network 2: B2[bond2, site_b]
    // Uses the SAME site_b index
    let tensor_b2 = TensorDynLen::new(
        vec![bond2.clone(), site_b.clone()],
        vec![4, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![2.0; 12]))),
    );

    // Create TreeTN 2
    let tn2: TreeTN<DynId, NoSymmSpace, String> = TreeTN::from_tensors(
        vec![tensor_a2, tensor_b2],
        vec!["A".to_string(), "B".to_string()],
    ).unwrap();

    // Verify same topology
    assert!(tn1.same_topology(&tn2));
    assert_eq!(tn1.node_count(), 2);
    assert_eq!(tn2.node_count(), 2);

    // Contract with center at B
    // Since all site indices are contracted (inner product case),
    // all tensors collapse into the center node
    let result = tn1.contract_zipup(&tn2, &"B".to_string(), None, None).unwrap();

    // For inner product: all nodes get absorbed into center
    // Result should have 1 node (the center B) containing a scalar
    assert_eq!(result.node_count(), 1);

    // Result should have 0 edges (single node)
    assert_eq!(result.edge_count(), 0);

    // The center node should exist
    assert!(result.node_index(&"B".to_string()).is_some());
}

/// Test contract_zipup with partial site contraction (external indices remain).
/// This simulates computing <ψ|O|φ> where O is an MPO.
#[test]
fn test_contract_zipup_partial_contraction() {
    // Create indices
    // Shared site indices (will be contracted between tn1 and tn2)
    let site_a = Index::new_dyn(2);  // Physical index at node A
    let site_b = Index::new_dyn(3);  // Physical index at node B

    // External index (exists only in tn2, will remain in result)
    let site_external_a = Index::new_dyn(2);  // External index at A (like MPO output)
    let site_external_b = Index::new_dyn(3);  // External index at B (like MPO output)

    // Bond indices (different for each network, will be replaced by sim_internal_inds)
    let bond1 = Index::new_dyn(4);  // Bond for network 1
    let bond2 = Index::new_dyn(4);  // Bond for network 2

    // Network 1 (like bra <ψ|):
    // A1[site_a, bond1] -- B1[bond1, site_b]
    let tensor_a1 = TensorDynLen::new(
        vec![site_a.clone(), bond1.clone()],
        vec![2, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 8]))),
    );

    let tensor_b1 = TensorDynLen::new(
        vec![bond1.clone(), site_b.clone()],
        vec![4, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12]))),
    );

    let tn1: TreeTN<DynId, NoSymmSpace, String> = TreeTN::from_tensors(
        vec![tensor_a1, tensor_b1],
        vec!["A".to_string(), "B".to_string()],
    ).unwrap();

    // Network 2 (like O|φ> with MPO indices):
    // A2[site_a, site_external_a, bond2] -- B2[bond2, site_b, site_external_b]
    // Has both shared and external site indices at each node
    let tensor_a2 = TensorDynLen::new(
        vec![site_a.clone(), site_external_a.clone(), bond2.clone()],
        vec![2, 2, 4],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![2.0; 16]))),
    );

    let tensor_b2 = TensorDynLen::new(
        vec![bond2.clone(), site_b.clone(), site_external_b.clone()],
        vec![4, 3, 3],
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![2.0; 36]))),
    );

    let tn2: TreeTN<DynId, NoSymmSpace, String> = TreeTN::from_tensors(
        vec![tensor_a2, tensor_b2],
        vec!["A".to_string(), "B".to_string()],
    ).unwrap();

    // Verify same topology
    assert!(tn1.same_topology(&tn2));

    // Contract with center at B
    let result = tn1.contract_zipup(&tn2, &"B".to_string(), None, None).unwrap();

    // Result should preserve structure because both nodes have external indices
    assert_eq!(result.node_count(), 2);
    assert_eq!(result.edge_count(), 1);

    // Check that node A exists and has the external index
    let node_a = result.node_index(&"A".to_string()).unwrap();
    let tensor_a = result.tensor(node_a).unwrap();
    assert!(tensor_a.indices.iter().any(|idx| idx.id == site_external_a.id));

    // Check that node B exists and has the external index
    let node_b = result.node_index(&"B".to_string()).unwrap();
    let tensor_b = result.tensor(node_b).unwrap();
    assert!(tensor_b.indices.iter().any(|idx| idx.id == site_external_b.id));
}

// ============================================================================
// Helper function for comparing contract_zipup vs contract_naive
// ============================================================================

use tensor4all_treetn::{SiteIndexNetwork, random_treetn_f64, LinkSpace};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

/// Helper to test contract_zipup vs contract_naive for a given topology.
///
/// Creates two random TreeTNs with the given site network and link space,
/// contracts them using both methods, and verifies the results match.
///
/// # Arguments
/// * `site_network` - The site index network defining topology and site indices
/// * `link_space` - The bond dimensions
/// * `center` - The center node name for contract_zipup
/// * `seed1`, `seed2` - Random seeds for the two networks
/// * `rtol` - Relative tolerance for comparison
fn compare_zipup_vs_naive(
    site_network: &SiteIndexNetwork<String, DynId>,
    link_space: LinkSpace<String>,
    center: &str,
    seed1: u64,
    seed2: u64,
    rtol: f64,
) {
    // Create two random TreeTNs
    let mut rng1 = ChaCha8Rng::seed_from_u64(seed1);
    let mut rng2 = ChaCha8Rng::seed_from_u64(seed2);
    let tn1 = random_treetn_f64(&mut rng1, site_network, link_space.clone());
    let tn2 = random_treetn_f64(&mut rng2, site_network, link_space);

    // Contract using naive method (reference)
    let naive_result = tn1.contract_naive(&tn2)
        .expect("contract_naive should succeed");

    // Contract using zipup method
    let zipup_result = tn1.contract_zipup(&tn2, &center.to_string(), None, None)
        .expect("contract_zipup should succeed");

    // Convert zipup result to tensor for comparison
    let zipup_tensor = zipup_result.contract_to_tensor()
        .expect("contract_to_tensor should succeed");

    // Compare using distance
    let dist = naive_result.distance(&zipup_tensor);
    assert!(
        dist < rtol,
        "contract_zipup vs contract_naive distance too large: {} >= {}",
        dist, rtol
    );
}

// ============================================================================
// Topology tests using helper function
// ============================================================================

/// Test contract_zipup vs contract_naive for 2-node chain: A -- B
#[test]
fn test_zipup_vs_naive_2node_chain() {
    let mut site_network = SiteIndexNetwork::<String, DynId>::new();
    let site_a = Index::new_dyn(2);
    let site_b = Index::new_dyn(3);
    site_network.add_node("A".to_string(), HashSet::from([site_a])).unwrap();
    site_network.add_node("B".to_string(), HashSet::from([site_b])).unwrap();
    site_network.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(4), "B", 42, 123, 1e-10);
}

/// Test contract_zipup vs contract_naive for 3-node chain: A -- B -- C
#[test]
fn test_zipup_vs_naive_3node_chain() {
    let mut site_network = SiteIndexNetwork::<String, DynId>::new();
    let site_a = Index::new_dyn(2);
    let site_b = Index::new_dyn(2);
    let site_c = Index::new_dyn(2);
    site_network.add_node("A".to_string(), HashSet::from([site_a])).unwrap();
    site_network.add_node("B".to_string(), HashSet::from([site_b])).unwrap();
    site_network.add_node("C".to_string(), HashSet::from([site_c])).unwrap();
    site_network.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    site_network.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    // Test with center at B (middle)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "B", 42, 123, 1e-10);

    // Test with center at A (end)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "A", 42, 123, 1e-10);

    // Test with center at C (other end)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "C", 42, 123, 1e-10);
}

/// Test contract_zipup vs contract_naive for star topology:
///       A
///       |
///   C - B - D
#[test]
fn test_zipup_vs_naive_star() {
    let mut site_network = SiteIndexNetwork::<String, DynId>::new();
    let site_a = Index::new_dyn(2);
    let site_b = Index::new_dyn(2);
    let site_c = Index::new_dyn(2);
    let site_d = Index::new_dyn(2);
    site_network.add_node("A".to_string(), HashSet::from([site_a])).unwrap();
    site_network.add_node("B".to_string(), HashSet::from([site_b])).unwrap();
    site_network.add_node("C".to_string(), HashSet::from([site_c])).unwrap();
    site_network.add_node("D".to_string(), HashSet::from([site_d])).unwrap();
    site_network.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    site_network.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    site_network.add_edge(&"B".to_string(), &"D".to_string()).unwrap();

    // Test with center at B (hub)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "B", 42, 123, 1e-10);

    // Test with center at A (leaf)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "A", 42, 123, 1e-10);
}

/// Test contract_zipup vs contract_naive for Y-shaped topology:
///   A - B - C
///       |
///       D
#[test]
fn test_zipup_vs_naive_y_shape() {
    let mut site_network = SiteIndexNetwork::<String, DynId>::new();
    let site_a = Index::new_dyn(2);
    let site_b = Index::new_dyn(2);
    let site_c = Index::new_dyn(2);
    let site_d = Index::new_dyn(2);
    site_network.add_node("A".to_string(), HashSet::from([site_a])).unwrap();
    site_network.add_node("B".to_string(), HashSet::from([site_b])).unwrap();
    site_network.add_node("C".to_string(), HashSet::from([site_c])).unwrap();
    site_network.add_node("D".to_string(), HashSet::from([site_d])).unwrap();
    site_network.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    site_network.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    site_network.add_edge(&"B".to_string(), &"D".to_string()).unwrap();

    // Test with center at B
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "B", 42, 123, 1e-10);

    // Test with center at D
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "D", 42, 123, 1e-10);
}

/// Test contract_zipup vs contract_naive for 5-node chain: A -- B -- C -- D -- E
#[test]
fn test_zipup_vs_naive_5node_chain() {
    let mut site_network = SiteIndexNetwork::<String, DynId>::new();
    for name in ["A", "B", "C", "D", "E"] {
        let site = Index::new_dyn(2);
        site_network.add_node(name.to_string(), HashSet::from([site])).unwrap();
    }
    site_network.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    site_network.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    site_network.add_edge(&"C".to_string(), &"D".to_string()).unwrap();
    site_network.add_edge(&"D".to_string(), &"E".to_string()).unwrap();

    // Test with center at C (middle)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(2), "C", 42, 123, 1e-10);

    // Test with center at E (end)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(2), "E", 42, 123, 1e-10);
}
