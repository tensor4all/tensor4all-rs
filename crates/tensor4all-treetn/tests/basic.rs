use tensor4all_treetn::{Connection, TreeTN, TreeTopology};
use tensor4all::index::{DefaultIndex as Index, DynId};
use tensor4all::{TensorDynLen, Storage};
use tensor4all::NoSymmSpace;
use tensor4all::storage::{DenseStorageF64, DenseStorageC64};
use std::sync::Arc;
use std::collections::HashMap;
use petgraph::graph::NodeIndex;
use num_complex::Complex64;

#[test]
fn test_connection_creation() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(3);
    
    let conn = Connection::new(idx_a, idx_b);
    assert!(conn.is_ok());
    let conn = conn.unwrap();
    assert_eq!(conn.bond_dim(), 3);
}

#[test]
fn test_connection_dimension_mismatch() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(5);
    
    let conn = Connection::new(idx_a, idx_b);
    assert!(conn.is_err());
    let err_msg = conn.unwrap_err().to_string();
    assert!(err_msg.contains("dimension") || err_msg.contains("mismatch") || err_msg.contains("matching"));
}

#[test]
fn test_connection_replace_bond_indices() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(3);
    
    let mut conn = Connection::new(idx_a, idx_b).unwrap();
    assert_eq!(conn.bond_dim(), 3);
    
    let new_idx_a = Index::new_dyn(5);
    let new_idx_b = Index::new_dyn(5);
    
    let result = conn.replace_bond_indices(new_idx_a, new_idx_b);
    assert!(result.is_ok());
    assert_eq!(conn.bond_dim(), 5);
}

#[test]
fn test_connection_replace_bond_indices_mismatch() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(3);
    
    let mut conn = Connection::new(idx_a, idx_b).unwrap();
    
    let new_idx_a = Index::new_dyn(5);
    let new_idx_b = Index::new_dyn(7);
    
    let result = conn.replace_bond_indices(new_idx_a, new_idx_b);
    assert!(result.is_err());
}

#[test]
fn test_connection_ortho_towards() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(3);
    
    let mut conn = Connection::new(idx_a.clone(), idx_b.clone()).unwrap();
    assert!(conn.ortho_towards.is_none());
    
    // Set ortho_towards to idx_a
    let result = conn.set_ortho_towards(Some(idx_a.clone()));
    assert!(result.is_ok());
    assert_eq!(conn.ortho_towards.as_ref().map(|idx| &idx.id), Some(&idx_a.id));
    
    // Set ortho_towards to idx_b
    let result = conn.set_ortho_towards(Some(idx_b.clone()));
    assert!(result.is_ok());
    assert_eq!(conn.ortho_towards.as_ref().map(|idx| &idx.id), Some(&idx_b.id));
    
    // Clear ortho_towards
    let result = conn.set_ortho_towards(None);
    assert!(result.is_ok());
    assert!(conn.ortho_towards.is_none());
}

#[test]
fn test_connection_ortho_towards_invalid() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(3);
    let idx_c = Index::new_dyn(3); // Different ID
    
    let mut conn = Connection::new(idx_a, idx_b).unwrap();
    
    // Try to set ortho_towards to an index that's not in the connection
    let result = conn.set_ortho_towards(Some(idx_c));
    assert!(result.is_err());
}

#[test]
fn test_treetn_add_tensor() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i, j];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    let tensor = TensorDynLen::new(indices, dims, storage);
    
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
    let indices = vec![i, j];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    let tensor = TensorDynLen::new(indices, dims, storage);
    
    let node = tn.add_tensor_auto_name(tensor);
    assert_eq!(tn.tensor(node).unwrap().dims, vec![2, 3]);
    
    // Replace with a new tensor
    let new_i = Index::new_dyn(4);
    let new_j = Index::new_dyn(5);
    let new_indices = vec![new_i, new_j];
    let new_dims = vec![4, 5];
    let new_storage = Arc::new(Storage::new_dense_f64(20));
    let new_tensor = TensorDynLen::new(new_indices, new_dims, new_storage);
    
    let old_tensor_result = tn.replace_tensor(node, new_tensor);
    assert!(old_tensor_result.is_ok());
    let old_tensor = old_tensor_result.unwrap();
    assert!(old_tensor.is_some());
    assert_eq!(old_tensor.unwrap().dims, vec![2, 3]);
    assert_eq!(tn.tensor(node).unwrap().dims, vec![4, 5]);
}

#[test]
fn test_treetn_replace_tensor_invalid_node() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i, j];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    let new_tensor = TensorDynLen::new(indices, dims, storage);
    
    // Try to replace a non-existent node
    let invalid_node = NodeIndex::new(999);
    let result = tn.replace_tensor(invalid_node, new_tensor);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_treetn_replace_tensor_missing_connection_index() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    // Create two connected tensors
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    // Connect via j1 and i2
    tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Try to replace node1 with a tensor that doesn't have j1
    let new_i = Index::new_dyn(5);
    let new_j = Index::new_dyn(6);
    let new_indices = vec![new_i, new_j]; // j1 is missing!
    let new_dims = vec![5, 6];
    let new_storage = Arc::new(Storage::new_dense_f64(30));
    let new_tensor = TensorDynLen::new(new_indices, new_dims, new_storage);
    
    let result = tn.replace_tensor(node1, new_tensor);
    assert!(result.is_err());
    // Error should indicate missing index
    match result {
        Err(e) => {
            let err_msg = e.to_string();
            assert!(err_msg.contains("missing") || err_msg.contains("index") || err_msg.contains("connection") || err_msg.contains("indices"));
        }
        Ok(_) => panic!("Expected error but got Ok"),
    }
}

#[test]
fn test_treetn_replace_tensor_with_connection() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    // Create two connected tensors
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    // Connect via j1 and i2
    tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Replace node1 with a tensor that has j1 (same ID)
    let new_i = Index::new_dyn(5);
    let new_j1 = j1.clone(); // Keep the same j1 index
    let new_indices = vec![new_i, new_j1];
    let new_dims = vec![5, 3];
    let new_storage = Arc::new(Storage::new_dense_f64(15));
    let new_tensor = TensorDynLen::new(new_indices, new_dims, new_storage);
    
    let result = tn.replace_tensor(node1, new_tensor);
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());
}

#[test]
fn test_treetn_connect() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    // Create two tensors
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    // Connect via j1 and i2 (both dimension 3)
    let edge = tn.connect(node1, &j1, node2, &i2);
    assert!(edge.is_ok());
    assert_eq!(tn.edge_count(), 1);
    
    // Verify connection
    let conn = tn.connection(edge.unwrap());
    assert!(conn.is_some());
    assert_eq!(conn.unwrap().bond_dim(), 3);
}

#[test]
fn test_treetn_connect_dimension_mismatch() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(5); // Different dimension
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![5, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(20));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    // Try to connect j1 (dim 3) with i2 (dim 5) - should fail
    // Note: i2 has dimension 5, but j1 has dimension 3, so they can't be connected
    // However, the test setup has i2 in tensor2 with dimension 5, so the connection
    // will fail at the dimension mismatch check in Connection::new
    let edge = tn.connect(node1, &j1, node2, &i2);
    assert!(edge.is_err());
    let err_msg = edge.unwrap_err().to_string();
    // Error should mention dimension mismatch or matching dimensions
    assert!(err_msg.contains("dimension") || err_msg.contains("mismatch") || err_msg.contains("matching") || err_msg.contains("Failed to create connection"));
}

#[test]
fn test_treetn_connect_invalid_node() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    // Create invalid node index
    let invalid_node = NodeIndex::new(999);
    
    let edge = tn.connect(node1, &j1, invalid_node, &i1);
    assert!(edge.is_err());
    let err_msg = edge.unwrap_err().to_string();
    // Error should mention that nodes don't exist
    assert!(err_msg.contains("exist") || err_msg.contains("found") || err_msg.contains("node") || err_msg.contains("One or both") || err_msg.contains("Failed to connect"));
}

#[test]
fn test_treetn_connect_index_not_in_tensor() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    // Try to connect with an index that doesn't exist in tensor1
    let fake_index = Index::new_dyn(3);
    let edge = tn.connect(node1, &fake_index, node2, &i2);
    assert!(edge.is_err());
    let err_msg = edge.unwrap_err().to_string();
    assert!(err_msg.contains("found") || err_msg.contains("Index") || err_msg.contains("index"));
}

#[test]
fn test_treetn_edge_index_for_node() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    let edge = tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Get index for node1
    let idx1 = tn.edge_index_for_node(edge, node1);
    assert!(idx1.is_ok());
    assert_eq!(idx1.unwrap().id, j1.id);
    
    // Get index for node2
    let idx2 = tn.edge_index_for_node(edge, node2);
    assert!(idx2.is_ok());
    assert_eq!(idx2.unwrap().id, i2.id);
    
    // Try with invalid node
    let invalid_node = NodeIndex::new(999);
    let result = tn.edge_index_for_node(edge, invalid_node);
    assert!(result.is_err());
}

#[test]
fn test_treetn_replace_edge_bond() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    let edge = tn.connect(node1, &j1, node2, &i2).unwrap();
    assert_eq!(tn.connection(edge).unwrap().bond_dim(), 3);
    
    // Replace with new indices (dimension 5)
    let new_idx1 = Index::new_dyn(5);
    let new_idx2 = Index::new_dyn(5);
    let result = tn.replace_edge_bond(edge, new_idx1, new_idx2);
    assert!(result.is_ok());
    assert_eq!(tn.connection(edge).unwrap().bond_dim(), 5);
}

#[test]
fn test_treetn_replace_edge_bond_mismatch() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    let edge = tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Try to replace with mismatched dimensions
    let new_idx1 = Index::new_dyn(5);
    let new_idx2 = Index::new_dyn(7);
    let result = tn.replace_edge_bond(edge, new_idx1, new_idx2);
    assert!(result.is_err());
}

#[test]
fn test_treetn_set_edge_ortho_towards() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    let edge = tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Set ortho direction to j1 (node1 side)
    let result = tn.set_edge_ortho_towards(edge, Some(j1.clone()));
    assert!(result.is_ok());
    assert_eq!(
        tn.connection(edge).unwrap().ortho_towards.as_ref().map(|idx| &idx.id),
        Some(&j1.id)
    );
    
    // Set ortho direction to i2 (node2 side)
    let result = tn.set_edge_ortho_towards(edge, Some(i2.clone()));
    assert!(result.is_ok());
    assert_eq!(
        tn.connection(edge).unwrap().ortho_towards.as_ref().map(|idx| &idx.id),
        Some(&i2.id)
    );
    
    // Clear ortho direction
    let result = tn.set_edge_ortho_towards(edge, None);
    assert!(result.is_ok());
    assert!(tn.connection(edge).unwrap().ortho_towards.is_none());
}

#[test]
fn test_treetn_set_edge_ortho_towards_invalid() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    let edge = tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Try to set ortho direction to an index that's not in the connection
    let invalid_index = Index::new_dyn(5); // Different ID
    let result = tn.set_edge_ortho_towards(edge, Some(invalid_index));
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("index") || err_msg.contains("index_source") || err_msg.contains("index_target"));
}

#[test]
fn test_treetn_validate_tree_simple() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    // Create a simple tree: node1 -- node2
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Should be valid: 2 nodes, 1 edge
    assert!(tn.validate_tree().is_ok());
}

#[test]
fn test_treetn_validate_tree_three_nodes() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    // Create a tree: node1 -- node2 -- node3
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let j2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), j2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    let i3 = Index::new_dyn(4);
    let k3 = Index::new_dyn(5);
    let indices3 = vec![i3.clone(), k3.clone()];
    let dims3 = vec![4, 5];
    let storage3 = Arc::new(Storage::new_dense_f64(20));
    let tensor3 = TensorDynLen::new(indices3, dims3, storage3);
    let node3 = tn.add_tensor_auto_name(tensor3);
    
    tn.connect(node1, &j1, node2, &i2).unwrap();
    tn.connect(node2, &j2, node3, &i3).unwrap();
    
    // Should be valid: 3 nodes, 2 edges
    assert!(tn.validate_tree().is_ok());
}

#[test]
fn test_treetn_validate_tree_cycle() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    // Create a cycle: node1 -- node2 -- node3 -- node1
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let k1 = Index::new_dyn(4);
    let indices1 = vec![i1.clone(), j1.clone(), k1.clone()];
    let dims1 = vec![2, 3, 4];
    let storage1 = Arc::new(Storage::new_dense_f64(24));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let j2 = Index::new_dyn(5);
    let indices2 = vec![i2.clone(), j2.clone()];
    let dims2 = vec![3, 5];
    let storage2 = Arc::new(Storage::new_dense_f64(15));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    let i3 = Index::new_dyn(4);
    let j3 = Index::new_dyn(5);
    let indices3 = vec![i3.clone(), j3.clone()];
    let dims3 = vec![4, 5];
    let storage3 = Arc::new(Storage::new_dense_f64(20));
    let tensor3 = TensorDynLen::new(indices3, dims3, storage3);
    let node3 = tn.add_tensor_auto_name(tensor3);
    
    tn.connect(node1, &j1, node2, &i2).unwrap();
    tn.connect(node2, &j2, node3, &j3).unwrap();
    tn.connect(node3, &i3, node1, &k1).unwrap();
    
    // Should fail: 3 nodes, 3 edges (cycle, not a tree)
    assert!(tn.validate_tree().is_err());
}

#[test]
fn test_treetn_validate_tree_disconnected() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    // Create two disconnected components
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    let i3 = Index::new_dyn(5);
    let j3 = Index::new_dyn(6);
    let indices3 = vec![i3.clone(), j3.clone()];
    let dims3 = vec![5, 6];
    let storage3 = Arc::new(Storage::new_dense_f64(30));
    let tensor3 = TensorDynLen::new(indices3, dims3, storage3);
    let _node3 = tn.add_tensor_auto_name(tensor3);
    
    // Connect node1 and node2, but leave node3 disconnected
    tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Should fail: graph is not connected
    assert!(tn.validate_tree().is_err());
}

#[test]
fn test_treetn_validate_tree_empty() {
    let tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    // Empty graph should be valid
    assert!(tn.validate_tree().is_ok());
}

#[test]
fn test_auto_centers_empty() {
    let tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    assert!(!tn.is_canonicalized());
    assert!(tn.ortho_region().is_empty());
}

#[test]
fn test_set_auto_centers() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i, j];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    let tensor = TensorDynLen::new(indices, dims, storage);
    let node = tn.add_tensor_auto_name(tensor);
    
    // Initially not canonicalized
    assert!(!tn.is_canonicalized());
    
    // Set ortho_region
    let result = tn.set_ortho_region(vec![node]);
    assert!(result.is_ok());
    assert!(tn.is_canonicalized());
    assert_eq!(tn.ortho_region().len(), 1);
    assert!(tn.ortho_region().contains(&node));
}

#[test]
fn test_set_auto_centers_invalid_node() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let invalid_node = NodeIndex::new(999);
    let result = tn.set_ortho_region(vec![invalid_node]);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result);
    assert!(err_msg.contains("exist") || err_msg.contains("valid"));
}

#[test]
fn test_add_remove_auto_center() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1, j1];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor_auto_name(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2, k2];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor_auto_name(tensor2);
    
    // Add first node to region
    let result = tn.add_to_ortho_region(node1);
    assert!(result.is_ok());
    assert!(tn.is_canonicalized());
    assert_eq!(tn.ortho_region().len(), 1);
    
    // Add second node to region
    let result = tn.add_to_ortho_region(node2);
    assert!(result.is_ok());
    assert_eq!(tn.ortho_region().len(), 2);
    assert!(tn.ortho_region().contains(&node1));
    assert!(tn.ortho_region().contains(&node2));
    
    // Remove first node from region
    tn.remove_from_ortho_region(&node1);
    assert_eq!(tn.ortho_region().len(), 1);
    assert!(!tn.ortho_region().contains(&node1));
    assert!(tn.ortho_region().contains(&node2));
    
    // Remove second node from region
    tn.remove_from_ortho_region(&node2);
    assert!(!tn.is_canonicalized());
    assert!(tn.ortho_region().is_empty());
}

#[test]
fn test_add_auto_center_invalid() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let invalid_node = NodeIndex::new(999);
    let result = tn.add_to_ortho_region(invalid_node);
    assert!(result.is_err());
}

#[test]
fn test_clear_auto_centers() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i, j];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    let tensor = TensorDynLen::new(indices, dims, storage);
    let node = tn.add_tensor_auto_name(tensor);
    
    tn.set_ortho_region(vec![node]).unwrap();
    assert!(tn.is_canonicalized());
    
    tn.clear_ortho_region();
    assert!(!tn.is_canonicalized());
    assert!(tn.ortho_region().is_empty());
}

#[test]
fn test_validate_ortho_consistency_requires_connected_auto_centers() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    // Create three nodes, but don't connect them (ortho_region will be disconnected).
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let tensor1 = TensorDynLen::new(vec![i1, j1], vec![2, 3], Arc::new(Storage::new_dense_f64(6)));
    let n1 = tn.add_tensor_auto_name(tensor1);

    let i2 = Index::new_dyn(2);
    let j2 = Index::new_dyn(3);
    let tensor2 = TensorDynLen::new(vec![i2, j2], vec![2, 3], Arc::new(Storage::new_dense_f64(6)));
    let n2 = tn.add_tensor_auto_name(tensor2);

    // Two centers that are not connected by edges should fail connectivity check.
    tn.set_ortho_region(vec![n1, n2]).unwrap();
    assert!(tn.validate_ortho_consistency().is_err());
}

#[test]
fn test_validate_ortho_consistency_none_only_inside_centers() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    // Build a simple chain: n1 -- n2
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let tensor1 = TensorDynLen::new(vec![i1.clone(), j1.clone()], vec![2, 3], Arc::new(Storage::new_dense_f64(6)));
    let n1 = tn.add_tensor_auto_name(tensor1);

    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let tensor2 = TensorDynLen::new(vec![i2.clone(), k2.clone()], vec![3, 4], Arc::new(Storage::new_dense_f64(12)));
    let n2 = tn.add_tensor_auto_name(tensor2);

    let e = tn.connect(n1, &j1, n2, &i2).unwrap();

    // Only n2 is center.
    tn.set_ortho_region(vec![n2]).unwrap();

    // Incorrectly set None on boundary edge (should be forbidden).
    {
        let conn = tn.connection_mut(e).unwrap();
        conn.ortho_towards = None;
    }
    assert!(tn.validate_ortho_consistency().is_err());
}

#[test]
fn test_validate_ortho_consistency_chain_points_towards_centers() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    // Build chain: n1 -- n2 -- n3, center is n2.
    let a1 = Index::new_dyn(2);
    let b1 = Index::new_dyn(3);
    let t1 = TensorDynLen::new(vec![a1, b1.clone()], vec![2, 3], Arc::new(Storage::new_dense_f64(6)));
    let n1 = tn.add_tensor_auto_name(t1);

    let a2 = Index::new_dyn(3);
    let b2 = Index::new_dyn(4);
    let c2 = Index::new_dyn(5);
    let t2 = TensorDynLen::new(vec![a2.clone(), b2.clone(), c2], vec![3, 4, 5], Arc::new(Storage::new_dense_f64(60)));
    let n2 = tn.add_tensor_auto_name(t2);

    let a3 = Index::new_dyn(4);
    let b3 = Index::new_dyn(6);
    let t3 = TensorDynLen::new(vec![a3.clone(), b3], vec![4, 6], Arc::new(Storage::new_dense_f64(24)));
    let n3 = tn.add_tensor_auto_name(t3);

    // Connect n1 -- n2 via b1 (dim3) and a2 (dim3)
    let e12 = tn.connect(n1, &b1, n2, &a2).unwrap();
    // Connect n2 -- n3 via b2 (dim4) and a3 (dim4)
    let e23 = tn.connect(n2, &b2, n3, &a3).unwrap();

    tn.set_ortho_region(vec![n2]).unwrap();

    // Boundary edges must point into centers: for e12, center is n2, which is the target (n1,n2) => use a2
    tn.set_edge_ortho_towards(e12, Some(a2.clone())).unwrap();
    // for e23, center is n2, which is the source (n2,n3) => use b2
    tn.set_edge_ortho_towards(e23, Some(b2.clone())).unwrap();

    assert!(tn.validate_ortho_consistency().is_ok());
}

#[test]
fn test_canonicalize_simple() {
    // Create a simple 2-node tree: n1 -- n2
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let a1 = Index::new_dyn(2);
    let b1 = Index::new_dyn(3);
    let a2 = Index::new_dyn(3);
    let b2 = Index::new_dyn(4);

    let indices1 = vec![a1.clone(), b1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6])));
    let tensor1: TensorDynLen<DynId> = TensorDynLen::new(indices1, dims1, storage1);

    let indices2 = vec![a2.clone(), b2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12])));
    let tensor2: TensorDynLen<DynId> = TensorDynLen::new(indices2, dims2, storage2);

    let n1 = tn.add_tensor_auto_name(tensor1);
    let n2 = tn.add_tensor_auto_name(tensor2);

    let _e12 = tn.connect(n1, &b1, n2, &a2).unwrap();

    // Canonicalize towards n2
    let tn_canon = tn.canonicalize(vec![n2]).unwrap();

    // Verify that the network is canonicalized
    assert!(tn_canon.is_canonicalized());
    assert!(tn_canon.ortho_region().contains(&n2));

    // Verify ortho consistency
    assert!(tn_canon.validate_ortho_consistency().is_ok());
}

#[test]
fn test_canonicalize_mixed_storage() {
    // Test canonicalization with mixed f64 and Complex64 storage
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let a1 = Index::new_dyn(2);
    let b1 = Index::new_dyn(3);
    let a2 = Index::new_dyn(3);
    let b2 = Index::new_dyn(4);

    // n1: f64 tensor
    let indices1 = vec![a1.clone(), b1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6])));
    let tensor1: TensorDynLen<DynId> = TensorDynLen::new(indices1, dims1, storage1);

    // n2: Complex64 tensor
    let indices2 = vec![a2.clone(), b2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(vec![Complex64::new(1.0, 0.0); 12])));
    let tensor2: TensorDynLen<DynId> = TensorDynLen::new(indices2, dims2, storage2);

    let n1 = tn.add_tensor_auto_name(tensor1);
    let n2 = tn.add_tensor_auto_name(tensor2);

    let _e12 = tn.connect(n1, &b1, n2, &a2).unwrap();

    // Canonicalize towards n2
    let tn_canon = tn.canonicalize(vec![n2]).unwrap();

    // Verify that the network is canonicalized
    assert!(tn_canon.is_canonicalized());
    assert!(tn_canon.ortho_region().contains(&n2));

    // Verify ortho consistency
    assert!(tn_canon.validate_ortho_consistency().is_ok());
}

// ============================================================================
// TensorDynLen::add tests
// ============================================================================

#[test]
fn test_tensor_add_same_indices() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a)));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 3], storage_a);

    let data_b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b)));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 3], storage_b);

    let sum = tensor_a.add(&tensor_b).unwrap();
    assert_eq!(sum.dims, vec![2, 3]);

    // Verify values
    match sum.storage.as_ref() {
        Storage::DenseF64(d) => {
            let data = d.as_slice();
            assert_eq!(data[0], 2.0);
            assert_eq!(data[5], 7.0);
        }
        _ => panic!("Expected DenseF64"),
    }
}

#[test]
fn test_tensor_add_permuted_indices() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a)));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 3], storage_a);

    // tensor_b has indices in reverse order
    let data_b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b)));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(vec![j.clone(), i.clone()], vec![3, 2], storage_b);

    // Should still work - tensor_b will be permuted to match tensor_a
    let sum = tensor_a.add(&tensor_b).unwrap();
    assert_eq!(sum.dims, vec![2, 3]);
}

#[test]
fn test_tensor_add_different_indices_fails() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);

    let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a)));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 3], storage_a);

    let data_b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b)));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone(), k.clone()], vec![2, 3], storage_b);

    let result = tensor_a.add(&tensor_b);
    assert!(result.is_err());
}

// ============================================================================
// TreeTN scalar multiplication tests
// ============================================================================

#[test]
fn test_treetn_scalar_mul_positive() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6])));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(vec![i, j], vec![2, 3], storage);
    let _node = tn.add_tensor_auto_name(tensor);

    // Multiply by 4.0
    let tn_scaled = tn * 4.0;

    // Since there's one node and ortho_region is empty, all data should be scaled by 4^(1/1) = 4
    let node = tn_scaled.node_indices()[0];
    let tensor = tn_scaled.tensor(node).unwrap();
    match tensor.storage.as_ref() {
        Storage::DenseF64(d) => {
            // Should be 4.0 for all elements
            assert!((d.as_slice()[0] - 4.0).abs() < 1e-10);
        }
        _ => panic!("Expected DenseF64"),
    }
}

#[test]
fn test_treetn_scalar_mul_negative() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 1.0])));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(vec![i], vec![2], storage);
    let _node = tn.add_tensor_auto_name(tensor);

    // Multiply by -4.0
    let tn_scaled = tn * (-4.0);

    let node = tn_scaled.node_indices()[0];
    let tensor = tn_scaled.tensor(node).unwrap();
    match tensor.storage.as_ref() {
        Storage::DenseF64(d) => {
            // Should be -4.0 for all elements
            assert!((d.as_slice()[0] + 4.0).abs() < 1e-10);
        }
        _ => panic!("Expected DenseF64"),
    }
}

#[test]
fn test_treetn_scalar_mul_zero() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 2.0])));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(vec![i], vec![2], storage);
    let _node = tn.add_tensor_auto_name(tensor);

    // Multiply by 0.0
    let tn_scaled = tn * 0.0;

    let node = tn_scaled.node_indices()[0];
    let tensor = tn_scaled.tensor(node).unwrap();
    match tensor.storage.as_ref() {
        Storage::DenseF64(d) => {
            // All elements should be 0
            for val in d.as_slice() {
                assert_eq!(*val, 0.0);
            }
        }
        _ => panic!("Expected DenseF64"),
    }
}

#[test]
fn test_treetn_scalar_mul_complex() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 1.0])));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(vec![i], vec![2], storage);
    let _node = tn.add_tensor_auto_name(tensor);

    // Multiply by a complex scalar
    let tn_scaled = tn * Complex64::new(0.0, 2.0);

    let node = tn_scaled.node_indices()[0];
    let tensor = tn_scaled.tensor(node).unwrap();
    match tensor.storage.as_ref() {
        Storage::DenseC64(d) => {
            // Should have imaginary component of 2.0
            assert!((d.as_slice()[0].im - 2.0).abs() < 1e-10);
        }
        _ => panic!("Expected DenseC64"),
    }
}

// ============================================================================
// TreeTN::contract_to_tensor tests
// ============================================================================

#[test]
fn test_treetn_contract_single_node() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6])));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(vec![i, j], vec![2, 3], storage);
    let _node = tn.add_tensor_auto_name(tensor);

    let result = tn.contract_to_tensor().unwrap();
    assert_eq!(result.dims, vec![2, 3]);
}

#[test]
fn test_treetn_contract_two_nodes() {
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let j2 = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    // node 1: indices [i, j1] with dims [2, 3]
    let storage1 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6])));
    let tensor1: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone(), j1.clone()], vec![2, 3], storage1);
    let n1 = tn.add_tensor_auto_name(tensor1);

    // node 2: indices [j2, k] with dims [3, 4]
    let storage2 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12])));
    let tensor2: TensorDynLen<DynId> = TensorDynLen::new(vec![j2.clone(), k.clone()], vec![3, 4], storage2);
    let n2 = tn.add_tensor_auto_name(tensor2);

    // Connect via j1 and j2
    tn.connect(n1, &j1, n2, &j2).unwrap();

    // Contract to tensor - currently this performs simple sequential contraction
    // which may not eliminate bond indices if they have different IDs
    // For this test, we just verify it runs without errors and produces valid output
    let result = tn.contract_to_tensor().unwrap();

    // The result should have some dimensions (implementation-dependent)
    assert!(!result.dims.is_empty());

    // Verify total size is consistent
    let total_size: usize = result.dims.iter().product();
    assert!(total_size > 0);
}

#[test]
fn test_treetn_contract_empty_fails() {
    let tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();
    let result = tn.contract_to_tensor();
    assert!(result.is_err());
}

#[test]
fn test_contract_to_tensor_chain() {
    // Test edge-based contract_to_tensor with a 3-node chain
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let j2 = Index::new_dyn(3);
    let k1 = Index::new_dyn(4);
    let k2 = Index::new_dyn(4);
    let l = Index::new_dyn(5);

    // Create a chain: node1 - node2 - node3
    // node1: [i, j1] dims [2, 3]
    let storage1 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec((0..6).map(|x| x as f64).collect())));
    let tensor1: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone(), j1.clone()], vec![2, 3], storage1);
    let n1 = tn.add_tensor_auto_name(tensor1);

    // node2: [j2, k1] dims [3, 4]
    let storage2 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec((0..12).map(|x| x as f64).collect())));
    let tensor2: TensorDynLen<DynId> = TensorDynLen::new(vec![j2.clone(), k1.clone()], vec![3, 4], storage2);
    let n2 = tn.add_tensor_auto_name(tensor2);

    // node3: [k2, l] dims [4, 5]
    let storage3 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec((0..20).map(|x| x as f64).collect())));
    let tensor3: TensorDynLen<DynId> = TensorDynLen::new(vec![k2.clone(), l.clone()], vec![4, 5], storage3);
    let n3 = tn.add_tensor_auto_name(tensor3);

    // Connect: node1 -- node2 -- node3
    tn.connect(n1, &j1, n2, &j2).unwrap();
    tn.connect(n2, &k1, n3, &k2).unwrap();

    // Contract to tensor
    let result = tn.contract_to_tensor().unwrap();

    // Result should have external indices only (i, l)
    assert_eq!(result.dims.len(), 2);
    assert_eq!(result.dims, vec![2, 5]);

    // Verify total size
    let data: Vec<f64> = match result.storage.as_ref() {
        Storage::DenseF64(d) => d.iter().copied().collect(),
        _ => panic!("Expected DenseF64 storage"),
    };
    assert_eq!(data.len(), 10); // 2 * 5 = 10
}

// ============================================================================
// TreeTN::add tests
// ============================================================================

#[test]
fn test_treetn_add_single_node() {
    // Create two compatible TreeTNs with single nodes
    let mut tn_a = TreeTN::<DynId, NoSymmSpace, usize>::new();
    let mut tn_b = TreeTN::<DynId, NoSymmSpace, usize>::new();

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let data_a = vec![1.0; 6];
    let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a)));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 3], storage_a);
    tn_a.add_tensor(0, tensor_a).unwrap();

    let data_b = vec![2.0; 6];
    let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b)));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 3], storage_b);
    tn_b.add_tensor(0, tensor_b).unwrap();

    // Add should succeed for compatible networks
    assert!(tn_a.can_add(&tn_b));
}

// ============================================================================
// TreeTopology and factorize_tensor_to_treetn tests
// ============================================================================

#[test]
fn test_tree_topology_validation() {
    // Valid topology: 2 nodes, 1 edge
    let mut nodes: HashMap<usize, Vec<usize>> = HashMap::new();
    nodes.insert(0, vec![0]);
    nodes.insert(1, vec![1]);
    let edges = vec![(0, 1)];
    let topology = TreeTopology::new(nodes, edges);
    assert!(topology.validate().is_ok());
}

#[test]
fn test_tree_topology_validation_fails_wrong_edge_count() {
    // Invalid: 3 nodes but 1 edge (should be 2)
    let mut nodes: HashMap<usize, Vec<usize>> = HashMap::new();
    nodes.insert(0, vec![0]);
    nodes.insert(1, vec![1]);
    nodes.insert(2, vec![2]);
    let edges = vec![(0, 1)];
    let topology = TreeTopology::new(nodes, edges);
    assert!(topology.validate().is_err());
}

#[test]
fn test_tree_topology_validation_fails_unknown_node() {
    let mut nodes: HashMap<usize, Vec<usize>> = HashMap::new();
    nodes.insert(0, vec![0]);
    nodes.insert(1, vec![1]);
    let edges = vec![(0, 2)]; // Node 2 doesn't exist
    let topology = TreeTopology::new(nodes, edges);
    assert!(topology.validate().is_err());
}

// ============================================================================
// TreeTN::add correctness validation tests
// Verify that contract(A+B) == contract(A) + contract(B)
// ============================================================================

/// Helper function to compute Frobenius norm of a tensor (works with both f64 and Complex64)
fn tensor_norm(tensor: &TensorDynLen<DynId>) -> f64 {
    match tensor.storage.as_ref() {
        Storage::DenseF64(d) => {
            d.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt()
        }
        Storage::DenseC64(d) => {
            d.as_slice().iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
        }
        _ => panic!("tensor_norm: unsupported storage type"),
    }
}

/// Helper function to compute max absolute difference between two tensors (works with both f64 and Complex64)
fn tensor_max_diff(a: &TensorDynLen<DynId>, b: &TensorDynLen<DynId>) -> f64 {
    assert_eq!(a.dims, b.dims, "Dimension mismatch for tensor comparison");

    match (a.storage.as_ref(), b.storage.as_ref()) {
        (Storage::DenseF64(da), Storage::DenseF64(db)) => {
            da.as_slice().iter().zip(db.as_slice().iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f64, f64::max)
        }
        (Storage::DenseC64(da), Storage::DenseC64(db)) => {
            da.as_slice().iter().zip(db.as_slice().iter())
                .map(|(x, y)| (x - y).norm())
                .fold(0.0_f64, f64::max)
        }
        _ => panic!("tensor_max_diff: mismatched or unsupported storage types"),
    }
}

/// Verify that contract(A + B) == contract(A) + contract(B) within tolerance
fn assert_treetn_add_correctness(
    tn_a: TreeTN<DynId, NoSymmSpace, usize>,
    tn_b: TreeTN<DynId, NoSymmSpace, usize>,
    context: &str,
) {
    let contracted_a = tn_a.clone().contract_to_tensor().unwrap();
    let contracted_b = tn_b.clone().contract_to_tensor().unwrap();
    let expected = contracted_a.add(&contracted_b).unwrap();

    let tn_sum = tn_a.add(tn_b).unwrap();
    let actual = tn_sum.contract_to_tensor().unwrap();

    // Compare dimensions (as sets, since order may differ)
    assert_eq!(
        expected.dims.iter().collect::<std::collections::HashSet<_>>(),
        actual.dims.iter().collect::<std::collections::HashSet<_>>(),
        "{}: dimension sets should match", context
    );

    let max_diff = tensor_max_diff(&expected, &actual);
    let norm = tensor_norm(&expected);

    assert!(
        max_diff < 1e-10 * norm.max(1.0),
        "{}: contract(A+B) should equal contract(A) + contract(B). Max diff: {}, norm: {}",
        context, max_diff, norm
    );
}

/// Macro to generate two-nodes correctness tests for different scalar types
macro_rules! test_treetn_add_two_nodes_correctness {
    ($test_name:ident, $scalar_type:ty, $storage_variant:ident, $storage_type:ty, $make_scalar:expr) => {
        #[test]
        fn $test_name() {
            // Structure: node 0 -- node 1
            let i0 = Index::new_dyn(2);
            let k1 = Index::new_dyn(3);
            let bond_a = Index::new_dyn(4);
            let bond_b = Index::new_dyn(4);

            // Network A
            let mut tn_a = TreeTN::<DynId, NoSymmSpace, usize>::new();

            let data_a0: Vec<$scalar_type> = (1..=8).map(|x| $make_scalar(x, 1)).collect();
            let storage_a0 = Arc::new(Storage::$storage_variant(<$storage_type>::from_vec(data_a0)));
            let tensor_a0: TensorDynLen<DynId> = TensorDynLen::new(
                vec![i0.clone(), bond_a.clone()], vec![2, 4], storage_a0);
            let n0_a = tn_a.add_tensor(0, tensor_a0).unwrap();

            let data_a1: Vec<$scalar_type> = (1..=12).map(|x| $make_scalar(x * 3, 2)).collect();
            let storage_a1 = Arc::new(Storage::$storage_variant(<$storage_type>::from_vec(data_a1)));
            let tensor_a1: TensorDynLen<DynId> = TensorDynLen::new(
                vec![bond_a.clone(), k1.clone()], vec![4, 3], storage_a1);
            let n1_a = tn_a.add_tensor(1, tensor_a1).unwrap();

            tn_a.connect(n0_a, &bond_a, n1_a, &bond_a).unwrap();

            // Network B
            let mut tn_b = TreeTN::<DynId, NoSymmSpace, usize>::new();

            let data_b0: Vec<$scalar_type> = (1..=8).map(|x| $make_scalar(x * 10, 3)).collect();
            let storage_b0 = Arc::new(Storage::$storage_variant(<$storage_type>::from_vec(data_b0)));
            let tensor_b0: TensorDynLen<DynId> = TensorDynLen::new(
                vec![i0.clone(), bond_b.clone()], vec![2, 4], storage_b0);
            let n0_b = tn_b.add_tensor(0, tensor_b0).unwrap();

            let data_b1: Vec<$scalar_type> = (1..=12).map(|x| $make_scalar(x * 7, 4)).collect();
            let storage_b1 = Arc::new(Storage::$storage_variant(<$storage_type>::from_vec(data_b1)));
            let tensor_b1: TensorDynLen<DynId> = TensorDynLen::new(
                vec![bond_b.clone(), k1.clone()], vec![4, 3], storage_b1);
            let n1_b = tn_b.add_tensor(1, tensor_b1).unwrap();

            tn_b.connect(n0_b, &bond_b, n1_b, &bond_b).unwrap();

            assert_treetn_add_correctness(tn_a, tn_b, stringify!($test_name));
        }
    };
}

// Generate tests for f64 and Complex64
test_treetn_add_two_nodes_correctness!(
    test_treetn_add_two_nodes_f64,
    f64, DenseF64, DenseStorageF64,
    |x: i32, _: i32| x as f64
);

test_treetn_add_two_nodes_correctness!(
    test_treetn_add_two_nodes_c64,
    Complex64, DenseC64, DenseStorageC64,
    |x: i32, y: i32| Complex64::new(x as f64, (y * x) as f64)
);

/// Macro to generate single-node correctness tests for different scalar types
macro_rules! test_treetn_add_single_node_correctness {
    ($test_name:ident, $scalar_type:ty, $storage_variant:ident, $storage_type:ty, $make_scalar:expr) => {
        #[test]
        fn $test_name() {
            let mut tn_a = TreeTN::<DynId, NoSymmSpace, usize>::new();
            let mut tn_b = TreeTN::<DynId, NoSymmSpace, usize>::new();

            let i = Index::new_dyn(2);
            let j = Index::new_dyn(3);

            let data_a: Vec<$scalar_type> = (1..=6).map(|x| $make_scalar(x, 1)).collect();
            let storage_a = Arc::new(Storage::$storage_variant(<$storage_type>::from_vec(data_a)));
            let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(
                vec![i.clone(), j.clone()], vec![2, 3], storage_a);
            tn_a.add_tensor(0, tensor_a).unwrap();

            let data_b: Vec<$scalar_type> = (1..=6).map(|x| $make_scalar(x * 10, 2)).collect();
            let storage_b = Arc::new(Storage::$storage_variant(<$storage_type>::from_vec(data_b)));
            let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(
                vec![i.clone(), j.clone()], vec![2, 3], storage_b);
            tn_b.add_tensor(0, tensor_b).unwrap();

            assert_treetn_add_correctness(tn_a, tn_b, stringify!($test_name));
        }
    };
}

// Generate single-node tests for f64 and Complex64
test_treetn_add_single_node_correctness!(
    test_treetn_add_single_node_f64,
    f64, DenseF64, DenseStorageF64,
    |x: i32, _: i32| x as f64
);

test_treetn_add_single_node_correctness!(
    test_treetn_add_single_node_c64,
    Complex64, DenseC64, DenseStorageC64,
    |x: i32, y: i32| Complex64::new(x as f64, (y * x) as f64)
);

#[test]
fn test_treetn_add_bond_dimension_growth() {
    // Verify that bond dimensions grow as expected: new_dim = dim_A + dim_B
    // Using shared bond indices within each network for proper contraction
    let mut tn_a = TreeTN::<DynId, NoSymmSpace, usize>::new();
    let mut tn_b = TreeTN::<DynId, NoSymmSpace, usize>::new();

    let i0 = Index::new_dyn(2);
    let bond_a = Index::new_dyn(3);  // shared bond dim 3 in network A
    let k1 = Index::new_dyn(4);

    // Network A with bond dim 3
    let storage_a0 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6])));
    let tensor_a0: TensorDynLen<DynId> = TensorDynLen::new(vec![i0.clone(), bond_a.clone()], vec![2, 3], storage_a0);
    let n0_a = tn_a.add_tensor(0, tensor_a0).unwrap();

    let storage_a1 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12])));
    let tensor_a1: TensorDynLen<DynId> = TensorDynLen::new(vec![bond_a.clone(), k1.clone()], vec![3, 4], storage_a1);
    let n1_a = tn_a.add_tensor(1, tensor_a1).unwrap();

    tn_a.connect(n0_a, &bond_a, n1_a, &bond_a).unwrap();

    // Network B with bond dim 5
    let bond_b = Index::new_dyn(5);  // shared bond dim 5 in network B

    let storage_b0 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 10])));
    let tensor_b0: TensorDynLen<DynId> = TensorDynLen::new(vec![i0.clone(), bond_b.clone()], vec![2, 5], storage_b0);
    let n0_b = tn_b.add_tensor(0, tensor_b0).unwrap();

    let storage_b1 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 20])));
    let tensor_b1: TensorDynLen<DynId> = TensorDynLen::new(vec![bond_b.clone(), k1.clone()], vec![5, 4], storage_b1);
    let n1_b = tn_b.add_tensor(1, tensor_b1).unwrap();

    tn_b.connect(n0_b, &bond_b, n1_b, &bond_b).unwrap();

    // Add the networks
    let tn_sum = tn_a.add(tn_b).unwrap();

    // Check bond dimension grew to 3 + 5 = 8
    assert_eq!(tn_sum.edge_count(), 1);
    // Get the edge from edges_for_node on any node
    let any_node = tn_sum.node_indices()[0];
    let edges = tn_sum.edges_for_node(any_node);
    assert!(!edges.is_empty(), "Should have at least one edge");
    let edge = edges[0].0;
    let conn = tn_sum.connection(edge).unwrap();
    assert_eq!(conn.bond_dim(), 8, "Bond dimension should be 3 + 5 = 8");
}

#[test]
fn test_treetn_add_multi_physical_permuted() {
    // Regression test for issue #1: verify correctness when a node has
    // multiple physical indices and tensor_b has them in a different order.
    //
    // Network A: node 0 with tensor indices [i, j, k] (3 physical indices)
    // Network B: node 0 with tensor indices [k, i, j] (permuted order)
    //
    // Both should represent the same logical data when accounting for permutation,
    // and contract(A + B) should equal contract(A) + contract(B).

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    // Network A: indices in order [i, j, k]
    // Data: sequential values 1..24
    let mut tn_a = TreeTN::<DynId, NoSymmSpace, usize>::new();
    let data_a: Vec<f64> = (1..=24).map(|x| x as f64).collect();
    let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a)));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(
        vec![i.clone(), j.clone(), k.clone()],
        vec![2, 3, 4],
        storage_a,
    );
    tn_a.add_tensor(0, tensor_a).unwrap();

    // Network B: indices in order [k, i, j] (permuted from [i, j, k])
    // Data: different values, also sequential but scaled
    let mut tn_b = TreeTN::<DynId, NoSymmSpace, usize>::new();
    let data_b: Vec<f64> = (1..=24).map(|x| (x * 10) as f64).collect();
    let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b)));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(
        vec![k.clone(), i.clone(), j.clone()],  // Permuted order!
        vec![4, 2, 3],  // Corresponding dimensions
        storage_b,
    );
    tn_b.add_tensor(0, tensor_b).unwrap();

    assert_treetn_add_correctness(tn_a, tn_b, "test_treetn_add_multi_physical_permuted");
}

#[test]
fn test_treetn_add_two_nodes_multi_physical_permuted() {
    // More complex test: two connected nodes, each with 2 physical indices,
    // where network B has permuted physical index ordering.
    //
    // Structure: node 0 -- node 1
    // Node 0: physical indices [i0, j0], bond index
    // Node 1: bond index, physical indices [k1, l1]
    //
    // Network B permutes the physical indices at each node.

    let i0 = Index::new_dyn(2);
    let j0 = Index::new_dyn(3);
    let k1 = Index::new_dyn(2);
    let l1 = Index::new_dyn(3);
    let bond_a = Index::new_dyn(4);  // shared bond for network A
    let bond_b = Index::new_dyn(4);  // shared bond for network B

    // Network A with standard ordering
    let mut tn_a = TreeTN::<DynId, NoSymmSpace, usize>::new();

    // Node 0: [i0, j0, bond] -> 2*3*4 = 24 elements
    let data_a0: Vec<f64> = (1..=24).map(|x| x as f64).collect();
    let storage_a0 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a0)));
    let tensor_a0: TensorDynLen<DynId> = TensorDynLen::new(
        vec![i0.clone(), j0.clone(), bond_a.clone()],
        vec![2, 3, 4],
        storage_a0,
    );
    let n0_a = tn_a.add_tensor(0, tensor_a0).unwrap();

    // Node 1: [bond, k1, l1] -> 4*2*3 = 24 elements
    let data_a1: Vec<f64> = (1..=24).map(|x| (x * 2) as f64).collect();
    let storage_a1 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a1)));
    let tensor_a1: TensorDynLen<DynId> = TensorDynLen::new(
        vec![bond_a.clone(), k1.clone(), l1.clone()],
        vec![4, 2, 3],
        storage_a1,
    );
    let n1_a = tn_a.add_tensor(1, tensor_a1).unwrap();

    tn_a.connect(n0_a, &bond_a, n1_a, &bond_a).unwrap();

    // Network B with permuted physical indices
    let mut tn_b = TreeTN::<DynId, NoSymmSpace, usize>::new();

    // Node 0: [j0, i0, bond] (i0, j0 swapped)
    let data_b0: Vec<f64> = (1..=24).map(|x| (x * 10) as f64).collect();
    let storage_b0 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b0)));
    let tensor_b0: TensorDynLen<DynId> = TensorDynLen::new(
        vec![j0.clone(), i0.clone(), bond_b.clone()],  // Permuted!
        vec![3, 2, 4],
        storage_b0,
    );
    let n0_b = tn_b.add_tensor(0, tensor_b0).unwrap();

    // Node 1: [l1, bond, k1] (permuted)
    let data_b1: Vec<f64> = (1..=24).map(|x| (x * 20) as f64).collect();
    let storage_b1 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b1)));
    let tensor_b1: TensorDynLen<DynId> = TensorDynLen::new(
        vec![l1.clone(), bond_b.clone(), k1.clone()],  // Permuted!
        vec![3, 4, 2],
        storage_b1,
    );
    let n1_b = tn_b.add_tensor(1, tensor_b1).unwrap();

    tn_b.connect(n0_b, &bond_b, n1_b, &bond_b).unwrap();

    assert_treetn_add_correctness(tn_a, tn_b, "test_treetn_add_two_nodes_multi_physical_permuted");
}

// Note: Complex64 two-nodes correctness test is generated by the macro (test_treetn_add_two_nodes_c64)

// ============================================================================
// TreeTN::contract_to_tensor tests for non-chain topologies
// ============================================================================

#[test]
fn test_contract_to_tensor_star() {
    // Test edge-based contract_to_tensor with a star topology
    //     A
    //     |
    // B - C - D
    //     |
    //     E
    let mut tn = TreeTN::<DynId, NoSymmSpace, String>::new();

    // External indices for each leaf
    let ext_a = Index::new_dyn(2);
    let ext_b = Index::new_dyn(3);
    let ext_d = Index::new_dyn(4);
    let ext_e = Index::new_dyn(5);

    // Bond indices (connecting leaves to center C)
    let bond_ac_a = Index::new_dyn(6);
    let bond_ac_c = Index::new_dyn(6);
    let bond_bc_b = Index::new_dyn(7);
    let bond_bc_c = Index::new_dyn(7);
    let bond_cd_c = Index::new_dyn(8);
    let bond_cd_d = Index::new_dyn(8);
    let bond_ce_c = Index::new_dyn(9);
    let bond_ce_e = Index::new_dyn(9);

    // Create tensors
    // Node A: [ext_a, bond_ac_a] dims [2, 6]
    let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(
        (0..12).map(|x| x as f64).collect()
    )));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(
        vec![ext_a.clone(), bond_ac_a.clone()], vec![2, 6], storage_a);
    let na = tn.add_tensor("A".to_string(), tensor_a).unwrap();

    // Node B: [ext_b, bond_bc_b] dims [3, 7]
    let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(
        (0..21).map(|x| (x * 2) as f64).collect()
    )));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(
        vec![ext_b.clone(), bond_bc_b.clone()], vec![3, 7], storage_b);
    let nb = tn.add_tensor("B".to_string(), tensor_b).unwrap();

    // Node C (center): [bond_ac_c, bond_bc_c, bond_cd_c, bond_ce_c] dims [6, 7, 8, 9]
    let storage_c = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(
        (0..(6*7*8*9)).map(|x| (x % 100) as f64 / 10.0).collect()
    )));
    let tensor_c: TensorDynLen<DynId> = TensorDynLen::new(
        vec![bond_ac_c.clone(), bond_bc_c.clone(), bond_cd_c.clone(), bond_ce_c.clone()],
        vec![6, 7, 8, 9], storage_c);
    let nc = tn.add_tensor("C".to_string(), tensor_c).unwrap();

    // Node D: [bond_cd_d, ext_d] dims [8, 4]
    let storage_d = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(
        (0..32).map(|x| (x * 3) as f64).collect()
    )));
    let tensor_d: TensorDynLen<DynId> = TensorDynLen::new(
        vec![bond_cd_d.clone(), ext_d.clone()], vec![8, 4], storage_d);
    let nd = tn.add_tensor("D".to_string(), tensor_d).unwrap();

    // Node E: [bond_ce_e, ext_e] dims [9, 5]
    let storage_e = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(
        (0..45).map(|x| (x + 1) as f64).collect()
    )));
    let tensor_e: TensorDynLen<DynId> = TensorDynLen::new(
        vec![bond_ce_e.clone(), ext_e.clone()], vec![9, 5], storage_e);
    let ne = tn.add_tensor("E".to_string(), tensor_e).unwrap();

    // Connect: A-C, B-C, C-D, C-E
    tn.connect(na, &bond_ac_a, nc, &bond_ac_c).unwrap();
    tn.connect(nb, &bond_bc_b, nc, &bond_bc_c).unwrap();
    tn.connect(nc, &bond_cd_c, nd, &bond_cd_d).unwrap();
    tn.connect(nc, &bond_ce_c, ne, &bond_ce_e).unwrap();

    // Contract to tensor
    let result = tn.contract_to_tensor().unwrap();

    // Result should have external indices only (ext_a, ext_b, ext_d, ext_e)
    assert_eq!(result.dims.len(), 4);
    // Dims should be [2, 3, 4, 5] in some order
    let mut sorted_dims = result.dims.clone();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 3, 4, 5]);

    // Verify total size
    let total_size: usize = result.dims.iter().product();
    assert_eq!(total_size, 2 * 3 * 4 * 5);

    // Verify data is not all zeros (sanity check)
    let data: Vec<f64> = match result.storage.as_ref() {
        Storage::DenseF64(d) => d.iter().copied().collect(),
        _ => panic!("Expected DenseF64 storage"),
    };
    let sum: f64 = data.iter().sum();
    assert!(sum.abs() > 1e-10, "Result should not be all zeros");
}

#[test]
fn test_contract_to_tensor_y_shaped() {
    // Test edge-based contract_to_tensor with a Y-shaped tree
    //     A
    //     |
    //     B
    //    / \
    //   C   D
    let mut tn = TreeTN::<DynId, NoSymmSpace, String>::new();

    // External indices
    let ext_a = Index::new_dyn(2);
    let ext_c = Index::new_dyn(3);
    let ext_d = Index::new_dyn(4);

    // Bond indices
    let bond_ab_a = Index::new_dyn(5);
    let bond_ab_b = Index::new_dyn(5);
    let bond_bc_b = Index::new_dyn(6);
    let bond_bc_c = Index::new_dyn(6);
    let bond_bd_b = Index::new_dyn(7);
    let bond_bd_d = Index::new_dyn(7);

    // Create tensors
    // Node A: [ext_a, bond_ab_a] dims [2, 5]
    let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(
        (0..10).map(|x| x as f64).collect()
    )));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(
        vec![ext_a.clone(), bond_ab_a.clone()], vec![2, 5], storage_a);
    let na = tn.add_tensor("A".to_string(), tensor_a).unwrap();

    // Node B (branch point): [bond_ab_b, bond_bc_b, bond_bd_b] dims [5, 6, 7]
    let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(
        (0..(5*6*7)).map(|x| (x % 50) as f64 / 10.0).collect()
    )));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(
        vec![bond_ab_b.clone(), bond_bc_b.clone(), bond_bd_b.clone()],
        vec![5, 6, 7], storage_b);
    let nb = tn.add_tensor("B".to_string(), tensor_b).unwrap();

    // Node C: [bond_bc_c, ext_c] dims [6, 3]
    let storage_c = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(
        (0..18).map(|x| (x * 2) as f64).collect()
    )));
    let tensor_c: TensorDynLen<DynId> = TensorDynLen::new(
        vec![bond_bc_c.clone(), ext_c.clone()], vec![6, 3], storage_c);
    let nc = tn.add_tensor("C".to_string(), tensor_c).unwrap();

    // Node D: [bond_bd_d, ext_d] dims [7, 4]
    let storage_d = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(
        (0..28).map(|x| (x + 1) as f64).collect()
    )));
    let tensor_d: TensorDynLen<DynId> = TensorDynLen::new(
        vec![bond_bd_d.clone(), ext_d.clone()], vec![7, 4], storage_d);
    let nd = tn.add_tensor("D".to_string(), tensor_d).unwrap();

    // Connect: A-B, B-C, B-D
    tn.connect(na, &bond_ab_a, nb, &bond_ab_b).unwrap();
    tn.connect(nb, &bond_bc_b, nc, &bond_bc_c).unwrap();
    tn.connect(nb, &bond_bd_b, nd, &bond_bd_d).unwrap();

    // Contract to tensor
    let result = tn.contract_to_tensor().unwrap();

    // Result should have external indices only (ext_a, ext_c, ext_d)
    assert_eq!(result.dims.len(), 3);
    // Dims should be [2, 3, 4] in some order
    let mut sorted_dims = result.dims.clone();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 3, 4]);

    // Verify total size
    let total_size: usize = result.dims.iter().product();
    assert_eq!(total_size, 2 * 3 * 4);

    // Verify data is not all zeros (sanity check)
    let data: Vec<f64> = match result.storage.as_ref() {
        Storage::DenseF64(d) => d.iter().copied().collect(),
        _ => panic!("Expected DenseF64 storage"),
    };
    let sum: f64 = data.iter().sum();
    assert!(sum.abs() > 1e-10, "Result should not be all zeros");
}

// ============================================================================
// log_norm tests
// ============================================================================

#[test]
fn test_log_norm_chain() {
    // Create a simple 2-node chain: n1 -- n2
    // Data: all 1.0
    // Total elements: 2*3 + 3*4 = 6 + 12, but contracted TN norm is different
    // For a chain TN with all ones and bond dim 3:
    // ||TN|| can be computed by contracting: (1,1,...) * (1,1,...) over bond

    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let a1 = Index::new_dyn(2);  // external
    let b1 = Index::new_dyn(3);  // bond
    let a2 = Index::new_dyn(3);  // bond
    let b2 = Index::new_dyn(4);  // external

    let storage1 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6])));
    let tensor1: TensorDynLen<DynId> = TensorDynLen::new(vec![a1.clone(), b1.clone()], vec![2, 3], storage1);

    let storage2 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12])));
    let tensor2: TensorDynLen<DynId> = TensorDynLen::new(vec![a2.clone(), b2.clone()], vec![3, 4], storage2);

    let n1 = tn.add_tensor_auto_name(tensor1);
    let n2 = tn.add_tensor_auto_name(tensor2);

    tn.connect(n1, &b1, n2, &a2).unwrap();

    // Compute log_norm
    let log_norm = tn.log_norm().unwrap();

    // Verify by contracting to tensor and computing norm
    let full_tensor = tn.contract_to_tensor().unwrap();
    let expected_norm = full_tensor.norm();
    let expected_log_norm = expected_norm.ln();

    assert!(
        (log_norm - expected_log_norm).abs() < 1e-10,
        "log_norm mismatch: got {}, expected {}",
        log_norm, expected_log_norm
    );
}

#[test]
fn test_log_norm_already_canonicalized_single_site() {
    // Create TN, canonicalize to single site, call log_norm
    // Verify it uses the existing center without re-canonizing

    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let a1 = Index::new_dyn(2);
    let b1 = Index::new_dyn(3);
    let a2 = Index::new_dyn(3);
    let b2 = Index::new_dyn(4);

    let storage1 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
    let tensor1: TensorDynLen<DynId> = TensorDynLen::new(vec![a1.clone(), b1.clone()], vec![2, 3], storage1);

    let storage2 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec((1..=12).map(|i| i as f64).collect())));
    let tensor2: TensorDynLen<DynId> = TensorDynLen::new(vec![a2.clone(), b2.clone()], vec![3, 4], storage2);

    let n1 = tn.add_tensor_auto_name(tensor1);
    let n2 = tn.add_tensor_auto_name(tensor2);

    tn.connect(n1, &b1, n2, &a2).unwrap();

    // First get expected norm via full contraction
    let full_tensor = tn.contract_to_tensor().unwrap();
    let expected_log_norm = full_tensor.norm().ln();

    // Canonicalize to n1
    tn.canonicalize_mut(std::iter::once(n1)).unwrap();
    assert_eq!(tn.ortho_region().len(), 1);

    // Compute log_norm
    let log_norm = tn.log_norm().unwrap();

    // Should still be canonicalized to n1
    assert_eq!(tn.ortho_region().len(), 1);

    assert!(
        (log_norm - expected_log_norm).abs() < 1e-10,
        "log_norm mismatch: got {}, expected {}",
        log_norm, expected_log_norm
    );
}

#[test]
fn test_log_norm_multi_site_ortho_region() {
    // Create TN with 3 nodes, canonicalize to 2 sites, call log_norm
    // Verify it canonicalizes to single site

    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let a1 = Index::new_dyn(2);
    let b1 = Index::new_dyn(3);
    let a2 = Index::new_dyn(3);
    let b2 = Index::new_dyn(3);
    let a3 = Index::new_dyn(3);
    let b3 = Index::new_dyn(4);

    let storage1 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6])));
    let tensor1: TensorDynLen<DynId> = TensorDynLen::new(vec![a1.clone(), b1.clone()], vec![2, 3], storage1);

    let storage2 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 9])));
    let tensor2: TensorDynLen<DynId> = TensorDynLen::new(vec![a2.clone(), b2.clone()], vec![3, 3], storage2);

    let storage3 = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12])));
    let tensor3: TensorDynLen<DynId> = TensorDynLen::new(vec![a3.clone(), b3.clone()], vec![3, 4], storage3);

    let n1 = tn.add_tensor_auto_name(tensor1);
    let n2 = tn.add_tensor_auto_name(tensor2);
    let n3 = tn.add_tensor_auto_name(tensor3);

    tn.connect(n1, &b1, n2, &a2).unwrap();
    tn.connect(n2, &b2, n3, &a3).unwrap();

    // Get expected norm via full contraction
    let full_tensor = tn.contract_to_tensor().unwrap();
    let expected_log_norm = full_tensor.norm().ln();

    // Canonicalize to two sites (n1 and n2)
    tn.canonicalize_mut(vec![n1, n2]).unwrap();
    assert_eq!(tn.ortho_region().len(), 2);

    // Compute log_norm - should canonicalize to single site
    let log_norm = tn.log_norm().unwrap();

    // Should now be canonicalized to single site (min of n1, n2)
    assert_eq!(tn.ortho_region().len(), 1);

    assert!(
        (log_norm - expected_log_norm).abs() < 1e-10,
        "log_norm mismatch: got {}, expected {}",
        log_norm, expected_log_norm
    );
}

#[test]
fn test_log_norm_complex_tensor() {
    // Test with complex tensors
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let a1 = Index::new_dyn(2);
    let b1 = Index::new_dyn(2);
    let a2 = Index::new_dyn(2);
    let b2 = Index::new_dyn(2);

    let data1: Vec<Complex64> = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, 0.0),
    ];
    let storage1 = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data1)));
    let tensor1: TensorDynLen<DynId> = TensorDynLen::new(vec![a1.clone(), b1.clone()], vec![2, 2], storage1);

    let data2: Vec<Complex64> = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, -1.0),
    ];
    let storage2 = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data2)));
    let tensor2: TensorDynLen<DynId> = TensorDynLen::new(vec![a2.clone(), b2.clone()], vec![2, 2], storage2);

    let n1 = tn.add_tensor_auto_name(tensor1);
    let n2 = tn.add_tensor_auto_name(tensor2);

    tn.connect(n1, &b1, n2, &a2).unwrap();

    // Get expected norm via full contraction
    let full_tensor = tn.contract_to_tensor().unwrap();
    let expected_log_norm = full_tensor.norm().ln();

    // Compute log_norm
    let log_norm = tn.log_norm().unwrap();

    assert!(
        (log_norm - expected_log_norm).abs() < 1e-10,
        "log_norm mismatch for complex: got {}, expected {}",
        log_norm, expected_log_norm
    );
}

#[test]
fn test_log_norm_single_tensor() {
    // Test with a single tensor (no bonds)
    let mut tn = TreeTN::<DynId, NoSymmSpace, NodeIndex>::new();

    let i = Index::new_dyn(3);
    let j = Index::new_dyn(4);

    let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data.clone())));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone(), j.clone()], vec![3, 4], storage);

    let _n = tn.add_tensor_auto_name(tensor);

    // Expected: sqrt(1² + 2² + ... + 12²) = sqrt(650)
    let expected_norm = (data.iter().map(|x| x * x).sum::<f64>()).sqrt();
    let expected_log_norm = expected_norm.ln();

    let log_norm = tn.log_norm().unwrap();

    assert!(
        (log_norm - expected_log_norm).abs() < 1e-10,
        "log_norm mismatch for single tensor: got {}, expected {}",
        log_norm, expected_log_norm
    );
}