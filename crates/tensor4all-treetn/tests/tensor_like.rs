//! Tests for TreeTN operations.
//!
//! Note: TreeTN does NOT implement TensorLike (see TENSOR_LIKE_DESIGN.md).
//! This file tests TreeTN's own API which provides similar functionality:
//! - `site_index_network()` instead of `external_indices()`
//! - `contract_to_tensor()` instead of `to_tensor()`
//!
//! Migrated from tensor4all-rs main branch with type adjustments:
//! - TreeTN<DynId, NoSymmSpace, String> -> TreeTN<TensorDynLen, String>
//! - Index<DynId> -> DynIndex
//! - TensorDynLen<DynId, NoSymmSpace> -> TensorDynLen
//! - idx.id -> idx.id() (IndexLike trait method)

use tensor4all_core::{DynIndex, IndexLike, StorageScalar, TensorDynLen, TensorLike};
use tensor4all_treetn::TreeTN;

/// Helper to create a simple tensor with given indices
fn make_tensor(indices: Vec<DynIndex>) -> TensorDynLen {
    let total_size: usize = indices.iter().map(|idx| idx.dim()).product();
    let data: Vec<f64> = (0..total_size).map(|i| i as f64).collect();
    let storage = f64::dense_storage(data);
    TensorDynLen::from_indices(indices, storage)
}

/// Helper to collect all site (physical) indices from a TreeTN.
/// This corresponds to `external_indices()` in the original TensorLike trait.
fn collect_site_indices(tn: &TreeTN<TensorDynLen, String>) -> Vec<DynIndex> {
    let mut indices = Vec::new();
    for node_name in tn.node_names() {
        if let Some(site_space) = tn.site_space(&node_name) {
            indices.extend(site_space.iter().cloned());
        }
    }
    indices
}

#[test]
fn test_treetn_site_indices_single_node() {
    // Create a TreeTN with a single node containing a 2x3 tensor
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let tensor = make_tensor(vec![i.clone(), j.clone()]);

    // Use from_tensors to create the network
    let tn =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    // Site indices should be all indices (since no connections)
    let site_indices = collect_site_indices(&tn);
    assert_eq!(site_indices.len(), 2);

    // Check that the indices have the correct dimensions
    let sizes: Vec<usize> = site_indices.iter().map(|idx| idx.dim()).collect();
    assert!(sizes.contains(&2));
    assert!(sizes.contains(&3));
}

#[test]
fn test_treetn_site_indices_connected_nodes() {
    // Create a TreeTN with two connected nodes
    // Node A: indices (i, bond_ab)
    // Node B: indices (bond_ab, j)
    // After connecting, site indices should be (i, j) - not bond_ab

    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let bond_ab = DynIndex::new_dyn(4);

    let tensor_a = make_tensor(vec![i.clone(), bond_ab.clone()]);
    let tensor_b = make_tensor(vec![bond_ab.clone(), j.clone()]);

    // from_tensors automatically connects tensors that share common indices
    let tn = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![tensor_a, tensor_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Site indices should be i and j (not bond_ab)
    let site_indices = collect_site_indices(&tn);
    assert_eq!(site_indices.len(), 2);

    // Verify the site indices are i and j (not bond)
    let site_ids: Vec<_> = site_indices.iter().map(|idx| idx.id()).collect();
    assert!(site_ids.contains(&i.id()));
    assert!(site_ids.contains(&j.id()));
    assert!(!site_ids.contains(&bond_ab.id()));
}

#[test]
fn test_treetn_num_site_indices() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(4);

    let tensor_a = make_tensor(vec![i.clone(), bond.clone()]);
    let tensor_b = make_tensor(vec![bond.clone(), j.clone()]);

    let tn = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![tensor_a, tensor_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    let site_indices = collect_site_indices(&tn);
    assert_eq!(site_indices.len(), 2);
}

#[test]
fn test_treetn_contract_to_tensor() {
    // Create a simple MPS-like network: A -- B
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(2);

    // Tensor A: 2x2 with values
    let tensor_a = TensorDynLen::from_indices(
        vec![i.clone(), bond.clone()],
        f64::dense_storage(vec![1.0, 0.0, 0.0, 1.0]), // identity-like
    );

    // Tensor B: 2x3 with values
    let tensor_b = TensorDynLen::from_indices(
        vec![bond.clone(), j.clone()],
        f64::dense_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );

    let tn = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![tensor_a, tensor_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Contract to tensor using the explicit method (exponential cost!)
    let result = tn.contract_to_tensor().expect("contract_to_tensor should succeed");

    // Result should be 2x3 (dimensions of i and j)
    assert_eq!(result.external_indices().len(), 2);
    let dims: Vec<usize> = result.external_indices().iter().map(|idx| idx.dim()).collect();
    assert!(dims.contains(&2));
    assert!(dims.contains(&3));
}

#[test]
fn test_treetn_site_indices_deterministic_ordering() {
    // Create a TreeTN with multiple nodes and verify ordering is deterministic
    let idx_a = DynIndex::new_dyn(2);
    let idx_b = DynIndex::new_dyn(3);
    let idx_c = DynIndex::new_dyn(4);

    // Add nodes in non-alphabetical order (C, A, B)
    let tn = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![
            make_tensor(vec![idx_c.clone()]),
            make_tensor(vec![idx_a.clone()]),
            make_tensor(vec![idx_b.clone()]),
        ],
        vec!["C".to_string(), "A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Site indices - node names are deterministic (from the same TreeTN)
    let site_indices = collect_site_indices(&tn);
    assert_eq!(site_indices.len(), 3);

    // First call
    let ids1: Vec<_> = site_indices.iter().map(|idx| idx.id()).collect();

    // Second call should give same order
    let site_indices2 = collect_site_indices(&tn);
    let ids2: Vec<_> = site_indices2.iter().map(|idx| idx.id()).collect();

    assert_eq!(ids1, ids2, "site_indices should be deterministic");
}

// Note: The following tests from the original file have been removed because
// TreeTN does NOT implement TensorLike in the index-like branch.
// See TENSOR_LIKE_DESIGN.md for the design rationale.
//
// Removed tests:
// - test_treetn_tensor_like_object_safety
// - test_treetn_clone_trait_object
// - test_treetn_as_any
// - test_treetn_downcast_via_trait_object
// - test_mixed_downcast
//
// These tests tested dyn TensorLike functionality which no longer exists.
// For heterogeneous tensor collections, use an enum wrapper instead.
