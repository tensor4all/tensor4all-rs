//! Tests for TensorLike trait implementation for TreeTN.

use tensor4all_treetn::TreeTN;
use tensor4all_core::index::{Index, DynId, NoSymmSpace};
use tensor4all_core::{TensorDynLen, TensorLike, TensorLikeDowncast, DefaultTagSet, StorageScalar};

/// Helper to create a simple tensor with given indices
fn make_tensor(indices: Vec<Index<DynId, NoSymmSpace>>) -> TensorDynLen<DynId> {
    let total_size: usize = indices.iter().map(|idx| idx.size()).product();
    let data: Vec<f64> = (0..total_size).map(|i| i as f64).collect();
    let storage = f64::dense_storage(data);
    TensorDynLen::from_indices(indices, storage)
}

#[test]
fn test_treetn_external_indices_single_node() {
    // Create a TreeTN with a single node containing a 2x3 tensor
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let tensor = make_tensor(vec![i.clone(), j.clone()]);

    // Use new to create the network
    let tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
        vec![tensor],
        vec!["A".to_string()],
    ).unwrap();

    // External indices should be all indices (since no connections)
    let external = tn.external_indices();
    assert_eq!(external.len(), 2);

    // Check that the indices have the correct dimensions
    let sizes: Vec<usize> = external.iter().map(|idx| idx.size()).collect();
    assert!(sizes.contains(&2));
    assert!(sizes.contains(&3));
}

#[test]
fn test_treetn_external_indices_connected_nodes() {
    // Create a TreeTN with two connected nodes
    // Node A: indices (i, bond_ab)
    // Node B: indices (bond_ab, j)
    // After connecting, external indices should be (i, j)

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let bond_ab = Index::new_dyn(4);

    let tensor_a = make_tensor(vec![i.clone(), bond_ab.clone()]);
    let tensor_b = make_tensor(vec![bond_ab.clone(), j.clone()]);

    // new automatically connects tensors that share common indices
    let tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
        vec![tensor_a, tensor_b],
        vec!["A".to_string(), "B".to_string()],
    ).unwrap();

    // External indices should be i and j (not bond_ab)
    let external = tn.external_indices();
    assert_eq!(external.len(), 2);

    // Verify the external indices are i and j (not bond)
    let external_ids: Vec<_> = external.iter().map(|idx| idx.id).collect();
    assert!(external_ids.contains(&i.id));
    assert!(external_ids.contains(&j.id));
    assert!(!external_ids.contains(&bond_ab.id));
}

#[test]
fn test_treetn_num_external_indices() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let bond = Index::new_dyn(4);

    let tensor_a = make_tensor(vec![i.clone(), bond.clone()]);
    let tensor_b = make_tensor(vec![bond.clone(), j.clone()]);

    let tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
        vec![tensor_a, tensor_b],
        vec!["A".to_string(), "B".to_string()],
    ).unwrap();

    assert_eq!(tn.num_external_indices(), 2);
}

#[test]
fn test_treetn_to_tensor() {
    // Create a simple MPS-like network: A -- B
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let bond = Index::new_dyn(2);

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

    let tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
        vec![tensor_a, tensor_b],
        vec!["A".to_string(), "B".to_string()],
    ).unwrap();

    // Contract to tensor
    let result = tn.to_tensor().expect("to_tensor should succeed");

    // Result should be 2x3 (dimensions of i and j)
    assert_eq!(result.indices.len(), 2);
    let dims: Vec<usize> = result.indices.iter().map(|idx| idx.size()).collect();
    assert!(dims.contains(&2));
    assert!(dims.contains(&3));
}

#[test]
fn test_treetn_external_indices_deterministic_ordering() {
    // Create a TreeTN with multiple nodes and verify ordering is deterministic
    let idx_a = Index::new_dyn(2);
    let idx_b = Index::new_dyn(3);
    let idx_c = Index::new_dyn(4);

    // Add nodes in non-alphabetical order (C, A, B)
    let tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
        vec![
            make_tensor(vec![idx_c.clone()]),
            make_tensor(vec![idx_a.clone()]),
            make_tensor(vec![idx_b.clone()]),
        ],
        vec!["C".to_string(), "A".to_string(), "B".to_string()],
    ).unwrap();

    // External indices should be sorted by node name first
    let external = tn.external_indices();
    assert_eq!(external.len(), 3);

    // First call
    let ids1: Vec<DynId> = external.iter().map(|idx| idx.id).collect();

    // Second call should give same order
    let external2 = tn.external_indices();
    let ids2: Vec<DynId> = external2.iter().map(|idx| idx.id).collect();

    assert_eq!(ids1, ids2, "external_indices should be deterministic");
}

#[test]
fn test_treetn_tensor_like_object_safety() {
    let i = Index::new_dyn(2);
    let tensor = make_tensor(vec![i.clone()]);

    let tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
        vec![tensor],
        vec!["A".to_string()],
    ).unwrap();

    // Use as trait object
    let trait_obj: &dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = DefaultTagSet> = &tn;

    assert_eq!(trait_obj.num_external_indices(), 1);
    assert_eq!(trait_obj.external_indices().len(), 1);
}

#[test]
fn test_treetn_clone_trait_object() {
    let i = Index::new_dyn(2);
    let tensor = make_tensor(vec![i.clone()]);

    let tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
        vec![tensor],
        vec!["A".to_string()],
    ).unwrap();

    // Box as trait object
    let boxed: Box<dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = DefaultTagSet>> =
        Box::new(tn);

    // Clone via dyn-clone
    let cloned = dyn_clone::clone_box(&*boxed);

    assert_eq!(boxed.num_external_indices(), cloned.num_external_indices());
}

#[test]
fn test_treetn_as_any() {
    let i = Index::new_dyn(2);
    let tensor = make_tensor(vec![i.clone()]);

    let tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
        vec![tensor],
        vec!["A".to_string()],
    ).unwrap();

    // Get as Any
    let any_ref = tn.as_any();

    // Should be able to downcast back to TreeTN
    let downcast = any_ref.downcast_ref::<TreeTN<DynId, NoSymmSpace, String>>();
    assert!(downcast.is_some());

    let downcast_tn = downcast.unwrap();
    assert_eq!(downcast_tn.node_count(), 1);
}

#[test]
fn test_treetn_downcast_via_trait_object() {
    let i = Index::new_dyn(2);
    let tensor = make_tensor(vec![i.clone()]);

    let tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
        vec![tensor],
        vec!["A".to_string()],
    ).unwrap();

    // Use as trait object
    let trait_obj: &dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = DefaultTagSet> = &tn;

    // Use TensorLikeDowncast extension trait
    assert!(trait_obj.is::<TreeTN<DynId, NoSymmSpace, String>>());

    let downcast = trait_obj.downcast_ref::<TreeTN<DynId, NoSymmSpace, String>>();
    assert!(downcast.is_some());
    assert_eq!(downcast.unwrap().node_count(), 1);

    // Should not downcast to wrong type
    assert!(!trait_obj.is::<TensorDynLen<DynId>>());
    assert!(trait_obj.downcast_ref::<TensorDynLen<DynId>>().is_none());
}

#[test]
fn test_mixed_downcast() {
    // Create a TensorDynLen and a TreeTN, put them in a Vec of trait objects,
    // and show we can downcast each to the correct type.

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let tensor = make_tensor(vec![i.clone(), j.clone()]);
    let tn = TreeTN::<DynId, NoSymmSpace, String>::from_tensors(
        vec![make_tensor(vec![i.clone()])],
        vec!["A".to_string()],
    ).unwrap();

    let objects: Vec<Box<dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = DefaultTagSet>>> = vec![
        Box::new(tensor),
        Box::new(tn),
    ];

    // First should be TensorDynLen
    assert!(objects[0].is::<TensorDynLen<DynId>>());
    assert!(!objects[0].is::<TreeTN<DynId, NoSymmSpace, String>>());

    // Second should be TreeTN
    assert!(!objects[1].is::<TensorDynLen<DynId>>());
    assert!(objects[1].is::<TreeTN<DynId, NoSymmSpace, String>>());

    // Downcast and verify properties
    let tensor_ref = objects[0].downcast_ref::<TensorDynLen<DynId>>().unwrap();
    assert_eq!(tensor_ref.num_external_indices(), 2);

    let tn_ref = objects[1].downcast_ref::<TreeTN<DynId, NoSymmSpace, String>>().unwrap();
    assert_eq!(tn_ref.node_count(), 1);
}
