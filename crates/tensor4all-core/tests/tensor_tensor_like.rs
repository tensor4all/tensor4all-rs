//! Tests for TensorLike trait implementation.

use tensor4all_core::index::{DynId, Index, NoSymmSpace};
use tensor4all_core::DefaultTagSet;
use tensor4all_core::{StorageScalar, TensorDynLen, TensorLike, TensorLikeDowncast};

/// Helper to create a simple tensor with given dimensions
fn make_tensor(dims: &[usize]) -> TensorDynLen<DynId> {
    let indices: Vec<Index<DynId, NoSymmSpace>> = dims.iter().map(|&d| Index::new_dyn(d)).collect();
    let total_size: usize = dims.iter().product();
    let data: Vec<f64> = (0..total_size).map(|i| i as f64).collect();
    let storage = f64::dense_storage(data);
    TensorDynLen::from_indices(indices, storage)
}

#[test]
fn test_tensor_like_external_indices() {
    let tensor = make_tensor(&[2, 3, 4]);

    // Use TensorLike trait
    let external_indices = tensor.external_indices();
    assert_eq!(external_indices.len(), 3);

    // Check dimensions through the indices
    assert_eq!(external_indices[0].size(), 2);
    assert_eq!(external_indices[1].size(), 3);
    assert_eq!(external_indices[2].size(), 4);
}

#[test]
fn test_tensor_like_num_external_indices() {
    let tensor = make_tensor(&[5, 6]);

    assert_eq!(tensor.num_external_indices(), 2);
}

#[test]
fn test_tensor_like_to_tensor() {
    let original = make_tensor(&[2, 3]);

    // to_tensor should return a clone
    let cloned = <TensorDynLen<DynId> as TensorLike>::to_tensor(&original)
        .expect("to_tensor should succeed");

    assert_eq!(cloned.indices.len(), original.indices.len());
    assert_eq!(cloned.dims, original.dims);
}

#[test]
fn test_tensor_like_tensordot_basic() {
    // Create two tensors: A(i,j) and B(j,k)
    // Contract over j to get C(i,k)
    let i = Index::<DynId, NoSymmSpace>::new_dyn(2);
    let j = Index::<DynId, NoSymmSpace>::new_dyn(3);
    let k = Index::<DynId, NoSymmSpace>::new_dyn(4);

    // Tensor A: 2x3 matrix
    let a_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = TensorDynLen::from_indices(vec![i.clone(), j.clone()], f64::dense_storage(a_data));

    // Tensor B: 3x4 matrix (use a copy of j with same id)
    let j_copy = Index::new(j.id, j.symm.clone());
    let b_data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let b = TensorDynLen::from_indices(vec![j_copy.clone(), k.clone()], f64::dense_storage(b_data));

    // Create pairs with DefaultTagSet
    let pairs = vec![(
        Index::<DynId, NoSymmSpace, DefaultTagSet>::new_with_tags(
            j.id,
            j.symm.clone(),
            DefaultTagSet::default(),
        ),
        Index::<DynId, NoSymmSpace, DefaultTagSet>::new_with_tags(
            j_copy.id,
            j_copy.symm.clone(),
            DefaultTagSet::default(),
        ),
    )];

    // Use TensorLike::tensordot (via default implementation)
    let c = <TensorDynLen<DynId> as TensorLike>::tensordot(&a, &b, &pairs)
        .expect("tensordot should succeed");

    // Result should be 2x4
    assert_eq!(c.dims, vec![2, 4]);
}

#[test]
fn test_tensor_like_object_safety() {
    let tensor = make_tensor(&[2, 3]);

    // Use trait object
    let trait_obj: &dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = DefaultTagSet> = &tensor;

    // Should be able to call methods on trait object
    assert_eq!(trait_obj.num_external_indices(), 2);

    let external = trait_obj.external_indices();
    assert_eq!(external.len(), 2);
}

#[test]
fn test_tensor_like_clone_trait_object() {
    let tensor = make_tensor(&[2, 3]);

    // Box the tensor as a trait object
    let boxed: Box<dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = DefaultTagSet>> =
        Box::new(tensor);

    // Clone the boxed trait object (via dyn-clone)
    let cloned = dyn_clone::clone_box(&*boxed);

    // Both should have the same properties
    assert_eq!(boxed.num_external_indices(), cloned.num_external_indices());
}

#[test]
fn test_tensor_like_as_any() {
    let tensor = make_tensor(&[2, 3]);

    // Get as Any
    let any_ref = tensor.as_any();

    // Should be able to downcast back to TensorDynLen<DynId>
    let downcast = any_ref.downcast_ref::<TensorDynLen<DynId>>();
    assert!(downcast.is_some());

    let downcast_tensor = downcast.unwrap();
    assert_eq!(downcast_tensor.dims, vec![2, 3]);
}

#[test]
fn test_tensor_like_downcast_via_trait_object() {
    let tensor = make_tensor(&[2, 3]);

    // Use as trait object
    let trait_obj: &dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = DefaultTagSet> = &tensor;

    // Use TensorLikeDowncast extension trait
    assert!(trait_obj.is::<TensorDynLen<DynId>>());

    let downcast = trait_obj.downcast_ref::<TensorDynLen<DynId>>();
    assert!(downcast.is_some());
    assert_eq!(downcast.unwrap().dims, vec![2, 3]);

    // Should not downcast to wrong type
    assert!(!trait_obj.is::<String>());
    assert!(trait_obj.downcast_ref::<String>().is_none());
}

#[test]
fn test_tensor_like_downcast_boxed() {
    let tensor = make_tensor(&[4, 5]);

    // Box as trait object
    let boxed: Box<dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = DefaultTagSet>> =
        Box::new(tensor);

    // Downcast via as_any
    let downcast = boxed.as_any().downcast_ref::<TensorDynLen<DynId>>();
    assert!(downcast.is_some());
    assert_eq!(downcast.unwrap().dims, vec![4, 5]);

    // Also via extension trait
    assert!(boxed.is::<TensorDynLen<DynId>>());
}
