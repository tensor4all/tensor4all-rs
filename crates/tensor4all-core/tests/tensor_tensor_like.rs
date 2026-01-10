//! Tests for TensorLike trait implementation.

use tensor4all_core::index::{DynId, Index};
use tensor4all_core::DynIndex;
use tensor4all_core::{StorageScalar, TensorDynLen, TensorLike};

/// Helper to create a simple tensor with given dimensions
fn make_tensor(dims: &[usize]) -> TensorDynLen {
    let indices: Vec<DynIndex> = dims.iter().map(|&d| Index::new_dyn(d)).collect();
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
    use tensor4all_core::index_like::IndexLike;
    assert_eq!(external_indices[0].dim(), 2);
    assert_eq!(external_indices[1].dim(), 3);
    assert_eq!(external_indices[2].dim(), 4);
}

#[test]
fn test_tensor_like_num_external_indices() {
    let tensor = make_tensor(&[5, 6]);

    assert_eq!(tensor.num_external_indices(), 2);
}

#[test]
fn test_tensor_like_tensordot_basic() {
    // Create two tensors: A(i,j) and B(j,k)
    // Contract over j to get C(i,k)
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);
    let k = Index::<DynId>::new_dyn(4);

    // Tensor A: 2x3 matrix
    let a_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = TensorDynLen::from_indices(vec![i.clone(), j.clone()], f64::dense_storage(a_data));

    // Tensor B: 3x4 matrix (use a copy of j with same id)
    let j_copy = Index::new(j.id.clone(), j.dim);
    let b_data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let b = TensorDynLen::from_indices(vec![j_copy.clone(), k.clone()], f64::dense_storage(b_data));

    // Create pairs with DynIndex
    let pairs: Vec<(DynIndex, DynIndex)> = vec![(j.clone(), j_copy.clone())];

    // Use TensorLike::tensordot
    let c = <TensorDynLen as TensorLike>::tensordot(&a, &b, &pairs)
        .expect("tensordot should succeed");

    // Result should be 2x4
    assert_eq!(c.dims, vec![2, 4]);
}

// Note: trait object tests removed - TensorLike is now fully generic and does not support dyn
