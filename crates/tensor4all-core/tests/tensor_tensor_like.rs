//! Tests for TensorLike trait implementation.

use tensor4all_core::index::{DynId, Index};
use tensor4all_core::DynIndex;
use tensor4all_core::{AllowedPairs, StorageScalar, TensorDynLen, TensorIndex, TensorLike};

/// Helper to create a simple tensor with given dimensions
fn make_tensor(dims: &[usize]) -> TensorDynLen {
    let indices: Vec<DynIndex> = dims.iter().map(|&d| Index::new_dyn(d)).collect();
    let total_size: usize = dims.iter().product();
    let data: Vec<f64> = (0..total_size).map(|i| i as f64).collect();
    let storage = f64::dense_storage_with_shape(data, dims);
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
fn test_tensor_like_contract_basic() {
    // Create two tensors: A(i,j) and B(j,k)
    // Contract over j to get C(i,k)
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);
    let k = Index::<DynId>::new_dyn(4);

    // Tensor A: 2x3 matrix
    let a_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = TensorDynLen::from_indices(vec![i.clone(), j.clone()], f64::dense_storage_with_shape(a_data, &[2, 3]));

    // Tensor B: 3x4 matrix (use a copy of j with same id)
    let j_copy = Index::new(j.id, j.dim);
    let b_data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let b = TensorDynLen::from_indices(vec![j_copy.clone(), k.clone()], f64::dense_storage_with_shape(b_data, &[3, 4]));

    // Use TensorLike::contract - auto-detects contractable pairs via is_contractable
    let c = <TensorDynLen as TensorLike>::contract(&[&a, &b], AllowedPairs::All)
        .expect("contract should succeed");

    // Result should be 2x4
    assert_eq!(c.dims, vec![2, 4]);
}

#[test]
fn test_contract_allowed_pairs_specified() {
    // Create three tensors: A(i,j), B(j,k), C(k,l)
    // With AllowedPairs::Specified(&[(0, 1), (1, 2)]) - A-B and B-C pairs allowed
    // j is shared between A and B, k is shared between B and C
    // All tensors form a connected chain: A-j-B-k-C
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);
    let k = Index::<DynId>::new_dyn(4);
    let l = Index::<DynId>::new_dyn(5);

    // Tensor A: 2x3 matrix (i, j)
    let a_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = TensorDynLen::from_indices(vec![i.clone(), j.clone()], f64::dense_storage_with_shape(a_data, &[2, 3]));

    // Tensor B: 3x4 matrix (j, k) - j has same id as A's j
    let j_copy = Index::new(j.id, j.dim);
    let b_data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let b = TensorDynLen::from_indices(vec![j_copy.clone(), k.clone()], f64::dense_storage_with_shape(b_data, &[3, 4]));

    // Tensor C: 4x5 matrix (k, l) - k has same id as B's k
    let k_copy = Index::new(k.id, k.dim);
    let c_data: Vec<f64> = (0..20).map(|x| x as f64).collect();
    let c = TensorDynLen::from_indices(vec![k_copy.clone(), l.clone()], f64::dense_storage_with_shape(c_data, &[4, 5]));

    // Contract with specified pairs
    // j is contracted between A and B (in pair (0,1))
    // k is contracted between B and C (in pair (1,2))
    let result = <TensorDynLen as TensorLike>::contract(
        &[&a, &b, &c],
        AllowedPairs::Specified(&[(0, 1), (1, 2)]),
    )
    .expect("contract should succeed");

    // Result should have: i (from A, dim=2), l (from C, dim=5)
    // j and k are contracted
    assert_eq!(result.dims.len(), 2);
    let mut sorted_dims = result.dims.clone();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 5]);
}

#[test]
fn test_contract_specified_empty_with_common_indices_errors() {
    // AllowedPairs::Specified(&[]) with tensors that share index IDs should error
    // because outer_product requires tensors to have no common indices
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);

    // Tensor A: 2x3 matrix
    let a_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = TensorDynLen::from_indices(vec![i.clone(), j.clone()], f64::dense_storage_with_shape(a_data, &[2, 3]));

    // Tensor B: 2x3 matrix (use copies of i and j with same ids)
    let i_copy = Index::new(i.id, i.dim);
    let j_copy = Index::new(j.id, j.dim);
    let b_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let b = TensorDynLen::from_indices(
        vec![i_copy.clone(), j_copy.clone()],
        f64::dense_storage_with_shape(b_data, &[2, 3]),
    );

    // With empty allowed pairs and tensors that share index IDs,
    // outer_product will fail because tensors have common indices
    let result = <TensorDynLen as TensorLike>::contract(&[&a, &b], AllowedPairs::Specified(&[]));

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("common indices"));
}

#[test]
fn test_contract_specified_empty_outer_product() {
    // AllowedPairs::Specified(&[]) with tensors that have different index IDs
    // should succeed via outer product
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);
    let k = Index::<DynId>::new_dyn(4);
    let l = Index::<DynId>::new_dyn(5);

    // Tensor A: 2x3 matrix with indices (i, j)
    let a_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = TensorDynLen::from_indices(vec![i.clone(), j.clone()], f64::dense_storage_with_shape(a_data, &[2, 3]));

    // Tensor B: 4x5 matrix with indices (k, l) - different from a
    let b_data: Vec<f64> = (0..20).map(|x| x as f64).collect();
    let b = TensorDynLen::from_indices(vec![k.clone(), l.clone()], f64::dense_storage_with_shape(b_data, &[4, 5]));

    // With empty allowed pairs and different index IDs, outer product succeeds
    let result =
        <TensorDynLen as TensorLike>::contract(&[&a, &b], AllowedPairs::Specified(&[])).unwrap();

    // Result should have 4 indices (i, j, k, l)
    assert_eq!(result.dims.len(), 4);
    let mut sorted_dims = result.dims.clone();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 3, 4, 5]);
}

#[test]
fn test_contract_specified_disconnected_outer_product() {
    // AllowedPairs::Specified(&[(0,1), (2,3)]) with 4 tensors
    // This creates a disconnected graph: {A,B} and {C,D}
    // Each component contracts within itself, then outer product combines them
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);

    let a = TensorDynLen::from_indices(vec![i.clone()], f64::dense_storage_with_shape(vec![1.0, 2.0], &[2]));
    let i_copy = Index::new(i.id, i.dim);
    let b = TensorDynLen::from_indices(vec![i_copy.clone()], f64::dense_storage_with_shape(vec![3.0, 4.0], &[2]));
    let c = TensorDynLen::from_indices(vec![j.clone()], f64::dense_storage_with_shape(vec![5.0, 6.0, 7.0], &[3]));
    let j_copy = Index::new(j.id, j.dim);
    let d = TensorDynLen::from_indices(
        vec![j_copy.clone()],
        f64::dense_storage_with_shape(vec![8.0, 9.0, 10.0], &[3]),
    );

    // Disconnected pairs: (0,1) and (2,3)
    // A and B contract i, C and D contract j, then outer product combines results
    let result = <TensorDynLen as TensorLike>::contract(
        &[&a, &b, &c, &d],
        AllowedPairs::Specified(&[(0, 1), (2, 3)]),
    )
    .unwrap();

    // A(i) * B(i) contracts to scalar (dim 0)
    // C(j) * D(j) contracts to scalar (dim 0)
    // Outer product of two scalars is a scalar
    assert_eq!(result.dims.len(), 0);
}

// Note: trait object tests removed - TensorLike is now fully generic and does not support dyn
