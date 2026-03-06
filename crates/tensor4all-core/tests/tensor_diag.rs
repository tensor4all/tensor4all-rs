use num_complex::Complex64;
use std::sync::Arc;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::TensorLike;
use tensor4all_core::{
    diag_tensor_dyn_len, diag_tensor_dyn_len_c64, is_diag_tensor, AnyScalar, Storage, TensorDynLen,
};

#[test]
fn test_diag_tensor_creation() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];

    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_data.clone());
    assert_eq!(tensor.dims(), vec![3, 3]);
    assert!(is_diag_tensor(&tensor));
}

#[test]
#[should_panic(expected = "DiagTensor requires all indices to have the same dimension")]
fn test_diag_tensor_validation_different_dims() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0];

    let _tensor = diag_tensor_dyn_len(vec![i, j], diag_data);
}

#[test]
fn test_diag_tensor_sum() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];

    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_data);
    let sum: AnyScalar = tensor.sum();
    assert!(!sum.is_complex());
    assert!((sum.real() - 6.0).abs() < 1e-10);
}

#[test]
fn test_diag_tensor_scale_preserves_diag_structure() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], vec![1.0, -2.0, 4.0]);

    let scaled = tensor.scale(AnyScalar::new_real(-0.5)).unwrap();

    assert!(is_diag_tensor(&scaled));
    let expected = diag_tensor_dyn_len(vec![i, j], vec![-0.5, 1.0, -2.0]);
    assert!(scaled.isapprox(&expected, 1e-12, 0.0));
}

#[test]
fn test_diag_tensor_permute() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];

    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone(), k.clone()], diag_data.clone());

    // Permute: data should not change for DiagTensor
    let permuted = tensor.permute(&[2, 0, 1]);
    assert_eq!(permuted.dims(), vec![3, 3, 3]);
    let expected = diag_tensor_dyn_len(vec![k, i, j], diag_data);
    assert!(permuted.isapprox(&expected, 1e-12, 0.0));
}

#[test]
fn test_diag_tensor_contract_diag_diag_all_contracted() {
    // Create two 2x2 DiagTensors and contract all indices
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let diag_a = vec![1.0, 2.0];
    let diag_b = vec![3.0, 4.0];

    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_a);
    let tensor_b = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_b);

    // Contract all indices: result should be scalar (inner product)
    let result = tensor_a.contract(&tensor_b);

    // Result should be scalar: 1*3 + 2*4 = 11
    assert_eq!(result.dims().len(), 0);
    assert!((result.only().real() - 11.0).abs() < 1e-12);
}

#[test]
fn test_diag_tensor_contract_diag_diag_partial() {
    // Create A[i, j] and B[j, k], contract along j
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let diag_a = vec![1.0, 2.0, 3.0];
    let diag_b = vec![4.0, 5.0, 6.0];

    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_a);
    let tensor_b = diag_tensor_dyn_len(vec![j.clone(), k.clone()], diag_b);

    // Contract along j: result should be DiagTensor[i, k]
    let result = tensor_a.contract(&tensor_b);

    assert_eq!(result.dims(), vec![3, 3]);
    assert!(is_diag_tensor(&result));

    // Result diagonal should be element-wise product: [1*4, 2*5, 3*6] = [4, 10, 18]
    let expected = diag_tensor_dyn_len(vec![i, k], vec![4.0, 10.0, 18.0]);
    assert!(result.isapprox(&expected, 1e-12, 0.0));
}

#[test]
fn test_diag_tensor_contract_diag_dense() {
    // Create DiagTensor A[i, j] and DenseTensor B[j, k]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let diag_a = vec![1.0, 2.0];

    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_a);

    // Create DenseTensor B[j, k] with all ones
    let indices_b = vec![j.clone(), k.clone()];
    let dims_b = vec![2, 2];
    use tensor4all_core::storage::DenseStorageF64;
    let storage_b = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(vec![1.0; 4], &dims_b));
    let tensor_b: TensorDynLen = TensorDynLen::new(indices_b, Arc::new(storage_b));

    // Contract along j: result should be DenseTensor[i, k]
    let result = tensor_a.contract(&tensor_b);

    assert_eq!(result.dims(), vec![2, 2]);
    let expected = TensorDynLen::from_dense_f64(vec![i, k], vec![1.0, 1.0, 2.0, 2.0]);
    assert!(result.isapprox(&expected, 1e-12, 0.0));
}

#[test]
fn test_diag_tensor_convert_to_dense() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];

    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_data);
    let dims = tensor.dims();
    let dense_storage = tensor.storage().to_dense_storage(&dims);
    let dense_tensor = TensorDynLen::new(vec![i.clone(), j.clone()], Arc::new(dense_storage));
    let expected = TensorDynLen::from_dense_f64(
        vec![i, j],
        vec![
            1.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 3.0,
        ],
    );
    assert!(dense_tensor.isapprox(&expected, 1e-12, 0.0));
}

#[test]
fn test_diag_tensor_rank3() {
    // Test DiagTensor with rank 3
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let diag_data = vec![1.0, 2.0];

    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone(), k.clone()], diag_data.clone());
    assert_eq!(tensor.dims(), vec![2, 2, 2]);
    assert!(is_diag_tensor(&tensor));

    // Sum should work
    let sum: AnyScalar = tensor.sum();
    assert!(!sum.is_complex());
    assert!((sum.real() - 3.0).abs() < 1e-10);
}

#[test]
fn test_diag_tensor_complex() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let diag_data = vec![Complex64::new(1.0, 0.5), Complex64::new(2.0, 1.0)];

    let tensor = diag_tensor_dyn_len_c64(vec![i.clone(), j.clone()], diag_data.clone());
    assert_eq!(tensor.dims(), vec![2, 2]);
    assert!(is_diag_tensor(&tensor));

    // Sum should work
    let sum: AnyScalar = tensor.sum();
    assert!(sum.is_complex());
    let z: Complex64 = sum.into();
    assert!((z.re - 3.0).abs() < 1e-10);
    assert!((z.im - 1.5).abs() < 1e-10);
}

#[test]
fn test_diag_tensor_complex_axpby_preserves_diag_structure() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let diag_a = vec![Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)];
    let diag_b = vec![Complex64::new(0.5, -1.0), Complex64::new(3.0, 0.25)];

    let tensor_a = diag_tensor_dyn_len_c64(vec![i.clone(), j.clone()], diag_a.clone());
    let tensor_b = diag_tensor_dyn_len_c64(vec![i.clone(), j.clone()], diag_b.clone());

    let a = AnyScalar::new_real(2.0);
    let b = AnyScalar::new_complex(-0.5, 1.0);
    let result = tensor_a.axpby(a, &tensor_b, b).unwrap();

    assert!(is_diag_tensor(&result));
    let b_c = Complex64::new(-0.5, 1.0);
    let expected_diag: Vec<Complex64> = diag_a
        .iter()
        .zip(diag_b.iter())
        .map(|(&x, &y)| Complex64::new(2.0, 0.0) * x + b_c * y)
        .collect();
    let expected = diag_tensor_dyn_len_c64(vec![i, j], expected_diag);
    assert!(result.isapprox(&expected, 1e-12, 0.0));
}

#[test]
fn test_diag_tensor_contract_rank3() {
    // Test contraction of rank-3 DiagTensors
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let l = Index::new_dyn(2);
    let diag_a = vec![1.0, 2.0];
    let diag_b = vec![3.0, 4.0];

    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone(), k.clone()], diag_a);
    let tensor_b = diag_tensor_dyn_len(vec![k.clone(), l.clone()], diag_b);

    // Contract along k: result should be DiagTensor[i, j, l]
    let result = tensor_a.contract(&tensor_b);

    assert_eq!(result.dims(), vec![2, 2, 2]);
    assert!(is_diag_tensor(&result));

    // Result diagonal should be element-wise product: [1*3, 2*4] = [3, 8]
    let expected = diag_tensor_dyn_len(vec![i, j, l], vec![3.0, 8.0]);
    assert!(result.isapprox(&expected, 1e-12, 0.0));
}
