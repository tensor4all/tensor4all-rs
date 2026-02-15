use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{TensorDynLen, TensorLike};

#[test]
fn test_sub_identical_tensors_is_zero() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let a = TensorDynLen::from_dense_f64(
        vec![i.clone(), j.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );

    let diff = &a - &a;
    assert!(diff.norm() < 1e-14);
    assert!(diff.maxabs() < 1e-14);
}

#[test]
fn test_sub_different_tensors() {
    let i = Index::new_dyn(2);
    let a = TensorDynLen::from_dense_f64(vec![i.clone()], vec![3.0, 5.0]);
    let b = TensorDynLen::from_dense_f64(vec![i.clone()], vec![1.0, 2.0]);

    let diff = &a - &b;
    let data = diff.to_vec_f64().unwrap();
    assert!((data[0] - 2.0).abs() < 1e-14);
    assert!((data[1] - 3.0).abs() < 1e-14);
}

#[test]
fn test_sub_permuted_indices() {
    // a[i,j] - b[j,i] should auto-permute
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let a = TensorDynLen::from_dense_f64(
        vec![i.clone(), j.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let b = TensorDynLen::from_dense_f64(
        vec![j.clone(), i.clone()],
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], // transposed
    );

    let diff = &a - &b;
    assert!(diff.maxabs() < 1e-14);
}

#[test]
fn test_neg() {
    let i = Index::new_dyn(3);
    let a = TensorDynLen::from_dense_f64(vec![i.clone()], vec![1.0, -2.0, 3.0]);

    let neg_a = -&a;
    let data = neg_a.to_vec_f64().unwrap();
    assert!((data[0] - (-1.0)).abs() < 1e-14);
    assert!((data[1] - 2.0).abs() < 1e-14);
    assert!((data[2] - (-3.0)).abs() < 1e-14);
}

#[test]
fn test_maxabs() {
    let i = Index::new_dyn(4);
    let a = TensorDynLen::from_dense_f64(vec![i.clone()], vec![1.0, -5.0, 3.0, -2.0]);
    assert!((a.maxabs() - 5.0).abs() < 1e-14);
}

#[test]
fn test_maxabs_scalar() {
    let s = TensorDynLen::scalar_f64(-7.0);
    assert!((s.maxabs() - 7.0).abs() < 1e-14);
}

#[test]
fn test_isapprox_identical() {
    let i = Index::new_dyn(3);
    let a = TensorDynLen::from_dense_f64(vec![i.clone()], vec![1.0, 2.0, 3.0]);
    assert!(a.isapprox(&a, 0.0, 0.0));
}

#[test]
fn test_isapprox_atol() {
    let i = Index::new_dyn(2);
    let a = TensorDynLen::from_dense_f64(vec![i.clone()], vec![1.0, 2.0]);
    let b = TensorDynLen::from_dense_f64(vec![i.clone()], vec![1.0, 2.01]);

    // ||a - b|| = 0.01
    assert!(a.isapprox(&b, 0.1, 0.0)); // atol=0.1 > 0.01
    assert!(!a.isapprox(&b, 0.001, 0.0)); // atol=0.001 < 0.01
}

#[test]
fn test_isapprox_rtol() {
    let i = Index::new_dyn(2);
    let a = TensorDynLen::from_dense_f64(vec![i.clone()], vec![100.0, 200.0]);
    let b = TensorDynLen::from_dense_f64(vec![i.clone()], vec![100.0, 201.0]);

    // ||a - b|| = 1.0, max(||a||, ||b||) ≈ 224
    // rtol * max_norm ≈ 0.01 * 224 ≈ 2.24 > 1.0
    assert!(a.isapprox(&b, 0.0, 0.01));
    // rtol * max_norm ≈ 0.001 * 224 ≈ 0.224 < 1.0
    assert!(!a.isapprox(&b, 0.0, 0.001));
}

#[test]
fn test_isapprox_index_mismatch_returns_false() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let a = TensorDynLen::from_dense_f64(vec![i.clone()], vec![1.0, 2.0]);
    let b = TensorDynLen::from_dense_f64(vec![j.clone()], vec![1.0, 2.0, 3.0]);

    // Different indices → sub fails → isapprox returns false
    assert!(!a.isapprox(&b, 1e10, 1e10));
}

#[test]
fn test_sub_operator_owned() {
    let i = Index::new_dyn(2);
    let a = TensorDynLen::from_dense_f64(vec![i.clone()], vec![5.0, 10.0]);
    let b = TensorDynLen::from_dense_f64(vec![i.clone()], vec![1.0, 3.0]);

    // owned - owned
    let diff = a.clone() - b.clone();
    let data = diff.to_vec_f64().unwrap();
    assert!((data[0] - 4.0).abs() < 1e-14);
    assert!((data[1] - 7.0).abs() < 1e-14);

    // owned - ref
    let diff2 = a.clone() - &b;
    assert!(diff2.isapprox(&diff, 1e-14, 0.0));

    // ref - owned
    let diff3 = &a - b.clone();
    assert!(diff3.isapprox(&diff, 1e-14, 0.0));
}
