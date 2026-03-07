use std::sync::Arc;

use tensor4all_core::{
    factorize, qr, svd, AnyScalar, Canonical, FactorizeOptions, Index, Storage, TensorDynLen,
};
use tensor4all_tensorbackend::tenferro_dyadtensor::{AdMode, AdTensor, DynAdTensor};
use tensor4all_tensorbackend::tenferro_tensor::{MemoryOrder, Tensor};

fn native_f64_tensor(primal: &[f64], tangent: &[f64], dims: &[usize]) -> DynAdTensor {
    let primal = Tensor::<f64>::from_slice(primal, dims, MemoryOrder::RowMajor)
        .expect("valid primal tensor");
    let tangent = Tensor::<f64>::from_slice(tangent, dims, MemoryOrder::RowMajor)
        .expect("valid tangent tensor");
    AdTensor::new_forward(primal, tangent).unwrap().into()
}

#[test]
fn sum_preserves_forward_native_payload() {
    let i = Index::new_dyn(2);
    let tensor = TensorDynLen::from_native(
        vec![i],
        native_f64_tensor(&[1.0, 2.0], &[0.25, -0.75], &[2]),
    )
    .unwrap();

    let sum = tensor.sum();

    assert_eq!(sum.mode(), AdMode::Forward);
    assert_eq!(sum.primal().as_f64(), Some(3.0));
    assert_eq!(sum.tangent().and_then(|x| x.as_f64()), Some(-0.5));
}

#[test]
fn only_preserves_forward_native_payload() {
    let tensor =
        TensorDynLen::from_native(vec![], native_f64_tensor(&[2.5], &[0.75], &[])).unwrap();

    let only = tensor.only();

    assert_eq!(only.mode(), AdMode::Forward);
    assert_eq!(only.primal().as_f64(), Some(2.5));
    assert_eq!(only.tangent().and_then(|x| x.as_f64()), Some(0.75));
}

#[test]
fn inner_product_preserves_forward_native_payload() {
    let i = Index::new_dyn(2);
    let lhs = TensorDynLen::from_native(
        vec![i.clone()],
        native_f64_tensor(&[1.0, 2.0], &[0.1, 0.2], &[2]),
    )
    .unwrap();
    let rhs =
        TensorDynLen::from_native(vec![i], native_f64_tensor(&[3.0, 4.0], &[1.0, -1.0], &[2]))
            .unwrap();

    let inner = lhs.inner_product(&rhs).unwrap();

    assert_eq!(inner.mode(), AdMode::Forward);
    assert_eq!(inner.primal().as_f64(), Some(11.0));
    let tangent = inner.tangent().and_then(|x| x.as_f64()).unwrap();
    assert!(
        (tangent - 0.1).abs() < 1e-12,
        "unexpected tangent: {tangent}"
    );
}

#[test]
fn qr_preserves_forward_native_payload() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = TensorDynLen::from_native(
        vec![i.clone(), j.clone()],
        native_f64_tensor(&[3.0, 0.0, 0.0, 2.0], &[0.5, 0.0, 0.0, -0.25], &[2, 2]),
    )
    .unwrap();

    let (q, r) = qr::<f64>(&tensor, std::slice::from_ref(&i)).unwrap();

    assert_eq!(q.as_native().mode(), AdMode::Forward);
    assert_eq!(r.as_native().mode(), AdMode::Forward);
    assert!(
        (q.sum()
            .tangent()
            .and_then(|x| x.as_f64())
            .unwrap_or_default())
        .abs()
            < 1e-12
    );
    let r_tangent = r.sum().tangent().and_then(|x| x.as_f64()).unwrap();
    assert!(
        (r_tangent - 0.25).abs() < 1e-12,
        "unexpected QR tangent: {r_tangent}"
    );
}

#[test]
fn svd_preserves_forward_native_payload() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = TensorDynLen::from_native(
        vec![i.clone(), j.clone()],
        native_f64_tensor(&[3.0, 0.0, 0.0, 2.0], &[0.5, 0.0, 0.0, -0.25], &[2, 2]),
    )
    .unwrap();

    let (u, s, v) = svd::<f64>(&tensor, std::slice::from_ref(&i)).unwrap();

    assert_eq!(u.as_native().mode(), AdMode::Forward);
    assert_eq!(s.as_native().mode(), AdMode::Forward);
    assert_eq!(v.as_native().mode(), AdMode::Forward);
    let s_tangent = s.sum().tangent().and_then(|x| x.as_f64()).unwrap();
    assert!(
        (s_tangent - 0.25).abs() < 1e-12,
        "unexpected SVD tangent: {s_tangent}"
    );
}

#[test]
fn factorize_svd_preserves_forward_native_payload() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = TensorDynLen::from_native(
        vec![i.clone(), j.clone()],
        native_f64_tensor(&[3.0, 0.0, 0.0, 2.0], &[0.5, 0.0, 0.0, -0.25], &[2, 2]),
    )
    .unwrap();

    let result = factorize(
        &tensor,
        std::slice::from_ref(&i),
        &FactorizeOptions::svd().with_canonical(Canonical::Left),
    )
    .unwrap();

    assert_eq!(result.left.as_native().mode(), AdMode::Forward);
    assert_eq!(result.right.as_native().mode(), AdMode::Forward);
    let right_tangent = result
        .right
        .sum()
        .tangent()
        .and_then(|x| x.as_f64())
        .unwrap();
    assert!(
        (right_tangent - 0.25).abs() < 1e-12,
        "unexpected factorize tangent: {right_tangent}"
    );
}

#[test]
fn rank1_native_snapshots_stay_dense() {
    let i = Index::new_dyn(3);
    let tensor = TensorDynLen::from_native(
        vec![i],
        native_f64_tensor(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0], &[3]),
    )
    .unwrap();

    let scaled = tensor.scale(AnyScalar::new_real(2.0)).unwrap();

    assert!(matches!(scaled.storage().as_ref(), Storage::DenseF64(_)));
}

#[test]
fn plain_dense_storage_auto_seeds_native_payload() {
    let i = Index::new_dyn(2);
    let tensor = TensorDynLen::from_storage(
        vec![i],
        Arc::new(Storage::DenseF64(
            tensor4all_core::storage::DenseStorageF64::from_vec_with_shape(vec![1.0, 2.0], &[2]),
        )),
    )
    .unwrap();

    assert_eq!(tensor.as_native().mode(), AdMode::Primal);
}

#[test]
fn plain_diag_storage_auto_seeds_native_diag_payload() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::from_storage(
        vec![i, j],
        Arc::new(Storage::new_diag_f64(vec![1.0, 2.0, 3.0])),
    )
    .unwrap();

    let native = tensor.as_native();
    assert_eq!(native.mode(), AdMode::Primal);
    assert!(native.is_diag());
}

#[test]
fn from_native_into_native_round_trips_mode_and_dims() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = TensorDynLen::from_native(
        vec![i, j],
        native_f64_tensor(&[1.0, 2.0, 3.0, 4.0], &[0.5, 0.0, 0.0, -0.25], &[2, 2]),
    )
    .unwrap();

    let native = tensor.into_native();
    assert_eq!(native.mode(), AdMode::Forward);
    assert_eq!(native.dims(), vec![2, 2]);
}
