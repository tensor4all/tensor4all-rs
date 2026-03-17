use tenferro::AdMode;
use tensor4all_core::{
    factorize, forward_ad, is_diag_tensor, qr, svd, AnyScalar, Canonical, FactorizeOptions, Index,
    Storage, TensorDynLen,
};

fn forward_tensor(primal: TensorDynLen, tangent: TensorDynLen) -> TensorDynLen {
    forward_ad::dual_level(|fw| fw.make_dual(&primal, &tangent)).unwrap()
}

fn assert_same_tensor_data(lhs: &TensorDynLen, rhs: &TensorDynLen) {
    assert_eq!(lhs.dims(), rhs.dims());
    assert_eq!(lhs.is_diag(), rhs.is_diag());
    assert_eq!(lhs.is_f64(), rhs.is_f64());
    assert_eq!(lhs.is_complex(), rhs.is_complex());
    if lhs.is_f64() {
        assert_eq!(lhs.to_vec_f64().unwrap(), rhs.to_vec_f64().unwrap());
    } else {
        assert_eq!(lhs.to_vec_c64().unwrap(), rhs.to_vec_c64().unwrap());
    }
}

#[test]
fn sum_preserves_forward_payload_via_dual_level() {
    let i = Index::new_dyn(2);
    let tensor = forward_tensor(
        TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap(),
        TensorDynLen::from_dense(vec![i], vec![0.25, -0.75]).unwrap(),
    );

    let sum = tensor.sum();

    assert_eq!(sum.mode(), AdMode::Forward);
    assert_eq!(sum.primal().as_f64(), Some(3.0));
    assert_eq!(sum.tangent().and_then(|x| x.as_f64()), Some(-0.5));
}

#[test]
fn only_preserves_forward_payload_via_dual_level() {
    let tensor = forward_tensor(
        TensorDynLen::from_dense(vec![], vec![2.5]).unwrap(),
        TensorDynLen::from_dense(vec![], vec![0.75]).unwrap(),
    );

    let only = tensor.only();

    assert_eq!(only.mode(), AdMode::Forward);
    assert_eq!(only.primal().as_f64(), Some(2.5));
    assert_eq!(only.tangent().and_then(|x| x.as_f64()), Some(0.75));
}

#[test]
fn inner_product_preserves_forward_payload_via_dual_level() {
    let i = Index::new_dyn(2);
    let lhs = forward_tensor(
        TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap(),
        TensorDynLen::from_dense(vec![i.clone()], vec![0.1, 0.2]).unwrap(),
    );
    let rhs = forward_tensor(
        TensorDynLen::from_dense(vec![i.clone()], vec![3.0, 4.0]).unwrap(),
        TensorDynLen::from_dense(vec![i], vec![1.0, -1.0]).unwrap(),
    );

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
fn qr_preserves_forward_payload_via_dual_level() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = forward_tensor(
        TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![3.0, 0.0, 0.0, 2.0]).unwrap(),
        TensorDynLen::from_dense(vec![i.clone(), j], vec![0.5, 0.0, 0.0, -0.25]).unwrap(),
    );

    let (q, r) = qr::<f64>(&tensor, std::slice::from_ref(&i)).unwrap();

    assert_eq!(q.mode(), AdMode::Forward);
    assert_eq!(r.mode(), AdMode::Forward);
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
fn svd_preserves_forward_payload_via_dual_level() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = forward_tensor(
        TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![3.0, 0.0, 0.0, 2.0]).unwrap(),
        TensorDynLen::from_dense(vec![i.clone(), j], vec![0.5, 0.0, 0.0, -0.25]).unwrap(),
    );

    let (u, s, v) = svd::<f64>(&tensor, std::slice::from_ref(&i)).unwrap();

    assert_eq!(u.mode(), AdMode::Forward);
    assert_eq!(s.sum().mode(), AdMode::Forward);
    assert_eq!(v.mode(), AdMode::Forward);
    let s_tangent = s.sum().tangent().and_then(|x| x.as_f64()).unwrap();
    assert!(
        (s_tangent - 0.25).abs() < 1e-12,
        "unexpected SVD tangent: {s_tangent}"
    );
}

#[test]
fn factorize_svd_preserves_forward_payload_via_dual_level() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = forward_tensor(
        TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![3.0, 0.0, 0.0, 2.0]).unwrap(),
        TensorDynLen::from_dense(vec![i.clone(), j], vec![0.5, 0.0, 0.0, -0.25]).unwrap(),
    );

    let result = factorize(
        &tensor,
        std::slice::from_ref(&i),
        &FactorizeOptions::svd().with_canonical(Canonical::Left),
    )
    .unwrap();

    assert_eq!(result.left.mode(), AdMode::Forward);
    assert_eq!(result.right.mode(), AdMode::Forward);
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
fn forward_ad_unpack_dual_restores_primal_and_tangent() {
    let i = Index::new_dyn(2);
    let primal = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    let tangent = TensorDynLen::from_dense(vec![i], vec![0.5, -0.25]).unwrap();

    let (unpacked_primal, unpacked_tangent) = forward_ad::dual_level(|fw| {
        let dual = fw.make_dual(&primal, &tangent)?;
        fw.unpack_dual(&dual)
    })
    .unwrap();

    assert_same_tensor_data(&unpacked_primal, &primal);
    assert_same_tensor_data(&unpacked_tangent.unwrap(), &tangent);
}

#[test]
fn rank1_native_snapshots_stay_dense() {
    let i = Index::new_dyn(3);
    let tensor = forward_tensor(
        TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0, 3.0]).unwrap(),
        TensorDynLen::from_dense(vec![i], vec![0.0, 0.0, 0.0]).unwrap(),
    );

    let scaled = tensor.scale(AnyScalar::new_real(2.0)).unwrap();
    let snapshot = scaled.storage();

    assert!(snapshot.is_dense());
    assert!(!snapshot.is_diag());
    assert_eq!(scaled.to_vec_f64().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn plain_dense_storage_auto_seeds_native_payload() {
    let i = Index::new_dyn(2);
    let tensor = TensorDynLen::from_storage(
        vec![i],
        Storage::from_dense_f64_col_major(vec![1.0, 2.0], &[2])
            .map(std::sync::Arc::new)
            .unwrap(),
    )
    .unwrap();

    assert_eq!(tensor.mode(), AdMode::Primal);
    assert_eq!(tensor.to_vec_f64().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn plain_diag_storage_auto_seeds_native_diag_payload() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::from_storage(
        vec![i, j],
        Storage::from_diag_f64_col_major(vec![1.0, 2.0, 3.0], 2)
            .map(std::sync::Arc::new)
            .unwrap(),
    )
    .unwrap();

    assert_eq!(tensor.mode(), AdMode::Primal);
    assert!(is_diag_tensor(&tensor));
}
