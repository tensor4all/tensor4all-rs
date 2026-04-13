use tensor4all_core::{contract_multi, AllowedPairs, Index, Storage, TensorDynLen};

#[test]
fn plain_dense_storage_auto_seeds_native_payload() {
    let i = Index::new_dyn(2);
    let tensor = TensorDynLen::from_storage(
        vec![i],
        Storage::from_dense_col_major(vec![1.0, 2.0], &[2])
            .map(std::sync::Arc::new)
            .unwrap(),
    )
    .unwrap();

    assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn plain_diag_storage_preserves_diag_metadata() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::from_storage(
        vec![i, j],
        Storage::from_diag_col_major(vec![1.0, 2.0, 3.0], 2)
            .map(std::sync::Arc::new)
            .unwrap(),
    )
    .unwrap();

    assert_eq!(
        tensor.to_vec::<f64>().unwrap(),
        vec![
            1.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 3.0,
        ]
    );
    assert!(tensor.is_diag());
}

#[test]
fn contraction_without_grad_returns_rank_zero_scalar() {
    let i = Index::new_dyn(3);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    let ones = TensorDynLen::from_dense(vec![i], vec![1.0, 1.0, 1.0]).unwrap();

    let result = contract_multi(&[&a, &ones], AllowedPairs::All).unwrap();

    assert!(result.indices().is_empty());
    assert_eq!(result.to_vec::<f64>().unwrap(), vec![6.0]);
}

#[test]
fn backward_accumulates_until_clear_grad() {
    let i = Index::new_dyn(3);
    let x = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0, 3.0])
        .unwrap()
        .enable_grad();
    let ones = TensorDynLen::from_dense(vec![i], vec![1.0, 1.0, 1.0]).unwrap();

    let loss = contract_multi(&[&x, &ones], AllowedPairs::All).unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0]);

    let loss = contract_multi(&[&x, &ones], AllowedPairs::All).unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.to_vec::<f64>().unwrap(), vec![2.0, 2.0, 2.0]);

    x.clear_grad().unwrap();
    assert!(x.grad().unwrap().is_none());
}

#[test]
fn tracks_grad_and_detach_report_leaf_state() {
    let scalar = TensorDynLen::scalar(2.0).unwrap();
    assert!(!scalar.tracks_grad());

    let tracked = scalar.enable_grad();
    assert!(tracked.tracks_grad());

    let detached = tracked.detach();
    assert!(!detached.tracks_grad());
    assert!(tracked.tracks_grad());
}
