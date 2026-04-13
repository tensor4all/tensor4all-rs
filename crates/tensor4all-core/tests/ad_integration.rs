use tensor4all_core::{Index, TensorDynLen};

#[test]
fn tensor_sum_returns_rank_zero_anyscalar_with_backward() {
    let i = Index::new_dyn(3);
    let x = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0, 3.0])
        .unwrap()
        .enable_grad();

    let loss = x.sum();
    assert!(loss.tracks_grad());

    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn tensor_only_preserves_tracking_for_scalar_tensor() {
    let x = TensorDynLen::scalar(2.0).unwrap().enable_grad();

    let scalar = x.only();
    let loss = &scalar * &scalar;
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert!((grad.only().real() - 4.0).abs() < 1e-12);
}
