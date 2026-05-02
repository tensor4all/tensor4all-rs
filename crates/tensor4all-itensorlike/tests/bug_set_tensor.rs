use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::TensorDynLen;
use tensor4all_itensorlike::{TensorTrain, TensorTrainError};

#[test]
fn set_tensor_returns_an_error_for_an_invalid_site() {
    let site = Index::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![site.clone()], vec![1.0, 0.0]).unwrap();
    let replacement = TensorDynLen::from_dense(vec![site.clone()], vec![0.5, 0.5]).unwrap();
    let mut tt = TensorTrain::new(vec![tensor]).unwrap();

    let result = tt.set_tensor(1, replacement);

    assert!(matches!(
        result,
        Err(TensorTrainError::SiteOutOfBounds { site: 1, length: 1 })
    ));
}

#[test]
fn set_tensor_replaces_the_tensor_at_the_requested_site() {
    let site = Index::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![site.clone()], vec![1.0, 0.0]).unwrap();
    let replacement = TensorDynLen::from_dense(vec![site.clone()], vec![0.5, 0.5]).unwrap();
    let mut tt = TensorTrain::new(vec![tensor]).unwrap();

    assert!(tt.set_tensor(0, replacement).is_ok());
    assert_eq!(tt.tensor(0).to_vec::<f64>().unwrap(), vec![0.5, 0.5]);
}
