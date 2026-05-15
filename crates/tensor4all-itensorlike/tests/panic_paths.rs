use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::{TensorTrain, TensorTrainError};

fn site_tensor(size: usize, data: Vec<f64>) -> TensorDynLen {
    let site = DynIndex::new_dyn(size);
    TensorDynLen::from_dense(vec![site], data).unwrap()
}

#[test]
fn tensor_accessors_report_invalid_sites() {
    let mut tt = TensorTrain::default();

    let err = tt.tensor(0).unwrap_err();
    assert!(matches!(
        err,
        TensorTrainError::SiteOutOfBounds { site: 0, length: 0 }
    ));

    let err = tt.tensor_mut(0).unwrap_err();
    assert!(matches!(
        err,
        TensorTrainError::SiteOutOfBounds { site: 0, length: 0 }
    ));
}

#[test]
fn inner_reports_length_mismatch() {
    let left = TensorTrain::new(vec![site_tensor(2, vec![1.0, 2.0])]).unwrap();
    let right = TensorTrain::default();

    let err = left.inner(&right).unwrap_err();
    assert!(err.to_string().contains("same length"));
}

#[test]
fn sim_linkinds_reports_success_without_panic_wrapper() {
    let tt = TensorTrain::new(vec![site_tensor(2, vec![1.0, 2.0])]).unwrap();
    let simmed = tt.sim_linkinds().unwrap();

    assert_eq!(simmed.len(), 1);
    assert_eq!(
        simmed.tensor(0).unwrap().to_vec::<f64>().unwrap(),
        vec![1.0, 2.0]
    );
}
