//! TreeTN failures keep their typed diagnostic as `TensorTrainError`'s
//! `std::error::Error::source`.

use std::error::Error;

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_itensorlike::{TensorTrain, TensorTrainError};
use tensor4all_treetn::TreeTNOperationError;

fn two_site_train() -> TensorTrain {
    let s0 = DynIndex::new_dyn(2);
    let link = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(2);
    let t0 =
        IdxTensor::from_dense(vec![s0, link.clone()], (0..6).map(|i| i as f64).collect()).unwrap();
    let t1 = IdxTensor::from_dense(vec![link, s1], (0..6).map(|i| i as f64).collect()).unwrap();
    TensorTrain::new(vec![t0, t1]).unwrap()
}

#[test]
fn set_tensor_preserves_typed_treetn_error_source() {
    let mut tt = two_site_train();
    // The replacement lacks the link index, so TreeTN refuses the replacement.
    let replacement =
        IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0_f64, 2.0_f64]).unwrap();

    let error = tt.set_tensor(0, replacement).unwrap_err();

    assert!(matches!(error, TensorTrainError::TreeTn { .. }));
    let source = error.source().expect("TreeTn error keeps its source");
    assert!(
        source.downcast_ref::<TreeTNOperationError>().is_some(),
        "source is the original TreeTN diagnostic, got {source}"
    );
}
