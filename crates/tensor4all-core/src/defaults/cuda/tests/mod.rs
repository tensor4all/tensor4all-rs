use std::sync::Arc;

use tensor4all_tensorbackend::{Storage, StorageKind};

use super::*;
use crate::{DynIndex, IdxTensor};

#[test]
fn structured_transfer_rejection_needs_no_cuda_context() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let storage = Arc::new(Storage::from_diag_col_major(vec![1.0_f64, 2.0], 2).unwrap());
    let tensor = IdxTensor::from_structured_storage(vec![i, j], storage).unwrap();

    assert!(matches!(
        precheck(&tensor, "upload"),
        Err(IdxTensorCudaError::UnsupportedStorage {
            kind: StorageKind::Diagonal,
            ..
        })
    ));
}

#[test]
fn tracked_transfer_rejection_needs_no_cuda_context() {
    let tensor = IdxTensor::scalar(1.0_f64).unwrap().enable_grad().unwrap();

    assert!(matches!(
        precheck(&tensor, "upload"),
        Err(IdxTensorCudaError::Tracked { .. })
    ));
}
