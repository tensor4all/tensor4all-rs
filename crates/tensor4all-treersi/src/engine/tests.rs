use super::readback_error;
use crate::TreeRsiError;
use tensor4all_core::{DynIndex, IdxTensor, IdxTensorError};
use tensor4all_tensorbackend::{CpuExecutionContextError, ExecutionContextEntryError};

#[test]
fn a_session_reentry_readback_failure_keeps_its_typed_source() {
    let error = IdxTensorError::EngineEntry {
        source: ExecutionContextEntryError::Cpu(CpuExecutionContextError::Reentered),
    };
    match readback_error(error) {
        TreeRsiError::Numerical { source } => {
            assert!(
                source.to_string().contains("session"),
                "the re-entry diagnostic must survive: {source}"
            );
            assert!(
                std::error::Error::source(&*source).is_some(),
                "the typed re-entry diagnostic must stay in the chain"
            );
        }
        other => panic!("a failed readback must be Numerical, got {other:?}"),
    }
}

#[test]
fn another_scalar_type_is_reported_as_a_scalar_kind() {
    let site = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense(vec![site], vec![1.0_f64, 2.0]).unwrap();
    let error = tensor.to_vec::<f32>().unwrap_err();
    assert!(
        error.is_payload_scalar_type_mismatch(),
        "a dtype mismatch must be classified at to_vec: {error}"
    );
    assert!(matches!(
        readback_error(error),
        TreeRsiError::ScalarKind { .. }
    ));
}
