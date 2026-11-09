use super::readback_error;
use crate::TreeRsiError;
use tensor4all_core::{DynIndex, IdxTensor, IdxTensorError};
use tensor4all_tensorbackend::{CpuExecutionContextError, ExecutionContextEntryError};

#[test]
fn a_session_reentry_readback_failure_keeps_its_typed_source() {
    let error = IdxTensorError::EngineEntry {
        source: ExecutionContextEntryError::Cpu(CpuExecutionContextError::Reentered),
    };
    let TreeRsiError::Numerical { source } = readback_error(error) else {
        panic!("a failed readback must be Numerical");
    };
    // The chain itself has to identify the re-entry, not its message text: a
    // dtype mismatch is the only class that does not keep a source.
    assert!(
        matches!(
            source.downcast_ref::<IdxTensorError>(),
            Some(IdxTensorError::EngineEntry {
                source: ExecutionContextEntryError::Cpu(CpuExecutionContextError::Reentered)
            })
        ),
        "the readback must keep the context re-entry as its source: {source}"
    );
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
