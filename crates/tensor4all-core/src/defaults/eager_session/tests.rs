use std::error::Error;

use tensor4all_tensorbackend::CpuExecutionContextError;

use crate::IdxTensorError;

#[test]
fn context_error_conversion_keeps_the_typed_variant() {
    let error = IdxTensorError::from(CpuExecutionContextError::Reentered);
    assert!(matches!(
        error,
        IdxTensorError::EngineEntry {
            source: tensor4all_tensorbackend::ExecutionContextEntryError::Cpu(
                CpuExecutionContextError::Reentered
            )
        }
    ));
    assert!(error.source().is_some());
}
