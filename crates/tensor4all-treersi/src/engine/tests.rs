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

#[cfg(feature = "tenferro-cuda")]
mod cuda_arm {
    use super::super::{row_interpolation, Exec, Matrix, RrLUOptions};
    use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, CudaExecutionContext};

    /// The CUDA arm's rank decision runs on the CPU context it was handed, not
    /// on the process-global one: holding a session of the global context
    /// must not make it fail with a re-entry error. Needs CUDA ordinal 0 only
    /// to construct the (otherwise unused) device context.
    #[test]
    #[ignore = "requires visible CUDA ordinal 0"]
    fn cuda_row_interpolation_uses_the_handed_cpu_context() {
        let cuda = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
        let owned = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap());
        let matrix = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]);
        let global = tensor4all_tensorbackend::default_cpu_execution_context();
        let result = global
            .with_concrete_session(|_held| {
                row_interpolation::<f64>(
                    &mut Exec::Cuda(&cuda, &owned),
                    matrix,
                    RrLUOptions::default(),
                )
            })
            .expect("entering the global session succeeds");
        assert!(result.is_ok(), "{result:?}");
    }
}
