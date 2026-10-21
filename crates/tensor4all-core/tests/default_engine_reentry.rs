//! An implicit eager entry nested inside the process-global CPU engine borrow.
//!
//! The implicit frontend uses the process-global default CPU context's engine.
//! While that engine is already borrowed on this thread, a nested entry must be
//! reported as the typed re-entry error instead of waiting for the engine.

use std::sync::Arc;

use tensor4all_core::{
    factorize_in, DynIndex, ExecutionContext, FactorizeOptions, IdxTensor, IdxTensorError,
};
use tensor4all_tensorbackend::{
    with_default_backend, CpuBackend, CpuExecutionContext, CpuExecutionContextError,
};

#[test]
fn implicit_entry_inside_the_default_engine_borrow_reports_reentered() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

    let error = with_default_backend(|_engine| {
        tensor
            .permute(&[1, 0])
            .expect_err("an implicit eager entry cannot re-enter the borrowed default engine")
    });

    match error {
        IdxTensorError::EngineEntry {
            source: CpuExecutionContextError::Reentered,
        } => {}
        other => panic!("expected the typed re-entry refusal, got {other:?}"),
    }
}

/// A context-aware factorization runs on the caller's engine even while the
/// process-global default engine is borrowed on the same thread.
#[test]
fn context_factorization_runs_while_the_default_engine_is_borrowed() {
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )));
    let i = DynIndex::new_dyn(4);
    let j = DynIndex::new_dyn(3);
    let data: Vec<f64> = (0..12).map(|value| value as f64).collect();
    let tensor = IdxTensor::from_dense_in(&context, vec![i.clone(), j.clone()], data).unwrap();

    for options in [FactorizeOptions::qr(), FactorizeOptions::svd()] {
        let result = with_default_backend(|_engine| {
            factorize_in(&tensor, std::slice::from_ref(&i), &options, &context)
                .expect("the context factorization must not reach the borrowed default engine")
        });
        assert!(result.rank > 0);
        assert_eq!(result.left.dims()[0], 4);
        assert_eq!(result.right.dims().last().copied(), Some(3));
    }
}
