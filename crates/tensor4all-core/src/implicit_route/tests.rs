use std::error::Error;
use std::sync::Arc;

use tensor4all_tensorbackend::ExecutionContextEntryError;

use crate::IdxTensorError;

#[test]
fn refusal_is_the_typed_no_context_entry_error() {
    let error = super::refusal();
    assert!(matches!(
        error.downcast_ref::<ExecutionContextEntryError>(),
        Some(ExecutionContextEntryError::NoContext)
    ));
    // Recovered as a typed `IdxTensorError` at the public boundary.
    let typed = IdxTensorError::from(ExecutionContextEntryError::NoContext);
    assert!(matches!(
        typed,
        IdxTensorError::EngineEntry {
            source: ExecutionContextEntryError::NoContext
        }
    ));
    assert!(typed.source().is_some());
}

#[cfg(feature = "global-defaults")]
#[test]
fn global_build_resolves_the_process_global_state() {
    let runtime = super::eager_runtime().unwrap();
    assert!(Arc::ptr_eq(
        &runtime,
        &tensor4all_tensorbackend::default_eager_ctx().unwrap()
    ));
    let context = super::default_cpu_execution_context().unwrap();
    assert!(Arc::ptr_eq(
        &context,
        &tensor4all_tensorbackend::default_cpu_execution_context()
    ));
}

#[cfg(not(feature = "global-defaults"))]
#[test]
fn explicit_only_build_refuses_instead_of_choosing_an_engine() {
    assert!(matches!(
        super::eager_runtime(),
        Err(ExecutionContextEntryError::NoContext)
    ));
    assert!(matches!(
        super::default_cpu_execution_context(),
        Err(ExecutionContextEntryError::NoContext)
    ));
}
