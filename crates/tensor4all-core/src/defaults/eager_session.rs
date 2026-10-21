//! Engine selection for the implicit and context-scoped eager frontends.
//!
//! A context-aware entry point (`*_in(&ExecutionContext, ..)`) runs on that
//! context's engine. An implicit entry point has no context to consult: it
//! resolves the engine from the eager runtime of the operand it acts on, using
//! the process-global default CPU context only when that runtime has no live
//! owner. This is the transitional implicit route, which coexists with the
//! engine-explicit API and will be removed with it.

use std::sync::Arc;

use tenferro::ExecutionEngine;
use tenferro_ad::{EagerRuntime, EagerSession};
use tenferro_tensor::Engine;
use tensor4all_tensorbackend::{with_runtime_engine, ExecutionContext};

use crate::IdxTensorError;

/// Reject a context this build cannot execute in before resolving an engine.
///
/// In the mixed configuration of `tensor4all-rs#869` the backend exposes
/// `ExecutionContext::Cuda` while core is built without CUDA support. The
/// caller's `validate_context` normally rejects it first; this keeps an entry
/// that reaches here from silently falling back to the implicit CPU engine.
#[cfg(not(feature = "tenferro-cuda"))]
fn ensure_supported_context(context: Option<&ExecutionContext>) -> Result<(), IdxTensorError> {
    if context.is_some_and(|context| !matches!(context, ExecutionContext::Cpu(_))) {
        return Err(super::idx_tensor::unsupported_execution_context_error());
    }
    Ok(())
}

#[cfg(feature = "tenferro-cuda")]
fn ensure_supported_context(_: Option<&ExecutionContext>) -> Result<(), IdxTensorError> {
    Ok(())
}

/// Run one eager session on the engine selected by `context`.
///
/// `None` selects the engine of the CPU or CUDA context that owns `runtime`'s
/// eager runtime (falling back to the process-global default CPU context when
/// that runtime has no live owner; the transitional implicit route).
/// `Some(context)` selects that exact context's engine; a nested entry on the
/// same thread is reported as the context's typed
/// [`CpuExecutionContextError::Reentered`](tensor4all_tensorbackend::CpuExecutionContextError::Reentered)
/// or `CudaExecutionContextError::Reentered` instead of waiting for the engine it
/// already holds.
pub(crate) fn with_eager_session<T, E>(
    runtime: &Arc<EagerRuntime>,
    context: Option<&ExecutionContext>,
    f: impl FnOnce(&mut EagerSession<'_>) -> std::result::Result<T, E>,
) -> std::result::Result<T, IdxTensorError>
where
    E: From<tenferro_ad::Error> + std::error::Error + Send + Sync + 'static,
{
    with_eager_session_typed(runtime, context, f)?.map_err(eager_error)
}

/// Run one eager session and keep the callback's own error type on failure.
///
/// The outer error is the engine-entry failure; the inner result is the
/// callback's. A caller that inspects the callback's typed diagnostic (for
/// example a fallback that preserves the original AD error) uses this form
/// instead of [`with_eager_session`], which folds both into
/// [`IdxTensorError`].
pub(crate) fn with_eager_session_typed<T, E>(
    runtime: &Arc<EagerRuntime>,
    context: Option<&ExecutionContext>,
    f: impl FnOnce(&mut EagerSession<'_>) -> std::result::Result<T, E>,
) -> std::result::Result<std::result::Result<T, E>, IdxTensorError>
where
    E: From<tenferro_ad::Error>,
{
    ensure_supported_context(context)?;
    match context {
        Some(ExecutionContext::Cpu(context)) => context
            .with_session(|engine| runtime.with_eager_session(engine, f))
            .map_err(IdxTensorError::from),
        #[cfg(feature = "tenferro-cuda")]
        Some(ExecutionContext::Cuda(context)) => context
            .with_session(|engine| runtime.with_eager_session(engine, f))
            .map_err(IdxTensorError::from),
        None => with_runtime_engine(runtime, |engine| runtime.with_eager_session(engine, f))
            .map_err(IdxTensorError::from),
        #[cfg(not(feature = "tenferro-cuda"))]
        _ => with_runtime_engine(runtime, |engine| runtime.with_eager_session(engine, f))
            .map_err(IdxTensorError::from),
    }
}

/// Run one engine operation on the engine selected by `context`.
///
/// The contract is the same as [`with_eager_session`], for operations that take
/// a [`tenferro_tensor::Engine`] directly instead of an eager session. `runtime`
/// is the eager runtime of the operand the operation acts on, and decides the
/// engine on the implicit (`None`) route.
pub(crate) fn with_engine<T, E>(
    runtime: &Arc<EagerRuntime>,
    context: Option<&ExecutionContext>,
    f: impl FnOnce(&mut dyn Engine) -> std::result::Result<T, E>,
) -> std::result::Result<T, IdxTensorError>
where
    E: std::error::Error + Send + Sync + 'static,
{
    ensure_supported_context(context)?;
    let result = match context {
        Some(ExecutionContext::Cpu(context)) => context
            .with_session(|engine| f(engine))
            .map_err(IdxTensorError::from),
        #[cfg(feature = "tenferro-cuda")]
        Some(ExecutionContext::Cuda(context)) => context
            .with_session(|engine| f(engine))
            .map_err(IdxTensorError::from),
        None => with_runtime_engine(runtime, |engine| f(engine)).map_err(IdxTensorError::from),
        #[cfg(not(feature = "tenferro-cuda"))]
        _ => with_runtime_engine(runtime, |engine| f(engine)).map_err(IdxTensorError::from),
    };
    result?.map_err(eager_error)
}

/// Run one execution-engine operation on the engine selected by `context`.
///
/// `backward` and the other derivative entry points need
/// [`tenferro::ExecutionEngine`] rather than the narrower
/// [`tenferro_tensor::Engine`]; the engine selection is otherwise identical to
/// [`with_engine`].
pub(crate) fn with_execution_engine<T, E>(
    runtime: &Arc<EagerRuntime>,
    context: Option<&ExecutionContext>,
    f: impl FnOnce(&mut dyn ExecutionEngine) -> std::result::Result<T, E>,
) -> std::result::Result<T, IdxTensorError>
where
    E: std::error::Error + Send + Sync + 'static,
{
    ensure_supported_context(context)?;
    let result = match context {
        Some(ExecutionContext::Cpu(context)) => context
            .with_session(|engine| f(engine))
            .map_err(IdxTensorError::from),
        #[cfg(feature = "tenferro-cuda")]
        Some(ExecutionContext::Cuda(context)) => context
            .with_session(|engine| f(engine))
            .map_err(IdxTensorError::from),
        None => with_runtime_engine(runtime, |engine| f(engine)).map_err(IdxTensorError::from),
        #[cfg(not(feature = "tenferro-cuda"))]
        _ => with_runtime_engine(runtime, |engine| f(engine)).map_err(IdxTensorError::from),
    };
    result?.map_err(eager_error)
}

fn eager_error(error: impl std::error::Error + Send + Sync + 'static) -> IdxTensorError {
    IdxTensorError::materialization(anyhow::Error::new(error))
}

#[cfg(test)]
mod tests {
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
}
