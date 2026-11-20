//! Engine selection for the implicit and context-scoped eager frontends.
//!
//! A context-aware entry point (`*_in(&ExecutionContext, ..)`) runs on that
//! context's engine. An implicit entry point has no context to consult: it
//! resolves the engine from the eager runtime of the operand it acts on, using
//! the process-global default CPU context only when that runtime has no live
//! owner. This is the transitional implicit route, which coexists with the
//! engine-explicit API and will be removed with it.
//!
//! [`with_eager_on`] is the engine-taking entry of stage A2 (#1393): it runs a
//! caller-owned [`Runtime`] on the caller's `&mut` engine after validating the
//! operands, with no context, lock or owner lookup.

use std::sync::Arc;

use tenferro::{EngineTarget, ExecutionContextIdentity, ExecutionEngine};
use tenferro_ad::{EagerRuntime, EagerSession};
use tenferro_tensor::{Engine, MemoryKind, TensorRead};
use tensor4all_tensorbackend::{with_runtime_engine, CpuEngine, ExecutionContext, Runtime};

use crate::{IdxTensor, IdxTensorError};

/// Run one eager session of `runtime` on the caller's engine, after validating
/// the operands.
///
/// Validation runs before `f`, so before any RNG draw or kernel the operation
/// performs, and in this order:
///
/// 1. every operand that carries an eager value belongs to `runtime`, else
///    [`IdxTensorError::RuntimeMismatch`];
/// 2. `runtime` supports `exec` ([`Runtime::check_engine`]), else
///    [`IdxTensorError::Operation`] carrying the typed
///    [`RuntimeError::UnsupportedEngine`](tensor4all_tensorbackend::RuntimeError::UnsupportedEngine);
/// 3. `exec` can read every operand where it is placed, else
///    [`IdxTensorError::PlacementMismatch`]. An operand with no eager value is
///    host data, which only a CPU engine reads.
///
/// Nothing is transferred. Crate-private until stage B2 of #1376 gives the
/// public operations their `exec` argument.
// Stage A2: the first non-test callers arrive with stage B2 of #1376.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn with_eager_on<T, E>(
    operation: &'static str,
    runtime: &Runtime,
    operands: &[&IdxTensor],
    exec: &mut dyn ExecutionEngine,
    f: impl FnOnce(&mut EagerSession<'_>) -> std::result::Result<T, E>,
) -> std::result::Result<T, IdxTensorError>
where
    E: From<tenferro_ad::Error> + std::error::Error + Send + Sync + 'static,
{
    validate_operands(operation, runtime, operands, exec)?;
    runtime.with_eager(exec, f).map_err(eager_error)
}

/// The entry validation of [`with_eager_on`], without running anything.
fn validate_operands(
    operation: &'static str,
    runtime: &Runtime,
    operands: &[&IdxTensor],
    exec: &dyn ExecutionEngine,
) -> std::result::Result<(), IdxTensorError> {
    let expected = runtime.id();
    for operand in operands {
        if let Some(value) = operand.eager_value() {
            let found = value.ctx_id();
            if found != expected {
                return Err(IdxTensorError::RuntimeMismatch {
                    operation,
                    expected,
                    found,
                });
            }
        }
    }
    let target = runtime
        .check_engine(exec)
        .map_err(|source| IdxTensorError::Operation {
            operation,
            source: Arc::new(source),
        })?;
    for operand in operands {
        let readable = match operand.eager_value() {
            Some(value) => engine_reads(&target, &value.tensor_read()),
            None => is_cpu_engine(&target),
        };
        if !readable {
            let placement = match operand.eager_value() {
                Some(value) => {
                    let read = value.tensor_read();
                    format!(
                        "{:?} (allocation domain {:?})",
                        read.placement(),
                        read.allocation_domain()
                    )
                }
                None => "host memory (no eager value)".to_owned(),
            };
            return Err(IdxTensorError::PlacementMismatch {
                operation,
                placement,
                engine_id: target.engine_id().as_str().to_owned(),
            });
        }
    }
    Ok(())
}

fn is_cpu_engine(target: &EngineTarget) -> bool {
    target.context_identity() == ExecutionContextIdentity::of::<CpuEngine>()
}

/// Whether an engine with `target` reads `value` where it is placed.
///
/// A CPU engine reads host memory outside any allocation domain, or inside its
/// own; any other engine reads only values in its own allocation domain.
fn engine_reads(target: &EngineTarget, value: &TensorRead<'_>) -> bool {
    let domain = value.allocation_domain();
    let own_domain = domain.is_some() && domain == target.allocation_domain();
    if !is_cpu_engine(target) {
        return own_domain;
    }
    match value.placement().memory_kind {
        MemoryKind::PinnedHost | MemoryKind::UnpinnedHost => domain.is_none() || own_domain,
        MemoryKind::Managed => own_domain,
        MemoryKind::Device | MemoryKind::Other(_) => false,
    }
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
        // INVARIANT: reachable only in the mixed configuration of
        // tensor4all-rs#869 (tensorbackend built with CUDA, core without), where
        // `ExecutionContext::Cuda` exists but this build cannot execute in it; in
        // every other configuration `Cpu` is the only variant.
        #[cfg(not(feature = "tenferro-cuda"))]
        #[allow(unreachable_patterns)]
        Some(_) => Err(super::idx_tensor::unsupported_execution_context_error()),
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
    let result = match context {
        Some(ExecutionContext::Cpu(context)) => context
            .with_session(|engine| f(engine))
            .map_err(IdxTensorError::from),
        #[cfg(feature = "tenferro-cuda")]
        Some(ExecutionContext::Cuda(context)) => context
            .with_session(|engine| f(engine))
            .map_err(IdxTensorError::from),
        None => with_runtime_engine(runtime, |engine| f(engine)).map_err(IdxTensorError::from),
        // INVARIANT: reachable only in the mixed configuration of
        // tensor4all-rs#869 (tensorbackend built with CUDA, core without), where
        // `ExecutionContext::Cuda` exists but this build cannot execute in it; in
        // every other configuration `Cpu` is the only variant.
        #[cfg(not(feature = "tenferro-cuda"))]
        #[allow(unreachable_patterns)]
        Some(_) => Err(super::idx_tensor::unsupported_execution_context_error()),
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
    let result = match context {
        Some(ExecutionContext::Cpu(context)) => context
            .with_session(|engine| f(engine))
            .map_err(IdxTensorError::from),
        #[cfg(feature = "tenferro-cuda")]
        Some(ExecutionContext::Cuda(context)) => context
            .with_session(|engine| f(engine))
            .map_err(IdxTensorError::from),
        None => with_runtime_engine(runtime, |engine| f(engine)).map_err(IdxTensorError::from),
        // INVARIANT: reachable only in the mixed configuration of
        // tensor4all-rs#869 (tensorbackend built with CUDA, core without), where
        // `ExecutionContext::Cuda` exists but this build cannot execute in it; in
        // every other configuration `Cpu` is the only variant.
        #[cfg(not(feature = "tenferro-cuda"))]
        #[allow(unreachable_patterns)]
        Some(_) => Err(super::idx_tensor::unsupported_execution_context_error()),
    };
    result?.map_err(eager_error)
}

fn eager_error(error: impl std::error::Error + Send + Sync + 'static) -> IdxTensorError {
    IdxTensorError::materialization(anyhow::Error::new(error))
}

#[cfg(test)]
mod tests;
