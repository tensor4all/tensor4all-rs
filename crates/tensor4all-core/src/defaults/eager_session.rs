//! Engine selection for the implicit and context-scoped eager frontends.
//!
//! Every entry names its engine route with [`EngineRoute`]. A context-aware entry
//! point (`*_in(&ExecutionContext, ..)`) passes [`EngineRoute::Context`] and runs
//! on that context's engine. An implicit entry point has no context to consult:
//! it passes [`EngineRoute::Legacy`], the documented migration bridge, which
//! resolves the engine from the eager runtime of the operand it acts on and uses
//! the process-global default CPU context only when that runtime has no live
//! owner (`global-defaults` builds). There is no `None` spelling: a route that
//! may reach the process-global engine says `Legacy` at the call site
//! (tensor4all-rs#1378). The legacy route is transitional and is removed with the
//! implicit API (#1376 stage D2).
//!
//! [`with_eager_on`] is the engine-taking entry of stage A2 (#1393): it runs a
//! caller-owned [`Runtime`] on the caller's `&mut` engine after validating the
//! operands, with no context, lock or owner lookup.

use std::sync::Arc;

use tenferro::{EngineTarget, ExecutionContextIdentity, ExecutionEngine};
use tenferro_ad::{EagerRuntime, EagerSession};
use tenferro_tensor::{Engine, MemoryKind, TensorRead};
use tensor4all_tensorbackend::{CpuEngine, ExecutionContext, Runtime};

use crate::implicit_route::with_runtime_engine_or_global;

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
/// This mirrors the input contracts the engines register with the pinned
/// tenferro (`tenferro-cpu` `cpu_input_signature`, `tenferro-gpu`
/// `cuda_input_signature`), so a value accepted here is not refused later by
/// `prepare`:
///
/// - a CPU engine reads host memory with no backend family and no allocation
///   domain, or host or managed memory a backend allocated in the engine's own
///   domain;
/// - any other engine reads only device memory a backend allocated in its own
///   allocation domain. The domain identifies the engine's device, so the
///   device ordinal needs no separate check.
fn engine_reads(target: &EngineTarget, value: &TensorRead<'_>) -> bool {
    let domain = value.allocation_domain();
    let family = value.backend_family();
    let in_own_domain = family.is_some()
        && target.allocation_domain().is_some()
        && domain == target.allocation_domain();
    let memory_kind = &value.placement().memory_kind;
    if !is_cpu_engine(target) {
        return *memory_kind == MemoryKind::Device && in_own_domain;
    }
    match memory_kind {
        MemoryKind::PinnedHost | MemoryKind::UnpinnedHost => {
            (family.is_none() && domain.is_none()) || in_own_domain
        }
        MemoryKind::Managed => in_own_domain,
        MemoryKind::Device | MemoryKind::Other(_) => false,
    }
}

/// Which engine an eager entry runs on.
///
/// There is deliberately no `Option<&ExecutionContext>` here: `None` used to mean
/// "the operand's owner, else the process-global context" without saying so
/// (tensor4all-rs#1378). Each caller now names its route.
#[derive(Clone, Copy, Debug)]
pub(crate) enum EngineRoute<'a> {
    /// The caller's context: exactly this context's engine, never another one.
    Context(&'a ExecutionContext),
    /// The legacy migration bridge (design doc section 7.0) for an implicit
    /// entry that has no engine yet: the engine of the context owning the
    /// operand's runtime, else the process-global context through
    /// [`with_runtime_engine_or_global`]. Removed in #1376 stage D2.
    Legacy,
}

impl<'a> EngineRoute<'a> {
    /// The route for a public `Option<&ExecutionContext>` parameter: the
    /// caller's context when given, else the [`Legacy`](Self::Legacy) bridge.
    ///
    /// Naming the bridge here keeps the choice visible where an optional
    /// context parameter (design doc family F2) is forwarded.
    pub(crate) fn context_or_legacy(context: Option<&'a ExecutionContext>) -> Self {
        context.map_or(Self::Legacy, Self::Context)
    }
}

/// Run one eager session on the engine selected by `route`.
///
/// [`EngineRoute::Legacy`] selects the engine of the CPU or CUDA context that
/// owns `runtime`'s eager runtime, falling back to the process-global default
/// CPU context when that runtime has no live owner (the explicit legacy bridge).
/// [`EngineRoute::Context`] selects that exact context's engine; a nested entry on the
/// same thread is reported as the context's typed
/// [`CpuExecutionContextError::Reentered`](tensor4all_tensorbackend::CpuExecutionContextError::Reentered)
/// or `CudaExecutionContextError::Reentered` instead of waiting for the engine it
/// already holds.
pub(crate) fn with_eager_session<T, E>(
    runtime: &Arc<EagerRuntime>,
    route: EngineRoute<'_>,
    f: impl FnOnce(&mut EagerSession<'_>) -> std::result::Result<T, E>,
) -> std::result::Result<T, IdxTensorError>
where
    E: From<tenferro_ad::Error> + std::error::Error + Send + Sync + 'static,
{
    with_eager_session_typed(runtime, route, f)?.map_err(eager_error)
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
    route: EngineRoute<'_>,
    f: impl FnOnce(&mut EagerSession<'_>) -> std::result::Result<T, E>,
) -> std::result::Result<std::result::Result<T, E>, IdxTensorError>
where
    E: From<tenferro_ad::Error>,
{
    match route {
        EngineRoute::Context(ExecutionContext::Cpu(context)) => context
            .with_session(|engine| runtime.with_eager_session(engine, f))
            .map_err(IdxTensorError::from),
        #[cfg(feature = "tenferro-cuda")]
        EngineRoute::Context(ExecutionContext::Cuda(context)) => context
            .with_session(|engine| runtime.with_eager_session(engine, f))
            .map_err(IdxTensorError::from),
        EngineRoute::Legacy => {
            with_runtime_engine_or_global(runtime, |engine| runtime.with_eager_session(engine, f))
                .map_err(IdxTensorError::from)
        }
        // INVARIANT: reachable only in the mixed configuration of
        // tensor4all-rs#869 (tensorbackend built with CUDA, core without), where
        // `ExecutionContext::Cuda` exists but this build cannot execute in it; in
        // every other configuration `Cpu` is the only variant.
        #[cfg(not(feature = "tenferro-cuda"))]
        #[allow(unreachable_patterns)]
        EngineRoute::Context(_) => Err(super::idx_tensor::unsupported_execution_context_error()),
    }
}

/// Run one engine operation on the engine selected by `context`.
///
/// The contract is the same as [`with_eager_session`], for operations that take
/// a [`tenferro_tensor::Engine`] directly instead of an eager session. `runtime`
/// is the eager runtime of the operand the operation acts on, and decides the
/// engine on the [`EngineRoute::Legacy`] route.
pub(crate) fn with_engine<T, E>(
    runtime: &Arc<EagerRuntime>,
    route: EngineRoute<'_>,
    f: impl FnOnce(&mut dyn Engine) -> std::result::Result<T, E>,
) -> std::result::Result<T, IdxTensorError>
where
    E: std::error::Error + Send + Sync + 'static,
{
    let result = match route {
        EngineRoute::Context(ExecutionContext::Cpu(context)) => context
            .with_session(|engine| f(engine))
            .map_err(IdxTensorError::from),
        #[cfg(feature = "tenferro-cuda")]
        EngineRoute::Context(ExecutionContext::Cuda(context)) => context
            .with_session(|engine| f(engine))
            .map_err(IdxTensorError::from),
        EngineRoute::Legacy => {
            with_runtime_engine_or_global(runtime, |engine| f(engine)).map_err(IdxTensorError::from)
        }
        // INVARIANT: reachable only in the mixed configuration of
        // tensor4all-rs#869 (tensorbackend built with CUDA, core without), where
        // `ExecutionContext::Cuda` exists but this build cannot execute in it; in
        // every other configuration `Cpu` is the only variant.
        #[cfg(not(feature = "tenferro-cuda"))]
        #[allow(unreachable_patterns)]
        EngineRoute::Context(_) => Err(super::idx_tensor::unsupported_execution_context_error()),
    };
    result?.map_err(eager_error)
}

/// The tenferro `dot_general` options for per-operand conjugation flags.
pub(crate) fn dot_general_conj_options(
    lhs_conj: bool,
    rhs_conj: bool,
) -> tenferro_tensor::DotGeneralOptions {
    let conj = |flag: bool| {
        if flag {
            tenferro_tensor::Conj::Yes
        } else {
            tenferro_tensor::Conj::No
        }
    };
    tenferro_tensor::DotGeneralOptions::default()
        .lhs(conj(lhs_conj))
        .rhs(conj(rhs_conj))
}

/// One axis index as the signed axis tenferro's operations take.
///
/// # Errors
///
/// Returns an error for an index above `isize::MAX`, which no tensor rank reaches.
pub(crate) fn signed_axis(axis: usize) -> anyhow::Result<isize> {
    isize::try_from(axis).map_err(|_| anyhow::anyhow!("axis {axis} exceeds isize::MAX"))
}

/// Axis indices as the signed axes tenferro's operations take.
///
/// # Errors
///
/// Returns an error for an index above `isize::MAX`, which no tensor rank reaches.
pub(crate) fn signed_axes(axes: &[usize]) -> anyhow::Result<Vec<isize>> {
    axes.iter().map(|&axis| signed_axis(axis)).collect()
}

fn eager_error(error: impl std::error::Error + Send + Sync + 'static) -> IdxTensorError {
    IdxTensorError::materialization(anyhow::Error::new(error))
}

#[cfg(test)]
mod tests;
