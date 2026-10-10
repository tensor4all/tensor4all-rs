//! Explicit and optional process-global tenferro CPU execution contexts.

use std::cell::Cell;
use std::sync::{Arc, Mutex, OnceLock};

use tenferro::{CompiledGraph, GraphCompiler, Runtime, Tensor, TracedGraph};
use tenferro_ad::{AdContext, EagerRuntime};
use tenferro_cpu::{BufferPoolStats, CpuBackend, CpuContext};
use tenferro_tensor::{BackendSession, BackendSessionHost};

/// Caller-owned execution domain used by context-aware tensor algorithms.
///
/// Values are validated against the exact runtime represented by the selected
/// context; no implicit host/device transfer is performed by this enum.
#[derive(Clone, Debug)]
pub enum ExecutionContext {
    /// Host execution through one caller-owned CPU context.
    Cpu(Arc<CpuExecutionContext>),
    /// CUDA execution through one caller-owned CUDA context.
    #[cfg(feature = "tenferro-cuda")]
    Cuda(Arc<crate::cuda::CudaExecutionContext>),
}

impl ExecutionContext {
    /// Check whether this is the process-global CPU context.
    ///
    /// Compatibility boundary: legacy CPU-global callers route through the
    /// historical host code paths (bitwise-identical numerics) while explicit
    /// contexts use the scoped primitives. Without the global-defaults
    /// feature there is no global context, so this always reports false.
    pub fn is_global_default_cpu(&self) -> bool {
        #[cfg(feature = "global-defaults")]
        {
            match self {
                ExecutionContext::Cpu(context) => {
                    let own = context.eager_runtime().map(|runtime| runtime.id());
                    let global = defaults::default_eager_ctx().map(|runtime| runtime.id());
                    matches!((own, global), (Ok(a), Ok(b)) if a == b)
                }
                #[cfg(feature = "tenferro-cuda")]
                ExecutionContext::Cuda(_) => false,
            }
        }
        #[cfg(not(feature = "global-defaults"))]
        {
            let _ = self;
            false
        }
    }
}

/// Error returned by explicit CPU context graph or eager-runtime operations.
///
/// The original tenferro diagnostic is retained as the error source.
///
/// # Examples
///
/// ```
/// use std::error::Error;
/// use std::sync::Arc;
/// use tensor4all_tensorbackend::CpuExecutionContextError;
///
/// let error = CpuExecutionContextError::Initialization {
///     component: "graph runtime",
///     source: Arc::new(std::io::Error::other("registration failed")),
/// };
/// assert!(error.source().is_some());
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum CpuExecutionContextError {
    /// A context-owned graph or eager runtime could not be initialized.
    #[error("failed to initialize {component}: {source}")]
    Initialization {
        /// Context component being initialized.
        component: &'static str,
        /// Original tenferro diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// tenferro rejected entry into the canonical CPU session.
    #[error("CPU canonical session entry failed: {source}")]
    SessionEntry {
        /// Original tenferro diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// Graph compilation or execution failed.
    #[error("CPU graph {operation} failed: {source}")]
    Graph {
        /// Graph operation that failed.
        operation: &'static str,
        /// Original tenferro diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

const CANONICAL_SESSION_REENTRY_MESSAGE: &str = "recursive tensorbackend canonical session entry";

/// Backend name carried by a canonical-session reentry rejection.
const CANONICAL_SESSION_BACKEND: &str = "CpuExecutionContext";

thread_local! {
    static CANONICAL_SESSION_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

struct CanonicalSessionGuard {
    previous: bool,
}

impl CanonicalSessionGuard {
    /// Whether a canonical session is already active on this thread.
    fn is_active() -> bool {
        CANONICAL_SESSION_ACTIVE.with(Cell::get)
    }

    /// Report a nested canonical session as the typed entry rejection instead of
    /// panicking, so a public entry point never aborts a caller.
    fn reentry_error() -> CpuExecutionContextError {
        CpuExecutionContextError::SessionEntry {
            source: Arc::new(tenferro_tensor::SessionEntryError::Reentered {
                backend: CANONICAL_SESSION_BACKEND,
            }),
        }
    }

    fn enter() -> Self {
        // The entry point rejected a nested session before this ran; reaching an
        // active guard here would mean a caller bypassed the canonical entry.
        debug_assert!(!Self::is_active(), "{CANONICAL_SESSION_REENTRY_MESSAGE}");
        CANONICAL_SESSION_ACTIVE.with(|active| Self {
            previous: active.replace(true),
        })
    }
}

impl Drop for CanonicalSessionGuard {
    fn drop(&mut self) {
        CANONICAL_SESSION_ACTIVE.with(|active| active.set(self.previous));
    }
}

/// Run one concrete session on `backend` under the canonical session guard.
///
/// # Errors
///
/// Returns [`CpuExecutionContextError::SessionEntry`] when tenferro rejects the
/// entry, for example because an execution is already active on this thread.
fn run_canonical_session<R>(
    backend: &mut CpuBackend,
    f: impl FnOnce(&mut dyn BackendSession) -> R,
) -> Result<R, CpuExecutionContextError> {
    backend
        .with_backend_session(|session| {
            let _guard = CanonicalSessionGuard::enter();
            f(session)
        })
        .map_err(|source| CpuExecutionContextError::SessionEntry {
            source: Arc::new(source),
        })
}

impl CpuExecutionContextError {
    fn initialization(
        component: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Initialization {
            component,
            source: Arc::new(source),
        }
    }

    fn graph(
        operation: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Graph {
            operation,
            source: Arc::new(source),
        }
    }
}

struct GraphState {
    compiler: GraphCompiler,
    runtime: Runtime,
    backend: CpuBackend,
}

/// Caller-owned CPU execution domain for plain, graph, and eager-AD work.
///
/// The supplied backend is the only source of CPU execution resources for every
/// entry from a thread that is not a Rayon worker. A plain session entered from a
/// Rayon worker runs inline and single-threaded on a context-local pool-less CPU
/// backend instead: a worker that waits for a pool install is handed more of the
/// enclosing pool's work, and that work may enter a session itself
/// (tensor4all-rs#830). Backend clones preserve its runtime identity; this
/// constructor never uses `CpuBackend::new`, `CpuContext::from_env`, or a
/// process-global fallback.
/// Graph preparation caches and the eager runtime are owned by this context and
/// are released when it is dropped.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let threads = context.with_backend(|backend| backend.num_threads());
/// assert_eq!(threads, 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CpuExecutionContext {
    backend: Mutex<CpuBackend>,
    inline: OnceLock<CpuBackend>,
    graph: OnceLock<Result<Mutex<GraphState>, CpuExecutionContextError>>,
    eager: OnceLock<Result<Arc<EagerRuntime>, CpuExecutionContextError>>,
}

impl std::fmt::Debug for CpuExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuExecutionContext")
            .field("graph_initialized", &self.graph.get().is_some())
            .field("eager_initialized", &self.eager.get().is_some())
            .finish_non_exhaustive()
    }
}

impl CpuExecutionContext {
    /// Create an execution context from a caller-selected CPU backend.
    ///
    /// Runtime construction is lazy, so creating a context cannot fail and does
    /// not allocate another executor or consult environment configuration.
    pub fn from_backend(backend: CpuBackend) -> Self {
        Self {
            backend: Mutex::new(backend),
            inline: OnceLock::new(),
            graph: OnceLock::new(),
            eager: OnceLock::new(),
        }
    }

    /// Run a plain tensor operation with this context's backend.
    ///
    /// The closure runs while the context-local backend lock is held. A poisoned
    /// lock is recovered because tenferro validates every new backend session.
    pub fn with_backend<R>(&self, f: impl FnOnce(&mut CpuBackend) -> R) -> R {
        let mut backend = match self.backend.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        f(&mut backend)
    }

    /// Run `f` in one canonical session of this context.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::SessionEntry`] when tenferro rejects
    /// the session entry.
    pub(crate) fn with_session<R>(
        &self,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> Result<R, CpuExecutionContextError> {
        if CanonicalSessionGuard::is_active() {
            return Err(CanonicalSessionGuard::reentry_error());
        }
        // A Rayon worker can be handed more of the enclosing pool's work while
        // tenferro installs this session into the context's own pool. That stolen
        // work may enter a session itself, on a thread whose tenferro execution is
        // already active, which tenferro rejects. Entering from a worker therefore
        // runs the session inline and single-threaded on that worker and leaves
        // parallelism to the enclosing pool.
        if rayon::current_thread_index().is_some() {
            let mut backend = self.inline_backend();
            return run_canonical_session(&mut backend, f);
        }
        let mut backend = match self.backend.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        run_canonical_session(&mut backend, f)
    }

    /// Backend used for one session entered from a Rayon worker.
    ///
    /// `CpuContext::with_threads(1)` owns no Rayon pool, so its executor runs
    /// every operation on the calling thread and never installs a session into a
    /// pool the caller does not belong to.
    fn inline_backend(&self) -> CpuBackend {
        self.inline
            .get_or_init(|| match CpuContext::with_threads(1) {
                Ok(context) => CpuBackend::from_context(Arc::new(context)),
                // INVARIANT: `CpuContext::with_threads` rejects only a zero worker
                // count, which this call never passes, so this arm is unreachable.
                // Keeping the supplied backend is a last resort rather than a
                // policy: it is the only remaining backend this context owns.
                Err(_) => self.backend_clone(),
            })
            .clone()
    }

    fn backend_clone(&self) -> CpuBackend {
        self.with_backend(|backend| backend.clone())
    }

    fn graph_state(&self) -> Result<&Mutex<GraphState>, CpuExecutionContextError> {
        self.graph
            .get_or_init(|| {
                let backend = self.backend_clone();
                build_graph_runtime(&backend).map(|runtime| {
                    Mutex::new(GraphState {
                        compiler: GraphCompiler::new(),
                        runtime,
                        backend,
                    })
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    fn with_graph_state<R>(
        &self,
        f: impl FnOnce(&mut GraphCompiler, &mut Runtime, &mut CpuBackend) -> R,
    ) -> Result<R, CpuExecutionContextError> {
        let mut graph = match self.graph_state()?.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let GraphState {
            compiler,
            runtime,
            backend,
        } = &mut *graph;
        Ok(f(compiler, runtime, backend))
    }

    /// Compile a backend-neutral traced graph using this context's compiler cache.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when graph-runtime initialization or
    /// graph compilation fails.
    pub fn compile_graph(
        &self,
        graph: &TracedGraph,
    ) -> Result<CompiledGraph, CpuExecutionContextError> {
        self.with_graph_state(|compiler, _, _| compiler.compile_traced_graph(graph))?
            .map_err(|source| CpuExecutionContextError::graph("compilation", source))
    }

    /// Execute a compiled graph in this context's runtime and prepared-plan cache.
    ///
    /// `CompiledGraph` is backend-neutral. Backend-prepared executables and
    /// workspaces never leave this context-owned runtime.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when runtime initialization,
    /// preparation, or execution fails.
    pub fn run_graph(
        &self,
        graph: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>, CpuExecutionContextError> {
        self.with_graph_state(|_, runtime, _| runtime.run_compiled(graph, inputs))?
            .map_err(|source| CpuExecutionContextError::graph("execution", source))
    }

    /// Return this context's eager reverse-AD runtime.
    ///
    /// Repeated calls return the same runtime and therefore the same eager
    /// compilation cache.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when linalg AD-rule or eager-runtime
    /// registration fails.
    pub fn eager_runtime(&self) -> Result<Arc<EagerRuntime>, CpuExecutionContextError> {
        self.eager
            .get_or_init(|| build_eager_runtime(self.backend_clone()))
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)
    }

    /// Return statistics for this context's runtime-owned graph caches.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when graph initialization or the
    /// cache statistics query fails.
    pub fn graph_cache_stats(
        &self,
    ) -> Result<tenferro::RuntimeCacheStats, CpuExecutionContextError> {
        self.with_graph_state(|_, runtime, _| runtime.cache_stats())?
            .map_err(|source| CpuExecutionContextError::graph("cache statistics", source))
    }

    /// Return retained-buffer statistics for this context's graph backend.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when graph initialization or the
    /// backend statistics query fails.
    pub fn graph_buffer_pool_stats(&self) -> Result<BufferPoolStats, CpuExecutionContextError> {
        self.with_graph_state(|_, _, backend| backend.buffer_pool_stats())?
            .map_err(|source| CpuExecutionContextError::graph("buffer-pool statistics", source))
    }

    /// Release retained buffers owned by this context's graph backend.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when graph initialization or reset
    /// fails.
    pub fn reset_graph_buffer_pool(&self) -> Result<(), CpuExecutionContextError> {
        self.with_graph_state(|_, _, backend| backend.reset_buffer_pool())?
            .map_err(|source| CpuExecutionContextError::graph("buffer-pool reset", source))
    }

    /// Recreate this context's graph runtime and release its prepared caches.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when runtime reconstruction or
    /// buffer release fails.
    pub fn reset_graph_runtime(&self) -> Result<(), CpuExecutionContextError> {
        self.with_graph_state(|compiler, runtime, backend| {
            let replacement = build_graph_runtime(backend)?;
            *compiler = GraphCompiler::new();
            let old = std::mem::replace(runtime, replacement);
            drop(old);
            backend
                .reset_buffer_pool()
                .map_err(|source| CpuExecutionContextError::graph("buffer-pool reset", source))
        })??;
        Ok(())
    }
}

fn build_graph_runtime(backend: &CpuBackend) -> Result<Runtime, CpuExecutionContextError> {
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            tenferro_cpu::runtime_engine_registration(backend).map_err(|source| {
                CpuExecutionContextError::initialization("graph CPU engine", source)
            })?,
        )
        .map_err(|source| CpuExecutionContextError::initialization("graph CPU engine", source))?;
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CpuBackend>(
                tenferro_cpu::runtime_engine_id().map_err(|source| {
                    CpuExecutionContextError::initialization("einsum extension", source)
                })?,
            )
            .map_err(|source| {
                CpuExecutionContextError::initialization("einsum extension", source)
            })?,
        )
        .map_err(|source| CpuExecutionContextError::initialization("einsum extension", source))?;
    builder
        .build()
        .map_err(|source| CpuExecutionContextError::initialization("graph runtime", source))
}

fn build_eager_runtime(backend: CpuBackend) -> Result<Arc<EagerRuntime>, CpuExecutionContextError> {
    let ad_context = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().map_err(|source| {
            CpuExecutionContextError::initialization("linalg AD rules", source)
        })?)
        .map_err(|source| CpuExecutionContextError::initialization("linalg AD rules", source))?
        .build()
        .map_err(|source| CpuExecutionContextError::initialization("AD context", source))?;
    let runtime = EagerRuntime::with_cpu_backend_and_ad_context(backend, &ad_context)
        .map_err(|source| CpuExecutionContextError::initialization("eager runtime", source))?;
    // [AI Supplied] Install the built-in extension modules before publishing
    // the shared context. Lazy first use reconfigures the runtime and advances
    // its epoch; doing that after tensors have prepared AD derivatives can
    // invalidate those prepared programs under parallel first use.
    let engine_id = tenferro_cpu::runtime_engine_id()
        .map_err(|source| CpuExecutionContextError::initialization("CPU runtime engine", source))?;
    let einsum_module = tenferro_einsum::extension_module::<CpuBackend>(engine_id.clone())
        .map_err(|source| CpuExecutionContextError::initialization("einsum extension", source))?;
    runtime
        .install_extension_module(einsum_module)
        .map_err(|source| CpuExecutionContextError::initialization("einsum runtime", source))?;
    let linalg_module = tenferro_linalg::extension_module::<CpuBackend>(engine_id)
        .map_err(|source| CpuExecutionContextError::initialization("linalg extension", source))?;
    runtime
        .install_extension_module(linalg_module)
        .map_err(|source| CpuExecutionContextError::initialization("linalg runtime", source))?;
    Ok(runtime)
}

#[cfg(feature = "global-defaults")]
mod defaults {
    use super::*;
    use tenferro_cpu::CpuContext;

    static DEFAULT_CONTEXT: OnceLock<Arc<CpuExecutionContext>> = OnceLock::new();

    #[cfg(test)]
    thread_local! {
        static FORCE_EAGER_CONTEXT_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    #[cfg(test)]
    static DEFAULT_CONTEXT_HITS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    /// Borrow the process-global CPU execution context.
    ///
    /// New code must take a caller-owned context; this accessor exists for the
    /// process-global convenience path and for callers that map the session-entry
    /// rejection into their own error type.
    pub(crate) fn default_context() -> &'static Arc<CpuExecutionContext> {
        DEFAULT_CONTEXT.get_or_init(|| {
            #[cfg(test)]
            DEFAULT_CONTEXT_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Arc::new(CpuExecutionContext::from_backend(CpuBackend::from_context(
                Arc::new(CpuContext::from_env()),
            )))
        })
    }

    /// Error returned when the process-global eager AD runtime cannot be initialized.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::sync::Arc;
    /// use tensor4all_tensorbackend::EagerContextError;
    ///
    /// let error = EagerContextError::Registration {
    ///     source: Arc::new(std::io::Error::other("registration failed")),
    /// };
    /// assert!(error.source().is_some());
    /// ```
    #[derive(Debug, Clone, thiserror::Error)]
    pub enum EagerContextError {
        /// The tenferro linalg AD extension rule could not be registered.
        #[error("failed to register tenferro linalg AD rule: {source}")]
        Registration {
            /// Original diagnostic returned by tenferro.
            #[source]
            source: Arc<dyn std::error::Error + Send + Sync + 'static>,
        },
    }

    /// Run a closure against the optional process-global CPU backend.
    pub fn with_default_backend<R>(f: impl FnOnce(&mut CpuBackend) -> R) -> R {
        default_context().with_backend(f)
    }

    /// Run `f` in one canonical session of the process-global context.
    ///
    /// The entry rejection stays inside tenferro's error type: it is reported as
    /// a `tenferro_tensor::Error` classified as runtime state, so callers keep
    /// one error type for session work while the source chain keeps tenferro's
    /// admission diagnostic.
    ///
    /// # Errors
    ///
    /// Returns the tenferro diagnostic when the session entry is rejected or
    /// when `f` fails.
    pub(crate) fn with_default_session<T: Send>(
        f: impl FnOnce(&mut dyn BackendSession) -> Result<T, tenferro_tensor::Error> + Send,
    ) -> Result<T, tenferro_tensor::Error> {
        default_context()
            .with_session(f)
            .map_err(|source| {
                tenferro_tensor::Error::runtime_state_source("canonical session entry", source)
            })
            .and_then(std::convert::identity)
    }

    pub(crate) fn with_default_graph_runtime<R>(
        f: impl FnOnce(&mut GraphCompiler, &Runtime, &mut CpuBackend) -> R,
    ) -> anyhow::Result<R> {
        default_context()
            .with_graph_state(|compiler, runtime, backend| f(compiler, runtime, backend))
            .map_err(anyhow::Error::new)
    }

    pub(crate) fn default_engine_buffer_pool_stats() -> anyhow::Result<BufferPoolStats> {
        default_context()
            .graph_buffer_pool_stats()
            .map_err(anyhow::Error::new)
    }

    pub(crate) fn reset_default_engine_buffer_pool() -> anyhow::Result<()> {
        default_context()
            .reset_graph_buffer_pool()
            .map_err(anyhow::Error::new)
    }

    pub(crate) fn reset_default_engine() -> anyhow::Result<()> {
        default_context()
            .reset_graph_runtime()
            .map_err(anyhow::Error::new)
    }

    /// Return the optional process-global eager context used by convenience APIs.
    ///
    /// # Errors
    ///
    /// Returns [`EagerContextError::Registration`] when eager runtime
    /// initialization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tensor4all_tensorbackend::default_eager_ctx;
    ///
    /// let first = default_eager_ctx().unwrap();
    /// let second = default_eager_ctx().unwrap();
    /// assert!(Arc::ptr_eq(&first, &second));
    /// ```
    pub fn default_eager_ctx() -> Result<Arc<EagerRuntime>, EagerContextError> {
        #[cfg(test)]
        if FORCE_EAGER_CONTEXT_FAILURE.with(std::cell::Cell::get) {
            return Err(EagerContextError::Registration {
                source: Arc::new(std::io::Error::other(
                    "forced default eager context registration failure",
                )),
            });
        }
        default_context()
            .eager_runtime()
            .map_err(|source| EagerContextError::Registration {
                source: Arc::new(source),
            })
    }

    /// Borrow the process-global CPU execution context.
    ///
    /// Compatibility entry for CPU-global convenience APIs (e.g. the legacy
    /// context-free SRC entry): host tensors constructed through the global
    /// default belong to this exact context, so they validate against it.
    /// New code must take a caller-owned context instead of consulting this.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{default_cpu_execution_context, ExecutionContext};
    ///
    /// let context = ExecutionContext::Cpu(default_cpu_execution_context());
    /// assert!(matches!(context, ExecutionContext::Cpu(_)));
    /// ```
    pub fn default_cpu_execution_context() -> Arc<CpuExecutionContext> {
        Arc::clone(default_context())
    }

    #[cfg(test)]
    pub(crate) fn default_context_hits() -> usize {
        DEFAULT_CONTEXT_HITS.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn with_forced_eager_context_failure<T>(f: impl FnOnce() -> T) -> T {
        let previous = FORCE_EAGER_CONTEXT_FAILURE.with(|failure| failure.replace(true));
        let result = f();
        FORCE_EAGER_CONTEXT_FAILURE.with(|failure| failure.set(previous));
        result
    }
}

#[cfg(feature = "global-defaults")]
pub(crate) use defaults::default_context;
#[cfg(all(test, feature = "global-defaults"))]
pub(crate) use defaults::with_forced_eager_context_failure;
#[cfg(feature = "global-defaults")]
pub use defaults::{
    default_cpu_execution_context, default_eager_ctx, with_default_backend, EagerContextError,
};
#[cfg(feature = "global-defaults")]
pub(crate) use defaults::{
    default_engine_buffer_pool_stats, reset_default_engine, reset_default_engine_buffer_pool,
    with_default_graph_runtime, with_default_session,
};

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;
    use tenferro::program::{CoreSemanticOp, ProgramInputSpec};
    use tenferro::{DType, TensorSessionOpsExt, TraceContext};
    use tenferro_ad::EagerTensor;

    fn context() -> CpuExecutionContext {
        CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap())
    }

    #[test]
    fn explicit_session_runs_a_concrete_operation() {
        let context = context();
        let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap();
        let result = context
            .with_session(|session| lhs.matmul(&rhs, session))
            .unwrap()
            .unwrap();

        assert_eq!(result.as_slice::<f64>().unwrap(), &[23.0, 34.0]);
    }

    #[test]
    fn recursive_session_entry_fails_before_lock_and_restores_guard() {
        let context = context();
        let error = context
            .with_session(|_| context.with_session(|_| ()))
            .expect("the outer session entry")
            .expect_err("a recursive canonical session entry must be rejected");
        let CpuExecutionContextError::SessionEntry { source } = &error else {
            panic!("the rejection must be a typed session entry error: {error}");
        };
        assert!(matches!(
            source.downcast_ref::<tenferro_tensor::SessionEntryError>(),
            Some(tenferro_tensor::SessionEntryError::Reentered { .. })
        ));

        // The guard is restored, so an independent entry still works.
        assert_eq!(context.with_session(|_| 7usize).unwrap(), 7);
    }

    #[test]
    fn explicit_plain_graph_and_eager_paths_share_only_the_supplied_backend() {
        let context = context();
        assert!(format!("{context:?}").contains("graph_initialized: false"));
        assert_eq!(context.with_backend(|backend| backend.num_threads()), 1);

        let mut trace = TraceContext::new();
        let input = trace
            .input(ProgramInputSpec::new(DType::F64, [2_usize.into()]))
            .unwrap();
        let output = trace.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
        let graph = trace.finish(&[output]).unwrap();
        let compiled = context.compile_graph(&graph).unwrap();
        let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();
        let output = context.run_graph(&compiled, &[&input]).unwrap();
        assert_eq!(output[0].as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
        context.run_graph(&compiled, &[&input]).unwrap();
        let cached = context.graph_cache_stats().unwrap().prepared_plans;
        assert!(cached.entries > 0);
        assert!(cached.hits > 0);
        context.reset_graph_runtime().unwrap();
        assert_eq!(
            context.graph_cache_stats().unwrap().prepared_plans.entries,
            0
        );

        let eager = context.eager_runtime().unwrap();
        assert!(Arc::ptr_eq(&eager, &context.eager_runtime().unwrap()));
    }

    #[test]
    fn separate_eager_contexts_reject_cross_context_operations() {
        let first = context().eager_runtime().unwrap();
        let second = context().eager_runtime().unwrap();
        let a = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
            first.clone(),
        )
        .unwrap();
        let b = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
            second,
        )
        .unwrap();
        let error = first
            .with_eager_session(|session| session.add(&a, &b))
            .expect_err("cross-context addition must be rejected");
        assert!(matches!(error, tenferro_ad::Error::ContextMismatch { .. }));
    }

    #[test]
    fn caller_supplied_context_stays_caller_owned_after_context_drop() {
        let executor = Arc::new(CpuContext::with_threads(1).unwrap());
        let context =
            CpuExecutionContext::from_backend(CpuBackend::from_context(Arc::clone(&executor)));
        assert_eq!(context.with_backend(|backend| backend.num_threads()), 1);
        drop(context);
        assert_eq!(executor.num_threads(), 1);
    }

    #[test]
    fn independent_contexts_do_not_share_a_backend_mutex() {
        let first = Arc::new(context());
        let second = Arc::new(context());
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let handles = [first, second].map(|context| {
            let entered_tx = entered_tx.clone();
            let release_rx = Arc::clone(&release_rx);
            std::thread::spawn(move || {
                context.with_backend(|_| {
                    entered_tx.send(()).unwrap();
                    release_rx.lock().unwrap().recv().unwrap();
                });
            })
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        release_tx.send(()).unwrap();
        release_tx.send(()).unwrap();
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn session_from_a_rayon_worker_completes_without_waiting_on_the_context_pool() {
        use rayon::prelude::*;

        // The context owns a two-worker Rayon pool, and the enclosing pool hands
        // the worker waiting for a session install more of its own items. A
        // session entry that installs into the context pool therefore re-enters a
        // session on a thread whose execution is already active; the one-worker
        // case makes that sequence certain.
        let context = Arc::new(CpuExecutionContext::from_backend(
            CpuBackend::with_threads(2).unwrap(),
        ));
        for enclosing_workers in [1usize, 2] {
            let pool = Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(enclosing_workers)
                    .build()
                    .unwrap(),
            );
            let (sender, receiver) = mpsc::channel();
            let context = Arc::clone(&context);
            std::thread::spawn(move || {
                let results = pool.install(|| {
                    (0..2usize)
                        .into_par_iter()
                        .map(|_| {
                            let lhs = Tensor::from_vec_col_major(
                                vec![2, 2],
                                vec![1.0_f64, 2.0, 3.0, 4.0],
                            )
                            .unwrap();
                            let rhs =
                                Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap();
                            // tenferro-rs #2004 rejects a worker entry it cannot
                            // wait for with a typed `SessionEntry` error instead of
                            // blocking on the context pool. Either outcome finishes,
                            // which is what this test observes. The worker path reaches
                            // tenferro through the process-global arbiter, which the
                            // rest of this parallel suite also uses, so a rejection is
                            // retried until the unrelated holder releases.
                            let deadline = std::time::Instant::now()
                                + std::time::Duration::from_secs(30);
                            let outcome = loop {
                                let outcome = context.with_session(|session| {
                                    lhs.matmul(&rhs, session)
                                });
                                let contended = matches!(
                                    &outcome,
                                    Err(CpuExecutionContextError::SessionEntry { source })
                                        if matches!(
                                            source.downcast_ref::<tenferro_tensor::SessionEntryError>(),
                                            Some(tenferro_tensor::SessionEntryError::Contended { .. })
                                        )
                                );
                                if contended && std::time::Instant::now() < deadline {
                                    std::thread::sleep(std::time::Duration::from_millis(1));
                                    continue;
                                }
                                break outcome;
                            };
                            match outcome {
                                Ok(Ok(product)) => {
                                    Some(product.as_slice::<f64>().unwrap().to_vec())
                                }
                                Ok(Err(error)) => panic!("session operation failed: {error}"),
                                Err(error) => {
                                    let CpuExecutionContextError::SessionEntry { source } = error
                                    else {
                                        panic!("worker entry must fail with the typed session error");
                                    };
                                    let rejection = source
                                        .downcast_ref::<tenferro_tensor::SessionEntryError>()
                                        .expect("the rejection must preserve tenferro's typed cause");
                                    assert!(
                                        matches!(
                                            rejection,
                                            tenferro_tensor::SessionEntryError::Contended { .. }
                                        ),
                                        "a worker cannot wait, so the rejection is contention: {rejection}"
                                    );
                                    None
                                }
                            }
                        })
                        .collect::<Vec<_>>()
                });
                let _ = sender.send(results);
            });

            let results = receiver.recv_timeout(Duration::from_secs(60)).expect(
                "a session entered from a Rayon worker must finish instead of waiting on the context pool",
            );
            assert_eq!(results.len(), 2, "enclosing workers = {enclosing_workers}");
            assert!(
                results.iter().any(Option::is_some),
                "at least one concurrent worker entry must complete a session, got {results:?}"
            );
            for values in results.into_iter().flatten() {
                assert_eq!(values, vec![23.0, 34.0]);
            }
        }
    }

    #[cfg(feature = "global-defaults")]
    #[test]
    fn explicit_paths_do_not_initialize_the_default_context() {
        let before = defaults::default_context_hits();
        let context = context();
        context.with_backend(|backend| assert_eq!(backend.num_threads(), 1));
        context.eager_runtime().unwrap();
        assert_eq!(defaults::default_context_hits(), before);
    }
}
