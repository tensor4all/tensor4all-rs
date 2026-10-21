//! Explicit and optional process-global tenferro CPU execution contexts.

use std::cell::Cell;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, OnceLock};

use tenferro::{
    CompiledGraph, ExecutionEngine, GraphCompiler, Runtime, Tensor, TensorRead, TracedGraph,
};
use tenferro_ad::{AdContext, EagerRuntime};
use tenferro_cpu::{BufferPoolStats, CpuEngine};

/// Failure reported when building a [`CpuBackend`] pool.
///
/// A pool build is the only fallible step of a [`CpuBackend`]: tenferro itself
/// never builds threads, so the caller's thread policy is the caller's error to
/// carry.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::CpuThreadError;
///
/// let error = CpuThreadError::ZeroThreads;
/// assert!(error.to_string().contains("at least one"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum CpuThreadError {
    /// A [`CpuBackend`] was asked for zero workers.
    #[error("a CPU backend needs at least one worker thread")]
    ZeroThreads,
    /// The Rayon pool for the requested worker count could not be built.
    #[error("failed to build the CPU worker pool: {source}")]
    Pool {
        /// Original Rayon diagnostic.
        #[source]
        source: Box<rayon::ThreadPoolBuildError>,
    },
}

/// Failure reported by [`set_default_cpu_threads`] when the default is fixed.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::AlreadyInitialized;
///
/// let error = AlreadyInitialized;
/// assert_eq!(
///     error.to_string(),
///     "the process-wide default CPU thread count is already initialized"
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the process-wide default CPU thread count is already initialized")]
pub struct AlreadyInitialized;

static DEFAULT_CPU_THREADS: OnceLock<usize> = OnceLock::new();

/// Fix the worker count [`CpuBackend::new`] uses for the rest of the process.
///
/// The value is read once, by the first [`CpuBackend::new`] call; a backend
/// built later keeps whatever policy it was built with. Call this before any
/// other thread can construct a default backend.
///
/// # Errors
///
/// Returns [`AlreadyInitialized`] when the default was already fixed, either by
/// an earlier call or by the first [`CpuBackend::new`].
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
/// use tensor4all_tensorbackend::{set_default_cpu_threads, AlreadyInitialized};
///
/// // Whichever call sets the default first wins, and every later one is
/// // rejected typed instead of panicking.
/// let _ = set_default_cpu_threads(NonZeroUsize::new(3).unwrap());
/// assert_eq!(
///     set_default_cpu_threads(NonZeroUsize::MAX),
///     Err(AlreadyInitialized)
/// );
/// ```
pub fn set_default_cpu_threads(threads: NonZeroUsize) -> Result<(), AlreadyInitialized> {
    DEFAULT_CPU_THREADS
        .set(threads.get())
        .map_err(|_| AlreadyInitialized)
}

/// The worker count [`CpuBackend::new`] uses: the fixed default, or one per
/// available core.
fn default_cpu_threads() -> usize {
    *DEFAULT_CPU_THREADS.get_or_init(|| {
        std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(1)
    })
}

/// tensor4all's owned CPU execution handle: one [`CpuEngine`].
///
/// tenferro no longer has a backend session or a `CpuContext`; a caller builds
/// a Rayon pool, hands it to [`CpuEngine::new`], and keeps the engine for as
/// long as it wants the thread policy, the buffer pool, the prepared-plan cache
/// and the contraction scratch to stay warm. This type is that caller-held
/// owner for tensor4all: the implicit frontend's resource owner during the
/// transition to engine-explicit APIs, which will replace it later.
///
/// It is **not** [`Clone`]. Cloning an execution domain was how the previous
/// session API shared one engine between callers; with a caller-held engine the
/// caller owns it outright, and a second execution domain needs a second
/// backend.
///
/// [`CpuBackend::new`] uses [`set_default_cpu_threads`] when it was called, and
/// otherwise one worker per available core. Upstream tenferro no longer reads
/// `RAYON_NUM_THREADS`; the previous default came from
/// `tenferro_cpu::CpuContext::from_env`, which read `RAYON_NUM_THREADS` (and fell
/// back to a single-threaded context when it was malformed) and otherwise used
/// the available parallelism. If the pool cannot be built, [`CpuBackend::new`]
/// falls back to [`CpuEngine::serial`] instead of panicking.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::CpuBackend;
/// use tenferro_cpu::CpuEngine;
///
/// // A serial backend owns no pool and runs every operation on the caller's thread.
/// let backend = CpuBackend::with_threads(1)?;
/// assert_eq!(backend.num_threads(), 1);
/// assert!(backend.engine().pool().is_none());
///
/// // A caller that already owns an engine can hand it in unchanged.
/// let backend = CpuBackend::from(CpuEngine::serial());
/// assert_eq!(backend.num_threads(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CpuBackend {
    engine: CpuEngine,
}

impl std::fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuBackend")
            .field("engine", &self.engine)
            .finish()
    }
}

impl CpuBackend {
    /// Build a backend with the default thread policy.
    ///
    /// The worker count is the value [`set_default_cpu_threads`] fixed, or one
    /// per available core. A worker stack of
    /// [`tenferro_cpu::DEFAULT_WORKER_STACK_BYTES`] is reserved for each worker,
    /// as the CPU providers require.
    ///
    /// The pool build is the one fallible step, and this constructor does not
    /// report it: if the pool cannot be built the backend is
    /// [`CpuEngine::serial`] instead, which is correct but uses no worker
    /// threads. Build the pool yourself and use [`CpuBackend::from`] when that
    /// difference matters.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// assert!(backend.num_threads() >= 1);
    /// ```
    pub fn new() -> Self {
        let threads = default_cpu_threads();
        Self::with_threads(threads).unwrap_or_else(|_| Self {
            engine: CpuEngine::serial(),
        })
    }

    /// Build a backend that may use `threads` workers.
    ///
    /// One worker builds a [`CpuEngine::serial`] engine, which owns no pool, so
    /// it stays usable from inside another pool. Larger counts build a Rayon
    /// pool with [`tenferro_cpu::thread_pool_builder`], so provider code gets the
    /// worker stack it needs.
    ///
    /// # Errors
    ///
    /// Returns [`CpuThreadError::ZeroThreads`] for zero, and
    /// [`CpuThreadError::Pool`] when Rayon cannot build the pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CpuBackend, CpuThreadError};
    ///
    /// assert!(matches!(
    ///     CpuBackend::with_threads(0),
    ///     Err(CpuThreadError::ZeroThreads)
    /// ));
    /// let backend = CpuBackend::with_threads(2)?;
    /// assert_eq!(backend.num_threads(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_threads(threads: usize) -> Result<Self, CpuThreadError> {
        if threads == 0 {
            return Err(CpuThreadError::ZeroThreads);
        }
        if threads == 1 {
            return Ok(Self {
                engine: CpuEngine::serial(),
            });
        }
        let pool = tenferro_cpu::thread_pool_builder()
            .num_threads(threads)
            .build()
            .map_err(|source| CpuThreadError::Pool {
                source: Box::new(source),
            })?;
        Ok(Self {
            engine: CpuEngine::new(pool),
        })
    }

    /// The number of workers operations on this backend may use.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// assert_eq!(CpuBackend::with_threads(1)?.num_threads(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn num_threads(&self) -> usize {
        self.engine.num_threads()
    }

    /// Borrow the engine this backend owns.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(1)?;
    /// assert_eq!(backend.engine().buffer_pool_len(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn engine(&self) -> &CpuEngine {
        &self.engine
    }

    /// Borrow the engine mutably, for one operation or one session.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// let mut backend = CpuBackend::with_threads(1)?;
    /// backend.engine_mut().reset_buffer_pool();
    /// assert_eq!(backend.engine().buffer_pool_len(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn engine_mut(&mut self) -> &mut CpuEngine {
        &mut self.engine
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for CpuBackend {
    type Target = CpuEngine;

    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

impl DerefMut for CpuBackend {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.engine
    }
}

impl From<CpuEngine> for CpuBackend {
    fn from(engine: CpuEngine) -> Self {
        Self { engine }
    }
}

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
    /// The calling thread is already inside a session of this context.
    ///
    /// One context owns one engine and lends it to one caller at a time. A
    /// session entered from inside another session on the same thread would
    /// wait for a lock that thread already holds, so it is reported as this
    /// typed error before any lock is taken.
    #[error("recursive CPU execution-context session entry on one thread")]
    Reentered,
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

    fn enter() -> Self {
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

/// The engine a context owns, plus the graph runtime derived from it.
///
/// The registration the graph runtime holds is derived from the engine, and
/// `Runtime::run` needs `&mut engine`, so both live under one lock. The
/// [`GraphCompiler`] is not here: it never touches the engine, so it has its own
/// lock and a compile never waits for a running graph.
struct EngineState {
    backend: CpuBackend,
    runtime: Option<Runtime>,
}

impl EngineState {
    fn new(backend: CpuBackend) -> Self {
        Self {
            backend,
            runtime: None,
        }
    }

    /// The graph runtime of this context, built from the engine on first use.
    fn runtime(&mut self) -> Result<&mut Runtime, CpuExecutionContextError> {
        let Self { backend, runtime } = self;
        match runtime {
            Some(runtime) => Ok(runtime),
            None => {
                let built = build_graph_runtime(backend)?;
                Ok(runtime.insert(built))
            }
        }
    }

    /// Replace the engine if a panic left it broken.
    ///
    /// A poisoned lock means a previous holder panicked while it had the engine
    /// borrowed. tenferro marks an engine broken when that happened during an
    /// operation, and refuses later calls on it. The engine is then replaced by
    /// a fresh one of the same thread count: this context owns the engine, so
    /// there is nothing else to preserve. If the replacement pool cannot be
    /// built the broken engine stays, and its operations report tenferro's typed
    /// broken-engine refusal instead.
    fn recover_from_panic(&mut self) {
        if !self.backend.engine().is_broken() {
            return;
        }
        let threads = self.backend.num_threads();
        if let Ok(replacement) = CpuBackend::with_threads(threads) {
            self.backend = replacement;
            // The prepared plans were made for the replaced engine's workspace;
            // rebuild the runtime against the new one.
            self.runtime = None;
        }
    }
}

/// Caller-owned CPU execution domain for plain, graph, and eager-AD work.
///
/// The context owns one [`CpuBackend`] and lends its engine to one caller at a
/// time. A session entered from a Rayon worker runs inline on a fresh serial
/// engine that shares the context engine's prepared-plan cache instead of
/// waiting for the context engine: a worker that waits for a pool install is
/// handed more of the enclosing pool's work, and that work may enter a session
/// itself (tensor4all-rs#830). Graph preparation caches, the engine's buffer
/// pool and the eager runtime are owned by this context and are released when
/// it is dropped.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tensor4all_tensorbackend::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let threads = context.with_backend(|backend| backend.num_threads());
/// assert_eq!(threads, 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CpuExecutionContext {
    engine: Mutex<EngineState>,
    compiler: Mutex<GraphCompiler>,
    eager: OnceLock<Result<Arc<EagerRuntime>, CpuExecutionContextError>>,
}

impl std::fmt::Debug for CpuExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuExecutionContext")
            .field("graph_initialized", &self.engine_initialized())
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
            engine: Mutex::new(EngineState::new(backend)),
            compiler: Mutex::new(GraphCompiler::new()),
            eager: OnceLock::new(),
        }
    }

    fn engine_initialized(&self) -> bool {
        self.engine
            .try_lock()
            .is_ok_and(|state| state.runtime.is_some())
    }

    /// Lock the engine state, recovering it from a panic that poisoned the lock.
    fn lock_engine(&self) -> std::sync::MutexGuard<'_, EngineState> {
        match self.engine.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                guard.recover_from_panic();
                guard
            }
        }
    }

    /// Lock the compiler, discarding one a panic left behind.
    ///
    /// A `GraphCompiler` holds planning state that a panic may have left half
    /// written; rebuilding it costs nothing a compile would not pay for the next
    /// graph anyway.
    fn lock_compiler(&self) -> std::sync::MutexGuard<'_, GraphCompiler> {
        match self.compiler.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                *guard = GraphCompiler::new();
                guard
            }
        }
    }

    /// Run a plain tensor operation with this context's engine.
    ///
    /// The closure runs while the context-local engine lock is held. A lock
    /// poisoned by a panic is recovered: a broken engine is replaced by a fresh
    /// one with the same thread count. The session entries recover the same way.
    ///
    /// This borrows the engine for the whole closure, so a session entry from
    /// inside `f` on this thread is rejected with
    /// [`CpuExecutionContextError::Reentered`] instead of waiting for the engine.
    /// Calling `with_backend` from inside a session of this context is the
    /// caller's obligation not to do: this entry cannot report an error, so it
    /// would wait for the engine the session already holds.
    pub fn with_backend<R>(&self, f: impl FnOnce(&mut CpuBackend) -> R) -> R {
        // The guard makes a nested session entry a typed error, not a wait.
        let _guard = CanonicalSessionGuard::enter();
        f(&mut self.lock_engine().backend)
    }

    /// Run `f` in one session of this context's engine.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::Reentered`] when this thread is
    /// already inside a session of this context.
    pub(crate) fn with_session<R>(
        &self,
        f: impl FnOnce(&mut CpuEngine) -> R,
    ) -> Result<R, CpuExecutionContextError> {
        // Checked before any lock: a nested entry would wait for a lock this
        // thread already holds.
        if CanonicalSessionGuard::is_active() {
            return Err(CpuExecutionContextError::Reentered);
        }
        let _guard = CanonicalSessionGuard::enter();
        // A worker runs inline on its own serial engine instead of waiting for a
        // pool it belongs to (tensor4all-rs#830). The fresh engine shares the
        // plan cache, whose entries are immutable prepared plans, so a stage
        // entered per item does not re-plan. It does **not** share the context
        // engine's buffer pool, extension caches, batch policy or allocation
        // domain: an entry from a worker therefore allocates and batches
        // independently, and only while it runs.
        if rayon::current_thread_index().is_some() {
            let plans = self.lock_engine().backend.engine().plan_cache().clone();
            let mut engine = CpuEngine::serial().with_plan_cache(plans);
            return Ok(f(&mut engine));
        }
        let mut state = self.lock_engine();
        Ok(f(state.backend.engine_mut()))
    }

    /// Compile a backend-neutral traced graph using this context's compiler cache.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::Graph`] when graph compilation fails.
    pub fn compile_graph(
        &self,
        graph: &TracedGraph,
    ) -> Result<CompiledGraph, CpuExecutionContextError> {
        self.lock_compiler()
            .compile_traced_graph(graph)
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
        let reads = inputs
            .iter()
            .map(|input| TensorRead::from_tensor(input))
            .collect::<Vec<_>>();
        self.run_graph_reads(graph, &reads)
    }

    /// Execute a compiled graph on borrowed tensor views.
    ///
    /// The read form is what the bridge uses: an operand that is already a
    /// borrowed view stays borrowed instead of being copied into an owner first.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when runtime initialization,
    /// preparation, or execution fails.
    pub(crate) fn run_graph_reads(
        &self,
        graph: &CompiledGraph,
        inputs: &[TensorRead<'_>],
    ) -> Result<Vec<Tensor>, CpuExecutionContextError> {
        let mut state = self.lock_engine();
        let EngineState { backend, runtime } = &mut *state;
        let runtime = match runtime {
            Some(runtime) => &*runtime,
            None => &*runtime.insert(build_graph_runtime(backend)?),
        };
        let target = backend.engine().target();
        let prepared = runtime
            .prepare(graph, inputs, std::slice::from_ref(&target))
            .map_err(|source| CpuExecutionContextError::graph("preparation", source))?;
        let outputs = {
            let mut engines: [&mut dyn ExecutionEngine; 1] = [backend.engine_mut()];
            runtime
                .run(&mut engines, &prepared, inputs)
                .map_err(|source| CpuExecutionContextError::graph("execution", source))?
        };
        outputs
            .into_iter()
            .map(|output| {
                output
                    .into_tensor(backend.engine_mut())
                    .map_err(|source| CpuExecutionContextError::graph("output", source))
            })
            .collect()
    }

    /// Return this context's eager reverse-AD runtime.
    ///
    /// The runtime is engine-independent: it is built once from an
    /// [`AdContext`] with the linalg AD rules and the einsum/linalg extension
    /// modules installed, and every call takes the caller's engine
    /// ([`EagerRuntime::with_eager_session`]). Repeated calls return the same
    /// runtime and therefore the same eager compilation cache.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when linalg AD-rule or eager-runtime
    /// registration fails.
    pub fn eager_runtime(&self) -> Result<Arc<EagerRuntime>, CpuExecutionContextError> {
        self.eager
            .get_or_init(build_eager_runtime)
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
        let mut state = self.lock_engine();
        let runtime = state.runtime()?;
        runtime
            .cache_stats()
            .map_err(|source| CpuExecutionContextError::graph("cache statistics", source))
    }

    /// Return retained-buffer statistics for this context's engine.
    ///
    /// `buffers` counts the retained allocations and `capacity_bytes` their
    /// total element capacity. A context whose pool has just been released by
    /// [`Self::reset_graph_buffer_pool`] reports zero.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when the engine lock cannot be
    /// recovered.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::program::{CoreSemanticOp, ProgramInputSpec};
    /// use tenferro::{DType, Tensor, TraceContext};
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    ///
    /// let mut trace = TraceContext::new();
    /// let input = trace.input(ProgramInputSpec::new(DType::F64, [2_usize.into()]))?;
    /// let output = trace.add_op(CoreSemanticOp::Neg, &[input])?[0];
    /// let graph = trace.finish(&[output])?;
    /// let compiled = context.compile_graph(&graph)?;
    /// let values = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0])?;
    /// let output = context.run_graph(&compiled, &[&values])?;
    /// assert_eq!(output[0].as_slice::<f64>()?, &[-1.0, 2.0]);
    ///
    /// // Releasing the pool empties it.
    /// context.reset_graph_buffer_pool()?;
    /// let after = context.graph_buffer_pool_stats()?;
    /// assert_eq!(after.buffers, 0);
    /// assert_eq!(after.capacity_bytes, 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn graph_buffer_pool_stats(&self) -> Result<BufferPoolStats, CpuExecutionContextError> {
        Ok(self.lock_engine().backend.engine().buffer_pool_stats())
    }

    /// Release retained buffers owned by this context's engine.
    ///
    /// The graph runtime, its compiled plans and its prepared-plan cache are
    /// kept, so callers can keep running previously compiled graphs afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when the engine lock cannot be
    /// recovered.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// context.reset_graph_buffer_pool()?;
    ///
    /// // The pool retains nothing afterwards, and the runtime stays usable.
    /// let stats = context.graph_buffer_pool_stats()?;
    /// assert_eq!(stats.buffers, 0);
    /// assert_eq!(stats.capacity_bytes, 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn reset_graph_buffer_pool(&self) -> Result<(), CpuExecutionContextError> {
        self.lock_engine().backend.engine_mut().reset_buffer_pool();
        Ok(())
    }

    /// Recreate this context's graph runtime and release its prepared caches.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when the engine lock cannot be
    /// recovered.
    pub fn reset_graph_runtime(&self) -> Result<(), CpuExecutionContextError> {
        {
            let mut state = self.lock_engine();
            state.runtime = None;
            state.backend.engine_mut().reset_buffer_pool();
        }
        *self.lock_compiler() = GraphCompiler::new();
        Ok(())
    }
}

fn build_graph_runtime(backend: &CpuBackend) -> Result<Runtime, CpuExecutionContextError> {
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            tenferro_cpu::runtime_engine_registration(backend.engine()).map_err(|source| {
                CpuExecutionContextError::initialization("graph CPU engine", source)
            })?,
        )
        .map_err(|source| CpuExecutionContextError::initialization("graph CPU engine", source))?;
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CpuEngine>(
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

fn build_eager_runtime() -> Result<Arc<EagerRuntime>, CpuExecutionContextError> {
    let ad_context = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().map_err(|source| {
            CpuExecutionContextError::initialization("linalg AD rules", source)
        })?)
        .map_err(|source| CpuExecutionContextError::initialization("linalg AD rules", source))?
        .build()
        .map_err(|source| CpuExecutionContextError::initialization("AD context", source))?;
    let runtime = EagerRuntime::with_ad_context(&ad_context)
        .map_err(|source| CpuExecutionContextError::initialization("eager runtime", source))?;
    // [AI Supplied] Install the built-in extension modules before publishing
    // the shared context. Lazy first use reconfigures the runtime and advances
    // its epoch; doing that after tensors have prepared AD derivatives can
    // invalidate those prepared programs under parallel first use.
    let engine_id = tenferro_cpu::runtime_engine_id()
        .map_err(|source| CpuExecutionContextError::initialization("CPU runtime engine", source))?;
    let einsum_module = tenferro_einsum::extension_module::<CpuEngine>(engine_id.clone())
        .map_err(|source| CpuExecutionContextError::initialization("einsum extension", source))?;
    runtime
        .install_extension_module(einsum_module)
        .map_err(|source| CpuExecutionContextError::initialization("einsum runtime", source))?;
    let linalg_module = tenferro_linalg::extension_module::<CpuEngine>(engine_id)
        .map_err(|source| CpuExecutionContextError::initialization("linalg extension", source))?;
    runtime
        .install_extension_module(linalg_module)
        .map_err(|source| CpuExecutionContextError::initialization("linalg runtime", source))?;
    Ok(runtime)
}

#[cfg(feature = "global-defaults")]
mod defaults {
    use super::*;

    static DEFAULT_CONTEXT: OnceLock<Arc<CpuExecutionContext>> = OnceLock::new();

    #[cfg(test)]
    thread_local! {
        static FORCE_EAGER_CONTEXT_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Borrow the process-global CPU execution context.
    ///
    /// New code must take a caller-owned context; this accessor exists for the
    /// process-global convenience path and for callers that map the session-entry
    /// rejection into their own error type.
    pub(crate) fn default_context() -> &'static Arc<CpuExecutionContext> {
        DEFAULT_CONTEXT
            .get_or_init(|| Arc::new(CpuExecutionContext::from_backend(CpuBackend::new())))
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

    /// Run `f` in one session of the process-global context's engine.
    ///
    /// The entry rejection stays inside tenferro's error type: it is reported as
    /// a `tenferro_tensor::Error` classified as runtime state, so callers keep
    /// one error type for session work.
    ///
    /// # Errors
    ///
    /// Returns the tenferro diagnostic when the entry is rejected or when `f`
    /// fails.
    pub(crate) fn with_default_session<T: Send>(
        f: impl FnOnce(&mut CpuEngine) -> Result<T, tenferro_tensor::Error> + Send,
    ) -> Result<T, tenferro_tensor::Error> {
        default_context()
            .with_session(f)
            .map_err(|source| {
                tenferro_tensor::Error::runtime_state_source("canonical session entry", source)
            })
            .and_then(std::convert::identity)
    }

    /// Run `f` against the process-global graph compiler.
    ///
    /// The compiler lock is released before the caller runs the compiled graph,
    /// so a compile never waits for a context engine that is busy running one.
    pub(crate) fn with_default_graph_compiler<R>(f: impl FnOnce(&mut GraphCompiler) -> R) -> R {
        f(&mut default_context().lock_compiler())
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
    with_default_graph_compiler, with_default_session,
};

#[cfg(test)]
mod tests;
