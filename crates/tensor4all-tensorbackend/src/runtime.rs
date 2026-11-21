//! The engine-free shared runtime (tensor4all-rs#1393, stage A2 of #1376).
//!
//! A [`Runtime`] holds what was prepared: the eager AD runtime, the graph
//! compiler and the compiled-program cache. It holds no engine and no engine
//! lock. Every computation borrows the caller's engine as
//! `exec: &mut dyn ExecutionEngine` for one call, so several engines can share
//! one runtime and the values it owns.

use std::sync::{Arc, Mutex, MutexGuard};

use tenferro::{
    CompiledGraph, EngineId, EngineTarget, ExecutionEngine, GraphCompiler, Runtime as GraphRuntime,
    RuntimeCacheStats, Tensor, TensorRead, TracedGraph,
};
use tenferro_ad::{ContextId, EagerRuntime, EagerSession, ErrorPhase};
use tenferro_cpu::CpuEngine;

/// Failure building a [`Runtime`] or running work through it.
///
/// # Examples
///
/// ```
/// use std::error::Error;
/// use std::sync::Arc;
/// use tensor4all_tensorbackend::RuntimeError;
///
/// let error = RuntimeError::Graph {
///     stage: "execution",
///     source: Arc::new(std::io::Error::other("kernel failed")),
/// };
/// assert!(error.source().is_some());
/// assert!(error.to_string().contains("execution"));
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum RuntimeError {
    /// A runtime component (AD rules, eager runtime, graph runtime or an
    /// extension module) could not be initialized.
    #[error("failed to initialize {component}: {source}")]
    Initialization {
        /// Runtime component being initialized.
        component: &'static str,
        /// Original tenferro diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// Graph compilation, preparation or execution failed.
    #[error("graph {stage} failed: {source}")]
    Graph {
        /// Graph stage that failed.
        stage: &'static str,
        /// Original tenferro diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// The engine is not one this runtime has extension modules for.
    ///
    /// A runtime serves every engine whose target matches the one it was built
    /// for: for [`Runtime::new`], any [`CpuEngine`] with the canonical runtime
    /// id and no allocation domain. An engine with another runtime id, another
    /// allocation domain or another engine type is refused here, before any
    /// preparation, instead of being registered lazily.
    #[error("engine `{engine_id}` is not supported by this runtime: {reason}")]
    UnsupportedEngine {
        /// The runtime id the refused engine reports.
        engine_id: String,
        /// Which part of the engine's target differs.
        reason: &'static str,
    },
}

impl RuntimeError {
    pub(crate) fn initialization(
        component: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Initialization {
            component,
            source: Arc::new(source),
        }
    }

    pub(crate) fn graph(
        stage: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Graph {
            stage,
            source: Arc::new(source),
        }
    }
}

/// The caller-owned runtime that several engines share.
///
/// A `Runtime` holds the eager AD runtime (AD rules, eager values and their
/// compiled derivative programs), the graph compiler and the compiled-program
/// cache. It holds **no engine and no engine lock**: every call borrows the
/// caller's engine as `exec: &mut dyn ExecutionEngine`, and the borrow is the
/// exclusivity. Cloning is cheap and every clone is the same runtime, so the
/// caller shares it across threads and tasks by clone.
///
/// Values belong to the runtime, not to the engine that computed them: a value
/// produced on one [`CpuEngine`] runs on any other `CpuEngine` that this runtime
/// supports (same runtime id, no allocation domain), whatever its thread count.
/// Nothing in tensor4all chooses an engine for the caller.
///
/// This is the engine-free half of what
/// [`CpuExecutionContext`](crate::CpuExecutionContext) bundles with an engine
/// lock; the contexts are now built on a private `Runtime` each.
///
/// # Examples
///
/// Two engines with different thread counts share one runtime:
///
/// ```
/// use tenferro_ad::EagerTensor;
/// use tensor4all_tensorbackend::{CpuBackend, CpuEngine, Runtime};
///
/// let rt = Runtime::new()?;
/// let mut serial = CpuEngine::serial();
/// let mut pooled = CpuBackend::with_threads(2)?.into_engine();
///
/// let x = EagerTensor::from_tensor_in(
///     tenferro::Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?,
///     rt.eager_runtime().clone(),
/// )?;
/// // Computed on the serial engine ...
/// let y = rt.with_eager(&mut serial, |session| session.add(&x, &x))?;
/// // ... and consumed on the pooled one.
/// let z = rt.with_eager(&mut pooled, |session| session.mul(&y, &x))?;
/// assert_eq!(z.value()?.as_slice::<f64>()?, &[2.0, 8.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

struct RuntimeInner {
    eager: Arc<EagerRuntime>,
    /// The graph path, or `None` for an engine kind with no graph path (CUDA).
    graph: Option<GraphPath>,
    /// Held only while compiling.
    compiler: Mutex<GraphCompiler>,
    /// The one engine target this runtime has extension modules for.
    target: EngineTarget,
}

/// A CPU graph runtime together with the target it was registered for, so a
/// reset can always rebuild it.
struct GraphPath {
    cpu: CpuTarget,
    /// Held only to clone or replace the handle, never while a graph runs.
    runtime: Mutex<GraphRuntime>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("id", &self.id())
            .field("engine_id", &self.inner.target.engine_id().as_str())
            .finish_non_exhaustive()
    }
}

impl Runtime {
    /// Build a runtime for CPU engines with the canonical runtime id.
    ///
    /// The linalg AD rules and the einsum/linalg extension modules are
    /// installed, and the CPU engine kind is registered with both the eager and
    /// the graph runtime, all here. A first use by any engine therefore never
    /// reconfigures the runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Runtime;
    ///
    /// let rt = Runtime::new()?;
    /// let shared = rt.clone();
    /// assert_eq!(shared.id(), rt.id());
    /// assert_ne!(Runtime::new()?.id(), rt.id());
    /// # Ok::<(), tensor4all_tensorbackend::RuntimeError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Initialization`] when an AD rule, an extension
    /// module or the CPU engine registration cannot be installed.
    pub fn new() -> Result<Self, RuntimeError> {
        Self::for_cpu_engine(&CpuTarget::of(&CpuEngine::serial()))
    }

    /// Build a runtime for CPU engines with `cpu`'s runtime id and allocation
    /// domain.
    ///
    /// Used by [`CpuExecutionContext`](crate::CpuExecutionContext), whose
    /// engine may carry a non-canonical runtime id or an allocation domain.
    pub(crate) fn for_cpu_engine(cpu: &CpuTarget) -> Result<Self, RuntimeError> {
        let mut probe = cpu.probe();
        let engine_id = ExecutionEngine::target(&probe).engine_id().clone();
        let target = ExecutionEngine::target(&probe);
        let eager = build_cpu_eager_runtime(&engine_id)?;
        // Register the engine kind now: a lazy registration on first use would
        // reconfigure the runtime and advance its epoch under prepared AD
        // programs (the hazard of `build_cpu_eager_runtime`'s module install).
        eager
            .with_eager_session(&mut probe, |_| Ok::<(), tenferro_ad::Error>(()))
            .map_err(|source| RuntimeError::initialization("eager CPU engine", source))?;
        let graph = build_cpu_graph_runtime(&probe, engine_id)?;
        let graph = GraphPath {
            cpu: cpu.clone(),
            runtime: Mutex::new(graph),
        };
        Ok(Self::from_parts(eager, Some(graph), target))
    }

    /// Build a runtime for one CUDA engine.
    ///
    /// The runtime holds the linalg AD rules and the CUDA einsum/linalg
    /// extension modules, and registers `engine`'s kind here, so a first use
    /// never reconfigures it. It serves engines with `engine`'s target only
    /// (the same allocation domain), and it has no graph path:
    /// [`Self::run_graph`] refuses with [`RuntimeError::UnsupportedEngine`].
    /// CPU engines are refused too; whether one runtime can serve both engine
    /// kinds is open in tensor4all-rs#1376.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{Runtime, RuntimeError};
    ///
    /// // Opening a CUDA engine needs a device, so only the signature is checked here.
    /// let build: fn(&mut tenferro_gpu::cuda::CudaEngine) -> Result<Runtime, RuntimeError> =
    ///     Runtime::for_cuda_engine;
    /// assert!(std::mem::size_of_val(&build) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Initialization`] when an AD rule, an extension
    /// module or the engine registration cannot be installed.
    #[cfg(feature = "tenferro-cuda")]
    pub fn for_cuda_engine(
        engine: &mut tenferro_gpu::cuda::CudaEngine,
    ) -> Result<Self, RuntimeError> {
        let target = ExecutionEngine::target(&*engine);
        let eager = build_cuda_eager_runtime(target.engine_id().clone())?;
        eager
            .with_eager_session(engine, |_| Ok::<(), tenferro_ad::Error>(()))
            .map_err(|source| RuntimeError::initialization("eager CUDA engine", source))?;
        Ok(Self::from_parts(eager, None, target))
    }

    /// Wrap an already built eager runtime, and optionally a graph path, for
    /// engines with `target`.
    fn from_parts(
        eager: Arc<EagerRuntime>,
        graph: Option<GraphPath>,
        target: EngineTarget,
    ) -> Self {
        Self {
            inner: Arc::new(RuntimeInner {
                eager,
                graph,
                compiler: Mutex::new(GraphCompiler::new()),
                target,
            }),
        }
    }

    /// Wrap an already built eager runtime with no graph path, for engines
    /// with `target` (the CUDA context's runtime).
    #[cfg(feature = "tenferro-cuda")]
    pub(crate) fn eager_only(eager: Arc<EagerRuntime>, target: EngineTarget) -> Self {
        Self::from_parts(eager, None, target)
    }

    /// Return this runtime's identity.
    ///
    /// Two handles are the same runtime exactly when their ids are equal. The
    /// id is the id of the eager runtime inside, which is what every value of
    /// this runtime carries
    /// ([`EagerTensor::ctx_id`](tenferro_ad::EagerTensor::ctx_id)).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Runtime;
    ///
    /// let rt = Runtime::new()?;
    /// assert_eq!(rt.id(), rt.eager_runtime().id());
    /// # Ok::<(), tensor4all_tensorbackend::RuntimeError>(())
    /// ```
    pub fn id(&self) -> ContextId {
        self.inner.eager.id()
    }

    /// Borrow the eager runtime that values of this runtime carry.
    ///
    /// Pass it to a value constructor (for example
    /// [`EagerTensor::from_tensor_in`](tenferro_ad::EagerTensor::from_tensor_in))
    /// to create a value in this runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerTensor;
    /// use tensor4all_tensorbackend::Runtime;
    ///
    /// let rt = Runtime::new()?;
    /// let x = EagerTensor::from_tensor_in(
    ///     tenferro::Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?,
    ///     rt.eager_runtime().clone(),
    /// )?;
    /// assert_eq!(x.ctx_id(), rt.id());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn eager_runtime(&self) -> &Arc<EagerRuntime> {
        &self.inner.eager
    }

    /// Check that this runtime can run work on `exec`.
    ///
    /// An engine is supported when its target (runtime id, engine type and
    /// allocation domain) equals the one this runtime was built for.
    /// [`Self::with_eager`] and [`Self::run_graph`] run this check first; call
    /// it directly to obtain the typed refusal before starting other work.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CpuEngine, Runtime, RuntimeError};
    ///
    /// let rt = Runtime::new()?;
    /// rt.check_engine(&CpuEngine::serial())?;
    ///
    /// let custom = tenferro::EngineId::new("example.cpu.v1").unwrap();
    /// let other = CpuEngine::serial().with_runtime_engine_id(custom);
    /// assert!(matches!(
    ///     rt.check_engine(&other),
    ///     Err(RuntimeError::UnsupportedEngine { .. })
    /// ));
    /// # Ok::<(), RuntimeError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::UnsupportedEngine`] when the engine's runtime
    /// id, engine type or allocation domain differs from this runtime's.
    pub fn check_engine(&self, exec: &dyn ExecutionEngine) -> Result<EngineTarget, RuntimeError> {
        let target = exec.target();
        let expected = &self.inner.target;
        let reason = if target.engine_id() != expected.engine_id() {
            "it reports a different runtime engine id"
        } else if target.context_identity() != expected.context_identity() {
            "it is a different engine type"
        } else if target.allocation_domain() != expected.allocation_domain() {
            "it has a different allocation domain"
        } else {
            return Ok(target);
        };
        Err(RuntimeError::UnsupportedEngine {
            engine_id: target.engine_id().as_str().to_owned(),
            reason,
        })
    }

    /// Run one eager session of this runtime on the caller's engine.
    ///
    /// `f` receives the session; every eager operation inside it runs on
    /// `exec`, and its results belong to this runtime. There is no lock, no
    /// re-entry guard and no substitute engine: `exec` is borrowed for the
    /// whole of `f`. Values of this runtime made on any supported engine can
    /// be combined inside the session.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerTensor;
    /// use tensor4all_tensorbackend::{CpuEngine, Runtime};
    ///
    /// let rt = Runtime::new()?;
    /// let x = EagerTensor::from_tensor_in(
    ///     tenferro::Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0])?,
    ///     rt.eager_runtime().clone(),
    /// )?;
    /// let y = rt.with_eager(&mut CpuEngine::serial(), |session| session.neg(&x))?;
    /// assert_eq!(y.value()?.as_slice::<f64>()?, &[-1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `f`'s own error unchanged. When this runtime does not support
    /// `exec` (the condition [`RuntimeError::UnsupportedEngine`] describes for
    /// [`Self::check_engine`]), returns `E::from` a [`tenferro_ad::Error`] with
    /// the unsupported-operation kind, before `f` runs.
    pub fn with_eager<T, E: From<tenferro_ad::Error>>(
        &self,
        exec: &mut dyn ExecutionEngine,
        f: impl FnOnce(&mut EagerSession<'_>) -> Result<T, E>,
    ) -> Result<T, E> {
        if let Err(refusal) = self.check_engine(exec) {
            return Err(E::from(tenferro_ad::Error::unsupported(
                "Runtime::with_eager",
                ErrorPhase::Execution,
                refusal.to_string(),
            )));
        }
        self.inner.eager.with_eager_session(exec, f)
    }

    /// Compile a backend-neutral traced graph with this runtime's compiler.
    ///
    /// The compiler lock is held only while compiling, so a compile never
    /// waits for a graph that is running.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::program::{CoreSemanticOp, ProgramInputSpec};
    /// use tenferro::{DType, TraceContext};
    /// use tensor4all_tensorbackend::Runtime;
    ///
    /// let rt = Runtime::new()?;
    /// let trace = TraceContext::new();
    /// let input = trace.input(ProgramInputSpec::new(DType::F64, [2_usize.into()]))?;
    /// let output = trace.add_op(CoreSemanticOp::Neg, &[input])?[0];
    /// let compiled = rt.compile_graph(&trace.finish(&[output])?)?;
    ///
    /// let values = tenferro::Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 0.5])?;
    /// let mut exec = tensor4all_tensorbackend::CpuEngine::serial();
    /// let out = rt.run_graph(&compiled, &[&values], &mut exec)?;
    /// assert_eq!(out[0].as_slice::<f64>()?, &[-3.0, -0.5]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Graph`] when compilation fails.
    pub fn compile_graph(&self, graph: &TracedGraph) -> Result<CompiledGraph, RuntimeError> {
        self.lock_compiler()
            .compile_traced_graph(graph)
            .map_err(|source| RuntimeError::graph("compilation", source))
    }

    /// Run a compiled graph on the caller's engine.
    ///
    /// The program is prepared for `exec`'s target in this runtime's
    /// compiled-program cache, so every supported engine reuses the same
    /// prepared plans. The engine comes last, after the operands.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::program::{CoreSemanticOp, ProgramInputSpec};
    /// use tenferro::{DType, Tensor, TraceContext};
    /// use tensor4all_tensorbackend::{CpuBackend, CpuEngine, Runtime};
    ///
    /// let rt = Runtime::new()?;
    /// let trace = TraceContext::new();
    /// let input = trace.input(ProgramInputSpec::new(DType::F64, [2_usize.into()]))?;
    /// let output = trace.add_op(CoreSemanticOp::Neg, &[input])?[0];
    /// let compiled = rt.compile_graph(&trace.finish(&[output])?)?;
    ///
    /// let values = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0])?;
    /// let mut serial = CpuEngine::serial();
    /// let mut pooled = CpuBackend::with_threads(2)?.into_engine();
    /// let a = rt.run_graph(&compiled, &[&values], &mut serial)?;
    /// let b = rt.run_graph(&compiled, &[&values], &mut pooled)?;
    /// assert_eq!(a[0].as_slice::<f64>()?, &[-1.0, 2.0]);
    /// assert_eq!(b[0].as_slice::<f64>()?, a[0].as_slice::<f64>()?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::UnsupportedEngine`] when this runtime does not
    /// support `exec` or has no graph path for it, and [`RuntimeError::Graph`]
    /// when preparation or execution fails.
    pub fn run_graph(
        &self,
        graph: &CompiledGraph,
        inputs: &[&Tensor],
        exec: &mut dyn ExecutionEngine,
    ) -> Result<Vec<Tensor>, RuntimeError> {
        let reads = inputs
            .iter()
            .map(|input| TensorRead::from_tensor(input))
            .collect::<Vec<_>>();
        self.run_graph_reads(graph, &reads, exec)
    }

    /// [`Self::run_graph`] on borrowed tensor views.
    pub(crate) fn run_graph_reads(
        &self,
        graph: &CompiledGraph,
        inputs: &[TensorRead<'_>],
        exec: &mut dyn ExecutionEngine,
    ) -> Result<Vec<Tensor>, RuntimeError> {
        let target = self.check_engine(exec)?;
        let Some(runtime) = self.graph_runtime() else {
            return Err(RuntimeError::UnsupportedEngine {
                engine_id: target.engine_id().as_str().to_owned(),
                reason: "this runtime has no graph path for it",
            });
        };
        let prepared = runtime
            .prepare(graph, inputs, std::slice::from_ref(&target))
            .map_err(|source| RuntimeError::graph("preparation", source))?;
        let outputs = {
            let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut *exec];
            runtime
                .run(&mut engines, &prepared, inputs)
                .map_err(|source| RuntimeError::graph("execution", source))?
        };
        outputs
            .into_iter()
            .map(|output| {
                output
                    .into_tensor(&mut *exec)
                    .map_err(|source| RuntimeError::graph("output", source))
            })
            .collect()
    }

    /// Return statistics for this runtime's compiled-program caches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Runtime;
    ///
    /// let rt = Runtime::new()?;
    /// assert_eq!(rt.graph_cache_stats()?.prepared_plans.entries, 0);
    /// # Ok::<(), tensor4all_tensorbackend::RuntimeError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::UnsupportedEngine`] when this runtime has no
    /// graph path, and [`RuntimeError::Graph`] when the statistics query fails.
    pub fn graph_cache_stats(&self) -> Result<RuntimeCacheStats, RuntimeError> {
        let Some(runtime) = self.graph_runtime() else {
            return Err(RuntimeError::UnsupportedEngine {
                engine_id: self.inner.target.engine_id().as_str().to_owned(),
                reason: "this runtime has no graph path",
            });
        };
        runtime
            .cache_stats()
            .map_err(|source| RuntimeError::graph("cache statistics", source))
    }

    /// Lock the compiler, discarding one a panic left behind.
    ///
    /// A `GraphCompiler` holds planning state that a panic may have left half
    /// updated, so a poisoned compiler is replaced by a fresh one.
    pub(crate) fn lock_compiler(&self) -> MutexGuard<'_, GraphCompiler> {
        match self.inner.compiler.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                *guard = GraphCompiler::new();
                self.inner.compiler.clear_poison();
                guard
            }
        }
    }

    /// Replace the graph runtime and the compiler with fresh ones.
    ///
    /// The eager runtime, and with it every value of this runtime, is kept.
    pub(crate) fn reset_graph(&self) -> Result<(), RuntimeError> {
        if let Some(graph) = &self.inner.graph {
            let probe = graph.cpu.probe();
            let engine_id = ExecutionEngine::target(&probe).engine_id().clone();
            let fresh = build_cpu_graph_runtime(&probe, engine_id)?;
            *lock_ignoring_poison(&graph.runtime) = fresh;
        }
        *self.lock_compiler() = GraphCompiler::new();
        Ok(())
    }

    fn graph_runtime(&self) -> Option<GraphRuntime> {
        self.inner
            .graph
            .as_ref()
            .map(|graph| lock_ignoring_poison(&graph.runtime).clone())
    }
}

/// The part of a CPU engine's target that registrations read: its runtime id
/// and its allocation domain.
#[derive(Clone)]
pub(crate) struct CpuTarget {
    engine_id: EngineId,
    domain: Option<Arc<dyn tenferro_tensor::SharedTensorAllocationDomain>>,
}

impl CpuTarget {
    pub(crate) fn of(engine: &CpuEngine) -> Self {
        Self {
            engine_id: ExecutionEngine::target(engine).engine_id().clone(),
            domain: engine.shared_allocation_domain().cloned(),
        }
    }

    /// A serial engine with this target; it stands for every CPU engine with
    /// the same target.
    fn probe(&self) -> CpuEngine {
        let probe = CpuEngine::serial().with_runtime_engine_id(self.engine_id.clone());
        match &self.domain {
            Some(domain) => probe.with_allocation_domain(Arc::clone(domain)),
            None => probe,
        }
    }
}

/// The graph handle is only cloned or replaced under its lock, so a poisoned
/// lock still holds a consistent handle.
fn lock_ignoring_poison<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|poisoned| {
        lock.clear_poison();
        poisoned.into_inner()
    })
}

/// Build the graph runtime for CPU engines under `engine_id`.
///
/// The registration reads only the engine's id and allocation domain, so the
/// serial probe stands for every CPU engine with that target.
fn build_cpu_graph_runtime(
    probe: &CpuEngine,
    engine_id: EngineId,
) -> Result<GraphRuntime, RuntimeError> {
    let mut builder = GraphRuntime::builder();
    builder
        .register_engine(
            tenferro_cpu::runtime_engine_registration_with_id(probe, engine_id.clone())
                .map_err(|source| RuntimeError::initialization("graph CPU engine", source))?,
        )
        .map_err(|source| RuntimeError::initialization("graph CPU engine", source))?;
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CpuEngine>(engine_id)
                .map_err(|source| RuntimeError::initialization("einsum extension", source))?,
        )
        .map_err(|source| RuntimeError::initialization("einsum extension", source))?;
    builder
        .build()
        .map_err(|source| RuntimeError::initialization("graph runtime", source))
}

/// Build the eager runtime for CPU engines under `engine_id`.
fn build_cpu_eager_runtime(engine_id: &EngineId) -> Result<Arc<EagerRuntime>, RuntimeError> {
    let ad_context = crate::context::build_ad_context()
        .map_err(|source| RuntimeError::initialization("linalg AD rules", source))?;
    let runtime = EagerRuntime::with_ad_context(&ad_context)
        .map_err(|source| RuntimeError::initialization("eager runtime", source))?;
    // [AI Supplied] Install the built-in extension modules before publishing
    // the shared runtime. Lazy first use reconfigures the runtime and advances
    // its epoch; doing that after tensors have prepared AD derivatives can
    // invalidate those prepared programs under parallel first use.
    //
    // The modules are installed under the *engine's own* runtime id: an engine
    // built with `CpuEngine::with_runtime_engine_id` would otherwise not find
    // its einsum/linalg extensions.
    let einsum_module = tenferro_einsum::extension_module::<CpuEngine>(engine_id.clone())
        .map_err(|source| RuntimeError::initialization("einsum extension", source))?;
    runtime
        .install_extension_module(einsum_module)
        .map_err(|source| RuntimeError::initialization("einsum runtime", source))?;
    let linalg_module = tenferro_linalg::extension_module::<CpuEngine>(engine_id.clone())
        .map_err(|source| RuntimeError::initialization("linalg extension", source))?;
    runtime
        .install_extension_module(linalg_module)
        .map_err(|source| RuntimeError::initialization("linalg runtime", source))?;
    Ok(runtime)
}

/// Build the eager runtime for CUDA engines under `engine_id`.
#[cfg(feature = "tenferro-cuda")]
pub(crate) fn build_cuda_eager_runtime(
    engine_id: EngineId,
) -> Result<Arc<EagerRuntime>, RuntimeError> {
    use tenferro_gpu::cuda::CudaEngine;

    let ad_context = crate::context::build_ad_context()
        .map_err(|source| RuntimeError::initialization("linalg AD rules", source))?;
    let runtime = EagerRuntime::with_ad_context(&ad_context)
        .map_err(|source| RuntimeError::initialization("eager runtime", source))?;
    let einsum_module = tenferro_einsum::extension_module::<CudaEngine>(engine_id.clone())
        .map_err(|source| RuntimeError::initialization("einsum extension", source))?;
    runtime
        .install_extension_module(einsum_module)
        .map_err(|source| RuntimeError::initialization("einsum runtime", source))?;
    let linalg_module = tenferro_linalg::extension_module::<CudaEngine>(engine_id)
        .map_err(|source| RuntimeError::initialization("linalg extension", source))?;
    runtime
        .install_extension_module(linalg_module)
        .map_err(|source| RuntimeError::initialization("linalg runtime", source))?;
    Ok(runtime)
}

#[cfg(test)]
mod tests;
