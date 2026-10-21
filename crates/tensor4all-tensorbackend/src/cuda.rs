//! Explicit CUDA execution context and transfer boundaries.

use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use tenferro::{ExecutionEngine, Tensor};
use tenferro_ad::{ContextId, EagerRuntime};
use tenferro_gpu::cuda::{
    cuda_devices, download_pending, upload_pending, CompletionUnproven, CudaEngine,
    PendingDownload, PendingUpload, PinnedHostBuffer,
};
use tenferro_tensor::{
    AllocationDomainId, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement,
    TensorDeviceTransfer, TensorRead,
};

use crate::context::{next_context_id, CanonicalSessionGuard};

/// The only CUDA ordinal selected by [`CudaExecutionContext`].
pub const CUDA_ORDINAL: u32 = 0;

/// Error returned by explicit CUDA context creation, placement validation, and
/// device transfer.
///
/// The original provider, runtime, or tensor diagnostic is retained as the
/// standard error source where one exists. CUDA operations never fall back to
/// a CPU backend or perform an implicit transfer.
///
/// # Remedies
///
/// - Initialization failures: make visible CUDA ordinal 0 and its runtime
///   libraries available, then retry [`CudaExecutionContext::new`].
/// - Placement failures: use [`CudaExecutionContext::upload_cuda`] for a host
///   tensor, or use one context consistently for resident values.
/// - Transfer failures: keep the explicit upload/download boundary and inspect
///   the retained source error.
///
/// # Examples
///
/// ```
/// use std::error::Error;
/// use std::sync::Arc;
/// use tensor4all_tensorbackend::CudaExecutionContextError;
///
/// let error = CudaExecutionContextError::Initialization {
///     component: "CUDA backend",
///     source: Arc::new(std::io::Error::other("driver unavailable")),
/// };
/// assert!(error.source().is_some());
/// assert!(error.to_string().contains("driver unavailable"));
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum CudaExecutionContextError {
    /// CUDA device or eager-runtime initialization failed.
    #[error(
        "failed to initialize CUDA {component}: {source}; make visible CUDA ordinal 0 and retry"
    )]
    Initialization {
        /// Context component that failed to initialize.
        component: &'static str,
        /// Original provider or runtime diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// An explicit upload or download failed.
    #[error("CUDA {operation} transfer failed: {source}; keep the transfer explicit and inspect the source error")]
    Transfer {
        /// Transfer operation that failed.
        operation: &'static str,
        /// Original backend transfer diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// A host/device placement does not satisfy the context contract.
    #[error("CUDA {operation} placement mismatch: expected a dense value on visible CUDA ordinal 0; use explicit upload_cuda/download or one context consistently")]
    PlacementMismatch {
        /// Operation that observed the mismatch.
        operation: &'static str,
    },
    /// The calling thread is already inside a session of this context.
    ///
    /// One context owns one CUDA engine and lends it to one caller at a time. A
    /// session entered from inside another session on the same thread would
    /// wait for a lock that thread already holds, so it is reported as this
    /// typed error before any lock is taken.
    #[error("recursive CUDA execution-context session entry on one thread")]
    Reentered,
    /// A resident allocation belongs to another CUDA allocation domain.
    #[error("CUDA {operation} received a foreign allocation domain; use one CudaExecutionContext for upload, computation, and download")]
    ForeignAllocation {
        /// Operation that observed the foreign allocation.
        operation: &'static str,
    },
    /// An input violates the host-only upload boundary.
    #[error("CUDA {operation} received an unsupported input: {reason}; {remedy}")]
    UnsupportedInput {
        /// Operation that rejected the input.
        operation: &'static str,
        /// Stable reason for rejecting the input.
        reason: &'static str,
        /// Corrective action for the caller.
        remedy: &'static str,
    },
    /// Synchronization of the context-owned CUDA stream failed.
    #[error("CUDA synchronization failed: {source}; retry after checking the CUDA runtime")]
    Synchronization {
        /// Original CUDA synchronization diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

/// The CUDA engine a [`CudaExecutionContext`] owns, plus its session identity.
///
/// The process-wide eager-runtime owner table holds this weakly, so an implicit
/// eager entry on a context-owned runtime can reach the same engine the owning
/// context lends to explicit calls. The engine is never cloned or replaced:
/// opening the same ordinal again would create a different stream and
/// allocation domain.
pub(crate) struct CudaEngineOwner {
    id: u64,
    engine: Mutex<CudaEngine>,
}

impl CudaEngineOwner {
    fn new(engine: CudaEngine) -> Self {
        Self {
            id: next_context_id(),
            engine: Mutex::new(engine),
        }
    }

    /// Lock the engine, recovering a lock poisoned by a panic.
    ///
    /// A panic leaves upstream's engine marked broken; there is no second engine
    /// to substitute, so the broken engine stays and refuses later operations
    /// with its own typed diagnostic.
    fn lock_engine(&self) -> std::sync::MutexGuard<'_, CudaEngine> {
        match self.engine.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Run `f` in one session of this owner's engine.
    pub(crate) fn with_session<R>(
        &self,
        f: impl FnOnce(&mut CudaEngine) -> R,
    ) -> Result<R, CudaExecutionContextError> {
        if CanonicalSessionGuard::is_active(self.id) {
            return Err(CudaExecutionContextError::Reentered);
        }
        let _guard = CanonicalSessionGuard::enter(self.id);
        let mut engine = self.lock_engine();
        Ok(f(&mut engine))
    }

    /// Run `f` in one session of this owner's engine, seen as the engine trait.
    ///
    /// This is the entry used by the eager-runtime owner table. It uses the same
    /// re-entry guard and the same single engine as [`Self::with_session`]; a
    /// Rayon worker is never given a second engine.
    pub(crate) fn with_execution_engine<R>(
        &self,
        f: impl FnOnce(&mut dyn ExecutionEngine) -> R,
    ) -> Result<R, CudaExecutionContextError> {
        self.with_session(|engine| f(engine))
    }
}

/// Caller-owned CUDA execution domain for explicit ordinal-0 transfers.
///
/// The context owns exactly one [`CudaEngine`] and lends it to one caller at a
/// time through [`Self::with_session`]. The eager runtime is engine-independent:
/// it is built once from an AD context with the linalg AD rules and the CUDA
/// einsum/linalg extension modules installed for the context engine's runtime
/// id, and every eager call takes the caller's engine. No CPU backend, second
/// CUDA engine, global CUDA context, or implicit transfer is used.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
/// use std::sync::Arc;
/// use tenferro_ad::EagerRuntime;
/// use tenferro::Tensor;
///
/// let new_context: fn() -> Result<CudaExecutionContext, CudaExecutionContextError> =
///     CudaExecutionContext::new;
/// let upload: fn(&CudaExecutionContext, &Tensor) -> Result<Tensor, CudaExecutionContextError> =
///     CudaExecutionContext::upload_cuda;
/// let eager: fn(&CudaExecutionContext) -> Result<Arc<EagerRuntime>, CudaExecutionContextError> =
///     CudaExecutionContext::eager_runtime;
/// assert_eq!(std::mem::size_of_val(&new_context), std::mem::size_of::<fn() -> Result<CudaExecutionContext, CudaExecutionContextError>>());
/// assert_eq!(std::mem::size_of_val(&upload), std::mem::size_of::<fn(&CudaExecutionContext, &Tensor) -> Result<Tensor, CudaExecutionContextError>>());
/// assert_eq!(std::mem::size_of_val(&eager), std::mem::size_of::<fn(&CudaExecutionContext) -> Result<Arc<EagerRuntime>, CudaExecutionContextError>>());
/// ```
pub struct CudaExecutionContext {
    owner: Arc<CudaEngineOwner>,
    eager: OnceLock<Result<Arc<EagerRuntime>, CudaExecutionContextError>>,
    engine_id: tenferro::EngineId,
    device_name: String,
}

impl fmt::Debug for CudaExecutionContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaExecutionContext")
            .field("ordinal", &CUDA_ORDINAL)
            .field("device_name", &self.device_name)
            .field("engine_id", &self.engine_id)
            .field("eager_initialized", &self.eager.get().is_some())
            .finish_non_exhaustive()
    }
}

/// A caller-owned pinned host buffer for a pending transfer.
pub struct CudaPinnedBuffer {
    buffer: PinnedHostBuffer,
}

impl std::fmt::Debug for CudaPinnedBuffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaPinnedBuffer")
            .field("byte_len", &self.buffer.len())
            .finish_non_exhaustive()
    }
}

impl CudaPinnedBuffer {
    /// Number of bytes this buffer holds.
    pub fn byte_len(&self) -> usize {
        self.buffer.len()
    }

    /// The staged bytes, for reading back what a download filled.
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    /// The staged bytes, mutably, for staging an upload's source.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }
}

/// One submitted device-to-host transfer.
///
/// Dropping the handle moves the transfer into the runtime's retirement queue, where a later query releases it
/// without waiting - provided the copy has completed and the queue can admit it, since the pinned implementation's
/// queue is bounded and a transfer that cannot be admitted is retained rather than released. Upstream's
/// [`PendingDownload`] owns the runtime handle it was submitted on, which keeps that runtime alive for the
/// retirement queue; no second engine is held. Releasing the last runtime owner is a different event: that
/// synchronizes initialized streams. Dropping never proves the copy; only [`PendingCudaDownload::wait`] hands back
/// filled bytes.
pub struct PendingCudaDownload {
    inner: PendingDownload,
}

impl std::fmt::Debug for PendingCudaDownload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PendingCudaDownload")
            .finish_non_exhaustive()
    }
}

impl PendingCudaDownload {
    /// Whether the copy's event has completed.
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Transfer`] when the event query itself fails for a reason other than
    /// "not ready".
    pub fn is_ready(&mut self) -> Result<bool, CudaExecutionContextError> {
        self.inner
            .is_ready()
            .map_err(|source| CudaExecutionContextError::Transfer {
                operation: "download readiness",
                source: Arc::new(source),
            })
    }

    /// Wait for the copy and return the device source plus the filled pinned buffer.
    ///
    /// The device source is handed back rather than consumed here, and the filled buffer is returned raw: turning
    /// those bytes into a host value is the caller's step, so its cost is not hidden behind the wait.
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Transfer`] when the copy reports an unproven completion; the filled
    /// buffer must then be discarded rather than read.
    pub fn wait(self) -> Result<(Tensor, CudaPinnedBuffer), CudaExecutionContextError> {
        let (tensor, buffer) = self.inner.wait().map_err(|source: CompletionUnproven| {
            CudaExecutionContextError::Transfer {
                operation: "download wait",
                source: Arc::new(source),
            }
        })?;
        Ok((tensor, CudaPinnedBuffer { buffer }))
    }
}

/// One submitted host-to-device transfer.
///
/// Dropping the handle moves the transfer into the runtime's retirement queue, where a later query releases it
/// without waiting - provided the copy has completed and the queue can admit it, since the pinned implementation's
/// queue is bounded and a transfer that cannot be admitted is retained rather than released. Upstream's
/// [`PendingUpload`] owns the runtime handle it was submitted on, which keeps that runtime alive for the
/// retirement queue; no second engine is held. Releasing the last runtime owner is a different event: that
/// synchronizes initialized streams. Dropping never proves the copy; only [`PendingCudaUpload::wait`] hands back
/// the written device tensor.
pub struct PendingCudaUpload {
    inner: PendingUpload,
}

impl std::fmt::Debug for PendingCudaUpload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PendingCudaUpload")
            .finish_non_exhaustive()
    }
}

impl PendingCudaUpload {
    /// Whether the copy's event has completed.
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Transfer`] when the event query itself fails for a reason other than
    /// "not ready".
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContextError, PendingCudaUpload};
    /// let is_ready: fn(&mut PendingCudaUpload) -> Result<bool, CudaExecutionContextError> =
    ///     PendingCudaUpload::is_ready;
    /// assert!(std::mem::size_of_val(&is_ready) > 0);
    /// ```
    pub fn is_ready(&mut self) -> Result<bool, CudaExecutionContextError> {
        self.inner
            .is_ready()
            .map_err(|source| CudaExecutionContextError::Transfer {
                operation: "upload readiness",
                source: Arc::new(source),
            })
    }

    /// Wait for the copy and return the written device destination plus the source buffer.
    ///
    /// The device destination is handed back rather than consumed here, and the source buffer is returned raw:
    /// it is the caller's staging storage, not a host value.
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Transfer`] when the copy reports an unproven completion; the device
    /// tensor must then be discarded rather than read.
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContextError, CudaPinnedBuffer, PendingCudaUpload};
    /// let wait: fn(
    ///     PendingCudaUpload,
    /// ) -> Result<(tenferro::Tensor, CudaPinnedBuffer), CudaExecutionContextError> =
    ///     PendingCudaUpload::wait;
    /// assert!(std::mem::size_of_val(&wait) > 0);
    /// ```
    pub fn wait(self) -> Result<(Tensor, CudaPinnedBuffer), CudaExecutionContextError> {
        let (tensor, buffer) = self.inner.wait().map_err(|source: CompletionUnproven| {
            CudaExecutionContextError::Transfer {
                operation: "upload wait",
                source: Arc::new(source),
            }
        })?;
        Ok((tensor, CudaPinnedBuffer { buffer }))
    }
}

impl CudaExecutionContext {
    /// Create a context for visible CUDA ordinal 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError, CUDA_ORDINAL};
    /// let constructor: fn() -> Result<CudaExecutionContext, CudaExecutionContextError> = CudaExecutionContext::new;
    /// assert_eq!(CUDA_ORDINAL, 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Initialization`] when device
    /// discovery or CUDA runtime initialization fails.
    pub fn new() -> Result<Self, CudaExecutionContextError> {
        let device = cuda_devices()
            .map_err(|source| CudaExecutionContextError::Initialization {
                component: "device discovery",
                source: Arc::new(source),
            })?
            .into_iter()
            .find(|device| device.id().ordinal() == CUDA_ORDINAL)
            .ok_or_else(|| CudaExecutionContextError::Initialization {
                component: "device discovery",
                source: Arc::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "visible CUDA ordinal 0 is unavailable",
                )),
            })?;
        let device_name = device.name().to_owned();
        let engine = CudaEngine::open(device.id()).map_err(|source| {
            CudaExecutionContextError::Initialization {
                component: "engine",
                source: Arc::new(source),
            }
        })?;
        let engine_id = engine.runtime_engine_id().clone();
        Ok(Self {
            owner: Arc::new(CudaEngineOwner::new(engine)),
            eager: OnceLock::new(),
            engine_id,
            device_name,
        })
    }

    /// Return the only supported visible CUDA ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CUDA_ORDINAL};
    /// let accessor: fn(&CudaExecutionContext) -> u32 = CudaExecutionContext::ordinal;
    /// assert_eq!(CUDA_ORDINAL, 0);
    /// ```
    pub const fn ordinal(&self) -> u32 {
        CUDA_ORDINAL
    }

    /// Return the discovered name of visible CUDA ordinal 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CudaExecutionContext;
    /// let accessor: for<'a> fn(&'a CudaExecutionContext) -> &'a str = CudaExecutionContext::device_name;
    /// assert!(std::mem::size_of_val(&accessor) > 0);
    /// ```
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Return this context's eager runtime identity, initializing it lazily.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let accessor: fn(&CudaExecutionContext) -> Result<tenferro_ad::ContextId, CudaExecutionContextError> = CudaExecutionContext::context_id;
    /// assert!(std::mem::size_of_val(&accessor) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Initialization`] when the CUDA
    /// eager runtime or its runtime registration cannot be created.
    pub fn context_id(&self) -> Result<ContextId, CudaExecutionContextError> {
        self.eager_runtime().map(|runtime| runtime.id())
    }

    /// Return the context-owned CUDA eager runtime.
    ///
    /// The runtime is engine-independent: it holds the AD rules and the CUDA
    /// einsum/linalg execution modules for this context engine's runtime id, and
    /// every eager call takes the caller's engine. Repeated calls return the
    /// same [`Arc`], so they share the eager compilation cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let accessor: fn(&CudaExecutionContext) -> Result<Arc<tenferro_ad::EagerRuntime>, CudaExecutionContextError> = CudaExecutionContext::eager_runtime;
    /// assert!(std::mem::size_of_val(&accessor) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Initialization`] when AD-rule or
    /// eager-runtime registration fails.
    pub fn eager_runtime(&self) -> Result<Arc<EagerRuntime>, CudaExecutionContextError> {
        let runtime = self
            .eager
            .get_or_init(|| build_eager_runtime(self.engine_id.clone()))
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)?;
        #[cfg(feature = "global-defaults")]
        crate::context::register_eager_runtime_owner(
            &runtime,
            crate::context::RuntimeOwner::Cuda(Arc::downgrade(&self.owner)),
        );
        Ok(runtime)
    }

    /// Run `f` in one session of this context's engine.
    ///
    /// The context lends its single [`CudaEngine`] to one caller at a time for
    /// the whole of `f`. A nested entry on the same thread is reported as
    /// [`CudaExecutionContextError::Reentered`] instead of waiting for the
    /// engine the session already holds.
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Reentered`] when this thread is
    /// already inside a session of this context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// use tenferro_gpu::cuda::CudaEngine;
    ///
    /// let session: fn(
    ///     &CudaExecutionContext,
    ///     fn(&mut CudaEngine) -> u32,
    /// ) -> Result<u32, CudaExecutionContextError> = CudaExecutionContext::with_session;
    /// assert!(std::mem::size_of_val(&session) > 0);
    /// ```
    pub fn with_session<R>(
        &self,
        f: impl FnOnce(&mut CudaEngine) -> R,
    ) -> Result<R, CudaExecutionContextError> {
        self.owner.with_session(f)
    }

    /// Upload a host tensor explicitly into this CUDA context.
    ///
    /// The source must be host-backed. A resident CUDA or foreign-backend
    /// tensor is rejected rather than copied implicitly.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let upload: fn(&CudaExecutionContext, &tenferro::Tensor) -> Result<tenferro::Tensor, CudaExecutionContextError> = CudaExecutionContext::upload_cuda;
    /// assert!(std::mem::size_of_val(&upload) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::UnsupportedInput`] for a
    /// non-host source, [`CudaExecutionContextError::Transfer`] for a backend
    /// transfer failure, or [`CudaExecutionContextError::PlacementMismatch`]
    /// for invalid placement metadata.
    pub fn upload_cuda(&self, tensor: &Tensor) -> Result<Tensor, CudaExecutionContextError> {
        let read = TensorRead::from_tensor(tensor);
        self.validate_host_placement_metadata(read.placement(), read.allocation_domain())?;
        self.with_session(|engine| engine.upload_host_tensor(read))?
            .map_err(|source| CudaExecutionContextError::Transfer {
                operation: "upload",
                source: Arc::new(source),
            })
    }

    /// Download a tensor resident on this exact CUDA context into host storage.
    ///
    /// This method never accepts host tensors and never downloads from a
    /// different allocation domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let download: fn(&CudaExecutionContext, &tenferro::Tensor) -> Result<tenferro::Tensor, CudaExecutionContextError> = CudaExecutionContext::download;
    /// assert!(std::mem::size_of_val(&download) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::PlacementMismatch`] for host,
    /// non-CUDA, or nonzero-ordinal inputs,
    /// [`CudaExecutionContextError::ForeignAllocation`] for a tensor owned by
    /// another CUDA context, or [`CudaExecutionContextError::Transfer`] when
    /// the explicit download fails.
    pub fn download(&self, tensor: &Tensor) -> Result<Tensor, CudaExecutionContextError> {
        self.validate_cuda_placement(tensor)?;
        let read = TensorRead::from_tensor(tensor);
        self.with_session(|engine| engine.download_to_host(read))?
            .map_err(|source| CudaExecutionContextError::Transfer {
                operation: "download",
                source: Arc::new(source),
            })
    }

    /// Allocate a pinned host buffer of `bytes` bytes.
    ///
    /// The bytes are the caller's: stage what the device expects (column-major, native endianness) through
    /// [`CudaPinnedBuffer::as_mut_slice`], and read a download's result through
    /// [`CudaPinnedBuffer::as_slice`].
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Transfer`] when the pinned allocation fails, and
    /// [`CudaExecutionContextError::PlacementMismatch`] when `bytes` is zero, which the pinned allocator rejects.
    pub fn allocate_pinned(
        &self,
        bytes: usize,
    ) -> Result<CudaPinnedBuffer, CudaExecutionContextError> {
        if bytes == 0 {
            return Err(CudaExecutionContextError::PlacementMismatch {
                operation: "allocate pinned buffer",
            });
        }
        self.with_session(|engine| PinnedHostBuffer::new(engine.runtime(), bytes))?
            .map(|buffer| CudaPinnedBuffer { buffer })
            .map_err(|source| CudaExecutionContextError::Transfer {
                operation: "allocate pinned buffer",
                source: Arc::new(source),
            })
    }

    /// Submit a device-to-host copy without waiting for it.
    ///
    /// Takes the device tensor **by value**: the pinned revision's pending entry consumes its source, and the
    /// value comes back from [`PendingCudaDownload::wait`]. Nothing is copied behind the caller's back to work
    /// around that - a caller that wants to keep its tensor should not use this route.
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::PlacementMismatch`] when the tensor is not resident on this exact
    /// context, and [`CudaExecutionContextError::Transfer`] when the submission is rejected (for example when the
    /// pinned buffer's length does not match). A rejection's recoverable package is **not** retained on this
    /// route - only its diagnostic - because that package is neither `Send` nor `Sync`; the buffer is dropped with
    /// it.
    pub fn download_pending(
        &self,
        tensor: Tensor,
        buffer: CudaPinnedBuffer,
    ) -> Result<PendingCudaDownload, CudaExecutionContextError> {
        self.validate_cuda_placement(&tensor)?;
        // Upstream's pending handle owns the runtime it was submitted on, which keeps that runtime alive for the
        // retirement queue; the engine itself is not cloned.
        let inner = self
            .with_session(|engine| download_pending(engine.runtime(), tensor, buffer.buffer))?
            .map_err(|source| CudaExecutionContextError::Transfer {
                operation: "download submit",
                // The rejected package is not `Send`; only its diagnostic can be retained here.
                source: Arc::new(std::io::Error::other(format!("{source:?}"))),
            })?;
        Ok(PendingCudaDownload { inner })
    }

    /// Submit a host-to-device copy without waiting for it.
    ///
    /// Takes the destination tensor **by value**: the pinned revision's pending entry consumes its destination,
    /// and the value comes back from [`PendingCudaUpload::wait`] together with the source buffer. Nothing is
    /// copied behind the caller's back to work around that - a caller that wants to keep its tensor should not
    /// use this route.
    ///
    /// Stage the bytes the device expects (column-major, native endianness) through
    /// [`CudaPinnedBuffer::as_mut_slice`] before submitting.
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::PlacementMismatch`] when the destination is not resident on this
    /// exact context, and [`CudaExecutionContextError::Transfer`] when the submission is rejected (for example
    /// when the pinned buffer's length does not match the destination). A rejection's recoverable package is
    /// **not** retained on this route - only its diagnostic - because that package is neither `Send` nor `Sync`;
    /// the buffer is dropped with it.
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{
    ///     CudaExecutionContext, CudaExecutionContextError, CudaPinnedBuffer, PendingCudaUpload,
    /// };
    /// let upload: fn(
    ///     &CudaExecutionContext,
    ///     CudaPinnedBuffer,
    ///     tenferro::Tensor,
    /// ) -> Result<PendingCudaUpload, CudaExecutionContextError> = CudaExecutionContext::upload_pending;
    /// assert!(std::mem::size_of_val(&upload) > 0);
    /// ```
    pub fn upload_pending(
        &self,
        source: CudaPinnedBuffer,
        destination: Tensor,
    ) -> Result<PendingCudaUpload, CudaExecutionContextError> {
        self.validate_cuda_placement(&destination)?;
        // Upstream's pending handle owns the runtime it was submitted on, which keeps that runtime alive for the
        // retirement queue; the engine itself is not cloned.
        let inner = self
            .with_session(|engine| upload_pending(engine.runtime(), source.buffer, destination))?
            .map_err(|source| CudaExecutionContextError::Transfer {
                operation: "upload submit",
                // The rejected package is not `Send`; only its diagnostic can be retained here.
                source: Arc::new(std::io::Error::other(format!("{source:?}"))),
            })?;
        Ok(PendingCudaUpload { inner })
    }

    /// Validate an owned tensor's placement without transferring or reading it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let validate: fn(&CudaExecutionContext, &tenferro::Tensor) -> Result<(), CudaExecutionContextError> = CudaExecutionContext::validate_cuda_placement;
    /// assert!(std::mem::size_of_val(&validate) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::PlacementMismatch`] when the tensor
    /// is not CUDA-resident on visible ordinal 0, or
    /// [`CudaExecutionContextError::ForeignAllocation`] when it belongs to a
    /// different CUDA context.
    pub fn validate_cuda_placement(
        &self,
        tensor: &Tensor,
    ) -> Result<(), CudaExecutionContextError> {
        let read = TensorRead::from_tensor(tensor);
        if read.backend_family() != Some("cuda") {
            return Err(CudaExecutionContextError::PlacementMismatch {
                operation: "validate CUDA placement",
            });
        }
        self.validate_cuda_placement_metadata(read.placement(), read.allocation_domain())
    }

    /// Validate placement metadata obtained from a borrowed eager value.
    ///
    /// This is a metadata-only check: it performs no host access and no device
    /// transfer. It exists so higher-level wrappers can validate an
    /// [`EagerRuntime`] value without first materializing an owned [`Tensor`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// use tenferro_tensor::{AllocationDomainId, Placement};
    /// let validate: fn(&CudaExecutionContext, &Placement, Option<AllocationDomainId>) -> Result<(), CudaExecutionContextError> = CudaExecutionContext::validate_cuda_placement_metadata;
    /// assert!(std::mem::size_of_val(&validate) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::PlacementMismatch`] for a
    /// non-CUDA-ordinal-0 placement and
    /// [`CudaExecutionContextError::ForeignAllocation`] for a different
    /// allocation domain.
    pub fn validate_cuda_placement_metadata(
        &self,
        placement: &Placement,
        allocation_domain: Option<AllocationDomainId>,
    ) -> Result<(), CudaExecutionContextError> {
        if !is_cuda_ordinal_zero_placement(placement) {
            return Err(CudaExecutionContextError::PlacementMismatch {
                operation: "validate CUDA placement",
            });
        }
        let expected = self.with_session(|engine| engine.runtime().allocation_domain())?;
        if allocation_domain != Some(expected) {
            return Err(CudaExecutionContextError::ForeignAllocation {
                operation: "validate CUDA placement",
            });
        }
        Ok(())
    }

    /// Validate metadata for a host-only upload source.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// use tenferro_tensor::{AllocationDomainId, Placement};
    /// let validate: fn(&CudaExecutionContext, &Placement, Option<AllocationDomainId>) -> Result<(), CudaExecutionContextError> = CudaExecutionContext::validate_host_placement_metadata;
    /// assert!(std::mem::size_of_val(&validate) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::UnsupportedInput`] when the source
    /// carries device or backend allocation metadata.
    pub fn validate_host_placement_metadata(
        &self,
        placement: &Placement,
        allocation_domain: Option<AllocationDomainId>,
    ) -> Result<(), CudaExecutionContextError> {
        validate_host_placement_metadata(placement, allocation_domain)
    }

    /// Synchronize work submitted through this context's CUDA runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError};
    /// let synchronize: fn(&CudaExecutionContext) -> Result<(), CudaExecutionContextError> = CudaExecutionContext::synchronize;
    /// assert!(std::mem::size_of_val(&synchronize) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaExecutionContextError::Synchronization`] when the CUDA
    /// stream cannot be synchronized.
    pub fn synchronize(&self) -> Result<(), CudaExecutionContextError> {
        self.with_session(|engine| engine.synchronize())?
            .map_err(|source| CudaExecutionContextError::Synchronization {
                source: Arc::new(source),
            })
    }
}

fn build_eager_runtime(
    engine_id: tenferro::EngineId,
) -> Result<Arc<EagerRuntime>, CudaExecutionContextError> {
    let ad_context = crate::context::build_ad_context().map_err(|source| {
        CudaExecutionContextError::Initialization {
            component: "linalg AD rules",
            source: Arc::new(source),
        }
    })?;
    let runtime = EagerRuntime::with_ad_context(&ad_context).map_err(|source| {
        CudaExecutionContextError::Initialization {
            component: "eager runtime",
            source: Arc::new(source),
        }
    })?;
    let einsum_module = tenferro_einsum::extension_module::<CudaEngine>(engine_id.clone())
        .map_err(|source| CudaExecutionContextError::Initialization {
            component: "einsum extension",
            source: Arc::new(source),
        })?;
    runtime
        .install_extension_module(einsum_module)
        .map_err(|source| CudaExecutionContextError::Initialization {
            component: "einsum runtime",
            source: Arc::new(source),
        })?;
    let linalg_module =
        tenferro_linalg::extension_module::<CudaEngine>(engine_id).map_err(|source| {
            CudaExecutionContextError::Initialization {
                component: "linalg extension",
                source: Arc::new(source),
            }
        })?;
    runtime
        .install_extension_module(linalg_module)
        .map_err(|source| CudaExecutionContextError::Initialization {
            component: "linalg runtime",
            source: Arc::new(source),
        })?;
    Ok(runtime)
}

fn validate_host_placement_metadata(
    placement: &Placement,
    allocation_domain: Option<AllocationDomainId>,
) -> Result<(), CudaExecutionContextError> {
    if placement.device.is_some()
        || matches!(placement.memory_kind, MemoryKind::Device)
        || allocation_domain.is_some()
    {
        return Err(CudaExecutionContextError::UnsupportedInput {
            operation: "upload",
            reason: "the source is not host-resident",
            remedy: "download it explicitly or provide a host tensor",
        });
    }
    Ok(())
}

fn is_cuda_ordinal_zero_placement(placement: &Placement) -> bool {
    placement.memory_kind == MemoryKind::Device
        && placement.device.as_ref()
            == Some(&DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: CUDA_ORDINAL as usize,
            })
}

#[cfg(test)]
mod tests;
