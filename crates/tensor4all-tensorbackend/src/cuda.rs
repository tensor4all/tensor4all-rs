//! Explicit CUDA execution context and transfer boundaries.

use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use tenferro::Tensor;
use tenferro_ad::{ContextId, EagerRuntime};
use tenferro_gpu::cuda::{
    cuda_devices, download_pending, CompletionUnproven, CudaBackend, PendingDownload,
    PinnedHostBuffer,
};
use tenferro_tensor::{
    AllocationDomainId, BackendSessionHost, DeviceId, DeviceKind, GpuBackendKind, MemoryKind,
    Placement, TensorRead,
};

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
    /// The CUDA backend rejected entry into its execution session.
    #[error("CUDA {operation} session entry failed: {source}; the backend rejects an entry it cannot admit")]
    SessionEntry {
        /// Operation that observed the rejection.
        operation: &'static str,
        /// Original backend admission diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
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

/// Caller-owned CUDA execution domain for explicit ordinal-0 transfers.
///
/// The context owns one [`CudaBackend`] and lazily creates one shared
/// [`EagerRuntime`] from a clone of that backend. Cloning preserves the
/// provider's runtime and allocation-domain identity. No CPU backend, global
/// CUDA context, or implicit transfer is used.
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
    backend: Mutex<CudaBackend>,
    eager: OnceLock<Result<Arc<EagerRuntime>, CudaExecutionContextError>>,
    device_name: String,
}

impl fmt::Debug for CudaExecutionContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaExecutionContext")
            .field("ordinal", &CUDA_ORDINAL)
            .field("device_name", &self.device_name)
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
/// without waiting - but only while something still owns the runtime, which is why the handle keeps a backend
/// clone. Releasing the last runtime owner is a different event: that synchronizes initialized streams. Dropping
/// never proves the copy; only [`PendingCudaDownload::wait`] hands back filled bytes.
pub struct PendingCudaDownload {
    inner: PendingDownload,
    _keep_alive: CudaBackend,
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
        let backend = CudaBackend::new(device.id()).map_err(|source| {
            CudaExecutionContextError::Initialization {
                component: "backend",
                source: Arc::new(source),
            }
        })?;
        Ok(Self {
            backend: Mutex::new(backend),
            eager: OnceLock::new(),
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
    /// Repeated calls return the same [`Arc`] and preserve the runtime and
    /// allocation identity shared with the context-owned backend.
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
    /// Returns [`CudaExecutionContextError::Initialization`] when eager
    /// runtime registration fails.
    pub fn eager_runtime(&self) -> Result<Arc<EagerRuntime>, CudaExecutionContextError> {
        self.eager
            .get_or_init(|| {
                let backend = self.backend_clone();
                EagerRuntime::with_cuda_backend(backend).map_err(|source| {
                    CudaExecutionContextError::Initialization {
                        component: "eager runtime",
                        source: Arc::new(source),
                    }
                })
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)
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
        self.with_backend_session(|session| session.upload_host_tensor(read))
            .map_err(|source| CudaExecutionContextError::SessionEntry {
                operation: "upload",
                source: Arc::new(source),
            })?
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
        self.with_backend_session(|session| session.download_to_host(read))
            .map_err(|source| CudaExecutionContextError::SessionEntry {
                operation: "download",
                source: Arc::new(source),
            })?
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
        self.with_backend(|backend| PinnedHostBuffer::new(backend.runtime(), bytes))
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
        // Keeps the runtime alive until the handle is gone: pinned buffers hold a weak runtime reference and leak
        // if it cannot be upgraded when they are destroyed.
        let keep_alive = self.with_backend(|backend| backend.clone());
        let inner = self
            .with_backend(|backend| download_pending(backend.runtime(), tensor, buffer.buffer))
            .map_err(|source| CudaExecutionContextError::Transfer {
                operation: "download submit",
                // The rejected package is not `Send`; only its diagnostic can be retained here.
                source: Arc::new(std::io::Error::other(format!("{source:?}"))),
            })?;
        Ok(PendingCudaDownload {
            inner,
            _keep_alive: keep_alive,
        })
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
        let expected = self.with_backend(|backend| backend.runtime().allocation_domain());
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
        self.with_backend(|backend| backend.runtime().synchronize())
            .map_err(|source| CudaExecutionContextError::Synchronization {
                source: Arc::new(source),
            })
    }

    fn with_backend<R>(&self, f: impl FnOnce(&mut CudaBackend) -> R) -> R {
        let mut backend = match self.backend.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        f(&mut backend)
    }

    fn backend_clone(&self) -> CudaBackend {
        self.with_backend(|backend| backend.clone())
    }

    fn with_backend_session<R: Send>(
        &self,
        f: impl FnOnce(&mut dyn tenferro_tensor::BackendSession) -> R + Send,
    ) -> Result<R, tenferro_tensor::SessionEntryError> {
        self.with_backend(|backend| backend.with_backend_session(f))
    }
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
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn source_chain_is_preserved_for_initialization() {
        let error = CudaExecutionContextError::Initialization {
            component: "backend",
            source: Arc::new(std::io::Error::other("driver unavailable")),
        };
        assert!(error.source().is_some());
        assert_eq!(error.source().unwrap().to_string(), "driver unavailable");
    }

    #[test]
    fn metadata_rejections_do_not_need_a_cuda_context() {
        let host = Placement::default();
        assert!(validate_host_placement_metadata(&host, None).is_ok());
        assert!(!is_cuda_ordinal_zero_placement(&host));

        let cuda = Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: CUDA_ORDINAL as usize,
            }),
            cpu_affinity: None,
        };
        assert!(is_cuda_ordinal_zero_placement(&cuda));
        assert!(matches!(
            validate_host_placement_metadata(&cuda, None),
            Err(CudaExecutionContextError::UnsupportedInput { .. })
        ));
    }
}
