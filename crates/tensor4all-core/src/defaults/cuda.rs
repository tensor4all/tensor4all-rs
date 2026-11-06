//! CUDA transfer boundaries for [`IdxTensor`].

use std::sync::Arc;

use tenferro_tensor::{DType, Tensor as NativeTensor};
use tensor4all_tensorbackend::{
    CudaExecutionContext, CudaExecutionContextError, CudaPinnedBuffer, PendingCudaDownload,
    StorageKind,
};

use super::idx_tensor::{IdxTensor, TensorStorageError};

/// Error returned by explicit [`IdxTensor`] CUDA transfer and residency checks.
///
/// CUDA transfer is deliberately limited to untracked, logically dense values.
/// The error retains deferred-storage and context failures as typed sources; it
/// never falls back to a CPU transfer or exposes runtime identifiers.
///
/// # Examples
///
/// ```
/// use tensor4all_core::defaults::IdxTensorCudaError;
///
/// let error = IdxTensorCudaError::Tracked { operation: "upload" };
/// assert!(error.to_string().contains("tracked"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum IdxTensorCudaError {
    /// The tensor has a deferred storage diagnostic that must be reported first.
    #[error("IdxTensor CUDA {operation} rejected deferred storage: {source}")]
    DeferredStorage {
        /// Operation that observed the deferred storage.
        operation: &'static str,
        /// Original storage diagnostic.
        #[source]
        source: TensorStorageError,
    },
    /// CUDA transfer does not preserve automatic-differentiation tracking.
    #[error(
        "IdxTensor CUDA {operation} rejects tracked values; provide an untracked dense tensor"
    )]
    Tracked {
        /// Operation that rejected the tracked tensor.
        operation: &'static str,
    },
    /// CUDA transfer does not accept diagonal or general structured storage.
    #[error("IdxTensor CUDA {operation} requires dense storage, got {kind:?}")]
    UnsupportedStorage {
        /// Operation that rejected the storage layout.
        operation: &'static str,
        /// Logical storage layout observed before any materialization.
        kind: StorageKind,
    },
    /// CUDA transfer requires canonical one-class-per-axis metadata.
    #[error("IdxTensor CUDA {operation} requires dense axis classes")]
    NonDenseAxisClasses {
        /// Operation that rejected the axis metadata.
        operation: &'static str,
    },
    /// A residency check found no eager value to inspect.
    #[error("IdxTensor CUDA {operation} requires an eager value")]
    MissingEagerInner {
        /// Operation that required the eager value.
        operation: &'static str,
    },
    /// The supplied CUDA context rejected placement or initialization metadata.
    #[error("IdxTensor CUDA {operation} context operation failed: {source}")]
    Context {
        /// Operation that observed the context failure.
        operation: &'static str,
        /// Original typed context diagnostic.
        #[source]
        source: CudaExecutionContextError,
    },
    /// The tensor belongs to another eager CUDA context.
    #[error("IdxTensor CUDA {operation} received a value from another context")]
    ForeignContext {
        /// Operation that observed the context mismatch.
        operation: &'static str,
    },
    /// Native eager wrapping or explicit transfer preparation failed.
    #[error("IdxTensor CUDA {operation} failed: {source}")]
    Operation {
        /// Operation that failed.
        operation: &'static str,
        /// Original native or eager diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

fn precheck(tensor: &IdxTensor, operation: &'static str) -> Result<(), IdxTensorCudaError> {
    if let Some(source) = tensor.deferred_storage_error() {
        return Err(IdxTensorCudaError::DeferredStorage {
            operation,
            source: source.clone(),
        });
    }
    if tensor.tracks_grad() {
        return Err(IdxTensorCudaError::Tracked { operation });
    }
    let kind = tensor.storage_kind();
    if kind != StorageKind::Dense {
        return Err(IdxTensorCudaError::UnsupportedStorage { operation, kind });
    }
    if !tensor
        .axis_classes()
        .iter()
        .copied()
        .eq(0..tensor.indices().len())
    {
        return Err(IdxTensorCudaError::NonDenseAxisClasses { operation });
    }
    Ok(())
}

fn operation_error<E>(operation: &'static str, source: E) -> IdxTensorCudaError
where
    E: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
{
    IdxTensorCudaError::Operation {
        operation,
        source: Arc::from(source.into()),
    }
}

fn context_error(operation: &'static str, source: CudaExecutionContextError) -> IdxTensorCudaError {
    IdxTensorCudaError::Context { operation, source }
}

impl IdxTensor {
    /// Upload this untracked, logically dense tensor into a CUDA context.
    ///
    /// The host value is duplicated before the explicit device transfer. The
    /// returned tensor keeps the original ordered indices and is owned by the
    /// supplied context's eager runtime.
    ///
    /// # Errors
    ///
    /// Returns [`IdxTensorCudaError`] when storage is deferred, tracked,
    /// structured, not eagerly materializable, unsupported by CUDA, or when the
    /// explicit context transfer fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{CudaExecutionContext, IdxTensor, IdxTensorCudaError};
    ///
    /// let upload: fn(
    ///     &IdxTensor,
    ///     &CudaExecutionContext,
    /// ) -> Result<IdxTensor, IdxTensorCudaError> = IdxTensor::upload_cuda;
    /// assert_eq!(
    ///     std::mem::size_of_val(&upload),
    ///     std::mem::size_of::<fn(
    ///         &IdxTensor,
    ///         &CudaExecutionContext,
    ///     ) -> Result<IdxTensor, IdxTensorCudaError>>(),
    /// );
    /// ```
    pub fn upload_cuda(&self, context: &CudaExecutionContext) -> Result<Self, IdxTensorCudaError> {
        precheck(self, "upload")?;
        let native = self
            .cuda_duplicate_native()
            .map_err(|source| operation_error("upload materialization", source))?;
        let uploaded = context
            .upload_cuda(&native)
            .map_err(|source| context_error("upload", source))?;
        let runtime = context
            .eager_runtime()
            .map_err(|source| context_error("upload", source))?;
        // A device tensor needs the owning engine to become an eager leaf; the
        // context lends the same engine it used for the upload.
        let inner = context
            .with_session(|engine| {
                runtime.with_eager_session(engine, |session| session.constant_from(uploaded))
            })
            .map_err(|source| context_error("upload eager session", source))?
            .map_err(|source| operation_error("upload eager wrapping", source))?;
        Self::from_inner(self.indices.clone(), inner)
            .map_err(|source| operation_error("upload tensor construction", source))
    }

    /// Download this CUDA-resident tensor into the host eager context.
    ///
    /// The residency and context checks are metadata-only. The resident value
    /// is then duplicated in its device allocation domain, downloaded
    /// explicitly, and reconstructed with the original ordered indices.
    ///
    /// # Errors
    ///
    /// Returns [`IdxTensorCudaError`] when the value is not an untracked dense
    /// tensor resident in the supplied context, or when explicit download or
    /// host reconstruction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{CudaExecutionContext, IdxTensor, IdxTensorCudaError};
    ///
    /// let download: fn(
    ///     &IdxTensor,
    ///     &CudaExecutionContext,
    /// ) -> Result<IdxTensor, IdxTensorCudaError> = IdxTensor::download;
    /// assert_eq!(
    ///     std::mem::size_of_val(&download),
    ///     std::mem::size_of::<fn(
    ///         &IdxTensor,
    ///         &CudaExecutionContext,
    ///     ) -> Result<IdxTensor, IdxTensorCudaError>>(),
    /// );
    /// ```
    pub fn download(&self, context: &CudaExecutionContext) -> Result<Self, IdxTensorCudaError> {
        self.validate_cuda_residency(context)?;
        let resident = self
            .cuda_duplicate_native()
            .map_err(|source| operation_error("download materialization", source))?;
        let host = context
            .download(&resident)
            .map_err(|source| context_error("download", source))?;
        Self::from_native(self.indices.clone(), host)
            .map_err(|source| operation_error("download tensor construction", source))
    }

    /// Submit a device-to-host transfer of this tensor without waiting for it.
    ///
    /// The host value is reconstructed in [`PendingCudaIdxDownload::wait`] from the filled pinned buffer,
    /// preserving this tensor's ordered indices, so submission and completion can be measured separately.
    ///
    /// # Errors
    ///
    /// Returns [`IdxTensorCudaError::DeferredStorage`] for a deferred storage payload,
    /// [`IdxTensorCudaError::Tracked`] for a tracked value and [`IdxTensorCudaError::Context`] when the tensor is
    /// not resident in this exact context. Returns [`IdxTensorCudaError::Operation`] when the tensor holds no
    /// elements (the pinned allocator rejects a zero-byte request) or when its shape or byte count overflows
    /// `usize`, and [`IdxTensorCudaError::Context`] again when the pinned allocation or the submission is
    /// refused.
    pub fn download_pending(
        &self,
        context: &CudaExecutionContext,
    ) -> Result<PendingCudaIdxDownload, IdxTensorCudaError> {
        self.validate_cuda_residency(context)?;
        let resident = self
            .cuda_duplicate_native()
            .map_err(|source| operation_error("download pending materialization", source))?;
        let dtype = resident.dtype();
        let shape = resident.shape().to_vec();
        let elements = shape
            .iter()
            .try_fold(1usize, |total, extent| total.checked_mul(*extent))
            .ok_or_else(|| {
                operation_error(
                    "download pending byte count",
                    "the shape product overflows usize",
                )
            })?;
        let bytes = elements.checked_mul(dtype_size(dtype)).ok_or_else(|| {
            operation_error(
                "download pending byte count",
                "the byte count overflows usize",
            )
        })?;
        if bytes == 0 {
            return Err(operation_error(
                "download pending allocation",
                "an empty tensor has no pinned buffer to fill; use the synchronous download",
            ));
        }
        let buffer = context
            .allocate_pinned(bytes)
            .map_err(|source| context_error("download pending allocation", source))?;
        let pending = context
            .download_pending(resident, buffer)
            .map_err(|source| context_error("download pending submit", source))?;
        Ok(PendingCudaIdxDownload {
            inner: pending,
            indices: self.indices.clone(),
            dtype,
            shape,
        })
    }

    /// Validate CUDA placement and eager-runtime identity without reading data.
    ///
    /// This checks the existing eager value's context identity and
    /// [`tenferro::TensorRead`] placement metadata. It does not duplicate,
    /// download, or otherwise access host tensor values.
    ///
    /// # Errors
    ///
    /// Returns [`IdxTensorCudaError`] when the tensor is deferred, tracked,
    /// structured, missing an eager value, placed on a different CUDA context,
    /// or rejected by the supplied context's placement validator.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{CudaExecutionContext, IdxTensor, IdxTensorCudaError};
    ///
    /// let validate: fn(
    ///     &IdxTensor,
    ///     &CudaExecutionContext,
    /// ) -> Result<(), IdxTensorCudaError> = IdxTensor::validate_cuda_residency;
    /// assert_eq!(
    ///     std::mem::size_of_val(&validate),
    ///     std::mem::size_of::<fn(
    ///         &IdxTensor,
    ///         &CudaExecutionContext,
    ///     ) -> Result<(), IdxTensorCudaError>>(),
    /// );
    /// ```
    pub fn validate_cuda_residency(
        &self,
        context: &CudaExecutionContext,
    ) -> Result<(), IdxTensorCudaError> {
        precheck(self, "validate CUDA residency")?;
        let inner = self
            .cuda_eager_inner()
            .ok_or(IdxTensorCudaError::MissingEagerInner {
                operation: "validate CUDA residency",
            })?;
        let expected_context = context
            .context_id()
            .map_err(|source| context_error("validate CUDA residency", source))?;
        if inner.ctx_id() != expected_context {
            return Err(IdxTensorCudaError::ForeignContext {
                operation: "validate CUDA residency",
            });
        }
        let read = inner.tensor_read();
        context
            .validate_cuda_placement_metadata(read.placement(), read.allocation_domain())
            .map_err(|source| context_error("validate CUDA residency", source))
    }
}

/// One submitted device-to-host transfer of an indexed tensor.
///
/// Dropping the handle moves the transfer into the runtime's retirement queue, where a later query releases it
/// without waiting - provided the copy has completed and the queue can admit it, since the pinned implementation's
/// queue is bounded and a transfer that cannot be admitted is retained rather than released. Reclamation also
/// needs something that still owns the runtime, and releasing the *last* runtime owner is a different event
/// that synchronizes initialized streams. Dropping never proves the copy; only
/// [`PendingCudaIdxDownload::wait`] hands back a host value.
pub struct PendingCudaIdxDownload {
    inner: PendingCudaDownload,
    indices: Vec<crate::DynIndex>,
    dtype: DType,
    shape: Vec<usize>,
}

impl std::fmt::Debug for PendingCudaIdxDownload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PendingCudaIdxDownload")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl PendingCudaIdxDownload {
    /// Whether the copy's event has completed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::defaults::{IdxTensorCudaError, PendingCudaIdxDownload};
    ///
    /// let ready: fn(&mut PendingCudaIdxDownload) -> Result<bool, IdxTensorCudaError> =
    ///     PendingCudaIdxDownload::is_ready;
    /// assert!(std::mem::size_of_val(&ready) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`IdxTensorCudaError::Context`] when the event query itself fails for a reason other than
    /// "not ready".
    pub fn is_ready(&mut self) -> Result<bool, IdxTensorCudaError> {
        self.inner
            .is_ready()
            .map_err(|source| context_error("download pending readiness", source))
    }

    /// Wait for the copy and return the host value with this tensor's ordered indices.
    ///
    /// The host payload is built from the filled pinned bytes and the tensor's own metadata; no second transfer
    /// happens here. Decoding and host construction are part of the wait, so their cost is not hidden behind it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::defaults::{IdxTensor, IdxTensorCudaError, PendingCudaIdxDownload};
    ///
    /// let wait: fn(PendingCudaIdxDownload) -> Result<IdxTensor, IdxTensorCudaError> =
    ///     PendingCudaIdxDownload::wait;
    /// assert!(std::mem::size_of_val(&wait) > 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`IdxTensorCudaError::Context`] when the copy reports an unproven completion (the filled buffer
    /// must then be discarded rather than read), and [`IdxTensorCudaError::Operation`] when the bytes cannot be
    /// reconstructed for the tensor's dtype.
    pub fn wait(self) -> Result<IdxTensor, IdxTensorCudaError> {
        let (_device, buffer) = self
            .inner
            .wait()
            .map_err(|source| context_error("download pending wait", source))?;
        let host = host_native_from_pinned(&buffer, self.dtype, &self.shape)
            .map_err(|source| operation_error("download pending reconstruction", source))?;
        IdxTensor::from_native(self.indices, host)
            .map_err(|source| operation_error("download pending tensor construction", source))
    }
}

/// The byte width of one element of `dtype`, or `0` for a dtype with no host layout.
fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::Bool => 1,
        DType::F32 | DType::I32 => 4,
        DType::F64 | DType::I64 | DType::C32 => 8,
        DType::C64 => 16,
        // An externally-owned dtype has no host layout this route can rebuild.
        DType::External(_) => 0,
    }
}

/// Rebuild a host tensor from a filled pinned buffer and the tensor's own metadata.
fn host_native_from_pinned(
    buffer: &CudaPinnedBuffer,
    dtype: DType,
    shape: &[usize],
) -> Result<NativeTensor, String> {
    let bytes = buffer.as_slice();
    let shape = shape.to_vec();
    match dtype {
        DType::F32 => NativeTensor::from_vec_col_major(shape, decode_f32(bytes)?)
            .map_err(|error| format!("{error}")),
        DType::F64 => NativeTensor::from_vec_col_major(shape, decode_f64(bytes)?)
            .map_err(|error| format!("{error}")),
        DType::I32 => NativeTensor::from_vec_col_major(shape, decode_i32(bytes)?)
            .map_err(|error| format!("{error}")),
        DType::I64 => NativeTensor::from_vec_col_major(shape, decode_i64(bytes)?)
            .map_err(|error| format!("{error}")),
        DType::C32 => NativeTensor::from_vec_col_major(shape, decode_c32(bytes)?)
            .map_err(|error| format!("{error}")),
        DType::C64 => NativeTensor::from_vec_col_major(shape, decode_c64(bytes)?)
            .map_err(|error| format!("{error}")),
        // The pinned pending route rejects a Bool copy and has no layout for an externally-owned dtype, so
        // neither can reach this reconstruction through the pending entry.
        DType::Bool | DType::External(_) => {
            Err(format!("the pending route does not reconstruct {dtype:?}"))
        }
    }
}

/// Decode `bytes` as native-endian values of one fixed width, rejecting a partial trailing value.
macro_rules! decode_exact {
    ($bytes:expr, $ty:ty, $width:literal) => {{
        let bytes: &[u8] = $bytes;
        if bytes.len() % $width != 0 {
            Err(format!(
                "a buffer of {} bytes is not a whole number of {}-byte values",
                bytes.len(),
                $width
            ))
        } else {
            Ok(bytes
                .as_chunks::<$width>()
                .0
                .iter()
                .map(|chunk| <$ty>::from_ne_bytes(*chunk))
                .collect::<Vec<$ty>>())
        }
    }};
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    decode_exact!(bytes, f32, 4)
}

fn decode_f64(bytes: &[u8]) -> Result<Vec<f64>, String> {
    decode_exact!(bytes, f64, 8)
}

fn decode_i32(bytes: &[u8]) -> Result<Vec<i32>, String> {
    decode_exact!(bytes, i32, 4)
}

fn decode_i64(bytes: &[u8]) -> Result<Vec<i64>, String> {
    decode_exact!(bytes, i64, 8)
}

/// Decode a buffer of native-endian complex values, one `(real, imaginary)` pair per value.
///
/// The real parts are decoded as one run and paired with their imaginary neighbours: the pinned representation is
/// `repr(C) { re, im }`, so the bytes of a complex value are its two scalars back to back.
fn decode_c32(bytes: &[u8]) -> Result<Vec<num_complex::Complex32>, String> {
    let values = decode_f32(bytes)?;
    if values.len() % 2 != 0 {
        return Err(format!(
            "a Complex32 buffer of {} values is not a whole number of pairs",
            values.len()
        ));
    }
    Ok(values
        .chunks_exact(2)
        .map(|pair| num_complex::Complex32::new(pair[0], pair[1]))
        .collect())
}

fn decode_c64(bytes: &[u8]) -> Result<Vec<num_complex::Complex64>, String> {
    let values = decode_f64(bytes)?;
    if values.len() % 2 != 0 {
        return Err(format!(
            "a Complex64 buffer of {} values is not a whole number of pairs",
            values.len()
        ));
    }
    Ok(values
        .chunks_exact(2)
        .map(|pair| num_complex::Complex64::new(pair[0], pair[1]))
        .collect())
}

#[cfg(test)]
mod tests;
