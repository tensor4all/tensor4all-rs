#![warn(missing_docs)]
//! Tensor storage and linear algebra backend for tensor4all.
//!
//! [`CpuExecutionContext`] is the canonical CPU integration path. It requires a
//! caller-supplied [`CpuBackend`] and owns plain, graph, and eager-AD runtime state.
//!
//! ## Feature flags
//!
//! - `explicit-context`: explicit CPU execution and logical tensor transfer.
//! - `global-defaults`: legacy process-global tensor operations.
//! - `backend-tenferro` (default): compatibility alias for `global-defaults`.

#[cfg(feature = "global-defaults")]
/// Dynamic scalar types supporting f32, f64, Complex32, and Complex64.
mod any_scalar;
#[cfg(feature = "global-defaults")]
/// Backend dispatch for dense linear algebra operations.
mod backend;
#[cfg(feature = "explicit-context")]
/// Explicit and optional process-global tenferro execution helpers.
mod context;
#[cfg(feature = "tenferro-cuda")]
/// Explicit visible-ordinal-0 CUDA execution and transfer boundaries.
mod cuda;
#[cfg(feature = "explicit-context")]
/// Einsum label helpers shared by both frontends.
mod einsum_ids;
#[cfg(feature = "explicit-context")]
/// Opt-in explicit/concrete execution frontend (issue #859 B1).
pub mod explicit;
#[cfg(feature = "global-defaults")]
/// Incremental QR state for successive randomized compression.
mod incremental_qr;
#[cfg(feature = "explicit-context")]
/// Backend-free tensor snapshots for execution-domain transfer.
mod logical_tensor;
#[cfg(feature = "global-defaults")]
/// Dense column-major matrix type and backend-backed matrix utilities.
mod matrix;
#[cfg(feature = "global-defaults")]
/// Process-level memory pressure helpers.
mod memory;
#[cfg(feature = "global-defaults")]
/// Tensor snapshot storage types and low-level dense/diagonal kernels.
mod storage;
#[cfg(feature = "global-defaults")]
pub(crate) mod tenferro_bridge;
#[cfg(feature = "global-defaults")]
/// Supported public tensor element types and native constructor hooks.
mod tensor_element;

#[cfg(feature = "global-defaults")]
pub use any_scalar::BackendScalar;
#[cfg(feature = "global-defaults")]
pub use backend::{
    full_piv_lu_backend, full_piv_lu_matrix, full_piv_lu_matrix_in, full_piv_lu_matrix_owned,
    qr_backend, qr_backend_in, solve_backend, solve_matrix, solve_matrix_in, solve_matrix_owned,
    src_error_estimate, src_error_estimate_general, svd_backend, svd_backend_in,
    triangular_solve_backend, triangular_solve_matrix, triangular_solve_matrix_in,
    triangular_solve_matrix_owned, BackendLinalgError, BackendLinalgScalar, FullPivLuMatrixResult,
    FullPivLuResult, FullPivLuScalar, MatrixSolveScalar, MatrixTriangularSolveScalar,
    SrcErrorEstimate, SvdResult,
};
#[cfg(feature = "global-defaults")]
pub use context::{
    default_cpu_execution_context, default_eager_ctx, with_default_backend, with_runtime_engine,
    EagerContextError, ExecutionContextEntryError,
};
#[cfg(feature = "explicit-context")]
pub use context::{
    set_default_cpu_threads, AlreadyInitialized, CpuBackend, CpuExecutionContext,
    CpuExecutionContextError, CpuThreadError, ExecutionContext,
};
#[cfg(feature = "tenferro-cuda")]
pub use cuda::{
    CudaExecutionContext, CudaExecutionContextError, CudaPinnedBuffer, PendingCudaDownload,
    PendingCudaUpload, CUDA_ORDINAL,
};
#[cfg(feature = "global-defaults")]
pub use incremental_qr::{IncrementalQr, IncrementalQrScalar};
#[cfg(feature = "explicit-context")]
pub use logical_tensor::{LogicalTensor, LogicalTensorData, LogicalTensorError};
#[cfg(feature = "global-defaults")]
pub use matrix::{
    batched_mat_mul_same_shape, batched_mat_mul_same_shape_owned, from_vec2d,
    grouped_mat_mul_shared, grouped_mat_mul_shared_in, grouped_mat_mul_shared_owned,
    grouped_mat_mul_shared_with_backend, hermitian_eigendecomposition,
    hermitian_exponential_first_column, lowest_hermitian_eigenpair, mat_mul, mat_mul_in,
    mat_mul_owned, submatrix, submatrix_argmax, swap_cols, swap_rows, transpose, try_from_vec2d,
    BlasMul, GroupedGemmError, GroupedGemmJob, HermitianEigenError, HermitianEigenScalar,
    HermitianEigendecomposition, HermitianEigenpair, Matrix, MatrixMulError, MatrixScalar,
    MatrixShapeError, MatrixTensorConversionError,
};
#[cfg(feature = "global-defaults")]
pub use memory::{release_process_allocator_cached_memory, AllocatorPressureRelief};
#[cfg(feature = "global-defaults")]
pub use storage::{
    contract_storage, make_mut_storage, min_dim, scale_target_dtype, NonfiniteFlags, Storage,
    StorageError, StorageKind, StorageResult, StorageScalar, StructuredStorage, SumFromStorage,
};
#[cfg(feature = "global-defaults")]
pub use tenferro_bridge::{
    axpby_native_tensor, axpby_storage_native, conj_native_tensor, contract_native_tensor,
    contract_storage_native, dense_native_tensor_from_col_major,
    dense_native_tensor_from_col_major_owned, diag_native_tensor_from_col_major,
    einsum_native_subscripts, einsum_native_tensor_reads, einsum_native_tensors,
    einsum_native_tensors_owned, native_tensor_primal_to_dense_col_major,
    native_tensor_primal_to_diag, native_tensor_primal_to_storage, outer_product_native_tensor,
    outer_product_storage_native, permute_native_tensor, permute_storage_native,
    print_and_reset_native_einsum_profile, qr_native_tensor, reset_native_einsum_profile,
    reshape_col_major_native_tensor, scale_native_tensor, scale_storage_native,
    storage_from_payload_native, storage_payload_native, storage_payload_native_read_input,
    storage_to_native_tensor, sum_native_tensor, svd_native_tensor, tangent_native_tensor,
    BridgeError, NativeTensorReadInput,
};
#[cfg(feature = "global-defaults")]
pub use tensor_element::TensorElement;

/// Report a violated internal invariant that no public input can reach.
///
/// This is the only sanctioned `panic!` in production code. Use it only for
/// (a) an internal invariant — a state no public input can reach — or (b) the
/// documented `# Panics` precondition of an infallible standard-trait
/// implementation that cannot return `Result`, such as an out-of-range
/// [`Index`](std::ops::Index)/[`IndexMut`](std::ops::IndexMut). Never use it
/// for user-reachable invalid input of a function that can return `Result`;
/// return a crate-local typed error instead.
///
/// `context` names the invariant and `error` carries the underlying
/// diagnostic, so the panic message is `"{context}: {error}"`. The library
/// panic audit allowlists this single `panic!` and reports every call site, so
/// invariant panics stay visible.
///
/// # Panics
///
/// Always panics and never returns.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::invariant_violated;
///
/// let payload = std::panic::catch_unwind(|| {
///     invariant_violated("rank mismatch", "expected 2 rows");
/// })
/// .unwrap_err();
/// let message = payload
///     .downcast_ref::<String>()
///     .expect("panic payload is a formatted string");
/// assert_eq!(message, "rank mismatch: expected 2 rows");
/// ```
#[doc(hidden)]
#[track_caller]
pub fn invariant_violated(context: &str, error: impl std::fmt::Display) -> ! {
    panic!("{context}: {error}")
}

#[cfg(all(test, feature = "global-defaults"))]
mod tests;
