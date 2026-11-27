#![warn(missing_docs)]
//! Tensor storage and linear algebra backend for tensor4all.
//!
//! [`CpuExecutionContext`] is the canonical CPU integration path. It requires a
//! caller-supplied [`CpuBackend`] and owns plain, graph, and eager-AD runtime state.
//!
//! ## Feature flags
//!
//! - `explicit-context` (default): the canonical caller-supplied surface: explicit CPU
//!   execution, logical tensor transfer, `Matrix`, `Storage`, `TensorElement`, the native
//!   tensor bridge and the `*_in` linalg and einsum entries.
//! - `global-defaults` (opt-in): legacy operations backed by one process-global context.
//! - `backend-tenferro`: compatibility alias for `global-defaults`.

#[cfg(feature = "explicit-context")]
/// Dynamic scalar types supporting f32, f64, Complex32, and Complex64.
mod any_scalar;
#[cfg(feature = "explicit-context")]
/// Backend dispatch for dense linear algebra operations.
mod backend;
#[cfg(feature = "explicit-context")]
/// Complete-pivoting LU of a square matrix stopped at a rank cap.
mod capped_full_piv_lu;
#[cfg(feature = "explicit-context")]
/// Explicit and optional process-global tenferro execution helpers.
mod context;
#[cfg(feature = "tenferro-cuda")]
/// Explicit visible-ordinal-0 CUDA execution and transfer boundaries.
mod cuda;
#[cfg(feature = "explicit-context")]
mod diagnostics;
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
#[cfg(feature = "explicit-context")]
/// Dense column-major matrix type and backend-backed matrix utilities.
mod matrix;
#[cfg(feature = "explicit-context")]
/// Process-level memory pressure helpers.
mod memory;
#[cfg(feature = "explicit-context")]
/// The engine-free shared runtime (`Runtime`) and its `&mut`-engine entries.
mod runtime;
#[cfg(feature = "explicit-context")]
/// Tensor snapshot storage types and low-level dense/diagonal kernels.
mod storage;
#[cfg(feature = "explicit-context")]
pub(crate) mod tenferro_bridge;
#[cfg(feature = "explicit-context")]
/// Supported public tensor element types and native constructor hooks.
mod tensor_element;

#[cfg(feature = "explicit-context")]
pub use any_scalar::{BackendScalar, ScalarTensorElement, ScalarValue};
#[cfg(feature = "global-defaults")]
pub use backend::{
    full_piv_lu_backend, full_piv_lu_matrix, full_piv_lu_matrix_owned, qr_backend, solve_backend,
    solve_matrix, solve_matrix_owned, src_error_estimate, src_error_estimate_general, svd_backend,
    triangular_solve_backend, triangular_solve_matrix, triangular_solve_matrix_owned,
};
#[cfg(feature = "explicit-context")]
pub use backend::{
    full_piv_lu_matrix_in, qr_backend_in, solve_matrix_in, src_error_estimate_general_in,
    svd_backend_in, triangular_solve_matrix_in, BackendLinalgError, BackendLinalgScalar,
    FullPivLuMatrixResult, FullPivLuResult, FullPivLuScalar, MatrixSolveScalar,
    MatrixTriangularSolveScalar, SrcErrorEstimate, SvdResult,
};
#[cfg(feature = "explicit-context")]
pub use capped_full_piv_lu::{capped_full_piv_lu, CappedFullPivLu, CappedFullPivLuScalar};
#[cfg(feature = "global-defaults")]
pub use context::{
    default_cpu_execution_context, default_eager_ctx, with_default_backend, EagerContextError,
};
#[cfg(feature = "explicit-context")]
pub use context::{
    default_cpu_wait_policy, set_default_cpu_threads, set_default_cpu_wait_policy,
    with_runtime_engine, with_runtime_engine_or_global, AlreadyInitialized, CpuBackend,
    CpuExecutionContext, CpuExecutionContextError, CpuThreadError, CpuWaitPolicy, ExecutionContext,
    ExecutionContextEntryError, WaitPolicyAlreadyInitialized,
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
    batched_mat_mul_same_shape, batched_mat_mul_same_shape_owned, hermitian_eigendecomposition,
    hermitian_exponential_first_column, lowest_hermitian_eigenpair, mat_mul, mat_mul_owned,
};
#[cfg(feature = "explicit-context")]
pub use matrix::{
    from_vec2d, hermitian_eigendecomposition_in, hermitian_exponential_first_column_in,
    lowest_hermitian_eigenpair_in, mat_mul_in, submatrix, submatrix_argmax, swap_cols, swap_rows,
    transpose, try_from_vec2d, BlasMul, HermitianEigenError, HermitianEigenScalar,
    HermitianEigendecomposition, HermitianEigenpair, Matrix, MatrixMulError, MatrixScalar,
    MatrixShapeError, MatrixTensorConversionError,
};
#[cfg(feature = "explicit-context")]
pub use memory::{release_process_allocator_cached_memory, AllocatorPressureRelief};
#[cfg(feature = "explicit-context")]
pub use runtime::{Runtime, RuntimeError};
#[cfg(feature = "explicit-context")]
pub use storage::{
    scale_target_dtype, NonfiniteFlags, Storage, StorageError, StorageKind, StorageResult,
    StorageScalar, StructuredStorage, SumFromStorage,
};
/// The engine trait every computing entry takes as `exec: &mut dyn ExecutionEngine`.
#[cfg(feature = "explicit-context")]
pub use tenferro::ExecutionEngine;
#[cfg(feature = "explicit-context")]
pub use tenferro_bridge::{
    conj_native_tensor, dense_native_tensor_from_col_major,
    dense_native_tensor_from_col_major_owned, dense_native_tensor_read_input_from_col_major,
    diag_native_tensor_from_col_major, einsum_native_tensor_reads_in, einsum_native_tensors_in,
    einsum_native_tensors_owned_in, native_tensor_primal_to_dense_col_major,
    print_and_reset_native_einsum_profile, reset_native_einsum_profile,
    storage_from_payload_native, storage_payload_native, storage_payload_native_read_input,
    storage_to_native_tensor, BridgeError, NativeEinsumPlan, NativeTensorReadInput,
};
#[cfg(feature = "global-defaults")]
pub use tenferro_bridge::{
    einsum_native_subscripts, einsum_native_tensor_reads, einsum_native_tensors,
    einsum_native_tensors_owned, native_tensor_primal_to_diag, qr_native_tensor,
};
/// The CUDA engine a caller opens and passes as `exec`, and its device id.
#[cfg(feature = "tenferro-cuda")]
pub use tenferro_gpu::cuda::{CudaDeviceId, CudaEngine};
#[cfg(feature = "explicit-context")]
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

// Execution vocabulary for downstream algorithms that borrow caller-owned lanes.
#[cfg(feature = "explicit-context")]
pub use tenferro_cpu::{threadteam::Exec, CpuEngine};
