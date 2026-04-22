#![warn(missing_docs)]
//! Tensor storage and linear algebra backend for tensor4all.
//!
//! This crate provides:
//! - [`Storage`]: Dynamic snapshot storage for logical tensor values
//! - [`StructuredStorage`]: `axis_classes`-aware materialized snapshots
//! - [`AnyScalar`]: Dynamic scalar type backed by rank-0 `tenferro::Tensor`
//! - tenferro-backed execution helpers for tensor algebra
//!
//! ## Feature Flags
//!
//! - `backend-tenferro` (default): Use tenferro backend for linalg/einsum

/// Dynamic scalar types supporting f64 and Complex64.
mod any_scalar;
/// Backend dispatch for SVD and QR operations.
mod backend;
/// Thread-local tenferro execution helpers.
mod context;
/// Tensor snapshot storage types and low-level dense/diagonal kernels.
mod storage;
pub(crate) mod tenferro_bridge;
/// Supported public tensor element types and native constructor hooks.
mod tensor_element;

pub use any_scalar::AnyScalar;
pub use backend::{qr_backend, svd_backend, BackendLinalgScalar, SvdResult};
pub use context::{default_eager_ctx, with_default_backend};
pub use storage::{
    contract_storage, make_mut_storage, mindim, Storage, StorageKind, StorageScalar,
    StructuredStorage, SumFromStorage,
};
pub use tenferro_bridge::{
    axpby_native_tensor, axpby_storage_native, conj_native_tensor, contract_native_tensor,
    contract_storage_native, dense_native_tensor_from_col_major, diag_native_tensor_from_col_major,
    einsum_native_tensors, native_tensor_primal_to_dense_c64_col_major,
    native_tensor_primal_to_dense_col_major, native_tensor_primal_to_dense_f64_col_major,
    native_tensor_primal_to_diag_c64, native_tensor_primal_to_diag_f64,
    native_tensor_primal_to_storage, outer_product_native_tensor, outer_product_storage_native,
    permute_native_tensor, permute_storage_native, print_and_reset_native_einsum_profile,
    qr_native_tensor, reset_native_einsum_profile, reshape_col_major_native_tensor,
    scale_native_tensor, scale_storage_native, storage_to_native_tensor, sum_native_tensor,
    svd_native_tensor, tangent_native_tensor,
};
pub use tensor_element::TensorElement;
