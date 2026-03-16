#![warn(missing_docs)]
//! Tensor storage and linear algebra backend for tensor4all.
//!
//! This crate provides:
//! - [`Storage`]: Dynamic tensor storage (f64/Complex64, Dense/Diag)
//! - [`AnyScalar`]: Dynamic scalar type backed by rank-0 `tenferro::Tensor`
//! - tenferro-backed execution helpers for tensor algebra
//!
//! ## Feature Flags
//!
//! - `backend-tenferro` (default): Use tenferro backend for linalg/einsum

/// Dynamic scalar types supporting f64 and Complex64.
pub mod any_scalar;
/// Backend dispatch for SVD and QR operations.
pub mod backend;
pub(crate) mod layout;
/// Tensor storage types (Dense and Diagonal).
pub mod storage;
pub(crate) mod tenferro_bridge;
/// Supported public tensor element types and native constructor hooks.
pub mod tensor_element;

pub use any_scalar::AnyScalar;
pub use storage::{
    contract_storage, make_mut_storage, mindim, DenseScalar, DenseStorage, DenseStorageC64,
    DenseStorageF64, DenseStorageFactory, DiagStorage, DiagStorageC64, DiagStorageF64, Storage,
    StorageScalar, SumFromStorage,
};
pub use tenferro_bridge::{
    axpby_native_tensor, axpby_storage_native, conj_native_tensor, contract_native_tensor,
    contract_storage_native, dense_native_tensor_from_row_major, diag_native_tensor_from_row_major,
    einsum_native_tensors, native_tensor_primal_to_dense_c64, native_tensor_primal_to_dense_f64,
    native_tensor_primal_to_diag_c64, native_tensor_primal_to_diag_f64,
    native_tensor_primal_to_storage, outer_product_native_tensor, outer_product_storage_native,
    permute_native_tensor, permute_storage_native, qr_native_tensor,
    reshape_row_major_native_tensor, scale_native_tensor, scale_storage_native,
    storage_to_native_tensor, sum_native_tensor, svd_native_tensor, tangent_native_tensor,
};
pub use tensor_element::TensorElement;
