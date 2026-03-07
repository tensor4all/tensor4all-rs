#![warn(missing_docs)]
//! Tensor storage and linear algebra backend for tensor4all.
//!
//! This crate provides:
//! - [`Storage`]: Dynamic tensor storage (f64/Complex64, Dense/Diag)
//! - [`AnyScalar`]: Dynamic scalar type (f64/Complex64)
//! - tenferro-backed dispatch for SVD/QR/einsum operations
//!
//! This crate re-exports `mdarray` for downstream use.
//! Linalg backend details are kept internal behind tensorbackend APIs.
//!
//! ## Feature Flags
//!
//! - `backend-tenferro` (default): Use tenferro backend for linalg/einsum

/// Dynamic scalar types supporting f64 and Complex64.
pub mod any_scalar;
/// Backend dispatch for SVD and QR operations.
pub mod backend;
/// Einstein summation operations.
pub mod einsum;
/// Tensor storage types (Dense and Diagonal).
pub mod storage;
pub(crate) mod tenferro_bridge;

pub use any_scalar::AnyScalar;
pub use backend::{qr_backend, svd_backend, SvdResult};
pub use storage::{
    contract_storage, make_mut_storage, mindim, storage_to_dtensor, DenseScalar, DenseStorage,
    DenseStorageC64, DenseStorageF64, DenseStorageFactory, DiagStorage, DiagStorageC64,
    DiagStorageF64, Storage, StorageScalar, SumFromStorage,
};
pub use tenferro_bridge::{
    axpby_storage_native, conj_dyn_ad_tensor_native, contract_dyn_ad_tensor_native,
    contract_storage_native, dyn_ad_tensor_primal_to_storage, einsum_dyn_ad_tensors_native,
    outer_product_dyn_ad_tensor_native, outer_product_storage_native, permute_dyn_ad_tensor_native,
    permute_storage_native, qr_dyn_ad_tensor_native, scale_storage_native,
    storage_to_dyn_ad_tensor, sum_dyn_ad_tensor_native, svd_dyn_ad_tensor_native,
};
pub use tenferro_dyadtensor::{DynAdScalar, DynAdTensor};

// Re-export underlying crates for downstream use.
pub use mdarray;
pub use tenferro_dyadtensor;
pub use tenferro_tensor;
