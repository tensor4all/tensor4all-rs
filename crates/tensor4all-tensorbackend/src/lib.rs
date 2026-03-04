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

// Re-export underlying crates for downstream use.
pub use mdarray;
