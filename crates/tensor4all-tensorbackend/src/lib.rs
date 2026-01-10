//! Tensor storage and linear algebra backend for tensor4all.
//!
//! This crate provides:
//! - [`Storage`]: Dynamic tensor storage (f64/Complex64, Dense/Diag)
//! - [`AnyScalar`]: Dynamic scalar type (f64/Complex64)
//! - Backend dispatch for SVD and QR operations via [`SvdResult`]
//!
//! This crate re-exports `mdarray` and `faer_traits` for downstream use.
//! The `mdarray-linalg` dependency is kept internal to isolate API changes.

pub mod any_scalar;
pub mod backend;
pub mod storage;

pub use any_scalar::AnyScalar;
pub use backend::{qr_backend, svd_backend, SvdResult};
pub use storage::{
    contract_storage, make_mut_storage, mindim, storage_to_dtensor, DenseStorageC64,
    DenseStorageF64, DenseStorageFactory, DiagStorageC64, DiagStorageF64, Storage, StorageScalar,
    SumFromStorage,
};

// Re-export underlying crates for downstream use
// Note: mdarray_linalg is NOT re-exported to keep its API internal
pub use faer_traits;
pub use mdarray;
