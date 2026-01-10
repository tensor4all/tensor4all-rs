//! Tensor storage and linear algebra backend for tensor4all.
//!
//! This crate provides:
//! - [`Storage`]: Dynamic tensor storage (f64/Complex64, Dense/Diag)
//! - [`AnyScalar`]: Dynamic scalar type (f64/Complex64)
//! - Backend dispatch for SVD and QR operations
//!
//! This crate also re-exports `mdarray`, `mdarray_linalg`, and `faer_traits`
//! for use by downstream crates.

pub mod any_scalar;
pub mod backend;
pub mod storage;

pub use any_scalar::AnyScalar;
pub use backend::{qr_backend, svd_backend};
pub use storage::{
    contract_storage, make_mut_storage, mindim, storage_to_dtensor, DenseStorageC64,
    DenseStorageF64, DenseStorageFactory, DiagStorageC64, DiagStorageF64, Storage, StorageScalar,
    SumFromStorage,
};

// Re-export underlying crates for downstream use
pub use faer_traits;
pub use mdarray;
pub use mdarray_linalg;
