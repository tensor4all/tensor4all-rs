//! Tensor storage and linear algebra backend for tensor4all.
//!
//! This crate provides:
//! - [`Storage`]: Dynamic tensor storage (f64/Complex64, Dense/Diag)
//! - [`AnyScalar`]: Dynamic scalar type (f64/Complex64)
//! - Backend dispatch for SVD and QR operations via [`SvdResult`]
//!
//! This crate re-exports `mdarray` and `faer_traits` for downstream use.
//! The `mdarray-linalg` dependency is kept internal to isolate API changes.
//!
//! ## Feature Flags
//!
//! - `backend-faer` (default): Use FAER for linear algebra operations
//! - `backend-lapack`: Use LAPACK for linear algebra operations
//! - `backend-libtorch`: Enable PyTorch/libtorch backend for tensor operations and autograd

pub mod any_scalar;
pub mod backend;
pub mod einsum;
pub mod storage;

// Torch backend module (feature-gated)
#[cfg(feature = "backend-libtorch")]
pub mod torch;

pub use any_scalar::AnyScalar;
pub use backend::{qr_backend, svd_backend, SvdResult};
pub use storage::{
    contract_storage, make_mut_storage, mindim, storage_to_dtensor, DenseScalar, DenseStorage,
    DenseStorageC64, DenseStorageF64, DenseStorageFactory, DiagStorage, DiagStorageC64,
    DiagStorageF64, Storage, StorageScalar, SumFromStorage,
};

// Re-export underlying crates for downstream use
// Note: mdarray_linalg is NOT re-exported to keep its API internal
pub use faer_traits;
pub use mdarray;
