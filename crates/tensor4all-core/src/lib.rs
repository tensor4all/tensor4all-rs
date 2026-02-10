//! Core tensor operations and types for tensor4all-rs.
//!
//! This crate provides the foundational types and operations for tensor networks:
//!
//! - **Index types**: [`DynIndex`], [`Index`], [`DynId`] for tensor indices
//! - **Tag sets**: [`TagSet`], [`TagSetLike`] for metadata tagging
//! - **Tensors**: [`TensorDynLen`] for dynamic-rank dense tensors
//! - **Operations**: Contraction, SVD, QR decomposition, factorization
//!
//! # Example
//!
//! ```
//! use tensor4all_core::{Index, DynIndex, TensorDynLen};
//!
//! // Create indices with dynamic identity
//! let i = Index::new_dyn(2);
//! let j = Index::new_dyn(3);
//!
//! // Create a tensor
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let t = TensorDynLen::from_dense_f64(vec![i.clone(), j.clone()], data);
//! ```

#![warn(missing_docs)]
// Common (tags, utilities, scalar)
pub mod global_default;
pub mod index_like;
pub mod scalar;
/// Stack-allocated fixed-capacity string types for ITensors.jl compatibility.
pub mod smallstring;
/// Tag set types for tensor metadata.
pub mod tagset;
pub mod truncation;

pub use scalar::CommonScalar;

// Default concrete type implementations (index, tensor, linalg, etc.)
pub mod defaults;

// Backwards compatibility: re-export defaults submodules as top-level modules
// This allows `tensor4all_core::index::...` to work
pub use defaults::index;

pub use defaults::{DefaultIndex, DefaultTagSet, DynId, DynIndex, Index, TagSet};
pub use index_like::{ConjState, IndexLike};

/// Index operations (replacement, set operations, contraction preparation).
pub mod index_ops;
pub use index_ops::{
    check_unique_indices, common_ind_positions, common_inds, hascommoninds, hasind, hasinds,
    noncommon_inds, prepare_contraction, prepare_contraction_pairs, replaceinds,
    replaceinds_in_place, union_inds, unique_inds, ContractionError, ContractionSpec,
    ReplaceIndsError,
};
pub use smallstring::{SmallChar, SmallString, SmallStringError};
pub use tagset::{Tag, TagSetError, TagSetLike};

// Tensor (storage, tensor types) - re-exported from tensor4all-tensorbackend
pub use tensor4all_tensorbackend::any_scalar;
pub use tensor4all_tensorbackend::storage;
pub mod tensor_index;
pub mod tensor_like;

pub use tensor_index::TensorIndex;

// Krylov subspace methods (GMRES, etc.)
pub mod krylov;

// Block tensor for block matrix GMRES
pub mod block_tensor;

// Backwards compatibility: re-export defaults::tensordynlen as tensor
pub use defaults::tensordynlen as tensor;

pub use any_scalar::AnyScalar;
pub use defaults::tensor_data::{TensorComponent, TensorData, TensorDataError};
pub use defaults::tensordynlen::{
    compute_permutation_from_indices, diag_tensor_dyn_len, diag_tensor_dyn_len_c64, is_diag_tensor,
    unfold_split, TensorAccess, TensorDynLen,
};
pub use storage::{
    make_mut_storage, mindim, storage_to_dtensor, DenseStorageFactory, Storage, StorageScalar,
    SumFromStorage,
};
pub use tensor_like::{
    AllowedPairs, Canonical, DirectSumResult, FactorizeAlg, FactorizeError, FactorizeOptions,
    FactorizeResult, TensorLike,
};

// Contraction - backwards compatibility
pub use defaults::contract;
pub use defaults::contract::{contract_connected, contract_multi};

// Linear algebra backend - re-exported from tensor4all-tensorbackend
pub use tensor4all_tensorbackend::backend;

// Re-export linear algebra modules from defaults for backwards compatibility
// This allows `tensor4all_core::svd::...`, `tensor4all_core::qr::...`, etc.
pub mod direct_sum {
    //! Re-export of direct sum operations.
    pub use crate::defaults::direct_sum::*;
}
pub mod factorize {
    //! Re-export of factorization operations.
    pub use crate::defaults::factorize::*;
}
pub mod qr {
    //! Re-export of QR decomposition operations.
    pub use crate::defaults::qr::*;
}
pub mod svd {
    //! Re-export of SVD decomposition operations.
    pub use crate::defaults::svd::*;
}

// Re-export linear algebra items for top-level access
pub use defaults::direct_sum::direct_sum;
pub use defaults::factorize::factorize;
pub use defaults::qr::{
    default_qr_rtol, qr, qr_c64, qr_with, set_default_qr_rtol, QrError, QrOptions,
};
pub use defaults::svd::{
    default_svd_rtol, set_default_svd_rtol, svd, svd_c64, svd_with, SvdError, SvdOptions,
};

// Global default and truncation utilities
pub use global_default::{GlobalDefault, InvalidRtolError};
pub use truncation::{DecompositionAlg, HasTruncationParams, TruncationParams};
