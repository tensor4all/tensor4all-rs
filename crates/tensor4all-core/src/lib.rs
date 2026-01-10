// Common (tags, utilities)
pub mod index_like;
pub mod smallstring;
pub mod tagset;

// Default concrete type implementations (index, tensor, linalg, etc.)
pub mod defaults;

// Backwards compatibility: re-export defaults submodules as top-level modules
// This allows `tensor4all_core::index::...` to work
pub use defaults::index;

pub use defaults::{DefaultIndex, DefaultTagSet, DynId, DynIndex, Index, NoSymmSpace, Symmetry, TagSet};
pub use index_like::{ConjState, IndexLike};

// Index operations (uses defaults::*)
pub mod index_ops;
pub use index_ops::{
    check_unique_indices, common_inds, hascommoninds, hasind, hasinds, noncommon_inds, replaceinds,
    replaceinds_in_place, union_inds, unique_inds, ReplaceIndsError,
};
pub use smallstring::{SmallChar, SmallString, SmallStringError};
pub use tagset::{Tag, TagSetError, TagSetLike};

// Tensor (storage, tensor types) - re-exported from tensor4all-tensorbackend
pub use tensor4all_tensorbackend::any_scalar;
pub use tensor4all_tensorbackend::storage;
pub mod tensor_like;

// Krylov subspace methods (GMRES, etc.)
pub mod krylov;

// Backwards compatibility: re-export defaults::tensordynlen as tensor
pub use defaults::tensordynlen as tensor;

pub use any_scalar::AnyScalar;
pub use storage::{
    make_mut_storage, mindim, storage_to_dtensor, DenseStorageFactory, Storage, StorageScalar,
    SumFromStorage,
};
pub use defaults::tensordynlen::{
    compute_permutation_from_indices, diag_tensor_dyn_len, diag_tensor_dyn_len_c64, is_diag_tensor,
    unfold_split, TensorAccess, TensorDynLen,
};
pub use tensor_like::{
    Canonical, DirectSumResult, FactorizeAlg, FactorizeError, FactorizeOptions, FactorizeResult, TensorLike,
};

// Contraction - backwards compatibility
pub use defaults::contract;
pub use defaults::contract::contract_multi;

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
