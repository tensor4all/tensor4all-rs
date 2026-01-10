// Common (index, tags, utilities)
pub mod algorithm;
pub mod index;
pub mod index_like;
pub mod index_ops;
pub mod smallstring;
pub mod tagset;

pub use algorithm::{
    CanonicalForm, CompressionAlgorithm, ContractionAlgorithm, FactorizeAlgorithm,
};
pub use index::{DefaultIndex, DefaultTagSet, DynId, Index, NoSymmSpace, Symmetry, TagSet};
pub use index_like::{DynIndex, IndexLike};
pub use index_ops::{
    check_unique_indices, common_inds, hascommoninds, hasind, hasinds, noncommon_inds, replaceinds,
    replaceinds_in_place, sim, sim_owned, union_inds, unique_inds, ReplaceIndsError,
};
pub use smallstring::{SmallChar, SmallString, SmallStringError};
pub use tagset::{Tag, TagSetError, TagSetLike};

// Tensor (storage, tensor types)
pub mod any_scalar;
pub mod storage;
pub mod tensor;
pub mod tensor_like;

pub use any_scalar::AnyScalar;
pub use storage::{
    make_mut_storage, mindim, storage_to_dtensor, DenseStorageFactory, Storage, StorageScalar,
    SumFromStorage,
};
pub use tensor::{
    compute_permutation_from_indices, diag_tensor_dyn_len, diag_tensor_dyn_len_c64, is_diag_tensor,
    unfold_split, TensorAccess, TensorDynLen,
};
pub use tensor_like::{TensorLike, TensorLikeDowncast};

// Contraction
pub mod contract;
pub use contract::contract_multi;

// Linear algebra (SVD, QR, factorize)
mod backend;
pub mod direct_sum;
pub mod factorize;
pub mod qr;
pub mod svd;

pub use direct_sum::direct_sum;
pub use factorize::{
    factorize, Canonical, FactorizeAlg, FactorizeError, FactorizeOptions, FactorizeResult,
};
pub use qr::{default_qr_rtol, qr, qr_c64, qr_with, set_default_qr_rtol, QrError, QrOptions};
pub use svd::{
    default_svd_rtol, set_default_svd_rtol, svd, svd_c64, svd_with, SvdError, SvdOptions,
};
