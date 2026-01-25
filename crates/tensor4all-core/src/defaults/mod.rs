//! Default concrete type implementations.
//!
//! This module provides the default concrete types for tensor network operations:
//!
//! - [`DynId`]: Runtime identity (UUID-based unique identifier)
//! - [`TagSet`]: Tag set for metadata (Arc-wrapped for cheap cloning)
//! - [`Index`]: Generic index type (`Index<Id, Tags>`)
//! - [`DynIndex`]: Default index type (`Index<DynId, TagSet>`)
//! - [`TensorDynLen`]: Dense tensor with dynamic rank
//!
//! Linear algebra operations:
//! - [`svd::svd`]: Singular Value Decomposition
//! - [`qr::qr`]: QR decomposition
//! - [`factorize::factorize`]: Unified factorization interface
//! - [`direct_sum::direct_sum`]: Direct sum of tensors
//!
//! These types are suitable for most tensor network applications and provide
//! a good balance of flexibility and performance.

pub mod index;
pub mod tensor_data;
/// Dynamic-length tensor implementation.
pub mod tensordynlen;

// Contraction
pub mod contract;

// Linear algebra modules
pub mod direct_sum;
pub mod factorize;
pub mod qr;
pub mod svd;

pub use contract::{
    build_diag_union, build_diag_union_from_components, build_diag_union_from_data, collect_sizes,
    contract_connected, contract_multi, remap_output_ids, remap_tensor_ids, AxisUnionFind,
};
pub use index::{DefaultIndex, DefaultTagSet, DynId, DynIndex, Index, TagSet};
pub use tensor_data::{TensorComponent, TensorData};
pub use tensordynlen::{
    compute_permutation_from_indices, diag_tensor_dyn_len, diag_tensor_dyn_len_c64, is_diag_tensor,
    unfold_split, TensorAccess, TensorDynLen,
};

// Re-export linear algebra functions and types
pub use direct_sum::direct_sum;
pub use factorize::{
    factorize, Canonical, FactorizeAlg, FactorizeError, FactorizeOptions, FactorizeResult,
};
pub use qr::{default_qr_rtol, qr, qr_c64, qr_with, set_default_qr_rtol, QrError, QrOptions};
pub use svd::{
    default_svd_rtol, set_default_svd_rtol, svd, svd_c64, svd_with, SvdError, SvdOptions,
};
