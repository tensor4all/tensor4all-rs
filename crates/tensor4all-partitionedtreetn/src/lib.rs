#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

//! Partitioned Tree Tensor Network subdomains for tensor4all.
//!
//! This crate provides [`Projector`], [`SubDomainTreeTN`], and
//! [`PartitionedTreeTN`] patch algebra, including TreeTN-general adaptive
//! patching. Stored patch data is eagerly masked, and partition metadata is
//! validated transactionally.
//!
//! [`adaptive_interpolation::patched_interpolate`] builds a partition directly
//! from a function: it runs any engine implementing
//! [`TreeInterpolator`](tensor4all_treetn::interpolation::TreeInterpolator)
//! (for example `tensor4all_treetci::TreeTciInterpolator`) patch by patch and
//! splits the patches that cannot be accepted. The accuracy requirement is an
//! [`ErrorNorm`] with an [`ErrorTolerance`]: by default the L2 error over the
//! whole domain, measured by the driver itself (a certificate where patches
//! are exact or measured exhaustively, up to a calibrated, not proven,
//! rounding model; a statistical estimate where they are sampled, with the
//! audit on, and otherwise only an acceptance statistic), or the M2 sampled
//! max-norm criterion of the engine. Sampled measurements, audits included,
//! can miss a localized feature that enters a patch only through a corner or
//! an edge; only a certified result is a guarantee. The crate
//! depends on the engine trait only, not on an engine crate.
//!
//! The representation follows the partitioned tensor-network approach used by
//! [PartitionedMPSs.jl](https://github.com/tensor4all/PartitionedMPSs.jl) and
//! the adaptive patching literature. The patch queue of the adaptive
//! interpolation driver derives from TCIAlgorithms.jl (MIT) through
//! `tensor4all-partitionedtt`; see the notice in
//! [`adaptive_interpolation`] and `LICENSE-TCIALGORITHMS-MIT`.

pub mod adaptive_interpolation;
mod error;
mod error_norm;
mod partitioned_tree_tn;
mod patching;
mod projector;
pub mod reconstruction;
mod subdomain_tree_tn;

pub use error::{PartitionedTreeTNError, Result};
pub use error_norm::{ErrorNorm, ErrorTolerance, L2Reference};
pub use partitioned_tree_tn::PartitionedTreeTN;
pub use patching::{
    add_with_patching, contract_adaptive, truncate_adaptive, PatchSplitStrategy, PatchingOptions,
};
pub use projector::Projector;
pub use subdomain_tree_tn::SubDomainTreeTN;

pub use tensor4all_core::{DynIndex, IdxTensor};
pub use tensor4all_treetn::{SiteIndexNetwork, TreeTN};
