//! Tree Tensor Cross Interpolation for `tensor4all-rs`.
//!
//! This crate is a Rust port of
//! [`TreeTCI.jl`](https://github.com/tensor4all/TreeTCI.jl).
//! Upstream `TreeTCI.jl/Project.toml` currently lists:
//!
//! - Ryo Watanabe <https://github.com/Ryo-wtnb11>
//!
//! Tree-specific pivot updates use [`matrixluci`] as the low-level pivot
//! substrate.

#![warn(missing_docs)]

/// High-level TreeTCI entry points.
pub mod api;
/// Assembly helpers from subtree-local pivots to global site-order indices.
pub mod assemble;
/// Batch views for global site-order evaluation.
pub mod batch;
/// Tree graph helpers and edge-bipartition utilities for TreeTCI.
pub mod graph;
/// Canonical subtree-key types.
pub mod key;
/// TreeTN materialization from converged TreeTCI pivots.
pub mod materialize;
/// Optimization loop and options for TreeTCI.
pub mod optimize;
/// Pivot candidate generation strategies.
pub mod proposer;
/// TreeTCI state and initial pivot seeding.
pub mod state;
/// Per-edge pivot update logic built on top of `matrixluci`.
pub mod update;
/// Edge visitation strategies.
pub mod visitor;

pub use api::{crossinterpolate_tree, crossinterpolate_tree_with_proposer};
pub use assemble::{assemble_global_point, assemble_points_column_major, MultiIndex};
pub use batch::{GlobalIndexBatch, OwnedGlobalIndexBatch};
pub use graph::{TreeTciEdge, TreeTciGraph};
pub use key::SubtreeKey;
pub use materialize::to_treetn;
pub use optimize::{optimize_default, optimize_with_proposer, TreeTciOptions};
pub use proposer::{
    DefaultProposer, PivotCandidateProposer, SimpleProposer, TruncatedDefaultProposer,
};
pub use state::SimpleTreeTci;
pub use update::update_edge;
pub use visitor::{AllEdges, EdgeVisitor};
