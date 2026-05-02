//! Tree Tensor Cross Interpolation for `tensor4all-rs`.
//!
//! This crate is a Rust port of
//! [`TreeTCI.jl`](https://github.com/tensor4all/TreeTCI.jl).
//! Upstream `TreeTCI.jl/Project.toml` currently lists:
//!
//! - Ryo Watanabe <https://github.com/Ryo-wtnb11>
//!
//! Tree-specific pivot updates use the tcicore LUCI pivot substrate.
//!
//! # Overview
//!
//! The main entry point is [`crossinterpolate2`], which approximates a
//! multi-dimensional function as a tree tensor network using tensor cross
//! interpolation.
//!
//! The workflow is:
//! 1. Define a tree graph via [`TreeTciGraph`] (or use [`TreeTciGraph::linear_chain`]).
//! 2. Configure options via [`TreeTciOptions`].
//! 3. Call [`crossinterpolate2`] with a batch evaluator, returning a `TreeTN`.
//!
//! # Example
//!
//! ```
//! use tensor4all_treetci::{
//!     crossinterpolate2, DefaultProposer, GlobalIndexBatch,
//!     TreeTciEdge, TreeTciGraph, TreeTciOptions,
//! };
//! use anyhow::Result;
//!
//! // Approximate f(i, j) = delta(i, j) on a 2-site tree
//! let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
//! let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
//!     let mut vals = Vec::with_capacity(batch.n_points());
//!     for p in 0..batch.n_points() {
//!         let i = batch.get(0, p).unwrap();
//!         let j = batch.get(1, p).unwrap();
//!         vals.push(if i == j { 1.0 } else { 0.0 });
//!     }
//!     Ok(vals)
//! };
//!
//! let options = TreeTciOptions { tolerance: 1e-10, max_iter: 5, ..Default::default() };
//! let (treetn, ranks, errors) = crossinterpolate2::<f64, _, _>(
//!     evaluate, vec![2, 2], graph, vec![], options, None, &DefaultProposer,
//! ).unwrap();
//!
//! assert!(errors.last().copied().unwrap_or(1.0) < 1e-8);
//! assert!(ranks.last().copied().unwrap_or(0) <= 2);
//! ```

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
/// Per-edge pivot update logic built on top of the tcicore LUCI substrate.
pub mod update;
/// Edge visitation strategies.
pub mod visitor;

#[cfg(test)]
mod test_support;

pub use api::crossinterpolate2;
pub use assemble::{assemble_global_point, assemble_points_column_major, MultiIndex};
pub use batch::{GlobalIndexBatch, OwnedGlobalIndexBatch};
pub use graph::{TreeTciEdge, TreeTciGraph};
pub use key::SubtreeKey;
pub use materialize::to_treetn;
pub use optimize::{optimize_default, optimize_with_proposer, TreeTciOptions};
pub use proposer::{
    DefaultProposer, PivotCandidateProposer, SimpleProposer, TruncatedDefaultProposer,
};
#[allow(deprecated)]
pub use state::SimpleTreeTci;
pub use state::TreeTCI2;
pub use visitor::{AllEdges, EdgeVisitor};
