#![warn(missing_docs)]
//! High-level Quantics TCI interface for function interpolation.
//!
//! This crate provides a user-friendly interface for interpolating functions
//! in Quantics Tensor Train (QTT) format. It wraps tensor4all-treetci and
//! quanticsgrids to provide seamless conversion between function domains
//! and quantics representations.
//!
//! This is a Rust port of [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl).
//!
//! # Important Conventions
//!
//! - **0-indexed grid indices**: All grid indices are **0-indexed** (matching the rest of
//!   the workspace). The first grid point is `[0, 0]`. QuanticsTCI.jl scripts must subtract 1.
//! - **Equal dimensions**: [`quanticscrossinterpolate_discrete`] and
//!   [`quanticscrossinterpolate_from_arrays`] require all dimensions to have the **same**
//!   number of points (same power of 2). Use [`quanticscrossinterpolate`] with an explicit
//!   [`DiscretizedGrid`] for non-uniform grids.
//! - **Power-of-2 grid sizes**: All grid dimensions must be powers of 2.
//!
//! # Overview
//!
//! The main workflow is:
//! 1. Create a grid describing your function's domain
//! 2. Call [`quanticscrossinterpolate`] (or the `_discrete` / `_from_arrays` variants)
//! 3. Use the resulting [`QuanticsTensorCI2`] for evaluation, integration, etc.
//!
//! # Example: Discrete Grid
//!
//! ```rust
//! use tensor4all_quanticstci::{
//!     quanticscrossinterpolate_discrete_batch, QtciOptions, QuanticsBatch,
//! };
//!
//! // Interpolate f(i, j) = i + j on a 16x16 grid.
//! // Discrete indices are 0-indexed.
//! let f = |batch: QuanticsBatch<'_, usize>| -> anyhow::Result<Vec<f64>> {
//!     Ok((0..batch.n_points())
//!         .map(|point| (batch.get(0, point).unwrap() + batch.get(1, point).unwrap()) as f64)
//!         .collect())
//! };
//! let sizes = vec![16, 16];
//!
//! let (qtci, _ranks, _errors) = quanticscrossinterpolate_discrete_batch(
//!     &sizes,
//!     f,
//!     None,
//!     QtciOptions::default(),
//! ).unwrap();
//!
//! let value = qtci.evaluate(&[4, 9]).unwrap();  // 0-indexed
//! assert!((value - 13.0).abs() < 1e-10);
//! ```
//!
//! # Example: Continuous Grid with `DiscretizedGrid`
//!
//! ```rust
//! use tensor4all_quanticstci::{
//!     quanticscrossinterpolate_batch, DiscretizedGrid, QtciOptions, QuanticsBatch,
//! };
//!
//! let grid = DiscretizedGrid::builder(&[4])  // 2^4 = 16 points
//!     .with_lower_bound(&[0.0])
//!     .with_upper_bound(&[1.0])
//!     .build()
//!     .unwrap();
//!
//! let f = |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
//!     Ok((0..batch.n_points())
//!         .map(|point| batch.get(0, point).unwrap().powi(2))
//!         .collect())
//! };
//!
//! let (qtci, _ranks, _errors) = quanticscrossinterpolate_batch(
//!     &grid,
//!     f,
//!     None,
//!     QtciOptions::default(),
//! ).unwrap();
//!
//! // integral() = sum * step_size (left Riemann sum of x^2 over [0, 1))
//! let integral = qtci.integral().unwrap();
//! assert!((integral - 1.0 / 3.0).abs() < 0.1); // rough Riemann sum with 16 points
//! ```
//!
//! # Choosing the Right API
//!
//! | Scenario | Function to use |
//! |---|---|
//! | Function on integer grid (e.g., lattice) | [`quanticscrossinterpolate_discrete_batch`] |
//! | Function on a continuous interval `[a, b)` | [`quanticscrossinterpolate_batch`] with [`DiscretizedGrid`] |
//! | Grid points given as explicit arrays | [`quanticscrossinterpolate_from_arrays_batch`] |
//! | Vector/tensor-valued function | [`quanticscrossinterpolate_multicomponent`] |
//!
//! Every entry point evaluates the target function **in batches**: it receives a
//! [`QuanticsBatch`], column-major `(n_dims, n_points)`, and returns one value
//! per requested point. A scalar function can be adapted with
//! [`pointwise_coordinate_batch`] or [`pointwise_index_batch`]. The point-wise
//! entry points (`quanticscrossinterpolate`, `quanticscrossinterpolate_discrete`,
//! `quanticscrossinterpolate_from_arrays`) are deprecated because they call the
//! target function once per point, which is the wrong boundary for vectorized
//! functions and language bindings.

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod prelude;

pub mod batch;
mod batched;
mod error;
mod options;
mod quantics_tci;

pub use batch::{
    pointwise_components_batch, pointwise_coordinate_batch, pointwise_index_batch, QuanticsBatch,
};
// The deprecated point-wise entry points stay re-exported so existing callers
// keep compiling with a warning; the deprecation fires at each call site.
#[allow(deprecated)]
pub use batched::{
    quanticscrossinterpolate_batched, quanticscrossinterpolate_multicomponent,
    QuanticsTensorCI2Batched,
};
pub use error::QuanticsTCIError;
pub use options::QtciOptions;
#[allow(deprecated)]
pub use quantics_tci::{
    quanticscrossinterpolate, quanticscrossinterpolate_batch, quanticscrossinterpolate_discrete,
    quanticscrossinterpolate_discrete_batch, quanticscrossinterpolate_from_arrays,
    quanticscrossinterpolate_from_arrays_batch, QuanticsTensorCI2,
};

// Re-export commonly used types from dependencies
pub use quanticsgrids::{DiscretizedGrid, InherentDiscreteGrid, UnfoldingScheme};
pub use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
pub use tensor4all_treetci::{DefaultProposer, TreeTciGraph, TreeTciOptions};
