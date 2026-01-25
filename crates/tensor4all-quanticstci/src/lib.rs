#![warn(missing_docs)]
//! High-level Quantics TCI interface for function interpolation.
//!
//! This crate provides a user-friendly interface for interpolating functions
//! in Quantics Tensor Train (QTT) format. It wraps tensor4all-tensorci and
//! quanticsgrids to provide seamless conversion between function domains
//! and quantics representations.
//!
//! This is a Rust port of [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl).
//!
//! # Overview
//!
//! The main workflow is:
//! 1. Create a grid describing your function's domain
//! 2. Call `quanticscrossinterpolate` with your function
//! 3. Use the resulting `QuanticsTensorCI2` for evaluation, integration, etc.
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};
//!
//! // Interpolate f(i, j) = i + j on a 16x16 grid
//! let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;
//! let sizes = vec![16, 16];
//!
//! let (qtci, ranks, errors) = quanticscrossinterpolate_discrete(
//!     &sizes,
//!     f,
//!     QtciOptions::default(),
//! ).unwrap();
//!
//! let value = qtci.evaluate(&[5, 10]).unwrap();
//! assert!((value - 15.0).abs() < 1e-10);
//! ```

mod options;
mod quantics_tci;

pub use options::QtciOptions;
pub use quantics_tci::{
    quanticscrossinterpolate, quanticscrossinterpolate_discrete,
    quanticscrossinterpolate_from_arrays, QuanticsTensorCI2,
};

// Re-export commonly used types from dependencies
pub use quanticsgrids::{DiscretizedGrid, InherentDiscreteGrid, UnfoldingScheme};
pub use tensor4all_tensorci::{PivotSearchStrategy, Scalar, TCI2Options, TensorCI2};
