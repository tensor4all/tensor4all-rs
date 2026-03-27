#![warn(missing_docs)]
//! Tensor Cross Interpolation (TCI) library
//!
//! This crate provides tensor cross interpolation algorithms for efficiently
//! approximating high-dimensional tensors as tensor trains.
//!
//! # Main algorithms
//!
//! - `TensorCI2`: Primary two-site TCI algorithm
//! - `crossinterpolate2`: Function to perform TCI2 interpolation
//! - `TensorCI1`: Legacy one-site TCI algorithm kept for compatibility
//! - `crossinterpolate1`: Legacy entry point for TCI1
//!
//! `TensorCI2` is the actively maintained path and uses `MatrixLUCI` from `tensor4all-tcicore`.
//! `TensorCI1` uses `MatrixACA` from `tensor4all-tcicore`.
//! `PivotSearchStrategy::Rook` uses lazy block-rook evaluation; when
//! `normalize_error` is enabled it normalizes by the maximum observed sample
//! value from the lazily requested entries rather than by a full-grid scan.
//!
//! # Example
//!
//! ```
//! use tensor4all_tensorci::{crossinterpolate2, TCI2Options};
//!
//! // Function to interpolate: f(i, j) = i + j + 1
//! let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
//! let local_dims = vec![4, 4];
//! let first_pivot = vec![vec![1, 1]];
//!
//! let (tci, ranks, errors) =
//!     crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
//!         f,
//!         None,
//!         local_dims,
//!         first_pivot,
//!         TCI2Options::default(),
//!     )
//!     .unwrap();
//! println!("TCI rank: {}", tci.rank());
//! ```

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod error;
pub mod globalpivot;
pub mod globalsearch;
pub mod integration;
pub mod optfirstpivot;
pub mod tensorci1;
pub mod tensorci2;

// Re-export main types
pub use error::{Result, TCIError};
pub use globalpivot::{DefaultGlobalPivotFinder, GlobalPivotFinder, GlobalPivotSearchInput};
pub use globalsearch::{estimate_true_error, floating_zone};
pub use optfirstpivot::opt_first_pivot;
pub use tensorci1::{crossinterpolate1, SweepStrategy, TCI1Options, TensorCI1};
pub use tensorci2::{
    crossinterpolate2, PivotSearchStrategy, Sweep2Strategy, TCI2Options, TensorCI2,
};

pub use tensor4all_tcicore::{
    CacheKey, CacheKeyError, CachedFunction, IndexInt, IndexSet, LocalIndex, MultiIndex, Scalar,
};
