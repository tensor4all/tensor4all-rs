#![warn(missing_docs)]
//! TCI Core infrastructure
//!
//! Shared foundation for tensor cross interpolation algorithms:
//! - Matrix CI: [`MatrixLUCI`], [`MatrixACA`], [`RrLU`]
//! - [`CachedFunction`]: Thread-safe cached function evaluation with wide key support
//! - [`IndexSet`]: Bidirectional index set for pivot management
//!
//! # Example
//!
//! ```
//! use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI, from_vec2d};
//!
//! let m = from_vec2d(vec![
//!     vec![1.0, 2.0, 3.0],
//!     vec![4.0, 5.0, 6.0],
//!     vec![7.0, 8.0, 9.0],
//! ]);
//!
//! let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
//! println!("Rank: {}", ci.rank());
//! ```

pub mod cached_function;
pub mod error;
pub mod indexset;
pub mod matrix;
pub mod matrixaca;
pub mod matrixlu;
pub mod matrixluci;
pub mod scalar;
pub mod traits;

pub use cached_function::cache_key::CacheKey;
pub use cached_function::error::CacheKeyError;
pub use cached_function::index_int::IndexInt;
pub use cached_function::CachedFunction;
pub use error::{MatrixCIError, Result};
pub use indexset::{IndexSet, LocalIndex, MultiIndex};
pub use matrix::{from_vec2d, Matrix};
pub use matrixaca::MatrixACA;
pub use matrixlu::{rrlu, rrlu_inplace, RrLU, RrLUOptions};
pub use matrixluci::MatrixLUCI;
pub use scalar::Scalar;
pub use traits::AbstractMatrixCI;
