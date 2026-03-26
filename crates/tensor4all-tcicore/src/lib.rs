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

pub mod error;
pub mod matrix;
pub mod matrixaca;
pub mod matrixlu;
pub mod matrixluci;
pub mod scalar;
pub mod traits;

pub use error::{MatrixCIError, Result};
pub use matrix::{Matrix, from_vec2d};
pub use matrixaca::MatrixACA;
pub use matrixlu::{RrLU, RrLUOptions, rrlu, rrlu_inplace};
pub use matrixluci::MatrixLUCI;
pub use scalar::Scalar;
pub use traits::AbstractMatrixCI;
