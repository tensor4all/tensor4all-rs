#![warn(missing_docs)]
//! Matrix Cross Interpolation library
//!
//! This crate provides matrix cross interpolation algorithms:
//! - [`MatrixLUCI`]: LU-based Cross Interpolation (wrapper around [`RrLU`])
//! - [`MatrixACA`]: Adaptive Cross Approximation
//! - [`RrLU`]: Rank-Revealing LU decomposition
//!
//! Both [`MatrixLUCI`] and [`MatrixACA`] implement the [`AbstractMatrixCI`] trait,
//! providing a unified interface for low-rank matrix approximation.
//!
//! # Example
//!
//! ```
//! use matrixci::{MatrixLUCI, AbstractMatrixCI, from_vec2d};
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
pub mod matrixaca;
pub mod matrixlu;
pub mod matrixluci;
pub mod scalar;
pub mod traits;
pub mod util;

// Re-export main types
pub use error::{MatrixCIError, Result};
pub use matrixaca::MatrixACA;
pub use matrixlu::{rrlu, rrlu_inplace, RrLU, RrLUOptions};
pub use matrixluci::MatrixLUCI;
pub use scalar::Scalar;
pub use traits::AbstractMatrixCI;
pub use util::{from_vec2d, Matrix};
