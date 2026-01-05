//! Contraction operations for tensor trains (TT-TT)
//!
//! This module re-exports TT contraction operations from `tensor4all-simpletensortrain`.
//! These operations are conceptually MPO-MPO contractions where the MPO
//! has trivial (dimension 1) "operator" indices.
//!
//! # Available operations
//!
//! - [`dot`]: Inner product (returns scalar)
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_mpocontraction::tt_contraction::{dot, ContractionOptions};
//! use crate::TensorTrain;
//!
//! let tt1 = TensorTrain::<f64>::constant(&[2, 3], 2.0);
//! let tt2 = TensorTrain::<f64>::constant(&[2, 3], 3.0);
//!
//! // Inner product
//! let inner = dot(&tt1, &tt2).unwrap();
//! ```

// Re-export TT contraction types and functions from tensor4all-simpletensortrain
pub use crate::contraction::{dot, ContractionOptions};

// Also re-export TensorTrain for convenience
pub use crate::TensorTrain;
