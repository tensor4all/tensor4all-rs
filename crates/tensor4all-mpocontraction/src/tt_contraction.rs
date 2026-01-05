//! Contraction operations for tensor trains (TT-TT)
//!
//! This module re-exports TT contraction operations from `tensor4all-tensortrain`.
//! These operations are conceptually MPO-MPO contractions where the MPO
//! has trivial (dimension 1) "operator" indices.
//!
//! # Available operations
//!
//! - [`hadamard`]: Element-wise (Hadamard) product
//! - [`dot`]: Inner product (returns scalar)
//! - [`hadamard_zipup`]: Hadamard product with on-the-fly compression
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_mpocontraction::tt_contraction::{hadamard, dot, ContractionOptions};
//! use tensor4all_tensortrain::TensorTrain;
//!
//! let tt1 = TensorTrain::<f64>::constant(&[2, 3], 2.0);
//! let tt2 = TensorTrain::<f64>::constant(&[2, 3], 3.0);
//!
//! // Element-wise product
//! let result = hadamard(&tt1, &tt2).unwrap();
//!
//! // Inner product
//! let inner = dot(&tt1, &tt2).unwrap();
//! ```

// Re-export TT contraction types and functions from tensor4all-tensortrain
pub use tensor4all_tensortrain::contraction::{
    dot, hadamard, hadamard_zipup, ContractionOptions,
};

// Also re-export TensorTrain for convenience
pub use tensor4all_tensortrain::TensorTrain;
