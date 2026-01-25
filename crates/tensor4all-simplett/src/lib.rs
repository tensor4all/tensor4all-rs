#![warn(missing_docs)]
//! Tensor Train (MPS) library
//!
//! This crate provides tensor train (also known as Matrix Product State) algorithms,
//! including:
//! - `TensorTrain`: The main tensor train structure
//! - `SiteTensorTrain`: Center-canonical form
//! - `VidalTensorTrain`: Vidal canonical form with explicit singular values
//! - `InverseTensorTrain`: Inverse form for efficient local updates
//! - Compression algorithms (LU, CI, SVD)
//! - Arithmetic operations (add, subtract, scale)
//!
//! # Example
//!
//! ```
//! use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};
//!
//! // Create a constant tensor train
//! let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
//!
//! // Evaluate at a specific index
//! let value = tt.evaluate(&[0, 1, 1]).unwrap();
//! println!("Value: {}", value);
//!
//! // Sum over all indices
//! let sum = tt.sum();
//! println!("Sum: {}", sum);
//! ```

pub mod arithmetic;
pub mod cache;
pub mod canonical;
pub mod compression;
pub mod contraction;
pub mod error;
pub mod mpo;
pub mod tensortrain;
pub mod traits;
pub mod types;
pub mod vidal;

// Re-export main types
pub use cache::TTCache;
pub use canonical::{center_canonicalize, SiteTensorTrain};
pub use compression::{CompressionMethod, CompressionOptions};
pub use contraction::{dot, ContractionOptions};
pub use error::{Result, TensorTrainError};
pub use tensortrain::TensorTrain;
pub use traits::{AbstractTensorTrain, TTScalar};
pub use types::{tensor3_from_data, tensor3_zeros, LocalIndex, MultiIndex, Tensor3, Tensor3Ops};
pub use vidal::{DiagMatrix, InverseTensorTrain, VidalTensorTrain};
