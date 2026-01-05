//! ITensorMPS.jl-inspired Tensor Train library with orthogonality tracking
//!
//! This crate provides a Rust implementation of Tensor Trains (also known as MPS)
//! with an API inspired by [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl).
//!
//! # Features
//!
//! - `ITensorTrain`: Main tensor train type with orthogonality tracking
//! - Canonicalization with SVD, LU, or CI algorithms
//! - Truncation with configurable tolerance and max rank
//! - Norm and inner product computations
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_itensortrain::{ITensorTrain, TruncateOptions};
//!
//! // Create a tensor train from tensors
//! let tt = ITensorTrain::new(tensors)?;
//!
//! // Orthogonalize to site 2
//! tt.orthogonalize(2)?;
//!
//! // Truncate with SVD
//! tt.truncate(&TruncateOptions::svd().with_rtol(1e-10))?;
//!
//! // Compute norm
//! let n = tt.norm();
//! ```
//!
//! # Differences from ITensorMPS.jl
//!
//! - Uses 0-indexed sites (Julia uses 1-indexed)
//! - Supports multiple canonicalization algorithms (SVD, LU, CI)
//! - Uses `conj` instead of `dag` (no index direction flipping without QN support)
//! - Each site can have multiple site indices (not just one physical index per site)

pub mod error;
pub mod options;
pub mod tensortrain;

pub use error::{ITensorTrainError, Result};
pub use options::{CanonicalMethod, TruncateAlg, TruncateOptions};
pub use tensortrain::ITensorTrain;
