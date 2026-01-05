//! MPO (Matrix Product Operator) contraction algorithms
//!
//! This module provides algorithms for contracting two Matrix Product Operators (MPOs),
//! which are tensor trains with 4D site tensors of shape (left_bond, site_dim_1, site_dim_2, right_bond).
//!
//! # Main Types
//!
//! - [`MPO`]: Basic Matrix Product Operator representation
//! - [`SiteMPO`]: Center-canonical form with orthogonality center
//! - [`VidalMPO`]: Vidal canonical form with explicit singular values
//! - [`InverseMPO`]: Inverse form for efficient local updates
//!
//! # Contraction Algorithms
//!
//! - [`contract_naive`]: Naive contraction (exact but memory-intensive)
//! - [`contract_zipup`]: Zip-up contraction with on-the-fly compression
//! - [`contract_fit`]: Variational fitting algorithm
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_simpletensortrain::mpo::{MPO, contract_naive, ContractionOptions};
//!
//! let mpo_a = MPO::constant(&[2, 2], 1.0);
//! let mpo_b = MPO::constant(&[2, 2], 2.0);
//! let result = contract_naive(&mpo_a, &mpo_b, None)?;
//! ```

pub mod error;
pub mod types;
pub mod mpo;
pub mod site_mpo;
pub mod vidal_mpo;
pub mod inverse_mpo;
pub mod factorize;
pub mod environment;
pub mod contraction;
pub mod contract_naive;
pub mod contract_zipup;
pub mod contract_fit;
pub mod dispatch;
pub mod tt_contraction;

// Re-export main types and functions
pub use error::{MPOError, Result};
pub use types::{Tensor4, Tensor4Ops, tensor4_zeros, tensor4_from_data};
pub use mpo::MPO;
pub use site_mpo::SiteMPO;
pub use vidal_mpo::VidalMPO;
pub use inverse_mpo::InverseMPO;
pub use factorize::{factorize, FactorizeMethod, FactorizeOptions, FactorizeResult};
pub use contraction::{Contraction, ContractionOptions};
pub use contract_naive::contract_naive;
pub use contract_zipup::contract_zipup;
pub use contract_fit::{contract_fit, FitOptions};
pub use dispatch::contract;
