//! Linear equation solvers for Tree Tensor Networks.
//!
//! This module provides solvers for linear systems of the form
//! `(a₀ + a₁ * A) * x = b` where A is a TTN operator and x, b are TTN states.
//!
//! # Modules
//!
//! - [`common`]: Shared infrastructure (environment cache, options, projected operator)
//! - [`square`]: Solver for V_in = V_out case (input and output spaces are the same)
//!
//! # Usage
//!
//! For most use cases, use the `square` solver:
//!
//! ```ignore
//! use tensor4all_treetn::linsolve::square::{square_linsolve, LinsolveOptions};
//!
//! let result = square_linsolve(&operator, &rhs, init, &center, LinsolveOptions::default())?;
//! ```

pub mod common;
pub mod square;

// Re-export commonly used types
pub use common::{EnvironmentCache, LinsolveOptions, NetworkTopology, ProjectedOperator};
pub use square::{
    square_linsolve, LinsolveVerifyReport, NodeVerifyDetail, ProjectedState, SquareLinsolveResult,
    SquareLinsolveUpdater,
};
