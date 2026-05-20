//! Common infrastructure for linear equation solvers.
//!
//! This module provides shared utilities used by both square (V_in = V_out)
//! and general (V_in ≠ V_out) linear equation solvers.

mod environment;
mod options;
mod projected_operator;

pub use environment::{EnvironmentCache, NetworkTopology};
pub use options::{GmresToleranceMode, LinsolveOptions};
pub use projected_operator::ProjectedOperator;
