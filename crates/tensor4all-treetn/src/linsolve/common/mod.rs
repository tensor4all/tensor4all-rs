//! Common infrastructure for linear equation solvers.
//!
//! This module provides shared utilities used by both square (V_in = V_out)
//! and general (V_in ≠ V_out) linear equation solvers.

mod environment;
mod local_solver;
mod options;
mod projected_operator;

pub(crate) use environment::CachedTopology;
pub use environment::{EnvironmentCache, NetworkTopology};
pub(crate) use local_solver::solve_local_system;
pub use local_solver::LocalSolver;
pub use options::{
    GmresToleranceMode, LinsolveOptions, ResidualEnrichment, ResidualEnrichmentRangeError,
};
pub use projected_operator::ProjectedOperator;
