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
pub use local_solver::LocalSolver;
pub(crate) use local_solver::solve_local_system;
pub use options::{GmresToleranceMode, LinsolveOptions};
pub use projected_operator::ProjectedOperator;
