//! Error types for MPO contraction operations

use thiserror::Error;

/// Result type for MPO operations
pub type Result<T> = std::result::Result<T, MPOError>;

/// Errors that can occur during MPO operations
#[derive(Error, Debug)]
pub enum MPOError {
    /// Dimension mismatch between tensors
    #[error("Dimension mismatch: tensor at site {site} has incompatible dimensions")]
    DimensionMismatch { site: usize },

    /// Bond dimension mismatch between adjacent tensors
    #[error("Bond dimension mismatch at site {site}: left tensor has right_dim={left_right}, right tensor has left_dim={right_left}")]
    BondDimensionMismatch {
        site: usize,
        left_right: usize,
        right_left: usize,
    },

    /// Shared dimension mismatch between two MPOs
    #[error("Shared dimension mismatch at site {site}: MPO A has site_dim_2={dim_a}, MPO B has site_dim_1={dim_b}")]
    SharedDimensionMismatch {
        site: usize,
        dim_a: usize,
        dim_b: usize,
    },

    /// Length mismatch between two MPOs
    #[error("MPO length mismatch: expected {expected}, got {got}")]
    LengthMismatch { expected: usize, got: usize },

    /// Invalid index provided
    #[error("Index out of bounds: index {index} at site {site} (max: {max})")]
    IndexOutOfBounds { site: usize, index: usize, max: usize },

    /// Empty MPO
    #[error("MPO is empty")]
    Empty,

    /// Invalid boundary conditions
    #[error("Invalid boundary conditions: first tensor must have left_dim=1, last tensor must have right_dim=1")]
    InvalidBoundary,

    /// Invalid orthogonality center
    #[error("Invalid orthogonality center: {center} is out of range [0, {max})")]
    InvalidCenter { center: usize, max: usize },

    /// Factorization error
    #[error("Factorization failed: {message}")]
    FactorizationError { message: String },

    /// Invalid operation
    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },

    /// Convergence failure
    #[error("Failed to converge after {sweeps} sweeps (final error: {error})")]
    ConvergenceFailure { sweeps: usize, error: f64 },
}
