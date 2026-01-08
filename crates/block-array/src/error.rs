//! Error types for blocked-mdarray operations.

use thiserror::Error;

/// Error type for blocked array operations.
#[derive(Debug, Error)]
pub enum BlockArrayError {
    /// Block partitions are incompatible for the operation.
    #[error("Incompatible block partitions for operation")]
    IncompatiblePartitions,

    /// Block index is out of bounds.
    #[error("Block index {index:?} out of bounds for shape {shape:?}")]
    BlockIndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },

    /// Shape mismatch in operation.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Operation requires a 2D array (matrix).
    #[error("Operation requires 2D array, got {0}D")]
    NotMatrix(usize),

    /// Invalid permutation.
    #[error("Invalid permutation: {0}")]
    InvalidPermutation(String),
}

/// Result type for blocked array operations.
pub type Result<T> = std::result::Result<T, BlockArrayError>;
