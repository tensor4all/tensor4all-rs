//! Error types for tensor train operations

use thiserror::Error;

/// Result type for tensor train operations
pub type Result<T> = std::result::Result<T, TensorTrainError>;

/// Errors that can occur during tensor train operations
#[derive(Error, Debug)]
pub enum TensorTrainError {
    /// Dimension mismatch between tensors
    #[error("Dimension mismatch: tensor at site {site} has incompatible dimensions")]
    DimensionMismatch {
        /// The site index where the mismatch occurred
        site: usize,
    },

    /// Invalid index provided
    #[error("Index out of bounds: index {index} at site {site} (max: {max})")]
    IndexOutOfBounds {
        /// The site index where the error occurred
        site: usize,
        /// The invalid index value
        index: usize,
        /// The maximum allowed index value
        max: usize,
    },

    /// Length mismatch in index set
    #[error("Index set length mismatch: expected {expected}, got {got}")]
    IndexLengthMismatch {
        /// The expected length
        expected: usize,
        /// The actual length provided
        got: usize,
    },

    /// Empty tensor train
    #[error("Tensor train is empty")]
    Empty,

    /// Invalid operation
    #[error("Invalid operation: {message}")]
    InvalidOperation {
        /// Description of the invalid operation
        message: String,
    },
}
