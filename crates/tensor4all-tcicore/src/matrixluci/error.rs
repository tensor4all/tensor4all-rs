//! Error types for matrixluci.

use thiserror::Error;

/// Errors that can occur during matrix LUCI operations.
#[derive(Debug, Error)]
pub enum MatrixLuciError {
    /// Invalid argument.
    #[error("Invalid argument: {message}")]
    InvalidArgument {
        /// Description of the invalid argument.
        message: String,
    },
    /// Singular pivot block encountered during lazy residual evaluation.
    #[error("Singular pivot block encountered")]
    SingularPivotBlock,
}

/// Result type for matrixluci operations.
pub type Result<T> = std::result::Result<T, MatrixLuciError>;
