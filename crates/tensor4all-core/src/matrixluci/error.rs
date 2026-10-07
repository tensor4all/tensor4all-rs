//! Error types for matrixluci.

use thiserror::Error;

/// Errors that can occur during matrix LUCI operations.
#[derive(Debug, Error)]
pub(crate) enum MatrixLuciError {
    /// Non-finite matrix input or residual.
    #[error("Non-finite values in {matrix}")]
    NaNEncountered { matrix: &'static str },
    /// Invalid argument.
    #[error("Invalid argument: {message}")]
    InvalidArgument {
        /// Description of the invalid argument.
        message: String,
    },
}

/// Result type for matrixluci operations.
pub(crate) type Result<T> = std::result::Result<T, MatrixLuciError>;
