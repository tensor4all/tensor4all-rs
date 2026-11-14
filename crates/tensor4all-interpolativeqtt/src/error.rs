//! Error types for interpolative QTT construction.

use thiserror::Error;

/// Result type used by interpolative QTT APIs.
pub type Result<T> = std::result::Result<T, InterpolativeQttError>;

/// Error type for interpolative QTT construction.
///
/// # Examples
///
/// ```
/// use tensor4all_interpolativeqtt::{get_chebyshev_grid, InterpolativeQttError};
///
/// let err = get_chebyshev_grid(0).unwrap_err();
/// assert!(matches!(err, InterpolativeQttError::InvalidArgument { .. }));
/// ```
#[derive(Debug, Error)]
pub enum InterpolativeQttError {
    /// The caller supplied an invalid argument.
    #[error("invalid argument: {message}")]
    InvalidArgument {
        /// Human-readable explanation of the invalid argument.
        message: String,
    },

    /// The evaluator failed; its original error is retained as the source.
    #[error("patch evaluator failed: {source}")]
    Evaluation {
        /// Original evaluator error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A batch evaluator returned the wrong number of values.
    #[error("patch evaluator returned {actual} values for {expected} points")]
    BatchLength {
        /// Requested point count.
        expected: usize,
        /// Returned value count.
        actual: usize,
    },

    /// A function value or interpolation residual was non-finite.
    #[error("patch evaluation or residual is non-finite")]
    NonFinite,

    /// A box cannot be split further within the configured limits.
    #[error("patch at {prefix:?} did not converge: error {error_estimate}, rank {rank}; increase max_depth or max_bond_dim")]
    NotConverged {
        /// Fixed leading bits per axis.
        prefix: Vec<Vec<usize>>,
        /// Maximum absolute interpolation residual at the validation points.
        error_estimate: f64,
        /// Largest bond dimension.
        rank: usize,
    },

    /// Splitting would exceed the configured patch budget.
    #[error("patch budget {limit} exceeded; increase max_patches")]
    PatchLimit {
        /// Configured maximum number of leaf boxes.
        limit: usize,
    },

    /// A supplied execution engine refused an evaluator call.
    #[error("patch execution failed: {source}")]
    Execution {
        /// Original backend error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// An interpolation matrix product failed.
    #[error(transparent)]
    MatrixMultiply(#[from] tensor4all_tensorbackend::MatrixMulError),

    /// The lane executor could not run the wave.
    #[cfg(feature = "parallel")]
    #[error("patch lane execution failed: {0}")]
    FanOut(#[source] hataori::MapInError),

    /// Underlying tensor train operation failed.
    #[error(transparent)]
    SimpleTensorTrain(#[from] tensor4all_simplett::SimpleTensorTrainError),
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> InterpolativeQttError {
    InterpolativeQttError::InvalidArgument {
        message: message.into(),
    }
}
