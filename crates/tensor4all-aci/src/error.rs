//! Error types for Alternating Cross Interpolation.

use thiserror::Error;

/// Result type for Alternating Cross Interpolation operations.
pub type Result<T> = std::result::Result<T, AciError>;

/// Errors that can occur while constructing or running ACI operations.
///
/// Use this error type when validating ACI inputs, options, initial tensor-train
/// guesses, and operator callbacks. Errors from the lower-level matrix-CI and
/// tensor-train crates are preserved through the corresponding variants.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::AciError;
///
/// let err = AciError::LengthMismatch {
///     expected: 6,
///     got: 5,
/// };
/// assert!(err.to_string().contains("length"));
///
/// let err = AciError::Operator {
///     message: "operator failed at point 3".to_string(),
/// };
/// assert!(err.to_string().contains("operator failed"));
///
/// let err = AciError::BatchIndexOutOfBounds {
///     axis: "input",
///     index: 4,
///     len: 3,
/// };
/// assert!(err.to_string().contains("input index out of bounds"));
///
/// let err = AciError::Backend {
///     operation: "left frame update at input 0, site 1".to_string(),
///     source: "engine refused the product".into(),
/// };
/// assert!(err.to_string().contains("left frame update at input 0, site 1"));
/// assert_eq!(
///     std::error::Error::source(&err).map(ToString::to_string).as_deref(),
///     Some("engine refused the product")
/// );
/// ```
#[derive(Debug, Error)]
pub enum AciError {
    /// No input tensors or input points were provided.
    #[error("ACI requires at least one input and one interpolation point")]
    EmptyInputs,

    /// A flat buffer length did not match the shape implied by the caller.
    #[error("ACI length mismatch: expected {expected} values, got {got}")]
    LengthMismatch {
        /// Number of values implied by the requested shape.
        expected: usize,
        /// Number of values supplied by the caller.
        got: usize,
    },

    /// A tensor-train site dimension did not match the expected value.
    #[error("ACI site shape mismatch at site {site}: expected {expected}, got {got}")]
    SiteDimMismatch {
        /// Zero-based site where the mismatch occurred.
        site: usize,
        /// Expected site dimension.
        expected: usize,
        /// Actual site dimension.
        got: usize,
    },

    /// A batch input or point index was outside the corresponding axis length.
    #[error("ACI batch {axis} index out of bounds: index {index}, len {len}")]
    BatchIndexOutOfBounds {
        /// Batch axis name, such as `"input"` or `"point"`.
        axis: &'static str,
        /// Zero-based index requested by the caller.
        index: usize,
        /// Number of valid entries along the axis.
        len: usize,
    },

    /// A configuration value or combination of values is invalid.
    #[error("Invalid ACI options: {message}")]
    InvalidOptions {
        /// Description of the invalid option value or relationship.
        message: String,
    },

    /// The supplied initial tensor-train guess is incompatible with the ACI problem.
    #[error("Invalid ACI initial guess: {message}")]
    InvalidInitialGuess {
        /// Description of the incompatible initial guess.
        message: String,
    },

    /// The execution backend could not carry out a matrix product of the sweep.
    ///
    /// Raised when the CPU execution context refuses a session (for example a
    /// reentrant entry, or a build without a process-global context reached
    /// through a context-free entry point), when the engine rejects the
    /// product, or when the product's operand shapes disagree. It reports a
    /// backend or shape failure, not a problem with the initial guess; that one
    /// is [`AciError::InvalidInitialGuess`]. The original error is kept as
    /// `source`.
    #[error("ACI backend failure in {operation}: {source}")]
    Backend {
        /// The product that failed, such as `"left frame update at input 0, site 1"`.
        operation: String,
        /// The backend, session or shape error that stopped it.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// An elementwise operator returned a buffer with the wrong length.
    #[error("ACI operator output length mismatch: expected {expected}, got {got}")]
    OperatorOutputLength {
        /// Expected number of output values.
        expected: usize,
        /// Number of output values returned by the operator.
        got: usize,
    },

    /// An elementwise operator reported an error.
    #[error("ACI operator error: {message}")]
    Operator {
        /// Operator-provided error message.
        message: String,
    },

    /// Error propagated from matrix cross interpolation.
    #[error("Matrix CI error: {0}")]
    MatrixCI(#[from] tensor4all_core::MatrixCIError),

    /// Error propagated from tensor-train construction or manipulation.
    #[error("Tensor train error: {0}")]
    SimpleTensorTrain(#[from] tensor4all_simplett::SimpleTensorTrainError),
}
