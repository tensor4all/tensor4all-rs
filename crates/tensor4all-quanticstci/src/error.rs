//! Error types for quantics tensor cross interpolation operations.

use thiserror::Error;

/// Result type for quantics TCI operations.
pub type Result<T> = std::result::Result<T, QuanticsTCIError>;

/// Error returned by quantics TCI construction, evaluation, and grid operations.
///
/// # Remedies
/// - Grid/quantics-format failures: validate grid parameters (bounds, number
///   of bits, physical dimension) before construction.
/// - Tolerance/termination failures: relax `tolerance`/`max_bond_dim` or
///   increase sweep limits when convergence is not reached.
/// - Backend failures: the wrapped source chain identifies the failing stage.
#[derive(Debug, Error)]
pub enum QuanticsTCIError {
    /// Invalid quantics grid or interpolation configuration.
    #[error("invalid quantics configuration: {message}")]
    InvalidConfiguration {
        /// Description of the invalid configuration.
        message: String,
    },
    /// The requested operation requires a discretized (continuous) grid.
    #[error("original coordinates are only available for discretized grids")]
    DiscreteGridRequired,
    /// An underlying evaluation, coordinate conversion, or backend operation
    /// failed.
    #[error("quantics TCI operation failed: {source}")]
    Operation {
        /// Original diagnostic, preserving the full source chain.
        #[source]
        source: anyhow::Error,
    },
}

impl From<anyhow::Error> for QuanticsTCIError {
    fn from(source: anyhow::Error) -> Self {
        Self::Operation { source }
    }
}

impl From<tensor4all_treetci::TreeTciError> for QuanticsTCIError {
    fn from(source: tensor4all_treetci::TreeTciError) -> Self {
        Self::Operation {
            source: anyhow::Error::new(source),
        }
    }
}

#[cfg(test)]
mod tests;
