//! Error types for tree tensor cross interpolation operations.

use thiserror::Error;

/// Result type for tree-TCI operations.
pub type Result<T> = std::result::Result<T, TreeTciError>;

/// Error returned by tree tensor cross interpolation construction and
/// traversal operations.
///
/// # Remedies
/// - Tree-shape or topology failures: validate the tree structure (leaf
///   counts, ranks) before construction.
/// - Tolerance/termination failures: relax `rtol`/`max_bond_dim` or increase
///   the number of sweeps when convergence is not reached.
/// - Backend failures: the wrapped source chain identifies the failing stage.
#[derive(Debug, Error)]
pub enum TreeTciError {
    /// Invalid optimization configuration.
    #[error("invalid tree-TCI configuration: {message}")]
    InvalidConfiguration {
        /// Description of the invalid option value.
        message: String,
    },
    /// The tree graph is invalid (disconnected, missing vertices, or
    /// inconsistent edge structure).
    #[error("invalid tree-TCI graph: {message}")]
    InvalidGraph {
        /// Description of the invalid graph structure.
        message: String,
    },
    /// An index or batch coordinate is out of bounds.
    #[error("index out of bounds: {message}")]
    IndexOutOfBounds {
        /// Description of the invalid index.
        message: String,
    },
    /// An underlying construction, materialization, or backend operation
    /// failed.
    #[error("tree-TCI operation failed: {source}")]
    Operation {
        /// Original diagnostic, preserving the full source chain.
        #[source]
        source: anyhow::Error,
    },
}

impl From<anyhow::Error> for TreeTciError {
    fn from(source: anyhow::Error) -> Self {
        Self::Operation { source }
    }
}

impl From<tensor4all_core::IdxTensorError> for TreeTciError {
    fn from(source: tensor4all_core::IdxTensorError) -> Self {
        Self::Operation {
            source: anyhow::Error::new(source),
        }
    }
}

impl From<tensor4all_treetn::TreeTNOperationError> for TreeTciError {
    fn from(source: tensor4all_treetn::TreeTNOperationError) -> Self {
        Self::Operation {
            source: anyhow::Error::new(source),
        }
    }
}

#[cfg(test)]
mod tests;
