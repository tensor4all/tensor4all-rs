//! Error types for TensorTrain operations.

use std::sync::Arc;

use thiserror::Error;

/// Result type for TensorTrain operations.
pub type Result<T> = std::result::Result<T, TensorTrainError>;

use tensor4all_core::IdxTensorError;
use tensor4all_treetn::TreeTNOperationError;

/// Errors that can occur in [`TensorTrain`](crate::TensorTrain) operations.
#[derive(Debug, Error)]
pub enum TensorTrainError {
    /// Tensor train is empty (has no tensors).
    #[error("Tensor train is empty")]
    Empty,

    /// Site index is out of bounds.
    #[error("Site index {site} is out of bounds (tensor train has {length} sites)")]
    SiteOutOfBounds {
        /// The requested site index.
        site: usize,
        /// The total number of sites in the tensor train.
        length: usize,
    },

    /// Bond shape mismatch between adjacent tensors.
    #[error("Bond shape mismatch at site {site}: left tensor has right dim {left_dim}, right tensor has left dim {right_dim}")]
    BondDimensionMismatch {
        /// The site index where the mismatch occurred.
        site: usize,
        /// The right bond dimension of the left tensor.
        left_dim: usize,
        /// The left bond dimension of the right tensor.
        right_dim: usize,
    },

    /// Tensor train does not have a well-defined orthogonality center.
    #[error("Tensor train does not have a well-defined orthogonality center (ortho_lims = {start}..{end})")]
    NoOrthogonalityCenter {
        /// The start of the orthogonality limits range.
        start: usize,
        /// The end of the orthogonality limits range.
        end: usize,
    },

    /// Invalid tensor structure for tensor train.
    #[error("Invalid tensor structure: {message}")]
    InvalidStructure {
        /// A description of the structural issue.
        message: String,
    },

    /// Factorization error.
    #[error("Factorization error: {0}")]
    Factorize(#[from] tensor4all_core::FactorizeError),

    /// A typed IdxTensor metric or materialization error.
    #[error("IdxTensor operation error: {source}")]
    IdxTensor {
        /// The typed tensor diagnostic.
        #[source]
        source: tensor4all_core::IdxTensorError,
    },

    /// A TreeTN operation failed while building or updating this tensor train.
    #[error("{context}: {source}")]
    TreeTn {
        /// Context describing the TensorTrain operation that failed.
        context: String,
        /// Original TreeTN diagnostic, preserved as the error source.
        #[source]
        source: TreeTNOperationError,
    },

    /// General operation error.
    #[error("Operation error: {message}")]
    OperationError {
        /// A description of the operation error.
        message: String,
    },

    /// An operation error retaining an owned backend source diagnostic.
    #[error("Operation error: {message}: {source}")]
    OperationErrorSource {
        /// Context describing the failed operation.
        message: String,
        /// Original operation diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

impl From<anyhow::Error> for TensorTrainError {
    fn from(source: anyhow::Error) -> Self {
        Self::operation_source("TensorTrain operation failed", source)
    }
}

impl From<IdxTensorError> for TensorTrainError {
    fn from(source: IdxTensorError) -> Self {
        Self::IdxTensor { source }
    }
}

impl TensorTrainError {
    /// Wrap a TreeTN diagnostic with TensorTrain-level context.
    pub(crate) fn tree_tn(context: impl Into<String>, source: TreeTNOperationError) -> Self {
        Self::TreeTn {
            context: context.into(),
            source,
        }
    }

    pub(crate) fn operation_source(message: impl Into<String>, source: anyhow::Error) -> Self {
        Self::OperationErrorSource {
            message: message.into(),
            source: Arc::from(source.into_boxed_dyn_error()),
        }
    }
}
