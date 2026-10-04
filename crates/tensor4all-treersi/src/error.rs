//! Typed errors for tree recursive sketched interpolation.

use tensor4all_treetn::TreeTNOperationError;
use thiserror::Error;

/// A result returned by tree RSI operations.
///
/// Related types: [`TreeRsiError`] classifies validation, resource, numerical,
/// and tensor-network failures.
///
/// # Examples
///
/// ```
/// use tensor4all_treersi::{Result, TreeRsiError};
///
/// fn reject() -> Result<()> {
///     Err(TreeRsiError::NoInputs)
/// }
/// assert!(matches!(reject(), Err(TreeRsiError::NoInputs)));
/// ```
pub type Result<T> = std::result::Result<T, TreeRsiError>;

/// An error reported while validating or running tree RSI.
///
/// Invalid inputs, options and probes are reported before sketch contraction;
/// resource limits are checked before the corresponding allocation.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treersi::{hadamard_many, TreeRsiError, TreeRsiOptions};
/// use tensor4all_treetn::TreeTN;
///
/// let site = DynIndex::new_dyn(2);
/// let tree = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![site], vec![1.0_f64, 2.0])?],
///     vec![0usize],
/// )?;
/// let options = TreeRsiOptions {
///     max_bond_dim: Some(0),
///     ..TreeRsiOptions::default()
/// };
/// let error = hadamard_many::<f64, _>(&[tree], &options).unwrap_err();
/// assert!(matches!(
///     error,
///     TreeRsiError::InvalidOption { option: "max_bond_dim", .. }
/// ));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TreeRsiError {
    /// No input tensor network was supplied.
    #[error("tree RSI requires at least one input tensor network")]
    NoInputs,

    /// An input tensor network contained no nodes.
    #[error("tree RSI input {input} has no nodes")]
    EmptyTree {
        /// Zero-based input position.
        input: usize,
    },

    /// An input's labeled topology differs from input 0.
    #[error("tree RSI input {input} has a different labeled topology than input 0")]
    TopologyMismatch {
        /// Zero-based input position.
        input: usize,
    },

    /// An input's physical indices at a node differ from input 0.
    #[error("tree RSI input {input} has different physical indices at node {node}")]
    PhysicalIndexMismatch {
        /// Zero-based input position.
        input: usize,
        /// Debug rendering of the node name.
        node: String,
    },

    /// An option value is outside its documented domain.
    #[error("invalid tree RSI option `{option}`: {message}")]
    InvalidOption {
        /// Option field name.
        option: &'static str,
        /// Explanation of the constraint.
        message: &'static str,
    },

    /// Explicit probes are missing or have the wrong shape.
    #[error("invalid tree RSI probes: {message}")]
    InvalidProbes {
        /// Description of the missing or malformed probe.
        message: String,
    },

    /// A requested allocation exceeds the configured element limit.
    #[error("tree RSI {resource} needs {requested} elements, above the limit {limit}")]
    ResourceLimit {
        /// Resource being allocated.
        resource: &'static str,
        /// Requested number of scalar elements.
        requested: usize,
        /// Configured limit.
        limit: usize,
    },

    /// A size computation overflowed `usize`.
    #[error("tree RSI size computation overflowed: {context}")]
    SizeOverflow {
        /// Computation that overflowed.
        context: &'static str,
    },

    /// A local matrix contained NaN or an infinity.
    #[error("tree RSI produced a nonfinite value at node {node} during {stage}")]
    NonFiniteValue {
        /// Debug rendering of the node name.
        node: String,
        /// Algorithm stage that produced the value.
        stage: &'static str,
    },

    /// A nonzero value cannot survive the scalar type's dynamic range.
    #[error("tree RSI cannot represent a nonzero value during {stage}; use a wider scalar type or rescale the inputs")]
    DynamicRange {
        /// Numerical stage that lost range.
        stage: &'static str,
    },

    /// A tree bond has zero dimension.
    #[error("tree RSI encountered a zero-dimensional bond at node {node}")]
    ZeroBond {
        /// Node adjacent to the invalid bond.
        node: String,
    },

    /// An input tensor belongs to a different execution context.
    #[error("tree RSI input context mismatch: {0}")]
    InputContext(#[source] TreeTNOperationError),

    /// An input core could not be read with the requested scalar type.
    #[error("tree RSI scalar kind mismatch: {message}")]
    ScalarKind {
        /// Underlying conversion message.
        message: String,
    },

    /// The execution context cannot run tree RSI, or an input does not belong
    /// to it.
    #[error("tree RSI execution context error: {message}")]
    Context {
        /// Which input or context was rejected and why.
        message: String,
    },

    /// A dense tensor, contraction, or factorization failed.
    #[error("tree RSI numerical operation failed: {source}")]
    Numerical {
        /// Original failure, retained for inspection.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A tree tensor-network operation failed.
    #[error(transparent)]
    Tree(#[from] TreeTNOperationError),

    /// An internal consistency check failed.
    #[error("tree RSI internal invariant violated: {message}")]
    InternalInvariant {
        /// Description of the violated invariant.
        message: &'static str,
    },
}

impl TreeRsiError {
    pub(crate) fn numerical(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        Self::Numerical {
            source: error.into(),
        }
    }
}
