//! Public error types for tensor4all-treetn APIs.

use thiserror::Error;

/// Error returned when selecting external indices by numbered tags fails.
///
/// Use this for APIs that resolve tags such as `"k=1"`, `"k=2"`, ... into a
/// concrete ordered index list.
///
/// # Examples
/// ```
/// use tensor4all_treetn::NumberedTagSelectionError;
///
/// let err = NumberedTagSelectionError::MissingTag {
///     tag: "k=3".to_string(),
/// };
/// assert_eq!(err.to_string(), "no external index with tag k=3");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NumberedTagSelectionError {
    /// The tag prefix is malformed for numbered-tag lookup.
    #[error("invalid numbered tag prefix {tag_prefix}: prefix must not contain '='")]
    InvalidPrefix {
        /// Prefix provided by the caller.
        tag_prefix: String,
    },
    /// The requested numeric range overflows `usize`.
    #[error("numbered tag range overflows usize at offset {offset} from start {start_index}")]
    RangeOverflow {
        /// First requested numeric suffix.
        start_index: usize,
        /// Offset from `start_index` that overflowed.
        offset: usize,
    },
    /// A required numbered tag was not present on any external index.
    #[error("no external index with tag {tag}")]
    MissingTag {
        /// Missing tag, including the numeric suffix.
        tag: String,
    },
    /// A required numbered tag matched more than one external index.
    #[error("found more than one external index with tag {tag}")]
    AmbiguousTag {
        /// Ambiguous tag, including the numeric suffix.
        tag: String,
    },
}

/// Error returned when rebinding a [`LinearOperator`](crate::LinearOperator)'s
/// true input or output indices fails.
///
/// The error reports whether the failed binding belonged to the input or output
/// map through the `role` field.
///
/// # Examples
/// ```
/// use tensor4all_treetn::LinearOperatorIndexBindingError;
///
/// let err = LinearOperatorIndexBindingError::DimensionMismatch {
///     role: "input",
///     old_dim: 2,
///     new_dim: 3,
/// };
/// assert!(err.to_string().contains("input index shape mismatch"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LinearOperatorIndexBindingError {
    /// A replacement would change an index dimension.
    #[error("{role} index shape mismatch: old dimension {old_dim} vs new dimension {new_dim}")]
    DimensionMismatch {
        /// Binding role, currently `"input"` or `"output"`.
        role: &'static str,
        /// Dimension of the source true index.
        old_dim: usize,
        /// Dimension of the requested replacement index.
        new_dim: usize,
    },
    /// The same source index was named more than once.
    #[error("duplicate {role} index binding for {index}")]
    DuplicateSourceIndex {
        /// Binding role, currently `"input"` or `"output"`.
        role: &'static str,
        /// Debug representation of the duplicate source index.
        index: String,
    },
    /// The source index does not appear in the corresponding mapping.
    #[error("{role} index {index} not found in operator mappings")]
    MissingSourceIndex {
        /// Binding role, currently `"input"` or `"output"`.
        role: &'static str,
        /// Debug representation of the missing source index.
        index: String,
    },
}

/// Error returned when applying a linear operator after explicit index binding
/// fails.
///
/// This separates binding failures from failures in the underlying apply
/// algorithm.
///
/// # Examples
/// ```
/// use tensor4all_treetn::{
///     LinearOperatorIndexApplyError,
///     LinearOperatorIndexBindingError,
/// };
///
/// let err = LinearOperatorIndexApplyError::BindIndices(
///     LinearOperatorIndexBindingError::MissingSourceIndex {
///         role: "input",
///         index: "i".to_string(),
///     },
/// );
/// assert!(err.to_string().contains("failed to bind operator indices"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LinearOperatorIndexApplyError {
    /// Explicit true-index rebinding failed before application.
    #[error("failed to bind operator indices: {0}")]
    BindIndices(#[from] LinearOperatorIndexBindingError),
    /// The underlying operator application failed.
    #[error("failed to apply rebound linear operator: {message}")]
    ApplyFailed {
        /// Error message from the underlying apply path, including context.
        message: String,
    },
}

/// Error returned by [`apply_linear_operator_certified`](crate::apply_linear_operator_certified)
/// and its index-binding form.
///
/// Every variant is a refusal, never a relaxation: the application does not fall
/// back to an unchecked or differently truncated run, and an unreachable budget
/// is reported instead of being widened.
///
/// # Examples
/// ```
/// use tensor4all_treetn::CertifiedApplyError;
///
/// let err = CertifiedApplyError::MemoryLimit {
///     estimated_bytes: 6 << 30,
///     limit_bytes: 4 << 30,
/// };
/// assert!(err.to_string().contains("memory limit"));
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CertifiedApplyError {
    /// The options are invalid: a budget that is negative or not finite.
    #[error("invalid certified-application options: {reason}")]
    InvalidOptions {
        /// What is wrong with the options.
        reason: String,
    },
    /// The state, or the operator on it, is not a chain. The certificate relies
    /// on the chain sweep, where the step errors are mutually orthogonal.
    #[error("certified application is unsupported here: {reason}")]
    UnsupportedTopology {
        /// Which part of the topology is unsupported.
        reason: String,
    },
    /// An estimated memory exceeds the limit: building the exact factors
    /// (checked before any factor is built), or a sweep step's working tensors
    /// next to the factors still stored (checked before that step's block is
    /// formed).
    ///
    /// # Remedy
    ///
    /// Raise [`CertifiedApplyOptions::memory_limit`](crate::CertifiedApplyOptions::memory_limit)
    /// if the machine has the memory, or apply the operator to smaller pieces of
    /// the state (finer patches), whose bonds are smaller.
    #[error(
        "certified application needs an estimated {estimated_bytes} bytes, above the memory limit \
         of {limit_bytes} bytes; raise the limit or apply the operator to finer patches"
    )]
    MemoryLimit {
        /// The estimate that exceeded the limit, in bytes.
        estimated_bytes: usize,
        /// The limit from the options.
        limit_bytes: usize,
    },
    /// Even keeping every direction the sweep can keep, the certified error of
    /// a step exceeds its share of the budget. This happens only for a budget
    /// at the level of floating-point roundoff.
    #[error(
        "error budget {budget:e} cannot be certified: step {step} is bounded by {step_bound:e} \
         at full rank, above its share {share:e}"
    )]
    BudgetUnreachable {
        /// The requested budget.
        budget: f64,
        /// Index of the sweep step, counted from the start of the sweep.
        step: usize,
        /// The smallest certified bound of that step.
        step_bound: f64,
        /// The step's share of the budget.
        share: f64,
    },
    /// Rebinding the operator's true indices failed before application.
    #[error("failed to bind operator indices: {0}")]
    BindIndices(#[from] LinearOperatorIndexBindingError),
    /// Extending the operator, transforming indices, or a tensor operation of
    /// the sweep failed.
    #[error("certified application failed: {source}")]
    Operation {
        /// The underlying failure.
        #[source]
        source: TreeTNOperationError,
    },
}

/// Error returned when applying a linear operator to state indices selected by
/// numbered tags.
///
/// This is used by [`apply_linear_operator_to_numbered_tags`](crate::apply_linear_operator_to_numbered_tags),
/// which resolves state indices tagged as `"k=1"`, `"k=2"`, ... and then uses
/// explicit index binding before applying the operator.
///
/// # Examples
/// ```
/// use tensor4all_treetn::LinearOperatorTaggedApplyError;
///
/// let err = LinearOperatorTaggedApplyError::MappingCountMismatch {
///     input_count: 2,
///     output_count: 3,
/// };
/// assert!(err.to_string().contains("input mappings"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LinearOperatorTaggedApplyError {
    /// State index selection from numbered tags failed.
    #[error("failed to select state indices by numbered tag: {0}")]
    SelectTaggedIndices(#[from] NumberedTagSelectionError),
    /// The operator does not have matching input and output mapping counts.
    #[error("operator has {input_count} input mappings but {output_count} output mappings")]
    MappingCountMismatch {
        /// Total number of true input mappings across all operator nodes.
        input_count: usize,
        /// Total number of true output mappings across all operator nodes.
        output_count: usize,
    },
    /// A selected state index lives on a node that has no operator mapping.
    #[error(
        "state index tagged {tag} on state node {node} has no operator {role} mapping; \
         operator node names must match the state node names"
    )]
    MissingNodeMapping {
        /// Mapping side that is missing: `"input"` or `"output"`.
        role: &'static str,
        /// Numbered tag that selected the state index, such as `"k=3"`.
        tag: String,
        /// Debug representation of the state node holding the selected index.
        node: String,
    },
    /// A node's selected-tag count differs from that node's mapping count.
    #[error(
        "state node {node} has {selected} selected tag(s) but {mappings} operator {role} mapping(s)"
    )]
    SelectedTagCountMismatch {
        /// Mapping side that mismatches: `"input"` or `"output"`.
        role: &'static str,
        /// Debug representation of the state/operator node.
        node: String,
        /// Number of selected state tags on the node.
        selected: usize,
        /// Number of operator mappings declared for the node.
        mappings: usize,
    },
    /// Explicit binding or application failed after tag selection.
    #[error("failed to apply operator to tagged indices: {0}")]
    ApplyToIndices(#[from] LinearOperatorIndexApplyError),
}

/// Error returned by selected-index TreeTN contraction helper APIs.
///
/// This covers helpers such as [`hadamard`](crate::hadamard),
/// [`weighted_sum_over_index_pairs`](crate::weighted_sum_over_index_pairs), and
/// [`sum_over_indices`](crate::sum_over_indices).
///
/// # Examples
/// ```
/// use tensor4all_treetn::SelectedIndexContractionError;
///
/// let err = SelectedIndexContractionError::DuplicateIndex {
///     index: "i".to_string(),
/// };
/// assert!(err.to_string().contains("duplicate index"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SelectedIndexContractionError {
    /// The same selected index was provided more than once.
    #[error("sum_over_indices: duplicate index {index}")]
    DuplicateIndex {
        /// Debug representation of the duplicated index.
        index: String,
    },
    /// A selected index appeared in more than one site space.
    #[error("sum_over_indices: index {index} appears in more than one site space")]
    IndexInMultipleSiteSpaces {
        /// Debug representation of the repeated index.
        index: String,
    },
    /// A selected index was not external to the input TreeTN.
    #[error("sum_over_indices: index {index} not found in external indices")]
    IndexNotFound {
        /// Debug representation of the missing index.
        index: String,
    },
    /// Construction of a local unit-weight tensor failed.
    #[error("sum_over_indices: failed to build ones tensor at node {node}: {message}")]
    BuildOnesTensor {
        /// Debug representation of the node.
        node: String,
        /// Error message from the tensor constructor.
        message: String,
    },
    /// Construction of the weight TreeTN failed.
    #[error("sum_over_indices: failed to build ones TreeTN: {message}")]
    BuildWeightsTree {
        /// Error message from the TreeTN constructor.
        message: String,
    },
    /// The underlying partial contraction failed.
    #[error("selected-index contraction failed: {message}")]
    PartialContractFailed {
        /// Error message from the partial contraction path, including context.
        message: String,
    },
}

pub(crate) fn format_anyhow_error(error: anyhow::Error) -> String {
    format!("{error:#}")
}

#[cfg(test)]
mod tests;

/// Error returned by general TreeTN and operator index operations.
///
/// Types the failures a caller can act on, and keeps `Operation` as the
/// catch-all for everything else (an invariant, storage or backend failure).
/// This mirrors `TreeTciError` in `tensor4all-treetci`.
///
/// The wrapped diagnostic stays reachable through both [`std::error::Error::source`] and
/// the variants' `source` field, so callers that still propagate into `anyhow::Result`
/// keep the full source chain.
///
/// # Remedies
/// - [`Self::InvalidConfiguration`]: correct the option or argument value the
///   message names.
/// - [`Self::InvalidGraph`]: the network cannot serve the request - a node, edge
///   or index is not part of it, or its structure is not a tree. Fix the network
///   or the names passed with it.
/// - [`Self::IndexOutOfBounds`]: the index or coordinate is outside the
///   dimension it is read from.
/// - [`Self::ScalarKindMismatch`]: request the element type the network stores.
/// - [`Self::Operation`]: the wrapped source chain identifies the failing stage.
#[derive(Debug, thiserror::Error)]
pub enum TreeTNOperationError {
    /// An option or argument value supplied by the caller is invalid or
    /// unsupported.
    #[error("invalid TreeTN operation configuration: {source}")]
    InvalidConfiguration {
        /// Original diagnostic.
        #[source]
        source: anyhow::Error,
    },
    /// The network cannot serve the request: an unknown node, edge or index, or
    /// a structure that is not a tree.
    #[error("invalid TreeTN graph: {source}")]
    InvalidGraph {
        /// Original diagnostic.
        #[source]
        source: anyhow::Error,
    },
    /// An index, coordinate or count is outside the dimension it is read from.
    #[error("index out of bounds: {source}")]
    IndexOutOfBounds {
        /// Original diagnostic.
        #[source]
        source: anyhow::Error,
    },
    /// The requested element type cannot represent the values the network
    /// stores.
    ///
    /// The mismatch payload is reachable through `Error::source`; the message is
    /// the payload's own `Display`, named here instead of forwarded through
    /// `#[error(transparent)]`, because transparent forwarding exposes the
    /// payload's `source()` (which it does not have) instead of the payload.
    ///
    /// `context` prefixes the layers [`From<anyhow::Error>`] found around the
    /// payload, already rendered as text, when the payload was the `anyhow`
    /// error object. Layers above a typed context or a foreign wrapper are not
    /// kept here (see [`From<anyhow::Error>`]).
    #[error("{context}{payload}")]
    ScalarKindMismatch {
        /// The dtype mismatch.
        #[source]
        payload: crate::EvaluatedScalarKindMismatch,
        /// Rendered `anyhow` context layers, each followed by `": "`.
        context: String,
    },
    /// An underlying tensor, storage, or backend operation failed, or an
    /// internal invariant was violated.
    #[error("TreeTN operation failed: {source}")]
    Operation {
        /// Original diagnostic, preserving the full source chain.
        #[source]
        source: anyhow::Error,
    },
}

impl TreeTNOperationError {
    /// An invalid option or argument value chosen by the caller.
    pub(crate) fn invalid_configuration(source: impl Into<anyhow::Error>) -> Self {
        Self::InvalidConfiguration {
            source: source.into(),
        }
    }

    /// The network or a name selected from it cannot serve the request.
    pub(crate) fn invalid_graph(source: impl Into<anyhow::Error>) -> Self {
        Self::InvalidGraph {
            source: source.into(),
        }
    }

    /// An index, coordinate or count outside the dimension it is read from.
    pub(crate) fn index_out_of_bounds(source: impl Into<anyhow::Error>) -> Self {
        Self::IndexOutOfBounds {
            source: source.into(),
        }
    }

    /// Prefixes whatever diagnostic the variant carries with `context`, keeping
    /// the variant: the classified counterpart of `anyhow::Context`. The
    /// context is kept as rendered text, not as its original type.
    pub(crate) fn with_context(self, context: impl std::fmt::Display) -> Self {
        let context = context.to_string();
        match self {
            Self::InvalidConfiguration { source } => Self::InvalidConfiguration {
                source: source.context(context),
            },
            Self::InvalidGraph { source } => Self::InvalidGraph {
                source: source.context(context),
            },
            Self::IndexOutOfBounds { source } => Self::IndexOutOfBounds {
                source: source.context(context),
            },
            Self::Operation { source } => Self::Operation {
                source: source.context(context),
            },
            Self::ScalarKindMismatch {
                payload,
                context: outer,
            } => Self::ScalarKindMismatch {
                payload,
                context: format!("{context}: {outer}"),
            },
        }
    }
}

/// Which variant an error reports, without the `anyhow` chain it carries.
///
/// Lets the classification be taken from a failure that `anyhow` only lends by
/// reference and then rebuilt around the chain that failure already holds.
#[derive(Clone)]
enum Class {
    InvalidConfiguration,
    InvalidGraph,
    IndexOutOfBounds,
    Operation,
    ScalarKindMismatch {
        payload: crate::EvaluatedScalarKindMismatch,
        context: String,
    },
}

impl Class {
    fn of(error: &TreeTNOperationError) -> Self {
        match error {
            TreeTNOperationError::InvalidConfiguration { .. } => Self::InvalidConfiguration,
            TreeTNOperationError::InvalidGraph { .. } => Self::InvalidGraph,
            TreeTNOperationError::IndexOutOfBounds { .. } => Self::IndexOutOfBounds,
            TreeTNOperationError::Operation { .. } => Self::Operation,
            TreeTNOperationError::ScalarKindMismatch { payload, context } => {
                Self::ScalarKindMismatch {
                    payload: *payload,
                    context: context.clone(),
                }
            }
        }
    }

    /// The variant around `source`. `ScalarKindMismatch` has no chain to hold
    /// it, so it keeps the failure's own payload and context unchanged: the
    /// links above it may be wrappers that already render the failure, and
    /// prefixing their text would repeat it on every crossing.
    fn into_error(self, source: anyhow::Error) -> TreeTNOperationError {
        match self {
            Self::InvalidConfiguration => TreeTNOperationError::InvalidConfiguration { source },
            Self::InvalidGraph => TreeTNOperationError::InvalidGraph { source },
            Self::IndexOutOfBounds => TreeTNOperationError::IndexOutOfBounds { source },
            Self::Operation => TreeTNOperationError::Operation { source },
            Self::ScalarKindMismatch { payload, context } => {
                TreeTNOperationError::ScalarKindMismatch { payload, context }
            }
        }
    }
}

impl From<crate::EvaluatedScalarKindMismatch> for TreeTNOperationError {
    fn from(payload: crate::EvaluatedScalarKindMismatch) -> Self {
        Self::ScalarKindMismatch {
            payload,
            context: String::new(),
        }
    }
}

impl From<anyhow::Error> for TreeTNOperationError {
    /// Internal `?` conversions land in the catch-all variant with their chain
    /// intact. An error that carries a classified variant keeps that variant.
    /// When it is the `anyhow` error object, the context layers above it are
    /// recorded as the message they render (a typed context cannot be recovered
    /// from `anyhow`). When a typed context or a foreign wrapper sits between,
    /// the variant holds the received chain unchanged, except
    /// `ScalarKindMismatch`, which has no chain: it keeps its own payload and
    /// context, and the links above it are dropped rather than re-rendered,
    /// because a wrapper may already render the mismatch.
    ///
    /// The classification is taken from the first `TreeTNOperationError` in the
    /// source chain (the failure), never from a typed context above it:
    /// `anyhow::downcast` prefers such a context over the object, and it is a
    /// note about the call, not the failure. A foreign wrapper whose `source()`
    /// exposes the failure classifies the same way, with or without a note.
    fn from(source: anyhow::Error) -> Self {
        // `chain()` runs outside in: the `anyhow` contexts above the error
        // object, then the object. A context link is not downcastable to `Self`
        // (it is a `ContextError`), so the first match is the failure.
        let mut notes: Vec<String> = Vec::new();
        let mut failure = None;
        for cause in source.chain() {
            match cause.downcast_ref::<Self>() {
                Some(error) => {
                    failure = Some(error);
                    break;
                }
                None => notes.push(cause.to_string()),
            }
        }
        let Some(failure) = failure else {
            return Self::Operation { source };
        };
        // `anyhow::downcast_ref` returns the outermost typed context before the
        // error object, so it equals `failure` only when the failure is the
        // object itself. Otherwise something sits between them: a typed context
        // or a foreign wrapper whose `source()` exposes the failure. `anyhow`
        // cannot tell those two apart, so both keep the failure's variant (a
        // note never changes the classification) around the chain `source`
        // already holds once, with nothing re-rendered into it.
        let is_object = source
            .downcast_ref::<Self>()
            .is_some_and(|outer| std::ptr::eq(outer, failure));
        if !is_object {
            return Class::of(failure).into_error(source);
        }
        let variant = match source.downcast::<Self>() {
            Ok(variant) => variant,
            Err(source) => return Self::Operation { source },
        };
        notes
            .into_iter()
            .rev()
            .fold(variant, |error, text| error.with_context(text))
    }
}

impl From<tensor4all_core::IdxTensorError> for TreeTNOperationError {
    fn from(source: tensor4all_core::IdxTensorError) -> Self {
        Self::Operation {
            source: anyhow::Error::new(source),
        }
    }
}
