//! Operator application with a certified truncation bound (#1286).

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{IndexLike, TensorLike};

use super::apply::{
    bind_linear_operator_indices, extend_operator_to_full_space, transform_output_to_true,
    transform_state_to_input,
};
use super::linear_operator::LinearOperator;
use super::Operator;
use crate::error::{CertifiedApplyError, TreeTNOperationError};
use crate::treetn::contraction::certified::DEFAULT_MEMORY_LIMIT;
use crate::treetn::TreeTN;

/// Error budget and memory limit of [`apply_linear_operator_certified`].
///
/// There is no default budget: the absolute error the application may discard
/// is the caller's decision, and it changes what the result's error means
/// (a proved bound instead of a measured difference).
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::CertifiedApplyOptions;
///
/// let options = CertifiedApplyOptions::new(1e-8);
/// assert_eq!(options.budget, 1e-8);
/// assert_eq!(options.memory_limit, CertifiedApplyOptions::DEFAULT_MEMORY_LIMIT);
/// assert_eq!(options.with_memory_limit(1 << 30).memory_limit, 1 << 30);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CertifiedApplyOptions {
    /// Absolute L2 (Frobenius) error the application may discard, in the units
    /// of the result. Must be finite and non-negative. The certified bound of
    /// the result never exceeds it.
    pub budget: f64,
    /// Upper limit, in bytes, on the estimated memory of the application.
    /// Default [`Self::DEFAULT_MEMORY_LIMIT`] (4 GiB). Building the exact
    /// factors is estimated from the bond dimensions before anything is
    /// allocated; each sweep step is checked from its actual dimensions, with
    /// the factors still resident, before its block is formed. Either estimate
    /// above the limit fails, and the second fails after the factors were
    /// built.
    pub memory_limit: usize,
}

impl CertifiedApplyOptions {
    /// The default memory limit: 4 GiB.
    pub const DEFAULT_MEMORY_LIMIT: usize = DEFAULT_MEMORY_LIMIT;

    /// Options with the absolute error `budget` and the default memory limit.
    pub fn new(budget: f64) -> Self {
        Self {
            budget,
            memory_limit: Self::DEFAULT_MEMORY_LIMIT,
        }
    }

    /// Replace the memory limit, in bytes.
    pub fn with_memory_limit(mut self, memory_limit: usize) -> Self {
        self.memory_limit = memory_limit;
        self
    }
}

/// The result of [`apply_linear_operator_certified`]: the truncated image and
/// a proved bound on what the truncation discarded.
///
/// [`Self::bound`] bounds `||A x - y||_2`, where `y` is [`Self::network`], for
/// the computed intermediate tensors: the roundoff made while contracting and
/// factorizing them is backend roundoff and is not part of the bound.
///
/// # Examples
///
/// See [`apply_linear_operator_certified`].
#[derive(Debug, Clone)]
pub struct CertifiedApplication<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    network: TreeTN<T, V>,
    bound: f64,
    step_bounds: Vec<f64>,
    estimated_bytes: usize,
    factor_bytes: usize,
}

impl<T, V> CertifiedApplication<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// The truncated image `y`, canonical at the state's default contraction
    /// centre (the smaller-named chain endpoint).
    pub fn network(&self) -> &TreeTN<T, V> {
        &self.network
    }

    /// The image by value.
    pub fn into_network(self) -> TreeTN<T, V> {
        self.network
    }

    /// Certified upper bound of `||A x - y||_2`, at most the budget.
    pub fn bound(&self) -> f64 {
        self.bound
    }

    /// Certified bound of each sweep step, in sweep order. The steps discard
    /// mutually orthogonal parts, so [`Self::bound`] is their Euclidean norm,
    /// rounded up.
    pub fn step_bounds(&self) -> &[f64] {
        &self.step_bounds
    }

    /// The largest memory estimate the limit was checked against, in bytes:
    /// building the factors, or a sweep step with the factors still stored.
    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }

    /// Bytes of the exact factors the sweep stored, as built.
    pub fn factor_bytes(&self) -> usize {
        self.factor_bytes
    }
}

/// Apply a [`LinearOperator`] to a chain state, truncating with a certified
/// error bound: compute `y` with `||A x - y||_2 <= options.budget`.
///
/// Use it when the exact application ([`apply_linear_operator`](crate::apply_linear_operator)
/// with a zero threshold) is too large to form and the truncation must be
/// charged as a proved bound rather than measured against an exact image.
///
/// The sweep runs along the chain. At each cut it keeps the smallest rank whose
/// certified error fits that step's share of the budget; the shares are
/// `budget / sqrt(N - 1)` and a step that spends less leaves the rest to later
/// steps. The projector of a step comes from the Schmidt spectrum of the
/// current state at that cut, weighted by an exact factor of the part not yet
/// swept, so the kept bonds are the ones a global SVD truncation of `A x` at
/// that accuracy would keep. The factors are built by untruncated QR before
/// the sweep, and their memory is estimated from the bond dimensions first.
/// The closing step discards nothing.
///
/// The operator may act on part of the state; its gaps are filled with
/// identities as in [`apply_linear_operator`](crate::apply_linear_operator).
///
/// # Arguments
///
/// * `operator` - The operator `A`. Its nodes must be nodes of the state.
/// * `state` - The state `x`. It must be a chain.
/// * `options` - The absolute error budget and the memory limit.
///
/// # Returns
///
/// The image with its bound, step bounds and memory figures; see
/// [`CertifiedApplication`].
///
/// # Errors
///
/// - [`CertifiedApplyError::InvalidOptions`] for a negative or non-finite budget;
/// - [`CertifiedApplyError::UnsupportedTopology`] when the state is not a chain,
///   or a full-coverage operator does not have the state's edges;
/// - [`CertifiedApplyError::MemoryLimit`] when an estimate exceeds
///   `options.memory_limit`: building the factors, checked before any factor
///   is built, or a sweep step, checked before its block is formed with the
///   factors still resident;
/// - [`CertifiedApplyError::BudgetUnreachable`] when a step cannot be certified
///   within its share even at full rank (a budget at roundoff level);
/// - [`CertifiedApplyError::Operation`] when the operator's nodes are not a
///   subset of the state's, the state is empty, or extending the operator,
///   transforming indices, or a tensor operation fails.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use tensor4all_core::{DynIndex, IdxTensor, TensorVectorSpace};
/// use tensor4all_treetn::{
///     apply_linear_operator, apply_linear_operator_certified, ApplyOptions,
///     CertifiedApplyOptions, IndexMapping, LinearOperator, TreeTN,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // A two-site product state and an operator acting on site "a" only.
/// let (a, b, bond) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2), DynIndex::new_dyn(1));
/// let state = TreeTN::<IdxTensor, &str>::from_tensors(
///     vec![
///         IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![1.0, 2.0])?,
///         IdxTensor::from_dense(vec![bond, b.clone()], vec![3.0, 4.0])?,
///     ],
///     vec!["a", "b"],
/// )?;
/// let (input, output) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
/// // Column-major [output, input]: diag(1, 2).
/// let diagonal = IdxTensor::from_dense(vec![output.clone(), input.clone()], vec![1.0, 0.0, 0.0, 2.0])?;
/// let operator = LinearOperator::new(
///     TreeTN::<IdxTensor, &str>::from_tensors(vec![diagonal], vec!["a"])?,
///     HashMap::from([("a", IndexMapping { true_index: a.clone(), internal_index: input })]),
///     HashMap::from([("a", IndexMapping { true_index: a.clone(), internal_index: output })]),
/// );
///
/// let certified = apply_linear_operator_certified(
///     &operator,
///     &state,
///     CertifiedApplyOptions::new(1e-10),
/// )?;
/// assert!(certified.bound() <= 1e-10);
/// // The image is rank one, so nothing needs to be discarded: it equals the
/// // exact application.
/// let exact = apply_linear_operator(&operator, &state, ApplyOptions::naive())?;
/// let error = certified.network().to_dense()?.sub(&exact.to_dense()?)?.norm()?;
/// assert!(error <= certified.bound().max(1e-14));
/// assert_eq!(certified.step_bounds().len(), 1);
/// # Ok(())
/// # }
/// ```
pub fn apply_linear_operator_certified<T, V>(
    operator: &LinearOperator<T, V>,
    state: &TreeTN<T, V>,
    options: CertifiedApplyOptions,
) -> Result<CertifiedApplication<T, V>, CertifiedApplyError>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if !options.budget.is_finite() || options.budget < 0.0 {
        return Err(CertifiedApplyError::InvalidOptions {
            reason: format!(
                "the budget must be finite and non-negative, got {}",
                options.budget
            ),
        });
    }
    let operation = |source: anyhow::Error| CertifiedApplyError::Operation {
        source: TreeTNOperationError::from(source),
    };
    let center = state.preferred_contraction_center().ok_or_else(|| {
        operation(anyhow::anyhow!(
            "apply_linear_operator_certified: empty state"
        ))
    })?;
    let state_nodes: HashSet<V> = state.node_names().into_iter().collect();
    let op_nodes: HashSet<V> = operator.node_names();
    let full_operator = if op_nodes == state_nodes {
        operator.clone()
    } else if op_nodes.is_subset(&state_nodes) {
        extend_operator_to_full_space(operator, state)
            .map_err(operation)?
            .0
    } else {
        return Err(operation(anyhow::anyhow!(
            "operator nodes {op_nodes:?} are not a subset of state nodes {state_nodes:?}"
        )));
    };
    let transformed = transform_state_to_input(&full_operator, state).map_err(operation)?;
    let chain = transformed.contract_certified_chain(
        full_operator.mpo(),
        &center,
        options.budget,
        options.memory_limit,
    )?;
    let network = transform_output_to_true(&full_operator, chain.network).map_err(operation)?;
    Ok(CertifiedApplication {
        network,
        bound: chain.bound,
        step_bounds: chain.step_bounds,
        estimated_bytes: chain.estimated_bytes,
        factor_bytes: chain.factor_bytes,
    })
}

/// [`apply_linear_operator_certified`] after explicitly binding selected
/// external indices, as [`apply_linear_operator_to_indices`](crate::apply_linear_operator_to_indices)
/// does for [`apply_linear_operator`](crate::apply_linear_operator).
///
/// # Arguments
///
/// * `operator` - Operator to apply.
/// * `state` - Chain state the rebound operator is applied to.
/// * `input_pairs` - `(operator_true_input, state_input)` replacements.
/// * `output_pairs` - `(operator_true_output, desired_output)` replacements.
/// * `options` - The absolute error budget and the memory limit.
///
/// # Returns
///
/// The certified application of the rebound operator.
///
/// # Errors
///
/// [`CertifiedApplyError::BindIndices`] when rebinding fails (see
/// [`bind_linear_operator_indices`](crate::bind_linear_operator_indices)), and
/// the errors of [`apply_linear_operator_certified`].
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use tensor4all_core::{DynIndex, IdxTensor, TensorVectorSpace};
/// use tensor4all_treetn::{
///     apply_linear_operator_certified_to_indices, CertifiedApplyOptions, IndexMapping,
///     LinearOperator, TreeTN,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let site = DynIndex::new_dyn(2);
/// let state = TreeTN::<IdxTensor, usize>::from_tensors(
///     vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 5.0])?],
///     vec![0],
/// )?;
/// // An operator built on its own indices: it negates its input.
/// let (op_input, op_output) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
/// let (input, output) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
/// let negate = IdxTensor::from_dense(vec![output.clone(), input.clone()], vec![-1.0, 0.0, 0.0, -1.0])?;
/// let operator = LinearOperator::new(
///     TreeTN::<IdxTensor, usize>::from_tensors(vec![negate], vec![0])?,
///     HashMap::from([(0, IndexMapping { true_index: op_input.clone(), internal_index: input })]),
///     HashMap::from([(0, IndexMapping { true_index: op_output.clone(), internal_index: output })]),
/// );
/// let certified = apply_linear_operator_certified_to_indices(
///     &operator,
///     &state,
///     &[(op_input, site.clone())],
///     &[(op_output, site.clone())],
///     CertifiedApplyOptions::new(0.0),
/// )?;
/// let expected = IdxTensor::from_dense(vec![site], vec![-3.0, -5.0])?;
/// assert!(certified.network().to_dense()?.sub(&expected)?.norm()? < 1e-14);
/// assert_eq!(certified.bound(), 0.0);
/// # Ok(())
/// # }
/// ```
pub fn apply_linear_operator_certified_to_indices<T, V>(
    operator: &LinearOperator<T, V>,
    state: &TreeTN<T, V>,
    input_pairs: &[(T::Index, T::Index)],
    output_pairs: &[(T::Index, T::Index)],
    options: CertifiedApplyOptions,
) -> Result<CertifiedApplication<T, V>, CertifiedApplyError>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let rebound = bind_linear_operator_indices(operator, input_pairs, output_pairs)?;
    apply_linear_operator_certified(&rebound, state, options)
}

#[cfg(test)]
mod tests;
