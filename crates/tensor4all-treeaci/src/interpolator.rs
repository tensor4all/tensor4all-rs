//! TreeACI as an element-wise interpolation engine.
//!
//! [`TreeAciInterpolator`] implements
//! [`ElementwiseInterpolator`](tensor4all_treetn::interpolation::ElementwiseInterpolator)
//! for every operator, so code written against that engine-independent
//! contract (such as the patched element-wise driver of
//! `tensor4all-partitionedtreetn`) can run tree ACI on each patch.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use rand::SeedableRng;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treetn::interpolation::{
    ElementwiseInterpolator, ElementwiseOperator, EvalContext, InterpolationError,
    InterpolationOutcome, InterpolationProblem, InterpolationTermination,
};
use tensor4all_treetn::TreeTN;

use crate::{
    tree_elementwise_batched_with_rng_on_session, Result, TreeAciError, TreeAciOptions,
    TreeAciResult, TreeAciScalar, TreeAciTermination, TreeElementwiseBatch,
};

/// Tree element-wise interpolation engine backed by tree ACI.
///
/// Implements
/// [`ElementwiseInterpolator`](tensor4all_treetn::interpolation::ElementwiseInterpolator)
/// for every operator type, so any pointwise map `f(x_1, ..., x_k)` of the
/// input networks can be interpolated. Use it where code is written against
/// that contract, for example as the per-patch engine of a patched
/// element-wise operation; call [`tree_elementwise_batched`](crate::tree_elementwise_batched)
/// directly for one unpatched run.
///
/// Each run calls [`tree_elementwise_batched_with_rng_on_session`] on the
/// session of the caller's [`EvalContext`], with the engine's
/// [`TreeAciOptions`] except that the problem supplies
///
/// - `tolerance` (the problem's absolute tolerance, which must be positive,
///   divided by the engine's [tolerance factor](Self::with_tolerance_factor)),
/// - `scale_tolerance` (always `false`: the tolerance is absolute),
/// - `max_bond_dim` (the problem's cap), and
/// - the random stream (a ChaCha8 stream seeded with the problem's seed).
///
/// Tree ACI takes no pivots, so the problem's initial pivots and global
/// search starts are not used, and the outcome carries no pivots.
///
/// A converged tree ACI run does not bound its error by `tolerance`: the
/// global guard accepts residuals up to
/// [`TreeAciOptions::global_tolerance_margin`] times the tolerance, and the
/// error estimate covers the sampled local matrices only. A caller that
/// accepts each run on a separate measurement at the problem's tolerance (as
/// the patched element-wise driver of `tensor4all-partitionedtreetn` does)
/// can give the kernel a tighter target with
/// [`with_tolerance_factor`](Self::with_tolerance_factor), which leaves the
/// problem's tolerance, and therefore that acceptance, unchanged.
///
/// The outcome maps tree ACI's result as follows:
///
/// - `termination`: `Converged` below the cap is `Converged`; `Converged` at
///   the cap and `RankLimited` are `BondCapReached`; `MaxSweeps` is
///   `IterationLimit`.
/// - `error_estimate`: the last sweep's largest local error, which with an
///   absolute tolerance is in the units of `f`; zero on the exact single-node
///   path. It is compared with the tightened tolerance inside the run, not
///   with the problem's.
/// - `max_sample_magnitude`: [`TreeAciResult::max_sample_magnitude`], which
///   excludes the global guard's samples.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, DynIndex, IdxTensor};
/// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
/// use tensor4all_treeaci::TreeAciInterpolator;
/// use tensor4all_treetn::interpolation::{
///     ElementwiseInterpolator, EvalContext, InterpolationProblem, InterpolationTermination,
///     Parallelism, Product,
/// };
/// use tensor4all_treetn::{NodeNameNetwork, TreeTN};
///
/// // Two rank-one inputs on the path 0 - 1: x(a, b) = (1 + a)(1 + b) and
/// // y(a, b) = (2 - a)(1 + 2 b).
/// let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
/// let rank_one = |left: [f64; 2], right: [f64; 2]| -> Result<TreeTN<IdxTensor, usize>, Box<dyn std::error::Error>> {
///     let bond = DynIndex::new_dyn(1);
///     Ok(TreeTN::from_tensors(
///         vec![
///             IdxTensor::from_dense(vec![a.clone(), bond.clone()], left.to_vec())?,
///             IdxTensor::from_dense(vec![bond, b.clone()], right.to_vec())?,
///         ],
///         vec![0usize, 1],
///     )?)
/// };
/// let inputs = [rank_one([1.0, 2.0], [1.0, 2.0])?, rank_one([2.0, 1.0], [1.0, 3.0])?];
///
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// topology.add_node(1)?;
/// topology.add_edge(&0, &1)?;
/// let node_sites = BTreeMap::from([(0usize, vec![a.clone()]), (1, vec![b.clone()])]);
/// let pivots = ColMajorArray::new(vec![0, 0], vec![2, 1])?;
/// let problem = InterpolationProblem::new(topology, node_sites, pivots, 1e-12, Some(4.try_into()?), 7)?;
///
/// let mut backend = CpuBackend::with_threads(1)?;
/// let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
/// // `Product` multiplies any scalar type, so name the scalar.
/// let outcome = ElementwiseInterpolator::<f64, _>::interpolate_elementwise(
///     &TreeAciInterpolator::default(),
///     &problem,
///     &inputs,
///     &Product,
///     &mut ctx,
/// )?;
///
/// assert_eq!(outcome.termination, InterpolationTermination::Converged);
/// // The product is (1 + a)(2 - a) (1 + b)(1 + 2 b).
/// let value = outcome.network.evaluate_point(&[a, b], &[1, 1])?;
/// assert!((value.real() - 2.0 * 1.0 * 2.0 * 3.0).abs() < 1e-10);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct TreeAciInterpolator {
    // Node-free: the fields that name nodes come from each run's problem.
    options: TreeAciOptions<()>,
    // Each run's tolerance is the problem's divided by this factor (>= 1).
    tolerance_factor: f64,
}

impl TreeAciInterpolator {
    /// Build an engine from tree ACI options.
    ///
    /// `tolerance`, `scale_tolerance`, `max_bond_dim`, and `rng_seed` are
    /// replaced by the problem on every run (see the type documentation); the
    /// other fields configure the run. `initial_guess` and `root` name nodes
    /// of one network and must be unset.
    ///
    /// # Errors
    ///
    /// Returns [`TreeAciError::InvalidOption`] when `initial_guess` or `root`
    /// is set, or when the options fail the checks of a tree ACI run (for
    /// example `min_sweeps` of zero or above `max_sweeps`, or a zero resource
    /// limit).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treeaci::{TreeAciError, TreeAciInterpolator, TreeAciOptions};
    ///
    /// let options = TreeAciOptions::<usize> { max_sweeps: 6, ..Default::default() };
    /// assert!(TreeAciInterpolator::new(options).is_ok());
    ///
    /// let rooted = TreeAciOptions { root: Some(0usize), ..Default::default() };
    /// let error = TreeAciInterpolator::new(rooted).unwrap_err();
    /// assert!(matches!(error, TreeAciError::InvalidOption { option: "root", .. }));
    /// ```
    pub fn new<W: crate::TreeAciNode>(options: TreeAciOptions<W>) -> Result<Self> {
        if options.initial_guess.is_some() {
            return Err(TreeAciError::InvalidOption {
                option: "initial_guess",
                message: "an interpolation engine runs on many problems and takes no initial guess",
            });
        }
        if options.root.is_some() {
            return Err(TreeAciError::InvalidOption {
                option: "root",
                message: "an interpolation engine runs on many problems and takes no root",
            });
        }
        crate::problem::validate_options(&options)?;
        Ok(Self {
            options: options.for_nodes(),
            tolerance_factor: 1.0,
        })
    }

    /// Run tree ACI with the problem's tolerance divided by `factor`.
    ///
    /// The problem's tolerance is what the caller accepts a result at; with a
    /// factor above one the kernel aims below it, which leaves room for tree
    /// ACI's convergence criterion (see the type documentation). The outcome's
    /// termination and error estimate refer to the tightened run. A larger
    /// factor gives larger ranks and more accurate results than requested;
    /// one (the default) runs at the problem's tolerance.
    ///
    /// # Arguments
    ///
    /// * `factor` - Finite and at least one.
    ///
    /// # Errors
    ///
    /// Returns [`TreeAciError::InvalidOption`] for a factor below one or not
    /// finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treeaci::{TreeAciError, TreeAciInterpolator};
    ///
    /// let engine = TreeAciInterpolator::default().with_tolerance_factor(3.0)?;
    /// assert_eq!(engine.tolerance_factor(), 3.0);
    /// assert_eq!(TreeAciInterpolator::default().tolerance_factor(), 1.0);
    ///
    /// let error = TreeAciInterpolator::default().with_tolerance_factor(0.5).unwrap_err();
    /// assert!(matches!(error, TreeAciError::InvalidOption { option: "tolerance_factor", .. }));
    /// # Ok::<(), TreeAciError>(())
    /// ```
    pub fn with_tolerance_factor(mut self, factor: f64) -> Result<Self> {
        if !(factor.is_finite() && factor >= 1.0) {
            return Err(TreeAciError::InvalidOption {
                option: "tolerance_factor",
                message: "the tolerance factor must be finite and at least one",
            });
        }
        self.tolerance_factor = factor;
        Ok(self)
    }

    /// The factor each run's tolerance is divided by; see
    /// [`with_tolerance_factor`](Self::with_tolerance_factor).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treeaci::TreeAciInterpolator;
    ///
    /// assert_eq!(TreeAciInterpolator::default().tolerance_factor(), 1.0);
    /// ```
    pub fn tolerance_factor(&self) -> f64 {
        self.tolerance_factor
    }

    /// The options of one run on `problem`.
    fn run_options<V: crate::TreeAciNode>(
        &self,
        problem: &InterpolationProblem<V>,
    ) -> TreeAciOptions<V> {
        TreeAciOptions {
            tolerance: problem.absolute_tolerance() / self.tolerance_factor,
            scale_tolerance: false,
            max_bond_dim: problem.max_bond_dim().map(|cap| cap.get()),
            rng_seed: problem.seed(),
            ..self.options.for_nodes()
        }
    }
}

impl Default for TreeAciInterpolator {
    /// An engine with [`TreeAciOptions::default`].
    fn default() -> Self {
        Self {
            options: TreeAciOptions::default(),
            tolerance_factor: 1.0,
        }
    }
}

impl<T, Op> ElementwiseInterpolator<T, Op> for TreeAciInterpolator
where
    T: TreeAciScalar,
    Op: ElementwiseOperator<T>,
{
    /// Run tree ACI on `inputs` mapped by `operator`.
    ///
    /// # Errors
    ///
    /// - [`InterpolationError::InvalidProblem`] when `inputs` is empty, the
    ///   first input's topology or per-node site sets differ from the
    ///   problem's, the inputs disagree with each other, the problem's
    ///   tolerance is zero, or tree ACI rejects the problem's size.
    /// - [`InterpolationError::Evaluator`] when `operator` fails or returns a
    ///   non-finite value; the operator's error is the source.
    /// - [`InterpolationError::Engine`] when tree ACI fails for any other
    ///   reason (a numerical step, a resource limit, or a session refusal).
    fn interpolate_elementwise<V>(
        &self,
        problem: &InterpolationProblem<V>,
        inputs: &[TreeTN<IdxTensor, V>],
        operator: &Op,
        ctx: &mut EvalContext<'_>,
    ) -> std::result::Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    {
        check_inputs(problem, inputs)?;
        if problem.absolute_tolerance() <= 0.0 {
            return Err(InterpolationError::InvalidProblem {
                message: "tree ACI needs a positive absolute tolerance".to_string(),
            });
        }
        let options = self.run_options(problem);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(problem.seed());
        // The operator's own error, kept so that it reaches the caller as the
        // source of `Evaluator` rather than as tree ACI's message string.
        let failure: RefCell<Option<anyhow::Error>> = RefCell::new(None);
        let callback = |batch: TreeElementwiseBatch<'_, T>, output: &mut [T]| -> Result<()> {
            let shape = [batch.n_inputs(), batch.n_points()];
            let applied = ColMajorArrayRef::new(batch.as_col_major_slice(), &shape)
                .map_err(anyhow::Error::from)
                .and_then(|values| operator.apply(values, output))
                .and_then(|()| {
                    crate::scalar::ensure_finite_values(output, "operator output")
                        .map_err(|_| anyhow::anyhow!("the operator returned a non-finite value"))
                });
            applied.map_err(|error| {
                let message = error.to_string();
                *failure.borrow_mut() = Some(error);
                TreeAciError::Callback { message }
            })
        };
        let result = tree_elementwise_batched_with_rng_on_session(
            callback,
            inputs,
            &options,
            &mut rng,
            ctx.session(),
        );
        let result = match result {
            Ok(result) => result,
            Err(error) => return Err(classify(error, failure.into_inner())),
        };
        Ok(outcome(result, problem))
    }
}

/// The first input carries the problem's topology and per-node site sets;
/// tree ACI checks the other inputs against the first.
fn check_inputs<V>(
    problem: &InterpolationProblem<V>,
    inputs: &[TreeTN<IdxTensor, V>],
) -> std::result::Result<(), InterpolationError>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let invalid = |message: &str| InterpolationError::InvalidProblem {
        message: message.to_string(),
    };
    let first = inputs
        .first()
        .ok_or_else(|| invalid("an element-wise interpolation needs at least one input"))?;
    if !first
        .site_index_network()
        .topology()
        .same_topology(problem.topology())
    {
        return Err(invalid(
            "the first input's topology differs from the problem's",
        ));
    }
    for (node, sites) in problem.node_sites() {
        let expected: HashSet<&DynIndex> = sites.iter().collect();
        let actual = first.site_space(node);
        let matches = actual.is_some_and(|actual| {
            actual.len() == expected.len() && actual.iter().all(|site| expected.contains(site))
        });
        if !matches {
            return Err(InterpolationError::InvalidProblem {
                message: format!(
                    "the first input's sites at node {node:?} differ from the problem's"
                ),
            });
        }
    }
    Ok(())
}

/// The interpolation error of a failed run.
fn classify(error: TreeAciError, operator_failure: Option<anyhow::Error>) -> InterpolationError {
    match (error, operator_failure) {
        (TreeAciError::Callback { .. }, Some(source)) => InterpolationError::Evaluator { source },
        (
            error @ (TreeAciError::NoInputs
            | TreeAciError::EmptyTree { .. }
            | TreeAciError::TopologyMismatch { .. }
            | TreeAciError::PhysicalIndexMismatch { .. }
            | TreeAciError::SizeOverflow { .. }),
            _,
        ) => InterpolationError::InvalidProblem {
            message: error.to_string(),
        },
        (error, _) => InterpolationError::Engine {
            source: anyhow::Error::new(error),
        },
    }
}

/// The outcome of a finished run.
fn outcome<V>(
    result: TreeAciResult<V>,
    problem: &InterpolationProblem<V>,
) -> InterpolationOutcome<V>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let final_rank = result
        .diagnostics
        .edge_ranks
        .iter()
        .map(|&(_, _, rank)| rank)
        .max()
        .unwrap_or(0);
    let at_cap = problem
        .max_bond_dim()
        .is_some_and(|cap| final_rank >= cap.get());
    let termination = match result.termination {
        TreeAciTermination::Converged if at_cap => InterpolationTermination::BondCapReached,
        TreeAciTermination::Converged => InterpolationTermination::Converged,
        TreeAciTermination::RankLimited => InterpolationTermination::BondCapReached,
        TreeAciTermination::MaxSweeps => InterpolationTermination::IterationLimit,
    };
    InterpolationOutcome {
        error_estimate: result.max_errors.last().copied().unwrap_or(0.0),
        max_sample_magnitude: result.max_sample_magnitude,
        network: result.tree,
        termination,
        pivots: None,
    }
}

#[cfg(test)]
mod tests;
