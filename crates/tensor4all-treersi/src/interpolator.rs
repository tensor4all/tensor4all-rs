//! Tree RSI as an element-wise interpolation engine for products.
//!
//! [`TreeRsiInterpolator`] implements
//! [`ElementwiseInterpolator`](tensor4all_treetn::interpolation::ElementwiseInterpolator)
//! for the [`Product`] operator only, so code written against that
//! engine-independent contract (such as the patched element-wise driver of
//! `tensor4all-partitionedtreetn`) can run tree RSI on each patch, and a
//! non-product operator is rejected at compile time.

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use rand::SeedableRng;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::interpolation::{
    ElementwiseInterpolator, EvalContext, InterpolationError, InterpolationOutcome,
    InterpolationProblem, InterpolationTermination, Product,
};
use tensor4all_treetn::TreeTN;

use crate::{
    hadamard_many_with_rng_on_session, Result, TreeRsiError, TreeRsiOptions, TreeRsiResult,
    TreeRsiScalar,
};

/// Tree element-wise interpolation engine for products, backed by tree RSI.
///
/// Implements
/// [`ElementwiseInterpolator`](tensor4all_treetn::interpolation::ElementwiseInterpolator)
/// for [`Product`] only: tree RSI computes the Hadamard product of its
/// inputs and no other map. Use it where code is written against that
/// contract, for example as the per-patch engine of a patched product; call
/// [`hadamard_many`](crate::hadamard_many) directly for one unpatched product.
///
/// Each run calls [`hadamard_many_with_rng_on_session`] on the session of the
/// caller's [`EvalContext`], with the engine's [`TreeRsiOptions`] except that
/// the problem supplies
///
/// - `max_bond_dim` (the problem's cap; a problem without a cap needs the
///   engine's `sketch_dim`), and
/// - the random stream (a ChaCha8 stream seeded with the problem's seed).
///
/// Tree RSI has no absolute tolerance and estimates no error: the problem's
/// tolerance is not used, and [`TreeRsiOptions::rel_tol`] (relative to the
/// largest local pivot) stays the engine's choice. It takes no pivots, so the
/// problem's initial pivots and global search starts are not used either.
///
/// The outcome is
///
/// - `termination`: `BondCapReached` when the largest output bond reaches the
///   cap, otherwise `ErrorNotMeasured`, since the run checks no error;
/// - `error_estimate`: `f64::INFINITY`, so that no comparison with a
///   tolerance passes on it;
/// - `max_sample_magnitude`: zero, since tree RSI evaluates no samples of the
///   product;
/// - `pivots`: `None`.
///
/// A caller must therefore measure the result; the patched driver of
/// `tensor4all-partitionedtreetn` does so under its L2 error norm.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, DynIndex, IdxTensor};
/// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
/// use tensor4all_treersi::TreeRsiInterpolator;
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
///             IdxTensor::from_dense_compact(vec![a.clone(), bond.clone()], left.to_vec())?,
///             IdxTensor::from_dense_compact(vec![bond, b.clone()], right.to_vec())?,
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
/// let outcome = ElementwiseInterpolator::<f64, _>::interpolate_elementwise(
///     &TreeRsiInterpolator::default(),
///     &problem,
///     &inputs,
///     &Product,
///     &mut ctx,
/// )?;
///
/// // Tree RSI checks no error: the caller measures the result.
/// assert_eq!(outcome.termination, InterpolationTermination::ErrorNotMeasured);
/// // The product is (1 + a)(2 - a) (1 + b)(1 + 2 b).
/// let value = outcome.network.evaluate_point(&[a, b], &[1, 1])?;
/// assert!((value.real() - 2.0 * 1.0 * 2.0 * 3.0).abs() < 1e-10);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Any operator other than [`Product`] does not compile:
///
/// ```compile_fail
/// use tensor4all_core::ColMajorArrayRef;
/// use tensor4all_treersi::TreeRsiInterpolator;
/// use tensor4all_treetn::interpolation::ElementwiseInterpolator;
///
/// fn engine<Op>(_: &impl ElementwiseInterpolator<f64, Op>, _: &Op) {}
/// let sum = |inputs: ColMajorArrayRef<'_, f64>, output: &mut [f64]| -> anyhow::Result<()> {
///     output.fill(inputs.data().iter().sum());
///     Ok(())
/// };
/// engine(&TreeRsiInterpolator::default(), &sum);
/// ```
#[derive(Clone, Debug)]
pub struct TreeRsiInterpolator {
    // Node-free: the fields that name nodes come from each run's problem.
    options: TreeRsiOptions<()>,
}

impl TreeRsiInterpolator {
    /// Build an engine from tree RSI options.
    ///
    /// `max_bond_dim` and `seed` are replaced by the problem on every run (see
    /// the type documentation); the other fields configure the run. `root` and
    /// `probes` name nodes of one network and must be unset.
    ///
    /// The cap comes from each problem, so it cannot be checked here: an engine
    /// without `sketch_dim` runs only problems that carry a cap, and returns
    /// [`InterpolationError::InvalidProblem`] for one without. The patched
    /// driver always passes a cap; set `sketch_dim` to run uncapped problems
    /// directly.
    ///
    /// # Errors
    ///
    /// Returns [`TreeRsiError::InvalidOption`] when `root` or `probes` is set,
    /// or when another field fails the checks of a tree RSI run (a zero
    /// `sketch_dim` or `max_local_elements`, or a negative or non-finite
    /// `rel_tol`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treersi::{TreeRsiError, TreeRsiInterpolator, TreeRsiOptions};
    ///
    /// let options = TreeRsiOptions::<usize> { rel_tol: 1e-10, ..Default::default() };
    /// assert!(TreeRsiInterpolator::new(options).is_ok());
    ///
    /// let rooted = TreeRsiOptions { root: Some(0usize), ..Default::default() };
    /// let error = TreeRsiInterpolator::new(rooted).unwrap_err();
    /// assert!(matches!(error, TreeRsiError::InvalidOption { option: "root", .. }));
    /// ```
    pub fn new<W>(options: TreeRsiOptions<W>) -> Result<Self> {
        if options.root.is_some() {
            return Err(TreeRsiError::InvalidOption {
                option: "root",
                message: "an interpolation engine runs on many problems and takes no root",
            });
        }
        if options.probes.is_some() {
            return Err(TreeRsiError::InvalidOption {
                option: "probes",
                message: "an interpolation engine runs on many problems and takes no probes",
            });
        }
        let options: TreeRsiOptions<()> = options.for_nodes();
        // The cap comes from each problem; check the other fields with one.
        crate::plan::validate_options(&TreeRsiOptions {
            max_bond_dim: Some(1),
            ..options.clone()
        })?;
        Ok(Self { options })
    }

    /// The options of one run on `problem`.
    fn run_options<V>(&self, problem: &InterpolationProblem<V>) -> TreeRsiOptions<V>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    {
        TreeRsiOptions {
            max_bond_dim: problem.max_bond_dim().map(|cap| cap.get()),
            seed: problem.seed(),
            ..self.options.for_nodes()
        }
    }
}

impl Default for TreeRsiInterpolator {
    /// An engine with [`TreeRsiOptions::default`].
    fn default() -> Self {
        Self {
            options: TreeRsiOptions::default(),
        }
    }
}

impl<T> ElementwiseInterpolator<T, Product> for TreeRsiInterpolator
where
    T: TreeRsiScalar,
{
    /// Run tree RSI on the product of `inputs`.
    ///
    /// # Errors
    ///
    /// - [`InterpolationError::InvalidProblem`] when `inputs` is empty, the
    ///   first input's topology or per-node site sets differ from the
    ///   problem's, the inputs disagree with each other, the problem has no
    ///   cap and the engine no `sketch_dim`, or a size overflows.
    /// - [`InterpolationError::Engine`] when tree RSI fails for any other
    ///   reason (an input it cannot read without a runtime, a resource limit,
    ///   a lost floating-point range, or a session failure).
    fn interpolate_elementwise<V>(
        &self,
        problem: &InterpolationProblem<V>,
        inputs: &[TreeTN<IdxTensor, V>],
        _operator: &Product,
        ctx: &mut EvalContext<'_>,
    ) -> std::result::Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    {
        check_inputs(problem, inputs)?;
        let options = self.run_options(problem);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(problem.seed());
        let result =
            hadamard_many_with_rng_on_session::<T, V, _>(inputs, &options, &mut rng, ctx.session())
                .map_err(classify)?;
        Ok(outcome(result, problem))
    }
}

/// The first input carries the problem's topology and per-node site sets;
/// tree RSI checks the other inputs against the first.
fn check_inputs<V>(
    problem: &InterpolationProblem<V>,
    inputs: &[TreeTN<IdxTensor, V>],
) -> std::result::Result<(), InterpolationError>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let invalid = |message: String| InterpolationError::InvalidProblem { message };
    let first = inputs.first().ok_or_else(|| {
        invalid("an element-wise interpolation needs at least one input".to_string())
    })?;
    if !first
        .site_index_network()
        .topology()
        .same_topology(problem.topology())
    {
        return Err(invalid(
            "the first input's topology differs from the problem's".to_string(),
        ));
    }
    for (node, sites) in problem.node_sites() {
        let expected: HashSet<&DynIndex> = sites.iter().collect();
        let matches = first.site_space(node).is_some_and(|actual| {
            actual.len() == expected.len() && actual.iter().all(|site| expected.contains(site))
        });
        if !matches {
            return Err(invalid(format!(
                "the first input's sites at node {node:?} differ from the problem's"
            )));
        }
    }
    Ok(())
}

/// The interpolation error of a failed run.
fn classify(error: TreeRsiError) -> InterpolationError {
    match error {
        error @ (TreeRsiError::NoInputs
        | TreeRsiError::EmptyTree { .. }
        | TreeRsiError::TopologyMismatch { .. }
        | TreeRsiError::PhysicalIndexMismatch { .. }
        | TreeRsiError::InvalidOption { .. }
        | TreeRsiError::SizeOverflow { .. }) => InterpolationError::InvalidProblem {
            message: error.to_string(),
        },
        error => InterpolationError::Engine {
            source: anyhow::Error::new(error),
        },
    }
}

/// The outcome of a finished run.
fn outcome<V>(
    result: TreeRsiResult<V>,
    problem: &InterpolationProblem<V>,
) -> InterpolationOutcome<V>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    // A single node has no bond (`max_rank` reports one for it), so it is
    // never at the cap.
    let final_rank = result
        .diagnostics
        .edges
        .iter()
        .map(|edge| edge.rank)
        .max()
        .unwrap_or(0);
    let at_cap = problem
        .max_bond_dim()
        .is_some_and(|cap| final_rank >= cap.get());
    InterpolationOutcome {
        network: result.tree,
        termination: if at_cap {
            InterpolationTermination::BondCapReached
        } else {
            InterpolationTermination::ErrorNotMeasured
        },
        error_estimate: f64::INFINITY,
        max_sample_magnitude: 0.0,
        pivots: None,
    }
}

#[cfg(test)]
mod tests;
