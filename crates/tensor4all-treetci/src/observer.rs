use crate::TreeTciEvaluationStats;

/// One completed iteration of the optimization loop, as [`TreeTciObserver::iteration`] receives it.
///
/// The values are the ones the run records for the iteration. `max_bond_dim` is
/// `ranks[iteration]` and `error` is `errors[iteration]` of the returned result,
/// bitwise. `error` is the bond error recorded after the iteration's edge
/// updates and before the global pivot injection of that iteration: pivots
/// injected afterwards are swept by the next iteration, so the next call can
/// report a different error for the same state.
///
/// Only the optimizer builds one.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct TreeTciIteration<'a> {
    /// Zero-based index of the iteration.
    pub iteration: usize,
    /// Maximum bond dimension after the iteration, `ranks[iteration]` of the result.
    pub max_bond_dim: usize,
    /// Maximum sampled bond error of the iteration, `errors[iteration]` of the result.
    pub error: f64,
    /// Evaluation accounting of the run so far; it never decreases between calls.
    pub evaluation: &'a TreeTciEvaluationStats,
}

/// Receives a report after every completed iteration of a TreeTCI run.
///
/// An observer is telemetry. It sees what the run has just recorded and can
/// neither change the run nor ask for an evaluation of the target function; that
/// separates it from [`EdgeVisitor`](crate::EdgeVisitor), which sets the edge
/// order and so the numerical result. The entry points take it as `&mut O`, so a
/// collector needs no interior mutability, and `&mut ()` observes nothing.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{
///     optimize_with_proposer, AllEdges, DefaultProposer, GlobalIndexBatch, TreeTCI2,
///     TreeTciGraph, TreeTciIteration, TreeTciObserver, TreeTciOptions,
/// };
///
/// /// Remembers the bond dimension after each iteration.
/// #[derive(Default)]
/// struct Progress {
///     max_bond_dims: Vec<usize>,
/// }
///
/// impl TreeTciObserver for Progress {
///     fn iteration(&mut self, it: &TreeTciIteration<'_>) {
///         self.max_bond_dims.push(it.max_bond_dim);
///     }
/// }
///
/// let graph = TreeTciGraph::linear_chain(3)?;
/// let mut state = TreeTCI2::<f64>::new(vec![2, 2, 2], graph)?;
/// state.add_global_pivots(&[vec![0, 0, 0]])?;
/// state.max_sample_value = 2.0;
/// let mut progress = Progress::default();
/// let result = optimize_with_proposer(
///     &mut state,
///     |batch: GlobalIndexBatch<'_>| Ok(vec![2.0; batch.n_points()]),
///     &TreeTciOptions::default(),
///     &DefaultProposer,
///     &AllEdges,
///     &mut progress,
/// )?;
/// assert_eq!(progress.max_bond_dims, result.ranks);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait TreeTciObserver {
    /// Called once per completed iteration, after the iteration's edge updates
    /// have been recorded and before the stopping checks, so the last iteration
    /// is reported too.
    ///
    /// The number of calls equals `ranks.len()` of the returned result, and call
    /// `k` reports `ranks[k]` and `errors[k]` exactly.
    fn iteration(&mut self, it: &TreeTciIteration<'_>);
}

/// The observer that ignores every iteration: pass `&mut ()` to observe nothing.
impl TreeTciObserver for () {
    fn iteration(&mut self, _it: &TreeTciIteration<'_>) {}
}
