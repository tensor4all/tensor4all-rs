use crate::error::Result as TreeTciResult;
use crate::evaluation::{BatchExecution, CachedExecution, Compatibility, Explicit};
use crate::{
    globalpivot::{find_global_pivots_erased, ScalarParts},
    update::update_edge,
    AllEdges, EdgeVisitor, GlobalIndexBatch, MultiIndex, PivotCandidateProposer, TreeTCI2,
    TreeTciEdge, TreeTciIteration, TreeTciObserver,
};
use anyhow::Result;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use tensor4all_core::CommonScalar;
use tensor4all_core::{MatrixLuciScalar as Scalar, RrLUOptions};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::interpolation::EvalContext;

/// The stopping condition reached by a TreeTCI optimization call.
///
/// Like the chain TCI's `TCI2Termination`, this distinguishes convergence
/// from resource limits. Convergence describes sampled pivot errors, not a
/// bound on the error over the entire index space.
/// Additional reasons may be added; exhaustive matches need a fallback arm.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::TreeTciTermination;
///
/// let reason = TreeTciTermination::MaxBondDimension;
/// assert_ne!(reason, TreeTciTermination::Converged);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TreeTciTermination {
    /// Three trailing iterations have error strictly below `tolerance`,
    /// final maximum rank equal to the minimum rank in that window, no newly
    /// accepted global pivots, and every edge's most recent update measured
    /// its bond error (see [`TreeTciTermination::DegenerateCandidateSet`]).
    ///
    /// Errors use the running maximum sample magnitude as their scale only
    /// when `scale_tolerance` is enabled.
    /// Fresh discovery is skipped on the final iteration, but enabled
    /// known-sample validation still runs there. This is a sampled criterion,
    /// not an exhaustive bound over the full index domain.
    Converged,
    /// The maximum rank reached `max_bond_dim` in three trailing iterations.
    ///
    /// This condition takes precedence over convergence even when the error
    /// is below tolerance. An enabled validation that still measures a positive
    /// error at or above tolerance returns `SampledErrorFloor { at_bond_cap: true }`
    /// instead. It does not imply that every edge has reached the cap.
    MaxBondDimension,
    /// `max_iter` iterations completed without either earlier stopping condition.
    ///
    /// A low error in the last iteration alone is insufficient for convergence.
    MaxIterations,
    /// Positive sampled residuals stay at or above tolerance at three equal
    /// completed-sweep pivot snapshots (including their order), or a saturated
    /// bond cap prevents resolving a measured residual. The snapshots span two equality comparisons;
    /// the first sweep may have changed its input pivots.
    /// The final entry of `errors` records this sampled floor on the configured
    /// scale. This is an observed stall, not a bound or proof of optimality.
    SampledErrorFloor {
        /// Whether the stall coincided with the caller's saturated bond cap.
        at_bond_cap: bool,
    },
    /// The convergence criterion of [`TreeTciTermination::Converged`] held on
    /// the recorded errors, but at least one edge could not measure its bond
    /// error in its most recent update.
    ///
    /// An edge's error is unmeasurable when a candidate side has only one
    /// entry, or the selected pivots exhaust the candidates at an endpoint
    /// of local dimension 1, while both full subtrees have more
    /// configurations than the stored rank. The recorded zero error
    /// then follows from candidate exhaustion rather than a measured residual
    /// (issues #1153 and #1352). If either subtree is fully represented at
    /// that rank, exhaustion is legitimate and the error counts as measured.
    /// Around a vertex of local dimension 1 whose adjacent bonds all have
    /// rank 1, the junction side is the product of single pivots (issue
    /// #1153). [`crate::DefaultProposer`] and
    /// [`crate::TruncatedDefaultProposer`] keep the pivots enriched there as
    /// permanent candidates, so an exactly rank-1 bond is measured and can be
    /// reported `Converged` (issue #1159); a custom proposer that drops them
    /// can reach this state. Higher-rank bonds can still exhaust their
    /// candidates even with the built-in proposers. The approximation may be
    /// inaccurate; enabling the global pivot search, supplying more initial
    /// pivots, or offering more candidates on that side can let the bonds grow.
    DegenerateCandidateSet,
}

/// Iteration diagnostics and stopping reason from optimizing a TreeTCI state.
///
/// The optimized state stays with the caller. [`crate::TreeTciRunResult`]
/// additionally contains a materialized tree tensor network.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{
///     optimize_default, TreeTCI2, TreeTciGraph, TreeTciOptions, TreeTciTermination,
/// };
///
/// let mut state = TreeTCI2::<f64>::new(vec![2, 2], TreeTciGraph::linear_chain(2)?)?;
/// state.add_global_pivots(&[vec![0, 0]])?;
/// let result = optimize_default(
///     &mut state, |batch| Ok(vec![2.0; batch.n_points()]),
///     &TreeTciOptions { seed: Some(0), ..Default::default() },
/// )?;
/// assert_eq!(result.termination, TreeTciTermination::Converged);
/// assert_eq!(result.ranks, vec![1; 3]);
/// assert_eq!(result.errors, vec![0.0; 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct TreeTciOptimizationResult {
    /// Maximum bond dimension after each completed iteration.
    pub ranks: Vec<usize>,
    /// Maximum sampled local/global residual after each iteration, normalized when requested.
    ///
    /// Has the same length as `ranks`. Each value combines local bond
    /// residuals, the bounded historical sample pool, all current retained pivot
    /// pairs, and new search observations. Validation also runs on final and
    /// capped sweeps. These are sampled
    /// diagnostics, not bounds on the full-network error.
    pub errors: Vec<f64>,
    /// The stopping condition evaluated by this call's optimization loop.
    pub termination: TreeTciTermination,
    /// Oracle evaluation and optional memo accounting for this optimization call.
    pub evaluation: crate::TreeTciEvaluationStats,
}

/// MVP optimization options for TreeTCI.
///
/// Controls convergence criteria, iteration limits, and bond dimension caps
/// for the tree tensor cross interpolation optimization loop.
///
/// # Defaults
///
/// | Field                     | Default       | Description                                         |
/// |---------------------------|---------------|-----------------------------------------------------|
/// | `tolerance`               | `1e-8`        | Stopping tolerance on the sampled local/global error |
/// | `max_iter`                | `20`          | Maximum number of edge-order iterations              |
/// | `max_bond_dim`            | `None`        | Maximum bond dimension (no cap by default)           |
/// | `scale_tolerance`         | `true`        | Scale tolerance by running max sample magnitude      |
/// | `enable_global_pivots`    | `true`        | Run automatic global pivot search after each sweep   |
/// | `nsearch`                 | `5`           | Random starting points for the global pivot search   |
/// | `global_search_starts`    | empty         | Extra caller-supplied starting points, walked first  |
/// | `max_nglobal_pivot`       | `5`           | Global pivots added per iteration                    |
/// | `tol_margin_global_search`| `10.0`        | Global pivot discovery margin over `abs_tol`        |
/// | `seed`                    | `None`        | RNG seed for the global pivot search                 |
/// | `validation_cache_bytes`  | 32 MiB        | Budget for replaying known oracle samples            |
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::TreeTciOptions;
///
/// // Default options
/// let opts = TreeTciOptions::default();
/// assert!((opts.tolerance - 1e-8).abs() < 1e-15);
/// assert_eq!(opts.max_iter, 20);
/// assert_eq!(opts.max_bond_dim, None);
/// assert!(opts.scale_tolerance);
/// assert!(opts.enable_global_pivots);
/// assert!(opts.global_search_starts.is_empty());
/// assert_eq!(opts.validation_cache_bytes, 32 * 1024 * 1024);
///
/// // Custom options for high-precision work
/// let opts = TreeTciOptions {
///     tolerance: 1e-12,
///     max_iter: 50,
///     max_bond_dim: Some(100),
///     scale_tolerance: true,
///     enable_global_pivots: true,
///     nsearch: 20,
///     global_search_starts: vec![vec![1, 1]],
///     max_nglobal_pivot: 10,
///     tol_margin_global_search: 5.0,
///     seed: Some(42),
///     evaluation_cache_bytes: None,
///     validation_cache_bytes: 32 * 1024 * 1024,
/// };
/// assert!((opts.tolerance - 1e-12).abs() < 1e-20);
/// assert_eq!(opts.max_iter, 50);
/// assert_eq!(opts.max_bond_dim, Some(100));
/// assert!(opts.enable_global_pivots);
/// assert_eq!(opts.global_search_starts, vec![vec![1, 1]]);
/// assert_eq!(opts.seed, Some(42));
/// ```
#[derive(Clone, Debug)]
pub struct TreeTciOptions {
    /// Stopping tolerance on the sampled local/global residual.
    ///
    /// The optimization loop monitors the maximum bond error across all edges
    /// and the largest residual observed by the enabled global search.
    /// When `scale_tolerance` is true, this error is divided by the running
    /// maximum observed sample magnitude, so the stopping threshold is
    /// `tolerance * max_sample_value`. Recommended range: `1e-6` to `1e-12`.
    /// Local LU truncation uses this same threshold in both global-search
    /// modes. A sampled violation with unchanged swept pivots reports a floor.
    /// The optimizer does not tighten or disable local truncation automatically.
    /// Zero is allowed: a zero residual neither satisfies strict convergence
    /// nor constitutes a sampled floor. An exact uncapped run then reaches
    /// `max_iter`; an exact capped run can report `MaxBondDimension`.
    /// Default: `1e-8`.
    pub tolerance: f64,

    /// Maximum number of edge-order iterations (outer sweeps).
    ///
    /// Each iteration visits all edges twice (two inner passes, in opposite
    /// directions with the default [`AllEdges`](crate::AllEdges)) and updates
    /// pivot sets. Typical values: 10--50. Default: `20`.
    pub max_iter: usize,

    /// Maximum bond dimension retained by the tcicore LUCI pivot substrate.
    ///
    /// Caps the number of pivots per edge bipartition. Use this to limit
    /// memory and computation for large problems. Once the maximal bond
    /// dimension has sat at this cap for three consecutive iterations, the
    /// loop stops even if `tolerance` has not been reached: every bond
    /// dimension of the result stays within the cap, but the approximation
    /// may be inaccurate when the cap is below the function's rank.
    /// Default: `None` (no cap).
    pub max_bond_dim: Option<usize>,

    /// Whether to scale the convergence tolerance by the running maximum
    /// observed sample magnitude.
    ///
    /// When `true`, the stopping criterion uses relative error
    /// `max_sampled_error / max_sample_value`, so `tolerance` is scaled by the
    /// largest sample seen so far. When `false`, the raw absolute sampled error is
    /// compared with `tolerance`. Default: `true`.
    pub scale_tolerance: bool,

    /// Whether to run an automatic global pivot search after each sweep.
    ///
    /// When `true`, the optimizer materializes the current approximation
    /// after each iteration and searches for multi-indices where
    /// `|f(idx) - tt(idx)|` is large, injecting the best finds via
    /// [`TreeTCI2::add_global_pivots`](crate::TreeTCI2::add_global_pivots).
    /// This recovers separated features that local pivot updates miss when
    /// the initial pivots sit in a single basin. A bounded recent-sample pool
    /// and its strongest residual witness are checked on every completed sweep,
    /// along with all current retained pivot pairs.
    /// Fresh discovery and injection are skipped on final or capped sweeps;
    /// known-sample validation still runs. Disabling this option also disables
    /// that history and its validation. Default: `true`.
    pub enable_global_pivots: bool,

    /// Number of random starting points for the global pivot search.
    ///
    /// Each starting point is locally optimized over all site coordinates.
    /// Larger values explore the index space more thoroughly at the cost of
    /// more evaluations. If no pivot qualifies and this is positive, the
    /// finder also walks every distinct retained full point across all edges.
    /// If none qualifies,
    /// each edge with a dimension-one endpoint gets up to `nsearch` restarts
    /// in each direction, preserving the other half of that retained point.
    /// Zero disables fresh random and confirmation starts; enabled validation
    /// of historical samples and current pivot pairs still runs.
    /// Drawn after [`Self::global_search_starts`]. Ignored when
    /// `enable_global_pivots` is `false`. Default: `5`.
    pub nsearch: usize,

    /// Extra starting points for the global pivot search, walked before the
    /// `nsearch` random ones.
    ///
    /// Each entry is a full-site multi-index in the state's site order: one
    /// coordinate per site, each below that site's local dimension. Use them
    /// to aim the search at a region the local pivot updates miss, such as a
    /// corner or edge of a patch; from a random start such a feature can look
    /// like a flat zero fiber, where the greedy walk stalls. A supplied start
    /// is walked even when `nsearch` is zero. Invalid entries are rejected
    /// before any evaluation: a wrong length or an out-of-range coordinate is
    /// a [`crate::TreeTciError::InvalidConfiguration`] from the optimizer and
    /// a [`crate::TreeTciError::Operation`] from a direct search call.
    /// Ignored (but still validated) when `enable_global_pivots` is `false`.
    /// Default: empty.
    pub global_search_starts: Vec<MultiIndex>,

    /// Maximum number of global pivots added per iteration.
    ///
    /// Ignored when `enable_global_pivots` is `false`. Default: `5`.
    pub max_nglobal_pivot: usize,

    /// Tolerance margin for the global pivot search.
    ///
    /// Exploratory candidates above `abs_tol * tol_margin_global_search`
    /// take priority. When none cross that margin, observed errors above
    /// `abs_tol` still qualify; unit-endpoint confirmation also uses
    /// `abs_tol`. Here `abs_tol` is the sweep's
    /// absolute tolerance (`tolerance * max_sample_value` when
    /// `scale_tolerance` is enabled). The value is always validated (it must
    /// be finite and nonnegative) but only consulted when `enable_global_pivots`
    /// is `true`. Default: `10.0`.
    pub tol_margin_global_search: f64,

    /// Random seed for the global pivot search.
    ///
    /// `None` seeds from OS entropy. Only used when `enable_global_pivots`
    /// is `true`. Default: `None`.
    pub seed: Option<u64>,

    /// Optional byte limit on the hash tables of the target memo.
    ///
    /// `None` (default) evaluates every request and preserves callback semantics.
    /// Use `Some(256 * 1024 * 1024)` for expensive deterministic targets.
    /// `Some(0)` deduplicates only within each batch and retains no values.
    /// At the limit the oldest generation of entries is evicted (reported as
    /// `evicted_entries`) and the new value is kept; while a table grows, the
    /// outgoing one is alive too, so the transient peak is at most 1.5 × the
    /// limit + 8 bytes.
    /// Memoization requires a fixed value for each point throughout the call.
    /// Edge candidate matrices are looked up through cache-owned side keys,
    /// which exist for the `u64` and `u128` key widths only, so an index space
    /// whose selected width exceeds 128 bits (but still fits the cache's 1024
    /// bits) returns [`crate::TreeTciError::Operation`] holding
    /// `CacheKeyError::SideKeyUnsupported` at run setup rather than falling back
    /// to whole-point lookups; a space beyond 1024 bits fails at cache
    /// construction with `CacheKeyError::Overflow`. Callback errors and wrong
    /// output lengths are never cached. The cache belongs to this run, is
    /// shared across updates/searches/final materialization, and is dropped
    /// at return. Bytes count the retained tables (buckets, entries, control
    /// bytes): the limit bounds those tables, not the transient per-call batch
    /// buffers (which grow with the points of one call), and not RSS overall.
    pub evaluation_cache_bytes: Option<usize>,
    /// Accounted byte budget for historical validation samples. Default: 32 MiB.
    /// Includes point/value payload and estimated table overhead, excluding
    /// spare capacity and RSS. Independent of oracle memoization. The history
    /// belongs to initialization and optimization of one call, is frozen
    /// before returning the requested materialization, and is dropped at return.
    /// As with interpolation itself, target values must be fixed during the
    /// call. At capacity, old unprotected slots are replaced in rotating order.
    /// The strongest residual witness measured on the current approximation
    /// is protected until its next validation; a stronger measured witness
    /// may replace it. The budget limits historical coverage without stopping
    /// interpolation. Current retained pivot pairs are checked independently.
    /// Zero disables historical storage; ignored when global checking is disabled.
    pub validation_cache_bytes: usize,
}

impl TreeTciOptions {
    pub(crate) fn validate(&self) -> TreeTciResult<()> {
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(crate::TreeTciError::InvalidConfiguration {
                message: "tolerance must be finite and nonnegative".to_string(),
            });
        }
        if self.max_iter == 0 {
            return Err(crate::TreeTciError::InvalidConfiguration {
                message: "max_iter must be positive".to_string(),
            });
        }
        if self.max_bond_dim == Some(0) {
            return Err(crate::TreeTciError::InvalidConfiguration {
                message: "max_bond_dim must be positive when specified".to_string(),
            });
        }
        if !self.tol_margin_global_search.is_finite() || self.tol_margin_global_search < 0.0 {
            return Err(crate::TreeTciError::InvalidConfiguration {
                message: "tol_margin_global_search must be finite and nonnegative".to_string(),
            });
        }
        Ok(())
    }
}

impl Default for TreeTciOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 20,
            max_bond_dim: None,
            scale_tolerance: true,
            enable_global_pivots: true,
            nsearch: 5,
            global_search_starts: Vec::new(),
            max_nglobal_pivot: 5,
            tol_margin_global_search: 10.0,
            seed: None,
            evaluation_cache_bytes: None,
            validation_cache_bytes: 32 * 1024 * 1024,
        }
    }
}

/// Optimize a TreeTCI state with the MVP strategy choices:
/// `AllEdges` visitation and [`DefaultProposer`](crate::DefaultProposer).
///
/// Returns [`TreeTciOptimizationResult`] with per-iteration ranks, errors,
/// and the stopping reason defined by [`TreeTciTermination`].
///
/// This is a convenience wrapper around [`optimize_with_proposer`] with the
/// default neighbor-product proposer.
///
/// The evaluator may be called several times per edge update; see
/// [`GlobalIndexBatch`](crate::GlobalIndexBatch#batch-sizes) for the batch
/// sizes.
///
/// Repeated calls on the same function/topology/dimensions support increasing
/// `max_bond_dim` or removing the cap. Current pivots and sampled normalization
/// scale carry over; proposers retain the current edge's pivots directly.
/// This call starts new diagnostics, a new convergence window and its own iteration budget,
/// including the final-iteration global-search skip. It need not match one
/// longer run. A failing evaluator can leave a partially updated state.
///
/// # Errors
///
/// Returns [`TreeTciError::InvalidConfiguration`](crate::TreeTciError::InvalidConfiguration)
/// for invalid options, and [`TreeTciError::Operation`](crate::TreeTciError::Operation)
/// when the evaluator fails, a pivot set is missing or malformed, or
/// materialization fails.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{
///     optimize_default, GlobalIndexBatch, TreeTCI2, TreeTciEdge,
///     TreeTciGraph, TreeTciOptions,
/// };
/// use anyhow::Result;
///
/// let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
/// let local_dims = vec![2, 2];
/// let mut state = TreeTCI2::<f64>::new(local_dims, graph).unwrap();
/// state.add_global_pivots(&[vec![0, 0]]).unwrap();
/// state.set_max_sample_value(1.0);
///
/// let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
///     let mut vals = Vec::with_capacity(batch.n_points());
///     for p in 0..batch.n_points() {
///         let i = batch.get(0, p).unwrap();
///         let j = batch.get(1, p).unwrap();
///         vals.push(if i == j { 1.0 } else { 0.0 });
///     }
///     Ok(vals)
/// };
///
/// let options = TreeTciOptions { tolerance: 1e-10, max_iter: 5, ..Default::default() };
/// let tensor4all_treetci::TreeTciOptimizationResult { ranks, errors, .. } = optimize_default(&mut state, evaluate, &options).unwrap();
///
/// // One entry per sweep actually run; the loop stops early once converged,
/// // so this may be less than max_iter (5).
/// assert_eq!(ranks, vec![2; 3]);
/// assert_eq!(ranks.len(), errors.len());
/// assert!(errors.last().copied().unwrap_or(1.0) < 1e-8);
/// ```
pub fn optimize_default<T, F>(
    state: &mut TreeTCI2<T>,
    evaluate: F,
    options: &TreeTciOptions,
) -> TreeTciResult<TreeTciOptimizationResult>
where
    T: Scalar + CommonScalar + FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts,
    F: FnMut(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    options.validate()?;
    optimize_with_proposer(
        state,
        evaluate,
        options,
        &crate::DefaultProposer,
        &AllEdges,
        &mut (),
    )
}

/// Optimize a TreeTCI state with a caller-supplied pivot candidate proposer
/// and edge visitor.
///
/// Returns [`TreeTciOptimizationResult`] with per-iteration ranks, errors,
/// and the stopping reason defined by [`TreeTciTermination`].
///
/// Use this when you need a custom proposer (e.g., [`SimpleProposer`](crate::SimpleProposer)
/// or [`TruncatedDefaultProposer`](crate::TruncatedDefaultProposer)) or a
/// custom edge visit order. Pass [`AllEdges`] to keep the default order.
///
/// # Edge visit order
///
/// `visitor` decides which edge is updated and when. It is called once per
/// inner pass, and its result must be a permutation of the graph's edges:
/// every edge exactly once, with no missing, duplicated, or unknown edge. That
/// contract is checked before the pass updates any state, so an invalid order
/// cannot leave a stale bond error on an edge or let the convergence window
/// pass with unswept edges. A contract violation returns
/// [`TreeTciError::InvalidConfiguration`](crate::TreeTciError::InvalidConfiguration).
/// Different valid orders generally reach different approximations; the
/// documented sequence is exactly the order returned by the visitor's
/// [`EdgeVisitor::visit_order_in_pass`] on each pass. [`AllEdges`] sweeps the
/// sorted edges forward on the first pass of an iteration and backward on the
/// second.
///
/// # Observing iterations
///
/// `observer` receives a [`TreeTciIteration`] after each completed iteration,
/// including the last, with the bond dimension and error the result records for
/// it and the evaluation accounting so far; pass `&mut ()` to observe nothing.
/// An observer cannot change the run and adds no evaluation of the target
/// function. The error is the one recorded before that iteration's global pivot
/// injection.
///
/// The evaluator may be called several times per edge update; see
/// [`GlobalIndexBatch`](crate::GlobalIndexBatch#batch-sizes) for the batch
/// sizes.
///
/// Candidate generation uses one ChaCha8 stream seeded by
/// [`PivotCandidateProposer::seed`] per optimization call; global searches use
/// a separate ChaCha8 stream governed by [`TreeTciOptions::seed`]. Use
/// [`optimize_with_proposer_with_rng`] to advance one caller-owned stream
/// through both operations, including across continued calls.
///
/// Repeated calls on the same function/topology/dimensions support increasing
/// `max_bond_dim` or removing the cap. Current pivots and sampled normalization
/// scale carry over; proposers retain the current edge's pivots directly.
/// This call starts new diagnostics, a new convergence window and its own iteration budget,
/// including the final-iteration global-search skip. It need not match one
/// longer run. A failing evaluator can leave a partially updated state.
///
/// # Errors
///
/// Returns [`TreeTciError::InvalidConfiguration`](crate::TreeTciError::InvalidConfiguration)
/// for invalid options or a visitor order that is not a permutation of the
/// graph's edges, and [`TreeTciError::Operation`](crate::TreeTciError::Operation)
/// when the evaluator fails, a pivot set is missing or malformed, or
/// materialization fails.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{
///     optimize_with_proposer, AllEdges, GlobalIndexBatch, SimpleProposer,
///     TreeTCI2, TreeTciEdge, TreeTciGraph, TreeTciOptions,
/// };
/// use anyhow::Result;
///
/// let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
/// let mut state = TreeTCI2::<f64>::new(vec![2, 2], graph).unwrap();
/// state.add_global_pivots(&[vec![0, 0]]).unwrap();
/// state.set_max_sample_value(1.0);
///
/// let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
///     let mut vals = Vec::with_capacity(batch.n_points());
///     for p in 0..batch.n_points() {
///         let i = batch.get(0, p).unwrap();
///         let j = batch.get(1, p).unwrap();
///         vals.push(if i == j { 1.0 } else { 0.0 });
///     }
///     Ok(vals)
/// };
///
/// let proposer = SimpleProposer::seeded(42);
/// let options = TreeTciOptions { tolerance: 1e-10, max_iter: 3, ..Default::default() };
/// let tensor4all_treetci::TreeTciOptimizationResult { ranks, errors, .. } = optimize_with_proposer(
///     &mut state, evaluate, &options, &proposer, &AllEdges, &mut (),
/// ).unwrap();
///
/// // One entry per sweep actually run; the loop stops early once converged,
/// // so this may be less than max_iter (3).
/// assert_eq!(ranks, vec![2; 3]);
/// assert_eq!(ranks.len(), errors.len());
/// ```
pub fn optimize_with_proposer<T, F, P, V, O>(
    state: &mut TreeTCI2<T>,
    evaluate: F,
    options: &TreeTciOptions,
    proposer: &P,
    visitor: &V,
    observer: &mut O,
) -> TreeTciResult<TreeTciOptimizationResult>
where
    T: Scalar + CommonScalar + FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts,
    F: FnMut(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
    V: EdgeVisitor,
    O: TreeTciObserver,
{
    options.validate()?;
    // The public optimizer keeps the process-global route: no caller-owned session.
    let mut evaluate = evaluate;
    let mut execution = Compatibility::new(&mut evaluate);
    with_seeded_streams(options, proposer, |streams| {
        optimize_memoized(
            state,
            &mut execution,
            options,
            proposer,
            visitor,
            observer,
            streams,
        )
    })
}

/// The single optimization body for every frontend: one memoizing owner around the
/// caller's execution route, then the sweep loop.
pub(crate) fn optimize_memoized<T, P, V, O>(
    state: &mut TreeTCI2<T>,
    execution: &mut dyn BatchExecution<T>,
    options: &TreeTciOptions,
    proposer: &P,
    visitor: &V,
    observer: &mut O,
    streams: &mut RandomStreams<'_>,
) -> TreeTciResult<TreeTciOptimizationResult>
where
    T: Scalar + CommonScalar + FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts,
    P: PivotCandidateProposer,
    V: EdgeVisitor,
    O: TreeTciObserver,
{
    let mut cached = CachedExecution::new(
        execution,
        &state.local_dims,
        options.evaluation_cache_bytes,
        options
            .enable_global_pivots
            .then_some(options.validation_cache_bytes),
    )?;
    let mut result = optimize_with_streams(
        state,
        &mut cached,
        options,
        proposer,
        visitor,
        observer,
        streams,
    )?;
    result.evaluation = cached.stats();
    Ok(result)
}

/// Optimize with an explicit proposer whose run-time execution runs in a caller-owned CPU context.
///
/// The same algorithm, options, stopping rule and diagnostics as [`optimize_with_proposer`]; what differs is the
/// resource the stage's session-bound work runs in. Without a context it goes through the process-global CPU
/// context; with one, the pivot search's materialization solve runs on `context`, as
/// [`crossinterpolate2_in`](crate::crossinterpolate2_in) does for a whole run.
///
/// The caller's evaluator and the pivot search's readout are unchanged and still use their existing routes, so this
/// is a caller-owned budget for the stage's session-bound work, not for everything the run touches.
///
/// # Errors
///
/// Returns [`crate::TreeTciError::InvalidConfiguration`] for invalid options,
/// [`crate::TreeTciError::InvalidGraph`] for a malformed topology,
/// [`crate::TreeTciError::IndexOutOfBounds`] for an out-of-range candidate, and
/// [`crate::TreeTciError::Operation`] when the evaluator or the pivot search's materialization solve fails.
/// A failing evaluator can leave a partially updated state.
pub fn optimize_with_proposer_in<T, F, P, O>(
    state: &mut TreeTCI2<T>,
    evaluate: F,
    options: &TreeTciOptions,
    proposer: &P,
    observer: &mut O,
    context: &tensor4all_tensorbackend::CpuExecutionContext,
) -> TreeTciResult<TreeTciOptimizationResult>
where
    T: Scalar + CommonScalar + FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts,
    F: FnMut(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
    O: TreeTciObserver,
{
    options.validate()?;
    let mut evaluate = evaluate;
    context
        .with_concrete_session(|session| {
            let mut contextual =
                |batch: GlobalIndexBatch<'_>, _ctx: &mut EvalContext<'_>| evaluate(batch);
            let mut context = crate::evaluation::serial_context(session);
            let mut execution = Explicit::new(&mut context, &mut contextual);
            with_seeded_streams(options, proposer, |streams| {
                optimize_memoized(
                    state,
                    &mut execution,
                    options,
                    proposer,
                    &AllEdges,
                    observer,
                    streams,
                )
            })
        })
        .map_err(|error| crate::TreeTciError::from(anyhow::Error::new(error)))?
}

/// Optimize with a caller-owned random stream.
///
/// Returns [`TreeTciOptimizationResult`], including the stopping reason.
/// Same as [`optimize_with_proposer`], but consumes `rng` for every global
/// pivot search of the run instead of deriving one generator from
/// [`TreeTciOptions::seed`], so the caller can reproduce or advance the run's
/// randomness and share one stream across several runs. Candidate generation
/// and global searches both consume the supplied stream directly. The seed
/// stored in a proposer and [`TreeTciOptions::seed`] are ignored here.
///
/// Repeated calls on the same function/topology/dimensions support increasing
/// `max_bond_dim` or removing the cap. Current pivots and sampled normalization
/// scale carry over; proposers retain the current edge's pivots directly.
/// This call starts new diagnostics, a new convergence window and its own iteration budget,
/// including the final-iteration global-search skip. It need not match one
/// longer run. A failing evaluator can leave a partially updated state.
///
/// # Errors
/// Returns [`TreeTciError::InvalidConfiguration`](crate::TreeTciError::InvalidConfiguration)
/// for invalid options, and [`TreeTciError::Operation`](crate::TreeTciError::Operation)
/// when the evaluator fails, a pivot set is missing or malformed, or
/// materialization fails.
/// # Examples
///
/// ```
/// use anyhow::Result;
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_treetci::{
///     optimize_with_proposer_with_rng, GlobalIndexBatch, SimpleProposer, TreeTCI2, TreeTciEdge,
///     TreeTciGraph, TreeTciOptions,
/// };
///
/// let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
/// let mut state = TreeTCI2::<f64>::new(vec![2, 2], graph).unwrap();
/// state.add_global_pivots(&[vec![0, 0]]).unwrap();
/// state.set_max_sample_value(1.0);
///
/// let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
///     let mut vals = Vec::with_capacity(batch.n_points());
///     for p in 0..batch.n_points() {
///         let i = batch.get(0, p).unwrap();
///         let j = batch.get(1, p).unwrap();
///         vals.push(if i == j { 1.0 } else { 0.0 });
///     }
///     Ok(vals)
/// };
///
/// let mut rng = ChaCha8Rng::seed_from_u64(3);
/// let options = TreeTciOptions { tolerance: 1e-10, max_iter: 1, ..Default::default() };
/// let tensor4all_treetci::TreeTciOptimizationResult { ranks, errors, .. } = optimize_with_proposer_with_rng(
///     &mut state, evaluate, &options, &SimpleProposer::default(), &mut (), &mut rng,
/// ).unwrap();
/// assert_eq!(ranks.len(), errors.len());
/// ```
pub fn optimize_with_proposer_with_rng<T, F, P, R, O>(
    state: &mut TreeTCI2<T>,
    evaluate: F,
    options: &TreeTciOptions,
    proposer: &P,
    observer: &mut O,
    rng: &mut R,
) -> TreeTciResult<TreeTciOptimizationResult>
where
    T: Scalar + CommonScalar + FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts,
    F: FnMut(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
    O: TreeTciObserver,
    R: rand::Rng + ?Sized,
{
    // Erase the caller's RNG type once so the run below is instantiated once
    // per scalar type instead of once per (scalar, RNG) pair.
    options.validate()?;
    let mut stream: &mut R = rng;
    let mut streams = RandomStreams::Shared(&mut stream);
    let mut evaluate = evaluate;
    let mut execution = Compatibility::new(&mut evaluate);
    optimize_memoized(
        state,
        &mut execution,
        options,
        proposer,
        &AllEdges,
        observer,
        &mut streams,
    )
}

/// Construct named seeded streams once at a high-level run boundary.
pub(crate) fn with_seeded_streams<P, T>(
    options: &TreeTciOptions,
    proposer: &P,
    run: impl FnOnce(&mut RandomStreams<'_>) -> T,
) -> T
where
    P: PivotCandidateProposer,
{
    let searches_run = options.enable_global_pivots && options.max_iter > 1;
    let mut global = match (searches_run, options.seed) {
        (true, None) => ChaCha8Rng::from_os_rng(),
        (_, seed) => ChaCha8Rng::seed_from_u64(seed.unwrap_or(0)),
    };
    let mut candidates = ChaCha8Rng::seed_from_u64(proposer.seed());
    run(&mut RandomStreams::Separate {
        candidates: &mut candidates,
        global: &mut global,
    })
}

// Dispatch at candidate/search batch boundaries, never per tensor element.
pub(crate) enum RandomStreams<'a> {
    Shared(&'a mut dyn rand::RngCore),
    Separate {
        candidates: &'a mut dyn rand::RngCore,
        global: &'a mut dyn rand::RngCore,
    },
}

impl RandomStreams<'_> {
    fn candidates(&mut self) -> &mut dyn rand::RngCore {
        match self {
            Self::Shared(rng) => *rng,
            Self::Separate { candidates, .. } => *candidates,
        }
    }
    fn global(&mut self) -> &mut dyn rand::RngCore {
        match self {
            Self::Shared(rng) => *rng,
            Self::Separate { global, .. } => *global,
        }
    }
}

/// Check the caller-supplied global search starts against the state.
///
/// `TreeTciOptions::validate` is shape-independent; this check needs the
/// state's local dimensions. It runs before any sweep, so an invalid start
/// fails without evaluating the target.
fn validate_global_search_starts(
    options: &TreeTciOptions,
    local_dims: &[usize],
) -> TreeTciResult<()> {
    for (index, start) in options.global_search_starts.iter().enumerate() {
        if start.len() != local_dims.len() {
            return Err(crate::TreeTciError::InvalidConfiguration {
                message: format!(
                    "global_search_starts[{index}] has {} coordinates but the state has {} sites",
                    start.len(),
                    local_dims.len()
                ),
            });
        }
        for (site, (&coordinate, &dimension)) in start.iter().zip(local_dims).enumerate() {
            if coordinate >= dimension {
                return Err(crate::TreeTciError::InvalidConfiguration {
                    message: format!(
                        "global_search_starts[{index}] has coordinate {coordinate} at site \
                         {site}, out of range for local dimension {dimension}"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Check that `order` is a permutation of the tree's edges.
///
/// The truncated proposer relies on each edge's kept pivots matching a
/// start-of-pass snapshot, so an order that omits, duplicates or invents an
/// edge would leave stale bond errors and could satisfy convergence falsely.
fn validate_edge_order<T>(state: &TreeTCI2<T>, order: &[TreeTciEdge]) -> TreeTciResult<()> {
    let expected = state.graph.edges();
    let expected_set: HashSet<TreeTciEdge> = expected.iter().copied().collect();
    let mut seen = HashSet::with_capacity(order.len());
    for &edge in order {
        if !expected_set.contains(&edge) {
            return Err(crate::TreeTciError::InvalidConfiguration {
                message: format!(
                    "edge visitor returned unknown edge {edge:?}; the tree has {} edges",
                    expected.len()
                ),
            });
        }
        if !seen.insert(edge) {
            return Err(crate::TreeTciError::InvalidConfiguration {
                message: format!("edge visitor returned edge {edge:?} more than once per pass"),
            });
        }
    }
    if let Some(missing) = expected.iter().find(|edge| !seen.contains(edge)) {
        return Err(crate::TreeTciError::InvalidConfiguration {
            message: format!(
                "edge visitor omitted edge {missing:?}; each edge must be visited exactly once per pass"
            ),
        });
    }
    Ok(())
}

pub(crate) fn optimize_with_streams<T, P, V, O>(
    state: &mut TreeTCI2<T>,
    execution: &mut dyn BatchExecution<T>,
    options: &TreeTciOptions,
    proposer: &P,
    visitor: &V,
    observer: &mut O,
    streams: &mut RandomStreams<'_>,
) -> TreeTciResult<TreeTciOptimizationResult>
where
    T: Scalar + CommonScalar + FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts,
    P: PivotCandidateProposer,
    V: EdgeVisitor,
    O: TreeTciObserver,
{
    options.validate()?;
    validate_global_search_starts(options, &state.local_dims)?;

    // Bonds around a vertex of local dimension 1 cannot grow from rank 1 by
    // local updates (#1153); enrich them before the first sweep, as xfac does
    // when it builds a `TensorTreeCI`.
    state.enrich_pivots_around_unit_vertices()?;

    let mut ranks = Vec::new();
    let mut errors = Vec::new();
    let mut nglobal_pivots_history: Vec<usize> = Vec::new();
    let mut termination = TreeTciTermination::MaxIterations;
    const INNER_EDGE_PASSES: usize = 2;
    // Mirrors `tensor4all-tensorci`'s `TensorCI2Options::ncheck_history` default
    // (see `convergence_criterion` in tensorci2.rs, itself a port of Julia's
    // `convergencecriterion`). A single below-tolerance sweep is not reliable:
    // the bond-error estimate can dip before the pivot search has actually
    // stabilized. Like TensorCI2, the window must also contain no newly added
    // global pivots: an iteration that injected pivots has not yet swept them,
    // so its error estimate is stale with respect to those pivots.
    const NCHECK_HISTORY: usize = 3;

    // Equal maximum ranks alone do not imply a stall: replacing pivots at the
    // same rank can still discover an independent direction (#1352). Retain
    // one exact pivot snapshot and require three equal completed-sweep states
    // (two equality comparisons). A changed snapshot starts a new window at one.
    let mut previous_pivots = options.enable_global_pivots.then(|| state.ijset.clone());
    let mut unchanged_pivot_sweeps = 0;
    for iteration in 0..options.max_iter {
        for pass in 0..INNER_EDGE_PASSES {
            let error_scale = if options.scale_tolerance && state.max_sample_value > 0.0 {
                state.max_sample_value
            } else {
                1.0
            };
            let kernel_options = RrLUOptions {
                rel_tol: 1e-14,
                abs_tol: options.tolerance * error_scale,
                max_bond_dim: options.max_bond_dim.unwrap_or(usize::MAX),
                left_orthogonal: true,
            };

            // The visitor must return a permutation of the tree's edges for
            // this pass. Validate before any update mutates state: an omitted
            // edge would keep its previous bond error and could satisfy
            // convergence falsely. The two subtree keys of an edge are not
            // changed by updates of other edges, so proposers can retain the
            // edge's current pivots without a full snapshot.
            let order = visitor.visit_order_in_pass(state, pass);
            validate_edge_order(state, &order)?;
            state.flush_pivot_errors();

            for edge in order {
                update_edge(
                    state,
                    edge,
                    execution,
                    &kernel_options,
                    proposer,
                    streams.candidates(),
                )?;
            }
        }

        if let Some(previous) = &mut previous_pivots {
            let unchanged = previous.len() == state.ijset.len()
                && state.ijset.iter().all(|(key, pivots)| {
                    previous.get(key).is_some_and(|old| {
                        old.shape() == pivots.shape() && old.data() == pivots.data()
                    })
                });
            if unchanged {
                unchanged_pivot_sweeps += 1;
            } else {
                unchanged_pivot_sweeps = 1;
                previous.clone_from(&state.ijset);
            }
        }
        let max_bond_dim = state.max_bond_dim();
        ranks.push(max_bond_dim);
        // Mirrors `TreeTCI.jl`'s `convergencecriterion` third disjunct
        // (branch `local-fix-convergence`, commit 06563dd): once the rank has
        // saturated at `max_bond_dim` for the whole trailing window, further
        // sweeps cannot reduce the error, so waiting for it to also cross
        // `tolerance` would just burn the remaining `max_iter` sweeps.
        //
        // The cap is known before discovery. We still validate known samples
        // on the capped approximation, without injecting further pivots. Injected pivots are only
        // consistent once a sweep has processed them: the injection projects
        // each pivot onto both sides of every edge with per-side
        // deduplication, so the two sides of an edge can hold different
        // column counts (and more than `max_bond_dim`) until the next sweep.
        // Stopping right after an injection would hand that unswept state to
        // `to_treetn`, which rejects it ("bond ranks disagree", #692).
        let bond_dim_saturated = ranks.len() >= NCHECK_HISTORY
            && options.max_bond_dim.is_some_and(|cap| {
                ranks[ranks.len() - NCHECK_HISTORY..]
                    .iter()
                    .all(|&r| r >= cap)
            });

        // Validate known samples on every completed approximation. Fresh
        // discoveries can be injected only when another sweep will process
        // them; final/capped stops still measure the known-sample floor.
        let mut sampled_error = state.max_bond_error();
        let mut pending_pivots = Vec::new();
        if options.enable_global_pivots {
            let error_scale = if options.scale_tolerance && state.max_sample_value > 0.0 {
                state.max_sample_value
            } else {
                1.0
            };
            let can_insert = !bond_dim_saturated && iteration + 1 < options.max_iter;
            let search = find_global_pivots_erased(
                state,
                execution,
                if can_insert { options.nsearch } else { 0 },
                if can_insert {
                    &options.global_search_starts
                } else {
                    &[]
                },
                if can_insert {
                    options.max_nglobal_pivot
                } else {
                    0
                },
                options.tol_margin_global_search,
                options.tolerance * error_scale,
                streams.global(),
            )?;
            sampled_error = sampled_error.max(search.max_error);
            if can_insert {
                nglobal_pivots_history.push(search.pivots.len());
                pending_pivots = search.pivots;
            } else {
                nglobal_pivots_history.push(0);
            }
        } else {
            nglobal_pivots_history.push(0);
        }

        let normalized_error = if options.scale_tolerance && state.max_sample_value > 0.0 {
            sampled_error / state.max_sample_value
        } else {
            sampled_error
        };
        errors.push(normalized_error);
        // Report the same pre-injection approximation that the local updates
        // and global residual search measured. Newly injected pivots are
        // processed by the next sweep.
        observer.iteration(&TreeTciIteration {
            iteration,
            max_bond_dim,
            error: normalized_error,
            evaluation: &execution.stats(),
        });
        if bond_dim_saturated {
            termination = if options.enable_global_pivots
                && normalized_error > 0.0
                && normalized_error >= options.tolerance
            {
                TreeTciTermination::SampledErrorFloor { at_bond_cap: true }
            } else {
                TreeTciTermination::MaxBondDimension
            };
            break;
        }

        // This Rust loop applies an early-convergence break rather than
        // unconditionally running every requested sweep. Its error comparison
        // uses the normalized errors when requested, consistent with tolerance.
        // The stopping contract and upstream comparison are recorded in
        // docs/design/treetci-termination.md (issues #834 and #835).
        //
        // Error-below-tolerance + rank-stable over the
        // trailing window, with no global pivots added anywhere in the window
        // (including this iteration's search, so a stop here never leaves
        // unswept injected pivots). The bond-dimension saturation disjunct is
        // checked above, before the search.
        if errors.len() >= NCHECK_HISTORY {
            let n = errors.len();
            let last_errors = &errors[n - NCHECK_HISTORY..];
            let last_ranks = &ranks[n - NCHECK_HISTORY..];
            let last_ngp = &nglobal_pivots_history[n - NCHECK_HISTORY..];
            let measured_stall = options.enable_global_pivots
                && unchanged_pivot_sweeps >= NCHECK_HISTORY
                && last_errors
                    .iter()
                    .all(|&e| e > 0.0 && e >= options.tolerance)
                && last_ranks.iter().all(|&rank| rank == last_ranks[0]);
            if measured_stall {
                termination = TreeTciTermination::SampledErrorFloor { at_bond_cap: false };
                break;
            }
            let errors_converged = last_errors.iter().all(|&e| e < options.tolerance);
            let no_global_pivots = last_ngp.iter().all(|&n| n == 0);
            let rank_stable = last_ranks.iter().min().copied().unwrap_or(0)
                == last_ranks.last().copied().unwrap_or(0);
            if errors_converged && no_global_pivots && rank_stable {
                // A zero error from exhausted candidates is not evidence of
                // convergence below the full bipartition rank (#1153, #1352).
                // Stop where `Converged` would have stopped and report that
                // the error was not seen.
                termination = if state.unmeasured_bonds.is_empty() {
                    TreeTciTermination::Converged
                } else {
                    TreeTciTermination::DegenerateCandidateSet
                };
                break;
            }
        }
        // Only a continuing iteration may leave injected pivots for its next
        // sweep. Floor stops return the measured, materializable
        // approximation rather than an unswept pivot state (#692).
        state.add_global_pivots(&pending_pivots)?;
    }

    Ok(TreeTciOptimizationResult {
        ranks,
        errors,
        termination,
        evaluation: Default::default(),
    })
}

#[cfg(test)]
mod tests;
