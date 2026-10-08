use crate::error::Result as TreeTciResult;
use crate::{
    globalpivot::{find_global_pivots_erased, ScalarParts},
    update::update_edge,
    AllEdges, EdgeVisitor, GlobalIndexBatch, PivotCandidateProposer, TreeTCI2,
};
use anyhow::Result;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::CommonScalar;
use tensor4all_core::{MatrixLuciScalar as Scalar, RrLUOptions};
use tensor4all_tensorbackend::FullPivLuScalar;

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
    /// final maximum rank equal to the minimum rank in that window, and
    /// no newly accepted global pivots.
    ///
    /// Errors are normalized only when `normalize_error` is enabled.
    /// Disabled searches and the search skipped on the final iteration
    /// contribute zero pivots; this reason does not certify an exhaustive
    /// residual check or a global search after that final iteration.
    Converged,
    /// The maximum rank reached `max_bond_dim` in three trailing iterations.
    ///
    /// This condition is checked before global search and convergence, so
    /// it takes precedence even when the sampled error is below tolerance.
    /// It does not imply that every edge has reached the cap.
    MaxBondDimension,
    /// `max_iter` iterations completed without either earlier stopping condition.
    ///
    /// A low error in the last iteration alone is insufficient for convergence.
    MaxIterations,
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
    /// Maximum sampled bond error after each iteration, normalized when requested.
    ///
    /// Has the same length as `ranks`; these are not full-network residuals.
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
/// | `tolerance`               | `1e-8`        | Relative stopping tolerance on normalized bond error |
/// | `max_iter`                | `20`          | Maximum number of edge-order iterations              |
/// | `max_bond_dim`            | `None`        | Maximum bond dimension (no cap by default)           |
/// | `normalize_error`         | `true`        | Normalize error by maximum sample magnitude          |
/// | `enable_global_pivots`    | `true`        | Run automatic global pivot search after each sweep   |
/// | `nsearch`                 | `5`           | Random starting points for the global pivot search   |
/// | `max_nglobal_pivot`       | `5`           | Global pivots added per iteration                    |
/// | `tol_margin_global_search`| `10.0`        | Global pivot acceptance margin over `abs_tol`        |
/// | `seed`                    | `None`        | RNG seed for the global pivot search                 |
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
/// assert!(opts.normalize_error);
/// assert!(opts.enable_global_pivots);
///
/// // Custom options for high-precision work
/// let opts = TreeTciOptions {
///     tolerance: 1e-12,
///     max_iter: 50,
///     max_bond_dim: Some(100),
///     normalize_error: true,
///     enable_global_pivots: true,
///     nsearch: 20,
///     max_nglobal_pivot: 10,
///     tol_margin_global_search: 5.0,
///     seed: Some(42),
///     evaluation_cache_bytes: None,
/// };
/// assert!((opts.tolerance - 1e-12).abs() < 1e-20);
/// assert_eq!(opts.max_iter, 50);
/// assert_eq!(opts.max_bond_dim, Some(100));
/// assert!(opts.enable_global_pivots);
/// assert_eq!(opts.seed, Some(42));
/// ```
#[derive(Clone, Debug)]
pub struct TreeTciOptions {
    /// Relative stopping tolerance on the normalized bond error.
    ///
    /// The optimization loop monitors the maximum bond error across all edges.
    /// When `normalize_error` is true, this error is divided by the maximum
    /// observed sample magnitude. Recommended range: `1e-6` to `1e-12`.
    /// Default: `1e-8`.
    pub tolerance: f64,

    /// Maximum number of edge-order iterations (outer sweeps).
    ///
    /// Each iteration visits all edges twice (two inner passes) and updates
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

    /// Whether to normalize the bond error by the maximum observed sample magnitude.
    ///
    /// When `true`, the stopping criterion uses relative error
    /// `max_bond_error / max_sample_value`. When `false`, the raw absolute
    /// bond error is used. Default: `true`.
    pub normalize_error: bool,

    /// Whether to run an automatic global pivot search after each sweep.
    ///
    /// When `true`, the optimizer materializes the current approximation
    /// after each iteration and searches for multi-indices where
    /// `|f(idx) - tt(idx)|` is large, injecting the best finds via
    /// [`TreeTCI2::add_global_pivots`](crate::TreeTCI2::add_global_pivots).
    /// This recovers separated features that local pivot updates miss when
    /// the initial pivots sit in a single basin. The search is skipped after
    /// the `max_iter`-th sweep and after the sweep at which the loop stops
    /// because the bond dimension saturated at `max_bond_dim`, so every
    /// injected pivot is processed by a later sweep. Default: `true`.
    pub enable_global_pivots: bool,

    /// Number of random starting points for the global pivot search.
    ///
    /// Each starting point is locally optimized over all site coordinates.
    /// Larger values explore the index space more thoroughly at the cost of
    /// more evaluations. Ignored when `enable_global_pivots` is `false`.
    /// Default: `5`.
    pub nsearch: usize,

    /// Maximum number of global pivots added per iteration.
    ///
    /// Ignored when `enable_global_pivots` is `false`. Default: `5`.
    pub max_nglobal_pivot: usize,

    /// Tolerance margin for the global pivot search.
    ///
    /// A candidate pivot is accepted when its interpolation error exceeds
    /// `abs_tol * tol_margin_global_search`, where `abs_tol` is the sweep's
    /// absolute tolerance (`tolerance * max_sample_value` when
    /// `normalize_error` is enabled). The value is always validated (it must
    /// be finite and nonnegative) but only consulted when `enable_global_pivots`
    /// is `true`. Default: `10.0`.
    pub tol_margin_global_search: f64,

    /// Random seed for the global pivot search.
    ///
    /// `None` seeds from OS entropy. Only used when `enable_global_pivots`
    /// is `true`. Default: `None`.
    pub seed: Option<u64>,

    /// Optional logical key/value byte budget for target memoization.
    ///
    /// `None` (default) evaluates every request and preserves callback semantics.
    /// Use `Some(256 * 1024 * 1024)` for expensive deterministic targets.
    /// `Some(0)` deduplicates only within each batch and retains no values.
    /// At capacity, inserts are skipped and reported; no entries are evicted.
    /// Memoization requires a fixed value for each point throughout the call,
    /// and supports index spaces up to 1024 bits. Callback errors and wrong
    /// output lengths are never cached. The cache belongs to this run, is
    /// shared across updates/searches/final materialization, and is dropped
    /// at return. Bytes exclude allocator overhead and do not bound total RSS.
    pub evaluation_cache_bytes: Option<usize>,
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
            normalize_error: true,
            enable_global_pivots: true,
            nsearch: 5,
            max_nglobal_pivot: 5,
            tol_margin_global_search: 10.0,
            seed: None,
            evaluation_cache_bytes: None,
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
/// for invalid options. It
/// also returns an error when the operation fails (a shape or index mismatch,
/// or a backend failure).
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
/// state.max_sample_value = 1.0;
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
    optimize_with_proposer(state, evaluate, options, &crate::DefaultProposer)
}

/// Optimize a TreeTCI state with `AllEdges` visitation and a caller-supplied
/// pivot candidate proposer.
///
/// Returns [`TreeTciOptimizationResult`] with per-iteration ranks, errors,
/// and the stopping reason defined by [`TreeTciTermination`].
///
/// Use this when you need a custom proposer (e.g., [`SimpleProposer`](crate::SimpleProposer)
/// or [`TruncatedDefaultProposer`](crate::TruncatedDefaultProposer)).
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
/// for invalid options. It
/// also returns an error when the operation fails (a shape or index mismatch,
/// or a backend failure).
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{
///     optimize_with_proposer, GlobalIndexBatch, SimpleProposer,
///     TreeTCI2, TreeTciEdge, TreeTciGraph, TreeTciOptions,
/// };
/// use anyhow::Result;
///
/// let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
/// let mut state = TreeTCI2::<f64>::new(vec![2, 2], graph).unwrap();
/// state.add_global_pivots(&[vec![0, 0]]).unwrap();
/// state.max_sample_value = 1.0;
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
///     &mut state, evaluate, &options, &proposer,
/// ).unwrap();
///
/// // One entry per sweep actually run; the loop stops early once converged,
/// // so this may be less than max_iter (3).
/// assert_eq!(ranks, vec![2; 3]);
/// assert_eq!(ranks.len(), errors.len());
/// ```
pub fn optimize_with_proposer<T, F, P>(
    state: &mut TreeTCI2<T>,
    evaluate: F,
    options: &TreeTciOptions,
    proposer: &P,
) -> TreeTciResult<TreeTciOptimizationResult>
where
    T: Scalar + CommonScalar + FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts,
    F: FnMut(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    options.validate()?;
    with_seeded_streams(options, proposer, |streams| {
        let evaluator = crate::evaluation::RunEvaluator::new(
            evaluate,
            &state.local_dims,
            options.evaluation_cache_bytes,
        )?;
        let mut result = optimize_with_streams(
            state,
            |batch| evaluator.call(batch),
            options,
            proposer,
            streams,
        )?;
        result.evaluation = evaluator.stats();
        Ok(result)
    })
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
/// for invalid options. It
/// also returns an error when the operation fails (a shape or index mismatch,
/// or a backend failure).
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
/// state.max_sample_value = 1.0;
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
///     &mut state, evaluate, &options, &SimpleProposer::default(), &mut rng,
/// ).unwrap();
/// assert_eq!(ranks.len(), errors.len());
/// ```
pub fn optimize_with_proposer_with_rng<T, F, P, R>(
    state: &mut TreeTCI2<T>,
    evaluate: F,
    options: &TreeTciOptions,
    proposer: &P,
    rng: &mut R,
) -> TreeTciResult<TreeTciOptimizationResult>
where
    T: Scalar + CommonScalar + FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts,
    F: FnMut(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
    R: rand::Rng + ?Sized,
{
    // Erase the caller's RNG type once so the run below is instantiated once
    // per scalar type instead of once per (scalar, RNG) pair.
    options.validate()?;
    let mut stream: &mut R = rng;
    let mut streams = RandomStreams::Shared(&mut stream);
    let evaluator = crate::evaluation::RunEvaluator::new(
        evaluate,
        &state.local_dims,
        options.evaluation_cache_bytes,
    )?;
    let mut result = optimize_with_streams(
        state,
        |batch| evaluator.call(batch),
        options,
        proposer,
        &mut streams,
    )?;
    result.evaluation = evaluator.stats();
    Ok(result)
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

pub(crate) fn optimize_with_streams<T, F, P>(
    state: &mut TreeTCI2<T>,
    evaluate: F,
    options: &TreeTciOptions,
    proposer: &P,
    streams: &mut RandomStreams<'_>,
) -> TreeTciResult<TreeTciOptimizationResult>
where
    T: Scalar + CommonScalar + FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    options.validate()?;

    let mut ranks = Vec::new();
    let mut errors = Vec::new();
    let mut nglobal_pivots_history: Vec<usize> = Vec::new();
    let mut termination = TreeTciTermination::MaxIterations;
    let visitor = AllEdges;
    const INNER_EDGE_PASSES: usize = 2;
    // Mirrors `tensor4all-tensorci`'s `TensorCI2Options::ncheck_history` default
    // (see `convergence_criterion` in tensorci2.rs, itself a port of Julia's
    // `convergencecriterion`). A single below-tolerance sweep is not reliable:
    // the bond-error estimate can dip before the pivot search has actually
    // stabilized. Like TensorCI2, the window must also contain no newly added
    // global pivots: an iteration that injected pivots has not yet swept them,
    // so its error estimate is stale with respect to those pivots.
    const NCHECK_HISTORY: usize = 3;

    for _iter in 0..options.max_iter {
        for _pass in 0..INNER_EDGE_PASSES {
            let error_scale = if options.normalize_error && state.max_sample_value > 0.0 {
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

            // INVARIANT: AllEdges visits each edge once in this pass. Its
            // two subtree keys are not changed by updates of other edges;
            // proposers can retain its current pivots without a full snapshot.
            state.flush_pivot_errors();

            for edge in visitor.visit_order(state) {
                update_edge(
                    state,
                    edge,
                    &evaluate,
                    &kernel_options,
                    proposer,
                    streams.candidates(),
                )?;
            }
        }

        ranks.push(state.max_bond_dim());
        let normalized_error = if options.normalize_error && state.max_sample_value > 0.0 {
            state.max_bond_error() / state.max_sample_value
        } else {
            state.max_bond_error()
        };
        errors.push(normalized_error);

        // Mirrors `TreeTCI.jl`'s `convergencecriterion` third disjunct
        // (branch `local-fix-convergence`, commit 06563dd): once the rank has
        // saturated at `max_bond_dim` for the whole trailing window, further
        // sweeps cannot reduce the error, so waiting for it to also cross
        // `tolerance` would just burn the remaining `max_iter` sweeps.
        //
        // This stop depends only on ranks already recorded, so it is decided
        // *before* the global pivot search. Injected pivots are only
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
        if bond_dim_saturated {
            termination = TreeTciTermination::MaxBondDimension;
            break;
        }

        // Global pivot search: after each sweep, materialize the current
        // approximation and inject pivots where |f - tt| is large, so
        // separated features that the local pivot updates miss are sampled
        // in the next sweep. Enabled by default; see
        // `TreeTciOptions::enable_global_pivots`. The search is skipped on
        // the final iteration: a pivot injected after the last sweep would
        // never be processed by a subsequent sweep, so the recorded error
        // and termination reason would not reflect it, and the unswept state
        // would not be materializable (see above).
        if options.enable_global_pivots && _iter + 1 < options.max_iter {
            let error_scale = if options.normalize_error && state.max_sample_value > 0.0 {
                state.max_sample_value
            } else {
                1.0
            };
            let abs_tol = options.tolerance * error_scale;
            let pivots = find_global_pivots_erased(
                state,
                &evaluate,
                options.nsearch,
                options.max_nglobal_pivot,
                options.tol_margin_global_search,
                abs_tol,
                streams.global(),
            )?;
            state.add_global_pivots(&pivots)?;
            nglobal_pivots_history.push(pivots.len());
        } else {
            nglobal_pivots_history.push(0);
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
            let errors_converged = last_errors.iter().all(|&e| e < options.tolerance);
            let no_global_pivots = last_ngp.iter().all(|&n| n == 0);
            let rank_stable = last_ranks.iter().min().copied().unwrap_or(0)
                == last_ranks.last().copied().unwrap_or(0);
            if errors_converged && no_global_pivots && rank_stable {
                termination = TreeTciTermination::Converged;
                break;
            }
        }
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
