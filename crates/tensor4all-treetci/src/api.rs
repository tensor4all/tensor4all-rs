use crate::batch::{checked_batch_len, EVALUATION_CHUNK_POINTS};
use crate::error::Result as TreeTciResult;
use crate::evaluation::{BatchExecution, CachedExecution, Compatibility, Explicit};
use crate::optimize::{with_seeded_streams, RandomStreams};
use crate::{
    materialize::to_treetn_chunked, optimize::optimize_with_streams, AllEdges, GlobalIndexBatch,
    MultiIndex, PivotCandidateProposer, TreeTCI2, TreeTciGraph, TreeTciOptions, TreeTciTermination,
};
use anyhow::Result;
use tensor4all_core::CommonScalar;
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::interpolation::EvalContext;
use tensor4all_treetn::TreeTN;

/// A materialized tree tensor network with iteration diagnostics and stopping reason.
///
/// Returned by [`crossinterpolate2`] and [`crossinterpolate2_with_rng`].
/// [`crate::TreeTciOptimizationResult`] contains the corresponding diagnostics
/// when optimizing a caller-owned state without materializing the network.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{
///     crossinterpolate2, DefaultProposer, TreeTciGraph, TreeTciOptions, TreeTciTermination,
/// };
///
/// let result = crossinterpolate2::<f64, _, _>(
///     |batch| Ok(vec![2.0; batch.n_points()]), vec![2, 2],
///     TreeTciGraph::linear_chain(2)?, vec![vec![0, 0]],
///     TreeTciOptions { seed: Some(0), ..Default::default() }, None, &DefaultProposer,
/// )?;
/// assert_eq!(result.termination, TreeTciTermination::Converged);
/// let dense = result.treetn.contract_to_tensor()?;
/// let expected = tensor4all_core::IdxTensor::from_dense(dense.indices().to_vec(), vec![2.0; 4])?;
/// assert!(dense.sub(&expected)?.maxabs()? < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct TreeTciRunResult {
    /// The materialized approximation; inspect `termination` before accepting it.
    pub treetn: TreeTN<tensor4all_core::IdxTensor, usize>,
    /// Maximum bond dimension after each completed iteration.
    pub ranks: Vec<usize>,
    /// Maximum sampled bond error per iteration, normalized when requested.
    ///
    /// Has the same length as `ranks`; it is not a full-network residual.
    pub errors: Vec<f64>,
    /// The stopping condition evaluated by the optimization loop.
    pub termination: TreeTciTermination,
    /// Oracle evaluation and optional memo accounting, including final materialization.
    pub evaluation: crate::TreeTciEvaluationStats,
}

/// Cross interpolate a function on a tree graph and return a `TreeTN`.
///
/// This is the unified entry point for tree tensor cross interpolation.
/// The `evaluate` closure receives batches of multi-indices and must return
/// one scalar per point. Edge candidate matrices and materialized site
/// tensors are split into calls of at most 65,536 points, so the closure
/// must not assume one call per matrix or tensor; see
/// [`GlobalIndexBatch`](crate::GlobalIndexBatch#batch-sizes).
/// The callback may borrow mutable state. With `evaluation_cache_bytes` enabled,
/// it must return a fixed value for each multi-index throughout the call:
/// memoization suppresses repeated calls, including during final materialization.
///
/// The `proposer` controls how pivot candidates are generated. Returns
/// [`TreeTciRunResult`] with the network, iteration histories, and stopping
/// reason; see [`TreeTciTermination`] for its sampled convergence criterion.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{
///     crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciEdge, TreeTciGraph,
///     TreeTciOptions,
/// };
/// use anyhow::Result;
///
/// // Approximate the 2-site identity function f(i, j) = 1 if i==j else 0
/// let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
/// let local_dims = vec![2, 2];
///
/// let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
///     let mut values = Vec::with_capacity(batch.n_points());
///     for p in 0..batch.n_points() {
///         let i = batch.get(0, p).unwrap();
///         let j = batch.get(1, p).unwrap();
///         values.push(if i == j { 1.0 } else { 0.0 });
///     }
///     Ok(values)
/// };
///
/// let options = TreeTciOptions {
///     tolerance: 1e-10,
///     max_iter: 10,
///     max_bond_dim: Some(10),
///     scale_tolerance: true,
///     ..Default::default()
/// };
///
/// let proposer = DefaultProposer;
/// let result = crossinterpolate2::<f64, _, _>(
///     evaluate,
///     local_dims,
///     graph,
///     vec![],
///     options,
///     None,
///     &proposer,
/// ).unwrap();
///
/// // The identity on a 2x2 space has rank 2
/// assert_eq!(result.ranks.last().copied(), Some(2));
/// assert_eq!(result.termination, tensor4all_treetci::TreeTciTermination::Converged);
/// // Error should converge to near zero
/// assert!(result.errors.last().copied().unwrap_or(1.0) < 1e-8);
/// ```
#[allow(clippy::too_many_arguments)]
/// # Errors
///
/// Returns [`crate::TreeTciError::InvalidConfiguration`] for invalid options.
/// Dimension mismatches, invalid initial pivots, an all-zero initial sample,
/// callback failures, and materialization/backend failures return
/// [`crate::TreeTciError::Operation`] with the underlying diagnostic.
///
pub fn crossinterpolate2<T, F, P>(
    evaluate: F,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: TreeTciOptions,
    center_site: Option<usize>,
    proposer: &P,
) -> TreeTciResult<TreeTciRunResult>
where
    T: FullPivLuScalar
        + CommonScalar
        + tensor4all_core::MatrixLuciScalar
        + tensor4all_core::TensorElement
        + crate::globalpivot::ScalarParts,
    F: FnMut(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    options.validate()?;
    let mut evaluate = evaluate;
    let mut execution = Compatibility::new(&mut evaluate);
    with_seeded_streams(&options, proposer, |streams| {
        crossinterpolate2_with_streams(
            &mut execution,
            local_dims,
            graph,
            initial_pivots,
            &options,
            center_site,
            proposer,
            streams,
        )
    })
}

/// Interpolate a tree tensor network with a caller-owned execution context.
///
/// The same algorithm, options, stopping rule and streaming as [`crossinterpolate2`]; what differs is the
/// resource the session-bound stages run in. Without a context they go through the process-global CPU context;
/// with one, the two materializations - the one the global pivot search performs and the final one - run on
/// the caller's context, through entries that consult no process-global context.
///
/// **What is not covered, so that this is a budget for part of the stage and not the whole run:** the caller's
/// own evaluator, the pivot search's readout (which for a homogeneous single-index tree takes raw message
/// kernels and cached columns, and can still reach the global session for larger grouped messages, narrow
/// scalars or generic layouts), and physical tensor publication, which adopts into the process-global eager
/// runtime. **A caller on a Rayon worker cannot use this entry point**: the concrete session entry rejects
/// worker threads rather than silently serializing or falling back to the global context, so the run fails
/// typed once a nonzero solve is reached. Call it from a thread that owns its execution, as the scoped and held
/// CPU session entries require.
///
/// # Errors
///
/// Returns [`crate::TreeTciError::InvalidConfiguration`] for invalid options, and
/// [`crate::TreeTciError::Operation`] for the dimension mismatches, invalid initial pivots, callback failures
/// and backend failures [`crossinterpolate2`] reports - including the concrete-session entry refusal when a
/// Rayon worker calls this entry point, because neither the optimizer's global-pivot work nor the
/// materialization solve can enter the caller's context there.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{crossinterpolate2_in, DefaultProposer, TreeTciGraph, TreeTciOptions};
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let result = crossinterpolate2_in::<f64, _, _>(
///     |batch| Ok(vec![2.0; batch.n_points()]), vec![2, 2],
///     TreeTciGraph::linear_chain(2)?, vec![vec![0, 0]],
///     TreeTciOptions::default(), None, &DefaultProposer, &context,
/// )?;
/// assert_eq!(result.ranks, vec![1; 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn crossinterpolate2_in<T, F, P>(
    evaluate: F,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: TreeTciOptions,
    center_site: Option<usize>,
    proposer: &P,
    context: &tensor4all_tensorbackend::CpuExecutionContext,
) -> TreeTciResult<TreeTciRunResult>
where
    T: FullPivLuScalar
        + CommonScalar
        + tensor4all_core::MatrixLuciScalar
        + tensor4all_core::TensorElement
        + crate::globalpivot::ScalarParts,
    F: FnMut(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    options.validate()?;
    let mut evaluate = evaluate;
    context
        .with_concrete_session(|session| {
            let mut contextual =
                |batch: GlobalIndexBatch<'_>, _ctx: &mut EvalContext<'_>| evaluate(batch);
            let mut context = crate::evaluation::serial_context(session);
            let mut execution = Explicit::new(&mut context, &mut contextual);
            with_seeded_streams(&options, proposer, |streams| {
                crossinterpolate2_with_streams(
                    &mut execution,
                    local_dims,
                    graph,
                    initial_pivots,
                    &options,
                    center_site,
                    proposer,
                    streams,
                )
            })
        })
        .map_err(|error| crate::TreeTciError::from(anyhow::Error::new(error)))?
}

/// Interpolate a tree tensor network on a caller-owned random stream and execution context.
///
/// The caller-stream counterpart of [`crossinterpolate2_in`]: the supplied stream drives every global pivot
/// search, and the optimizer's global-pivot work and the materialization solve run in `context`. The two are
/// independent, so a caller can advance its own randomness across runs while keeping those stages' budget to
/// itself. Physical tensor publication still adopts into the process-global eager runtime.
///
/// # Errors
///
/// Returns [`crate::TreeTciError::InvalidConfiguration`] for invalid options, and
/// [`crate::TreeTciError::Operation`] for the same dimension mismatches, invalid initial pivots, callback
/// failures, backend failures and concrete-session entry refusal [`crossinterpolate2_in`] reports.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_treetci::{crossinterpolate2_with_rng_in, DefaultProposer, TreeTciGraph, TreeTciOptions};
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let mut rng = ChaCha8Rng::seed_from_u64(0);
/// let result = crossinterpolate2_with_rng_in::<f64, _, _, _>(
///     |batch| Ok(vec![2.0; batch.n_points()]), vec![2, 2],
///     TreeTciGraph::linear_chain(2)?, vec![vec![0, 0]],
///     TreeTciOptions::default(), None, &DefaultProposer, &mut rng, &context,
/// )?;
/// assert_eq!(result.ranks, vec![1; 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn crossinterpolate2_with_rng_in<T, F, P, R>(
    evaluate: F,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: TreeTciOptions,
    center_site: Option<usize>,
    proposer: &P,
    rng: &mut R,
    context: &tensor4all_tensorbackend::CpuExecutionContext,
) -> TreeTciResult<TreeTciRunResult>
where
    T: FullPivLuScalar
        + CommonScalar
        + tensor4all_core::MatrixLuciScalar
        + tensor4all_core::TensorElement
        + crate::globalpivot::ScalarParts,
    F: FnMut(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
    R: rand::Rng + ?Sized,
{
    let mut stream: &mut R = rng;
    let mut streams = RandomStreams::Shared(&mut stream);
    let mut evaluate = evaluate;
    context
        .with_concrete_session(|session| {
            let mut contextual =
                |batch: GlobalIndexBatch<'_>, _ctx: &mut EvalContext<'_>| evaluate(batch);
            let mut context = crate::evaluation::serial_context(session);
            let mut execution = Explicit::new(&mut context, &mut contextual);
            crossinterpolate2_with_streams(
                &mut execution,
                local_dims,
                graph,
                initial_pivots,
                &options,
                center_site,
                proposer,
                &mut streams,
            )
        })
        .map_err(|error| crate::TreeTciError::from(anyhow::Error::new(error)))?
}

/// Interpolate a tree tensor network on a caller-owned random stream.
///
/// Same as [`crossinterpolate2`], but consumes `rng` for every global pivot
/// search of the run instead of deriving a generator from
/// [`TreeTciOptions::seed`], so the caller can reproduce or advance the run's
/// randomness and share one stream across runs. Candidate generation and
/// global searches both consume the supplied stream directly; proposer seeds
/// and `options.seed` are ignored.
/// Returns [`TreeTciRunResult`] with the network and the same stopping
/// diagnostics as [`crossinterpolate2`].
///
/// # Errors
/// Returns [`TreeTciError::InvalidConfiguration`](crate::TreeTciError::InvalidConfiguration)
/// for invalid options, and [`TreeTciError::Operation`](crate::TreeTciError::Operation)
/// for dimension mismatches, invalid initial pivots, an all-zero initial
/// sample, callback failures, and materialization/backend failures.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_treetci::{
///     crossinterpolate2_with_rng, DefaultProposer, TreeTciGraph,
///     TreeTciOptions, TreeTciTermination,
/// };
///
/// let mut rng = ChaCha8Rng::seed_from_u64(0);
/// let result = crossinterpolate2_with_rng::<f64, _, _, _>(
///     |batch| Ok(vec![2.0; batch.n_points()]), vec![2, 2],
///     TreeTciGraph::linear_chain(2)?, vec![vec![0, 0]],
///     TreeTciOptions::default(), None, &DefaultProposer, &mut rng,
/// )?;
/// assert_eq!(result.termination, TreeTciTermination::Converged);
/// assert_eq!(result.ranks, vec![1; 3]);
/// assert_eq!(result.errors, vec![0.0; 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn crossinterpolate2_with_rng<T, F, P, R>(
    evaluate: F,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: TreeTciOptions,
    center_site: Option<usize>,
    proposer: &P,
    rng: &mut R,
) -> TreeTciResult<TreeTciRunResult>
where
    T: FullPivLuScalar
        + CommonScalar
        + tensor4all_core::MatrixLuciScalar
        + tensor4all_core::TensorElement
        + crate::globalpivot::ScalarParts,
    F: FnMut(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
    R: rand::Rng + ?Sized,
{
    let mut stream: &mut R = rng;
    let mut streams = RandomStreams::Shared(&mut stream);
    let mut evaluate = evaluate;
    let mut execution = Compatibility::new(&mut evaluate);
    crossinterpolate2_with_streams(
        &mut execution,
        local_dims,
        graph,
        initial_pivots,
        &options,
        center_site,
        proposer,
        &mut streams,
    )
}

// INVARIANT: This shared run boundary preserves the eight explicit parameters
// of the public caller-stream entry point without bundling unrelated options.
#[allow(clippy::too_many_arguments)]
fn crossinterpolate2_with_streams<T, P>(
    execution: &mut dyn BatchExecution<T>,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: &TreeTciOptions,
    center_site: Option<usize>,
    proposer: &P,
    streams: &mut RandomStreams<'_>,
) -> TreeTciResult<TreeTciRunResult>
where
    T: FullPivLuScalar
        + CommonScalar
        + tensor4all_core::MatrixLuciScalar
        + tensor4all_core::TensorElement
        + crate::globalpivot::ScalarParts,
    P: PivotCandidateProposer,
{
    options.validate()?;
    if !(local_dims.len() == graph.n_sites()) {
        return Err(anyhow::anyhow!(
            "local_dims length {} must match graph site count {}",
            local_dims.len(),
            graph.n_sites()
        )
        .into());
    };

    let pivots = if initial_pivots.is_empty() {
        vec![vec![0; local_dims.len()]]
    } else {
        initial_pivots
    };

    let mut tci = TreeTCI2::<T>::new(local_dims, graph)?;
    tci.add_global_pivots(&pivots)?;

    let mut evaluator = CachedExecution::new(
        execution,
        &tci.local_dims,
        options.evaluation_cache_bytes,
    )?;

    // Initialize max_sample_value via batch evaluate
    let n_sites = tci.local_dims.len();
    let flat_len = checked_batch_len(n_sites, pivots.len())?;
    let mut flat = Vec::with_capacity(flat_len);
    for pivot in &pivots {
        flat.extend_from_slice(pivot);
    }
    let batch = GlobalIndexBatch::new(&flat, n_sites, pivots.len())?;
    let init_vals = evaluator.call(batch)?;
    if init_vals.len() != pivots.len() {
        return Err(anyhow::anyhow!(
            "initial evaluator returned {} values for {} pivots",
            init_vals.len(),
            pivots.len()
        )
        .into());
    }
    tci.max_sample_value = init_vals
        .iter()
        .map(|v| CommonScalar::abs_val(*v))
        .fold(0.0f64, f64::max);
    if !matches!(
        tci.max_sample_value.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    ) {
        return Err(anyhow::anyhow!("initial pivots must not all evaluate to zero").into());
    }

    let result = optimize_with_streams(
        &mut tci,
        &mut evaluator,
        options,
        proposer,
        &AllEdges,
        streams,
    )?;
    let treetn = to_treetn_chunked(
        &tci,
        &mut evaluator,
        center_site,
        EVALUATION_CHUNK_POINTS,
    )?;

    Ok(TreeTciRunResult {
        treetn,
        ranks: result.ranks,
        errors: result.errors,
        termination: result.termination,
        evaluation: evaluator.stats(),
    })
}
