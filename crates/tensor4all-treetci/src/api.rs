use crate::batch::checked_batch_len;
use crate::error::Result as TreeTciResult;
use crate::optimize::{with_seeded_streams, RandomStreams};
use crate::{
    materialize::to_treetn, optimize::optimize_with_streams, GlobalIndexBatch, MultiIndex,
    PivotCandidateProposer, TreeTCI2, TreeTciGraph, TreeTciOptions, TreeTciTermination,
};
use anyhow::Result;
use tensor4all_core::CommonScalar;
use tensor4all_tensorbackend::FullPivLuScalar;
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
///     normalize_error: true,
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
    with_seeded_streams(&options, proposer, |streams| {
        crossinterpolate2_with_streams(
            evaluate,
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
/// for invalid options. It
/// also returns an error when the operation fails (a shape or index mismatch,
/// or a backend failure).
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
    crossinterpolate2_with_streams(
        evaluate,
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
fn crossinterpolate2_with_streams<T, F, P>(
    evaluate: F,
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
    F: FnMut(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
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

    let evaluator = crate::evaluation::RunEvaluator::new(
        evaluate,
        &tci.local_dims,
        options.evaluation_cache_bytes,
    )?;
    let evaluate = |batch: GlobalIndexBatch<'_>| evaluator.call(batch);

    // Initialize max_sample_value via batch evaluate
    let n_sites = tci.local_dims.len();
    let flat_len = checked_batch_len(n_sites, pivots.len())?;
    let mut flat = Vec::with_capacity(flat_len);
    for pivot in &pivots {
        flat.extend_from_slice(pivot);
    }
    let batch = GlobalIndexBatch::new(&flat, n_sites, pivots.len())?;
    let init_vals = evaluate(batch)?;
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

    let result = optimize_with_streams(&mut tci, evaluate, options, proposer, streams)?;
    let treetn = to_treetn(&tci, evaluate, center_site)?;

    Ok(TreeTciRunResult {
        treetn,
        ranks: result.ranks,
        errors: result.errors,
        termination: result.termination,
        evaluation: evaluator.stats(),
    })
}
