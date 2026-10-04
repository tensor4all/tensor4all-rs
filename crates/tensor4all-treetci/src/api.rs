use crate::batch::checked_batch_len;
use crate::error::Result as TreeTciResult;
use crate::{
    materialize::to_treetn, optimize::optimize_with_proposer_with_rng, GlobalIndexBatch,
    MultiIndex, PivotCandidateProposer, TreeTCI2, TreeTciGraph, TreeTciOptions,
};
use anyhow::Result;
use rand::SeedableRng;
use tensor4all_core::CommonScalar;
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::TreeTN;

/// High-level TreeTCI return type:
/// `(treetn, ranks_per_iter, normalized_errors_per_iter)`.
///
/// - `treetn`: The materialized tree tensor network.
/// - `ranks_per_iter`: Maximum bond dimension at each iteration.
/// - `normalized_errors_per_iter`: Normalized bond error at each iteration.
pub type TreeTciRunResult = (
    TreeTN<tensor4all_core::IdxTensor, usize>,
    Vec<usize>,
    Vec<f64>,
);

/// Cross interpolate a function on a tree graph and return a `TreeTN`.
///
/// This is the unified entry point for tree tensor cross interpolation.
/// The `evaluate` closure receives batches of multi-indices and must return
/// one scalar per point. Edge candidate matrices and materialized site
/// tensors are split into calls of at most 65,536 points, so the closure
/// must not assume one call per matrix or tensor; see
/// [`GlobalIndexBatch`](crate::GlobalIndexBatch#batch-sizes).
///
/// The `proposer` controls how pivot candidates are generated.
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
/// let (treetn, ranks, errors) = crossinterpolate2::<f64, _, _>(
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
/// assert!(ranks.last().copied().unwrap_or(0) <= 2);
/// // Error should converge to near zero
/// assert!(errors.last().copied().unwrap_or(1.0) < 1e-8);
/// ```
#[allow(clippy::too_many_arguments)]
/// # Errors
///
/// Returns [`TreeTciError::InvalidConfiguration`](crate::TreeTciError::InvalidConfiguration)
/// for invalid options. It
/// also returns an error when the operation fails (a shape or index mismatch,
/// or a backend failure).
///
/// Interpolate a tree tensor network.
///
/// The seeded high-level path uses an explicitly named RNG (drawn from
/// [`TreeTciOptions::seed`], or OS entropy when unset and the global search is
/// enabled) and delegates to [`crossinterpolate2_with_rng`].
///
/// # Errors
///
/// Returns an error for invalid options, mismatched dimensions, pivot sets that
/// are empty or evaluate to zero, or a backend failure.
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
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    // Entropy is drawn only when a randomized global search actually runs.
    let searches_run = options.enable_global_pivots && options.max_iter > 1;
    let mut rng = match (searches_run, options.seed) {
        (true, None) => rand_chacha::ChaCha8Rng::from_os_rng(),
        (_, seed) => rand_chacha::ChaCha8Rng::seed_from_u64(seed.unwrap_or(0)),
    };
    crossinterpolate2_with_rng(
        evaluate,
        local_dims,
        graph,
        initial_pivots,
        options,
        center_site,
        proposer,
        &mut rng,
    )
}

/// Interpolate a tree tensor network on a caller-owned random stream.
///
/// Same as [`crossinterpolate2`], but consumes `rng` for every global pivot
/// search of the run instead of deriving a generator from
/// [`TreeTciOptions::seed`], so the caller can reproduce or advance the run's
/// randomness and share one stream across runs. Randomized proposers keep their
/// own generators (see #824); this controls the global searches only.
///
/// # Errors
/// Returns [`TreeTciError::InvalidConfiguration`](crate::TreeTciError::InvalidConfiguration)
/// for invalid options. It
/// also returns an error when the operation fails (a shape or index mismatch,
/// or a backend failure).
/// Interpolate a tree tensor network.
/// The seeded high-level path uses an explicitly named RNG (drawn from
/// [`TreeTciOptions::seed`], or OS entropy when unset and the global search is
/// enabled) and delegates to [`crossinterpolate2_with_rng`].
/// Returns an error for invalid options, mismatched dimensions, pivot sets that
/// are empty or evaluate to zero, or a backend failure.
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
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
    R: rand::Rng + ?Sized,
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

    let (ranks, errors) =
        optimize_with_proposer_with_rng(&mut tci, &evaluate, &options, proposer, rng)?;
    let treetn = to_treetn(&tci, &evaluate, center_site)?;

    Ok((treetn, ranks, errors))
}
