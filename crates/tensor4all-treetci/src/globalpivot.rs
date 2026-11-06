//! Automatic global pivot search for [`TreeTCI2`].
//!
//! Port of the `GlobalPivotFinder` machinery from the chain TCI2 crate
//! (`tensor4all-tensorci`): after each sweep the optimizer materializes the
//! current tree approximation and searches starting points with local
//! coordinate optimization for multi-indices where `|f(idx) - tt(idx)|` is
//! large. The starting points are the caller-supplied ones, walked first, and
//! the search's own random draws. Found pivots are injected via
//! [`TreeTCI2::add_global_pivots`] so the
//! next sweep samples regions the local pivot updates missed.
//!
//! The search is enabled by default (`TreeTciOptions::enable_global_pivots`)
//! and runs after every optimization sweep, except after the final sweep of a
//! run that stops at `max_iter` or through the bond-dimension saturation stop.

use crate::error::Result as TreeTciResult;
use crate::evaluation::{BatchExecution, Compatibility};
use crate::{
    batch::EVALUATION_CHUNK_POINTS, materialize::to_treetn_chunked, GlobalIndexBatch, MultiIndex,
    TreeTCI2,
};
use anyhow::Result;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::MatrixLuciScalar as Scalar;
use tensor4all_core::{floating_zone_walk, AnyScalar, ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

// INVARIANT: the argument list is the search's boundary (state, evaluator,
// search limits, tolerance, stream); grouping them into a struct would only
// move the same values through one more layer.
/// Search for multi-indices where the current approximation error is large.
///
/// Algorithm (mirrors `DefaultGlobalPivotFinder` in `tensor4all-tensorci`):
///
/// 1. Materialize the current [`TreeTCI2`] state as a `TreeTN`.
/// 2. Walk every caller-supplied start, then draw and walk `nsearch` random
///    starting points.
/// 3. Walk each start with [`floating_zone_walk`], retaining the maximizing
///    coordinate between site scans and repeating sweeps until no improvement,
///    an error above `10 * abs_tol * tol_margin`, or 100 sweeps.
/// 4. Keep points whose error exceeds `abs_tol * tol_margin`.
/// 5. Return at most `max_nglobal_pivot` distinct points.
///
/// Caller-supplied starts let a caller aim the search at a region it already
/// suspects, such as a corner of a patch. A corner-localized feature otherwise
/// looks like a flat zero fiber from a random start, where the greedy walk
/// stalls. The starts are walked even when `nsearch` is zero.
///
/// One [`TreeTNCachedEvaluator`] is reused across all starts and sweeps. Each
/// coordinate scan is a batch hinted around the varied site, sharing subtree
/// environments. The held coordinate is not evaluated again in its own scan.
/// Site-free junctions (`local_dims[site] == 1`) need no coordinate scan.
///
/// The returned pivots are full-site multi-indices ready for
/// [`TreeTCI2::add_global_pivots`].
///
/// # Arguments
///
/// * `state` -- current TreeTCI pivot state.
/// * `evaluate` -- batch evaluator, identical to the one passed to
///
///   [`optimize_with_proposer`](crate::optimize_with_proposer).
/// * `nsearch` -- number of random starting points, drawn after the
///   caller-supplied starts.
/// * `starts` -- caller-supplied starting points as full-site multi-indices,
///   walked before the random ones. Each must have one coordinate per site and
///   an in-range coordinate. An empty slice adds none.
/// * `max_nglobal_pivot` -- maximum number of pivots returned.
/// * `tol_margin` -- acceptance margin over `abs_tol`.
/// * `abs_tol` -- absolute interpolation-error threshold.
/// * `seed` -- Seed passed to `ChaCha8Rng` to generate the random starting
///   points. Pivot selection also depends on the target values and the
///   approximation readout, including its floating-point rounding.
///
/// # Returns
///
/// Up to `max_nglobal_pivot` distinct full-site multi-indices where the
/// current approximation is (likely) poor, strongest first. An empty vector
/// when nothing exceeds the threshold. Among positive equal errors, the lower
/// local value wins at each site; an all-zero scan keeps the held coordinate.
/// Across starts, the earlier start wins. This is a greedy local search and
/// can stall on flat zero fibers.
///
/// # Errors
///
/// Returns [`crate::TreeTciError::Operation`] when `abs_tol` or `tol_margin`
/// is negative or non-finite, when a start has the wrong length or an
/// out-of-range coordinate (both checked before any evaluation), when the
/// state cannot be materialized as a `TreeTN`, when the evaluator or the
/// approximation readout returns the wrong number of values or a non-finite
/// residual, or when building the candidate index array fails.
///
/// # Examples
///
/// ```
/// # fn main() -> anyhow::Result<()> {
/// use tensor4all_treetci::{
///     find_global_pivots, GlobalIndexBatch, TreeTCI2, TreeTciEdge, TreeTciGraph,
/// };
///
/// // f = 1 + 10 * delta_{(1, 1)} on two binary sites.
/// let evaluate = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
///     Ok(batch
///         .data()
///         .chunks(batch.n_sites())
///         .map(|point| if point == [1, 1] { 11.0 } else { 1.0 })
///         .collect())
/// };
///
/// // With the single pivot (0, 0) the rank-1 approximation is 1 everywhere,
/// // so (1, 1), with error 10, is the only point above `abs_tol = 1.0`.
/// let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)])?;
/// let mut state = TreeTCI2::<f64>::new(vec![2, 2], graph)?;
/// state.add_global_pivots(&[vec![0, 0]])?;
///
/// // The random starts are disabled and a caller start at (0, 1), one
/// // coordinate away from (1, 1), is walked instead; a walk from (0, 0) would
/// // see flat zero fibers.
/// let pivots = find_global_pivots(&state, evaluate, 0, &[vec![0, 1]], 2, 1.0, 1.0, 42)?;
/// assert_eq!(pivots, vec![vec![1, 1]]);
/// # Ok(())
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn find_global_pivots<T, F>(
    state: &TreeTCI2<T>,
    evaluate: F,
    nsearch: usize,
    starts: &[MultiIndex],
    max_nglobal_pivot: usize,
    tol_margin: f64,
    abs_tol: f64,
    seed: u64,
) -> TreeTciResult<Vec<MultiIndex>>
where
    T: FullPivLuScalar + Scalar + tensor4all_core::TensorElement + ScalarParts,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    // The seeded path uses an explicitly named RNG and delegates to the
    // caller-owned-stream entry point.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    find_global_pivots_with_rng(
        state,
        evaluate,
        nsearch,
        starts,
        max_nglobal_pivot,
        tol_margin,
        abs_tol,
        &mut rng,
    )
}

// Same argument boundary as [`find_global_pivots`].
/// Global pivot search on a caller-owned random stream.
///
/// Same as [`find_global_pivots`], but consumes `rng` directly for the random
/// starting points instead of deriving their stream from a seed, so a caller
/// can pin, advance, or share one stream across several searches. Pass a
/// `rand_chacha::ChaCha8Rng` when a deterministic algorithm is required.
///
/// The caller-supplied `starts` are walked before the random ones, so they do
/// not depend on `rng`.
///
/// # Errors
///
/// Returns [`crate::TreeTciError::Operation`] when `abs_tol` or `tol_margin`
/// is negative or non-finite, when a start has the wrong length or an
/// out-of-range coordinate (both checked before any evaluation), when the
/// state cannot be materialized as a `TreeTN`, when the evaluator or the
/// approximation readout returns the wrong number of values or a non-finite
/// residual, or when building the candidate index array fails.
#[allow(clippy::too_many_arguments)]
pub fn find_global_pivots_with_rng<T, F, R>(
    state: &TreeTCI2<T>,
    evaluate: F,
    nsearch: usize,
    starts: &[MultiIndex],
    max_nglobal_pivot: usize,
    tol_margin: f64,
    abs_tol: f64,
    rng: &mut R,
) -> TreeTciResult<Vec<MultiIndex>>
where
    T: FullPivLuScalar + Scalar + tensor4all_core::TensorElement + ScalarParts,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    R: Rng + ?Sized,
{
    // Erase the caller's RNG type once, so the search body below is
    // instantiated once per scalar type instead of once per (scalar, RNG) pair.
    let mut stream: &mut R = rng;
    let mut evaluate = evaluate;
    let mut execution = Compatibility::new(&mut evaluate);
    find_global_pivots_erased(
        state,
        &mut execution,
        nsearch,
        starts,
        max_nglobal_pivot,
        tol_margin,
        abs_tol,
        &mut stream,
    )
}

/// The search body on an already erased stream.
///
/// Callers inside the crate use this so the search is instantiated once per
/// scalar type.
// INVARIANT: the parameter list is the search's boundary (state, execution owner,
// search limits, tolerance, stream); grouping them into a struct would only move
// the same values through one more layer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_global_pivots_erased<T>(
    state: &TreeTCI2<T>,
    execution: &mut dyn BatchExecution<T>,
    nsearch: usize,
    starts: &[MultiIndex],
    max_nglobal_pivot: usize,
    tol_margin: f64,
    abs_tol: f64,
    rng: &mut dyn RngCore,
) -> TreeTciResult<Vec<MultiIndex>>
where
    T: FullPivLuScalar + Scalar + tensor4all_core::TensorElement + ScalarParts,
{
    let params = SearchParams {
        nsearch,
        max_nglobal_pivot,
        tol_margin,
        abs_tol,
    };
    search_with_readout(
        state,
        execution,
        params,
        starts,
        rng,
        cached_walk_readout,
        cached_walk_readout_on_session::<T>,
    )
}

/// Scalar parameters of one global pivot search, as passed to
/// [`find_global_pivots`].
#[derive(Clone, Copy, Debug)]
struct SearchParams {
    nsearch: usize,
    max_nglobal_pivot: usize,
    tol_margin: f64,
    abs_tol: f64,
}

/// A session-taking approximation readout for runs that own an [`EvalContext`].
///
/// It returns typed values, not [`AnyScalar`]s: an `AnyScalar` carries an eager
/// rank-zero tensor, and publishing one enters the process-global eager
/// runtime, which is exactly what a caller-held session forbids while it is
/// active.
type SessionReadout<T> = fn(
    &mut TreeTNCachedEvaluator<'_, usize>,
    &TreeTN<IdxTensor, usize>,
    &[DynIndex],
    ColMajorArrayRef<'_, usize>,
    Option<usize>,
    &mut tensor4all_tensorbackend::explicit::Session<'_>,
) -> Result<Vec<T>>;

/// Read one scan through the cache owned by this search. A site hint also
/// identifies single-candidate scans, whose varied site cannot be inferred.
fn cached_walk_readout(
    evaluator: &mut TreeTNCachedEvaluator<'_, usize>,
    _treetn: &TreeTN<IdxTensor, usize>,
    _site_indices: &[DynIndex],
    candidates: ColMajorArrayRef<'_, usize>,
    scan_site: Option<usize>,
) -> Result<Vec<AnyScalar>> {
    let hint = scan_site.map(EvaluationHint::around).unwrap_or_default();
    evaluator
        .evaluate_batched_with_hint(candidates, hint)
        .map_err(anyhow::Error::from)
}

/// Session counterpart of [`cached_walk_readout`], used when the run owner has a session.
///
/// The typed session entry point reads the cached values back as host scalars,
/// so the readout never constructs an [`AnyScalar`] and never enters the
/// process-global eager runtime.
fn cached_walk_readout_on_session<T: tensor4all_core::TensorElement>(
    evaluator: &mut TreeTNCachedEvaluator<'_, usize>,
    _treetn: &TreeTN<IdxTensor, usize>,
    _site_indices: &[DynIndex],
    candidates: ColMajorArrayRef<'_, usize>,
    scan_site: Option<usize>,
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
) -> Result<Vec<T>> {
    let hint = scan_site.map(EvaluationHint::around).unwrap_or_default();
    evaluator
        .evaluate_batched_typed_on_session::<T>(session, candidates, hint)
        .map_err(anyhow::Error::from)
}

/// [`find_global_pivots`] with the approximation readout supplied by the
/// caller. Production passes [`cached_walk_readout`]; tests also pass the
/// pointwise [`TreeTN::evaluate`] to check that the readout does not change
/// the selected pivots.
fn search_with_readout<T, R>(
    state: &TreeTCI2<T>,
    execution: &mut dyn BatchExecution<T>,
    params: SearchParams,
    starts: &[MultiIndex],
    rng: &mut dyn RngCore,
    mut readout: R,
    readout_on_session: SessionReadout<T>,
) -> TreeTciResult<Vec<MultiIndex>>
where
    T: FullPivLuScalar + Scalar + tensor4all_core::TensorElement + ScalarParts,
    R: FnMut(
        &mut TreeTNCachedEvaluator<'_, usize>,
        &TreeTN<IdxTensor, usize>,
        &[DynIndex],
        ColMajorArrayRef<'_, usize>,
        Option<usize>,
    ) -> Result<Vec<AnyScalar>>,
{
    let SearchParams {
        nsearch,
        max_nglobal_pivot,
        tol_margin,
        abs_tol,
    } = params;
    if !abs_tol.is_finite() || abs_tol < 0.0 {
        return Err(
            anyhow::anyhow!("global pivot search abs_tol must be finite and nonnegative").into(),
        );
    }
    if !tol_margin.is_finite() || tol_margin < 0.0 {
        return Err(anyhow::anyhow!(
            "global pivot search tol_margin must be finite and nonnegative"
        )
        .into());
    }
    if nsearch == 0 && starts.is_empty() {
        return Ok(Vec::new());
    }
    let n_sites = state.local_dims.len();
    for (index, start) in starts.iter().enumerate() {
        if start.len() != n_sites {
            return Err(anyhow::anyhow!(
                "global pivot search start {index} has {} coordinates, expected {n_sites}",
                start.len()
            )
            .into());
        }
        for (site, (&coordinate, &dimension)) in start.iter().zip(&state.local_dims).enumerate() {
            if coordinate >= dimension {
                return Err(anyhow::anyhow!(
                    "global pivot search start {index} has coordinate {coordinate} at site {site}, \
                     out of range for local dimension {dimension}"
                )
                .into());
            }
        }
    }
    if max_nglobal_pivot == 0 || n_sites == 0 {
        return Ok(Vec::new());
    }

    // Materialize the current approximation once per search. A degenerate or
    // inconsistent pivot state is a real error; propagate it rather than
    // silently skipping the search.
    let treetn = to_treetn_chunked(state, execution, None, EVALUATION_CHUNK_POINTS)?;
    let mut site_indices = Vec::with_capacity(n_sites);
    for site in 0..n_sites {
        let node = treetn
            .node_index(&site)
            .ok_or_else(|| anyhow::anyhow!("materialized tree missing site {site}"))?;
        let tensor = treetn
            .tensor(node)
            .ok_or_else(|| anyhow::anyhow!("materialized tree missing tensor for site {site}"))?;
        // `to_treetn` always stores the site index first.
        site_indices.push(tensor.indices()[0].clone());
    }

    // Cache ownership is one search, across every start/site/sweep. Scratch
    // is bounded by one site scan rather than nsearch * sum(local_dims).
    let mut cache =
        TreeTNCachedEvaluator::new(&treetn, &site_indices, CachedEvaluatorOptions::default())
            .map_err(anyhow::Error::from)?;
    let max_points = state
        .local_dims
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .saturating_sub(1)
        .max(1);
    let flat_capacity = n_sites
        .checked_mul(max_points)
        .ok_or_else(|| anyhow::anyhow!("global-pivot flat batch size overflowed usize"))?;
    let mut flat = Vec::with_capacity(flat_capacity);
    let mut start = Vec::with_capacity(n_sites);
    let threshold = abs_tol * tol_margin;
    let mut best = Vec::with_capacity(starts.len().saturating_add(nsearch));
    for provided in starts {
        let (error, point) = walk_from_start(
            state,
            execution,
            &mut cache,
            &treetn,
            &site_indices,
            n_sites,
            &mut flat,
            provided,
            threshold,
            &mut readout,
            readout_on_session,
        )?;
        if error > threshold {
            best.push((error, point));
        }
    }
    for _ in 0..nsearch {
        start.clear();
        start.extend(state.local_dims.iter().map(|&dim| rng.random_range(0..dim)));
        let (error, point) = walk_from_start(
            state,
            execution,
            &mut cache,
            &treetn,
            &site_indices,
            n_sites,
            &mut flat,
            &start,
            threshold,
            &mut readout,
            readout_on_session,
        )?;
        if error > threshold {
            best.push((error, point));
        }
    }

    // Keep the strongest distinct points.
    best.sort_by(|(a, _), (b, _)| b.total_cmp(a));
    let mut pivots: Vec<MultiIndex> = Vec::new();
    for (_, point) in best {
        if !pivots.contains(&point) {
            pivots.push(point);
            if pivots.len() >= max_nglobal_pivot {
                break;
            }
        }
    }
    Ok(pivots)
}

/// Walk one starting point and return its best error and the point that holds
/// it, using the search's shared cache and readout.
///
/// Shared by the caller-supplied starts and the random ones so both follow the
/// same walk, batching, and error checks.
// INVARIANT: the parameter list is one start's state plus the search-scoped
// owners it borrows; grouping them into a struct would only move the same
// references through one more layer.
#[allow(clippy::too_many_arguments)]
fn walk_from_start<T, R>(
    state: &TreeTCI2<T>,
    execution: &mut dyn BatchExecution<T>,
    cache: &mut TreeTNCachedEvaluator<'_, usize>,
    treetn: &TreeTN<IdxTensor, usize>,
    site_indices: &[DynIndex],
    n_sites: usize,
    flat: &mut Vec<usize>,
    start: &MultiIndex,
    threshold: f64,
    readout: &mut R,
    readout_on_session: SessionReadout<T>,
) -> TreeTciResult<(f64, MultiIndex)>
where
    T: FullPivLuScalar + Scalar + tensor4all_core::TensorElement + ScalarParts,
    R: FnMut(
        &mut TreeTNCachedEvaluator<'_, usize>,
        &TreeTN<IdxTensor, usize>,
        &[DynIndex],
        ColMajorArrayRef<'_, usize>,
        Option<usize>,
    ) -> Result<Vec<AnyScalar>>,
{
    // Match chain TCI / TensorCrossInterpolation.jl: at most 100 sweeps,
    // early exit above 10 * the acceptance threshold. Infinite early-stop
    // bounds disable only that exit; the sweep/no-improvement bounds remain.
    let (point, error) = floating_zone_walk::<_, anyhow::Error>(
        &state.local_dims,
        start,
        100,
        10.0 * threshold,
        |scan_site, points| {
            flat.clear();
            for point in points {
                flat.extend_from_slice(point);
            }
            let f_values = execution.call(GlobalIndexBatch::new(flat, n_sites, points.len())?)?;
            if f_values.len() != points.len() {
                return Err(anyhow::anyhow!(
                    "batch evaluator returned {} values for {} global-pivot candidates",
                    f_values.len(),
                    points.len(),
                ));
            }
            let shape = [n_sites, points.len()];
            let values = ColMajorArrayRef::new(flat, &shape).map_err(|error| {
                anyhow::anyhow!("failed to build candidate index array: {error}")
            })?;
            let errors: Vec<f64> = match execution.session() {
                Some(engine) => {
                    let mut session = tensor4all_tensorbackend::explicit::Session::new(engine);
                    let tt_values: Vec<T> = readout_on_session(
                        cache,
                        treetn,
                        site_indices,
                        values,
                        scan_site,
                        &mut session,
                    )?;
                    if tt_values.len() != points.len() {
                        return Err(anyhow::anyhow!(
                            "approximation readout returned {} values for {} global-pivot \
                             candidates",
                            tt_values.len(),
                            points.len(),
                        ));
                    }
                    f_values
                        .into_iter()
                        .zip(tt_values)
                        .map(|(f, tt)| typed_interp_error(f, tt))
                        .collect()
                }
                None => {
                    let tt_values = readout(cache, treetn, site_indices, values, scan_site)?;
                    if tt_values.len() != points.len() {
                        return Err(anyhow::anyhow!(
                            "approximation readout returned {} values for {} global-pivot \
                             candidates",
                            tt_values.len(),
                            points.len(),
                        ));
                    }
                    f_values
                        .into_iter()
                        .zip(tt_values)
                        .map(|(f, tt)| interp_error(f, tt))
                        .collect()
                }
            };
            if errors.iter().any(|error| !error.is_finite()) {
                return Err(anyhow::anyhow!("non-finite global pivot residual"));
            }
            Ok(errors)
        },
    )?;
    Ok((error, point))
}

/// Real and imaginary parts of a scalar, used to estimate the interpolation
/// error `|f(idx) - tt(idx)|` in `f64` space.
///
/// Implemented for `f32`, `f64`, `num_complex::Complex32`, and
/// `num_complex::Complex64`, the scalar types supported by the tree TCI2
/// optimization loop.
pub trait ScalarParts {
    /// Real part as `f64`.
    fn real_part(self) -> f64;
    /// Imaginary part as `f64` (zero for real scalars).
    fn imag_part(self) -> f64;
}

impl ScalarParts for f32 {
    fn real_part(self) -> f64 {
        self as f64
    }

    fn imag_part(self) -> f64 {
        0.0
    }
}

impl ScalarParts for f64 {
    fn real_part(self) -> f64 {
        self
    }

    fn imag_part(self) -> f64 {
        0.0
    }
}

impl ScalarParts for num_complex::Complex32 {
    fn real_part(self) -> f64 {
        self.re as f64
    }

    fn imag_part(self) -> f64 {
        self.im as f64
    }
}

impl ScalarParts for num_complex::Complex64 {
    fn real_part(self) -> f64 {
        self.re
    }

    fn imag_part(self) -> f64 {
        self.im
    }
}

fn interp_error<T: ScalarParts + Copy>(f_value: T, tt_value: AnyScalar) -> f64 {
    let re = f_value.real_part() - tt_value.real();
    let im = f_value.imag_part() - tt_value.imag();
    (re * re + im * im).sqrt()
}

/// [`interp_error`] for a typed approximation value, with the same `f64`
/// arithmetic after each part is widened by [`ScalarParts`].
fn typed_interp_error<T: ScalarParts + Copy>(f_value: T, tt_value: T) -> f64 {
    let re = f_value.real_part() - tt_value.real_part();
    let im = f_value.imag_part() - tt_value.imag_part();
    (re * re + im * im).sqrt()
}

#[cfg(test)]
mod tests;
