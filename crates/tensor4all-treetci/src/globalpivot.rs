//! Automatic global pivot search for [`TreeTCI2`].
//!
//! Port of the `GlobalPivotFinder` machinery from the chain TCI2 crate
//! (`tensor4all-tensorci`): after each sweep the optimizer materializes the
//! current tree approximation and searches random starting points with local
//! coordinate optimization for multi-indices where `|f(idx) - tt(idx)|` is
//! large. Found pivots are injected via [`TreeTCI2::add_global_pivots`] so the
//! next sweep samples regions the local pivot updates missed.
//!
//! The search is enabled by default (`TreeTciOptions::enable_global_pivots`)
//! and runs after every optimization sweep, except after the final sweep of a
//! run that stops at `max_iter` or through the bond-dimension saturation stop.

use crate::error::Result as TreeTciResult;
use crate::{materialize::to_treetn, GlobalIndexBatch, MultiIndex, TreeTCI2};
use anyhow::Result;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::MatrixLuciScalar as Scalar;
use tensor4all_core::{floating_zone_walk, AnyScalar, ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

/// Search for multi-indices where the current approximation error is large.
///
/// Algorithm (mirrors `DefaultGlobalPivotFinder` in `tensor4all-tensorci`):
///
/// 1. Materialize the current [`TreeTCI2`] state as a `TreeTN`.
/// 2. Draw `nsearch` random starting points.
/// 3. Walk each start with [`floating_zone_walk`], retaining the maximizing
///    coordinate between site scans and repeating sweeps until no improvement,
///    an error above `10 * abs_tol * tol_margin`, or 100 sweeps.
/// 4. Keep points whose error exceeds `abs_tol * tol_margin`.
/// 5. Return at most `max_nglobal_pivot` distinct points.
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
/// * `nsearch` -- number of random starting points.
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
/// Returns an error when the current state cannot be materialized as a
/// `TreeTN` (a rank mismatch or a singular pivot matrix), when the batch
/// evaluator returns a wrong number of values (a batch length mismatch),
/// when `abs_tol` or `tol_margin` is not finite and nonnegative (an
/// invalid configuration), when the candidate index array shape is
/// malformed (a shape mismatch), or when reading the materialized
/// approximation at the candidates fails (a contraction failure), or a
/// sampled interpolation residual is non-finite.
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
/// // Starts with a nonzero coordinate reach (1, 1); repeated finds
/// // are merged into one pivot. The (0, 0) start has flat zero fibers.
/// let pivots = find_global_pivots(&state, evaluate, 4, 2, 1.0, 1.0, 42)?;
/// assert_eq!(pivots, vec![vec![1, 1]]);
/// # Ok(())
/// # }
/// ```
pub fn find_global_pivots<T, F>(
    state: &TreeTCI2<T>,
    evaluate: F,
    nsearch: usize,
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
        max_nglobal_pivot,
        tol_margin,
        abs_tol,
        &mut rng,
    )
}

/// Global pivot search on a caller-owned random stream.
///
/// Same as [`find_global_pivots`], but consumes `rng` directly for the random
/// starting points instead of deriving their stream from a seed, so a caller
/// can pin, advance, or share one stream across several searches. Pass a
/// `rand_chacha::ChaCha8Rng` when a deterministic algorithm is required.
///
/// # Errors
/// Returns an error when the current state cannot be materialized as a
/// `TreeTN` (a rank mismatch or a singular pivot matrix), when the batch
/// evaluator returns a wrong number of values (a batch length mismatch),
/// when `abs_tol` or `tol_margin` is not finite and nonnegative (an
/// invalid configuration), when the candidate index array shape is
/// malformed (a shape mismatch), or when reading the materialized
/// approximation at the candidates fails (a contraction failure), or a
/// sampled interpolation residual is non-finite.
pub fn find_global_pivots_with_rng<T, F, R>(
    state: &TreeTCI2<T>,
    evaluate: F,
    nsearch: usize,
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
    find_global_pivots_erased(
        state,
        evaluate,
        nsearch,
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
pub(crate) fn find_global_pivots_erased<T, F>(
    state: &TreeTCI2<T>,
    evaluate: F,
    nsearch: usize,
    max_nglobal_pivot: usize,
    tol_margin: f64,
    abs_tol: f64,
    rng: &mut dyn RngCore,
) -> TreeTciResult<Vec<MultiIndex>>
where
    T: FullPivLuScalar + Scalar + tensor4all_core::TensorElement + ScalarParts,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    let params = SearchParams {
        nsearch,
        max_nglobal_pivot,
        tol_margin,
        abs_tol,
    };
    search_with_readout(state, evaluate, params, rng, cached_walk_readout)
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

/// [`find_global_pivots`] with the approximation readout supplied by the
/// caller. Production passes [`cached_walk_readout`]; tests also pass the
/// pointwise [`TreeTN::evaluate`] to check that the readout does not change
/// the selected pivots.
fn search_with_readout<T, F, R>(
    state: &TreeTCI2<T>,
    evaluate: F,
    params: SearchParams,
    rng: &mut dyn RngCore,
    mut readout: R,
) -> TreeTciResult<Vec<MultiIndex>>
where
    T: FullPivLuScalar + Scalar + tensor4all_core::TensorElement + ScalarParts,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
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
    if nsearch == 0 || max_nglobal_pivot == 0 {
        return Ok(Vec::new());
    }
    let n_sites = state.local_dims.len();
    if n_sites == 0 {
        return Ok(Vec::new());
    }

    // Materialize the current approximation once per search. A degenerate or
    // inconsistent pivot state is a real error; propagate it rather than
    // silently skipping the search.
    let treetn = to_treetn(state, &evaluate, None)?;
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
    let mut best = Vec::with_capacity(nsearch);
    for _ in 0..nsearch {
        start.clear();
        start.extend(state.local_dims.iter().map(|&dim| rng.random_range(0..dim)));
        // Match chain TCI / TensorCrossInterpolation.jl: at most 100 sweeps,
        // early exit above 10 * the acceptance threshold. Infinite early-stop
        // bounds disable only that exit; the sweep/no-improvement bounds remain.
        let (point, error) = floating_zone_walk::<_, anyhow::Error>(
            &state.local_dims,
            &start,
            100,
            10.0 * threshold,
            |scan_site, points| {
                flat.clear();
                for point in points {
                    flat.extend_from_slice(point);
                }
                let f_values = evaluate(GlobalIndexBatch::new(&flat, n_sites, points.len())?)?;
                if f_values.len() != points.len() {
                    return Err(anyhow::anyhow!(
                        "batch evaluator returned {} values for {} global-pivot candidates",
                        f_values.len(),
                        points.len(),
                    ));
                }
                let shape = [n_sites, points.len()];
                let values = ColMajorArrayRef::new(&flat, &shape).map_err(|error| {
                    anyhow::anyhow!("failed to build candidate index array: {error}")
                })?;
                let tt_values = readout(&mut cache, &treetn, &site_indices, values, scan_site)?;
                if tt_values.len() != points.len() {
                    return Err(anyhow::anyhow!(
                        "approximation readout returned {} values for {} global-pivot candidates",
                        tt_values.len(),
                        points.len(),
                    ));
                }
                f_values
                    .into_iter()
                    .zip(tt_values)
                    .map(|(f, tt)| {
                        let error = interp_error(f, tt);
                        if !error.is_finite() {
                            return Err(anyhow::anyhow!("non-finite global pivot residual"));
                        }
                        Ok(error)
                    })
                    .collect()
            },
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

#[cfg(test)]
mod tests;
