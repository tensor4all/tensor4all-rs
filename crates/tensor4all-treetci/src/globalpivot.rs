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
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::MatrixLuciScalar as Scalar;
use tensor4all_core::{AnyScalar, ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};

/// Search for multi-indices where the current approximation error is large.
///
/// Algorithm (mirrors `DefaultGlobalPivotFinder` in `tensor4all-tensorci`):
///
/// 1. Materialize the current [`TreeTCI2`] state as a `TreeTN`.
/// 2. Draw `nsearch` random starting points.
/// 3. For each starting point, sweep every site coordinate and keep the
///    point with the largest interpolation error `|f(idx) - tt(idx)|`.
/// 4. Keep points whose error exceeds `abs_tol * tol_margin`.
/// 5. Return at most `max_nglobal_pivot` distinct points.
///
/// The approximation is read at all `nsearch * sum(local_dims)` candidates
/// in one batch through a single [`TreeTNCachedEvaluator`], which shares
/// subtree environments between candidates instead of contracting the whole
/// network once per point.
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
/// * `seed` -- RNG seed; a fixed seed makes the search deterministic.
///
/// # Returns
///
/// Up to `max_nglobal_pivot` distinct full-site multi-indices where the
/// current approximation is (likely) poor, strongest first. An empty vector
/// when nothing exceeds the threshold. Among candidates with equal errors the
/// one generated first wins: within a start, the lower site and then the
/// lower local value; across starts, the earlier start.
///
/// # Errors
///
/// Returns an error when the current state cannot be materialized as a
/// `TreeTN` (a rank mismatch or a singular pivot matrix), when the batch
/// evaluator returns a wrong number of values (a batch length mismatch),
/// when `abs_tol` or `tol_margin` is not finite and nonnegative (an
/// invalid configuration), when the candidate index array shape is
/// malformed (a shape mismatch), or when reading the materialized
/// approximation at the candidates fails (a contraction failure).
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
    let params = SearchParams {
        nsearch,
        max_nglobal_pivot,
        tol_margin,
        abs_tol,
        seed,
    };
    search_with_readout(state, evaluate, params, cached_batched_readout)
}

/// Scalar parameters of one global pivot search, as passed to
/// [`find_global_pivots`].
#[derive(Clone, Copy, Debug)]
struct SearchParams {
    nsearch: usize,
    max_nglobal_pivot: usize,
    tol_margin: f64,
    abs_tol: f64,
    seed: u64,
}

/// Reads the materialized approximation at every candidate in one batch.
///
/// `candidates` has shape `[site_indices.len(), n_points]` in column-major
/// layout (one candidate per column, rows in `site_indices` order), which is
/// the layout [`TreeTNCachedEvaluator::evaluate_batched`] expects. The
/// evaluator is built once per search, so its environment caches are shared
/// by all candidates of that search. A plain batched call is used on purpose:
/// splitting the batch per varied site with `evaluate_batched_with_hint`
/// measured 1.6-1.9x slower on the issue #792 workloads.
fn cached_batched_readout(
    treetn: &TreeTN<IdxTensor, usize>,
    site_indices: &[DynIndex],
    candidates: ColMajorArrayRef<'_, usize>,
) -> Result<Vec<AnyScalar>> {
    let mut evaluator =
        TreeTNCachedEvaluator::new(treetn, site_indices, CachedEvaluatorOptions::default())
            .map_err(anyhow::Error::from)?;
    evaluator
        .evaluate_batched(candidates)
        .map_err(anyhow::Error::from)
}

/// [`find_global_pivots`] with the approximation readout supplied by the
/// caller. Production passes [`cached_batched_readout`]; tests also pass the
/// pointwise [`TreeTN::evaluate`] to check that the readout does not change
/// the selected pivots.
fn search_with_readout<T, F, R>(
    state: &TreeTCI2<T>,
    evaluate: F,
    params: SearchParams,
    readout: R,
) -> TreeTciResult<Vec<MultiIndex>>
where
    T: FullPivLuScalar + Scalar + tensor4all_core::TensorElement + ScalarParts,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    R: FnOnce(
        &TreeTN<IdxTensor, usize>,
        &[DynIndex],
        ColMajorArrayRef<'_, usize>,
    ) -> Result<Vec<AnyScalar>>,
{
    let SearchParams {
        nsearch,
        max_nglobal_pivot,
        tol_margin,
        abs_tol,
        seed,
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

    // Candidate points: for each random start, each site coordinate swept
    // over its full local dimension (same local search as the chain finder).
    let candidate_count = (0..n_sites)
        .try_fold(0usize, |count, site| {
            count.checked_add(state.local_dims[site])
        })
        .and_then(|per_start| per_start.checked_mul(nsearch))
        .ok_or_else(|| anyhow::anyhow!("global-pivot candidate count overflowed usize"))?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut points: Vec<MultiIndex> = Vec::with_capacity(candidate_count);
    for _ in 0..nsearch {
        let start: MultiIndex = (0..n_sites)
            .map(|site| rng.random_range(0..state.local_dims[site]))
            .collect();
        for site in 0..n_sites {
            for value in 0..state.local_dims[site] {
                let mut point = start.clone();
                point[site] = value;
                points.push(point);
            }
        }
    }

    // Evaluate the function at all candidates in one batch.
    let flat_len = n_sites
        .checked_mul(points.len())
        .ok_or_else(|| anyhow::anyhow!("global-pivot flat batch size overflowed usize"))?;
    let mut flat = Vec::with_capacity(flat_len);
    for point in &points {
        flat.extend_from_slice(point);
    }
    let f_values = evaluate(GlobalIndexBatch::new(&flat, n_sites, points.len())?)?;
    if f_values.len() != points.len() {
        return Err(anyhow::anyhow!(
            "batch evaluator returned {} values for {} global-pivot candidates",
            f_values.len(),
            points.len()
        )
        .into());
    }

    // Read the current approximation at all candidates in one batch. `flat`
    // holds one candidate per column of `n_sites` rows (column-major), in the
    // row order of `site_indices`.
    let shape = [n_sites, points.len()];
    let values_ref = ColMajorArrayRef::new(&flat, &shape)
        .map_err(|error| anyhow::anyhow!("failed to build candidate index array: {error}"))?;
    let tt_values = readout(&treetn, &site_indices, values_ref)?;
    if tt_values.len() != points.len() {
        return Err(anyhow::anyhow!(
            "approximation readout returned {} values for {} global-pivot candidates",
            tt_values.len(),
            points.len()
        )
        .into());
    }

    // Local search per starting point.
    //
    // Tie-breaking: the strict `>` keeps the first candidate in generation
    // order (lower site, then lower local value) among equal errors; below,
    // the stable sort keeps the earlier start among equal errors and the
    // deduplication keeps the first occurrence. The cached readout contracts
    // in a different order than the pointwise `TreeTN::evaluate` it replaced
    // (issue #792), so `tt` can differ from it in the last bits. That can only
    // reorder two distinct candidates whose errors agree to within that
    // rounding, i.e. candidates that are indistinguishable at working
    // precision; the rule above is applied unchanged to the computed errors.
    let mut best: Vec<(f64, MultiIndex)> = Vec::new();
    let mut point_index = 0usize;
    for _ in 0..nsearch {
        let mut start_best: Option<(f64, MultiIndex)> = None;
        for site in 0..n_sites {
            for _value in 0..state.local_dims[site] {
                let error = interp_error(f_values[point_index], tt_values[point_index].clone());
                if start_best
                    .as_ref()
                    .is_none_or(|(best_error, _)| error > *best_error)
                {
                    start_best = Some((error, points[point_index].clone()));
                }
                point_index += 1;
            }
        }
        if let Some((error, point)) = start_best
            && error > abs_tol * tol_margin
        {
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
