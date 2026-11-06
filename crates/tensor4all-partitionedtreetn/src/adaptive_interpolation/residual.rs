//! The residual-TT estimator of a patch's L2 error.
//!
//! The uniform sample of a patch is a decision statistic that can miss a
//! residual concentrated on a corner or an edge. This module builds the
//! estimator that replaces it under
//! [`L2Estimator::ResidualTt`](super::L2Estimator::ResidualTt): it
//! interpolates the residual `r = f - f~` with the same engine and reads the
//! exact norm of the resulting network.
//!
//! # Inputs
//!
//! The run uses the patch topology and the patch's active node sites, so its
//! site order is the patch's active site order. The caller supplies the
//! initial pivots and the extra global-search starts: the largest residuals of
//! the uniform sample, the patch corners, and the patch edges
//! ([`super::corners`]). The residual's own bond cap is
//! `max(ceil(chi_patch * ratio), floor)`, with `chi_patch` the largest bond
//! dimension of the measured patch (zero for the zero approximation).
//!
//! # Value
//!
//! `rms_P(r) = exp(log ||TT_r||) / sqrt(|P|)`, read with
//! [`TreeTN::log_norm_on_session`](tensor4all_treetn::TreeTN::log_norm_on_session).
//! Like every norm of the report, it shares the overflow and underflow limit
//! of `log_norm`: the center norm is squared before the logarithm, so a
//! residual norm above about `1.34e154` overflows and one below about
//! `1e-154` underflows to zero.
//!
//! # Non-convergence
//!
//! A capped residual TT captures the largest components first, so its norm is
//! a lower bound on the residual norm. When the run stops at its bond cap (or
//! at the engine's iteration limit) the caller uses the larger of this value
//! and the uniform sample's RMS. A run whose initial pivots are all exactly
//! zero returns
//! [`InterpolationError::AllSamplesZero`](tensor4all_treetn::interpolation::InterpolationError::AllSamplesZero)
//! before its global pivot search; the estimate is then `0` and no search ran.
//!
//! # Floating-point order
//!
//! One fresh [`TreeTNCachedEvaluator`] serves the whole run, centered at the
//! smallest node name, with [`EvaluationHint::default`] for every batch, like
//! a measurement. The residual run evaluates `f` through the patch cache, so
//! points the sample already evaluated are not evaluated again.

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{ColMajorArray, ColMajorArrayRef, CommonScalar, TensorElement};
use tensor4all_treetn::interpolation::{
    EvalContext, InterpolationError, InterpolationProblem, InterpolationTermination,
    TreeInterpolator,
};
use tensor4all_treetn::{
    CachedEvaluatorOptions, EvaluationHint, TreeTNCachedEvaluator, TreeTNOperationError,
};

use super::cache::{is_finite, PatchSampler};
use super::sampling::reserve_point_list;
use super::verify::{full_coordinates, MeasureError, MeasureTarget};

/// The residual run's absolute tolerance as a fraction of the patch
/// allowance: a tenth of `tau`, so that the estimate resolves the allowance
/// without spending rank on the part of the residual far below it.
pub(super) const RESIDUAL_TOLERANCE_FACTOR: f64 = 0.1;

/// The value a residual run produced.
#[derive(Debug)]
pub(super) struct ResidualEstimate {
    /// `||TT_r||_2 / sqrt(|P|)`.
    pub(super) rms: f64,
    /// The run did not converge, so the deciding estimate is the larger of
    /// this value and the uniform sample's RMS.
    pub(super) saturated: bool,
    /// The run stopped at its bond cap (a subset of `saturated`).
    pub(super) bond_capped: bool,
    /// The run's pivots with a nonzero residual, ranked by `|r|` (ties in
    /// pivot order), each point once.
    pub(super) pivots: Vec<Vec<usize>>,
}

/// The bond cap of a residual run: `max(ceil(chi_patch * ratio), floor)`.
pub(super) fn bond_cap(chi_patch: usize, ratio: f64, floor: usize) -> usize {
    let scaled = (chi_patch as f64 * ratio).ceil();
    let scaled = if scaled >= usize::MAX as f64 {
        usize::MAX
    } else {
        scaled as usize
    };
    scaled.max(floor).max(1)
}

/// Run the residual TT of one measured patch and report its RMS.
pub(super) fn estimate<T, V, E>(
    engine: &E,
    problem: &InterpolationProblem<V>,
    sampler: &PatchSampler<'_, T>,
    target: &MeasureTarget<'_, V>,
    patch_points: f64,
    ctx: &mut EvalContext<'_>,
) -> Result<ResidualEstimate, MeasureError>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
{
    let mut values = ResidualValues::new(sampler, target)?;
    // The engine wraps a callback failure in its own evaluator error, so the
    // typed failure is kept here and recovered below: a network readout or
    // point-list failure of the residual must not be blamed on the caller's
    // evaluator.
    let mut callback_error: Option<MeasureError> = None;
    let result = engine.interpolate(problem, ctx, |batch, ctx| {
        values.batch_residuals(batch, ctx).map_err(|error| {
            let message = error.to_string();
            callback_error = Some(error);
            anyhow::anyhow!("{message}")
        })
    });
    let outcome = match result {
        Ok(outcome) => outcome,
        Err(InterpolationError::AllSamplesZero) => {
            return Ok(ResidualEstimate {
                rms: 0.0,
                saturated: false,
                bond_capped: false,
                pivots: Vec::new(),
            });
        }
        Err(InterpolationError::Evaluator { source }) => {
            return Err(callback_error
                .take()
                .unwrap_or(MeasureError::Evaluator(source)));
        }
        Err(InterpolationError::InvalidProblem { message }) => {
            return Err(MeasureError::Network(format!(
                "the residual interpolation problem is invalid: {message}"
            )));
        }
        Err(InterpolationError::Engine { source }) => {
            return Err(MeasureError::Network(format!(
                "the residual interpolation engine failed: {source}"
            )));
        }
        Err(other) => {
            return Err(MeasureError::Network(format!(
                "the residual interpolation failed: {other}"
            )));
        }
    };
    let termination = outcome.termination;
    let saturated = termination != InterpolationTermination::Converged;
    let bond_capped = termination == InterpolationTermination::BondCapReached;
    let mut network = outcome.network;
    let log_norm = network
        .log_norm_on_session(ctx.session())
        .map_err(|error| {
            MeasureError::Network(format!("reading the residual network norm failed: {error}"))
        })?;
    let rms = (log_norm - 0.5 * patch_points.ln()).exp();
    let mut pivot_points = Vec::new();
    if let Some(pivots) = outcome.pivots.as_ref() {
        pivot_points = pivot_columns(pivots, target.active_dims)?;
    }
    let pivots = ranked_pivots(&mut values, &pivot_points, ctx)?;
    Ok(ResidualEstimate {
        rms,
        saturated,
        bond_capped,
        pivots,
    })
}

/// The columns of an outcome's pivots as active-coordinate points, checked
/// against the patch's active dimensions.
fn pivot_columns(
    pivots: &ColMajorArray<usize>,
    active_dims: &[usize],
) -> Result<Vec<Vec<usize>>, MeasureError> {
    let (Some(n_rows), Some(n_cols)) = (pivots.nrows(), pivots.ncols()) else {
        return Err(MeasureError::Network(format!(
            "the residual outcome pivots have shape {:?}, expected a 2D array",
            pivots.shape()
        )));
    };
    if n_rows != active_dims.len() {
        return Err(MeasureError::Network(format!(
            "the residual outcome pivots have {n_rows} rows, expected {} active sites",
            active_dims.len()
        )));
    }
    (0..n_cols)
        .map(|column| {
            let point = pivots.column(column).unwrap_or_default();
            for (slot, (&value, &dim)) in point.iter().zip(active_dims).enumerate() {
                if value >= dim {
                    return Err(MeasureError::Network(format!(
                        "the residual outcome pivot {column} has coordinate {value} at active \
                         site {slot}, out of range for dimension {dim}"
                    )));
                }
            }
            Ok(point.to_vec())
        })
        .collect()
}

/// The distinct points among `points` with a nonzero residual, ranked by
/// `|r|` and keeping the first occurrence of a repeated point.
fn ranked_pivots<T, V>(
    values: &mut ResidualValues<'_, '_, '_, T, V>,
    points: &[Vec<usize>],
    ctx: &mut EvalContext<'_>,
) -> Result<Vec<Vec<usize>>, MeasureError>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    if points.is_empty() {
        return Ok(Vec::new());
    }
    let mut flat = reserve_point_list(values.n_active, points.len(), "residual pivots")
        .map_err(MeasureError::PointList)?;
    for point in points {
        flat.extend_from_slice(point);
    }
    let residuals = values.flat_residuals(&flat, points.len(), ctx)?;
    let mut ranked: Vec<(usize, f64)> = residuals
        .iter()
        .enumerate()
        .map(|(index, value)| (index, value.abs_val()))
        .filter(|&(_, magnitude)| magnitude > 0.0)
        .collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    let mut seen = HashSet::new();
    Ok(ranked
        .into_iter()
        .filter(|&(index, _)| seen.insert(points[index].clone()))
        .map(|(index, _)| points[index].clone())
        .collect())
}

/// `f - f~` on active-coordinate batches, with one cached evaluator for the
/// patch network (or none for the zero approximation).
struct ResidualValues<'s, 'a, 'n, T, V>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    sampler: &'s PatchSampler<'a, T>,
    evaluator: Option<TreeTNCachedEvaluator<'n, V>>,
    fixed: &'n [Option<usize>],
    n_active: usize,
}

impl<'s, 'a, 'n, T, V> ResidualValues<'s, 'a, 'n, T, V>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    fn new(
        sampler: &'s PatchSampler<'a, T>,
        target: &MeasureTarget<'n, V>,
    ) -> Result<Self, MeasureError> {
        let evaluator = match target.network {
            None => None,
            Some(network) => {
                let center = network.node_names().into_iter().min();
                let options = CachedEvaluatorOptions {
                    center,
                    ..CachedEvaluatorOptions::default()
                };
                Some(
                    TreeTNCachedEvaluator::new(network, target.sites, options).map_err(
                        |error| {
                            MeasureError::Network(format!(
                                "building the residual evaluator failed: {error}"
                            ))
                        },
                    )?,
                )
            }
        };
        Ok(Self {
            sampler,
            evaluator,
            fixed: target.fixed,
            n_active: target.active_dims.len(),
        })
    }

    /// The residual at a column-major `[n_active, n_points]` batch.
    fn batch_residuals(
        &mut self,
        batch: ColMajorArrayRef<'_, usize>,
        ctx: &mut EvalContext<'_>,
    ) -> Result<Vec<T>, MeasureError> {
        let shape = batch.shape();
        if shape.len() != 2 || shape[0] != self.n_active {
            return Err(MeasureError::Evaluator(anyhow::anyhow!(
                "the residual run sent a batch of shape {shape:?}, expected 2 rows with {} \
                 active sites",
                self.n_active
            )));
        }
        self.flat_residuals(batch.data(), shape[1], ctx)
    }

    /// The residual at a column-major flat list of `n_points` points.
    fn flat_residuals(
        &mut self,
        flat: &[usize],
        n_points: usize,
        ctx: &mut EvalContext<'_>,
    ) -> Result<Vec<T>, MeasureError> {
        let shape = [self.n_active, n_points];
        let batch = ColMajorArrayRef::new(flat, &shape)
            .map_err(|error| MeasureError::Evaluator(anyhow::anyhow!("{error}")))?;
        let values = self
            .sampler
            .sample(batch, ctx)
            .map_err(MeasureError::Evaluator)?;
        let Some(evaluator) = self.evaluator.as_mut() else {
            return Ok(values);
        };
        let n_sites = self.fixed.len();
        let full = full_coordinates(flat, self.n_active, self.fixed, "residual full-coordinate")
            .map_err(MeasureError::PointList)?;
        let full_shape = [n_sites, n_points];
        let full_batch = ColMajorArrayRef::new(&full, &full_shape)
            .map_err(|error| MeasureError::Evaluator(anyhow::anyhow!("{error}")))?;
        let approximations = evaluator
            .evaluate_batched_typed_on_session::<T>(
                ctx.session(),
                full_batch,
                EvaluationHint::default(),
            )
            .map_err(|error: TreeTNOperationError| {
                MeasureError::Network(format!(
                    "evaluating the patch network for the residual failed: {error}"
                ))
            })?;
        if approximations.len() != n_points {
            return Err(MeasureError::Network(format!(
                "the patch network returned {} values for {n_points} residual points",
                approximations.len()
            )));
        }
        if let Some(index) = approximations.iter().position(|&value| !is_finite(value)) {
            return Err(MeasureError::Network(format!(
                "the patch network has a non-finite value at the point {:?}",
                &full[index * n_sites..(index + 1) * n_sites]
            )));
        }
        Ok(values
            .iter()
            .zip(approximations)
            .map(|(&value, approximation)| value - approximation)
            .collect())
    }
}
