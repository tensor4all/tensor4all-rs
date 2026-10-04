//! Driver-side measurement of patch networks.
//!
//! The floating-point operation order of a measurement must be a function of
//! the sorted node names, the positional legs of the stored node tensors, and
//! the batch contents only (see "Determinism" in
//! `docs/design/tree-patching-error-contract.md`). The driver's part of that
//! argument lives here: every measurement uses one fresh
//! [`TreeTNCachedEvaluator`], the smallest node name as a fixed center (no
//! greedy center search), [`EvaluationHint::default`] for every batch, and
//! points in a deterministic order evaluated in chunks of the fixed size
//! [`MEASUREMENT_CHUNK`].

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, TensorElement};
use tensor4all_treetn::{
    CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator, TreeTNOperationError,
};

use super::cache::{is_finite, PatchSampler};
use super::report::{
    GlobalL2Error, L2Measurement, MeasurementMethod, ToleranceNotMetBasis, GLOBAL_ROUNDING_MARGIN,
    MEASUREMENT_ROUNDING_FACTOR,
};
use super::sampling::{all_points, reserve_point_list, uniform_points};

/// Number of points per evaluator batch of a measurement. Fixed, so that the
/// batch composition and call history of a measurement depend only on its
/// point list.
pub(super) const MEASUREMENT_CHUNK: usize = 256;

/// Values of `network` at the column-major `[sites.len(), n_points]` points
/// `points`, with a fresh evaluator centered at the smallest node name, the
/// default hint, and chunks of [`MEASUREMENT_CHUNK`] points.
pub(super) fn network_values<T, V>(
    network: &TreeTN<IdxTensor, V>,
    sites: &[DynIndex],
    points: &[usize],
) -> Result<Vec<T>, TreeTNOperationError>
where
    T: TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let n_sites = sites.len();
    if n_sites == 0 || points.is_empty() {
        return Ok(Vec::new());
    }
    let center = network.node_names().into_iter().min();
    let options = CachedEvaluatorOptions {
        center,
        ..CachedEvaluatorOptions::default()
    };
    let mut evaluator = TreeTNCachedEvaluator::new(network, sites, options)?;
    let mut values = Vec::with_capacity(points.len() / n_sites);
    for chunk in points.chunks(MEASUREMENT_CHUNK * n_sites) {
        let shape = [n_sites, chunk.len() / n_sites];
        let batch = ColMajorArrayRef::new(chunk, &shape)
            .map_err(|error| TreeTNOperationError::from(anyhow::Error::new(error)))?;
        values.extend(evaluator.evaluate_batched_typed::<T>(batch, EvaluationHint::default())?);
    }
    Ok(values)
}

/// Which points a measurement uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PointPlan {
    /// Every point of the patch, column-major (first active site fastest);
    /// `count` is the patch's point count, whose point list fits in `usize`.
    Exhaustive { count: usize },
    /// `count` uniform points with replacement from the stream `seed`.
    Sampled { count: usize, seed: u64 },
}

impl PointPlan {
    /// The plan for a patch of `patch_count` points, `None` when the count
    /// or its point-list capacity cannot fit in a `Vec`: exhaustive up to
    /// `max(max_exhaustive_points, samples)` points, else sampled.
    pub(super) fn for_patch(
        patch_count: Option<usize>,
        samples: usize,
        max_exhaustive_points: usize,
        seed: u64,
    ) -> Self {
        match patch_count {
            Some(count) if count <= max_exhaustive_points.max(samples) => {
                Self::Exhaustive { count }
            }
            _ => Self::Sampled {
                count: samples,
                seed,
            },
        }
    }

    fn method(self) -> MeasurementMethod {
        match self {
            Self::Exhaustive { .. } => MeasurementMethod::Exhaustive,
            Self::Sampled { .. } => MeasurementMethod::Sampled,
        }
    }
}

/// A measurement with the points and residual magnitudes it used.
pub(super) struct Measured {
    pub(super) measurement: L2Measurement,
    /// Measured points in active coordinates, in measurement order.
    pub(super) points: Vec<Vec<usize>>,
    /// `|f - f~|` at every measured point.
    pub(super) residuals: Vec<f64>,
}

impl Measured {
    /// The distinct measured points with a residual above `tau`, largest
    /// residual first (ties in measurement order), at most `limit`.
    pub(super) fn worst_points(&self, tau: f64, limit: usize) -> Vec<Vec<usize>> {
        let mut order: Vec<usize> = (0..self.points.len())
            .filter(|&index| self.residuals[index] > tau)
            .collect();
        // A stable sort keeps measurement order among equal residuals.
        order.sort_by(|&left, &right| self.residuals[right].total_cmp(&self.residuals[left]));
        let mut seen = HashSet::new();
        order
            .into_iter()
            .filter(|&index| seen.insert(&self.points[index]))
            .take(limit)
            .map(|index| self.points[index].clone())
            .collect()
    }
}

/// Why a measurement failed.
pub(super) enum MeasureError {
    /// The requested measurement point list cannot be represented or reserved.
    PointList(String),
    /// The function evaluator failed or returned an unusable value.
    Evaluator(anyhow::Error),
    /// The patch network could not be evaluated or gave a non-finite value.
    Network(String),
}

/// What a measurement compares the function with.
pub(super) struct MeasureTarget<'a, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// The stored patch network, or `None` for the zero approximation.
    pub(super) network: Option<&'a TreeTN<IdxTensor, V>>,
    /// Every site of the problem, in site order.
    pub(super) sites: &'a [DynIndex],
    /// Fixed coordinate of every site, if any.
    pub(super) fixed: &'a [Option<usize>],
    /// Dimensions of the active sites, in site order.
    pub(super) active_dims: &'a [usize],
    /// `|P|` as `f64`.
    pub(super) patch_points: f64,
}

/// Measure `f - f~` on the points of `plan`. Values of `f` come through the
/// patch cache; values of the network from [`network_values`].
pub(super) fn measure<T, F, V>(
    sampler: &PatchSampler<'_, T, F>,
    target: &MeasureTarget<'_, V>,
    plan: PointPlan,
) -> Result<Measured, MeasureError>
where
    T: CommonScalar + TensorElement,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let n_active = target.active_dims.len();
    let flat = match plan {
        PointPlan::Exhaustive { count } => all_points(target.active_dims, count),
        PointPlan::Sampled { count, seed } => uniform_points(target.active_dims, count, seed),
    };
    let flat = flat.map_err(MeasureError::PointList)?;
    let n_points = flat.len() / n_active.max(1);
    let shape = [n_active, n_points];
    let batch = ColMajorArrayRef::new(&flat, &shape)
        .map_err(|error| MeasureError::Evaluator(anyhow::anyhow!("{error}")))?;
    let values = sampler.sample(batch).map_err(MeasureError::Evaluator)?;

    let approximations: Option<Vec<T>> = match target.network {
        None => None,
        Some(network) => {
            let mut full = reserve_point_list(target.fixed.len(), n_points, "full-coordinate")
                .map_err(MeasureError::PointList)?;
            for point in flat.chunks(n_active.max(1)) {
                let mut active = point.iter().copied();
                full.extend(
                    target
                        .fixed
                        .iter()
                        .map(|fixed| fixed.or_else(|| active.next()).unwrap_or_default()),
                );
            }
            let values = network_values::<T, V>(network, target.sites, &full).map_err(|error| {
                MeasureError::Network(format!("evaluating the patch network failed: {error}"))
            })?;
            if values.len() != n_points {
                return Err(MeasureError::Network(format!(
                    "the patch network returned {} values for {n_points} points",
                    values.len()
                )));
            }
            if let Some(index) = values.iter().position(|&value| !is_finite(value)) {
                return Err(MeasureError::Network(format!(
                    "the patch network has a non-finite value at the point {:?}",
                    &full[index * target.fixed.len()..(index + 1) * target.fixed.len()]
                )));
            }
            Some(values)
        }
    };
    let residuals: Vec<f64> = match &approximations {
        None => values.iter().map(|value| value.abs_val()).collect(),
        Some(approximations) => values
            .iter()
            .zip(approximations)
            .map(|(&value, &approximation)| (value - approximation).abs_val())
            .collect(),
    };
    let points = flat
        .chunks(n_active.max(1))
        .take(n_points)
        .map(<[usize]>::to_vec)
        .collect();
    Ok(Measured {
        measurement: statistics(&residuals, plan.method(), target.patch_points),
        points,
        residuals,
    })
}

/// The measurement of residual magnitudes with scaled accumulation, so that
/// finite residuals cannot overflow the sum of squares.
pub(super) fn statistics(
    residuals: &[f64],
    method: MeasurementMethod,
    patch_points: f64,
) -> L2Measurement {
    let n = residuals.len();
    let max_residual = residuals.iter().copied().fold(0.0_f64, f64::max);
    let mut measurement = L2Measurement {
        method,
        points: n,
        patch_points,
        rms: 0.0,
        mean_square_rel_std_error: 0.0,
        max_residual,
    };
    if n == 0 || max_residual == 0.0 {
        return measurement;
    }
    if !max_residual.is_finite() {
        measurement.rms = max_residual;
        return measurement;
    }
    let scaled: Vec<f64> = residuals
        .iter()
        .map(|&r| (r / max_residual) * (r / max_residual))
        .collect();
    let mean = scaled.iter().sum::<f64>() / n as f64;
    measurement.rms = max_residual * mean.sqrt();
    if method == MeasurementMethod::Sampled && n >= 2 {
        let variance = scaled.iter().map(|q| (q - mean) * (q - mean)).sum::<f64>() / (n - 1) as f64;
        measurement.mean_square_rel_std_error = (variance / n as f64).sqrt() / mean;
    }
    measurement
}

/// The root-mean-square value of exact values, with scaled accumulation.
pub(super) fn rms_of(magnitudes: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = magnitudes.clone().count();
    let largest = magnitudes.clone().fold(0.0_f64, f64::max);
    if n == 0 || largest == 0.0 {
        return 0.0;
    }
    let sum: f64 = magnitudes.map(|m| (m / largest) * (m / largest)).sum();
    largest * (sum / n as f64).sqrt()
}

/// A sum of squares `sum a_i^2` kept as `scale^2 * ssq` (LAPACK's `lassq`),
/// so that finite terms cannot overflow it.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ScaledSquares {
    scale: f64,
    ssq: f64,
}

impl ScaledSquares {
    /// Add `a^2` for a nonnegative `a`.
    pub(super) fn add(&mut self, a: f64) {
        if a == 0.0 {
            return;
        }
        if a > self.scale {
            self.ssq = 1.0 + self.ssq * (self.scale / a) * (self.scale / a);
            self.scale = a;
        } else {
            self.ssq += (a / self.scale) * (a / self.scale);
        }
    }

    /// `sqrt(sum a_i^2)`.
    pub(super) fn norm(&self) -> f64 {
        self.scale * self.ssq.sqrt()
    }
}

/// One accepted or zero patch as it enters the global error.
pub(super) struct Contribution<'a> {
    pub(super) patch_points: f64,
    pub(super) acceptance: &'a L2Measurement,
    pub(super) audit: Option<&'a L2Measurement>,
    /// `false` for a patch retained as `ToleranceNotMet`.
    pub(super) within_tolerance: bool,
}

/// The global error and the certified fraction of the domain, combined in
/// the given (canonical path) order. A `ToleranceNotMet` contribution turns
/// the M3 classification into [`GlobalL2Error::ToleranceNotMet`] with that
/// classification as its basis, and is excluded from the certified fraction.
pub(super) fn global_error(
    contributions: &[Contribution<'_>],
    domain_points: f64,
    tau: f64,
    approximation_rms: Option<f64>,
) -> (GlobalL2Error, f64) {
    let exact = |m: &L2Measurement| m.method != MeasurementMethod::Sampled;
    let certified_fraction: f64 = contributions
        .iter()
        .filter(|c| c.within_tolerance && exact(c.acceptance))
        .map(|c| c.patch_points / domain_points)
        .sum();
    let measured = classify(contributions, domain_points, tau, approximation_rms);
    if contributions.iter().all(|c| c.within_tolerance) {
        return (measured, certified_fraction);
    }
    let unmet_fraction = contributions
        .iter()
        .filter(|c| !c.within_tolerance)
        .map(|c| c.patch_points / domain_points)
        .sum();
    let (measured_rms, basis) = match measured {
        GlobalL2Error::Certified {
            rms_error,
            rounding_allowance_rms,
            rounding_limited,
            relative_error_bound,
        } => (
            rms_error,
            ToleranceNotMetBasis::ExactOrExhaustive {
                rounding_allowance_rms,
                rounding_limited,
                relative_error_bound,
            },
        ),
        GlobalL2Error::Audited {
            rms_error_estimate,
            mean_square_rel_std_error,
            relative_bound_estimate,
        } => (
            rms_error_estimate,
            ToleranceNotMetBasis::Audited {
                mean_square_rel_std_error,
                relative_bound_estimate,
            },
        ),
        GlobalL2Error::AcceptanceOnly {
            acceptance_statistic_rms,
        } => (
            acceptance_statistic_rms,
            ToleranceNotMetBasis::AcceptanceOnly,
        ),
        GlobalL2Error::ToleranceNotMet {
            measured_rms,
            basis,
            ..
        } => (measured_rms, basis),
    };
    let global = GlobalL2Error::ToleranceNotMet {
        measured_rms,
        unmet_fraction,
        basis,
    };
    (global, certified_fraction)
}

/// The M3 classification of the combined measurements: `Certified` when
/// every contribution is exact or exhaustive, `AcceptanceOnly` when some
/// sampled one has no audit, `Audited` otherwise. It never returns
/// `ToleranceNotMet`.
fn classify(
    contributions: &[Contribution<'_>],
    domain_points: f64,
    tau: f64,
    approximation_rms: Option<f64>,
) -> GlobalL2Error {
    let weight = |c: &Contribution<'_>| (c.patch_points / domain_points).sqrt();
    let exact = |m: &L2Measurement| m.method != MeasurementMethod::Sampled;

    let mut acceptance = ScaledSquares::default();
    for c in contributions {
        acceptance.add(weight(c) * c.acceptance.rms);
    }
    if contributions.iter().all(|c| exact(c.acceptance)) {
        let rms_error = acceptance.norm();
        let rounding =
            approximation_rms.map(|rms| MEASUREMENT_ROUNDING_FACTOR * f64::EPSILON * rms);
        let relative_error_bound = approximation_rms.zip(rounding).and_then(|(rms, rounding)| {
            let upper = rms_error * (1.0 + GLOBAL_ROUNDING_MARGIN) + rounding;
            let denominator = (1.0 - GLOBAL_ROUNDING_MARGIN) * rms - upper;
            (denominator > 0.0).then(|| upper / denominator)
        });
        return GlobalL2Error::Certified {
            rms_error,
            rounding_allowance_rms: rounding,
            rounding_limited: rounding.map(|rounding| rounding >= tau),
            relative_error_bound,
        };
    }
    if contributions
        .iter()
        .any(|c| !exact(c.acceptance) && c.audit.is_none())
    {
        return GlobalL2Error::AcceptanceOnly {
            acceptance_statistic_rms: acceptance.norm(),
        };
    }

    // Audited: exact and exhaustive contributions as measured, sampled ones
    // from their audits. Each term a_i = sqrt(w_i) rms_i; the variance of
    // the total mean square is sum (a_i^2 rel_i)^2.
    let terms: Vec<(f64, f64)> = contributions
        .iter()
        .map(|c| {
            let measurement = if exact(c.acceptance) {
                c.acceptance
            } else {
                c.audit.unwrap_or(c.acceptance)
            };
            (
                weight(c) * measurement.rms,
                measurement.mean_square_rel_std_error,
            )
        })
        .collect();
    let mut estimate = ScaledSquares::default();
    for &(a, _) in &terms {
        estimate.add(a);
    }
    let rms_error_estimate = estimate.norm();
    let largest = terms.iter().fold(0.0_f64, |m, &(a, _)| m.max(a));
    let mean_square_rel_std_error = if rms_error_estimate == 0.0 || !largest.is_finite() {
        0.0
    } else {
        let mut deviation = ScaledSquares::default();
        let mut total = 0.0;
        for &(a, rel) in &terms {
            let q = (a / largest) * (a / largest);
            total += q;
            deviation.add(q * rel);
        }
        deviation.norm() / total
    };
    let relative_bound_estimate = approximation_rms
        .filter(|&rms| rms > rms_error_estimate)
        .map(|rms| rms_error_estimate / (rms - rms_error_estimate));
    GlobalL2Error::Audited {
        rms_error_estimate,
        mean_square_rel_std_error,
        relative_bound_estimate,
    }
}

/// `||f~|| / sqrt(|X|)` from the log-norms of the accepted patches, combined
/// in the given order; `None` when some log-norm is not finite (`-inf`, a
/// zero patch, contributes zero).
pub(super) fn approximation_rms(
    patches: impl Iterator<Item = (Result<f64, TreeTNOperationError>, f64)>,
    domain_points: f64,
) -> Option<f64> {
    let mut sum = ScaledSquares::default();
    for (log_norm, patch_points) in patches {
        let log_norm = log_norm.ok()?;
        if log_norm == f64::NEG_INFINITY {
            continue;
        }
        if !log_norm.is_finite() {
            return None;
        }
        let rms = (log_norm - 0.5 * patch_points.ln()).exp();
        sum.add((patch_points / domain_points).sqrt() * rms);
    }
    let rms = sum.norm();
    rms.is_finite().then_some(rms)
}

#[cfg(test)]
mod tests;
