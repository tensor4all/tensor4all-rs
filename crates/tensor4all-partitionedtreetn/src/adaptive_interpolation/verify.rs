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
use tensor4all_tensorbackend::explicit::Session;
use tensor4all_treetn::interpolation::EvalContext;
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
    session: &mut Session<'_>,
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
        values.extend(evaluator.evaluate_batched_typed_on_session::<T>(
            session,
            batch,
            EvaluationHint::default(),
        )?);
    }
    Ok(values)
}

/// Complete a column-major `[n_active, n_points]` list of active coordinates
/// with the fixed coordinates, in site order. Returns an error if the full
/// point list cannot be represented or reserved.
pub(super) fn full_coordinates(
    active_points: &[usize],
    n_active: usize,
    fixed: &[Option<usize>],
    what: &str,
) -> Result<Vec<usize>, String> {
    let stride = n_active.max(1);
    let n_points = active_points.len() / stride;
    let mut full = reserve_point_list(fixed.len(), n_points, what)?;
    for point in active_points.chunks(stride) {
        let mut active = point.iter().copied();
        full.extend(
            fixed
                .iter()
                .map(|fixed| fixed.or_else(|| active.next()).unwrap_or_default()),
        );
    }
    Ok(full)
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
    /// The residual run's pivots with a nonzero residual, ranked by `|r|`
    /// (ties in pivot order), when a residual TT decided the measurement;
    /// empty otherwise. They are seeds for the next run, not witnesses.
    pub(super) residual_pivots: Vec<Vec<usize>>,
}

impl Measured {
    /// The distinct points that seed the next engine run, truncated to
    /// `limit`: the measured points with a residual above `tau`, largest
    /// first (ties in measurement order), then the residual run's nonzero
    /// pivots ranked by `|r|`.
    pub(super) fn worst_points(&self, tau: f64, limit: usize) -> Vec<Vec<usize>> {
        let mut order: Vec<usize> = (0..self.points.len())
            .filter(|&index| self.residuals[index] > tau)
            .collect();
        // A stable sort keeps measurement order among equal residuals.
        order.sort_by(|&left, &right| self.residuals[right].total_cmp(&self.residuals[left]));
        let mut seen = HashSet::new();
        let mut points: Vec<Vec<usize>> = Vec::new();
        for point in order
            .into_iter()
            .map(|index| &self.points[index])
            .chain(&self.residual_pivots)
        {
            if points.len() >= limit {
                break;
            }
            if seen.insert(point.clone()) {
                points.push(point.clone());
            }
        }
        points
    }
}

/// Why a measurement failed.
#[derive(Debug)]
pub(super) enum MeasureError {
    /// The requested measurement point list cannot be represented or reserved.
    PointList(String),
    /// The function evaluator failed or returned an unusable value.
    Evaluator(anyhow::Error),
    /// The patch network could not be evaluated or gave a non-finite value.
    Network(String),
}

impl std::fmt::Display for MeasureError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PointList(message) | Self::Network(message) => formatter.write_str(message),
            Self::Evaluator(source) => write!(formatter, "{source}"),
        }
    }
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
pub(super) fn measure<T, V>(
    sampler: &PatchSampler<'_, T>,
    target: &MeasureTarget<'_, V>,
    plan: PointPlan,
    ctx: &mut EvalContext<'_>,
) -> Result<Measured, MeasureError>
where
    T: CommonScalar + TensorElement,
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
    let values = sampler
        .sample(batch, ctx)
        .map_err(MeasureError::Evaluator)?;

    let approximations: Option<Vec<T>> = match target.network {
        None => None,
        Some(network) => {
            let full = full_coordinates(&flat, n_active, target.fixed, "full-coordinate")
                .map_err(MeasureError::PointList)?;
            let values = network_values::<T, V>(network, target.sites, &full, ctx.session())
                .map_err(|error| {
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
        residual_pivots: Vec::new(),
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
        residual_rms: None,
        residual_saturated: false,
        sample_rms: None,
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

/// The measurement of a residual TT: the deciding estimate is
/// `residual_rms`, or `max(residual_rms, sample.rms)` when the residual run
/// did not converge. The seed sample supplies `points`, `patch_points`,
/// `max_residual`, and `sample_rms`; its Monte-Carlo standard error is kept
/// only when the sample decided or co-decided the value.
pub(super) fn residual_statistics(
    sample: L2Measurement,
    residual_rms: f64,
    saturated: bool,
) -> L2Measurement {
    let rms = if saturated {
        residual_rms.max(sample.rms)
    } else {
        residual_rms
    };
    L2Measurement {
        method: MeasurementMethod::ResidualTt,
        points: sample.points,
        patch_points: sample.patch_points,
        rms,
        mean_square_rel_std_error: if saturated && sample.rms >= residual_rms {
            sample.mean_square_rel_std_error
        } else {
            0.0
        },
        max_residual: sample.max_residual,
        residual_rms: Some(residual_rms),
        residual_saturated: saturated,
        sample_rms: Some(sample.rms),
    }
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
/// so that finite terms cannot overflow it. An infinite term makes the norm
/// infinite; it never turns it into `NaN`.
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
        } else if self.scale.is_finite() {
            // An infinite sum absorbs every term; `inf / inf` would be NaN.
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

/// The values of one region that the global error combines: an accepted or
/// zero patch ([`Term::of`]), or a merged region ([`Term::merged`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Term {
    /// `|P|` as `f64`.
    pub(super) patch_points: f64,
    /// The deciding RMS error of the region.
    pub(super) rms: f64,
    /// Whether `rms` is exact up to rounding (an `Exact` or `Exhaustive`
    /// acceptance, or a merge of such regions).
    pub(super) certified: bool,
    /// The region's part of an audited estimate: the acceptance itself when
    /// certified, otherwise the audit; `None` for an unaudited estimate.
    pub(super) audited: Option<AuditedTerm>,
    /// `false` for a patch retained as `ToleranceNotMet`.
    pub(super) within_tolerance: bool,
}

/// One region's part of an audited estimate, in RMS units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct AuditedTerm {
    /// The estimated RMS error.
    pub(super) rms: f64,
    /// The RMS of the sampled part, which carries the Monte Carlo variance.
    pub(super) variance_rms: f64,
    /// The relative standard error of the sampled part's mean square.
    pub(super) mean_square_rel_std_error: f64,
}

impl Term {
    /// The term of one accepted or zero patch.
    pub(super) fn of(contribution: &Contribution<'_>) -> Self {
        let certified = matches!(
            contribution.acceptance.method,
            MeasurementMethod::Exact | MeasurementMethod::Exhaustive
        );
        // A residual-TT measurement carries the seed sample's standard error
        // only when the sample co-decided the value
        // (`L2Measurement::residual_saturated`); its variance is measured by
        // its sampled part.
        let audited_term = |measurement: &L2Measurement| AuditedTerm {
            rms: measurement.rms,
            variance_rms: measurement.sample_rms.unwrap_or(measurement.rms),
            mean_square_rel_std_error: measurement.mean_square_rel_std_error,
        };
        let audited = if certified {
            Some(audited_term(contribution.acceptance))
        } else {
            contribution.audit.map(audited_term)
        };
        Self {
            patch_points: contribution.patch_points,
            rms: contribution.acceptance.rms,
            certified,
            audited,
            within_tolerance: contribution.within_tolerance,
        }
    }

    /// The term of the region formed by disjoint `members` whose truncated
    /// sum differs from the exact sum by `truncation_rms` (an L2 norm over
    /// the region divided by the square root of its point count).
    ///
    /// By the triangle inequality the region's error is at most the members'
    /// combined error plus the truncation: `sqrt(sum w_i rms_i^2) +
    /// truncation_rms` with `w_i = |P_i| / |Q|`. The truncation is computed
    /// from networks, so it adds no variance: the sampled parts and their
    /// standard errors combine as they would in the global error, which a
    /// merge without truncation therefore leaves unchanged.
    pub(super) fn merged(members: &[Self], truncation_rms: f64) -> Self {
        let patch_points: f64 = members.iter().map(|m| m.patch_points).sum();
        let weight = |m: &Self| {
            if patch_points > 0.0 {
                (m.patch_points / patch_points).sqrt()
            } else {
                0.0
            }
        };
        let combined = |value: &dyn Fn(&Self) -> f64| {
            let mut sum = ScaledSquares::default();
            for member in members {
                sum.add(weight(member) * value(member));
            }
            sum.norm()
        };
        let rms = combined(&|m| m.rms) + truncation_rms;
        let audited = members
            .iter()
            .map(|m| m.audited)
            .collect::<Option<Vec<_>>>()
            .map(|parts| {
                let pair: Vec<(f64, AuditedTerm)> = members
                    .iter()
                    .zip(parts)
                    .map(|(member, part)| (weight(member), part))
                    .collect();
                let mut estimate = ScaledSquares::default();
                let mut variance = ScaledSquares::default();
                for &(w, part) in &pair {
                    estimate.add(w * part.rms);
                    variance.add(w * part.variance_rms);
                }
                let variance_rms = variance.norm();
                // A member contributes w_i^2 v_i^2 rel_i to the standard error
                // of the mean square; they combine in quadrature, and the
                // region's relative error is that over its own v^2.
                let mean_square_rel_std_error = if variance_rms > 0.0 && variance_rms.is_finite() {
                    let mut deviation = ScaledSquares::default();
                    for &(w, part) in &pair {
                        let share = (w * part.variance_rms / variance_rms).powi(2);
                        deviation.add(share * part.mean_square_rel_std_error);
                    }
                    deviation.norm()
                } else {
                    0.0
                };
                AuditedTerm {
                    rms: estimate.norm() + truncation_rms,
                    variance_rms,
                    mean_square_rel_std_error,
                }
            });
        Self {
            patch_points,
            rms,
            certified: members.iter().all(|m| m.certified),
            audited,
            within_tolerance: members.iter().all(|m| m.within_tolerance),
        }
    }
}

/// The global error and the certified fraction of the domain, combined in
/// the given (canonical path) order. A `ToleranceNotMet` contribution turns
/// the M3 classification into [`GlobalL2Error::ToleranceNotMet`] with that
/// classification as its basis, and is excluded from the certified fraction.
/// `epsilon` is the machine epsilon of the precision the patch networks were
/// evaluated in; it scales the rounding allowance.
pub(super) fn global_error(
    contributions: &[Contribution<'_>],
    domain_points: f64,
    tau: f64,
    approximation_rms: Option<f64>,
    epsilon: f64,
) -> (GlobalL2Error, f64) {
    let terms: Vec<Term> = contributions.iter().map(Term::of).collect();
    global_error_of_terms(&terms, domain_points, tau, approximation_rms, epsilon)
}

/// [`global_error`] of regions given as [`Term`]s, in the given order.
pub(super) fn global_error_of_terms(
    terms: &[Term],
    domain_points: f64,
    tau: f64,
    approximation_rms: Option<f64>,
    epsilon: f64,
) -> (GlobalL2Error, f64) {
    let certified_fraction: f64 = terms
        .iter()
        .filter(|t| t.within_tolerance && t.certified)
        .map(|t| t.patch_points / domain_points)
        .sum();
    let (rms, basis) = classify(terms, domain_points, tau, approximation_rms, epsilon);
    if terms.iter().all(|t| t.within_tolerance) {
        let global = match basis {
            ToleranceNotMetBasis::ExactOrExhaustive {
                rounding_allowance_rms,
                rounding_limited,
                relative_error_bound,
            } => GlobalL2Error::Certified {
                rms_error: rms,
                rounding_allowance_rms,
                rounding_limited,
                relative_error_bound,
            },
            ToleranceNotMetBasis::Audited {
                mean_square_rel_std_error,
                relative_bound_estimate,
            } => GlobalL2Error::Audited {
                rms_error_estimate: rms,
                mean_square_rel_std_error,
                relative_bound_estimate,
            },
            ToleranceNotMetBasis::AcceptanceOnly => GlobalL2Error::AcceptanceOnly {
                acceptance_statistic_rms: rms,
            },
        };
        return (global, certified_fraction);
    }
    let unmet_fraction = terms
        .iter()
        .filter(|t| !t.within_tolerance)
        .map(|t| t.patch_points / domain_points)
        .sum();
    let global = GlobalL2Error::ToleranceNotMet {
        measured_rms: rms,
        unmet_fraction,
        basis,
    };
    (global, certified_fraction)
}

/// The M3 classification of the combined measurements as its RMS value and
/// basis: exact or exhaustive when every contribution is (`Certified` in
/// M3), acceptance-only when some sampled one has no audit, audited
/// otherwise.
fn classify(
    terms: &[Term],
    domain_points: f64,
    tau: f64,
    approximation_rms: Option<f64>,
    epsilon: f64,
) -> (f64, ToleranceNotMetBasis) {
    let weight = |t: &Term| (t.patch_points / domain_points).sqrt();

    let mut acceptance = ScaledSquares::default();
    for t in terms {
        acceptance.add(weight(t) * t.rms);
    }
    if terms.iter().all(|t| t.certified) {
        let rms_error = acceptance.norm();
        let rounding = approximation_rms.map(|rms| MEASUREMENT_ROUNDING_FACTOR * epsilon * rms);
        let relative_error_bound = approximation_rms.zip(rounding).and_then(|(rms, rounding)| {
            let upper = rms_error * (1.0 + GLOBAL_ROUNDING_MARGIN) + rounding;
            let denominator = (1.0 - GLOBAL_ROUNDING_MARGIN) * rms - upper;
            (denominator > 0.0).then(|| upper / denominator)
        });
        let basis = ToleranceNotMetBasis::ExactOrExhaustive {
            rounding_allowance_rms: rounding,
            rounding_limited: rounding.map(|rounding| rounding >= tau),
            relative_error_bound,
        };
        return (rms_error, basis);
    }
    let Some(parts) = terms
        .iter()
        .map(|t| t.audited)
        .collect::<Option<Vec<AuditedTerm>>>()
    else {
        return (acceptance.norm(), ToleranceNotMetBasis::AcceptanceOnly);
    };

    // Audited: exact and exhaustive contributions as measured, sampled ones
    // from their audits. Each term a_i = sqrt(w_i) rms_i defines the estimate
    // a_i^2; the relative standard error is a Monte-Carlo diagnostic, so the
    // variance of a term is measured by its sampled part
    // (sqrt(w_i) * variance_rms_i) and its relative standard error is zero for
    // a converged residual-TT estimate.
    let terms: Vec<(f64, f64, f64)> = terms
        .iter()
        .zip(parts)
        .map(|(t, part)| {
            (
                weight(t) * part.rms,
                weight(t) * part.variance_rms,
                part.mean_square_rel_std_error,
            )
        })
        .collect();
    let mut estimate = ScaledSquares::default();
    for &(a, _, _) in &terms {
        estimate.add(a);
    }
    let rms_error_estimate = estimate.norm();
    let largest = terms.iter().fold(0.0_f64, |m, &(a, _, _)| m.max(a));
    let mean_square_rel_std_error = if rms_error_estimate == 0.0 || !largest.is_finite() {
        0.0
    } else {
        let mut total = 0.0;
        for &(a, _, _) in &terms {
            total += (a / largest) * (a / largest);
        }
        let mut deviation = ScaledSquares::default();
        for &(_, variance, rel) in &terms {
            let share = (variance / largest) * (variance / largest) / total;
            deviation.add(share * rel);
        }
        deviation.norm()
    };
    let relative_bound_estimate = approximation_rms
        .filter(|&rms| rms > rms_error_estimate)
        .map(|rms| rms_error_estimate / (rms - rms_error_estimate));
    let basis = ToleranceNotMetBasis::Audited {
        mean_square_rel_std_error,
        relative_bound_estimate,
    };
    (rms_error_estimate, basis)
}

/// `||f~|| / sqrt(|X|)` from the log-norms of the accepted patches, combined
/// in the given order; `None` when some log-norm was not measured or is not
/// finite (`-inf`, a zero patch, contributes zero).
pub(super) fn approximation_rms(
    patches: impl Iterator<Item = (Option<f64>, f64)>,
    domain_points: f64,
) -> Option<f64> {
    let mut sum = ScaledSquares::default();
    for (log_norm, patch_points) in patches {
        let log_norm = log_norm?;
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
