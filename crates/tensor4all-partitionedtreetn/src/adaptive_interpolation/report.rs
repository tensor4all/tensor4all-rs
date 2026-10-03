//! Records and the report of [`patched_interpolate`](super::patched_interpolate).

use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_treetn::interpolation::InterpolationTermination;

use crate::{ErrorTolerance, PartitionedTreeTN, Projector};

/// Relative margin covering the floating-point rounding of the budget
/// arithmetic: patch and domain sizes as `f64` (inexact above `2^53`), sums
/// over patches, and the conversion of `TreeTN::log_norm` into an RMS value
/// (an absolute error of about `eps * (|log_norm| + ln |P|)` in the
/// exponent, about `700 eps` for `|P| = 2^1000`). It covers up to about
/// `1e8` patches.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::GLOBAL_ROUNDING_MARGIN;
///
/// assert_eq!(GLOBAL_ROUNDING_MARGIN, 1e-8);
/// ```
pub const GLOBAL_ROUNDING_MARGIN: f64 = 1e-8;

/// The constant `c` of the first-order model `c * eps * ||f~||` of the
/// rounding of a measured residual, which comes from evaluating the patch
/// networks.
///
/// It is a model, not a proven bound: with heavy cancellation in a
/// contraction the evaluation error can exceed it. It never enters an
/// acceptance decision; it only sets the absolute rounding term of a
/// certified error ([`GlobalL2Error::Certified`]). The value is the largest
/// ratio `||evaluated - exact|| / (eps ||exact||)` observed in a calibration
/// on the test trees (162, rounded up from 161.23, for a network whose
/// contraction cancels to about 1% of its terms; random networks gave at
/// most 1.9 and interpolated patches at most 0.65) times a headroom factor
/// of 4. The calibration is the ignored test
/// `tests/adaptive_rounding_calibration.rs`.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::MEASUREMENT_ROUNDING_FACTOR;
///
/// assert_eq!(MEASUREMENT_ROUNDING_FACTOR, 4.0 * 162.0);
/// // For ||f~|| / sqrt(|X|) = 1 the rounding term is about 1.4e-13.
/// let rounding_rms = MEASUREMENT_ROUNDING_FACTOR * f64::EPSILON;
/// assert!(rounding_rms > 1e-13 && rounding_rms < 2e-13);
/// ```
pub const MEASUREMENT_ROUNDING_FACTOR: f64 = 648.0;

/// How the error of a patch was measured.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::MeasurementMethod;
///
/// assert_ne!(MeasurementMethod::Exhaustive, MeasurementMethod::Sampled);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeasurementMethod {
    /// The patch was built from all its values (at most one active site); its
    /// error is zero.
    Exact,
    /// The residual was evaluated at every point of the patch: exact up to
    /// floating-point rounding, a certificate unaffected by selection.
    Exhaustive,
    /// The residual was evaluated at fresh uniform points of the patch. As an
    /// acceptance measurement it is only a decision statistic; as an audit its
    /// mean square is an unbiased estimate (its RMS is not), never a bound.
    /// Unbiased refers to the average over draws: a residual concentrated on a
    /// small part of the patch, such as a localized feature that enters it
    /// only through a corner or an edge, can be missed by most draws, which
    /// then report a small error, and the standard error, computed from the
    /// same points, does not reveal the miss.
    Sampled,
}

/// One L2 measurement of a patch residual `r = f - f~_P` (under
/// [`ErrorNorm::L2`](crate::ErrorNorm::L2) only).
///
/// Values are in root-mean-square units, `rms_P(r) = sqrt(sum |r|^2 / |P|)`,
/// which cannot overflow for finite values; [`L2Measurement::error_norm`]
/// converts to the L2 norm over the patch.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, MeasurementMethod, PatchedInterpolationOptions,
/// };
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // One node with one site: the root is built exactly from its four values.
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     BTreeMap::from([(0usize, vec![DynIndex::new_dyn(4)])]),
///     ColMajorArray::new(vec![], vec![1, 0])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(batch.data().iter().map(|&x| 1.0 + x as f64).collect())
///     },
///     &PatchedInterpolationOptions::new(2),
/// )?;
/// let measurement = result.report.accepted[0].acceptance.as_ref().unwrap();
/// assert_eq!(measurement.method, MeasurementMethod::Exact);
/// assert_eq!((measurement.points, measurement.patch_points), (4, 4.0));
/// assert_eq!(measurement.rms, 0.0);
/// assert_eq!(measurement.error_norm(), Some(0.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct L2Measurement {
    /// How the residual was measured.
    pub method: MeasurementMethod,
    /// Points measured: `|P|` for `Exact` and `Exhaustive`, the drawn sample
    /// count (duplicates included) for `Sampled`.
    pub points: usize,
    /// `|P|` as `f64` (inexact above `2^53`).
    pub patch_points: f64,
    /// `rms_P(f - f~_P)` over the measured points; exact (up to rounding) for
    /// `Exact` and `Exhaustive`.
    pub rms: f64,
    /// Standard error of the mean square divided by the mean square; `0`
    /// unless `Sampled`, and `0` when the mean square is `0`, which carries
    /// no information. It is computed from the measured points, so it does
    /// not reveal a residual concentrated on points that were not drawn.
    pub mean_square_rel_std_error: f64,
    /// Largest `|f - f~_P|` over the measured points.
    pub max_residual: f64,
}

impl L2Measurement {
    /// The measured error as an L2 norm over the patch,
    /// `sqrt(patch_points) * rms`, or `None` when that product overflows.
    ///
    /// # Examples
    ///
    /// See [`L2Measurement`].
    pub fn error_norm(&self) -> Option<f64> {
        finite(self.patch_points.sqrt() * self.rms)
    }
}

/// The record of one accepted patch.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, PatchedInterpolationOptions,
/// };
/// use tensor4all_partitionedtreetn::ErrorNorm;
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::interpolation::InterpolationTermination;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // A single node with one site of dimension 4 is evaluated exactly.
/// let site = DynIndex::new_dyn(4);
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     BTreeMap::from([(0usize, vec![site])]),
///     ColMajorArray::new(vec![], vec![1, 0])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(batch.data().iter().map(|&x| x as f64 - 1.0).collect())
///     },
///     &PatchedInterpolationOptions::new(2).with_error_norm(ErrorNorm::sampled_max()),
/// )?;
/// let record = &result.report.accepted[0];
/// assert!(record.projector.is_empty());
/// assert_eq!(record.termination, InterpolationTermination::Converged);
/// assert_eq!(record.engine_error_estimate, 0.0);
/// assert_eq!(record.max_sample_magnitude, 2.0);
/// assert_eq!(record.max_bond_dim, 1);
/// assert_eq!(record.retries_used, 0);
/// // SampledMax runs no measurement.
/// assert!(record.acceptance.is_none() && record.audit.is_none());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchRecord {
    /// Projector of the patch (its fixed sites).
    pub projector: Projector,
    /// The engine's verdict; always
    /// [`InterpolationTermination::Converged`] for an accepted patch.
    pub termination: InterpolationTermination,
    /// The engine's raw error estimate in its own criterion's units (not an
    /// L2 error), `0` for an exactly evaluated patch.
    pub engine_error_estimate: f64,
    /// Largest sampled magnitude of the patch reported by the engine, or the
    /// largest exact value of an exactly evaluated patch.
    pub max_sample_magnitude: f64,
    /// Largest bond dimension of the patch network (one for an exactly
    /// evaluated patch).
    pub max_bond_dim: usize,
    /// Engine reruns of this patch after failed verifications, `0` if none.
    pub retries_used: usize,
    /// The measurement that accepted the patch (under L2 only).
    pub acceptance: Option<L2Measurement>,
    /// The independent audit of a `Sampled` acceptance (under L2 with
    /// [`VerificationOptions::audit`](super::VerificationOptions::audit)).
    pub audit: Option<L2Measurement>,
}

/// The record of one zero patch: a region approximated by zero and left out
/// of the partition, which treats an absent patch as zero.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, MeasurementMethod, PatchedInterpolationOptions,
/// };
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // f vanishes everywhere: the exact root is a zero patch.
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     BTreeMap::from([(0usize, vec![DynIndex::new_dyn(3)])]),
///     ColMajorArray::new(vec![], vec![1, 0])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(vec![0.0; batch.shape()[1]])
///     },
///     &PatchedInterpolationOptions::new(2),
/// )?;
/// let zero = &result.report.zero_patches[0];
/// assert!(zero.projector.is_empty());
/// assert_eq!(zero.acceptance.as_ref().unwrap().method, MeasurementMethod::Exact);
/// assert!(zero.audit.is_none());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ZeroPatchRecord {
    /// Projector of the patch.
    pub projector: Projector,
    /// The measurement of the zero approximation; `Some` under L2.
    pub acceptance: Option<L2Measurement>,
    /// The independent audit of a `Sampled` zero screen.
    pub audit: Option<L2Measurement>,
}

/// Where the L2 reference norm of a run came from.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::L2ReferenceSource;
///
/// assert_ne!(L2ReferenceSource::Given, L2ReferenceSource::ExactRoot);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum L2ReferenceSource {
    /// [`L2Reference::Given`](crate::L2Reference::Given).
    Given,
    /// Computed from all values of a root patch with at most one site.
    ExactRoot,
    /// No reference was needed: `rtol = 0` (and none was given).
    NotNeeded,
    /// [`L2Reference::MonteCarlo`](crate::L2Reference::MonteCarlo): the mean
    /// of `|f|^2` over uniform root samples.
    #[non_exhaustive]
    MonteCarlo {
        /// Number of root samples.
        samples: usize,
        /// Standard error of the estimated mean square divided by it (`0`
        /// when it is `0`).
        mean_square_rel_std_error: f64,
    },
}

/// Where the max-norm reference of a [`ErrorNorm::SampledMax`](crate::ErrorNorm::SampledMax)
/// run came from.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::MaxReferenceSource;
///
/// assert_ne!(MaxReferenceSource::Given, MaxReferenceSource::MaxOfRootCandidates);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MaxReferenceSource {
    /// The given `max_reference`.
    Given,
    /// The largest exact value of a root patch with at most one site.
    ExactRoot,
    /// The largest magnitude among the root patch's candidate samples.
    MaxOfRootCandidates,
}

/// The global L2 error `E = ||f - f~||` of a run, by what it can claim.
///
/// RMS values are `E / sqrt(|X|)`. Every acceptance measurement satisfies
/// `rms <= tau`, so a certified `rms_error` and `acceptance_statistic_rms`
/// do not exceed `tau` up to [`GLOBAL_ROUNDING_MARGIN`]; an audited estimate
/// can exceed `tau`, because it is reported as measured.
///
/// Bitwise reproducible for a fixed seed, a deterministic evaluator and
/// engine, and a reproducible network evaluation (see "Randomness and
/// determinism" in the module documentation), except the fields that depend
/// on [`L2ErrorReport::approximation_rms`]: `rounding_allowance_rms`,
/// `rounding_limited`, `relative_error_bound`, and
/// `relative_bound_estimate`.
///
/// # Examples
///
/// See [`L2ErrorReport`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GlobalL2Error {
    /// Every contribution is `Exact` or `Exhaustive`. The measured
    /// `rms_error` is at most `tau * (1 + GLOBAL_ROUNDING_MARGIN)`, and the
    /// true `E / sqrt(|X|)` exceeds it by at most `rounding_allowance_rms`,
    /// up to the rounding model of [`MEASUREMENT_ROUNDING_FACTOR`]. The bound
    /// is absolute with respect to the allowance used; with an estimated
    /// reference norm that allowance is itself random.
    #[non_exhaustive]
    Certified {
        /// Measured `E / sqrt(|X|)`.
        rms_error: f64,
        /// `MEASUREMENT_ROUNDING_FACTOR * eps * approximation_rms`; `None`
        /// when `approximation_rms` is `None`.
        rounding_allowance_rms: Option<f64>,
        /// `rounding_allowance_rms >= tau`: the allowance is not resolved by
        /// the measurement. `None` when `approximation_rms` is `None`.
        rounding_limited: Option<bool>,
        /// Conservative bound `E_up / ((1 - GLOBAL_ROUNDING_MARGIN) ||f~|| -
        /// E_up)` on `E / ||f||`, with `E_up = E (1 + GLOBAL_ROUNDING_MARGIN)
        /// + rounding`. `None` when the denominator is not positive or
        /// `approximation_rms` is `None`: then no relative statement exists.
        /// The bound relies on `||f~||` from `TreeTN::log_norm`.
        relative_error_bound: Option<f64>,
    },
    /// Every `Sampled` contribution has an audit: an estimate, not a bound,
    /// and not a guarantee. The audits sample uniformly, like the acceptance,
    /// so a residual concentrated on a small set (for example a localized
    /// feature that enters a patch only through a corner or an edge) can be
    /// missed by both; `rms_error_estimate` and `mean_square_rel_std_error`
    /// can then both be small while the true `E` is orders of magnitude
    /// larger. Only [`GlobalL2Error::Certified`] is a guarantee.
    #[non_exhaustive]
    Audited {
        /// Estimated `E / sqrt(|X|)` (exact and exhaustive contributions as
        /// measured, sampled ones from their audits).
        rms_error_estimate: f64,
        /// Relative standard error of `rms_error_estimate^2`, from the audit
        /// samples; it says nothing about a residual the audits did not draw.
        mean_square_rel_std_error: f64,
        /// Plug-in estimate of the bound `E / (||f~|| - E)` with the audited
        /// estimate of `E` inserted; not an unbiased estimate of `E / ||f||`,
        /// and too small whenever the estimate of `E` is.
        /// `None` when `||f~||` does not exceed the estimated `E` or
        /// `approximation_rms` is `None`. It relies on `||f~||` from
        /// `TreeTN::log_norm`.
        relative_bound_estimate: Option<f64>,
    },
    /// Some `Sampled` contribution has no audit: the combined acceptance
    /// statistics, which are neither a bound nor an estimate of `E`. No
    /// relative statement exists.
    #[non_exhaustive]
    AcceptanceOnly {
        /// Combined acceptance statistics in RMS units.
        acceptance_statistic_rms: f64,
    },
}

impl GlobalL2Error {
    /// The variant's RMS value: `rms_error`, `rms_error_estimate`, or
    /// `acceptance_statistic_rms`.
    ///
    /// # Examples
    ///
    /// See [`L2ErrorReport`].
    pub fn rms_value(&self) -> f64 {
        match self {
            Self::Certified { rms_error, .. } => *rms_error,
            Self::Audited {
                rms_error_estimate, ..
            } => *rms_error_estimate,
            Self::AcceptanceOnly {
                acceptance_statistic_rms,
            } => *acceptance_statistic_rms,
        }
    }
}

/// The L2 error part of a report under [`ErrorNorm::L2`](crate::ErrorNorm::L2).
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, GlobalL2Error, PatchedInterpolationOptions,
/// };
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // An exact root: certified with zero error and a zero relative bound.
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     BTreeMap::from([(0usize, vec![DynIndex::new_dyn(2)])]),
///     ColMajorArray::new(vec![], vec![1, 0])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(batch.data().iter().map(|&x| if x == 0 { 3.0 } else { 4.0 }).collect())
///     },
///     &PatchedInterpolationOptions::new(2),
/// )?;
/// let error = result.report.norm.l2_error().unwrap();
/// assert_eq!(error.domain_points, 2.0);
/// assert_eq!(error.certified_fraction, 1.0);
/// assert!(matches!(
///     error.global,
///     GlobalL2Error::Certified { rms_error, relative_error_bound: Some(bound), .. }
///         if rms_error == 0.0 && bound < 1e-12
/// ));
/// assert_eq!(error.global.rms_value(), 0.0);
/// assert_eq!(error.error_norm(), Some(0.0));
/// // ||f~|| = 5 up to rounding.
/// assert!((error.approximation_norm().unwrap() - 5.0).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct L2ErrorReport {
    /// `|X|` as `f64`.
    pub domain_points: f64,
    /// The global error by what it can claim.
    pub global: GlobalL2Error,
    /// Fraction of `|X|` whose contribution is `Exact` or `Exhaustive`.
    pub certified_fraction: f64,
    /// `||f~|| / sqrt(|X|)` from `TreeTN::log_norm` per accepted patch,
    /// combined in canonical path order; `None` when some patch's
    /// `log_norm` is not finite (`||f~_P||` above about `1.34e154`). Not
    /// covered by the bitwise determinism claim: the canonicalization it
    /// relies on is not audited for reproducibility. It feeds the report
    /// only, never a decision.
    pub approximation_rms: Option<f64>,
}

impl L2ErrorReport {
    /// The global error value in L2 units, `sqrt(|X|) * global.rms_value()`,
    /// or `None` on overflow. Its meaning (bound, estimate, or statistic)
    /// follows the [`GlobalL2Error`] variant.
    ///
    /// # Examples
    ///
    /// See [`L2ErrorReport`].
    pub fn error_norm(&self) -> Option<f64> {
        finite(self.domain_points.sqrt() * self.global.rms_value())
    }

    /// `||f~||`, or `None` when `approximation_rms` is `None` or the product
    /// overflows.
    ///
    /// # Examples
    ///
    /// See [`L2ErrorReport`].
    pub fn approximation_norm(&self) -> Option<f64> {
        self.approximation_rms
            .and_then(|rms| finite(self.domain_points.sqrt() * rms))
    }
}

/// The norm, reference, and allowance of a run, by norm.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, L2ReferenceSource, NormReport, PatchedInterpolationOptions,
/// };
/// use tensor4all_partitionedtreetn::ErrorTolerance;
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // The exact root gives S = ||(3, 4)|| = 5, so delta = rtol * S.
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// let options = PatchedInterpolationOptions::new(2)
///     .with_tolerance(ErrorTolerance { rtol: 1e-3, atol: 0.0 });
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     BTreeMap::from([(0usize, vec![DynIndex::new_dyn(2)])]),
///     ColMajorArray::new(vec![], vec![1, 0])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(batch.data().iter().map(|&x| if x == 0 { 3.0 } else { 4.0 }).collect())
///     },
///     &options,
/// )?;
/// let norm = &result.report.norm;
/// assert!(matches!(norm, NormReport::L2 { source: L2ReferenceSource::ExactRoot, .. }));
/// assert!((norm.reference_norm().unwrap() - 5.0).abs() < 1e-12);
/// assert!((norm.delta().unwrap() - 5e-3).abs() < 1e-15);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NormReport {
    /// A run under [`ErrorNorm::L2`](crate::ErrorNorm::L2).
    #[non_exhaustive]
    L2 {
        /// `S / sqrt(|X|)`, the RMS value of the reference; `None` only for
        /// [`L2ReferenceSource::NotNeeded`].
        reference_rms: Option<f64>,
        /// Where the reference came from.
        source: L2ReferenceSource,
        /// The RMS allowance `max(atol / sqrt(|X|), rtol * reference_rms)`,
        /// also the engine's absolute tolerance.
        tau: f64,
        /// The measured global error.
        error: L2ErrorReport,
    },
    /// A run under [`ErrorNorm::SampledMax`](crate::ErrorNorm::SampledMax).
    #[non_exhaustive]
    SampledMax {
        /// The max-norm reference, a function value (`0` for an all-zero
        /// exact root without a given reference).
        max_reference: f64,
        /// Where it came from.
        source: MaxReferenceSource,
        /// The engine's absolute tolerance `max(atol, rtol * max_reference)`.
        engine_tolerance: f64,
    },
}

impl NormReport {
    /// The L2 error report of an L2 run.
    ///
    /// # Examples
    ///
    /// See [`L2ErrorReport`].
    pub fn l2_error(&self) -> Option<&L2ErrorReport> {
        match self {
            Self::L2 { error, .. } => Some(error),
            Self::SampledMax { .. } => None,
        }
    }

    /// The L2 allowance `delta = sqrt(|X|) * tau` of an L2 run, or `None`
    /// under `SampledMax` or on overflow.
    ///
    /// # Examples
    ///
    /// See [`NormReport`].
    pub fn delta(&self) -> Option<f64> {
        match self {
            Self::L2 { tau, error, .. } => finite(error.domain_points.sqrt() * tau),
            Self::SampledMax { .. } => None,
        }
    }

    /// The L2 reference norm `S = sqrt(|X|) * reference_rms` of an L2 run,
    /// or `None` under `SampledMax`, without a reference, or on overflow.
    ///
    /// # Examples
    ///
    /// See [`NormReport`].
    pub fn reference_norm(&self) -> Option<f64> {
        match self {
            Self::L2 {
                reference_rms,
                error,
                ..
            } => reference_rms.and_then(|rms| finite(error.domain_points.sqrt() * rms)),
            Self::SampledMax { .. } => None,
        }
    }
}

/// Summary of one [`patched_interpolate`](super::patched_interpolate) run.
///
/// `accepted` and `zero_patches` are sorted by patch path, the lexicographic
/// order of the (position in the derived site order, coordinate) pairs of the
/// fixed sites in split order; this canonical order does not depend on the
/// processing order. Accepted and zero patches are pairwise disjoint and
/// together cover the domain.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, NormReport, PatchedInterpolationOptions,
/// };
/// use tensor4all_partitionedtreetn::ErrorNorm;
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // f(a, b) is b^2 for a = 0 and 1 + b for a = 1: rank two. With a cap of
/// // two the domain splits once at `a`; both halves are evaluated exactly.
/// let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// topology.add_node(1usize)?;
/// topology.add_edge(&0, &1)?;
/// let node_sites = BTreeMap::from([(0usize, vec![a.clone()]), (1, vec![b.clone()])]);
/// let f = |p: &[usize]| if p[0] == 1 { 1.0 + p[1] as f64 } else { (p[1] * p[1]) as f64 };
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     node_sites,
///     ColMajorArray::new(vec![1, 2], vec![2, 1])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(batch.data().chunks(2).map(f).collect())
///     },
///     &PatchedInterpolationOptions::new(2)
///         .with_error_norm(ErrorNorm::sampled_max_with_reference(4.0)),
/// )?;
/// let report = &result.report;
/// assert!(matches!(report.norm, NormReport::SampledMax { max_reference, .. } if max_reference == 4.0));
/// assert_eq!(report.splits, 1);
/// assert_eq!(report.accepted.len(), 2);
/// assert!(report.zero_patches.is_empty());
/// assert_eq!(report.accepted[0].projector.get(&a), Some(0));
/// assert_eq!(report.accepted[1].projector.get(&a), Some(1));
/// // Six points in total, each evaluated once; SampledMax measures nothing.
/// assert_eq!(report.function_evaluations, 6);
/// assert_eq!(report.measurement_evaluations, 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchedInterpolationReport {
    /// The requested tolerance.
    pub tolerance: ErrorTolerance,
    /// The norm, its reference and allowance, and under L2 the measured
    /// global error.
    pub norm: NormReport,
    /// Records of the accepted patches, in canonical path order.
    pub accepted: Vec<PatchRecord>,
    /// Records of the zero patches, in canonical path order. They are not
    /// part of the partition, which treats an absent patch as zero.
    pub zero_patches: Vec<ZeroPatchRecord>,
    /// Number of patches that were split.
    pub splits: usize,
    /// Number of points passed to the evaluator.
    pub function_evaluations: usize,
    /// Number of requested points served from a patch cache instead of the
    /// evaluator.
    pub cache_hits: usize,
    /// The part of `function_evaluations` made by the driver's own
    /// measurements: zero screens, verifications, audits, and a Monte Carlo
    /// reference estimate.
    pub measurement_evaluations: usize,
    /// The part of `measurement_evaluations` made by audits.
    pub audit_evaluations: usize,
    /// Number of measurements of engine outcomes that exceeded the
    /// allowance.
    pub verification_failures: usize,
    /// Number of engine reruns after failed verifications.
    pub engine_retries: usize,
}

/// Result of [`patched_interpolate`](super::patched_interpolate).
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, PatchedInterpolationOptions,
/// };
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // f vanishes everywhere: the root is a zero patch and the partition is empty.
/// let site = DynIndex::new_dyn(3);
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     BTreeMap::from([(0usize, vec![site])]),
///     ColMajorArray::new(vec![], vec![1, 0])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(vec![0.0; batch.shape()[1]])
///     },
///     &PatchedInterpolationOptions::new(2),
/// )?;
/// assert!(result.partition.is_empty());
/// assert_eq!(result.report.zero_patches.len(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct PatchedInterpolationResult<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// The accepted patches. Zero patches are absent (an absent patch is
    /// zero), so an all-zero function gives an empty partition.
    pub partition: PartitionedTreeTN<V>,
    /// Summary of the run.
    pub report: PatchedInterpolationReport,
}

/// `Some(value)` when it is finite.
pub(super) fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}
