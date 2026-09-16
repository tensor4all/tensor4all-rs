//! Repartition orthogonal targets using a fixed global L2 error budget.
//!
//! Unlike the local discarded-weight policy in [`crate::PatchingOptions`],
//! this module pins a reference scale before reconstruction and checks actual
//! network residuals before accepting approximations. For
//! [`ReconstructionTarget::from_partition`] and
//! [`ReconstructionTarget::from_tensor_products`] that scale is the exact target
//! L2 norm; [`ReconstructionTarget::from_subset_operator`] instead propagates a
//! preimage scale times an operator amplification factor, which is an upper
//! bound for a general operator. L2 means the unweighted discrete Frobenius
//! (Hilbert--Schmidt for operators) norm. No induced operator norm is used.
//! Floating-point roundoff is not rigorously bounded.
//!
//! The gain-driven merge, split, and superposition choices are independently
//! implemented from the supplied algorithm note, "Fourier transform of a patched
//! QTT" (2026-08-26/27). This module performs reconstruction, not a Fourier
//! transform or a QFT butterfly schedule.

use std::{fmt::Debug, hash::Hash};

use crate::{
    DynIndex, PartitionedTreeTN, PartitionedTreeTNError, PatchSplitStrategy, Projector, Result,
    SubDomainTreeTN,
};

mod engine;
mod schedule;
mod target;

pub use engine::reconstruct;
pub use schedule::{
    schedule_merge_refine, CoordinateGroup, CoverageContract, MergeRefineOptions,
    MergeRefineReport, MergeRefineResult,
};
pub use target::ReconstructionTarget;

/// Global L2 tolerances, separate from rank and partition selection policy.
///
/// The absolute allowance is `max(atol, rtol * target.reference_scale())`.
/// Unlike [`crate::PatchingOptions::cutoff`], `rtol` is a norm tolerance, not
/// a local discarded singular-value weight. Both fields must be finite and
/// nonnegative. Zero in both fields disables approximate compression.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::reconstruction::ReconstructionTolerance;
/// let tolerance = ReconstructionTolerance::default();
/// assert_eq!(tolerance.rtol, 1e-6);
/// assert_eq!(tolerance.atol, 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ReconstructionTolerance {
    /// Relative L2 tolerance against the fixed reference scale: the allowance
    /// is `max(atol, rtol * reference_scale)`. Default: `1e-6`.
    pub rtol: f64,
    /// Absolute L2 tolerance, useful for small targets. Default: zero.
    pub atol: f64,
}

impl Default for ReconstructionTolerance {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 0.0,
        }
    }
}

/// Gain and search policy for [`reconstruct`], independent of its error budget.
///
/// The rank is a soft goal: an over-goal term is retained when no permitted
/// split improves its rank. Each split fixes one whole external site index;
/// this also supports multiple site indices per tree node. QTT dyadic users
/// should supply binary indices MSB first in `patch_order` and select
/// [`PatchSplitStrategy::Sequential`] to preserve contiguous dyadic intervals.
/// Index significance is independent of the indices' placement on the tree.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::{reconstruction::ReconstructionOptions, PatchSplitStrategy};
/// let options = ReconstructionOptions::default();
/// assert_eq!(options.target_bond_dim, Some(64));
/// assert_eq!(options.max_regions, 1024);
/// assert_eq!(options.split_strategy, PatchSplitStrategy::ExactParameterGain);
/// assert!(options.patch_order.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct ReconstructionOptions {
    /// Desired maximum bond dimension per term, default `Some(64)`.
    /// `None` disables splitting; `Some(0)` is invalid. Never overrides accuracy.
    pub target_bond_dim: Option<usize>,
    /// Permitted split indices, matched by full identity and dimension.
    /// Empty means all external indices, in deterministic identity order.
    /// `Sequential` probes only the first unprojected index with dimension > 1;
    /// no gain or insufficient region capacity stops splitting that region,
    /// without skipping to later indices. `ExactParameterGain` probes all
    /// permitted candidates, with this order breaking ties. This list is
    /// independent of any QFT-selected subset.
    pub patch_order: Vec<DynIndex>,
    /// Existing adaptive-patching candidate strategy; default `ExactParameterGain`.
    /// Both strategies accept a split only when the maximum child term rank
    /// decreases. Use `Sequential` with explicit MSB-first `patch_order` for
    /// contiguous QTT intervals; leave the default for unrestricted gain search.
    pub split_strategy: PatchSplitStrategy,
    /// Maximum live output regions, default 1024; must be positive.
    /// Reaching this search limit retains valid higher-rank terms.
    pub max_regions: usize,
}

impl Default for ReconstructionOptions {
    fn default() -> Self {
        Self {
            target_bond_dim: Some(64),
            patch_order: Vec::new(),
            split_strategy: PatchSplitStrategy::default(),
            max_regions: 1024,
        }
    }
}

/// How [`ReconstructionTarget::from_subset_operator`] derives the reference scale.
///
/// The operator's amplification factor multiplies the preimage's reference scale:
///
/// ```text
/// s = 1          if unitary is specified
/// s = ||A||_F    otherwise
/// reference_scale = s * preimage.reference_scale()
/// absolute_tolerance = max(atol, rtol * reference_scale)
/// ```
///
/// The multiplication composes with the preimage's own scale, which is already a
/// propagated bound when that preimage is itself operator-derived; no output norm
/// is measured at any step. Here `A` is the operator restricted to the selected
/// sites. Spectator identity
/// factors are neither materialized nor counted, because
/// `||A ⊗ I||_2 = ||A||_2 <= ||A||_F`. For an `N`-dimensional unitary the
/// Frobenius norm is `sqrt(N)` while the induced 2-norm is `1`; `unitary = true`
/// assumes the latter and does not assert a unit Frobenius norm. Successive
/// applications multiply these factors instead of recomputing an output norm.
///
/// With `unitary = false` the resulting scale is a norm-based upper bound, not
/// the measured output norm, so `rtol` is relative to that scale rather than to
/// `||A x||_2`. The two coincide for a unitary acting on an exactly known input
/// scale.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::reconstruction::SubsetOperatorOptions;
/// let options = SubsetOperatorOptions::default();
/// assert!(!options.unitary);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SubsetOperatorOptions {
    /// Caller guarantee that the operator preserves the L2 norm, so the
    /// amplification factor is exactly `1`. Default: `false`, which scales by
    /// the operator's Frobenius norm.
    pub unitary: bool,
}

/// Numerical error accounting and final storage diagnostics from reconstruction.
///
/// `error_bound` is a triangle-inequality bound assembled from measured local
/// residual norms, combined by the Euclidean norm across disjoint regions.
/// It is an a posteriori numerical bound, not an interval-arithmetic certificate:
/// backend floating-point errors in addition, projection, and norms remain.
/// The reference scale never changes during reconstruction.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::{PartitionedTreeTN, reconstruction::*};
/// let target = ReconstructionTarget::from_partition(&PartitionedTreeTN::<usize>::new())?;
/// let output = reconstruct(&target, &0, ReconstructionTolerance::default(), &Default::default())?;
/// assert_eq!(output.report().reference_scale, 0.0);
/// assert_eq!(output.report().error_bound, 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ReconstructionReport {
    /// Pinned reference scale, fixed before any reconstruction. It is the exact
    /// target L2 norm for partition- and product-derived targets, and a
    /// norm-based upper bound for subset-operator targets.
    pub reference_scale: f64,
    /// Fixed `max(atol, rtol * reference_scale)` allowance.
    pub absolute_tolerance: f64,
    /// Accumulated measured-residual bound, at most `absolute_tolerance`.
    pub error_bound: f64,
    /// Number of retained disjoint regions (zero regions denote zero).
    pub region_count: usize,
    /// Number of network terms across all retained regions.
    pub term_count: usize,
    /// Largest retained bond dimension, zero for an empty result.
    pub max_bond_dim: usize,
    /// Number of accepted region splits, excluding discarded probes.
    pub split_count: usize,
    /// Number of accepted pairwise merges in the final regions.
    pub merge_count: usize,
}

#[derive(Debug, Clone)]
struct Region<V: Clone + Hash + Eq + Send + Sync + Debug> {
    projector: Projector,
    terms: Vec<SubDomainTreeTN<V>>,
}

/// Disjoint regions with one or more eagerly masked TreeTN terms per region.
///
/// Terms within a region may overlap and need not be orthogonal. Their norm
/// squares must not simply be added. [`PartitionedTreeTN`] is the special case
/// with one term per region; [`Self::into_partition`] checks this condition and
/// never implicitly sums terms. The report accompanies the represented value.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::{PartitionedTreeTN, reconstruction::*};
/// let target = ReconstructionTarget::from_partition(&PartitionedTreeTN::<usize>::new())?;
/// let output = reconstruct(&target, &0, Default::default(), &Default::default())?;
/// assert_eq!(output.regions().count(), 0);
/// assert_eq!(output.into_partition()?.norm()?, 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ReconstructedTreeTN<V = usize>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    regions: Vec<Region<V>>,
    report: ReconstructionReport,
}

impl<V: Clone + Hash + Eq + Ord + Send + Sync + Debug> ReconstructedTreeTN<V> {
    /// Borrow each region's projector and its eagerly masked superposition terms.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, reconstruction::*};
    /// let target = ReconstructionTarget::from_partition(&PartitionedTreeTN::<usize>::new())?;
    /// let output = reconstruct(&target, &0, Default::default(), &Default::default())?;
    /// assert_eq!(output.regions().count(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn regions(&self) -> impl Iterator<Item = (&Projector, &[SubDomainTreeTN<V>])> {
        self.regions
            .iter()
            .map(|region| (&region.projector, region.terms.as_slice()))
    }

    /// Borrow the fixed reference scale, measured error accounting, and final counts.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, reconstruction::*};
    /// let target = ReconstructionTarget::from_partition(&PartitionedTreeTN::<usize>::new())?;
    /// let output = reconstruct(&target, &0, Default::default(), &Default::default())?;
    /// assert_eq!(output.report().absolute_tolerance, 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn report(&self) -> &ReconstructionReport {
        &self.report
    }

    /// Consume a result with at most one term per region as a strict partition.
    ///
    /// # Errors
    /// Returns [`PartitionedTreeTNError::InvalidOptions`] when a region retains a superposition; inspect
    /// [`Self::regions`] to use its terms individually. Propagates partition
    /// topology, site identity, dtype, or projector validation errors.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, reconstruction::*};
    /// let target = ReconstructionTarget::from_partition(&PartitionedTreeTN::<usize>::new())?;
    /// let output = reconstruct(&target, &0, Default::default(), &Default::default())?;
    /// assert_eq!(output.into_partition()?.norm()?, 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_partition(self) -> Result<PartitionedTreeTN<V>> {
        single_term_partition(
            self.regions
                .into_iter()
                .map(|region| (region.projector, region.terms)),
        )
    }
}

/// Validate the global L2 tolerance pair shared by every reconstruction entry point.
fn validate_tolerance(tolerance: ReconstructionTolerance) -> Result<()> {
    if !tolerance.rtol.is_finite()
        || tolerance.rtol < 0.0
        || !tolerance.atol.is_finite()
        || tolerance.atol < 0.0
    {
        return Err(invalid("rtol and atol must be finite and nonnegative"));
    }
    Ok(())
}

/// The global L2 allowance `max(atol, rtol * scale)`.
fn absolute_allowance(tolerance: ReconstructionTolerance, scale: f64) -> Result<f64> {
    finite(tolerance.atol.max(finite(tolerance.rtol * scale)?))
}

/// Convert regions with at most one term each into a strict partition.
fn single_term_partition<V>(
    regions: impl IntoIterator<Item = (Projector, Vec<SubDomainTreeTN<V>>)>,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut terms = Vec::new();
    for (_, region_terms) in regions {
        if region_terms.len() > 1 {
            return Err(invalid(
                "a region retains multiple terms; use regions() to access the superposition",
            ));
        }
        terms.extend(region_terms);
    }
    PartitionedTreeTN::from_subdomains(terms)
}

fn invalid(reason: &'static str) -> PartitionedTreeTNError {
    PartitionedTreeTNError::InvalidOptions {
        operation: "reconstruction",
        reason,
    }
}

fn finite(value: f64) -> Result<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(PartitionedTreeTNError::NonFiniteAdaptiveValue)
    }
}
