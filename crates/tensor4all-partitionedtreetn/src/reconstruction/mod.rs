//! Repartition orthogonal targets using a fixed global L2 error budget.
//!
//! Unlike the local discarded-weight policy in [`crate::PatchingOptions`],
//! this module pins the target norm before reconstruction and checks actual
//! network residuals before accepting approximations. L2 means the unweighted
//! discrete Frobenius (Hilbert--Schmidt for operators) norm. No induced operator
//! norm is used. Floating-point roundoff is not rigorously bounded.
//!
//! The gain-driven merge, split, and superposition choices are independently
//! implemented from the supplied algorithm note, "Fourier transform of a patched
//! QTT" (2026-08-26/27). This module performs reconstruction, not a Fourier
//! transform or a QFT butterfly schedule.

use std::{fmt::Debug, hash::Hash};

use crate::{
    DynIndex, PartitionedTreeTN, PartitionedTreeTNError, Projector, Result, SubDomainTreeTN,
};

mod engine;
mod target;

pub use engine::reconstruct;
pub use target::ReconstructionTarget;

/// Global L2 tolerances, separate from rank and partition selection policy.
///
/// The absolute allowance is `max(atol, rtol * target.reference_norm())`.
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
    /// Relative L2 tolerance against the original target. Default: `1e-6`.
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
/// should supply the appropriate binary indices in `split_indices`.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::reconstruction::ReconstructionOptions;
/// let options = ReconstructionOptions::default();
/// assert_eq!(options.target_bond_dim, Some(64));
/// assert_eq!(options.max_regions, 1024);
/// ```
#[derive(Debug, Clone)]
pub struct ReconstructionOptions {
    /// Desired maximum bond dimension per term, default `Some(64)`.
    /// `None` disables splitting; `Some(0)` is invalid. Never overrides accuracy.
    pub target_bond_dim: Option<usize>,
    /// Permitted split indices, matched by full identity and dimension.
    /// Empty means all external indices, in deterministic identity order.
    /// Candidates are ranked by resulting logical parameter count, with input
    /// order breaking ties. This list is independent of any QFT-selected subset.
    pub split_indices: Vec<DynIndex>,
    /// Maximum live output regions, default 1024; must be positive.
    /// Reaching this search limit retains valid higher-rank terms.
    pub max_regions: usize,
}

impl Default for ReconstructionOptions {
    fn default() -> Self {
        Self {
            target_bond_dim: Some(64),
            split_indices: Vec::new(),
            max_regions: 1024,
        }
    }
}

/// Numerical error accounting and final storage diagnostics from reconstruction.
///
/// `error_bound` is a triangle-inequality bound assembled from measured local
/// residual norms, combined by the Euclidean norm across disjoint regions.
/// It is an a posteriori numerical bound, not an interval-arithmetic certificate:
/// backend floating-point errors in addition, projection, and norms remain.
/// The original reference norm never changes during reconstruction.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::{PartitionedTreeTN, reconstruction::*};
/// let target = ReconstructionTarget::from_partition(&PartitionedTreeTN::<usize>::new())?;
/// let output = reconstruct(&target, &0, ReconstructionTolerance::default(), &Default::default())?;
/// assert_eq!(output.report().reference_norm, 0.0);
/// assert_eq!(output.report().error_bound, 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ReconstructionReport {
    /// Original target L2 norm, computed before any reconstruction.
    pub reference_norm: f64,
    /// Fixed `max(atol, rtol * reference_norm)` allowance.
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

    /// Borrow the fixed reference norm, measured error accounting, and final counts.
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
        if self.regions.iter().any(|region| region.terms.len() > 1) {
            return Err(invalid(
                "a region retains multiple terms; use regions() to access the superposition",
            ));
        }
        PartitionedTreeTN::from_subdomains(self.regions.into_iter().flat_map(|r| r.terms).collect())
    }
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
