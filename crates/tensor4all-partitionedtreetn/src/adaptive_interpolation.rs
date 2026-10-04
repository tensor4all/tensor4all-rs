//! Adaptive patched interpolation of a function on a tree.
//!
//! [`patched_interpolate`] runs a tree interpolation engine (any
//! [`TreeInterpolator`]) on the whole domain of a function. Wherever a patch
//! cannot be accepted, it fixes the next site of a given order and retries on
//! every child region, unless an optional minimum patch size stops the
//! splitting. The result is a [`PartitionedTreeTN`] with disjoint, eagerly
//! masked patches, together with a [`PatchedInterpolationReport`].
//!
//! # Derivation notice
//!
//! The patch queue, the accept-or-split flow, and pivot recycling are derived
//! from `adaptiveinterpolate`, `createpatch`, and `_globalpivots` in
//! [TCIAlgorithms.jl](https://github.com/tensor4all/TCIAlgorithms.jl) at
//! commit e501032278c9dd41b46c5851d8238169c8d178c5 (MIT license; Copyright
//! 2023 Ritter.Marc and contributors), through the chain driver of the
//! deprecated `tensor4all-partitionedtt` crate. See
//! `LICENSE-TCIALGORITHMS-MIT` in this crate. The tree generalization, the
//! evaluation cache, the sampled-zero policy, the re-embedding of fixed
//! sites, and the L2 error contract are original to this crate.
//!
//! # Error contract
//!
//! The accuracy requirement is [`PatchedInterpolationOptions::error_norm`]
//! with [`PatchedInterpolationOptions::tolerance`]:
//!
//! - [`ErrorNorm::L2`] (the default): the unweighted discrete L2 error over
//!   the whole domain `X`, `E^2 = sum over x of |f(x) - f~(x)|^2`, against
//!   the allowance `delta = max(atol, rtol * S)` for a reference L2 norm `S`
//!   of `f` ([`L2Reference`]). The driver keeps it in root-mean-square units:
//!   `tau = delta / sqrt(|X|)` is pinned once and every accepted or zero
//!   patch `P` must satisfy `rms_P(f - f~_P) <= tau`, which sums to
//!   `E <= delta` over the disjoint patches. Zero patches are charged like
//!   any other patch. The engine receives the absolute tolerance `tau`. The
//!   only exception is a patch retained by
//!   [`PatchedInterpolationOptions::min_patch_bits`] with
//!   [`PatchStatus::ToleranceNotMet`]: the run then reports
//!   [`GlobalL2Error::ToleranceNotMet`], and `E` can exceed `delta` without
//!   limit.
//! - [`ErrorNorm::SampledMax`]: the M2 criterion, the engine's sampled error
//!   estimate against `max(atol, rtol * max_reference)`, with `max_reference`
//!   a function value. No measurement runs; it is neither a certified bound
//!   nor a measured error, and makes no L2 claim.
//! - [`ErrorNorm::MaxAbs`] and [`ErrorNorm::WeightedL2`] are placeholders that
//!   fail with [`PatchedInterpolationError::UnsupportedNorm`] before any
//!   evaluation.
//!
//! **Certified, measured, and estimated errors.** Under L2 the driver
//! measures every patch error from values of `f` at points chosen
//! independently of the approximation, by one of three methods
//! ([`MeasurementMethod`]): `Exact` (the patch was built from all its
//! values) and `Exhaustive` (the residual was evaluated at every point: exact
//! up to rounding) are certified; `Sampled` (fresh uniform points) is
//! measured, not certified, and never a guarantee. A sampled acceptance
//! measurement is only a decision statistic: it is conditioned on the
//! acceptance it decided, and a residual concentrated on unsampled points is
//! missed. With [`VerificationOptions::audit`], an independent audit sample
//! drawn after the decision gives an unbiased estimate of the mean square
//! residual (its square root, the reported RMS, is not unbiased) with a
//! standard error; it is never a bound. No finite sample bounds the L2 error
//! of a black-box function.
//!
//! **Known limitation: localized features.** The acceptance sample and the
//! audit are both uniform, so both can miss a localized feature that enters
//! a patch only through a corner or an edge; the engine may also never
//! sample it there and converge at a low rank. The audit's standard error is
//! computed from the same points and says nothing about a residual
//! concentrated on a small set it did not draw: the audited estimate can then
//! be orders of magnitude below the true error with a small standard error.
//! More [`VerificationOptions::samples`] make such a miss less likely but do
//! not exclude it. The opt-in
//! [`PatchedInterpolationOptions::cache_candidates`] starts child patches on
//! values the parent already evaluated in them, so it can prevent a miss
//! where the parent sampled the feature but the child's engine would not; it
//! does not help when the parent never sampled the feature, nor when the
//! acceptance sample and the audit miss a feature the engine saw but did not
//! resolve. Only a `Certified` result (every contribution exact or
//! exhaustive) is a guarantee. The limitation is recorded, with a
//! reproduction and the measured effect of the option, under "Known
//! limitation: corner-localized misses" in
//! `docs/design/tree-patching-error-contract.md`.
//!
//! The report's [`GlobalL2Error`] says what the run can claim:
//!
//! - `ToleranceNotMet` when some patch was retained without meeting its
//!   allowance; it takes precedence, is never certified, and its `basis`
//!   classifies the measurements as below (an exact or exhaustive basis
//!   still bounds `E`, but the tolerance was not met).
//! - `Certified` when every contribution is exact or exhaustive:
//!   `E <= delta (1 + GLOBAL_ROUNDING_MARGIN) + MEASUREMENT_ROUNDING_FACTOR *
//!   eps * ||f~||`, an absolute bound with respect to the allowance used, up
//!   to a first-order model of the evaluation rounding (not a proven bound).
//!   `rounding_limited` says when that rounding term is at least `tau`.
//!   `relative_error_bound` bounds `E / ||f||` from the computable side,
//!   `E / ||f|| <= E / (||f~|| - E)`, with conservative margins; it is `None`
//!   when the denominator is not positive.
//! - `Audited` when every sampled contribution has an audit: an estimate of
//!   `E` with its standard error, and a plug-in estimate of the relative
//!   bound. Neither is a guarantee: both can be far too small when a
//!   localized feature was missed (see above).
//! - `AcceptanceOnly` otherwise: the combined acceptance statistics, neither a
//!   bound nor an estimate; no relative statement exists.
//!
//! `approximation_rms` (and with it the rounding term, the flag, and the
//! relative fields) is `None` when the norm of some patch exceeds about
//! `1.34e154`, where `TreeTN::log_norm` overflows.
//!
//! # Algorithm
//!
//! 1. The inputs are validated before any evaluation: the norm first, then
//!    the topology and sites (those of
//!    [`validate_layout`](tensor4all_treetn::interpolation::validate_layout)),
//!    `patch_order`, the tolerance, the reference, the other options, the
//!    verification options, the initial pivots, and under L2 the domain size
//!    and whether a reference is required.
//! 2. Patches are processed in FIFO order, starting from the whole domain.
//!    The reference is pinned at the root and never changes.
//! 3. Each patch owns an evaluation cache of its points; a split hands every
//!    cached value (measured values included) to the child that contains it,
//!    so no point is evaluated twice.
//! 4. A patch with at most one active (unfixed) site is evaluated exactly and
//!    needs no engine or measurement. Otherwise the candidate pivots of the
//!    patch are sampled: compatible user pivots, recycled pivots, the
//!    parent's worst points, with
//!    [`PatchedInterpolationOptions::cache_candidates`] the largest cached
//!    values of the patch, then random points up to `n_initial_pivots`. If
//!    every sample is exactly zero, the zero
//!    approximation is measured under L2 (and accepted as a zero patch if it
//!    fits, otherwise its largest measured points join the candidates) or
//!    accepted directly under `SampledMax`. Zero patches are reported in
//!    [`PatchedInterpolationReport::zero_patches`] and omitted from the
//!    partition.
//! 5. The engine runs on the active sites. A
//!    [`InterpolationTermination::Converged`] outcome strictly below the bond
//!    cap is re-embedded (every fixed site re-attached by a one-hot factor)
//!    and, under L2, measured on the stored network: exhaustively when the
//!    patch has at most `max(max_exhaustive_points, samples)` points,
//!    otherwise on `samples` fresh uniform points. It is accepted when
//!    `rms <= tau`. With [`CappedPatches::AcceptUpTo`], a
//!    [`InterpolationTermination::BondCapReached`] outcome of a patch with at
//!    most that many active sites (generalized bits) is checked against the
//!    cap and judged the same way (by the engine estimate under
//!    `SampledMax`).
//! 6. A failed verification of a converged run reruns the engine (up to
//!    [`VerificationOptions::retries`] times) with the worst measured points
//!    and the outcome's pivots added to the initial pivots; a failed capped
//!    run is not rerun. Then the patch splits at the next unfixed site of
//!    [`PatchedInterpolationOptions::patch_order`], passing the worst points
//!    to the children. Any other verdict splits the patch directly.
//! 7. With [`PatchedInterpolationOptions::min_patch_bits`], a split whose
//!    children would have fewer active sites than the minimum is not made:
//!    the patch is retained with its last engine run. A run judged already
//!    keeps its verdict; one that was not (it did not converge and was not
//!    capped-eligible) is measured once on the stream of its attempt (judged
//!    by its estimate under `SampledMax`). The record says
//!    [`PatchStatus::WithinTolerance`] or [`PatchStatus::ToleranceNotMet`].
//!    An exhausted `patch_order` keeps its errors.
//!
//! # Randomness and determinism
//!
//! Every patch derives its sub-seeds from [`PatchedInterpolationOptions::seed`]
//! and its path, the (position in the derived site order, coordinate) pairs
//! of its fixed sites: one for its random candidate pivots, one per engine
//! run, and one stream each for the zero screen, every verification, the
//! audit, and (at the root) the Monte Carlo reference. The generator is
//! SplitMix64, and a coordinate in `0..d` is drawn with Lemire's unbiased
//! multiply-shift method with rejection. Unlike other randomized algorithms
//! of this workspace, the driver offers no API taking a caller-owned
//! `&mut R`: one shared stream would make the randomness of a patch depend on
//! the processing order.
//!
//! For a fixed seed, a deterministic evaluator, and a deterministic engine,
//! the report and every stored node tensor (values and positional axis order)
//! are identical across runs on fresh threads within one process, provided
//! every measured network value is reproducible; this is tested for `f64` on
//! trees where every node carries exactly one site, where the measurement
//! takes the cached evaluator's raw kernels. Each measurement uses a fresh
//! `TreeTNCachedEvaluator` centered at the smallest node name, the default
//! hint, and fixed-size chunks. On other trees (a site-free node, a node with
//! several sites, or `f32`/`c32` data) the evaluator's generic path can
//! differ at rounding level between threads and processes, so an L2
//! acceptance near `tau` may then differ between runs; this is an open
//! evaluator issue
//! ([issue #795](https://github.com/tensor4all/tensor4all-rs/issues/795)).
//! The intended scope is bitwise identical results on the same machine and
//! build across threads, thread counts, and processes, with no cross-machine
//! promise. It is not reached yet: generic-path trees need that issue fixed,
//! and reproducibility across processes is not claimed until a two-process
//! test passes, which does not exist yet. The bitwise claim never
//! covers `approximation_rms` and the fields derived from it. The stored
//! patches are `TreeTN`s, so what is derived from them may still differ
//! across runs, for a single patch as for the whole partition, on any topology
//! ([issue #791](https://github.com/tensor4all/tensor4all-rs/issues/791)):
//! materializing (`to_dense`, `contract_to_tensor`,
//! [`PartitionedTreeTN::to_treetn`]) in axis order and at rounding level, and
//! the iteration order of `external_indices`, `site_space`, and `neighbors`.
//!
//! # Examples
//!
//! Interpolate `f(x) = 1 / (1 + x)` on eight points, `x = b0 + 2 b1 + 4 b2`,
//! with one binary site per node of a three-node chain, under the default
//! L2 norm with a known reference norm. A bond cap of two only
//! accepts rank-one patches, so the domain is split twice; every patch ends
//! up exact, so the global error is certified.
//!
//! ```
//! use std::collections::BTreeMap;
//! use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
//! use tensor4all_partitionedtreetn::adaptive_interpolation::{
//!     patched_interpolate, GlobalL2Error, PatchedInterpolationOptions,
//! };
//! use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance, L2Reference};
//! use tensor4all_treetci::TreeTciInterpolator;
//! use tensor4all_treetn::NodeNameNetwork;
//!
//! let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
//! let mut topology = NodeNameNetwork::new();
//! for node in 0..3usize {
//!     topology.add_node(node)?;
//! }
//! topology.add_edge(&0, &1)?;
//! topology.add_edge(&1, &2)?;
//! let node_sites: BTreeMap<usize, Vec<DynIndex>> =
//!     (0..3).map(|node| (node, vec![sites[node].clone()])).collect();
//!
//! let values: Vec<f64> = (0..8).map(|x| 1.0 / (1.0 + x as f64)).collect();
//! let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
//! let f = |point: &[usize]| 1.0 / (1.0 + (point[0] + 2 * point[1] + 4 * point[2]) as f64);
//! let evaluate = |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
//!     Ok(batch.data().chunks(3).map(f).collect())
//! };
//! let options = PatchedInterpolationOptions::new(2)
//!     .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
//!     .with_tolerance(ErrorTolerance { rtol: 1e-12, atol: 0.0 });
//! let result = patched_interpolate(
//!     &TreeTciInterpolator::default(),
//!     topology,
//!     node_sites,
//!     ColMajorArray::new(vec![0, 0, 0], vec![3, 1])?,
//!     evaluate,
//!     &options,
//! )?;
//!
//! // The splits fix sites 0 and 1; every patch then has one active site.
//! assert_eq!(result.report.splits, 3);
//! assert_eq!(result.partition.len(), 4);
//! assert_eq!(result.report.function_evaluations, 8);
//! let error = result.report.norm.l2_error().unwrap();
//! assert!(matches!(error.global, GlobalL2Error::Certified { rms_error, .. } if rms_error == 0.0));
//!
//! // Compare with the dense function once: materialize, subtract, norm.
//! let reference = IdxTensor::from_dense(sites.clone(), values)?;
//! let dense = result.partition.to_treetn()?.contract_to_tensor()?;
//! assert!(dense.sub(&reference)?.norm()? <= 1e-12 * norm);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod acceptance;
mod cache;
mod embed;
mod layout;
mod options;
mod report;
mod sampling;
#[cfg(test)]
mod tests;
mod verify;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use tensor4all_core::{
    ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, TensorElement,
};
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationOutcome, InterpolationProblem, InterpolationTermination,
    TreeInterpolator,
};
use tensor4all_treetn::{NodeNameNetwork, TreeTN};

use self::sampling::point_list_capacity;

use crate::error::PartitionedTreeTNError;
use crate::{ErrorNorm, L2Reference, PartitionedTreeTN, Projector, SubDomainTreeTN};

use acceptance::SizePolicy;
use cache::{Counters, PatchCache, PatchSampler};
use layout::SiteLayout;
use sampling::{patch_candidates, patch_seeds, PatchDomain, PatchSeeds};
use verify::{Contribution, MeasureError, MeasureTarget, Measured, PointPlan};

pub use options::{CappedPatches, PatchedInterpolationOptions, VerificationOptions};
pub use report::{
    GlobalL2Error, L2ErrorReport, L2Measurement, L2ReferenceSource, MaxReferenceSource,
    MeasurementMethod, NormReport, PatchRecord, PatchStatus, PatchedInterpolationReport,
    PatchedInterpolationResult, ToleranceNotMetBasis, ZeroPatchRecord, GLOBAL_ROUNDING_MARGIN,
    MEASUREMENT_ROUNDING_FACTOR,
};

/// Error returned by [`patched_interpolate`].
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, PatchedInterpolationError, PatchedInterpolationOptions,
///     PatchedInterpolationResult,
/// };
/// use tensor4all_partitionedtreetn::ErrorNorm;
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// let run = |options: &PatchedInterpolationOptions|
///  -> Result<PatchedInterpolationResult<usize>, PatchedInterpolationError> {
///     let mut topology = NodeNameNetwork::new();
///     topology.add_node(0usize).unwrap();
///     patched_interpolate(
///         &TreeTciInterpolator::default(),
///         topology,
///         BTreeMap::from([(0usize, vec![DynIndex::new_dyn(2)])]),
///         ColMajorArray::new(vec![], vec![1, 0]).unwrap(),
///         |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///             Ok(vec![1.0; batch.shape()[1]])
///         },
///         options,
///     )
/// };
/// // A cap of one could only accept patches nonzero on a single node.
/// let error = run(&PatchedInterpolationOptions::new(1)).unwrap_err();
/// assert!(matches!(error, PatchedInterpolationError::InvalidInput { .. }));
/// assert!(error.to_string().contains("max_bond_dim"));
/// // A placeholder norm fails before any other check.
/// let options = PatchedInterpolationOptions::new(1).with_error_norm(ErrorNorm::MaxAbs);
/// assert!(matches!(
///     run(&options).unwrap_err(),
///     PatchedInterpolationError::UnsupportedNorm { norm: ErrorNorm::MaxAbs }
/// ));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PatchedInterpolationError {
    /// The inputs or options are invalid, or a reference cannot be pinned.
    /// Reported before any evaluation, except an unpinnable reference (a
    /// zero `SampledMax` root sample, or a zero Monte Carlo estimate with
    /// `atol = 0`) and a measurement point list that the allocator cannot
    /// provide (`verification.samples` or `verification.max_exhaustive_points`
    /// too large for the available memory), which surface when the
    /// measurement is built.
    #[error("invalid patched interpolation input: {message}")]
    InvalidInput {
        /// The violated condition and, where possible, the remedy.
        message: String,
    },
    /// The selected norm is a placeholder without an implementation.
    /// Reported before any evaluation and before every other check; the
    /// driver never falls back to another norm.
    #[error(
        "the error norm {norm:?} is not implemented; use ErrorNorm::L2 (the default) or \
         ErrorNorm::SampledMax"
    )]
    UnsupportedNorm {
        /// The requested norm.
        norm: ErrorNorm,
    },
    /// Sampling, interpolating, or measuring one patch failed: the evaluator
    /// failed or returned a wrong number of values, a non-finite value, or a
    /// value whose magnitude overflows ([`InterpolationError::Evaluator`]); or
    /// the engine failed, returned an outcome that does not match the problem,
    /// reported `Converged` with a rank that reaches the bond cap, returned a
    /// network above the cap for a capped-eligible or retained run, or
    /// returned a network with a non-finite value at a measured point
    /// ([`InterpolationError::Engine`]).
    #[error("interpolation of the patch {projector:?} failed: {source}")]
    Interpolation {
        /// Projector of the failing patch.
        projector: Projector,
        /// The underlying interpolation error.
        #[source]
        source: InterpolationError,
    },
    /// Building a patch or the partition failed.
    #[error("building the patched partition failed: {source}")]
    Partition {
        /// The underlying partition error.
        #[source]
        source: PartitionedTreeTNError,
    },
    /// A patch did not converge, its last engine run was not measured, and
    /// every site of `patch_order` is already fixed in it.
    #[error(
        "the patch {projector:?} did not converge and every site of patch_order is fixed; \
         list more sites in patch_order or raise max_bond_dim"
    )]
    NoSplitIndexLeft {
        /// Projector of the patch that could not be split.
        projector: Projector,
    },
    /// The last engine run of a patch was measured (it converged below the
    /// cap, or reached the cap within [`CappedPatches::AcceptUpTo`]), its L2
    /// error exceeded its allowance after the retries, and every site of
    /// `patch_order` is already fixed in it. The minimum patch size never
    /// turns this into an acceptance.
    #[error(
        "the measured L2 error of the patch {projector:?} (rms {}) exceeds its allowance and \
         every site of patch_order is fixed; list more sites in patch_order, raise rtol or atol, \
         or check the reference norm",
        measurement.rms
    )]
    VerificationFailed {
        /// Projector of the patch.
        projector: Projector,
        /// Its last failed measurement.
        measurement: L2Measurement,
    },
    /// A resource limit of the options was exceeded.
    #[error(
        "patched interpolation exceeded its {resource} limit of {limit}; raise the limit or \
         max_bond_dim"
    )]
    ResourceLimit {
        /// Name of the exceeded option.
        resource: &'static str,
        /// The limit.
        limit: usize,
    },
}

impl From<PartitionedTreeTNError> for PatchedInterpolationError {
    fn from(source: PartitionedTreeTNError) -> Self {
        Self::Partition { source }
    }
}

/// Adaptively interpolate a function on a tree into disjoint patches.
///
/// # Arguments
///
/// * `engine` - Tree interpolation engine run on every patch with at least
///   two active sites, for example `tensor4all_treetci::TreeTciInterpolator`.
/// * `topology` - Tree topology with named nodes. Its node set must equal the
///   keys of `node_sites`.
/// * `node_sites` - Site indices of every node, possibly none for a node. The
///   derived site order ([`InterpolationProblem::derive_site_order`]: nodes
///   in ascending name order, each node's sites in the given order) lays out
///   `initial_pivots` and every evaluator batch.
/// * `initial_pivots` - Column-major `[n_sites, n_pivots]` array of
///   full-domain points in site order; zero columns are allowed. Each patch
///   starts from those compatible with it. Pivots where the function is large
///   make the zero screening reliable.
/// * `evaluate` - Batch evaluator. It receives a column-major
///   `[n_sites, n_points]` array of full-domain points in site order and
///   returns one finite value per point, with a finite magnitude.
/// * `options` - See [`PatchedInterpolationOptions`].
///
/// # Returns
///
/// The partition of accepted patches (eagerly masked, every site index
/// retained, one dtype `T`) and a [`PatchedInterpolationReport`].
///
/// # Errors
///
/// - [`PatchedInterpolationError::UnsupportedNorm`] first, before any other
///   check, for [`ErrorNorm::MaxAbs`] and [`ErrorNorm::WeightedL2`].
/// - [`PatchedInterpolationError::InvalidInput`] before any evaluation when
///   [`validate_layout`](tensor4all_treetn::interpolation::validate_layout)
///   rejects the topology or sites (with its message); a `patch_order` entry
///   is not a site of the problem (full identity and dimension) or is
///   repeated; `tolerance.rtol` or `tolerance.atol` is negative or not
///   finite; a given reference (`max_reference` or `L2Reference::Given`) is
///   not finite and positive; `max_bond_dim < 2`; `n_initial_pivots == 0`;
///   `max_patches == Some(0)`; `capped_patches` is
///   [`CappedPatches::AcceptUpTo`] with fewer bits than `min_patch_bits`;
///   `verification.samples < 2`; `initial_pivots`
///   is not a 2D array with one row per site and in-range coordinates;
///   `verification.samples` times the number of sites exceeds the capacity
///   of a point list (its byte length must fit a `Vec`); under L2, the
///   domain's point count is not finite in `f64`, or the reference is
///   [`L2Reference::Required`] while `rtol > 0` and the root has more than
///   one site. After the root sample: under `SampledMax` without a
///   `max_reference`, every candidate sample of a root that needs the engine
///   is exactly zero; under L2 with [`L2Reference::MonteCarlo`], the
///   estimate is zero and `atol = 0`. During a measurement, possibly after
///   evaluations: its point list cannot be reserved (the allocator refuses
///   `verification.samples` or, for an exhaustive measurement, up to
///   `max(verification.max_exhaustive_points, verification.samples)` points).
/// - [`PatchedInterpolationError::Interpolation`] when the evaluator fails,
///   returns a wrong number of values, or returns a value with a non-finite
///   component or with finite components whose magnitude overflows
///   ([`InterpolationError::Evaluator`]); when the engine fails (including
///   [`InterpolationError::AllSamplesZero`] after screening); or when an
///   engine outcome does not match the problem, reports `Converged` at a
///   bond dimension not strictly below the cap, exceeds the cap in a
///   capped-eligible or retained run, or evaluates to a non-finite value at
///   a measured point ([`InterpolationError::Engine`]).
/// - [`PatchedInterpolationError::NoSplitIndexLeft`] when a patch does not
///   converge, its last run was not measured, and every site of
///   `patch_order` is fixed.
/// - [`PatchedInterpolationError::VerificationFailed`] when the measured last
///   run of a patch (converged, or capped-eligible) fails its L2 measurement
///   after the retries and every site of `patch_order` is fixed.
/// - [`PatchedInterpolationError::ResourceLimit`] when more than
///   `max_patches` patches would be processed.
/// - [`PatchedInterpolationError::Partition`] when building a patch network
///   or the partition fails.
///
/// # Examples
///
/// A branched tree whose junction `"c"` has degree three and carries no
/// site. `f` vanishes where the leaf site `x` is zero; that half is a zero
/// patch, measured exhaustively. Every patch here has at most 1024 points,
/// so the run is certified.
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate, GlobalL2Error, MeasurementMethod, PatchedInterpolationOptions,
/// };
/// use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance, L2Reference};
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// let (x, y, z) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3), DynIndex::new_dyn(3));
/// let mut topology = NodeNameNetwork::new();
/// for node in ["c", "x", "y", "z"] {
///     topology.add_node(node.to_string())?;
/// }
/// for leaf in ["x", "y", "z"] {
///     topology.add_edge(&"c".to_string(), &leaf.to_string())?;
/// }
/// let node_sites = BTreeMap::from([
///     ("c".to_string(), vec![]),
///     ("x".to_string(), vec![x.clone()]),
///     ("y".to_string(), vec![y.clone()]),
///     ("z".to_string(), vec![z.clone()]),
/// ]);
/// // Site order [x, y, z]; f = x (1 + y + z)^2 has rank three across the y edge.
/// let f = |p: &[usize]| (p[0] * (1 + p[1] + p[2]).pow(2)) as f64;
/// let values: Vec<f64> = (0..18).map(|k| f(&[k % 2, (k / 2) % 3, k / 6])).collect();
/// let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
/// let options = PatchedInterpolationOptions::new(3)
///     .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
///     .with_tolerance(ErrorTolerance { rtol: 1e-10, atol: 0.0 })
///     .with_patch_order(vec![x.clone(), y.clone()]);
/// let result = patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     node_sites,
///     ColMajorArray::new(vec![1, 2, 2], vec![3, 1])?,
///     |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///         Ok(batch.data().chunks(3).map(f).collect())
///     },
///     &options,
/// )?;
/// let zero = &result.report.zero_patches[0];
/// assert_eq!(zero.projector.get(&x), Some(0));
/// assert_eq!(zero.acceptance.as_ref().unwrap().method, MeasurementMethod::Exhaustive);
/// let error = result.report.norm.l2_error().unwrap();
/// assert!(matches!(error.global, GlobalL2Error::Certified { .. }));
/// let delta = result.report.norm.delta().unwrap();
///
/// let reference = IdxTensor::from_dense(vec![x, y, z], values)?;
/// let dense = result.partition.to_treetn()?.contract_to_tensor()?;
/// assert!(dense.sub(&reference)?.norm()? <= delta + 1e-12 * norm);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn patched_interpolate<T, V, E, F>(
    engine: &E,
    topology: NodeNameNetwork<V>,
    node_sites: BTreeMap<V, Vec<DynIndex>>,
    initial_pivots: ColMajorArray<usize>,
    evaluate: F,
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<V>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T> + Sync,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>> + Send + Sync,
{
    if matches!(
        options.error_norm,
        ErrorNorm::MaxAbs | ErrorNorm::WeightedL2
    ) {
        return Err(PatchedInterpolationError::UnsupportedNorm {
            norm: options.error_norm,
        });
    }
    let layout = SiteLayout::validated(topology, node_sites, &initial_pivots, options)?;
    let driver = Driver {
        engine,
        evaluate: &evaluate,
        layout,
        initial_pivots,
        options,
        policy: SizePolicy::new(options),
        max_reference: Cell::new(None),
        l2: Cell::new(None),
        counters: Counters::default(),
        measurement_evaluations: Cell::new(0),
        audit_evaluations: Cell::new(0),
        verification_failures: Cell::new(0),
        engine_retries: Cell::new(0),
        scalar: PhantomData,
    };
    driver.pin_references_known_in_advance();
    driver.run()
}

fn invalid(message: impl Into<String>) -> PatchedInterpolationError {
    PatchedInterpolationError::InvalidInput {
        message: message.into(),
    }
}

/// One patch waiting in the queue.
struct Patch<T> {
    /// (position, coordinate) of every fixed site, in split order.
    path: Vec<(usize, usize)>,
    /// Fixed coordinate of every site of the site order, if any.
    fixed: Vec<Option<usize>>,
    cache: PatchCache<T>,
    /// Full-domain pivots recycled from the parent's outcome.
    recycled: Vec<Vec<usize>>,
    /// Full-domain worst points of the parent's last failed measurement.
    worst: Vec<Vec<usize>>,
}

/// An accepted patch: its record, network, and `|P|`.
struct Accepted<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    record: PatchRecord,
    subdomain: SubDomainTreeTN<V>,
    patch_points: f64,
}

/// What processing one patch produced.
enum Verdict<T, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    Accepted(Box<Accepted<V>>),
    Zero(ZeroPatchRecord, f64),
    Split(Vec<Patch<T>>),
}

/// The pinned L2 reference and allowance.
#[derive(Clone, Copy, Debug)]
struct L2Pin {
    reference_rms: Option<f64>,
    source: L2ReferenceSource,
    tau: f64,
}

/// A failed verification: its measurement and its worst points in active
/// coordinates.
struct Failure {
    measurement: L2Measurement,
    worst: Vec<Vec<usize>>,
}

/// What resolving an unaccepted patch needs from its processing.
struct PatchContext<'a> {
    path: Vec<(usize, usize)>,
    fixed: &'a [Option<usize>],
    active: &'a [usize],
    active_dims: &'a [usize],
    /// `|P|` as `f64`.
    patch_points: f64,
    /// `|P|`, or `None` when its exhaustive point list cannot fit in a `Vec`.
    patch_count: Option<usize>,
    seeds: &'a PatchSeeds,
    /// `tau` under L2, the engine tolerance under `SampledMax`.
    tolerance: f64,
}

/// The last engine run of a patch that was not accepted.
struct Unaccepted<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    outcome: InterpolationOutcome<V>,
    attempt: usize,
    /// The checked, re-embedded network of a run that was judged and failed
    /// (measured under L2, estimate-checked under `SampledMax`); `None` for
    /// a run that was not judged.
    rejected: Option<SubDomainTreeTN<V>>,
    /// The last failed L2 measurement of the patch: of this run when
    /// `rejected` is set under L2, possibly of an earlier run when it is not,
    /// and `None` under `SampledMax`.
    failure: Option<Failure>,
}

fn max_magnitude<T: CommonScalar>(values: &[T]) -> f64 {
    values
        .iter()
        .map(|value| value.abs_val())
        .fold(0.0, f64::max)
}

fn evaluator_error(projector: &Projector, source: anyhow::Error) -> PatchedInterpolationError {
    PatchedInterpolationError::Interpolation {
        projector: projector.clone(),
        source: InterpolationError::Evaluator { source },
    }
}

fn engine_error(projector: &Projector, message: String) -> PatchedInterpolationError {
    PatchedInterpolationError::Interpolation {
        projector: projector.clone(),
        source: InterpolationError::Engine {
            source: anyhow::anyhow!(message),
        },
    }
}

fn measure_error(projector: &Projector, error: MeasureError) -> PatchedInterpolationError {
    match error {
        // The list length is bounded by `verification.samples` or
        // `verification.max_exhaustive_points`, so the remedy is an option
        // change: `InvalidInput`, not `ResourceLimit` (whose remedy is to
        // raise a limit).
        MeasureError::PointList(message) => invalid(format!(
            "could not construct the verification point list of the patch {projector:?}: \
             {message}; lower verification.samples or verification.max_exhaustive_points"
        )),
        MeasureError::Evaluator(source) => evaluator_error(projector, source),
        MeasureError::Network(message) => engine_error(projector, message),
    }
}

/// A violated internal invariant, reported as a construction failure.
fn internal(source: impl Into<anyhow::Error>) -> PatchedInterpolationError {
    PatchedInterpolationError::Partition {
        source: PartitionedTreeTNError::TensorConstruction {
            source: source.into(),
        },
    }
}

/// An outcome's pivots in active coordinates, checked for shape and range.
fn active_pivots(
    pivots: Option<&ColMajorArray<usize>>,
    active: &[usize],
    dims: &[usize],
) -> Result<Vec<Vec<usize>>, String> {
    let Some(pivots) = pivots else {
        return Ok(Vec::new());
    };
    let (Some(n_rows), Some(n_cols)) = (pivots.nrows(), pivots.ncols()) else {
        return Err(format!(
            "the outcome pivots have shape {:?}, expected a 2D array",
            pivots.shape()
        ));
    };
    if n_rows != active.len() {
        return Err(format!(
            "the outcome pivots have {n_rows} rows, expected {} active sites",
            active.len()
        ));
    }
    (0..n_cols)
        .map(|column| {
            let local = pivots.column(column).unwrap_or_default();
            for (&position, &value) in active.iter().zip(local) {
                if value >= dims[position] {
                    return Err(format!(
                        "the outcome pivot {column} has coordinate {value} for a site of \
                         dimension {}",
                        dims[position]
                    ));
                }
            }
            Ok(local.to_vec())
        })
        .collect()
}

/// Complete active-coordinate points with the fixed coordinates.
fn complete_points(points: &[Vec<usize>], fixed: &[Option<usize>]) -> Vec<Vec<usize>> {
    points
        .iter()
        .map(|local| {
            let mut active = local.iter().copied();
            fixed
                .iter()
                .map(|fixed| fixed.or_else(|| active.next()).unwrap_or_default())
                .collect()
        })
        .collect()
}

/// The added initial pivots of a rerun: the worst points of the failed run,
/// then its outcome pivots, without points already among the base
/// candidates or earlier in the list, truncated to `limit`.
fn added_pivots(
    base: &HashSet<Vec<usize>>,
    worst: &[Vec<usize>],
    outcome_pivots: &[Vec<usize>],
    limit: usize,
) -> Vec<Vec<usize>> {
    let mut seen = base.clone();
    worst
        .iter()
        .chain(outcome_pivots)
        .filter(|point| seen.insert((*point).clone()))
        .take(limit)
        .cloned()
        .collect()
}

/// State of one [`patched_interpolate`] run.
struct Driver<'a, T, V, E, F>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    engine: &'a E,
    evaluate: &'a F,
    layout: SiteLayout<V>,
    initial_pivots: ColMajorArray<usize>,
    options: &'a PatchedInterpolationOptions,
    policy: SizePolicy,
    /// SampledMax: the given reference, or the one pinned from the root.
    max_reference: Cell<Option<(f64, MaxReferenceSource)>>,
    /// L2: the pinned reference and allowance.
    l2: Cell<Option<L2Pin>>,
    counters: Counters,
    measurement_evaluations: Cell<usize>,
    audit_evaluations: Cell<usize>,
    verification_failures: Cell<usize>,
    engine_retries: Cell<usize>,
    scalar: PhantomData<T>,
}

impl<T, V, E, F> Driver<'_, T, V, E, F>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>,
{
    fn is_l2(&self) -> bool {
        matches!(self.options.error_norm, ErrorNorm::L2 { .. })
    }

    /// Pin the references that do not depend on the root before any
    /// evaluation: a given L2 norm, no L2 reference when `rtol = 0`, and a
    /// given `max_reference`.
    fn pin_references_known_in_advance(&self) {
        let tolerance = self.options.tolerance;
        match self.options.error_norm {
            ErrorNorm::L2 {
                reference: L2Reference::Given(norm),
                ..
            } => {
                self.pin_l2(
                    Some(norm / self.layout.domain_points.sqrt()),
                    L2ReferenceSource::Given,
                );
            }
            ErrorNorm::L2 { .. } if tolerance.rtol == 0.0 => {
                self.pin_l2(None, L2ReferenceSource::NotNeeded);
            }
            ErrorNorm::SampledMax {
                max_reference: Some(scale),
                ..
            } => self
                .max_reference
                .set(Some((scale, MaxReferenceSource::Given))),
            _ => {}
        }
    }

    /// Pin the L2 reference and `tau`, unless already pinned:
    /// `tau = max(atol / sqrt(|X|), rtol * reference_rms)`.
    fn pin_l2(&self, reference_rms: Option<f64>, source: L2ReferenceSource) {
        if self.l2.get().is_some() {
            return;
        }
        let tolerance = self.options.tolerance;
        let floor = tolerance.atol / self.layout.domain_points.sqrt();
        let tau = match reference_rms {
            Some(rms) => floor.max(tolerance.rtol * rms),
            None => floor,
        };
        self.l2.set(Some(L2Pin {
            reference_rms,
            source,
            tau,
        }));
    }

    fn l2_pin(&self) -> Result<L2Pin, PatchedInterpolationError> {
        self.l2
            .get()
            .ok_or_else(|| internal(anyhow::anyhow!("the L2 reference is not pinned")))
    }

    fn run(self) -> Result<PatchedInterpolationResult<V>, PatchedInterpolationError> {
        let mut queue = VecDeque::from([Patch {
            path: Vec::new(),
            fixed: vec![None; self.layout.sites.len()],
            cache: PatchCache::new(self.layout.dims.clone()),
            recycled: Vec::new(),
            worst: Vec::new(),
        }]);
        let mut processed = 0usize;
        let mut splits = 0usize;
        let mut accepted = Vec::new();
        let mut zeros = Vec::new();
        while let Some(patch) = queue.pop_front() {
            if let Some(limit) = self.options.max_patches {
                if processed == limit {
                    return Err(PatchedInterpolationError::ResourceLimit {
                        resource: "max_patches",
                        limit,
                    });
                }
            }
            processed += 1;
            let path = patch.path.clone();
            let projector = self.layout.projector(&path)?;
            match self.process(patch, &projector)? {
                Verdict::Accepted(entry) => accepted.push((path, *entry)),
                Verdict::Zero(record, patch_points) => zeros.push((path, record, patch_points)),
                Verdict::Split(children) => {
                    splits += 1;
                    queue.extend(children);
                }
            }
        }

        // Canonical order: lexicographic over the (position, coordinate) path.
        accepted.sort_by(|left, right| left.0.cmp(&right.0));
        zeros.sort_by(|left, right| left.0.cmp(&right.0));
        let norm = self.norm_report(&accepted, &zeros)?;
        let mut records = Vec::with_capacity(accepted.len());
        let mut subdomains = Vec::with_capacity(accepted.len());
        for (_, entry) in accepted {
            records.push(entry.record);
            subdomains.push(entry.subdomain);
        }
        let partition = PartitionedTreeTN::from_disjoint_subdomains(subdomains)?;
        Ok(PatchedInterpolationResult {
            partition,
            report: PatchedInterpolationReport {
                tolerance: self.options.tolerance,
                norm,
                accepted: records,
                zero_patches: zeros.into_iter().map(|(_, record, _)| record).collect(),
                splits,
                function_evaluations: self.counters.evaluations.get(),
                cache_hits: self.counters.cache_hits.get(),
                measurement_evaluations: self.measurement_evaluations.get(),
                audit_evaluations: self.audit_evaluations.get(),
                verification_failures: self.verification_failures.get(),
                engine_retries: self.engine_retries.get(),
            },
        })
    }

    /// The norm part of the report. Under L2 the global error and the
    /// approximation norm are combined in canonical path order.
    #[allow(clippy::type_complexity)]
    fn norm_report(
        &self,
        accepted: &[(Vec<(usize, usize)>, Accepted<V>)],
        zeros: &[(Vec<(usize, usize)>, ZeroPatchRecord, f64)],
    ) -> Result<NormReport, PatchedInterpolationError> {
        if !self.is_l2() {
            let (max_reference, source) = self
                .max_reference
                .get()
                .unwrap_or((0.0, MaxReferenceSource::ExactRoot));
            return Ok(NormReport::SampledMax {
                max_reference,
                source,
                engine_tolerance: self.options.tolerance.allowance(max_reference),
            });
        }
        let pin = self.l2_pin()?;
        let domain_points = self.layout.domain_points;
        let missing = || internal(anyhow::anyhow!("an L2 patch has no acceptance measurement"));

        // Accepted and zero patches merged in canonical path order.
        let mut contributions: Vec<(&[(usize, usize)], Contribution<'_>)> = Vec::new();
        for (path, entry) in accepted {
            contributions.push((
                path,
                Contribution {
                    patch_points: entry.patch_points,
                    acceptance: entry.record.acceptance.as_ref().ok_or_else(missing)?,
                    audit: entry.record.audit.as_ref(),
                    within_tolerance: entry.record.status == PatchStatus::WithinTolerance,
                },
            ));
        }
        for (path, record, patch_points) in zeros {
            contributions.push((
                path,
                Contribution {
                    patch_points: *patch_points,
                    acceptance: record.acceptance.as_ref().ok_or_else(missing)?,
                    audit: record.audit.as_ref(),
                    within_tolerance: true,
                },
            ));
        }
        contributions.sort_by(|left, right| left.0.cmp(right.0));
        let contributions: Vec<Contribution<'_>> =
            contributions.into_iter().map(|(_, c)| c).collect();

        let approximation_rms = verify::approximation_rms(
            accepted.iter().map(|(_, entry)| {
                (
                    entry.subdomain.data().clone().log_norm(),
                    entry.patch_points,
                )
            }),
            domain_points,
        );
        let (global, certified_fraction) =
            verify::global_error(&contributions, domain_points, pin.tau, approximation_rms);
        Ok(NormReport::L2 {
            reference_rms: pin.reference_rms,
            source: pin.source,
            tau: pin.tau,
            error: L2ErrorReport {
                domain_points,
                global,
                certified_fraction,
                approximation_rms,
            },
        })
    }

    fn sampler<'s>(
        &'s self,
        fixed: &'s [Option<usize>],
        n_active: usize,
        cache: PatchCache<T>,
    ) -> PatchSampler<'s, T, F> {
        PatchSampler {
            evaluate: self.evaluate,
            fixed,
            n_active,
            counters: &self.counters,
            cache: RefCell::new(cache),
        }
    }

    /// Measure `f - f~` (or `f` for the zero approximation) on a patch and
    /// count its new evaluations as measurement (and audit) evaluations.
    fn measure(
        &self,
        sampler: &PatchSampler<'_, T, F>,
        target: &MeasureTarget<'_, V>,
        plan: PointPlan,
        audit: bool,
        projector: &Projector,
    ) -> Result<Measured, PatchedInterpolationError> {
        let before = self.counters.evaluations.get();
        let measured = verify::measure(sampler, target, plan);
        let added = self.counters.evaluations.get() - before;
        Counters::add(&self.measurement_evaluations, added);
        if audit {
            Counters::add(&self.audit_evaluations, added);
        }
        measured.map_err(|error| measure_error(projector, error))
    }

    /// The audit of a sampled acceptance, if enabled. Audit points never
    /// become pivots: the patch is accepted, and its cache is dropped.
    fn audit(
        &self,
        sampler: &PatchSampler<'_, T, F>,
        target: &MeasureTarget<'_, V>,
        acceptance: &L2Measurement,
        seeds: &PatchSeeds,
        projector: &Projector,
    ) -> Result<Option<L2Measurement>, PatchedInterpolationError> {
        if acceptance.method != MeasurementMethod::Sampled || !self.options.verification.audit {
            return Ok(None);
        }
        let plan = PointPlan::Sampled {
            count: self.options.verification.samples,
            seed: seeds.audit(),
        };
        let audit = self.measure(sampler, target, plan, true, projector)?;
        Ok(Some(audit.measurement))
    }

    /// The measurement plan of a patch with the given point count.
    fn plan(&self, patch_count: Option<usize>, seed: u64) -> PointPlan {
        let verification = self.options.verification;
        PointPlan::for_patch(
            patch_count,
            verification.samples,
            verification.max_exhaustive_points,
            seed,
        )
    }

    /// Pin the L2 reference of a root that needs the engine from a Monte
    /// Carlo estimate on the root's reference stream.
    fn pin_monte_carlo(
        &self,
        sampler: &PatchSampler<'_, T, F>,
        zero_target: &MeasureTarget<'_, V>,
        seeds: &PatchSeeds,
        projector: &Projector,
    ) -> Result<(), PatchedInterpolationError> {
        let samples = self.options.verification.samples;
        let plan = PointPlan::Sampled {
            count: samples,
            seed: seeds.scale(),
        };
        // The residual of the zero approximation is |f|.
        let estimate = self.measure(sampler, zero_target, plan, false, projector)?;
        let reference_rms = estimate.measurement.rms;
        if reference_rms == 0.0 && self.options.tolerance.atol == 0.0 {
            return Err(invalid(format!(
                "the Monte Carlo reference norm is zero: all {samples} uniform root samples are \
                 exactly zero; give L2Reference::Given(norm), a positive tolerance.atol, or \
                 more verification.samples"
            )));
        }
        self.pin_l2(
            Some(reference_rms),
            L2ReferenceSource::MonteCarlo {
                samples,
                mean_square_rel_std_error: estimate.measurement.mean_square_rel_std_error,
            },
        );
        Ok(())
    }

    fn process(
        &self,
        patch: Patch<T>,
        projector: &Projector,
    ) -> Result<Verdict<T, V>, PatchedInterpolationError> {
        let Patch {
            path,
            fixed,
            cache,
            recycled,
            worst,
        } = patch;
        let layout = &self.layout;
        let active: Vec<usize> = (0..layout.sites.len())
            .filter(|&position| fixed[position].is_none())
            .collect();
        if active.len() <= 1 {
            return self.exact_patch(&fixed, &active, cache, projector);
        }
        let active_dims: Vec<usize> = active.iter().map(|&p| layout.dims[p]).collect();
        let patch_points: f64 = active_dims.iter().map(|&dim| dim as f64).product();

        let seeds = patch_seeds(self.options.seed, &path);
        let domain = PatchDomain {
            dims: &layout.dims,
            fixed: &fixed,
            active: &active,
            layout: cache.layout(),
        };
        // `None` when the point count or its exhaustive point-list capacity
        // cannot fit in a Vec: such a patch is measured by sampling.
        let patch_count = active_dims
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .filter(|&count| point_list_capacity(active.len(), count).is_some());
        // User pivots come first inside `patch_candidates`; then the recycled
        // pivots, the parent's worst points, and (opt-in) the largest cached
        // values of the patch, each kept once.
        let cached = if self.options.cache_candidates {
            let points =
                cache.largest_points(self.options.n_initial_pivots, |value| value.abs_val());
            complete_points(&points, &fixed)
        } else {
            Vec::new()
        };
        let prior: Vec<Vec<usize>> = recycled.into_iter().chain(worst).chain(cached).collect();
        let candidates = patch_candidates(
            &domain,
            &self.initial_pivots,
            &prior,
            self.options.n_initial_pivots,
            seeds.candidates,
        );
        let sampler = self.sampler(&fixed, active.len(), cache);
        let zero_target = MeasureTarget {
            network: None,
            sites: &layout.sites,
            fixed: &fixed,
            active_dims: &active_dims,
            patch_points,
        };
        if self.is_l2() && self.l2.get().is_none() {
            self.pin_monte_carlo(&sampler, &zero_target, &seeds, projector)?;
        }
        let shape = [active.len(), candidates.count];
        let batch = ColMajorArrayRef::new(&candidates.points, &shape).map_err(internal)?;
        let samples = sampler
            .sample(batch)
            .map_err(|source| evaluator_error(projector, source))?;
        let largest = max_magnitude(&samples);
        let mut base: Vec<Vec<usize>> = candidates
            .points
            .chunks(active.len())
            .map(<[usize]>::to_vec)
            .collect();
        let limit = self.options.max_bond_dim - 1;

        let tolerance = if self.is_l2() {
            let tau = self.l2_pin()?.tau;
            if largest == 0.0 {
                // Zero screen: measure the zero approximation.
                let plan = self.plan(patch_count, seeds.zero_screen());
                let screen = self.measure(&sampler, &zero_target, plan, false, projector)?;
                if screen.measurement.rms <= tau {
                    let audit = self.audit(
                        &sampler,
                        &zero_target,
                        &screen.measurement,
                        &seeds,
                        projector,
                    )?;
                    let record = ZeroPatchRecord {
                        projector: projector.clone(),
                        acceptance: Some(screen.measurement),
                        audit,
                    };
                    return Ok(Verdict::Zero(record, patch_points));
                }
                // The measured points with f != 0, largest |f| first, join
                // the base candidates; they are cached. Every candidate
                // sample is zero, so none of them is a candidate already.
                base.extend(screen.worst_points(0.0, limit));
            }
            tau
        } else {
            let scale = match self.max_reference.get() {
                Some((scale, _)) => scale,
                None if largest == 0.0 => {
                    return Err(invalid(format!(
                        "the max_reference of ErrorNorm::SampledMax cannot be pinned: all {} \
                         candidate samples of the root patch are exactly zero; give \
                         ErrorNorm::sampled_max_with_reference(max_abs) or initial pivots in \
                         the support of the function",
                        candidates.count
                    )));
                }
                None => {
                    self.max_reference
                        .set(Some((largest, MaxReferenceSource::MaxOfRootCandidates)));
                    largest
                }
            };
            if largest == 0.0 {
                let record = ZeroPatchRecord {
                    projector: projector.clone(),
                    acceptance: None,
                    audit: None,
                };
                return Ok(Verdict::Zero(record, patch_points));
            }
            self.options.tolerance.allowance(scale)
        };

        let interpolation_error = |source| PatchedInterpolationError::Interpolation {
            projector: projector.clone(),
            source,
        };
        let base_set: HashSet<Vec<usize>> = base.iter().cloned().collect();
        let retries = self.options.verification.retries;
        let mut added: Vec<Vec<usize>> = Vec::new();
        let mut failure: Option<Failure> = None;
        let mut attempt = 0usize;
        let (outcome, rejected) = loop {
            let initial: Vec<usize> = base.iter().chain(&added).flatten().copied().collect();
            let n_initial = base.len() + added.len();
            let problem = InterpolationProblem::new(
                layout.topology.clone(),
                layout.active_node_sites(&fixed),
                ColMajorArray::new(initial, vec![active.len(), n_initial]).map_err(internal)?,
                tolerance,
                NonZeroUsize::new(self.options.max_bond_dim),
                seeds.engine_run(attempt),
            )
            .map_err(interpolation_error)?;
            let outcome = self
                .engine
                .interpolate(&problem, |batch| sampler.sample(batch))
                .map_err(interpolation_error)?;
            let converged = outcome.termination == InterpolationTermination::Converged;
            if !converged
                && !self
                    .policy
                    .capped_eligible(outcome.termination, active.len())
            {
                break (outcome, None);
            }
            let subdomain = self.checked_subdomain(&outcome, &fixed, projector)?;
            let mut record = self.record(projector, &outcome, &subdomain, attempt);
            if !self.is_l2() {
                // A capped-eligible run is judged by the engine estimate.
                if converged || outcome.error_estimate <= tolerance {
                    return Ok(accepted(record, subdomain, patch_points));
                }
                break (outcome, Some(subdomain));
            }

            // Verify the re-embedded patch that would be stored.
            let target = MeasureTarget {
                network: Some(subdomain.data()),
                sites: &layout.sites,
                fixed: &fixed,
                active_dims: &active_dims,
                patch_points,
            };
            let plan = self.plan(patch_count, seeds.verify(attempt));
            let measured = self.measure(&sampler, &target, plan, false, projector)?;
            if measured.measurement.rms <= tolerance {
                record.audit =
                    self.audit(&sampler, &target, &measured.measurement, &seeds, projector)?;
                record.acceptance = Some(measured.measurement);
                return Ok(accepted(record, subdomain, patch_points));
            }
            Counters::add(&self.verification_failures, 1);
            // Every distinct worst point, largest first: a rerun drops the
            // base candidates before truncating (step 9); a split passes the
            // first `max_bond_dim - 1` of them (step 10).
            let all_worst = measured.worst_points(tolerance, usize::MAX);
            let worst: Vec<Vec<usize>> = all_worst.iter().take(limit).cloned().collect();
            failure = Some(Failure {
                measurement: measured.measurement,
                worst,
            });
            // Only a converged run is rerun: a capped run has exhausted its
            // rank, and more pivots only raise its starting rank.
            if converged && attempt < retries {
                let pivots = active_pivots(outcome.pivots.as_ref(), &active, &layout.dims)
                    .map_err(|message| engine_error(projector, message))?;
                added = added_pivots(&base_set, &all_worst, &pivots, limit);
                Counters::add(&self.engine_retries, 1);
                attempt += 1;
                continue;
            }
            break (outcome, Some(subdomain));
        };
        let context = PatchContext {
            path,
            fixed: &fixed,
            active: &active,
            active_dims: &active_dims,
            patch_points,
            patch_count,
            seeds: &seeds,
            tolerance,
        };
        let last = Unaccepted {
            outcome,
            attempt,
            rejected,
            failure,
        };
        self.resolve(context, last, sampler, projector)
    }

    /// The layout-checked, re-embedded network of an outcome the driver may
    /// use. A `Converged` outcome must be strictly below the bond cap (the
    /// M1 contract); any other outcome must not exceed it.
    fn checked_subdomain(
        &self,
        outcome: &InterpolationOutcome<V>,
        fixed: &[Option<usize>],
        projector: &Projector,
    ) -> Result<SubDomainTreeTN<V>, PatchedInterpolationError> {
        embed::check_outcome_layout(&outcome.network, &self.layout, fixed)
            .map_err(|message| engine_error(projector, message))?;
        let subdomain = self.subdomain(&outcome.network, fixed, projector)?;
        let cap = self.options.max_bond_dim;
        let bond = subdomain.max_bond_dim();
        if outcome.termination == InterpolationTermination::Converged && bond >= cap {
            return Err(engine_error(
                projector,
                format!(
                    "the engine reported Converged with bond dimension {bond}, not strictly below \
                     the cap {cap}"
                ),
            ));
        }
        if bond > cap {
            return Err(engine_error(
                projector,
                format!(
                    "the engine reported {:?} with bond dimension {bond}, above the cap {cap}",
                    outcome.termination
                ),
            ));
        }
        Ok(subdomain)
    }

    /// The record of an engine run, within its tolerance until judged
    /// otherwise, without measurements.
    fn record(
        &self,
        projector: &Projector,
        outcome: &InterpolationOutcome<V>,
        subdomain: &SubDomainTreeTN<V>,
        attempt: usize,
    ) -> PatchRecord {
        PatchRecord {
            projector: projector.clone(),
            termination: outcome.termination,
            engine_error_estimate: outcome.error_estimate,
            max_sample_magnitude: outcome.max_sample_magnitude,
            max_bond_dim: subdomain.max_bond_dim(),
            retries_used: attempt,
            acceptance: None,
            audit: None,
            status: PatchStatus::WithinTolerance,
        }
    }

    /// Resolve a patch that was not accepted: split it at the next unfixed
    /// site of the split order, retain it when the minimum blocks that
    /// split, or fail when no split site is left.
    fn resolve(
        &self,
        context: PatchContext<'_>,
        last: Unaccepted<V>,
        sampler: PatchSampler<'_, T, F>,
        projector: &Projector,
    ) -> Result<Verdict<T, V>, PatchedInterpolationError> {
        let layout = &self.layout;
        let fixed = context.fixed;
        let active = context.active;
        let Some(&split_position) = layout
            .split_order
            .iter()
            .find(|&&position| fixed[position].is_none())
        else {
            // The minimum never turns an exhausted patch_order into an
            // acceptance; a measured, failed last run is reported as such.
            return Err(match (last.rejected, last.failure) {
                (Some(_), Some(failure)) => PatchedInterpolationError::VerificationFailed {
                    projector: projector.clone(),
                    measurement: failure.measurement,
                },
                _ => PatchedInterpolationError::NoSplitIndexLeft {
                    projector: projector.clone(),
                },
            });
        };
        if self.policy.split_blocked(active.len()) {
            return self.retain_blocked(&context, last, &sampler, projector);
        }
        let Unaccepted {
            outcome, failure, ..
        } = last;
        let path = context.path;
        let recycled = if self.options.recycle_pivots {
            let pivots = active_pivots(outcome.pivots.as_ref(), active, &layout.dims)
                .map_err(|message| engine_error(projector, message))?;
            complete_points(&pivots, fixed)
        } else {
            Vec::new()
        };
        let worst = failure
            .map(|failure| complete_points(&failure.worst, fixed))
            .unwrap_or_default();
        let slot = active
            .iter()
            .position(|&position| position == split_position)
            .ok_or_else(|| internal(anyhow::anyhow!("the split site is not active")))?;
        let inside = |points: &[Vec<usize>], value: usize| -> Vec<Vec<usize>> {
            points
                .iter()
                .filter(|point| point[split_position] == value)
                .cloned()
                .collect()
        };
        let children = sampler
            .cache
            .into_inner()
            .split(slot)
            .into_iter()
            .enumerate()
            .map(|(value, cache)| {
                let mut child_path = path.clone();
                child_path.push((split_position, value));
                let mut child_fixed = fixed.to_vec();
                child_fixed[split_position] = Some(value);
                Patch {
                    path: child_path,
                    fixed: child_fixed,
                    cache,
                    recycled: inside(&recycled, value),
                    worst: inside(&worst, value),
                }
            })
            .collect();
        Ok(Verdict::Split(children))
    }

    /// Evaluate a patch with at most one active site on all its points and
    /// build its network without the engine.
    fn exact_patch(
        &self,
        fixed: &[Option<usize>],
        active: &[usize],
        cache: PatchCache<T>,
        projector: &Projector,
    ) -> Result<Verdict<T, V>, PatchedInterpolationError> {
        // Every point of the patch: `0..d` for one active site of dimension
        // `d`, or the single empty point when no site is active.
        let n_points: usize = active.iter().map(|&p| self.layout.dims[p]).product();
        let points: Vec<usize> = active.iter().flat_map(|_| 0..n_points).collect();
        let shape = [active.len(), n_points];
        let sampler = self.sampler(fixed, active.len(), cache);
        let batch = ColMajorArrayRef::new(&points, &shape).map_err(internal)?;
        let values = sampler
            .sample(batch)
            .map_err(|source| evaluator_error(projector, source))?;
        let largest = max_magnitude(&values);
        let patch_points = n_points as f64;
        let acceptance = if self.is_l2() {
            // Only the root can be exact before the reference is pinned; the
            // network multiplies the exact values by one-hot factors, so the
            // error is exactly zero and no measurement runs.
            self.pin_l2(
                Some(verify::rms_of(values.iter().map(|value| value.abs_val()))),
                L2ReferenceSource::ExactRoot,
            );
            Some(L2Measurement {
                method: MeasurementMethod::Exact,
                points: n_points,
                patch_points,
                rms: 0.0,
                mean_square_rel_std_error: 0.0,
                max_residual: 0.0,
            })
        } else {
            if self.max_reference.get().is_none() {
                self.max_reference
                    .set(Some((largest, MaxReferenceSource::ExactRoot)));
            }
            None
        };
        if largest == 0.0 {
            let record = ZeroPatchRecord {
                projector: projector.clone(),
                acceptance,
                audit: None,
            };
            return Ok(Verdict::Zero(record, patch_points));
        }
        let network = embed::exact_active_network(&self.layout, active, values)?;
        let subdomain = self.subdomain(&network, fixed, projector)?;
        let record = PatchRecord {
            projector: projector.clone(),
            termination: InterpolationTermination::Converged,
            engine_error_estimate: 0.0,
            max_sample_magnitude: largest,
            max_bond_dim: subdomain.max_bond_dim(),
            retries_used: 0,
            acceptance,
            audit: None,
            status: PatchStatus::WithinTolerance,
        };
        Ok(accepted(record, subdomain, patch_points))
    }

    /// Re-embed the fixed sites into an active-site network and wrap the
    /// already masked result as a patch.
    fn subdomain(
        &self,
        network: &TreeTN<IdxTensor, V>,
        fixed: &[Option<usize>],
        projector: &Projector,
    ) -> Result<SubDomainTreeTN<V>, PatchedInterpolationError> {
        let data = embed::embed_fixed_sites::<T, V>(network, &self.layout, fixed)?;
        Ok(SubDomainTreeTN::from_masked_data(
            data,
            projector.clone(),
            None,
        )?)
    }
}

fn accepted<T, V>(
    record: PatchRecord,
    subdomain: SubDomainTreeTN<V>,
    patch_points: f64,
) -> Verdict<T, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    Verdict::Accepted(Box::new(Accepted {
        record,
        subdomain,
        patch_points,
    }))
}
