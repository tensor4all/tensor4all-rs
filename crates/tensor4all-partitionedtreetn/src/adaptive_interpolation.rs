//! Adaptive patched interpolation of a function on a tree.
//!
//! [`patched_interpolate`] runs a tree interpolation engine (any
//! [`TreeInterpolator`]) on the whole domain of a function. Wherever a patch
//! cannot be accepted, it fixes the next site of a given order and retries on
//! every child region, unless an optional minimum patch size stops the
//! splitting. The result is a [`PartitionedTreeTN`] with disjoint, eagerly
//! masked patches, together with a [`PatchedInterpolationReport`].
//!
//! # Entries
//!
//! Three entries run the same driver, which is one wave loop: the BFS
//! generations of the patch queue are dispatched, merged in canonical path
//! order, and the splits of a wave form the next one. They differ only in where
//! a wave runs.
//!
//! - [`patched_interpolate`] runs every patch on the caller's session, one
//!   after another. The evaluator is `FnMut`, because one owner exists for the
//!   whole run.
//! - [`patched_interpolate_on_lanes`] fans a wave out over the lanes of a
//!   caller-owned [`TenferroLanes`](hataori_tenferro::TenferroLanes): each patch
//!   runs in the child session of the lane that pulled it, and the evaluator
//!   must be `Fn + Sync`. The root wave and every wave smaller than the lane
//!   count run one patch at a time, so a single item keeps the backend's whole
//!   team.
//! - `patched_interpolate_mpi` (feature `hataori-mpi` or `hataori-mpi-rsmpi-rt`)
//!   distributes each wave over MPI ranks through `hataori::pmap_lanes`; the
//!   root owns the queue, the merge, the pinned references and the result.
//!
//! # Execution and publication
//!
//! Every numerical step of a run - the engine call, the re-embedding of fixed
//! sites, the measurement of a stored patch and the readout of its norm - runs
//! on one session: the session of the caller's [`EvalContext`], or the child
//! session of the lane that runs the patch. The evaluator receives that same
//! context, so a caller-owned execution capability reaches every stage and an
//! evaluator that needs numerical work of its own can issue it on the session.
//! No step enters the process-global session, an eager runtime or a
//! thread-local default, so a run completes while another thread holds the
//! process-global backend and needs no warm-up of it.
//!
//! The publication boundary is explicit: every patch network is published as
//! compact dense host storage (`IdxTensor`) on the session that built it, never
//! adopted into the process-global eager runtime. The returned
//! [`PartitionedTreeTN`] and [`PatchedInterpolationReport`] are therefore
//! usable after the session has closed. A patch also reads its own log-norm on
//! that same session, so the final report assembly is arithmetic only and needs
//! no session - which is what lets a distributed run assemble on the root
//! alone.
//!
//! # Scheduling and determinism
//!
//! The schedule never changes the result contract: every patch derives its
//! random sub-seeds from the seed and its own path, the wave merge sorts by
//! canonical path, counters are per patch and summed, and the references are
//! pinned while the single root patch runs. `max_patches` is checked before a
//! wave is dispatched: at most the remaining allowance of that wave runs, in
//! canonical order, and the run then reports the same
//! [`PatchedInterpolationError::ResourceLimit`] as before.
//!
//! # Derivation notice
//!
//! The patch queue, the accept-or-split flow, and pivot recycling are derived
//! from `adaptiveinterpolate`, `createpatch`, and `_globalpivots` in
//! [TCIAlgorithms.jl](https://github.com/tensor4all/TCIAlgorithms.jl) at
//! commit e501032278c9dd41b46c5851d8238169c8d178c5 (MIT license; Copyright
//! 2023 Ritter.Marc and contributors), through the chain driver of the
//! retired `tensor4all-partitionedtt` crate (removed in #915 D3). See
//! `LICENSE-TCIALGORITHMS-MIT` in this crate. The intermediate Rust chain
//! driver was introduced by Samuel Badr (@SamuelBadr) in
//! [tensor4all-rs commit 69a24e7e](https://github.com/tensor4all/tensor4all-rs/commit/69a24e7e).
//! The tree generalization, the evaluation cache, the sampled-zero policy,
//! the re-embedding of fixed sites, and the L2 error contract are original
//! to this crate.
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
//! 2. Patches are processed wave by wave in the FIFO order of the queue,
//!    starting from the whole domain. The reference is pinned while the root
//!    patch runs and never changes; because the root is the only patch of wave
//!    0, a fan-out can pin it once and share it with every later wave.
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
//! every measured network value is reproducible. The schedule is not part of
//! that contract either: the wave merge sorts by canonical path, the counters
//! are per patch, and a lanes run at one or two lanes reproduces the serial
//! report on such fixtures. This is tested for `f64` on
//! trees where every node carries exactly one site, where the measurement
//! takes the cached evaluator's raw kernels. Each measurement uses a fresh
//! `TreeTNCachedEvaluator` centered at the smallest node name, the default
//! hint, and fixed-size chunks. On other trees (a site-free node, a node with
//! several sites, or `f32`/`c32` data) the evaluator's generic path can
//! differ at rounding level between threads and processes, so an L2
//! acceptance near `tau` may then differ between runs; this is an open
//! evaluator limitation tracked upstream in
//! [tenferro-rs #1963](https://github.com/tensor4all/tenferro-rs/issues/1963).
//! [Issue #795](https://github.com/tensor4all/tensor4all-rs/issues/795) closed
//! the tracking and test gap through PR #816; it did not fix the planner.
//! The intended scope is bitwise identical results on the same machine and
//! build across threads, thread counts, and processes, with no cross-machine
//! promise. It is not reached yet: generic-path measurements need deterministic
//! contraction paths, and reproducibility across processes is not claimed
//! until a two-process test passes, which does not exist yet. The bitwise claim
//! never covers `approximation_rms` and the fields derived from it. Issue #791
//! was fixed by PR #793: TreeTN construction, external-site order, and dense
//! output order no longer inherit the old hash-iteration nondeterminism.
//! `site_space` remains a set with unspecified iteration order; generic cached
//! evaluation still has the separate planner limitation above.
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
//! use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
//! use tensor4all_treetci::TreeTciInterpolator;
//! use tensor4all_treetn::interpolation::{EvalContext, Parallelism};
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
//! let evaluate = |batch: ColMajorArrayRef<'_, usize>, _ctx: &mut EvalContext<'_>| -> anyhow::Result<Vec<f64>> {
//!     Ok(batch.data().chunks(3).map(f).collect())
//! };
//! let options = PatchedInterpolationOptions::new(2)
//!     .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
//!     .with_tolerance(ErrorTolerance { rtol: 1e-12, atol: 0.0 });
//! let pivots = ColMajorArray::new(vec![0, 0, 0], vec![3, 1])?;
//! let result = {
//!     let mut backend = CpuBackend::with_threads(1).expect("1T CPU backend");
//!     let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
//!         patched_interpolate(
//!             &TreeTciInterpolator::default(),
//!             topology,
//!             node_sites,
//!             pivots,
//!             &mut ctx,
//!             evaluate,
//!             &options,
//!         )
//! }?;
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
mod fan_out;
mod layout;
#[cfg(any(feature = "hataori-mpi", feature = "hataori-mpi-rsmpi-rt"))]
mod mpi;
mod options;
mod report;
mod sampling;
#[cfg(test)]
mod tests;
mod verify;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use tensor4all_core::{
    ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, TensorElement,
};
use tensor4all_treetn::interpolation::{
    EvalContext, InterpolationError, InterpolationOutcome, InterpolationProblem,
    InterpolationTermination, TreeInterpolator,
};
use tensor4all_treetn::{NodeNameNetwork, TreeTN};

use self::sampling::point_list_capacity;

use crate::error::PartitionedTreeTNError;
use crate::subdomain_tree_tn::ScalarKind;
use crate::{ErrorNorm, L2Reference, PartitionedTreeTN, Projector, SubDomainTreeTN};

use acceptance::SizePolicy;
use cache::{Counters, PatchCache, PatchSampler};
use layout::SiteLayout;
use sampling::{patch_candidates, patch_seeds, PatchDomain, PatchSeeds};
use verify::{Contribution, MeasureError, MeasureTarget, Measured, PointPlan};

pub use fan_out::patched_interpolate_on_lanes;
#[cfg(any(feature = "hataori-mpi", feature = "hataori-mpi-rsmpi-rt"))]
pub use mpi::patched_interpolate_mpi;
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
/// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
/// use tensor4all_treetn::interpolation::{EvalContext, Parallelism};
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
///     {
///     let mut backend = CpuBackend::with_threads(1).expect("1T CPU backend");
///     let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
///         patched_interpolate(
///         &TreeTciInterpolator::default(),
///         topology,
///         BTreeMap::from([(0usize, vec![DynIndex::new_dyn(2)])]),
///         ColMajorArray::new(vec![], vec![1, 0]).unwrap(), &mut ctx,
///         |batch: ColMajorArrayRef<'_, usize>, _ctx: &mut EvalContext<'_>| -> anyhow::Result<Vec<f64>> {
///             Ok(vec![1.0; batch.shape()[1]])
///         },
///         options,
///     )
/// }
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
    /// The scheduler fan-out failed: a lane executor that could not run, join,
    /// or clean up a lane, or an MPI transport, placement, or wire failure. A
    /// patch failure caused by the inputs, the evaluator, or an option is
    /// reported under its own variant, and a distributed run reports a
    /// root-side failure here on every rank that is not the root.
    #[error("the patched interpolation fan-out failed: {message}")]
    FanOut {
        /// The failure message the fan-out converged on.
        message: String,
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
/// * `ctx` - The run's explicit execution capability, forwarded to every
///   engine call. The driver opens no session, installs no pool and consults no
///   process-global or thread-local default; pass the held session's context
///   ([`EvalContext::new`]) for a serial run, or a scheduler lane's context once
///   the caller owns the lanes.
/// * `evaluate` - Batch evaluator. It receives a column-major
///   `[n_sites, n_points]` array of full-domain points in site order and the
///   run's context, and returns one finite value per point, with a finite
///   magnitude. An evaluator that needs numerical work of its own issues it
///   through the context's session; an evaluator that only computes host
///   values can ignore the context.
/// * `options` - See [`PatchedInterpolationOptions`].
///
/// # Returns
///
/// The partition of accepted patches (eagerly masked, every site index
/// retained, one dtype `T`) and a [`PatchedInterpolationReport`]. Both are
/// published as compact dense host storage on the caller's session, so they
/// stay usable after the session closes.
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
///   An exact or candidate point list, or a split child collection, that cannot
///   be represented or reserved also returns `InvalidInput`; candidate counts
///   are clamped to the patch domain before their capacity is checked.
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
/// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
/// use tensor4all_treetn::interpolation::{EvalContext, Parallelism};
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
/// let result = {
///     let mut backend = CpuBackend::with_threads(1).expect("1T CPU backend");
///     let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
///         patched_interpolate(
///     &TreeTciInterpolator::default(),
///     topology,
///     node_sites,
///     ColMajorArray::new(vec![1, 2, 2], vec![3, 1]).unwrap(), &mut ctx,
///     |batch: ColMajorArrayRef<'_, usize>, _ctx: &mut EvalContext<'_>| -> anyhow::Result<Vec<f64>> {
///         Ok(batch.data().chunks(3).map(f).collect())
///     },
///     &options,
/// )
/// }?;
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
    ctx: &mut EvalContext<'_>,
    evaluate: F,
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<V>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T> + Sync,
    F: FnMut(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>>,
{
    let plan = Plan::validated(engine, topology, node_sites, initial_pivots, options)?;
    let mut waves = SerialWaves {
        ctx,
        evaluate: RefCell::new(evaluate),
    };
    let merged = run_waves(&plan, &mut waves)?;
    assemble(&plan, merged)
}

fn invalid(message: impl Into<String>) -> PatchedInterpolationError {
    PatchedInterpolationError::InvalidInput {
        message: message.into(),
    }
}

/// The caller's evaluator as one patch uses it.
///
/// A serial run has one mutable owner for the whole run, which a [`RefCell`]
/// lends to each patch in turn; a fanned-out run shares an immutable evaluator
/// across its lanes. Both reach a patch's sampler through this one trait, so
/// the per-patch body - engine, measurement, audit, publication - has a single
/// implementation for every entry.
trait Evaluate<T> {
    /// The values of the evaluator at a column-major
    /// `[n_sites, n_points]` batch of full-domain points.
    fn evaluate(
        &self,
        batch: ColMajorArrayRef<'_, usize>,
        ctx: &mut EvalContext<'_>,
    ) -> anyhow::Result<Vec<T>>;
}

/// A serial run's evaluator: one `FnMut` behind its cell.
struct Exclusive<'a, F>(&'a RefCell<F>);

impl<T, F> Evaluate<T> for Exclusive<'_, F>
where
    F: FnMut(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>>,
{
    fn evaluate(
        &self,
        batch: ColMajorArrayRef<'_, usize>,
        ctx: &mut EvalContext<'_>,
    ) -> anyhow::Result<Vec<T>> {
        let mut evaluate = self.0.borrow_mut();
        (*evaluate)(batch, ctx)
    }
}

/// A fanned-out run's evaluator: one `Fn` shared by the lanes.
struct Shared<'a, F>(&'a F);

impl<T, F> Evaluate<T> for Shared<'_, F>
where
    F: Fn(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>>,
{
    fn evaluate(
        &self,
        batch: ColMajorArrayRef<'_, usize>,
        ctx: &mut EvalContext<'_>,
    ) -> anyhow::Result<Vec<T>> {
        (*self.0)(batch, ctx)
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

/// An accepted patch with its canonical path.
#[allow(clippy::type_complexity)]
type AcceptedPatch<V> = (Vec<(usize, usize)>, Accepted<V>);

/// A zero patch with its canonical path and volume.
type ZeroPatch = (Vec<(usize, usize)>, ZeroPatchRecord, f64);

/// An accepted patch: its record, network, `|P|`, and the log-norm of its
/// network when the run reports an L2 error.
struct Accepted<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    record: PatchRecord,
    subdomain: SubDomainTreeTN<V>,
    patch_points: f64,
    /// `log ||f~_P||`, measured on the session that built the patch; `None`
    /// when the run has no L2 report or the norm overflowed.
    log_norm: Option<f64>,
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
#[derive(Clone, Copy, Debug, PartialEq)]
struct L2Pin {
    reference_rms: Option<f64>,
    source: L2ReferenceSource,
    tau: f64,
}

/// The reference norms of a run.
///
/// A run pins its references from its root patch, so the fields are still empty
/// while wave 0 runs and complete afterwards; every later wave reads them. The
/// order of the wave merge is what makes the pinned values a function of the
/// problem, not of the schedule.
#[derive(Clone, Copy, Debug, Default)]
struct References {
    /// Under L2: the pinned reference and allowance.
    l2: Option<L2Pin>,
    /// Under `SampledMax`: the given reference, or the one pinned from the root.
    max_reference: Option<(f64, MaxReferenceSource)>,
}

impl References {
    /// Adopt the pins of `other` this state does not hold yet.
    ///
    /// Two wave members can only disagree if the driver pinned a reference
    /// twice, which wave 0 (one root patch) cannot do; a disagreement is an
    /// internal error rather than a silently different run.
    fn merge(&mut self, other: Self) -> Result<(), PatchedInterpolationError> {
        match (self.l2, other.l2) {
            (None, Some(pin)) => self.l2 = Some(pin),
            (Some(mine), Some(theirs)) if mine != theirs => {
                return Err(internal(anyhow::anyhow!(
                    "two patches pinned different L2 references"
                )));
            }
            _ => {}
        }
        match (self.max_reference, other.max_reference) {
            (None, Some(reference)) => self.max_reference = Some(reference),
            (Some(mine), Some(theirs)) if mine != theirs => {
                return Err(internal(anyhow::anyhow!(
                    "two patches pinned different max references"
                )));
            }
            _ => {}
        }
        Ok(())
    }
}

/// One patch's access to the run's references.
enum Pins<'a> {
    /// A patch of wave 0, which is the only wave that pins a reference.
    Root(&'a mut References),
    /// A patch of any later wave: the references are pinned and read-only.
    Later(&'a References),
}

impl Pins<'_> {
    fn l2(&self) -> Option<L2Pin> {
        match self {
            Self::Root(references) => references.l2,
            Self::Later(references) => references.l2,
        }
    }

    fn max_reference(&self) -> Option<(f64, MaxReferenceSource)> {
        match self {
            Self::Root(references) => references.max_reference,
            Self::Later(references) => references.max_reference,
        }
    }

    fn l2_pin(&self) -> Result<L2Pin, PatchedInterpolationError> {
        self.l2()
            .ok_or_else(|| internal(anyhow::anyhow!("the L2 reference is not pinned")))
    }

    /// Pin the L2 reference unless one is pinned already.
    fn pin_l2(&mut self, pin: L2Pin) -> Result<(), PatchedInterpolationError> {
        match self {
            Self::Root(references) => {
                if references.l2.is_none() {
                    references.l2 = Some(pin);
                }
                Ok(())
            }
            Self::Later(references) => {
                if references.l2.is_none() {
                    return Err(internal(anyhow::anyhow!(
                        "a wave after the root pinned an L2 reference"
                    )));
                }
                Ok(())
            }
        }
    }

    /// Pin the max reference unless one is pinned already.
    fn pin_max_reference(
        &mut self,
        reference: (f64, MaxReferenceSource),
    ) -> Result<(), PatchedInterpolationError> {
        match self {
            Self::Root(references) => {
                if references.max_reference.is_none() {
                    references.max_reference = Some(reference);
                }
                Ok(())
            }
            Self::Later(references) => {
                if references.max_reference.is_none() {
                    return Err(internal(anyhow::anyhow!(
                        "a wave after the root pinned a max reference"
                    )));
                }
                Ok(())
            }
        }
    }

    /// The references as this patch sees them, for a wave merge.
    fn snapshot(&self) -> References {
        match self {
            Self::Root(references) => **references,
            Self::Later(references) => **references,
        }
    }
}

/// The references a wave starts from, with the write access wave 0 needs.
enum WavePins<'a> {
    /// Wave 0: the root patch may still pin.
    Root(&'a mut References),
    /// Any later wave: the pinned references, read-only.
    Later(&'a References),
}

impl<'a> WavePins<'a> {
    /// The references the wave starts from.
    fn start(&self) -> References {
        match self {
            Self::Root(references) => **references,
            Self::Later(references) => **references,
        }
    }

    /// Whether this wave may still pin a reference: only the root wave can.
    fn writable(&self) -> bool {
        matches!(self, Self::Root(_))
    }

    /// The access one patch of this wave sees.
    fn patch(&mut self) -> Pins<'_> {
        match self {
            Self::Root(references) => Pins::Root(references),
            Self::Later(references) => Pins::Later(references),
        }
    }
}

/// Evaluations and retries of one run, summed over its waves.
#[derive(Clone, Copy, Debug, Default)]
struct Totals {
    evaluations: usize,
    cache_hits: usize,
    measurement_evaluations: usize,
    audit_evaluations: usize,
    verification_failures: usize,
    engine_retries: usize,
}

impl Totals {
    fn add(&mut self, other: Self) {
        self.evaluations = self.evaluations.saturating_add(other.evaluations);
        self.cache_hits = self.cache_hits.saturating_add(other.cache_hits);
        self.measurement_evaluations = self
            .measurement_evaluations
            .saturating_add(other.measurement_evaluations);
        self.audit_evaluations = self
            .audit_evaluations
            .saturating_add(other.audit_evaluations);
        self.verification_failures = self
            .verification_failures
            .saturating_add(other.verification_failures);
        self.engine_retries = self.engine_retries.saturating_add(other.engine_retries);
    }
}

/// The counters of one patch while it runs.
///
/// One patch runs on one thread - on the caller's session or on one scheduler
/// lane - so its counters stay cell-based. They become a [`Totals`] delta when
/// the wave is merged, which keeps a run's report independent of the number of
/// lanes and of the assignment of patches to lanes.
#[derive(Default)]
struct Tally {
    counters: Counters,
    measurement_evaluations: Cell<usize>,
    audit_evaluations: Cell<usize>,
    verification_failures: Cell<usize>,
    engine_retries: Cell<usize>,
}

impl Tally {
    fn add(counter: &Cell<usize>, amount: usize) {
        Counters::add(counter, amount);
    }

    fn totals(&self) -> Totals {
        Totals {
            evaluations: self.counters.evaluations.get(),
            cache_hits: self.counters.cache_hits.get(),
            measurement_evaluations: self.measurement_evaluations.get(),
            audit_evaluations: self.audit_evaluations.get(),
            verification_failures: self.verification_failures.get(),
            engine_retries: self.engine_retries.get(),
        }
    }
}

/// What processing one patch produced, in the form a wave merge consumes.
struct PatchRun<T, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// (position, coordinate) of every fixed site, in split order.
    path: Vec<(usize, usize)>,
    verdict: Verdict<T, V>,
    /// The counter deltas of this patch.
    totals: Totals,
    /// The references after this patch; only wave 0 can add a pin.
    references: References,
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

/// A failed verification: its measurement and its worst points in active
/// coordinates.
struct Failure {
    measurement: L2Measurement,
    worst: Vec<Vec<usize>>,
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

/// The validated configuration of one run: everything a patch needs that does
/// not change between waves.
///
/// One `Plan` is shared by every wave and, when the run fans out, by every
/// lane: it holds no counter and no reference, so it is immutable for the whole
/// run.
struct Plan<'a, T, V, E>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    engine: &'a E,
    layout: SiteLayout<V>,
    initial_pivots: ColMajorArray<usize>,
    options: &'a PatchedInterpolationOptions,
    policy: SizePolicy,
    scalar: PhantomData<T>,
}

impl<'a, T, V, E> Plan<'a, T, V, E>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
{
    /// Validate the inputs and options in their documented order.
    fn validated(
        engine: &'a E,
        topology: NodeNameNetwork<V>,
        node_sites: BTreeMap<V, Vec<DynIndex>>,
        initial_pivots: ColMajorArray<usize>,
        options: &'a PatchedInterpolationOptions,
    ) -> Result<Self, PatchedInterpolationError> {
        if matches!(
            options.error_norm,
            ErrorNorm::MaxAbs | ErrorNorm::WeightedL2
        ) {
            return Err(PatchedInterpolationError::UnsupportedNorm {
                norm: options.error_norm,
            });
        }
        let layout = SiteLayout::validated(topology, node_sites, &initial_pivots, options)?;
        Ok(Self {
            engine,
            layout,
            initial_pivots,
            options,
            policy: SizePolicy::new(options),
            scalar: PhantomData,
        })
    }

    fn is_l2(&self) -> bool {
        matches!(self.options.error_norm, ErrorNorm::L2 { .. })
    }

    /// The references that do not depend on the root, pinned before any
    /// evaluation: a given L2 norm, no L2 reference when `rtol = 0`, and a
    /// given `max_reference`.
    fn initial_references(&self) -> References {
        let tolerance = self.options.tolerance;
        let mut references = References::default();
        match self.options.error_norm {
            ErrorNorm::L2 {
                reference: L2Reference::Given(norm),
                ..
            } => {
                references.l2 = Some(self.l2_pin(
                    Some(norm / self.layout.domain_points.sqrt()),
                    L2ReferenceSource::Given,
                ));
            }
            ErrorNorm::L2 { .. } if tolerance.rtol == 0.0 => {
                references.l2 = Some(self.l2_pin(None, L2ReferenceSource::NotNeeded));
            }
            ErrorNorm::SampledMax {
                max_reference: Some(scale),
                ..
            } => references.max_reference = Some((scale, MaxReferenceSource::Given)),
            _ => {}
        }
        references
    }

    /// The L2 reference and allowance: `tau = max(atol / sqrt(|X|), rtol *
    /// reference_rms)`.
    fn l2_pin(&self, reference_rms: Option<f64>, source: L2ReferenceSource) -> L2Pin {
        let tolerance = self.options.tolerance;
        let floor = tolerance.atol / self.layout.domain_points.sqrt();
        let tau = match reference_rms {
            Some(rms) => floor.max(tolerance.rtol * rms),
            None => floor,
        };
        L2Pin {
            reference_rms,
            source,
            tau,
        }
    }

    /// Whether every reference this run needs is pinned.
    ///
    /// True once wave 0 has run: the root patch is the only patch whose
    /// evaluation can pin a reference, so a later wave that meets an unpinned
    /// run would be a driver bug.
    fn is_pinned(&self, references: &References) -> bool {
        if self.is_l2() {
            references.l2.is_some()
        } else {
            references.max_reference.is_some()
        }
    }
}

impl<T, V, E> Plan<'_, T, V, E>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
{
    fn sampler<'s>(
        &'s self,
        evaluate: &'s dyn Evaluate<T>,
        tally: &'s Tally,
        fixed: &'s [Option<usize>],
        n_active: usize,
        cache: PatchCache<T>,
    ) -> PatchSampler<'s, T> {
        PatchSampler {
            evaluate,
            fixed,
            n_active,
            counters: &tally.counters,
            cache: RefCell::new(cache),
        }
    }

    /// Measure `f - f~` (or `f` for the zero approximation) on a patch and
    /// count its new evaluations as measurement (and audit) evaluations.
    #[allow(clippy::too_many_arguments)]
    fn measure(
        &self,
        tally: &Tally,
        sampler: &PatchSampler<'_, T>,
        target: &MeasureTarget<'_, V>,
        plan: PointPlan,
        audit: bool,
        projector: &Projector,
        ctx: &mut EvalContext<'_>,
    ) -> Result<Measured, PatchedInterpolationError> {
        let before = tally.counters.evaluations.get();
        let measured = verify::measure(sampler, target, plan, ctx);
        let added = tally.counters.evaluations.get() - before;
        Tally::add(&tally.measurement_evaluations, added);
        if audit {
            Tally::add(&tally.audit_evaluations, added);
        }
        measured.map_err(|error| measure_error(projector, error))
    }

    /// The audit of a sampled acceptance, if enabled. Audit points never
    /// become pivots: the patch is accepted, and its cache is dropped.
    #[allow(clippy::too_many_arguments)]
    fn audit(
        &self,
        tally: &Tally,
        sampler: &PatchSampler<'_, T>,
        target: &MeasureTarget<'_, V>,
        acceptance: &L2Measurement,
        seeds: &PatchSeeds,
        projector: &Projector,
        ctx: &mut EvalContext<'_>,
    ) -> Result<Option<L2Measurement>, PatchedInterpolationError> {
        if acceptance.method != MeasurementMethod::Sampled || !self.options.verification.audit {
            return Ok(None);
        }
        let plan = PointPlan::Sampled {
            count: self.options.verification.samples,
            seed: seeds.audit(),
        };
        let audit = self.measure(tally, sampler, target, plan, true, projector, ctx)?;
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
    #[allow(clippy::too_many_arguments)]
    fn pin_monte_carlo(
        &self,
        pins: &mut Pins<'_>,
        tally: &Tally,
        sampler: &PatchSampler<'_, T>,
        zero_target: &MeasureTarget<'_, V>,
        seeds: &PatchSeeds,
        projector: &Projector,
        ctx: &mut EvalContext<'_>,
    ) -> Result<(), PatchedInterpolationError> {
        let samples = self.options.verification.samples;
        let plan = PointPlan::Sampled {
            count: samples,
            seed: seeds.scale(),
        };
        // The residual of the zero approximation is |f|.
        let estimate = self.measure(tally, sampler, zero_target, plan, false, projector, ctx)?;
        let reference_rms = estimate.measurement.rms;
        if reference_rms == 0.0 && self.options.tolerance.atol == 0.0 {
            return Err(invalid(format!(
                "the Monte Carlo reference norm is zero: all {samples} uniform root samples are \
                 exactly zero; give L2Reference::Given(norm), a positive tolerance.atol, or \
                 more verification.samples"
            )));
        }
        pins.pin_l2(self.l2_pin(
            Some(reference_rms),
            L2ReferenceSource::MonteCarlo {
                samples,
                mean_square_rel_std_error: estimate.measurement.mean_square_rel_std_error,
            },
        ))
    }

    /// Process one patch: accept it, prove it zero, or split it.
    #[allow(clippy::too_many_arguments)]
    fn process(
        &self,
        patch: Patch<T>,
        projector: &Projector,
        pins: &mut Pins<'_>,
        evaluate: &dyn Evaluate<T>,
        tally: &Tally,
        ctx: &mut EvalContext<'_>,
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
            return self.exact_patch(
                pins, evaluate, tally, &fixed, &active, cache, projector, ctx,
            );
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
        )
        .map_err(invalid)?;
        let sampler = self.sampler(evaluate, tally, &fixed, active.len(), cache);
        let zero_target = MeasureTarget {
            network: None,
            sites: &layout.sites,
            fixed: &fixed,
            active_dims: &active_dims,
            patch_points,
        };
        if self.is_l2() && pins.l2().is_none() {
            self.pin_monte_carlo(pins, tally, &sampler, &zero_target, &seeds, projector, ctx)?;
        }
        let shape = [active.len(), candidates.count];
        let batch = ColMajorArrayRef::new(&candidates.points, &shape).map_err(internal)?;
        let samples = sampler
            .sample(batch, ctx)
            .map_err(|source| evaluator_error(projector, source))?;
        let largest = max_magnitude(&samples);
        let mut base: Vec<Vec<usize>> = candidates
            .points
            .chunks(active.len())
            .map(<[usize]>::to_vec)
            .collect();
        let limit = self.options.max_bond_dim - 1;

        let tolerance = if self.is_l2() {
            let tau = pins.l2_pin()?.tau;
            if largest == 0.0 {
                // Zero screen: measure the zero approximation.
                let plan = self.plan(patch_count, seeds.zero_screen());
                let screen =
                    self.measure(tally, &sampler, &zero_target, plan, false, projector, ctx)?;
                if screen.measurement.rms <= tau {
                    let audit = self.audit(
                        tally,
                        &sampler,
                        &zero_target,
                        &screen.measurement,
                        &seeds,
                        projector,
                        ctx,
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
            let scale = match pins.max_reference() {
                Some((scale, _)) => scale,
                None if largest == 0.0 => {
                    return Err(invalid(format!(
                        "the max_reference of ErrorNorm::SampledMax cannot be pinned: all {} \
                         candidate samples of the root patch are exactly zero; give \
                         ErrorNorm::sampled_max_with_reference(max_abs) or initial pivots in the \
                         support of the function",
                        candidates.count
                    )));
                }
                None => {
                    pins.pin_max_reference((largest, MaxReferenceSource::MaxOfRootCandidates))?;
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
                .interpolate(&problem, ctx, |batch, ctx| sampler.sample(batch, ctx))
                .map_err(interpolation_error)?;
            let converged = outcome.termination == InterpolationTermination::Converged;
            if !converged
                && !self
                    .policy
                    .capped_eligible(outcome.termination, active.len())
            {
                break (outcome, None);
            }
            let subdomain = self.checked_subdomain(&outcome, &fixed, projector, ctx)?;
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
            let measured = self.measure(tally, &sampler, &target, plan, false, projector, ctx)?;
            if measured.measurement.rms <= tolerance {
                record.audit = self.audit(
                    tally,
                    &sampler,
                    &target,
                    &measured.measurement,
                    &seeds,
                    projector,
                    ctx,
                )?;
                record.acceptance = Some(measured.measurement);
                return Ok(accepted(record, subdomain, patch_points));
            }
            Tally::add(&tally.verification_failures, 1);
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
                Tally::add(&tally.engine_retries, 1);
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
        self.resolve(context, last, sampler, tally, projector, ctx)
    }

    /// The layout-checked, re-embedded network of an outcome the driver may
    /// use. A `Converged` outcome must be strictly below the bond cap (the
    /// M1 contract); any other outcome must not exceed it.
    fn checked_subdomain(
        &self,
        outcome: &InterpolationOutcome<V>,
        fixed: &[Option<usize>],
        projector: &Projector,
        ctx: &mut EvalContext<'_>,
    ) -> Result<SubDomainTreeTN<V>, PatchedInterpolationError> {
        embed::check_outcome_layout(&outcome.network, &self.layout, fixed)
            .map_err(|message| engine_error(projector, message))?;
        let subdomain = self.subdomain(&outcome.network, fixed, projector, ctx)?;
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
    #[allow(clippy::too_many_arguments)]
    fn resolve(
        &self,
        context: PatchContext<'_>,
        last: Unaccepted<V>,
        sampler: PatchSampler<'_, T>,
        tally: &Tally,
        projector: &Projector,
        ctx: &mut EvalContext<'_>,
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
            return self.retain_blocked(&context, last, &sampler, tally, projector, ctx);
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
        let mut children = sampling::reserve_vec(layout.dims[split_position], "patch children")
            .map_err(invalid)?;
        let caches = sampler.cache.into_inner().split(slot).map_err(invalid)?;
        for (value, cache) in caches.into_iter().enumerate() {
            let mut child_path = path.clone();
            child_path.push((split_position, value));
            let mut child_fixed = fixed.to_vec();
            child_fixed[split_position] = Some(value);
            children.push(Patch {
                path: child_path,
                fixed: child_fixed,
                cache,
                recycled: inside(&recycled, value),
                worst: inside(&worst, value),
            });
        }
        Ok(Verdict::Split(children))
    }

    /// Evaluate a patch with at most one active site on all its points and
    /// build its network without the engine.
    #[allow(clippy::too_many_arguments)]
    fn exact_patch(
        &self,
        pins: &mut Pins<'_>,
        evaluate: &dyn Evaluate<T>,
        tally: &Tally,
        fixed: &[Option<usize>],
        active: &[usize],
        cache: PatchCache<T>,
        projector: &Projector,
        ctx: &mut EvalContext<'_>,
    ) -> Result<Verdict<T, V>, PatchedInterpolationError> {
        // Every point of the patch: `0..d` for one active site of dimension
        // `d`, or the single empty point when no site is active.
        let n_points: usize = active.iter().map(|&p| self.layout.dims[p]).product();
        let active_dims: Vec<usize> = active.iter().map(|&p| self.layout.dims[p]).collect();
        let points = sampling::all_points(&active_dims, n_points).map_err(invalid)?;
        let shape = [active.len(), n_points];
        let sampler = self.sampler(evaluate, tally, fixed, active.len(), cache);
        let batch = ColMajorArrayRef::new(&points, &shape).map_err(internal)?;
        let values = sampler
            .sample(batch, ctx)
            .map_err(|source| evaluator_error(projector, source))?;
        let largest = max_magnitude(&values);
        let patch_points = n_points as f64;
        let acceptance = if self.is_l2() {
            // Only the root can be exact before the reference is pinned; the
            // network multiplies the exact values by one-hot factors, so the
            // error is exactly zero and no measurement runs.
            pins.pin_l2(self.l2_pin(
                Some(verify::rms_of(values.iter().map(|value| value.abs_val()))),
                L2ReferenceSource::ExactRoot,
            ))?;
            Some(L2Measurement {
                method: MeasurementMethod::Exact,
                points: n_points,
                patch_points,
                rms: 0.0,
                mean_square_rel_std_error: 0.0,
                max_residual: 0.0,
            })
        } else {
            if pins.max_reference().is_none() {
                pins.pin_max_reference((largest, MaxReferenceSource::ExactRoot))?;
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
        let subdomain = self.subdomain(&network, fixed, projector, ctx)?;
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
        ctx: &mut EvalContext<'_>,
    ) -> Result<SubDomainTreeTN<V>, PatchedInterpolationError> {
        let data = embed::embed_fixed_sites::<T, V>(network, &self.layout, fixed, ctx.session())?;
        Ok(SubDomainTreeTN::from_masked_data(
            data,
            projector.clone(),
            None,
        )?)
    }

    /// The norm part of the report. Under L2 the global error and the
    /// approximation norm are combined in canonical path order; the assembly
    /// is arithmetic only, because every patch measured its own network on the
    /// session that built it.
    #[allow(clippy::type_complexity)]
    fn norm_report(
        &self,
        references: &References,
        accepted: &[(Vec<(usize, usize)>, Accepted<V>)],
        zeros: &[(Vec<(usize, usize)>, ZeroPatchRecord, f64)],
    ) -> Result<NormReport, PatchedInterpolationError> {
        if !self.is_l2() {
            let (max_reference, source) = references
                .max_reference
                .unwrap_or((0.0, MaxReferenceSource::ExactRoot));
            return Ok(NormReport::SampledMax {
                max_reference,
                source,
                engine_tolerance: self.options.tolerance.allowance(max_reference),
            });
        }
        let pin = references
            .l2
            .ok_or_else(|| internal(anyhow::anyhow!("the L2 reference is not pinned")))?;
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

        let approximations = accepted
            .iter()
            .map(|(_, entry)| (entry.log_norm, entry.patch_points));
        let approximation_rms = verify::approximation_rms(approximations, domain_points);
        // The rounding model scales with the coarsest precision the patch
        // networks were evaluated in.
        let mut epsilon = f64::EPSILON;
        for (_, entry) in accepted {
            let kind = entry
                .subdomain
                .scalar_kind()
                .map_err(|source| PatchedInterpolationError::Partition { source })?;
            epsilon = epsilon.max(kind.map_or(f64::EPSILON, ScalarKind::epsilon));
        }
        let (global, certified_fraction) = verify::global_error(
            &contributions,
            domain_points,
            pin.tau,
            approximation_rms,
            epsilon,
        );
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
        log_norm: None,
    }))
}

/// The root patch: the whole domain, with no fixed site.
fn root_patch<T, V>(layout: &SiteLayout<V>) -> Patch<T>
where
    T: CommonScalar,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    Patch {
        path: Vec::new(),
        fixed: vec![None; layout.sites.len()],
        cache: PatchCache::new(layout.dims.clone()),
        recycled: Vec::new(),
        worst: Vec::new(),
    }
}

/// Process one patch: everything a run does per patch, on the session it is
/// given.
///
/// This is the body every entry shares: the serial driver calls it on the
/// caller's session, a lane calls it on the lane's session, and an MPI rank
/// calls it on the session behind its own lane. It owns one patch's counters
/// and never touches another patch's state, so a wave of patches is
/// independent work.
#[allow(clippy::too_many_arguments)]
fn patch_outcome<T, V, E>(
    plan: &Plan<'_, T, V, E>,
    mut pins: Pins<'_>,
    evaluate: &dyn Evaluate<T>,
    patch: Patch<T>,
    ctx: &mut EvalContext<'_>,
) -> Result<PatchRun<T, V>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
{
    let tally = Tally::default();
    let path = patch.path.clone();
    let projector = plan.layout.projector(&path)?;
    let mut verdict = plan.process(patch, &projector, &mut pins, evaluate, &tally, ctx)?;
    if plan.is_l2() {
        if let Verdict::Accepted(entry) = &mut verdict {
            // The log-norm of the stored network is measured here, on the
            // session that built the patch: the final report is then pure
            // arithmetic and needs no session of its own.
            let mut data = entry.subdomain.data().clone();
            entry.log_norm = data.log_norm_on_session(ctx.session()).ok();
        }
    }
    Ok(PatchRun {
        path,
        verdict,
        totals: tally.totals(),
        references: pins.snapshot(),
    })
}

/// The driver's decision after one wave, which a distributed dispatcher must
/// agree on collectively.
#[derive(Clone, Debug)]
enum WaveControl {
    /// Another wave follows.
    Continue,
    /// The queue is empty: assemble the result.
    Stop,
    /// The run failed; every rank reports this message.
    Fail(String),
}

/// How the wave loop runs its waves.
///
/// The serial entry runs every wave on the caller's session; the lanes entry
/// spreads a wave over a scheduler's lanes; the MPI entry distributes a wave
/// over ranks. All three consume and produce the same records, so the patch
/// body and the wave merge have one implementation.
trait Waves<T, V, E>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    /// Run one wave and return one outcome per patch, in wave order.
    ///
    /// Non-root participants of a distributed run receive an empty wave and
    /// must still take part in the collective; only the root's outcomes enter
    /// the wave merge.
    fn run_wave(
        &mut self,
        plan: &Plan<'_, T, V, E>,
        patches: Vec<Patch<T>>,
        pins: WavePins<'_>,
    ) -> Result<Vec<PatchRun<T, V>>, PatchedInterpolationError>;

    /// Agree on `control` after a wave.
    ///
    /// Returns the control every participant observed and the references every
    /// participant uses for the next wave: a distributed dispatcher broadcasts
    /// the root's control and its pinned references, a local one returns them.
    fn agree(
        &mut self,
        control: WaveControl,
        references: References,
    ) -> Result<(WaveControl, References), PatchedInterpolationError>;
}

/// Merge one wave's outcomes in canonical path order.
///
/// Counters are summed, references are pinned (wave 0 only), accepted and zero
/// patches are collected, and the children of the wave's splits form the next
/// wave, again in canonical path order so that the schedule is a function of
/// the problem.
#[allow(clippy::type_complexity)]
fn merge_wave<T, V>(
    runs: Vec<PatchRun<T, V>>,
    references: &mut References,
    totals: &mut Totals,
    accepted: &mut Vec<AcceptedPatch<V>>,
    zeros: &mut Vec<ZeroPatch>,
    splits: &mut usize,
) -> Result<Vec<Patch<T>>, PatchedInterpolationError>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    let mut runs = runs;
    // Canonical order: lexicographic over the (position, coordinate) path.
    runs.sort_by(|left, right| left.path.cmp(&right.path));
    let mut next = Vec::new();
    for run in runs {
        totals.add(run.totals);
        references.merge(run.references)?;
        let PatchRun { path, verdict, .. } = run;
        match verdict {
            Verdict::Accepted(entry) => accepted.push((path, *entry)),
            Verdict::Zero(record, patch_points) => zeros.push((path, record, patch_points)),
            Verdict::Split(children) => {
                *splits += 1;
                next.extend(children);
            }
        }
    }
    next.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(next)
}

/// Everything a finished wave loop produced, before the report is assembled.
///
/// A distributed run merges on its root only, so the merge and the assembly are
/// separate steps: every participant runs the waves, and only the root turns the
/// merged records into a result.
struct RunMerged<T, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    references: References,
    totals: Totals,
    accepted: Vec<AcceptedPatch<V>>,
    zeros: Vec<ZeroPatch>,
    splits: usize,
    scalar: PhantomData<T>,
}

/// Assemble the result of a finished run from its wave merges.
fn assemble<T, V, E>(
    plan: &Plan<'_, T, V, E>,
    merged: RunMerged<T, V>,
) -> Result<PatchedInterpolationResult<V>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
{
    let RunMerged {
        references,
        totals,
        mut accepted,
        mut zeros,
        splits,
        scalar: _,
    } = merged;
    // Canonical order: lexicographic over the (position, coordinate) path.
    accepted.sort_by(|left, right| left.0.cmp(&right.0));
    zeros.sort_by(|left, right| left.0.cmp(&right.0));
    let norm = plan.norm_report(&references, &accepted, &zeros)?;
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
            tolerance: plan.options.tolerance,
            norm,
            accepted: records,
            zero_patches: zeros.into_iter().map(|(_, record, _)| record).collect(),
            splits,
            function_evaluations: totals.evaluations,
            cache_hits: totals.cache_hits,
            measurement_evaluations: totals.measurement_evaluations,
            audit_evaluations: totals.audit_evaluations,
            verification_failures: totals.verification_failures,
            engine_retries: totals.engine_retries,
        },
    })
}

/// Run the patch waves of a validated plan through `waves`.
///
/// Waves are the BFS generations of the patch queue. One wave is dispatched at
/// a time, its outcomes are merged in canonical path order, and the splits of
/// the wave form the next one. The `max_patches` allowance is checked before a
/// wave is dispatched: at most the remaining number of patches run (in
/// canonical order) and the run then reports the same
/// [`PatchedInterpolationError::ResourceLimit`] as a per-patch check.
fn run_waves<T, V, E, W>(
    plan: &Plan<'_, T, V, E>,
    waves: &mut W,
) -> Result<RunMerged<T, V>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
    W: Waves<T, V, E>,
{
    let mut references = plan.initial_references();
    let mut wave: Vec<Patch<T>> = vec![root_patch(&plan.layout)];
    let mut wave_index = 0usize;
    let mut processed = 0usize;
    let mut splits = 0usize;
    let mut totals = Totals::default();
    let mut accepted = Vec::new();
    let mut zeros = Vec::new();
    loop {
        // The allowance is checked before the wave is dispatched, so a wave
        // that does not fit runs its remaining patches and then fails.
        let mut resource_limit = None;
        if let Some(limit) = plan.options.max_patches {
            let remaining = limit.saturating_sub(processed);
            if wave.len() > remaining {
                wave.truncate(remaining);
                resource_limit = Some(limit);
            }
        }
        processed = processed.saturating_add(wave.len());
        let runs = {
            let pins = if wave_index == 0 {
                WavePins::Root(&mut references)
            } else {
                WavePins::Later(&references)
            };
            waves.run_wave(plan, std::mem::take(&mut wave), pins)?
        };
        let merged = merge_wave(
            runs,
            &mut references,
            &mut totals,
            &mut accepted,
            &mut zeros,
            &mut splits,
        );
        let mut next = match merged {
            Ok(next) => next,
            Err(error) => {
                // A distributed dispatcher has to agree before a root-side
                // failure returns here; a local one returns the control as it
                // received it.
                waves.agree(WaveControl::Fail(error.to_string()), references)?;
                return Err(error);
            }
        };
        wave_index += 1;
        let typed = resource_limit.map(|limit| PatchedInterpolationError::ResourceLimit {
            resource: "max_patches",
            limit,
        });
        let control = match &typed {
            Some(error) => WaveControl::Fail(error.to_string()),
            None if next.is_empty() => WaveControl::Stop,
            None => WaveControl::Continue,
        };
        let (control, agreed) = waves.agree(control, references)?;
        references = agreed;
        if wave_index == 1 && !plan.is_pinned(&references) {
            return Err(internal(anyhow::anyhow!(
                "the root wave pinned no reference for a run that needs one"
            )));
        }
        match control {
            WaveControl::Continue => wave = std::mem::take(&mut next),
            WaveControl::Stop => break,
            WaveControl::Fail(message) => {
                return Err(typed.unwrap_or(PatchedInterpolationError::FanOut { message }));
            }
        }
    }
    Ok(RunMerged {
        references,
        totals,
        accepted,
        zeros,
        splits,
        scalar: PhantomData,
    })
}

/// The serial driver: every wave runs on the caller's session, one patch after
/// another.
struct SerialWaves<'a, 'ctx, F> {
    ctx: &'a mut EvalContext<'ctx>,
    /// The caller's evaluator, lent to one patch at a time.
    evaluate: RefCell<F>,
}

impl<T, V, E, F> Waves<T, V, E> for SerialWaves<'_, '_, F>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
    F: FnMut(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>>,
{
    fn run_wave(
        &mut self,
        plan: &Plan<'_, T, V, E>,
        patches: Vec<Patch<T>>,
        mut pins: WavePins<'_>,
    ) -> Result<Vec<PatchRun<T, V>>, PatchedInterpolationError> {
        let evaluate = Exclusive(&self.evaluate);
        let mut runs = Vec::with_capacity(patches.len());
        for patch in patches {
            let pins = pins.patch();
            runs.push(patch_outcome(plan, pins, &evaluate, patch, self.ctx)?);
        }
        Ok(runs)
    }

    fn agree(
        &mut self,
        control: WaveControl,
        references: References,
    ) -> Result<(WaveControl, References), PatchedInterpolationError> {
        Ok((control, references))
    }
}
