//! TreeTN-general adaptive patching.
//!
//! The partition representation follows the independent Rust successor's
//! relationship to [PartitionedMPSs.jl](https://github.com/tensor4all/PartitionedMPSs.jl).
//! The default (volume-proportional, absolute squared-tail) policy follows the
//! published method “Adaptive Patching for Tensor Train Computations”
//! ([arXiv:2602.22372](https://arxiv.org/abs/2602.22372));
//! [`BudgetAllocation::Norm`] instead applies the local relative squared-tail
//! policy of [`TreeTN::truncate`](tensor4all_treetn::TreeTN::truncate). This
//! module does not implement adaptive interpolation or copy TCIAlgorithms.jl
//! code.
//! Local rank-triggered stopping is motivated by Sec. 4.3 of that paper. The
//! lazy group loop independently preserves the existing per-contribution
//! split heuristic; it does not infer a final sum's rank from one term.

mod report;

pub use report::{PatchCapRecord, PatchingReport};

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{PartitionedTreeTNError, Result};
use crate::partitioned_tree_tn::PartitionedTreeTN;
use crate::projector::Projector;
use crate::subdomain_tree_tn::{
    contraction_output_site_indices, ensure_center, sort_by_patch_order,
    validate_contraction_options, PatchOrderKey, ScalarKind, SubDomainContractionOutcome,
    SubDomainTreeTN,
};
use report::record_patch;
use tensor4all_core::{validate_svd_truncation_options, SvdTruncationPolicy};
use tensor4all_treetn::{contraction::ContractionOptions, SumEngine, TruncationOptions};

mod contraction_plan;

/// Records of one closing truncation, in the order the patches were retained.
type CapRecords<V> = Vec<(PatchOrderKey<V>, PatchCapRecord)>;

#[derive(Debug)]
struct PatchStats<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    subdomain: SubDomainTreeTN<V>,
    volume: usize,
    norm_squared: f64,
}

/// Strategy used to choose the next full site index for patch splitting.
///
/// `Sequential` takes the first available index in `PatchingOptions::patch_order`.
/// `ExactParameterGain` forms, budget-truncates, and logically counts every
/// candidate's children, selecting the candidate with the smallest total.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::PatchSplitStrategy;
///
/// assert_eq!(PatchSplitStrategy::default(), PatchSplitStrategy::ExactParameterGain);
/// assert_ne!(PatchSplitStrategy::Sequential, PatchSplitStrategy::ExactParameterGain);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PatchSplitStrategy {
    /// Split the first available index from the explicit patch order.
    Sequential,
    /// Select the split with the smallest checked logical child parameter count.
    #[default]
    ExactParameterGain,
}

/// How the dimensionless [`PatchingOptions::cutoff`] is allocated to patches.
///
/// Both rules distribute the same total weight, because patch supports are
/// disjoint (`Σ V_p = V` and `Σ ||F_p||² = ||F||²`); only the distribution
/// differs. `cutoff` is therefore a cross-patch budget under `Volume`, not an
/// SVD truncation knob; use [`SvdTruncationPolicy`](tensor4all_core::SvdTruncationPolicy)
/// or `TreeTN` options for the latter.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::{BudgetAllocation, PatchingOptions};
///
/// assert_eq!(BudgetAllocation::default(), BudgetAllocation::Volume);
/// assert_eq!(PatchingOptions::default().allocation, BudgetAllocation::Volume);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum BudgetAllocation {
    /// Uniform absolute accuracy over the domain: patch `p` gets the absolute
    /// squared-weight budget `cutoff * ||F||² * V_p / V` and is dropped when
    /// `||F_p||²` is at most that budget.
    ///
    /// This is the default and follows arXiv:2602.22372. Small-magnitude,
    /// large-volume regions are truncated hard or dropped, while a sharp peak
    /// keeps a high relative precision and drives splits around itself.
    #[default]
    Volume,
    /// Uniform relative accuracy per patch: every patch is truncated with the
    /// actual relative squared tail-sum policy (threshold `cutoff`) of
    /// [`TreeTN::truncate`](tensor4all_treetn::TreeTN::truncate), evaluated
    /// separately at each local SVD.
    ///
    /// No patch is dropped, and no absolute budget is inherited by children:
    /// split candidates are truncated with the same relative policy. This is
    /// best effort for the whole network, like `Volume`.
    Norm,
}

/// Options for TreeTN-general adaptive patching and truncation.
///
/// The scalar truncation surface follows the ITensorMPS `cutoff` convention: a
/// dimensionless, non-negative global **discarded-weight** budget shared by
/// the patches of one operation. [`BudgetAllocation`] selects how that budget
/// is spread over the patches. Under the default [`BudgetAllocation::Volume`]
/// each patch receives the absolute squared-tail budget
/// `local_cutoff_p = cutoff * ||F||² * volume_p / total_volume`; under
/// [`BudgetAllocation::Norm`] every local SVD instead applies the relative
/// squared tail-sum cutoff `cutoff` of `TreeTN::truncate`. Either way `cutoff`
/// is a cross-patch budget, not an SVD truncation-policy knob, and is best
/// effort for the final whole-network error.
///
/// `max_bond_dim` is the post-truncation hard cap; `None` leaves the bond
/// dimension uncapped. An empty `patch_order` lets adaptive patching consider
/// all unprojected site indices in deterministic full-index order. A nonempty
/// order may be partial, but every entry must be an exact full site-index
/// identity of the input TreeTN with a compatible dimension.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::{
///     BudgetAllocation, PatchSplitStrategy, PatchingOptions, SumEngine,
/// };
///
/// let options = PatchingOptions::default();
/// assert_eq!(options.cutoff, 1.0e-24);
/// assert_eq!(options.allocation, BudgetAllocation::Volume);
/// assert_eq!(options.max_bond_dim, Some(100));
/// assert!(options.patch_order.is_empty());
/// assert_eq!(options.split_strategy, PatchSplitStrategy::ExactParameterGain);
/// assert_eq!(options.sum_engine, SumEngine::Binary);
/// assert_eq!(options.group_sum_limit_bytes, 4 * 1024 * 1024 * 1024);
/// ```
#[derive(Debug, Clone)]
pub struct PatchingOptions {
    /// Dimensionless, non-negative discarded-weight budget shared across the
    /// patches of one operation.
    ///
    /// Under [`BudgetAllocation::Volume`] the whole local threshold
    /// `cutoff * ||F||² * volume_p / total_volume` is derived per operation
    /// from the operation's input norm `||F||` and applied independently at
    /// every local SVD of the patch. Under [`BudgetAllocation::Norm`] each
    /// local SVD uses the relative squared tail-sum cutoff `cutoff` instead.
    /// Smaller values retain more information. `cutoff = 0` disables threshold
    /// truncation, leaving only the hard `max_bond_dim` cap. This is best
    /// effort for the final whole-network error: reuse across several bonds or
    /// repeated truncation stages does not provide a global error bound.
    pub cutoff: f64,

    /// Rule that allocates [`Self::cutoff`] to patches.
    ///
    /// Defaults to [`BudgetAllocation::Volume`], which preserves the published
    /// volume-proportional numerics. Use [`BudgetAllocation::Norm`] for a
    /// per-patch relative accuracy that matches `TreeTN::truncate`'s relative
    /// cutoff locally.
    pub allocation: BudgetAllocation,

    /// Optional maximum retained bond dimension. `Some(0)` is invalid.
    ///
    /// This is a hard cap and takes precedence over `cutoff`. When the
    /// budget-truncated probe of a patch still exceeds this cap, adaptive
    /// patching splits the exact patch if an unprojected index is available.
    /// A patch that still exceeds the cap when no split candidate remains is
    /// truncated to the cap by the closing truncation.
    ///
    /// The result cannot tell a patch that the cap truncated from one whose
    /// exact rank equals the cap. [`add_with_patching_with_report`] and
    /// [`contract_adaptive_with_report`] name the patches the cap truncated,
    /// with their bond dimensions ([`PatchCapRecord`]); the functions without
    /// `_with_report` keep their signature and return no report.
    pub max_bond_dim: Option<usize>,

    /// Full site-index order used by [`PatchSplitStrategy::Sequential`].
    ///
    /// Entries are matched by complete index identity, including tags and
    /// prime level; an entry absent from the TreeTN is rejected.
    /// An empty order uses the completed result's node/site order. Adaptive
    /// Sequential contraction then completes its first contribution before
    /// selecting a candidate; an explicit order permits probing from the first pair.
    pub patch_order: Vec<tensor4all_core::DynIndex>,

    /// Candidate-selection strategy used for over-cap patches.
    pub split_strategy: PatchSplitStrategy,

    /// How exact sums of several patches are built: the equal-key patches
    /// [`add_with_patching`] combines, and the contributions to one output
    /// projector that [`contract_adaptive`] sums before its probe truncation.
    ///
    /// Defaults to [`SumEngine::Binary`], which adds the patches one by one and
    /// assumes few patches per sum. [`SumEngine::Nary`] builds each sum in one
    /// pass, much faster for many patches, and [`SumEngine::NaryWithin`] does
    /// so only while the estimated size of the built sum fits `limit_bytes`.
    /// The estimate is
    /// [`ExactSum::estimated_nary_bytes`](tensor4all_treetn::ExactSum::estimated_nary_bytes):
    /// for every node, the product of its site dimensions and of its summed
    /// bond dimensions, times 16 bytes. The engine changes only how a sum is
    /// built, not its value, the truncation that follows, or the split rule.
    ///
    /// A `NaryWithin` fallback to the add chain is not reported by these
    /// routines, and the [`PatchingReport`] of the `_with_report` variants
    /// covers only the bond cap. To see which method a given sum takes,
    /// measure it with
    /// [`TreeTN::exact_sum`](tensor4all_treetn::TreeTN::exact_sum) or
    /// [`PartitionedTreeTN::to_treetn_with`]. A reporting variant for the
    /// engine may be added when a caller needs it.
    pub sum_engine: SumEngine,

    /// Largest estimated size, in bytes, of one exact group sum: the
    /// contributions to one output projector in [`contract_adaptive`], or the
    /// equal-key input patches in [`add_with_patching`].
    ///
    /// The estimate is
    /// [`TreeTN::estimate_exact_sum_bytes`](tensor4all_treetn::TreeTN::estimate_exact_sum_bytes),
    /// taken before the sum is formed; a larger sum is refused with
    /// [`PartitionedTreeTNError::GroupSumLimit`] instead of being allocated.
    /// The estimate counts the built sum only. The add chain of
    /// [`SumEngine::Binary`] holds two partial sums at its last add, and the
    /// truncation that follows needs a transient on top of the sum (see
    /// [`SumEngine`]), so the peak can be several times the estimate.
    ///
    /// Defaults to [`Self::DEFAULT_GROUP_SUM_LIMIT_BYTES`] (4 GiB). A group of
    /// one patch is never checked. `usize::MAX` disables the check.
    pub group_sum_limit_bytes: usize,
}

impl PatchingOptions {
    /// The default [`Self::group_sum_limit_bytes`]: 4 GiB.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::PatchingOptions;
    ///
    /// assert_eq!(PatchingOptions::DEFAULT_GROUP_SUM_LIMIT_BYTES, 4 << 30);
    /// ```
    pub const DEFAULT_GROUP_SUM_LIMIT_BYTES: usize = 4 << 30;
}

impl Default for PatchingOptions {
    fn default() -> Self {
        Self {
            // Behavior-parity translation of the superseded global `rtol =
            // 1e-12` via `cutoff = old_rtol^2`; see the design record.
            cutoff: 1.0e-24,
            allocation: BudgetAllocation::Volume,
            max_bond_dim: Some(100),
            patch_order: Vec::new(),
            split_strategy: PatchSplitStrategy::default(),
            sum_engine: SumEngine::Binary,
            group_sum_limit_bytes: Self::DEFAULT_GROUP_SUM_LIMIT_BYTES,
        }
    }
}

/// Add subdomains with automatic TreeTN patch splitting.
///
/// Input patches that share an equal projector key are first summed by strict
/// subdomain addition, then the result is validated as one homogeneous,
/// exact-topology partition. Over-cap patches are split along full site
/// indices and retried until no permitted split remains. The explicit `center`
/// is used for every TreeTN truncation and is never stored in the result.
///
/// # Project-first splits
///
/// As in [`contract_adaptive`], a split is only probed on a truncated value;
/// its children come from the exact addends. The probe is the patch with its
/// cutoff budget applied and no bond cap, because a capped probe cannot tell an
/// exact rank from a truncated one; it decides only whether and where to split.
/// Each child is the projection of the exact parent, compressed at a
/// dtype-scaled round-off threshold (`64 * epsilon`) so projection-created null
/// bond space is removed without spending the caller's budget. The cutoff
/// budget and the bond cap are therefore spent once, on the final leaves,
/// rather than once per split level with every level's loss inherited by all
/// leaves below it. A leaf that is already below its budget still reaches that
/// closing pass, which drops it, so the other leaves' budgets stay fractions of
/// the whole input weight.
///
/// # Arguments
///
/// * `subdomains` - Eagerly masked patches with mutually disjoint projectors.
/// * `center` - Existing named TreeTN node used as every truncation center.
/// * `options` - Discarded-weight budget and its [`BudgetAllocation`], bond
///   cap, split order, and split strategy.
///
/// # Returns
///
/// A new partition containing the retained, budget-truncated leaves. The split
/// loop stops when no unprojected split candidate remains; the final adaptive
/// truncation still applies `options.max_bond_dim` to every retained leaf. The
/// whole-network error is best effort and is not bounded by `options.cutoff`:
/// one local budget is reused at every bond of a leaf's truncation sweep.
///
/// A leaf that the cap truncated is not visible in this return value, because
/// its bonds then sit exactly at the cap like an exact rank would.
/// [`add_with_patching_with_report`] runs the same computation and reports
/// which leaves the cap truncated ([`PatchCapRecord`]).
///
/// # Errors
///
/// Returns [`PartitionedTreeTNError::InvalidOptions`] for invalid cutoff,
/// cap, or split-order options; [`PartitionedTreeTNError::GroupSumLimit`] when
/// the exact sum of the patches sharing one projector key would exceed
/// `options.group_sum_limit_bytes`; [`PartitionedTreeTNError::Empty`] only when a
/// nonempty operand is required by a downstream TreeTN operation;
/// [`PartitionedTreeTNError::InvalidCenter`] when `center` is absent;
/// [`PartitionedTreeTNError::TopologyMismatch`],
/// [`PartitionedTreeTNError::SiteIndexMismatch`], or
/// [`PartitionedTreeTNError::DTypeMismatch`] for inconsistent inputs;
/// [`PartitionedTreeTNError::VolumeOverflow`] or
/// [`PartitionedTreeTNError::LogicalParameterCountOverflow`] for checked
/// arithmetic overflow; and typed TreeTN/backend errors for masking or
/// truncation failures. No input patch is mutated.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     add_with_patching, PatchSplitStrategy, PatchingOptions, SubDomainTreeTN,
/// };
/// use tensor4all_treetn::TreeTN;
///
/// let site0 = DynIndex::new_dyn(2);
/// let bond = DynIndex::new_dyn(2);
/// let site1 = DynIndex::new_dyn(2);
/// let left = IdxTensor::from_dense(
///     vec![site0.clone(), bond.clone()],
///     vec![1.0_f64, 0.0, 0.0, 1.0],
/// )?;
/// let right = IdxTensor::from_dense(
///     vec![bond, site1],
///     vec![1.0_f64, 0.0, 0.0, 1.0],
/// )?;
/// let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?;
/// let patch = SubDomainTreeTN::from_treetn(tree)?;
/// let result = add_with_patching(
///     vec![patch],
///     &0,
///     &PatchingOptions {
///         cutoff: 0.0,
///         max_bond_dim: Some(1),
///         patch_order: vec![site0],
///         split_strategy: PatchSplitStrategy::Sequential,
///         ..PatchingOptions::default()
///     },
/// )?;
/// assert_eq!(result.len(), 2);
/// assert!(result.values().all(|patch| patch.max_bond_dim() <= 1));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn add_with_patching<V>(
    subdomains: Vec<SubDomainTreeTN<V>>,
    center: &V,
    options: &PatchingOptions,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    add_with_patching_inner(subdomains, center, options, None)
}

/// Add subdomains with automatic TreeTN patch splitting, and report what the
/// bond cap truncated.
///
/// The computation, the arguments, the errors and the returned partition are
/// exactly those of [`add_with_patching`]; see there for the split rules. This
/// variant additionally returns a [`PatchingReport`] with one [`PatchCapRecord`]
/// per returned patch, so a caller can tell a patch that `options.max_bond_dim`
/// truncated from one whose exact rank equals the cap.
///
/// A patch is reported as cap-truncated when the same cutoff applied without
/// the cap retains a larger bond dimension while one of the returned patch's
/// bonds sits exactly at the cap, which is the case in which the split loop
/// ran out of `patch_order` indices. A patch whose exact rank equals the cap,
/// and one that the cutoff alone brought below it, are not reported. Reports
/// are in canonical patch order. The discarded weight is not reported: the
/// truncation returns no singular values, so it would need a second
/// decomposition of the patch. Recording costs one extra truncation of each
/// patch that keeps a bond exactly at the cap; the plain function pays
/// nothing.
///
/// # Errors
///
/// Returns the errors of [`add_with_patching`]; the report adds none.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     add_with_patching_with_report, PatchSplitStrategy, PatchingOptions, SubDomainTreeTN,
/// };
/// use tensor4all_treetn::TreeTN;
///
/// // A chain over bits `a, b | c` whose bond has exact rank 2 after `a` is
/// // fixed, so a cap of 1 truncates both patches with no index left to split.
/// let (a, b, c) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2), DynIndex::new_dyn(2));
/// let bond = DynIndex::new_dyn(2);
/// let left = IdxTensor::from_dense(
///     vec![a.clone(), b, bond.clone()],
///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
/// )?;
/// let right = IdxTensor::from_dense(vec![bond, c], vec![1.0_f64, 0.0, 0.0, 1.0])?;
/// let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?;
/// let options = PatchingOptions {
///     cutoff: 0.0,
///     max_bond_dim: Some(1),
///     patch_order: vec![a],
///     split_strategy: PatchSplitStrategy::Sequential,
///     ..PatchingOptions::default()
/// };
/// let (result, report) =
///     add_with_patching_with_report(vec![SubDomainTreeTN::from_treetn(tree)?], &0, &options)?;
/// assert_eq!(result.len(), 2);
/// assert_eq!(report.truncated_by_cap().count(), 2);
/// assert!(report.patches.iter().all(|record| record.bond_dims == vec![1]));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn add_with_patching_with_report<V>(
    subdomains: Vec<SubDomainTreeTN<V>>,
    center: &V,
    options: &PatchingOptions,
) -> Result<(PartitionedTreeTN<V>, PatchingReport)>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut records = Vec::new();
    let truncated = add_with_patching_inner(subdomains, center, options, Some(&mut records))?;
    Ok((truncated, PatchingReport::new(records)))
}

/// The body of [`add_with_patching_with_report`], with an optional sink for the
/// cap records. `None` skips the extra truncation that decides the flag, so the
/// plain entry point really does pay nothing for the report.
fn add_with_patching_inner<V>(
    subdomains: Vec<SubDomainTreeTN<V>>,
    center: &V,
    options: &PatchingOptions,
    mut records: Option<&mut CapRecords<V>>,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    validate_patching_options(options)?;
    let subdomains = combine_equal_key_subdomains(subdomains, options)?;
    let partitioned = PartitionedTreeTN::from_subdomains(subdomains)?;
    if partitioned.is_empty() {
        return Ok(partitioned);
    }

    let template = partitioned
        .values()
        .next()
        .ok_or(PartitionedTreeTNError::Empty)?;
    ensure_center(template.data(), center)?;
    validate_patch_order(template, options)?;

    // Pin the reference squared norm and total volume once from the combined
    // input partition. Split and re-truncation stages reuse these two numbers so
    // under the `Volume` allocation `cutoff` keeps meaning "fraction of the
    // original weight" throughout the operation and child patches inherit
    // volume-proportional cutoffs. `Norm` needs neither value: every stage
    // uses the per-factorization relative policy instead.
    let reference = match options.allocation {
        BudgetAllocation::Volume => {
            let (_, total_volume, reference_norm_squared) = patch_stats_and_totals(&partitioned)?;
            Some((total_volume, reference_norm_squared))
        }
        BudgetAllocation::Norm => None,
    };

    // Project-first: the patches carried between iterations stay exact. The
    // budget-truncated form of a patch is only a split-decision probe; the
    // children of a split are projections of the exact patch. Truncation is
    // therefore applied once, to the final leaves.
    let mut working = partitioned
        .into_iter()
        .map(|(_, subdomain)| subdomain)
        .collect::<Vec<_>>();

    loop {
        if let Some((total_volume, reference_norm_squared)) = reference {
            working = assign_volume_budgets(
                working,
                options.cutoff,
                reference_norm_squared,
                total_volume,
            )?;
        }

        let mut next = Vec::new();
        let mut split_any = false;
        for subdomain in working {
            // A patch at or below its budget has no probe: it is never split
            // and stays in the result set, so the closing truncation still
            // derives every other leaf's budget from the whole input weight and
            // drops the patch itself.
            let probe = split_decision_probe(&subdomain, center, options)?;
            let over_cap = probe.as_ref().is_some_and(|probe| {
                options
                    .max_bond_dim
                    .is_some_and(|cap| probe.max_bond_dim() > cap)
            });
            let index = match probe {
                Some(probe) if over_cap => choose_split_index(&probe, center, options)?,
                _ => None,
            };
            let Some(index) = index else {
                next.push(subdomain);
                continue;
            };

            split_any = true;
            for value in 0..index.dim {
                if let Some(child) = project_if_present(&subdomain, &index, value, center)? {
                    next.push(child);
                }
            }
        }

        if !split_any {
            let partitioned = PartitionedTreeTN::from_subdomains(next)?;
            let truncated = truncate_adaptive_reporting(
                &partitioned,
                center,
                options.cutoff,
                options.allocation,
                options.max_bond_dim,
                records.as_deref_mut(),
            )?;
            return Ok(truncated);
        }
        working = next;
    }
}

/// Contract two partitions and adaptively truncate the output.
///
/// Contraction uses the already eager-masked patches and the supplied TreeTN
/// contraction options. Compatible pairs are grouped by predicted output
/// projector before numerical work, in logical left-then-right patch order.
/// Unequal overlapping output keys are refined onto a disjoint value grid of
/// their fixed output indices before any masking or contraction, subject to
/// [`crate::AUTOMATIC_REFINEMENT_LIMIT`]. Operand restrictions are masked
/// exactly and cached per original patch; contributions retain pair order.
/// With a cap and [`PatchSplitStrategy::Sequential`], a group stops at its
/// first saturated contribution and retries projected original operands when
/// a split candidate exists. An explicit patch order permits the first pair
/// to use the TreeTN outcome hook; an empty order completes the first pair
/// to preserve its actual node/site order. Chain, Fit and randomized contractions
/// complete each attempted pair, but later pairs can still be skipped.
/// [`PatchSplitStrategy::ExactParameterGain`], an absent cap or an exhausted
/// split order needs all contributions. A completed group's exact sum is
/// truncated once, and the output is then retruncated with budgets computed
/// from the corrected output norm. Saturation is a refinement heuristic,
/// not an error certificate or a proof that the sum has the same rank.
///
/// # Global operators
///
/// A group holds every pair with one output projector, and its exact sum has
/// the sum of the contributions' bonds. An operator that mixes every value of
/// the indices the patches fix, such as a Fourier transform applied to patches
/// of the transformed variable, leaves no constraint on the output: every pair
/// lands in one group, and splitting on an output index does not reduce it.
/// That sum is checked against [`PatchingOptions::group_sum_limit_bytes`]
/// before it is formed. Apply such an operator with
/// [`schedule_merge_refine`](crate::reconstruction::schedule_merge_refine), or
/// the patched Fourier transform of `tensor4all-quanticstransform` built on
/// it, which never forms this sum.
///
/// # Arguments
///
/// * `left` - First validated partition.
/// * `right` - Second validated partition.
/// * `center` - Existing node used by contraction and adaptive truncation.
/// * `contract_options` - Non-dense TreeTN contraction method and rank policy.
/// * `patching_options` - Output discarded-weight budget and its allocation,
///   bond cap, and split options (`patch_order` candidates, `split_strategy`).
///
/// # Returns
///
/// A new partition containing compatible pairwise contractions and their
/// truncated output patches. The whole-network error is best effort and is not
/// bounded by `patching_options.cutoff`.
///
/// An output patch that the cap truncated is not visible in this return value.
/// [`contract_adaptive_with_report`] runs the same computation and reports
/// which patches the cap truncated ([`PatchCapRecord`]).
///
/// # Errors
///
/// Returns [`PartitionedTreeTNError::InvalidOptions`] for invalid contraction
/// or patching options or a forbidden contraction method,
/// [`PartitionedTreeTNError::TopologyMismatch`] or
/// [`PartitionedTreeTNError::SiteIndexMismatch`] for incompatible patches,
/// [`PartitionedTreeTNError::DTypeMismatch`] for mixed dtypes,
/// [`PartitionedTreeTNError::InvalidCenter`] when `center` is absent,
/// [`PartitionedTreeTNError::OverlappingProjectors`] for invalid operand layouts,
/// [`PartitionedTreeTNError::RefinementTooLarge`] when the overlapping outputs'
/// candidate grid exceeds [`crate::AUTOMATIC_REFINEMENT_LIMIT`],
/// [`PartitionedTreeTNError::GroupSumLimit`] when one group's exact sum would
/// exceed `patching_options.group_sum_limit_bytes`, and
/// [`PartitionedTreeTNError::TreeTN`],
/// [`PartitionedTreeTNError::VolumeOverflow`], or
/// [`PartitionedTreeTNError::NonFiniteAdaptiveValue`] from contraction or
/// output truncation. Errors in skipped numerical work are not evaluated.
/// Neither input is mutated.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     contract_adaptive, PatchingOptions, PartitionedTreeTN, SubDomainTreeTN,
/// };
/// use tensor4all_treetn::{contraction::ContractionOptions, TreeTN};
///
/// let site = DynIndex::new_dyn(2);
/// let make = |values| -> Result<_, Box<dyn std::error::Error>> {
///     let tensor = IdxTensor::from_dense(vec![site.clone()], values)?;
///     Ok(SubDomainTreeTN::from_treetn(TreeTN::from_tensors(
///         vec![tensor],
///         vec![0usize],
///     )?)?)
/// };
/// let left = PartitionedTreeTN::from_subdomain(make(vec![1.0_f64, 2.0])?)?;
/// let right = PartitionedTreeTN::from_subdomain(make(vec![3.0_f64, 4.0])?)?;
/// let result = contract_adaptive(
///     &left,
///     &right,
///     &0,
///     &ContractionOptions::default(),
///     &PatchingOptions {
///         cutoff: 0.0,
///         max_bond_dim: Some(1),
///         ..PatchingOptions::default()
///     },
/// )?;
/// assert_eq!(result.len(), 1);
/// assert!(result.values().next().ok_or("missing result")?.all_indices().is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_adaptive<V>(
    left: &PartitionedTreeTN<V>,
    right: &PartitionedTreeTN<V>,
    center: &V,
    contract_options: &ContractionOptions,
    patching_options: &PatchingOptions,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    contract_adaptive_with_selection(
        left,
        right,
        center,
        contract_options,
        patching_options,
        None,
        None,
    )
}

/// Contract two partitions and adaptively truncate the output, reporting what
/// the bond cap truncated.
///
/// The computation, the arguments, the errors and the returned partition are
/// exactly those of [`contract_adaptive`]. This variant additionally returns a
/// [`PatchingReport`] with one [`PatchCapRecord`] per returned output patch, so
/// a caller can tell a patch that `patching_options.max_bond_dim` truncated
/// from one whose exact rank equals the cap. A patch is reported as
/// cap-truncated when the same cutoff applied without the cap retains a larger
/// bond dimension while one of the returned patch's bonds sits exactly at the
/// cap; a patch whose exact rank equals the cap is not reported. The discarded
/// weight is not reported, because the truncation returns no singular values.
/// Recording costs one extra truncation of each patch that keeps a bond
/// exactly at the cap; the plain function pays nothing.
///
/// # Errors
///
/// Returns the errors of [`contract_adaptive`]; the report adds none.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     contract_adaptive_with_report, PatchingOptions, PartitionedTreeTN, SubDomainTreeTN,
/// };
/// use tensor4all_treetn::{contraction::ContractionOptions, TreeTN};
///
/// let site = DynIndex::new_dyn(2);
/// let make = |values| -> Result<_, Box<dyn std::error::Error>> {
///     let tensor = IdxTensor::from_dense(vec![site.clone()], values)?;
///     Ok(SubDomainTreeTN::from_treetn(TreeTN::from_tensors(
///         vec![tensor],
///         vec![0usize],
///     )?)?)
/// };
/// let left = PartitionedTreeTN::from_subdomain(make(vec![1.0_f64, 2.0])?)?;
/// let right = PartitionedTreeTN::from_subdomain(make(vec![3.0_f64, 4.0])?)?;
/// let (result, report) = contract_adaptive_with_report(
///     &left,
///     &right,
///     &0,
///     &ContractionOptions::default(),
///     &PatchingOptions {
///         cutoff: 0.0,
///         max_bond_dim: Some(1),
///         ..PatchingOptions::default()
///     },
/// )?;
/// // The contraction is a scalar patch: it has no bond for the cap to set.
/// assert_eq!(result.len(), 1);
/// assert_eq!(report.patches.len(), 1);
/// assert!(report.patches[0].bond_dims.is_empty());
/// assert!(!report.patches[0].cap_truncated);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_adaptive_with_report<V>(
    left: &PartitionedTreeTN<V>,
    right: &PartitionedTreeTN<V>,
    center: &V,
    contract_options: &ContractionOptions,
    patching_options: &PatchingOptions,
) -> Result<(PartitionedTreeTN<V>, PatchingReport)>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut records = Vec::new();
    let truncated = contract_adaptive_with_selection(
        left,
        right,
        center,
        contract_options,
        patching_options,
        None,
        Some(&mut records),
    )?;
    Ok((truncated, PatchingReport::new(records)))
}

/// Contract two partitions and adaptively truncate the output, forming only
/// the outputs whose projector is in `wanted`.
///
/// This is [`contract_adaptive`] restricted to a selection, as
/// [`PartitionedTreeTN::contract_selected`] restricts
/// [`PartitionedTreeTN::contract`]. A patch pair whose output projector is not
/// in `wanted` is skipped and never contracted. A selected output group is
/// split as in [`contract_adaptive`], so each returned patch has a wanted
/// projector or, after overlap resolution or a split, a refinement of one
/// ([`Projector::is_subset_of`] a wanted key).
/// Selection compares the original pair output keys before refinement. An
/// unselected group's contributions stay excluded even where its region
/// overlaps a selected key. Only selected keys enter the refinement-grid limit;
/// selecting all original keys gives the ordinary adaptive operation.
///
/// # Budget rule
///
/// The adaptive truncation budget is taken over the selected output only.
/// Under [`BudgetAllocation::Volume`], `||F||²` and the total volume in
/// `cutoff * ||F||² * volume_p / total_volume` are the totals of the selected
/// patches. A wanted block can therefore be truncated, or dropped, differently
/// from the same block in a [`contract_adaptive`] call over all outputs, and
/// differently from a call that selects it together with other blocks.
/// [`BudgetAllocation::Norm`] uses a per-patch relative policy and has no
/// totals, so its budgets do not depend on the selection.
///
/// # Arguments
///
/// * `left` - First validated partition.
/// * `right` - Second validated partition.
/// * `center` - Existing node used by contraction and adaptive truncation.
/// * `contract_options` - Non-dense TreeTN contraction method and rank policy.
/// * `patching_options` - Output discarded-weight budget and its allocation,
///   bond cap, and split options.
/// * `wanted` - Exact output projector keys, compared by equality with the
///   pre-split output key of each pair. Duplicates are ignored. A key that no
///   pair produces has no output; an empty slice returns an empty partition.
///   A pair whose key overlaps a wanted key without being equal to it is not
///   formed, so a wanted region holds only the pairs whose key equals it.
///
/// # Returns
///
/// A new partition containing the retained, truncated patches of the selected
/// outputs. The whole-network error is best effort and is not bounded by
/// `patching_options.cutoff`.
///
/// # Errors
///
/// Returns the errors of [`contract_adaptive`]:
/// [`PartitionedTreeTNError::InvalidOptions`] for invalid contraction or
/// patching options or a forbidden contraction method,
/// [`PartitionedTreeTNError::TopologyMismatch`] or
/// [`PartitionedTreeTNError::SiteIndexMismatch`] for incompatible patches,
/// [`PartitionedTreeTNError::DTypeMismatch`] for mixed dtypes,
/// [`PartitionedTreeTNError::InvalidCenter`] when `center` is absent,
/// [`PartitionedTreeTNError::OverlappingProjectors`] for invalid operand layouts,
/// [`PartitionedTreeTNError::RefinementTooLarge`] when the selected overlapping
/// outputs' candidate grid exceeds [`crate::AUTOMATIC_REFINEMENT_LIMIT`], and
/// [`PartitionedTreeTNError::TreeTN`],
/// [`PartitionedTreeTNError::VolumeOverflow`], or
/// [`PartitionedTreeTNError::NonFiniteAdaptiveValue`] from contraction or
/// output truncation. The operand checks run even when no pair is selected.
/// Errors in skipped numerical work are not evaluated. Neither input is mutated.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     contract_adaptive_selected, PatchingOptions, PartitionedTreeTN, Projector,
///     SubDomainTreeTN,
/// };
/// use tensor4all_treetn::{contraction::ContractionOptions, TreeTN};
///
/// // Row sums of the column-major matrix [[1, 3], [2, 4]] over (x, k).
/// let x = DynIndex::new_dyn(2);
/// let k = DynIndex::new_dyn(2);
/// let matrix = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![x.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])?],
///     vec![0usize],
/// )?;
/// let rows = PartitionedTreeTN::from_subdomains(vec![
///     SubDomainTreeTN::new(matrix.clone(), Projector::from_pairs([(x.clone(), 0)])?)?,
///     SubDomainTreeTN::new(matrix, Projector::from_pairs([(x.clone(), 1)])?)?,
/// ])?;
/// let ones = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(
///     TreeTN::from_tensors(vec![IdxTensor::from_dense(vec![k], vec![1.0_f64, 1.0])?], vec![0usize])?,
/// )?)?;
///
/// let row0 = Projector::from_pairs([(x, 0)])?;
/// let result = contract_adaptive_selected(
///     &rows,
///     &ones,
///     &0,
///     &ContractionOptions::default(),
///     &PatchingOptions {
///         cutoff: 0.0,
///         ..PatchingOptions::default()
///     },
///     &[row0.clone()],
/// )?;
/// assert_eq!(result.len(), 1);
/// let patch = result.get(&row0).ok_or("missing row 0")?;
/// assert_eq!(patch.data().to_dense()?.to_vec::<f64>()?, vec![4.0, 0.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_adaptive_selected<V>(
    left: &PartitionedTreeTN<V>,
    right: &PartitionedTreeTN<V>,
    center: &V,
    contract_options: &ContractionOptions,
    patching_options: &PatchingOptions,
    wanted: &[Projector],
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let wanted: HashSet<Projector> = wanted.iter().cloned().collect();
    contract_adaptive_with_selection(
        left,
        right,
        center,
        contract_options,
        patching_options,
        Some(&wanted),
        None,
    )
}

fn contract_adaptive_with_selection<V>(
    left: &PartitionedTreeTN<V>,
    right: &PartitionedTreeTN<V>,
    center: &V,
    contract_options: &ContractionOptions,
    patching_options: &PatchingOptions,
    wanted: Option<&HashSet<Projector>>,
    records: Option<&mut CapRecords<V>>,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    // Reject invalid contraction options before the empty-operand shortcut so
    // a bad policy or `max_bond_dim == 0` cannot pass through a no-op path.
    validate_contraction_options(contract_options)?;
    validate_patching_options(patching_options)?;

    let groups = contraction_plan::plan_groups(
        left,
        right,
        center,
        wanted,
        crate::AUTOMATIC_REFINEMENT_LIMIT,
    )?;
    let mut subdomains = Vec::new();
    for pairs in groups {
        subdomains.extend(contract_group_project_first(
            pairs,
            center,
            contract_options,
            patching_options,
        )?);
    }
    let contracted = PartitionedTreeTN::from_subdomains(subdomains)?;
    let truncated = truncate_adaptive_reporting(
        &contracted,
        center,
        patching_options.cutoff,
        patching_options.allocation,
        patching_options.max_bond_dim,
        records,
    )?;
    Ok(truncated)
}

fn contract_group_project_first<V>(
    pairs: Vec<(SubDomainTreeTN<V>, SubDomainTreeTN<V>)>,
    center: &V,
    contract_options: &ContractionOptions,
    patching_options: &PatchingOptions,
) -> Result<Vec<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let sequential = patching_options.split_strategy == PatchSplitStrategy::Sequential
        && patching_options.max_bond_dim.is_some();
    let threshold = patching_options
        .max_bond_dim
        .and_then(std::num::NonZeroUsize::new);
    // An explicit order depends only on the common output site space and
    // projector. An implicit order uses the first completed contribution's
    // actual site layout, preserving method-specific axis order and pruning.
    let mut split_index = if sequential && !patching_options.patch_order.is_empty() {
        if let Some((left, right)) = pairs.first() {
            let output = contraction_output_site_indices(left.data(), right.data())?;
            let projector = left
                .projector()
                .intersection(right.projector())
                .ok_or(PartitionedTreeTNError::ProjectorConflict)?;
            patching_options
                .patch_order
                .iter()
                .find(|index| output.contains(index) && !projector.is_projected_at(index))
                .cloned()
        } else {
            None
        }
    } else {
        None
    };

    #[cfg(test)]
    let group_id = GROUP_PAIR_PROBES.with(|probes| {
        let mut probes = probes.borrow_mut();
        let id = probes.len();
        probes.push((pairs.len(), 0, 0));
        id
    });
    let mut contributions = Vec::with_capacity(pairs.len());
    for (left, right) in &pairs {
        #[cfg(test)]
        GROUP_PAIR_PROBES.with(|probes| probes.borrow_mut()[group_id].1 += 1);
        #[cfg(test)]
        crate::partitioned_tree_tn::CONTRACTED_PAIRS.with(|count| count.set(count.get() + 1));
        let contribution = if let Some((index, threshold)) = split_index.as_ref().zip(threshold) {
            #[cfg(test)]
            GROUP_PAIR_PROBES.with(|probes| probes.borrow_mut()[group_id].2 += 1);
            match left.contract_with_outcome(right, center, contract_options.clone(), threshold)? {
                Some(SubDomainContractionOutcome::Saturated) => {
                    return split_group_project_first(
                        &pairs,
                        index,
                        center,
                        contract_options,
                        patching_options,
                    );
                }
                Some(SubDomainContractionOutcome::Completed(contribution)) => Some(contribution),
                None => None,
            }
        } else {
            left.contract(right, center, contract_options.clone())?
        };
        let Some(contribution) = contribution else {
            continue;
        };
        if sequential {
            if contributions.is_empty() && patching_options.patch_order.is_empty() {
                split_index = choose_split_index(&contribution, center, patching_options)?;
            }
            // Bondless contributions have max_bond_dim() == 1 although the
            // TreeTN outcome completes with an empty report. Keep the old rule.
            if patching_options
                .max_bond_dim
                .is_some_and(|cap| contribution.max_bond_dim() >= cap)
            {
                if let Some(index) = &split_index {
                    return split_group_project_first(
                        &pairs,
                        index,
                        center,
                        contract_options,
                        patching_options,
                    );
                }
            }
        }
        contributions.push(contribution);
    }
    if contributions.is_empty() {
        return Ok(Vec::new());
    }
    // Exact-add the whole group without intermediate truncation (strict TreeTN
    // addition), then truncate the completed group exactly once. Repeated
    // add-and-truncate was hash-order dependent and re-approximated the sum at
    // every contribution; the single post-add truncation caps the group once.
    // The sum is refused before it is formed when it would exceed the limit:
    // a global operator puts every pair in this one group.
    let terms: Vec<_> = contributions.iter().collect();
    let mut probe = SubDomainTreeTN::exact_sum_within(
        &terms,
        patching_options.sum_engine,
        patching_options.group_sum_limit_bytes,
        "contract_adaptive",
    )?;
    let truncation = truncation_options_from_contract(contract_options);
    probe.truncate(center, truncation)?;

    let Some(cap) = patching_options.max_bond_dim else {
        return Ok(vec![probe]);
    };
    let saturated = probe.max_bond_dim() >= cap
        || contributions
            .iter()
            .any(|contribution| contribution.max_bond_dim() >= cap);
    if !saturated {
        return Ok(vec![probe]);
    }
    let probe = with_split_budget(probe, patching_options.cutoff, patching_options.allocation)?;
    let Some(index) = choose_split_index(&probe, center, patching_options)? else {
        return Ok(vec![probe]);
    };
    #[cfg(test)]
    DISCARDED_GROUP_PROBES.with(|count| count.set(count.get() + 1));
    split_group_project_first(&pairs, &index, center, contract_options, patching_options)
}

/// Attach the inherited per-patch budget a split decision needs, if the
/// allocation has one.
///
/// [`BudgetAllocation::Volume`] derives the absolute budget of a single
/// standalone patch (`cutoff * ||F_p||^2`); [`BudgetAllocation::Norm`] needs no
/// absolute budget because split decisions use the relative policy directly, so
/// any inherited absolute budget is left untouched and unused.
fn with_split_budget<V>(
    subdomain: SubDomainTreeTN<V>,
    cutoff: f64,
    allocation: BudgetAllocation,
) -> Result<SubDomainTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    match allocation {
        BudgetAllocation::Volume => {
            let reference_norm_squared = subdomain.norm_squared()?;
            let total_volume = subdomain_volume(&subdomain)?;
            assign_volume_budgets(
                vec![subdomain],
                cutoff,
                reference_norm_squared,
                total_volume,
            )?
            .pop()
            .ok_or(PartitionedTreeTNError::Empty)
        }
        BudgetAllocation::Norm => Ok(subdomain),
    }
}

#[cfg(test)]
thread_local! {
    /// Per group of `contract_group_project_first`: (available pairs,
    /// attempted pairs, outcome calls).
    static GROUP_PAIR_PROBES: std::cell::RefCell<Vec<(usize, usize, usize)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Exact group sums built by `contract_group_project_first` and then
    /// discarded because the group was split.
    static DISCARDED_GROUP_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Split a saturated contraction group at `index`: project the original
/// operands of every pair to each coordinate and retry each child group.
fn split_group_project_first<V>(
    pairs: &[(SubDomainTreeTN<V>, SubDomainTreeTN<V>)],
    index: &tensor4all_core::DynIndex,
    center: &V,
    contract_options: &ContractionOptions,
    patching_options: &PatchingOptions,
) -> Result<Vec<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut children = Vec::with_capacity(index.dim);
    for value in 0..index.dim {
        let mut child_pairs = Vec::with_capacity(pairs.len());
        let mut projected_left = HashMap::new();
        let mut projected_right = HashMap::new();
        for (left, right) in pairs {
            let left_key = left.projector().clone();
            if !projected_left.contains_key(&left_key) {
                projected_left.insert(
                    left_key.clone(),
                    project_if_present(left, index, value, center)?,
                );
            }
            let right_key = right.projector().clone();
            if !projected_right.contains_key(&right_key) {
                projected_right.insert(
                    right_key.clone(),
                    project_if_present(right, index, value, center)?,
                );
            }
            if let (Some(child_left), Some(child_right)) = (
                projected_left.get(&left_key).cloned().flatten(),
                projected_right.get(&right_key).cloned().flatten(),
            ) {
                child_pairs.push((child_left, child_right));
            }
        }
        children.extend(contract_group_project_first(
            child_pairs,
            center,
            contract_options,
            patching_options,
        )?);
    }
    Ok(children)
}

fn project_if_present<V>(
    subdomain: &SubDomainTreeTN<V>,
    index: &tensor4all_core::DynIndex,
    value: usize,
    center: &V,
) -> Result<Option<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if !subdomain
        .all_indices()
        .iter()
        .any(|candidate| candidate == index)
    {
        return Ok(Some(subdomain.clone()));
    }
    let Some(mut projected) =
        subdomain.project(&Projector::from_pairs([(index.clone(), value)])?)?
    else {
        return Ok(None);
    };
    let threshold = 64.0
        * projected
            .scalar_kind()?
            .map_or(f64::EPSILON, ScalarKind::epsilon);
    projected.truncate(
        center,
        TruncationOptions::new().with_svd_policy(SvdTruncationPolicy::new(threshold)),
    )?;
    Ok(Some(projected))
}

fn truncation_options_from_contract(options: &ContractionOptions) -> TruncationOptions {
    let mut truncation = TruncationOptions::default();
    if let Some(policy) = options.svd_policy {
        truncation = truncation.with_svd_policy(policy);
    }
    if let Some(max_bond_dim) = options.max_bond_dim {
        truncation = truncation.with_max_bond_dim(max_bond_dim);
    }
    truncation
}

/// Truncate a partition with the budgets of one [`BudgetAllocation`] rule.
///
/// The reference squared norm `||F||²` and total patch volume are measured
/// once from the eager stored patches. Under [`BudgetAllocation::Volume`] a
/// patch receives the absolute local threshold
/// `local_cutoff_p = cutoff * ||F||² * volume_p / total_volume`; its
/// unprojected site dimensions define `volume`, while projected dimensions
/// contribute one. That whole threshold is applied independently at every
/// local SVD truncation of the patch, and patches whose norm is at most their
/// local cutoff are dropped. Under [`BudgetAllocation::Norm`] every local SVD
/// of a patch instead applies the relative squared tail-sum policy with
/// threshold `cutoff` (the ITensor-style relative cutoff of
/// [`TreeTN::truncate`](tensor4all_treetn::TreeTN::truncate)), and no patch is
/// dropped.
///
/// # Best effort, not a global bound
///
/// `cutoff` bounds the discarded-squared-weight of one local factorization.
/// Reuse of the same local cutoff across several bonds or repeated truncation
/// stages does not provide a global error bound; the final whole-network
/// relative error is best effort and may exceed `cutoff`. Use `max_bond_dim`
/// when a hard retained-rank bound is required.
///
/// # Arguments
///
/// * `partitioned` - Homogeneous partition with eagerly masked patches.
/// * `center` - Existing TreeTN node used for every local truncation.
/// * `cutoff` - Finite non-negative dimensionless discarded-weight cutoff;
///   `cutoff = 0` disables threshold truncation.
/// * `allocation` - Rule that spreads `cutoff` over the patches.
/// * `max_bond_dim` - Optional positive maximum retained bond dimension.
///
/// # Returns
///
/// A new partition with the retained patches. Under
/// [`BudgetAllocation::Volume`] the patches also carry their assigned absolute
/// squared-cutoff metadata; under [`BudgetAllocation::Norm`] they carry none,
/// because their truncation was relative. The input partition is unchanged,
/// including when truncation fails.
///
/// # Errors
///
/// Returns [`PartitionedTreeTNError::InvalidOptions`] for invalid tolerance or
/// rank cap, [`PartitionedTreeTNError::InvalidCenter`] for an absent center,
/// [`PartitionedTreeTNError::VolumeOverflow`] for checked volume arithmetic,
/// [`PartitionedTreeTNError::NonFiniteAdaptiveValue`] for a non-finite norm or
/// budget, and typed TreeTN/backend errors for truncation failures. Stored
/// topology, site assignment, and dtype mismatches are rejected before a patch
/// can be dropped as a shortcut.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     truncate_adaptive, BudgetAllocation, PartitionedTreeTN, Projector, SubDomainTreeTN,
/// };
/// use tensor4all_treetn::TreeTN;
///
/// let site = DynIndex::new_dyn(2);
/// let make = |values, coordinate| -> Result<_, Box<dyn std::error::Error>> {
///     let tensor = IdxTensor::from_dense(vec![site.clone()], values)?;
///     let tree = TreeTN::from_tensors(vec![tensor], vec![0usize])?;
///     Ok(SubDomainTreeTN::new(
///         tree,
///         Projector::from_pairs([(site.clone(), coordinate)])?,
///     )?)
/// };
/// let partition = PartitionedTreeTN::from_subdomains(vec![
///     make(vec![10.0_f64, 1.0e12], 0)?,
///     make(vec![1.0e12, 0.01], 1)?,
/// ])?;
/// // Volume drops the negligible patch because its norm is below the budget.
/// let dropped = truncate_adaptive(
///     &partition,
///     &0,
///     0.1,
///     BudgetAllocation::Volume,
///     Some(2),
/// )?;
/// assert_eq!(dropped.len(), 1);
/// assert!(dropped.contains(&Projector::from_pairs([(site.clone(), 0)])?));
/// // Norm never drops a nonzero patch and keeps both.
/// let kept = truncate_adaptive(
///     &partition,
///     &0,
///     0.1,
///     BudgetAllocation::Norm,
///     Some(2),
/// )?;
/// assert_eq!(kept.len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn truncate_adaptive<V>(
    partitioned: &PartitionedTreeTN<V>,
    center: &V,
    cutoff: f64,
    allocation: BudgetAllocation,
    max_bond_dim: Option<usize>,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    truncate_adaptive_reporting(partitioned, center, cutoff, allocation, max_bond_dim, None)
}

/// [`truncate_adaptive`] with an optional sink for what the cap truncated.
///
/// `records` is `Some` only for the `_with_report` entry points. For every
/// retained patch whose bond sits exactly at the cap it costs one extra
/// truncation of that patch with the same threshold and no cap, which is what
/// separates a rank the cap set from a rank the cutoff left at the cap. The
/// plain entry point passes `None` and pays nothing.
fn truncate_adaptive_reporting<V>(
    partitioned: &PartitionedTreeTN<V>,
    center: &V,
    cutoff: f64,
    allocation: BudgetAllocation,
    max_bond_dim: Option<usize>,
    records: Option<&mut CapRecords<V>>,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    validate_adaptive_truncation_options(cutoff, max_bond_dim)?;
    partitioned.validate_contents()?;
    if partitioned.is_empty() {
        return Ok(PartitionedTreeTN::new());
    }

    let template = partitioned
        .values()
        .next()
        .ok_or(PartitionedTreeTNError::Empty)?;
    ensure_center(template.data(), center)?;

    match allocation {
        BudgetAllocation::Volume => {
            truncate_adaptive_volume(partitioned, center, cutoff, max_bond_dim, records)
        }
        BudgetAllocation::Norm => {
            truncate_adaptive_norm(partitioned, center, cutoff, max_bond_dim, records)
        }
    }
}

/// Volume allocation: absolute, volume-proportional squared-tail budgets.
fn truncate_adaptive_volume<V>(
    partitioned: &PartitionedTreeTN<V>,
    center: &V,
    cutoff: f64,
    max_bond_dim: Option<usize>,
    mut records: Option<&mut CapRecords<V>>,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let (stats, total_volume, total_norm_squared) = patch_stats_and_totals(partitioned)?;
    if total_volume == 0 {
        return Ok(PartitionedTreeTN::new());
    }

    let global_cutoff_squared = cutoff * total_norm_squared;
    if !global_cutoff_squared.is_finite() {
        return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
    }

    let mut retained = Vec::with_capacity(stats.len());
    for stat in stats {
        let local_cutoff_squared =
            global_cutoff_squared * (stat.volume as f64 / total_volume as f64);
        if !local_cutoff_squared.is_finite() {
            return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
        }
        if stat.norm_squared <= local_cutoff_squared {
            continue;
        }

        let subdomain = truncate_patch_and_record(
            stat.subdomain,
            center,
            PatchThreshold::AbsoluteSquared(local_cutoff_squared),
            max_bond_dim,
            records.as_deref_mut(),
        )?;
        retained.push(subdomain.with_budget_squared(local_cutoff_squared)?);
    }

    PartitionedTreeTN::from_subdomains(retained)
}

/// Norm allocation: per-patch relative squared tail-sum cutoffs.
///
/// Each patch is truncated with the relative policy, so no patch is dropped
/// and no accumulated absolute budget is read or attached.
fn truncate_adaptive_norm<V>(
    partitioned: &PartitionedTreeTN<V>,
    center: &V,
    cutoff: f64,
    max_bond_dim: Option<usize>,
    mut records: Option<&mut CapRecords<V>>,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut retained = Vec::with_capacity(partitioned.len());
    for subdomain in partitioned.values() {
        let subdomain = subdomain.clone();
        if !subdomain.norm_squared()?.is_finite() {
            return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
        }
        let subdomain = truncate_patch_and_record(
            subdomain,
            center,
            PatchThreshold::RelativeSquared(cutoff),
            max_bond_dim,
            records.as_deref_mut(),
        )?;
        retained.push(subdomain.without_budget_squared());
    }

    PartitionedTreeTN::from_subdomains(retained)
}

/// Truncate one patch, and record whether the cap set one of its ranks.
///
/// Recording costs one further truncation of the same patch, with the same
/// threshold and no cap, and only for a patch that keeps a bond exactly at the
/// cap: a patch below the cap cannot have been capped, and a patch without
/// bonds has nothing to cap.
fn truncate_patch_and_record<V>(
    subdomain: SubDomainTreeTN<V>,
    center: &V,
    threshold: PatchThreshold,
    max_bond_dim: Option<usize>,
    records: Option<&mut CapRecords<V>>,
) -> Result<SubDomainTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let Some(records) = records else {
        let mut subdomain = subdomain;
        truncate_subdomain_with_threshold(&mut subdomain, center, threshold, max_bond_dim)?;
        return Ok(subdomain);
    };

    let mut truncated = subdomain.clone();
    truncate_subdomain_with_threshold(&mut truncated, center, threshold, max_bond_dim)?;
    let cap_truncated = match max_bond_dim {
        Some(cap) if truncated.data().link_dims().contains(&cap) => {
            let mut uncapped = subdomain;
            truncate_subdomain_with_threshold(&mut uncapped, center, threshold, None)?;
            uncapped.data().link_dims().iter().any(|&dim| dim > cap)
        }
        _ => false,
    };
    records.push(record_patch(&truncated, cap_truncated));
    Ok(truncated)
}

/// Sum input patches that share an equal projector key, then return the list.
///
/// `add_with_patching` is named for addition, so two patches that describe the
/// same projector region (for example two unprojected patches over one site)
/// must be added by strict subdomain addition rather than silently replacing
/// each other through `from_subdomains`' last-write-wins key semantics.
fn combine_equal_key_subdomains<V>(
    subdomains: Vec<SubDomainTreeTN<V>>,
    options: &PatchingOptions,
) -> Result<Vec<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    // Patches with one key are summed in input order, as the former running
    // `add` did.
    let mut groups: HashMap<Projector, Vec<SubDomainTreeTN<V>>> = HashMap::new();
    for subdomain in subdomains {
        groups
            .entry(subdomain.projector().clone())
            .or_default()
            .push(subdomain);
    }
    groups
        .into_values()
        .map(|mut group| {
            if group.len() == 1 {
                return group.pop().ok_or(PartitionedTreeTNError::Empty);
            }
            let terms: Vec<_> = group.iter().collect();
            SubDomainTreeTN::exact_sum_within(
                &terms,
                options.sum_engine,
                options.group_sum_limit_bytes,
                "add_with_patching",
            )
        })
        .collect()
}

fn validate_patching_options(options: &PatchingOptions) -> Result<()> {
    validate_adaptive_truncation_options(options.cutoff, options.max_bond_dim)?;
    for (position, index) in options.patch_order.iter().enumerate() {
        if index.dim == 0 {
            return Err(PartitionedTreeTNError::InvalidOptions {
                operation: "patching",
                reason: "patch_order cannot contain zero-dimensional indices",
            });
        }
        if options.patch_order[..position]
            .iter()
            .any(|previous| previous == index)
        {
            return Err(PartitionedTreeTNError::InvalidOptions {
                operation: "patching",
                reason: "patch_order cannot contain duplicate full indices",
            });
        }
    }
    Ok(())
}

fn validate_adaptive_truncation_options(cutoff: f64, max_bond_dim: Option<usize>) -> Result<()> {
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "patching",
            reason: "cutoff must be finite and non-negative",
        });
    }
    validate_svd_truncation_options(max_bond_dim, None).map_err(|_| {
        PartitionedTreeTNError::InvalidOptions {
            operation: "patching",
            reason: "max_bond_dim must be at least 1",
        }
    })?;
    Ok(())
}

fn validate_patch_order<V>(subdomain: &SubDomainTreeTN<V>, options: &PatchingOptions) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let indices = subdomain.all_indices();
    for requested in &options.patch_order {
        let Some(matched) = indices.iter().find(|candidate| *candidate == requested) else {
            return Err(PartitionedTreeTNError::InvalidOptions {
                operation: "patching",
                reason: "patch_order must contain only full TreeTN site indices",
            });
        };
        // `DynIndex` equality excludes the dimension, so an identity-matching
        // entry with a different dimension must be rejected before any split
        // uses the aliased dimension for masking or indexing.
        if requested.dim != matched.dim {
            return Err(PartitionedTreeTNError::SiteIndexMismatch);
        }
    }
    Ok(())
}

fn assign_volume_budgets<V>(
    subdomains: Vec<SubDomainTreeTN<V>>,
    cutoff: f64,
    reference_norm_squared: f64,
    total_volume: usize,
) -> Result<Vec<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if subdomains.is_empty() {
        return Ok(subdomains);
    }
    if total_volume == 0 {
        return Ok(Vec::new());
    }

    let global_cutoff_squared = cutoff * reference_norm_squared;
    if !global_cutoff_squared.is_finite() {
        return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
    }

    subdomains
        .into_iter()
        .map(|subdomain| {
            let volume = subdomain_volume(&subdomain)?;
            let local_cutoff_squared =
                global_cutoff_squared * (volume as f64 / total_volume as f64);
            if !local_cutoff_squared.is_finite() {
                return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
            }
            subdomain.with_budget_squared(local_cutoff_squared)
        })
        .collect()
}

/// Split-decision probe of one exact patch.
///
/// The probe is the patch with its cutoff budget applied and no bond cap: a
/// capped value could not distinguish an exact rank from a truncated one. It
/// only decides whether and where to split; the patch itself stays exact.
///
/// Under [`BudgetAllocation::Volume`] the patch's assigned absolute budget is
/// used and a patch at or below it returns `None`: it is never split, and the
/// final adaptive truncation drops it. Under [`BudgetAllocation::Norm`] the
/// relative policy is used and every patch is probed.
fn split_decision_probe<V>(
    subdomain: &SubDomainTreeTN<V>,
    center: &V,
    options: &PatchingOptions,
) -> Result<Option<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let threshold = match options.allocation {
        BudgetAllocation::Volume => {
            let budget_squared = inherited_budget_squared(subdomain)?;
            if subdomain.norm_squared()? <= budget_squared {
                return Ok(None);
            }
            PatchThreshold::AbsoluteSquared(budget_squared)
        }
        BudgetAllocation::Norm => PatchThreshold::RelativeSquared(options.cutoff),
    };
    let mut probe = subdomain.clone();
    truncate_subdomain_with_threshold(&mut probe, center, threshold, None)?;
    Ok(Some(probe))
}

/// Inherited absolute budget of one patch under [`BudgetAllocation::Volume`].
///
/// Split decisions need the budget that the enclosing operation assigned, so a
/// missing budget is an internal option error rather than a default.
fn inherited_budget_squared<V>(subdomain: &SubDomainTreeTN<V>) -> Result<f64>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    subdomain
        .budget_squared()
        .ok_or(PartitionedTreeTNError::InvalidOptions {
            operation: "patching",
            reason: "adaptive split decisions require assigned patch budgets",
        })
}

fn patch_stats<V>(partitioned: &PartitionedTreeTN<V>) -> Result<Vec<PatchStats<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut stats = Vec::with_capacity(partitioned.len());
    let mut patches: Vec<_> = partitioned.values().collect();
    sort_by_patch_order(
        &mut patches,
        |left, right| left.patch_order_key().cmp(right.patch_order_key()),
        |patch| patch.content_digest(),
    );
    for subdomain in patches {
        let volume = subdomain_volume(subdomain)?;
        let norm_squared = subdomain.norm_squared()?;
        if !norm_squared.is_finite() {
            return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
        }
        stats.push(PatchStats {
            subdomain: subdomain.clone(),
            volume,
            norm_squared,
        });
    }
    Ok(stats)
}

fn patch_stats_and_totals<V>(
    partitioned: &PartitionedTreeTN<V>,
) -> Result<(Vec<PatchStats<V>>, usize, f64)>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let stats = patch_stats(partitioned)?;
    let total_volume = stats.iter().try_fold(0usize, |total, stat| {
        total
            .checked_add(stat.volume)
            .ok_or(PartitionedTreeTNError::VolumeOverflow)
    })?;
    let total_norm_squared = checked_finite_sum(stats.iter().map(|stat| stat.norm_squared))?;
    Ok((stats, total_volume, total_norm_squared))
}

fn subdomain_volume<V>(subdomain: &SubDomainTreeTN<V>) -> Result<usize>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    checked_product(
        subdomain.all_indices().into_iter().map(|index| {
            if index.dim == 0 {
                return Err(PartitionedTreeTNError::InvalidOptions {
                    operation: "patching",
                    reason: "TreeTN site indices must have positive dimensions",
                });
            }
            Ok(if subdomain.projector().is_projected_at(&index) {
                1
            } else {
                index.dim
            })
        }),
        || PartitionedTreeTNError::VolumeOverflow,
    )
}

fn checked_product(
    factors: impl IntoIterator<Item = Result<usize>>,
    overflow: fn() -> PartitionedTreeTNError,
) -> Result<usize> {
    factors.into_iter().try_fold(1usize, |product, factor| {
        product.checked_mul(factor?).ok_or_else(overflow)
    })
}

fn checked_finite_sum(values: impl IntoIterator<Item = f64>) -> Result<f64> {
    values.into_iter().try_fold(0.0, |sum, value| {
        if !value.is_finite() {
            return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
        }
        let next = sum + value;
        if next.is_finite() {
            Ok(next)
        } else {
            Err(PartitionedTreeTNError::NonFiniteAdaptiveValue)
        }
    })
}

/// Squared discarded-weight threshold applied at each local SVD of a patch.
///
/// `AbsoluteSquared` is the `Volume` allocation's per-patch budget;
/// `RelativeSquared` is the `Norm` allocation's per-factorization relative
/// cutoff. Both use the squared-value measure and the discarded-tail-sum rule.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PatchThreshold {
    /// Drop a suffix while its cumulative squared weight stays below this
    /// absolute threshold.
    AbsoluteSquared(f64),
    /// Drop a suffix while its cumulative squared weight stays below this
    /// fraction of the factorization's own total squared weight.
    RelativeSquared(f64),
}

impl PatchThreshold {
    fn threshold(self) -> f64 {
        match self {
            Self::AbsoluteSquared(threshold) | Self::RelativeSquared(threshold) => threshold,
        }
    }

    fn policy(self) -> SvdTruncationPolicy {
        let policy = SvdTruncationPolicy::new(self.threshold())
            .with_squared_values()
            .with_discarded_tail_sum();
        match self {
            Self::AbsoluteSquared(_) => policy.with_absolute(),
            Self::RelativeSquared(_) => policy.with_relative(),
        }
    }
}

/// Truncate one subdomain with its whole discarded-weight threshold.
///
/// The whole `threshold` is passed to every local SVD of the truncation sweep.
/// Reuse across several bonds or repeated truncation stages is intentional and
/// best effort; it provides no global error bound, so there is no per-edge
/// division and no consumable remaining-budget bookkeeping.
fn truncate_subdomain_with_threshold<V>(
    subdomain: &mut SubDomainTreeTN<V>,
    center: &V,
    threshold: PatchThreshold,
    max_bond_dim: Option<usize>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let cutoff = threshold.threshold();
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
    }
    if max_bond_dim == Some(0) {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "patching",
            reason: "max_bond_dim must be at least 1",
        });
    }

    // Discarded-tail-sum cutoff on squared singular values: the local
    // factorization drops a suffix while its cumulative squared discarded
    // weight stays below the (absolute or relative) threshold.
    let mut options = TruncationOptions::default().with_svd_policy(threshold.policy());
    if let Some(max_bond_dim) = max_bond_dim {
        options = options.with_max_bond_dim(max_bond_dim);
    }
    subdomain.truncate(center, options)
}

fn choose_split_index<V>(
    subdomain: &SubDomainTreeTN<V>,
    center: &V,
    options: &PatchingOptions,
) -> Result<Option<tensor4all_core::DynIndex>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let candidates = split_candidates(subdomain, options);
    if candidates.is_empty() {
        return Ok(None);
    }

    match options.split_strategy {
        PatchSplitStrategy::Sequential => Ok(candidates.into_iter().next()),
        PatchSplitStrategy::ExactParameterGain => {
            choose_exact_parameter_gain_split(subdomain, center, options, candidates)
        }
    }
}

fn split_candidates<V>(
    subdomain: &SubDomainTreeTN<V>,
    options: &PatchingOptions,
) -> Vec<tensor4all_core::DynIndex>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut names = subdomain.data().node_names();
    names.sort();
    let all_indices: Vec<_> = names
        .iter()
        .flat_map(|name| subdomain.data().node_site_indices(name).unwrap_or_default())
        .collect();
    let raw_candidates: Vec<_> = if options.patch_order.is_empty() {
        all_indices.clone()
    } else {
        options.patch_order.clone()
    };

    raw_candidates
        .into_iter()
        .filter(|index| !subdomain.is_projected_at(index))
        .filter(|index| all_indices.iter().any(|candidate| candidate == index))
        .fold(Vec::new(), |mut candidates, index| {
            if !candidates.iter().any(|candidate| candidate == &index) {
                candidates.push(index);
            }
            candidates
        })
}

fn choose_exact_parameter_gain_split<V>(
    subdomain: &SubDomainTreeTN<V>,
    center: &V,
    options: &PatchingOptions,
    candidates: Vec<tensor4all_core::DynIndex>,
) -> Result<Option<tensor4all_core::DynIndex>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut best: Option<(usize, tensor4all_core::DynIndex)> = None;
    for candidate in candidates {
        let candidate_count = split_child_parameter_count(subdomain, center, &candidate, options)?;
        if best
            .as_ref()
            .is_none_or(|(best_count, _)| candidate_count < *best_count)
        {
            best = Some((candidate_count, candidate));
        }
    }
    Ok(best.map(|(_, index)| index))
}

/// Count the logical parameters of the truncated children of one split.
///
/// The truncation of each projected child follows the active allocation:
/// an inherited absolute budget under [`BudgetAllocation::Volume`] (children
/// below it contribute nothing) and the relative policy under
/// [`BudgetAllocation::Norm`] (no child is discarded).
fn split_child_parameter_count<V>(
    subdomain: &SubDomainTreeTN<V>,
    center: &V,
    index: &tensor4all_core::DynIndex,
    options: &PatchingOptions,
) -> Result<usize>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let children = split_subdomain(subdomain, index, options.allocation)?;
    children.into_iter().try_fold(0usize, |total, mut child| {
        let threshold = match options.allocation {
            BudgetAllocation::Volume => {
                let budget_squared = inherited_budget_squared(&child)?;
                if child.norm_squared()? <= budget_squared {
                    return Ok(total);
                }
                PatchThreshold::AbsoluteSquared(budget_squared)
            }
            BudgetAllocation::Norm => PatchThreshold::RelativeSquared(options.cutoff),
        };
        truncate_subdomain_with_threshold(&mut child, center, threshold, options.max_bond_dim)?;
        let child_count = logical_parameter_count(&child)?;
        total
            .checked_add(child_count)
            .ok_or(PartitionedTreeTNError::LogicalParameterCountOverflow)
    })
}

pub(crate) fn logical_parameter_count<V>(subdomain: &SubDomainTreeTN<V>) -> Result<usize>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut total = 0usize;
    for name in subdomain.data().node_names() {
        let node = subdomain
            .data()
            .node_index(&name)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node name has no node index"))?;
        let tensor = subdomain
            .data()
            .tensor(node)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node has no tensor"))?;
        let local = checked_product(tensor.dims().into_iter().map(Ok), || {
            PartitionedTreeTNError::LogicalParameterCountOverflow
        })?;
        total = total
            .checked_add(local)
            .ok_or(PartitionedTreeTNError::LogicalParameterCountOverflow)?;
    }
    Ok(total)
}

/// Split one patch along `index` into one child per coordinate.
///
/// Under [`BudgetAllocation::Volume`] each child inherits an equal share of the
/// parent's assigned absolute budget. Under [`BudgetAllocation::Norm`] no
/// absolute budget is inherited or carried: children are truncated with the
/// relative policy when a split decision needs it.
fn split_subdomain<V>(
    subdomain: &SubDomainTreeTN<V>,
    index: &tensor4all_core::DynIndex,
    allocation: BudgetAllocation,
) -> Result<Vec<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if index.dim == 0 {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "patching",
            reason: "cannot split along a zero-dimensional index",
        });
    }

    let child_budget_squared = match allocation {
        BudgetAllocation::Volume => subdomain
            .budget_squared()
            .map(|budget_squared| budget_squared / index.dim as f64),
        BudgetAllocation::Norm => None,
    };
    let mut children = Vec::with_capacity(index.dim);
    for value in 0..index.dim {
        let projector = Projector::from_pairs([(index.clone(), value)])?;
        let child = subdomain
            .project(&projector)?
            .ok_or(PartitionedTreeTNError::ProjectorConflict)?;
        children.push(match child_budget_squared {
            Some(budget_squared) => child.with_budget_squared(budget_squared)?,
            None if allocation == BudgetAllocation::Norm => child.without_budget_squared(),
            None => child,
        });
    }
    Ok(children)
}

#[cfg(test)]
mod tests;
