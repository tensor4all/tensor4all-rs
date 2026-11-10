//! Sibling merging of a patched interpolation result (milestone M6b).
//!
//! The driver splits a patch whenever it cannot be accepted and never undoes
//! a split, so its children can end up far below the bond cap: the parent may
//! have failed on one localized feature, a sampling miss, or an engine that
//! stopped early. [`merge_siblings`] walks the split tree bottom-up and
//! replaces a complete group of sibling patches by their truncated sum when
//! that keeps the run's L2 contract and stays below the cap.
//!
//! The rule spends only the slack that the members' measured errors leave
//! inside the merged region `Q`: with members `P_i` of measured errors `e_i`
//! and truncation residual `t_Q`, a merge is accepted when
//! `sqrt(sum e_i^2) + t_Q <= tau * sqrt(|Q|)`. Every region of the result
//! therefore still meets its volume-proportional allowance, and the global
//! bound `E <= delta` of the run holds without a second budget. The design is
//! recorded in `docs/design/tree-patching-sibling-merge.md`.
//!
//! # Derivation notice
//!
//! Merging sibling patches after an adaptive operation follows
//! `_mergesmallpatches` in `src/adaptivematmul.jl` of
//! [TCIAlgorithms.jl](https://github.com/tensor4all/TCIAlgorithms.jl) (MIT
//! license; Copyright 2023 Ritter.Marc and contributors), which keeps a sum
//! unless its bond dimension reaches the cap. The error accounting against
//! the producer's measured errors, the complete-group rule, and the
//! zero-patch handling are original to this crate.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_treetn::SumEngine;

use super::report::{L2ErrorReport, NormReport, PatchStatus, PatchedInterpolationReport};
use super::verify::{approximation_rms, global_error_of_terms, Contribution, Term};
use super::{internal, invalid, PatchedInterpolationError, PatchedInterpolationResult};
use crate::reconstruction::truncate_toward_allowance;
use crate::subdomain_tree_tn::ScalarKind;
use crate::{DynIndex, PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN};

/// Options of [`merge_siblings`].
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::SiblingMergeOptions;
///
/// // By default the run's own bond cap applies.
/// assert_eq!(SiblingMergeOptions::default().max_bond_dim, None);
/// let options = SiblingMergeOptions::new().with_max_bond_dim(8);
/// assert_eq!(options.max_bond_dim, Some(8));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SiblingMergeOptions {
    /// The bond cap of a merged patch: its largest bond dimension must be
    /// strictly below it. `None` (the default) uses the run's
    /// [`PatchedInterpolationReport::max_bond_dim`]; a smaller value keeps
    /// merged patches smaller than the run's patches may be. A value above the
    /// run's cap is rejected, since it would produce patches the run could not
    /// have accepted. When in doubt, keep the default.
    pub max_bond_dim: Option<usize>,
}

impl SiblingMergeOptions {
    /// The default options: the run's bond cap.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::SiblingMergeOptions;
    ///
    /// assert_eq!(SiblingMergeOptions::new(), SiblingMergeOptions::default());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set [`Self::max_bond_dim`], which must be positive and at most the
    /// run's cap ([`merge_siblings`] checks both).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::SiblingMergeOptions;
    ///
    /// assert_eq!(SiblingMergeOptions::new().with_max_bond_dim(4).max_bond_dim, Some(4));
    /// ```
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }
}

/// One accepted merge: a complete group of sibling patches replaced by one
/// patch on their parent region.
///
/// RMS values are over the merged region `Q`. The region's error is at most
/// `members_rms + truncation_rms`, by the triangle inequality, and that sum
/// is at most the run's `tau` up to rounding.
///
/// # Examples
///
/// See [`merge_siblings`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SiblingMergeRecord {
    /// The parent region: the members' projector without the last fixed
    /// site of the split order.
    pub region: Projector,
    /// The members' projectors in canonical order. A member can itself be
    /// the region of an earlier merge.
    pub members: Vec<Projector>,
    /// `sqrt(sum |P_i| rms_i^2 / |Q|)`, the members' combined deciding error.
    pub members_rms: f64,
    /// `||sum g_i - g_Q|| / sqrt(|Q|)`, the measured truncation of the sum,
    /// computed from the explicit difference network. Zero when the sum was
    /// kept untruncated.
    pub truncation_rms: f64,
    /// Largest bond dimension of the merged patch, `0` for a region whose
    /// members are all zero patches (no network is stored).
    pub max_bond_dim: usize,
}

/// What [`merge_siblings`] did, and the L2 error of the merged result.
///
/// # Examples
///
/// See [`merge_siblings`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SiblingMergeReport {
    /// The bond cap that was applied.
    pub max_bond_dim: usize,
    /// The accepted merges in the order they were applied: deepest groups
    /// first, in canonical order within one depth.
    pub merges: Vec<SiblingMergeRecord>,
    /// Complete groups rejected because the merged patch reached the cap.
    pub rejected_by_cap: usize,
    /// Complete groups rejected because a member was retained as
    /// [`PatchStatus::ToleranceNotMet`].
    pub rejected_by_status: usize,
    /// The L2 error of the merged result, classified as the run's error is
    /// (see [`GlobalL2Error`](super::GlobalL2Error)). A merged region is
    /// certified when all its members are, since the truncation is computed
    /// from networks. Without an accepted merge it equals the run's error.
    pub error: L2ErrorReport,
}

/// Result of [`merge_siblings`].
///
/// # Examples
///
/// See [`merge_siblings`].
#[derive(Debug)]
#[non_exhaustive]
pub struct SiblingMergeResult<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// The merged partition. Zero regions are absent, as in the run's
    /// partition.
    pub partition: PartitionedTreeTN<V>,
    /// The run's report, unchanged: its records describe the engine runs that
    /// produced the patches, merged or not.
    pub producer: PatchedInterpolationReport,
    /// The merges and the error of the merged partition.
    pub merge: SiblingMergeReport,
}

/// A leaf of the split tree: an accepted patch, a zero patch, or a merged
/// region.
struct Leaf<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    projector: Projector,
    /// `None` for a zero region.
    network: Option<SubDomainTreeTN<V>>,
    term: Term,
    /// `ln ||g||`; `-inf` for a zero region.
    log_norm: Option<f64>,
}

/// Merge complete groups of sibling patches of a patched interpolation result
/// without breaking its L2 error contract.
///
/// The patches of `result` form a prefix tree over
/// [`PatchedInterpolationReport::patch_order`]. A group is complete when all
/// `d` children of a parent region (one per value of the parent's next split
/// site) are leaves: accepted patches, zero patches, or regions merged
/// before. Groups are tried bottom-up, deepest first, in canonical order
/// within a depth; groups of one depth are disjoint, so the result does not
/// depend on that order. A merged region joins its own parent's group, and a
/// rejected group blocks every ancestor.
///
/// A group with members `P_i`, measured errors `e_i` (from the run's
/// acceptance measurements), and region `Q`:
///
/// 1. is rejected when a member was retained as
///    [`PatchStatus::ToleranceNotMet`], since it has no slack and a merge
///    would hide the miss inside a larger region;
/// 2. becomes a zero region when every member is a zero patch;
/// 3. otherwise its members are summed exactly, and the sum is truncated
///    toward the slack `tau * sqrt(|Q|) - sqrt(sum e_i^2)` when that is
///    positive. The truncation residual `t_Q` is measured on the explicit
///    difference network; a candidate exceeding the slack is discarded and
///    the exact sum kept with `t_Q = 0`;
/// 4. is accepted when the merged patch's largest bond dimension is strictly
///    below the cap.
///
/// Every region of the result then satisfies
/// `||f - g_Q|| <= sqrt(sum e_i^2) + t_Q <= tau * sqrt(|Q|)`, so the run's
/// bound `E <= delta` still holds, of the same kind as before: certified when
/// every member's measurement is exact or exhaustive, an estimate otherwise
/// (the truncation is computed, not sampled). Rounding of the difference
/// network is of the order of `eps * ||sum g_i||` and is not part of any
/// decision, as in the run.
///
/// # Arguments
///
/// * `result` - The result of an [`ErrorNorm::L2`](crate::ErrorNorm::L2) run
///   of [`patched_interpolate`](super::patched_interpolate) or its lanes or
///   MPI entry.
/// * `center` - An existing node of the patch networks, the canonical center
///   of truncation. Unused when the partition is empty.
/// * `options` - The bond cap of merged patches; see [`SiblingMergeOptions`].
///
/// # Returns
///
/// The merged partition, the run's report unchanged, and the merge report
/// with the L2 error of the merged partition.
///
/// # Errors
///
/// Returns [`PatchedInterpolationError::InvalidInput`] for a `SampledMax`
/// result (it makes no L2 claim), a cap of zero or above the run's cap, or a
/// result whose patches are not a prefix tree over its `patch_order`;
/// [`PatchedInterpolationError::Partition`] for an unknown `center`
/// ([`PartitionedTreeTNError::InvalidCenter`]) and for failures of the sum,
/// truncation, or norm.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     merge_siblings, patched_interpolate, GlobalL2Error, NormReport,
///     PatchedInterpolationOptions, SiblingMergeOptions, VerificationOptions,
/// };
/// use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance, L2Reference};
/// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::interpolation::{EvalContext, Parallelism};
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // sin(40 x^2) on 256 points of a star with a degree-3 hub and two binary
/// // sites per node; x is read from the eight bits, most significant first.
/// let f = |p: &[usize]| {
///     let x = p.iter().fold(0.0, |a, &b| 2.0 * a + b as f64) / 256.0;
///     (40.0 * x * x).sin()
/// };
/// let mut topology = NodeNameNetwork::new();
/// let mut node_sites = BTreeMap::new();
/// let mut sites = Vec::new();
/// for node in 0..4usize {
///     topology.add_node(node)?;
///     let node_site_list = vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
///     sites.extend(node_site_list.iter().cloned());
///     node_sites.insert(node, node_site_list);
/// }
/// for leaf in 1..4usize {
///     topology.add_edge(&0, &leaf)?;
/// }
/// let norm = (0..256usize)
///     .map(|i| f(&(0..8).map(|k| (i >> (7 - k)) & 1).collect::<Vec<_>>()).powi(2))
///     .sum::<f64>()
///     .sqrt();
/// let options = PatchedInterpolationOptions::new(4)
///     .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
///     .with_tolerance(ErrorTolerance { rtol: 1e-3, atol: 0.0 })
///     .with_verification(VerificationOptions::new().with_max_exhaustive_points(1 << 12))
///     .with_patch_order(sites);
/// let result = {
///     let mut backend = CpuBackend::with_threads(1).expect("1T CPU backend");
///     let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
///     patched_interpolate(
///         &TreeTciInterpolator::default(),
///         topology,
///         node_sites,
///         ColMajorArray::new(vec![], vec![8, 0]).unwrap(),
///         &mut ctx,
///         |batch: ColMajorArrayRef<'_, usize>, _ctx: &mut EvalContext<'_>| -> anyhow::Result<Vec<f64>> {
///             Ok(batch.data().chunks(8).map(f).collect())
///         },
///         &options,
///     )
/// }?;
/// let NormReport::L2 { tau, .. } = result.report.norm else { unreachable!() };
/// let patches = result.partition.len();
///
/// let merged = merge_siblings(result, &0, &SiblingMergeOptions::new())?;
/// // Each accepted merge replaces two sibling patches by one below the cap,
/// // within the region's share of the run's allowance.
/// assert!(!merged.merge.merges.is_empty());
/// assert_eq!(merged.partition.len(), patches - merged.merge.merges.len());
/// for record in &merged.merge.merges {
///     assert!(record.max_bond_dim < 4);
///     assert!(record.members_rms + record.truncation_rms <= tau * (1.0 + 1e-12));
/// }
/// // Exhaustively measured members keep the merged error certified within tau.
/// let GlobalL2Error::Certified { rms_error, .. } = merged.merge.error.global else {
///     unreachable!()
/// };
/// assert!(rms_error <= tau * (1.0 + 1e-12));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn merge_siblings<V>(
    result: PatchedInterpolationResult<V>,
    center: &V,
    options: &SiblingMergeOptions,
) -> Result<SiblingMergeResult<V>, PatchedInterpolationError>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let PatchedInterpolationResult { partition, report } = result;
    let NormReport::L2 { tau, error, .. } = &report.norm else {
        return Err(invalid(
            "sibling merging needs an ErrorNorm::L2 result; a SampledMax run makes no L2 claim \
             whose slack a merge could spend",
        ));
    };
    let (tau, domain_points) = (*tau, error.domain_points);
    let cap = match options.max_bond_dim {
        None => report.max_bond_dim,
        Some(0) => {
            return Err(invalid(
                "SiblingMergeOptions::max_bond_dim must be positive",
            ))
        }
        Some(cap) if cap > report.max_bond_dim => {
            return Err(invalid(format!(
                "SiblingMergeOptions::max_bond_dim ({cap}) exceeds the run's cap ({}); a merged \
                 patch must be one the run could have accepted",
                report.max_bond_dim
            )))
        }
        Some(cap) => cap,
    };
    if let Some(template) = partition.values().next() {
        if template.data().node_index(center).is_none() {
            return Err(PartitionedTreeTNError::InvalidCenter.into());
        }
    }

    let mut leaves = initial_leaves(partition, &report)?;
    let mut merge = SiblingMergeReport {
        max_bond_dim: cap,
        merges: Vec::new(),
        rejected_by_cap: 0,
        rejected_by_status: 0,
        error: error.clone(),
    };
    let deepest = leaves.keys().map(Vec::len).max().unwrap_or(0);
    for depth in (1..=deepest).rev() {
        let site = &report.patch_order[depth - 1];
        let mut groups: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
        for key in leaves.keys().filter(|key| key.len() == depth) {
            *groups.entry(key[..depth - 1].to_vec()).or_default() += 1;
        }
        for (parent, members) in groups {
            if members != site.dim {
                continue;
            }
            let keys: Vec<Vec<usize>> = (0..site.dim)
                .map(|value| {
                    let mut key = parent.clone();
                    key.push(value);
                    key
                })
                .collect();
            let group: Vec<&Leaf<V>> = keys
                .iter()
                .map(|key| leaves.get(key).ok_or_else(missing_leaf))
                .collect::<Result<_, _>>()?;
            match try_merge(&group, site, center, tau, cap)? {
                Attempt::Accepted(accepted) => {
                    let Accepted { leaf, record } = *accepted;
                    for key in &keys {
                        leaves.remove(key);
                    }
                    leaves.insert(parent, leaf);
                    merge.merges.push(record);
                }
                Attempt::RejectedByCap => merge.rejected_by_cap += 1,
                Attempt::RejectedByStatus => merge.rejected_by_status += 1,
            }
        }
    }

    // The canonical order of the run (lexicographic over the split path) is
    // the order of the value keys, because every patch fixes a prefix of the
    // same split order.
    let leaves: Vec<Leaf<V>> = leaves.into_values().collect();
    if !merge.merges.is_empty() {
        merge.error = merged_error(&leaves, domain_points, tau)?;
    }
    let partition = PartitionedTreeTN::from_disjoint_subdomains(
        leaves.into_iter().filter_map(|leaf| leaf.network).collect(),
    )?;
    Ok(SiblingMergeResult {
        partition,
        producer: report,
        merge,
    })
}

/// The leaves of the run, keyed by their values along the split order.
fn initial_leaves<V>(
    partition: PartitionedTreeTN<V>,
    report: &PatchedInterpolationReport,
) -> Result<BTreeMap<Vec<usize>, Leaf<V>>, PatchedInterpolationError>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let mut patches: HashMap<Projector, SubDomainTreeTN<V>> = partition.into_patches();
    let mut leaves = BTreeMap::new();
    let missing = || internal(anyhow::anyhow!("an L2 patch has no acceptance measurement"));
    for record in &report.accepted {
        let acceptance = record.acceptance.as_ref().ok_or_else(missing)?;
        let network = patches.remove(&record.projector).ok_or_else(|| {
            invalid(format!(
                "the result's partition has no patch for the accepted record {:?}",
                record.projector
            ))
        })?;
        let term = Term::of(&Contribution {
            patch_points: acceptance.patch_points,
            acceptance,
            audit: record.audit.as_ref(),
            within_tolerance: record.status == PatchStatus::WithinTolerance,
        });
        let leaf = Leaf {
            projector: record.projector.clone(),
            network: Some(network),
            term,
            log_norm: record.log_norm,
        };
        insert_leaf(&mut leaves, leaf, &report.patch_order)?;
    }
    for record in &report.zero_patches {
        let acceptance = record.acceptance.as_ref().ok_or_else(missing)?;
        let term = Term::of(&Contribution {
            patch_points: acceptance.patch_points,
            acceptance,
            audit: record.audit.as_ref(),
            within_tolerance: true,
        });
        let leaf = Leaf {
            projector: record.projector.clone(),
            network: None,
            term,
            log_norm: Some(f64::NEG_INFINITY),
        };
        insert_leaf(&mut leaves, leaf, &report.patch_order)?;
    }
    if !patches.is_empty() {
        return Err(invalid(
            "the result's partition has patches without an accepted record",
        ));
    }
    Ok(leaves)
}

/// Insert `leaf` under its values along `patch_order`, checking that its
/// projector fixes a prefix of that order.
fn insert_leaf<V>(
    leaves: &mut BTreeMap<Vec<usize>, Leaf<V>>,
    leaf: Leaf<V>,
    patch_order: &[DynIndex],
) -> Result<(), PatchedInterpolationError>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    let key: Vec<usize> = patch_order
        .iter()
        .map_while(|site| leaf.projector.get(site))
        .collect();
    if key.len() != leaf.projector.len() {
        return Err(invalid(format!(
            "the patch {:?} does not fix a prefix of the result's patch_order",
            leaf.projector
        )));
    }
    if leaves.insert(key, leaf).is_some() {
        return Err(invalid(
            "the result has two patches with the same projector",
        ));
    }
    Ok(())
}

struct Accepted<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    leaf: Leaf<V>,
    record: SiblingMergeRecord,
}

enum Attempt<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    Accepted(Box<Accepted<V>>),
    RejectedByCap,
    RejectedByStatus,
}

/// Decide one complete group; see [`merge_siblings`] for the rule.
fn try_merge<V>(
    group: &[&Leaf<V>],
    site: &DynIndex,
    center: &V,
    tau: f64,
    cap: usize,
) -> Result<Attempt<V>, PatchedInterpolationError>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    if group.iter().any(|leaf| !leaf.term.within_tolerance) {
        return Ok(Attempt::RejectedByStatus);
    }
    let first = group.first().ok_or_else(missing_leaf)?;
    let mut region = first.projector.clone();
    region.remove(site);
    let members: Vec<Term> = group.iter().map(|leaf| leaf.term).collect();
    let combined = Term::merged(&members, 0.0);
    let member_projectors = group.iter().map(|leaf| leaf.projector.clone()).collect();

    // The members' networks restricted to the region in metadata only: the
    // data stay masked to each member, so the sum is masked to the union.
    let networks: Vec<SubDomainTreeTN<V>> = group
        .iter()
        .filter_map(|leaf| leaf.network.as_ref())
        .map(|network| {
            SubDomainTreeTN::from_masked_data(network.data().clone(), region.clone(), None)
        })
        .collect::<Result<_, _>>()?;
    if networks.is_empty() {
        let record = SiblingMergeRecord {
            region: region.clone(),
            members: member_projectors,
            members_rms: combined.rms,
            truncation_rms: 0.0,
            max_bond_dim: 0,
        };
        let leaf = Leaf {
            projector: region,
            network: None,
            term: combined,
            log_norm: Some(f64::NEG_INFINITY),
        };
        return Ok(Attempt::Accepted(Box::new(Accepted { leaf, record })));
    }

    let terms: Vec<&SubDomainTreeTN<V>> = networks.iter().collect();
    let sum = SubDomainTreeTN::exact_sum(
        &terms,
        SumEngine::NaryWithin {
            limit_bytes: SumEngine::DEFAULT_NARY_LIMIT_BYTES,
        },
    )?;
    let root_points = combined.patch_points.sqrt();
    // The slack in L2 units; `tau * sqrt(|Q|)` can overflow only when the
    // networks' own norms do.
    let slack = (tau - combined.rms) * root_points;
    let (merged, residual) = if slack > 0.0 && slack.is_finite() {
        truncate_toward_allowance(&sum, center, slack)?
    } else {
        (sum, 0.0)
    };
    let max_bond_dim = merged.max_bond_dim();
    if max_bond_dim >= cap {
        return Ok(Attempt::RejectedByCap);
    }
    let truncation_rms = if residual > 0.0 {
        residual / root_points
    } else {
        0.0
    };
    let mut data = merged.data().clone();
    let log_norm = data.log_norm().ok().filter(|value| !value.is_nan());
    let record = SiblingMergeRecord {
        region: region.clone(),
        members: member_projectors,
        members_rms: combined.rms,
        truncation_rms,
        max_bond_dim,
    };
    let leaf = Leaf {
        projector: region,
        network: Some(merged),
        term: Term::merged(&members, truncation_rms),
        log_norm,
    };
    Ok(Attempt::Accepted(Box::new(Accepted { leaf, record })))
}

/// The L2 error of the merged leaves, assembled as the run assembles its own.
fn merged_error<V>(
    leaves: &[Leaf<V>],
    domain_points: f64,
    tau: f64,
) -> Result<L2ErrorReport, PatchedInterpolationError>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let terms: Vec<Term> = leaves.iter().map(|leaf| leaf.term).collect();
    let approximation_rms = approximation_rms(
        leaves
            .iter()
            .filter(|leaf| leaf.network.is_some())
            .map(|leaf| (leaf.log_norm, leaf.term.patch_points)),
        domain_points,
    );
    // The rounding model scales with the coarsest precision of the networks.
    let mut epsilon = f64::EPSILON;
    for network in leaves.iter().filter_map(|leaf| leaf.network.as_ref()) {
        let kind = network.scalar_kind()?;
        epsilon = epsilon.max(kind.map_or(f64::EPSILON, ScalarKind::epsilon));
    }
    let (global, certified_fraction) =
        global_error_of_terms(&terms, domain_points, tau, approximation_rms, epsilon);
    Ok(L2ErrorReport {
        domain_points,
        global,
        certified_fraction,
        approximation_rms,
    })
}

fn missing_leaf() -> PatchedInterpolationError {
    internal(anyhow::anyhow!("a complete sibling group lost a member"))
}

#[cfg(test)]
mod tests;
