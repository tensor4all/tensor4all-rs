//! Sibling merging of patches produced outside the adaptive driver.
//!
//! [`merge_sibling_patches`] applies the rule of [`merge_siblings`] to any
//! partition whose per-patch L2 errors are known, for example piecewise
//! polynomial patches embedded as projected networks.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

use super::super::verify::{AuditedTerm, Term};
use super::super::{invalid, PatchedInterpolationError};
#[cfg(doc)]
use super::merge_siblings;
use super::{insert_leaf, merge_leaves, Leaf, SiblingMergeReport};
use crate::reconstruction::truncate_toward_allowance;
use crate::{DynIndex, PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN};

/// The L2 error of one patch produced outside
/// [`patched_interpolate`](super::patched_interpolate), the input of
/// [`merge_sibling_patches`].
///
/// `rms` is the patch's error over its own points divided by the square root
/// of their number, `||f - g_P|| / sqrt(|P|)`. A per-value bound such as a
/// sampled maximum is a valid (conservative) `rms`. Set `certified` only when
/// `rms` is exact up to rounding, for example measured exhaustively.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchErrorBound;
/// use tensor4all_partitionedtreetn::Projector;
///
/// let bound = PatchErrorBound::new(Projector::new(), 1e-6);
/// assert_eq!(bound.rms, 1e-6);
/// assert!(!bound.certified);
/// assert!(PatchErrorBound::new(Projector::new(), 0.0).with_certified(true).certified);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PatchErrorBound {
    /// The patch's projector, equal to the projector of its network in the
    /// partition, or of a zero patch absent from the partition.
    pub projector: Projector,
    /// The patch's RMS error, finite and nonnegative.
    pub rms: f64,
    /// Whether `rms` is exact up to rounding rather than an estimate.
    pub certified: bool,
}

impl PatchErrorBound {
    /// An uncertified bound `rms` for the patch `projector`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchErrorBound;
    /// use tensor4all_partitionedtreetn::Projector;
    ///
    /// assert_eq!(PatchErrorBound::new(Projector::new(), 2.0).rms, 2.0);
    /// ```
    pub fn new(projector: Projector, rms: f64) -> Self {
        Self {
            projector,
            rms,
            certified: false,
        }
    }

    /// Set [`Self::certified`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::PatchErrorBound;
    /// use tensor4all_partitionedtreetn::Projector;
    ///
    /// assert!(PatchErrorBound::new(Projector::new(), 0.0).with_certified(true).certified);
    /// ```
    pub fn with_certified(mut self, certified: bool) -> Self {
        self.certified = certified;
        self
    }
}

/// Options of [`merge_sibling_patches`].
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::adaptive_interpolation::SiblingPatchMergeOptions;
///
/// let options = SiblingPatchMergeOptions::new(32).with_leaf_slack_share(0.5);
/// assert_eq!(options.max_bond_dim, 32);
/// assert_eq!(options.leaf_slack_share, 0.5);
/// assert_eq!(SiblingPatchMergeOptions::new(32).leaf_slack_share, 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SiblingPatchMergeOptions {
    /// The bond cap: a merged patch's largest bond dimension must be strictly
    /// below it. Positive.
    pub max_bond_dim: usize,
    /// The share of each input patch's own slack, `tau * sqrt(|P|) - rms *
    /// sqrt(|P|)`, spent on truncating that patch before any merge; in
    /// `[0, 1]`, default `0` (input patches are kept as given). Patches built
    /// without compression, for example exact polynomial embeddings, need a
    /// positive share to reach small bonds; the rest of the slack is left to
    /// the merges. When in doubt, use `0.5`.
    pub leaf_slack_share: f64,
}

impl SiblingPatchMergeOptions {
    /// The options with bond cap `max_bond_dim` and no leaf truncation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::SiblingPatchMergeOptions;
    ///
    /// assert_eq!(SiblingPatchMergeOptions::new(8).max_bond_dim, 8);
    /// ```
    pub fn new(max_bond_dim: usize) -> Self {
        Self {
            max_bond_dim,
            leaf_slack_share: 0.0,
        }
    }

    /// Set [`Self::leaf_slack_share`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::adaptive_interpolation::SiblingPatchMergeOptions;
    ///
    /// let options = SiblingPatchMergeOptions::new(8).with_leaf_slack_share(0.25);
    /// assert_eq!(options.leaf_slack_share, 0.25);
    /// ```
    pub fn with_leaf_slack_share(mut self, share: f64) -> Self {
        self.leaf_slack_share = share;
        self
    }
}

/// Result of [`merge_sibling_patches`].
///
/// # Examples
///
/// See [`merge_sibling_patches`].
#[derive(Debug)]
#[non_exhaustive]
pub struct SiblingPatchMergeResult<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// The merged partition. Zero regions are absent.
    pub partition: PartitionedTreeTN<V>,
    /// The merges and the L2 error of the merged partition.
    pub merge: SiblingMergeReport,
    /// Input patches replaced by a truncation before merging (see
    /// [`SiblingPatchMergeOptions::leaf_slack_share`]).
    pub truncated_leaves: usize,
}

/// Merge complete groups of sibling patches of any patched network whose
/// per-patch L2 errors are known, without exceeding the RMS budget `tau`.
///
/// This is [`merge_siblings`] for patches that did not come from
/// [`patched_interpolate`](super::patched_interpolate), for example piecewise
/// polynomial patches embedded as projected networks. The rule is the same:
/// a complete group of siblings along `patch_order` is replaced by its sum,
/// truncated toward the slack `tau * sqrt(|Q|) - sqrt(sum |P_i| rms_i^2)`, and
/// the merge is accepted when the merged largest bond dimension is strictly
/// below `max_bond_dim`. Merged regions join their parent's group, so merging
/// proceeds recursively toward the root. Every region of the result keeps
/// `||f - g_Q|| <= tau * sqrt(|Q|)`, so the global error stays at most
/// `tau * sqrt(N)` for `N` domain points, provided each input patch met its
/// own allowance. A patch whose `rms` exceeds `tau` is kept and blocks the
/// merges of its ancestors.
///
/// Choose `tau` from a target relative L2 error `rtol` and the norm of the
/// function as `tau = rtol * ||f|| / sqrt(N)`. Merges only spend what the
/// patches leave unused, so patches built well below `tau` merge further.
///
/// # Arguments
///
/// * `partition` - The patches. Every network must carry all physical sites;
///   the domain size `N` is the product of their dimensions.
/// * `patch_errors` - One bound per patch of `partition`, plus one per zero
///   patch that is absent from it. Together they must form a prefix tree over
///   `patch_order`.
/// * `patch_order` - The split order: every projector fixes a prefix of it.
/// * `tau` - The RMS error budget, finite and nonnegative.
/// * `center` - An existing node of the patch networks, the canonical center
///   of truncation.
/// * `options` - The bond cap of merged patches and the share of each patch's
///   slack spent truncating it before merging; see [`SiblingPatchMergeOptions`].
///
/// # Returns
///
/// The merged partition and the merge report, whose `error` is the L2 error
/// of the merged partition assembled from the bounds and truncations.
///
/// # Errors
///
/// Returns [`PatchedInterpolationError::InvalidInput`] for a negative or
/// non-finite `tau` or `rms`, a zero cap, a `leaf_slack_share` outside
/// `[0, 1]`, an empty `patch_errors`, an
/// all-zero input (no network fixes the domain size), a patch without a bound
/// or a bound duplicated, or projectors that are not a prefix tree over
/// `patch_order`; [`PatchedInterpolationError::Partition`] for an unknown
/// `center` and for failures of the sum, truncation, or norm.
///
/// # Examples
///
/// ```
/// use tensor4all_core::ColMajorArray;
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     merge_sibling_patches, PatchErrorBound, SiblingPatchMergeOptions,
/// };
/// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
/// use tensor4all_treetn::TreeTN;
///
/// // A rank-one function on three binary sites, split into its two halves.
/// let core = |a: f64, b: f64| ColMajorArray::new(vec![a, b], vec![1, 2, 1]).unwrap();
/// let (tree, sites) =
///     TreeTN::from_chain_cores(vec![core(1.0, 2.0), core(3.0, 4.0), core(5.0, 6.0)])?;
/// let whole = SubDomainTreeTN::from_treetn(tree)?;
/// let mut halves = Vec::new();
/// let mut bounds = Vec::new();
/// for value in 0..2 {
///     let projector = Projector::from_pairs([(sites[0].clone(), value)])?;
///     halves.push(whole.project(&projector)?.ok_or("empty half")?);
///     bounds.push(PatchErrorBound::new(projector, 0.0).with_certified(true));
/// }
/// let partition = PartitionedTreeTN::from_subdomains(halves)?;
///
/// let options = SiblingPatchMergeOptions::new(4);
/// let merged = merge_sibling_patches(partition, bounds, &sites, 1e-10, &0, &options)?;
/// // The halves sum back to the rank-one whole, within the budget.
/// assert_eq!(merged.partition.len(), 1);
/// assert_eq!(merged.merge.merges.len(), 1);
/// assert_eq!(merged.merge.merges[0].max_bond_dim, 1);
/// let norm = merged.partition.norm()?;
/// assert!((norm - (5.0_f64 * 25.0 * 61.0).sqrt()).abs() < 1e-10);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn merge_sibling_patches<V>(
    partition: PartitionedTreeTN<V>,
    patch_errors: Vec<PatchErrorBound>,
    patch_order: &[DynIndex],
    tau: f64,
    center: &V,
    options: &SiblingPatchMergeOptions,
) -> Result<SiblingPatchMergeResult<V>, PatchedInterpolationError>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    if !(tau.is_finite() && tau >= 0.0) {
        return Err(invalid(format!(
            "tau must be finite and nonnegative, got {tau}"
        )));
    }
    let SiblingPatchMergeOptions {
        max_bond_dim,
        leaf_slack_share,
    } = *options;
    if max_bond_dim == 0 {
        return Err(invalid("max_bond_dim must be positive"));
    }
    if !(0.0..=1.0).contains(&leaf_slack_share) {
        return Err(invalid(format!(
            "leaf_slack_share must be in [0, 1], got {leaf_slack_share}"
        )));
    }
    if patch_errors.is_empty() {
        return Err(invalid("patch_errors must contain at least one patch"));
    }
    let template = partition.values().next().ok_or_else(|| {
        invalid("the partition has no network to fix the domain size; an all-zero input has nothing to merge")
    })?;
    if template.data().node_index(center).is_none() {
        return Err(PartitionedTreeTNError::InvalidCenter.into());
    }
    let network = template.site_index_network();
    let domain_points: f64 = network
        .node_names()
        .into_iter()
        .filter_map(|node| network.site_space(node))
        .flat_map(|space| space.iter().map(|index| index.dim as f64))
        .product();

    let mut patches: HashMap<Projector, SubDomainTreeTN<V>> = partition.into_iter().collect();
    let mut leaves = BTreeMap::new();
    let mut truncated_leaves = 0;
    for bound in patch_errors {
        if !(bound.rms.is_finite() && bound.rms >= 0.0) {
            return Err(invalid(format!(
                "the patch {:?} has a negative or non-finite rms {}",
                bound.projector, bound.rms
            )));
        }
        let fixed: f64 = bound
            .projector
            .iter()
            .map(|(index, _)| index.dim as f64)
            .product();
        let patch_points = domain_points / fixed;
        let mut network = patches.remove(&bound.projector);
        // Optional truncation of the patch toward its share of its own slack;
        // the measured residual adds to the patch's error.
        let mut rms = bound.rms;
        let slack = leaf_slack_share * (tau - bound.rms) * patch_points.sqrt();
        if let Some(source) = network
            .as_ref()
            .filter(|_| slack > 0.0 && slack.is_finite())
        {
            let (candidate, residual) = truncate_toward_allowance(source, center, slack)?;
            if residual > 0.0 {
                rms += residual / patch_points.sqrt();
                network = Some(candidate);
                truncated_leaves += 1;
            }
        }
        let audited = bound.certified.then_some(AuditedTerm {
            rms,
            variance_rms: rms,
            mean_square_rel_std_error: 0.0,
        });
        let term = Term {
            patch_points,
            rms,
            certified: bound.certified,
            audited,
            within_tolerance: bound.rms <= tau,
        };
        let log_norm = match &network {
            Some(network) => {
                let mut data = network.data().clone();
                data.log_norm().ok().filter(|value| !value.is_nan())
            }
            None => Some(f64::NEG_INFINITY),
        };
        let leaf = Leaf {
            projector: bound.projector,
            network,
            term,
            log_norm,
        };
        insert_leaf(&mut leaves, leaf, patch_order)?;
    }
    if !patches.is_empty() {
        return Err(invalid(
            "the partition has patches without a PatchErrorBound",
        ));
    }
    let (partition, merge) = merge_leaves(
        leaves,
        patch_order,
        center,
        tau,
        max_bond_dim,
        domain_points,
        None,
    )?;
    Ok(SiblingPatchMergeResult {
        partition,
        merge,
        truncated_leaves,
    })
}
