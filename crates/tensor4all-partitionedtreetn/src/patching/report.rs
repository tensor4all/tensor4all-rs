//! Records of what the `max_bond_dim` cap did to the patches of one adaptive
//! patching operation.
//!
//! [`add_with_patching`](super::add_with_patching) and
//! [`contract_adaptive`](super::contract_adaptive) keep their signature and
//! return no report. Their `_with_report` siblings run the same computation and
//! return the report beside the partition.

use std::fmt::Debug;

use crate::projector::Projector;
use crate::subdomain_tree_tn::{PatchOrderKey, SubDomainTreeTN};

/// One returned patch and what the bond cap did to it.
///
/// The closing truncation of an operation applies the cutoff budget and the
/// bond cap in one sweep, so a returned patch whose bonds sit at the cap looks
/// the same whether the cap or its exact rank set them. This record separates
/// the two.
///
/// # Discarded weight
///
/// The record carries no discarded weight. `TreeTN::truncate` returns no
/// singular values, so the weight the cap discarded would need a separate
/// decomposition of the patch; the flag and the bond dimensions are what the
/// operation already has.
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
/// // A chain over bits `a, b | c` whose bond keeps exact rank 2 after `a` is
/// // fixed, so a cap of 1 truncates both patches with no index left to split.
/// let (a, b, c) = (
///     DynIndex::new_dyn(2),
///     DynIndex::new_dyn(2),
///     DynIndex::new_dyn(2),
/// );
/// let bond = DynIndex::new_dyn(2);
/// let left = IdxTensor::from_dense(
///     vec![a.clone(), b, bond.clone()],
///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
/// )?;
/// let right = IdxTensor::from_dense(vec![bond, c], vec![1.0_f64, 0.0, 0.0, 1.0])?;
/// let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?;
/// let (result, report) = add_with_patching_with_report(
///     vec![SubDomainTreeTN::from_treetn(tree)?],
///     &0,
///     &PatchingOptions {
///         cutoff: 0.0,
///         max_bond_dim: Some(1),
///         patch_order: vec![a],
///         split_strategy: PatchSplitStrategy::Sequential,
///         ..PatchingOptions::default()
///     },
/// )?;
/// assert_eq!(result.len(), 2);
/// for record in &report.patches {
///     assert!(record.cap_truncated);
///     assert_eq!(record.bond_dims, vec![1]);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PatchCapRecord {
    /// Projector of the returned patch.
    pub projector: Projector,
    /// Bond dimensions of the returned patch, sorted ascending. Empty for a
    /// patch without bonds.
    pub bond_dims: Vec<usize>,
    /// Whether the cap, not the cutoff, set a rank of this patch: one of its
    /// bonds sits exactly at the cap while the same cutoff applied without the
    /// cap retains a larger bond dimension.
    ///
    /// `false` for a patch whose exact rank equals the cap, for a patch that
    /// the cutoff alone brought below the cap, and for a patch without bonds.
    /// It is a patch-level statement, not a per-bond one, and it says nothing
    /// about the amount of discarded weight.
    pub cap_truncated: bool,
}

/// What the closing truncation of one adaptive patching operation did to the
/// returned patches.
///
/// The records are what [`PatchCapRecord`] describes; that type carries the
/// worked example.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct PatchingReport {
    /// One record per returned patch, in canonical patch order.
    pub patches: Vec<PatchCapRecord>,
}

impl PatchingReport {
    /// Records of the returned patches whose rank the cap set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::PatchingReport;
    ///
    /// let report = PatchingReport::default();
    /// assert_eq!(report.truncated_by_cap().count(), 0);
    /// ```
    pub fn truncated_by_cap(&self) -> impl Iterator<Item = &PatchCapRecord> {
        self.patches.iter().filter(|record| record.cap_truncated)
    }

    /// Build the report from the records of one closing truncation.
    ///
    /// The records are sorted by patch metadata so the report does not follow
    /// the partition's hash order; patches whose metadata key is equal keep
    /// the order they were recorded in.
    pub(super) fn new<V>(mut entries: Vec<(PatchOrderKey<V>, PatchCapRecord)>) -> Self
    where
        V: Ord,
    {
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        Self {
            patches: entries.into_iter().map(|(_, record)| record).collect(),
        }
    }
}

/// Record one truncated patch and whether the cap set one of its ranks.
pub(super) fn record_patch<V>(
    patch: &SubDomainTreeTN<V>,
    cap_truncated: bool,
) -> (PatchOrderKey<V>, PatchCapRecord)
where
    V: Clone + std::hash::Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut bond_dims = patch.data().link_dims();
    bond_dims.sort_unstable();
    (
        patch.patch_order_key().clone(),
        PatchCapRecord {
            projector: patch.projector().clone(),
            bond_dims,
            cap_truncated,
        },
    )
}
