//! Grouped per-patch mapping for [`PartitionedTreeTN`].

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::DynIndex;

use crate::error::{PartitionedTreeTNError, Result};
use crate::partitioned_tree_tn::PartitionedTreeTN;
use crate::projector::Projector;
use crate::subdomain_tree_tn::SubDomainTreeTN;

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Group patches that agree on the projected values of `group_by` and apply
    /// `f` once per group, collecting all returned patches into a validated
    /// partition.
    ///
    /// For each patch in `self`, its group key is the restriction of its
    /// projector to the indices in `group_by` that it projects. Groups are
    /// visited in deterministic ascending [`Projector`] order, and the patches
    /// within each group are ordered by [`Self::iter`], independent of
    /// insertion order.
    ///
    /// When `group_by` is empty and `self` is non-empty, every patch falls into
    /// one group with an empty [`Projector`] in [`Self::iter`] order. When
    /// `self` is empty, `f` is not called and an empty partition is returned.
    ///
    /// For each group, `f` receives the group's shared projector and the slice
    /// of patches in that group, and may return any number of patches (zero to
    /// drop the group, one to merge the group, or several). Every returned
    /// patch must agree with the group's projector on every index in
    /// `group_by`; it is never silently re-keyed. All returned patches across
    /// all groups must have distinct, pairwise disjoint projectors and share
    /// one named topology, site-index placement, and scalar dtype, validated
    /// through [`Self::from_subdomains`].
    ///
    /// The coverage contract of [`Self::coverage`] is preserved when `f` returns
    /// patches whose projector set matches `self`'s projectors identically. If
    /// `f` adds, drops, merges, or changes any projector, the old coverage
    /// cannot be transported in general, so the result carries no coverage
    /// (`None`); re-attach one with [`Self::with_coverage`].
    /// `f` is `Sync` so a lane-parallel
    /// variant can reuse the same closure bound without changing callers. On
    /// any error, `self` is left untouched.
    ///
    /// # Errors
    ///
    /// Returns the first error raised by `f`,
    /// [`PartitionedTreeTNError::ProjectorMismatch`] if a returned patch's
    /// projector disagrees with its group's projector on any index in
    /// `group_by`,
    /// [`PartitionedTreeTNError::OverlappingProjectors`] if two returned
    /// patches have equal or overlapping projectors, or the first validation
    /// error from [`Self::from_subdomains`]:
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`],
    /// [`PartitionedTreeTNError::DTypeMismatch`],
    /// or a patch validation error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{
    ///     PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
    /// };
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site_g = DynIndex::new_dyn(2);
    /// let site_s = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(1);
    /// let tree = TreeTN::from_tensors(
    ///     vec![
    ///         IdxTensor::from_dense(vec![site_g.clone(), bond.clone()], vec![1.0_f64, 2.0])?,
    ///         IdxTensor::from_dense(vec![bond, site_s.clone()], vec![3.0_f64, 4.0])?,
    ///     ],
    ///     vec![0usize, 1usize],
    /// )?;
    /// let partition = PartitionedTreeTN::from_subdomains(vec![
    ///     SubDomainTreeTN::new(
    ///         tree.clone(),
    ///         Projector::from_pairs([(site_g.clone(), 0), (site_s.clone(), 0)])?,
    ///     )?,
    ///     SubDomainTreeTN::new(
    ///         tree.clone(),
    ///         Projector::from_pairs([(site_g.clone(), 0), (site_s.clone(), 1)])?,
    ///     )?,
    ///     SubDomainTreeTN::new(
    ///         tree,
    ///         Projector::from_pairs([(site_g.clone(), 1), (site_s.clone(), 0)])?,
    ///     )?,
    /// ])?;
    ///
    /// let merged = partition.try_map_patch_groups(&[site_g.clone()], |group_proj, group| {
    ///     let mut sum = group[0].data().clone();
    ///     for patch in &group[1..] {
    ///         sum = sum.add(patch.data()).map_err(PartitionedTreeTNError::from)?;
    ///     }
    ///     Ok(vec![SubDomainTreeTN::new(sum, group_proj.clone())?])
    /// })?;
    ///
    /// assert_eq!(merged.len(), 2);
    /// assert!(merged.contains(&Projector::from_pairs([(site_g.clone(), 0)])?));
    /// assert!(merged.contains(&Projector::from_pairs([(site_g, 1)])?));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_map_patch_groups<F>(&self, group_by: &[DynIndex], f: F) -> Result<Self>
    where
        F: Fn(&Projector, &[&SubDomainTreeTN<V>]) -> Result<Vec<SubDomainTreeTN<V>>> + Sync,
    {
        if self.is_empty() {
            return Ok(Self::new().with_canonical_coverage(self.coverage().map(<[_]>::to_vec)));
        }

        let mut groups: HashMap<Projector, Vec<&SubDomainTreeTN<V>>> = HashMap::new();
        for (projector, patch) in self.iter() {
            let group_key = projector.filter_indices(group_by);
            groups.entry(group_key).or_default().push(patch);
        }

        let mut ordered_groups: Vec<(Projector, Vec<&SubDomainTreeTN<V>>)> =
            groups.into_iter().collect();
        ordered_groups.sort_by(|(left, _), (right, _)| left.canonical_cmp(right));

        let mut mapped = Vec::new();
        let mut seen_projectors = HashSet::new();
        for (group_proj, group_patches) in &ordered_groups {
            let out_patches = f(group_proj, group_patches)?;
            for out_patch in out_patches {
                for idx in group_by {
                    if out_patch.projector().get(idx) != group_proj.get(idx) {
                        return Err(PartitionedTreeTNError::ProjectorMismatch);
                    }
                }
                if !seen_projectors.insert(out_patch.projector().clone()) {
                    return Err(PartitionedTreeTNError::OverlappingProjectors);
                }
                mapped.push(out_patch);
            }
        }

        let result = Self::from_subdomains(mapped)?;
        // A coverage is stated in terms of projectors, so it can only be carried
        // over when `f` left the set of projectors unchanged.
        let projectors_preserved =
            seen_projectors.len() == self.len() && seen_projectors.iter().all(|p| self.contains(p));
        if projectors_preserved {
            Ok(result.with_canonical_coverage(self.coverage().map(<[_]>::to_vec)))
        } else {
            Ok(result)
        }
    }
}
