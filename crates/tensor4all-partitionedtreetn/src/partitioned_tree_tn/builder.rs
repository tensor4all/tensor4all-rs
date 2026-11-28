//! Constructing [`PartitionedTreeTN`] partitions from a layout of projectors.

use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

use crate::error::{PartitionedTreeTNError, Result};
use crate::partitioned_tree_tn::PartitionedTreeTN;
use crate::projector::index::has_overlapping_pair;
use crate::projector::Projector;
use crate::subdomain_tree_tn::SubDomainTreeTN;

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Build one patch per projector of `layout` from a function of the
    /// projector, in the order of `layout`.
    ///
    /// Returning `Ok(None)` from `build` leaves that patch absent rather than
    /// storing a zero patch. An empty `layout` (or a `build` callback that
    /// returns `Ok(None)` for every projector) yields an empty partition.
    ///
    /// Duplicate or overlapping projectors in `layout` are rejected before
    /// `build` is called. Each returned [`TreeTN`] is wrapped in a
    /// [`SubDomainTreeTN`] with its projector, and the collected patches are
    /// validated through [`Self::from_subdomains`]. `build` is `Sync` so a
    /// lane-parallel variant can reuse the same closure bound without changing
    /// callers.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`] before calling
    /// `build` if `layout` contains duplicate or overlapping projectors;
    /// propagates the first error returned by `build`; or returns the first
    /// validation error from [`SubDomainTreeTN::new`] or
    /// [`Self::from_subdomains`]:
    /// [`PartitionedTreeTNError::ProjectorIndexNotFound`],
    /// [`PartitionedTreeTNError::ProjectorCoordinateOutOfBounds`],
    /// [`PartitionedTreeTNError::InvalidTopology`],
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`],
    /// [`PartitionedTreeTNError::DTypeMismatch`],
    /// [`PartitionedTreeTNError::UnsupportedDType`],
    /// [`PartitionedTreeTNError::TreeTN`],
    /// [`PartitionedTreeTNError::TensorStorage`], or
    /// [`PartitionedTreeTNError::TensorConstruction`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(3);
    /// let p0 = Projector::from_pairs([(site.clone(), 0)])?;
    /// let p1 = Projector::from_pairs([(site.clone(), 1)])?;
    /// let p2 = Projector::from_pairs([(site.clone(), 2)])?;
    ///
    /// let partition = PartitionedTreeTN::try_from_projectors(&[p0.clone(), p1.clone(), p2.clone()], |p| {
    ///     match p.get(&site) {
    ///         Some(1) => Ok(None),
    ///         Some(val) => {
    ///             let v = (val + 1) as f64;
    ///             let tree = TreeTN::from_tensors(
    ///                 vec![IdxTensor::from_dense(vec![site.clone()], vec![v, v, v])?],
    ///                 vec![0usize],
    ///             )?;
    ///             Ok(Some(tree))
    ///         }
    ///         None => Ok(None),
    ///     }
    /// })?;
    ///
    /// assert_eq!(partition.len(), 2);
    /// assert!(partition.contains(&p0));
    /// assert!(!partition.contains(&p1));
    /// assert!(partition.contains(&p2));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_from_projectors<F>(layout: &[Projector], build: F) -> Result<Self>
    where
        F: Fn(&Projector) -> Result<Option<TreeTN<IdxTensor, V>>> + Sync,
    {
        let projector_refs: Vec<&Projector> = layout.iter().collect();
        if has_overlapping_pair(&projector_refs, false) {
            return Err(PartitionedTreeTNError::OverlappingProjectors);
        }

        let mut subdomains = Vec::with_capacity(layout.len());
        for projector in layout {
            if let Some(tree) = build(projector)? {
                subdomains.push(SubDomainTreeTN::new(tree, projector.clone())?);
            }
        }
        Self::from_subdomains(subdomains)
    }
}
