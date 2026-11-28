//! Restriction of a partition to the region selected by a list of projectors.

use std::fmt::Debug;
use std::hash::Hash;

use super::PartitionedTreeTN;
use crate::error::{PartitionedTreeTNError, Result};
use crate::projector::Projector;

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Restrict this partition to the region selected by `selection`, without
    /// truncation.
    ///
    /// The restriction is defined pairwise over `patches x selection`: for
    /// every `(patch, s)`, `patch`'s projector is intersected with `s`
    /// ([`Projector::intersection`]) and projected onto that intersection with
    /// [`SubDomainTreeTN::project`](crate::SubDomainTreeTN::project) unchanged;
    /// empty intersections are dropped. One patch that overlaps two disjoint
    /// selected projectors therefore splits into two patches. Because both the
    /// source patches and `selection` are pairwise disjoint, the resulting
    /// patches are pairwise disjoint as well.
    ///
    /// The number of pair projections is bounded by
    /// `self.len() * selection.len()`. An empty `selection` returns a valid
    /// empty partition.
    ///
    /// # Arguments
    ///
    /// * `selection` - Pairwise-disjoint projectors describing the region to
    ///   keep. Every point of every projector in `selection` must be covered by
    ///   the source partition's patches.
    ///
    /// # Returns
    ///
    /// A new [`PartitionedTreeTN`] containing the projected non-empty pairwise
    /// intersections, leaving `self` unchanged.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`PartitionedTreeTNError::OverlappingProjectors`] when `selection`
    ///   contains overlapping projectors or `self`'s stored layout is not
    ///   pairwise disjoint;
    /// - [`PartitionedTreeTNError::ProjectorIndexNotFound`] when a selected
    ///   projector references a site index absent from the source TreeTN site
    ///   space;
    /// - [`PartitionedTreeTNError::ProjectorCoordinateOutOfBounds`] when a
    ///   selected projector coordinate is out of range for the matched site
    ///   index;
    /// - [`PartitionedTreeTNError::InvalidOptions`] when a selected projector
    ///   extends outside the region covered by `self`'s patches (including when
    ///   `self` is empty and `selection` is non-empty);
    /// - plus the patch, dtype, and topology validation errors documented by
    ///   [`Self::from_subdomain`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s0 = DynIndex::new_dyn(2);
    /// let s1 = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(
    ///         vec![s0.clone(), s1.clone()],
    ///         vec![1.0_f64, 2.0, 3.0, 4.0],
    ///     )?],
    ///     vec![0usize],
    /// )?;
    /// let partition = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(
    ///     tree,
    ///     Projector::from_pairs([(s0.clone(), 0)])?,
    /// )?)?;
    ///
    /// let selection = [
    ///     Projector::from_pairs([(s0.clone(), 0), (s1.clone(), 0)])?,
    ///     Projector::from_pairs([(s0.clone(), 0), (s1.clone(), 1)])?,
    /// ];
    /// let restricted = partition.restrict(&selection)?;
    /// assert_eq!(restricted.len(), 2);
    /// assert!(restricted.contains(&selection[0]));
    /// assert!(restricted.contains(&selection[1]));
    /// assert!(partition.restrict(&[])?.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn restrict(&self, selection: &[Projector]) -> Result<Self> {
        self.validate_contents()?;
        if !Projector::are_disjoint(selection) {
            return Err(PartitionedTreeTNError::OverlappingProjectors);
        }
        if selection.is_empty() {
            return Ok(Self::new());
        }

        let ordered_patches = self.ordered_patches();
        if ordered_patches.is_empty() {
            return Err(PartitionedTreeTNError::InvalidOptions {
                operation: "PartitionedTreeTN::restrict",
                reason: "selected projector extends outside the source partition domain; select only regions covered by the source patches",
            });
        }

        let mut projected_patches = Vec::new();
        for selected in selection {
            let mut uncovered = vec![selected.clone()];
            for &patch in &ordered_patches {
                if let Some(projected) = patch.project(selected)? {
                    projected_patches.push(projected);
                    let mut next_uncovered = Vec::new();
                    for cell in uncovered {
                        if let Some(mut pieces) = cell.split_by(patch.projector()) {
                            next_uncovered.extend(pieces.drain(1..));
                        } else {
                            next_uncovered.push(cell);
                        }
                    }
                    uncovered = next_uncovered;
                }
            }
            if !uncovered.is_empty() {
                return Err(PartitionedTreeTNError::InvalidOptions {
                    operation: "PartitionedTreeTN::restrict",
                    reason: "selected projector extends outside the source partition domain; select only regions covered by the source patches",
                });
            }
        }

        Self::from_subdomains(projected_patches)
    }
}
