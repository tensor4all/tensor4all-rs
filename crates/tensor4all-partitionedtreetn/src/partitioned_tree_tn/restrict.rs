//! Restriction of a partition to the region selected by a list of projectors.

use std::fmt::Debug;
use std::hash::Hash;

use super::{
    ensure_covered, intersect_coverages, projector_covered_by, validate_coverage, PartitionedTreeTN,
};
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
    /// # Coverage contract
    ///
    /// The result's [`Self::coverage`] is `selection ∩ self.coverage()`, where
    /// `None` stands for the whole domain:
    /// - a selected region inside the source coverage with no stored patch is a
    ///   **known zero**: it needs no patch and stays inside the result's
    ///   coverage, so it can be selected;
    /// - a selected region outside the source coverage is **uncomputed**: it is
    ///   refused, never turned into a zero;
    /// - the result's coverage is `None` only when the source's coverage is
    ///   `None` and `selection` covers the whole domain, so a restriction never
    ///   declares more of the domain known than the source did.
    ///
    /// `None` means "defined on the whole space", while `Some(list)` is an
    /// explicit **known set**. A selection that merely enumerates every point
    /// of a dimension (for example one projector per value of an index) is
    /// still a list, so the result carries `Some(list)` rather than `None`:
    /// its coverage is that known set, not "all of the space" by definition.
    /// Only the single empty projector [`Projector::new`] selects the whole
    /// domain symbolically.
    ///
    /// The number of pair projections is bounded by
    /// `self.len() * selection.len()`. An empty `selection` returns an empty
    /// partition whose coverage is the empty list (nothing is known).
    ///
    /// # Arguments
    ///
    /// * `selection` - Pairwise-disjoint projectors describing the region to
    ///   keep. They may extend beyond the stored patches but must stay inside the
    ///   source coverage; see the coverage contract above.
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
    /// // The selection is only part of the domain, so the result is partial.
    /// assert_eq!(restricted.coverage().map(<[_]>::len), Some(2));
    ///
    /// // Selecting nothing leaves nothing known.
    /// let none = partition.restrict(&[])?;
    /// assert!(none.is_empty());
    /// assert_eq!(none.coverage().map(<[_]>::len), Some(0));
    ///
    /// // Selecting the whole domain of a partition defined everywhere keeps `None`.
    /// assert_eq!(partition.restrict(&[Projector::new()])?.coverage(), None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn restrict(&self, selection: &[Projector]) -> Result<Self> {
        self.validate_contents()?;
        if !Projector::are_disjoint(selection) {
            return Err(PartitionedTreeTNError::OverlappingProjectors);
        }
        // Validate the selection against the source site space (when known).
        if let Some(template) = self.values().next() {
            let (sites, _) = template
                .data()
                .all_site_indices()
                .map_err(PartitionedTreeTNError::from)?;
            validate_coverage(selection, Some(&sites), std::iter::empty::<Projector>())?;
        }
        for selected in selection {
            ensure_covered(selected, self.coverage())?;
        }

        let mut projected_patches = Vec::new();
        for patch in self.ordered_patches() {
            for selected in selection {
                if let Some(projected) = patch.project(selected)? {
                    projected_patches.push(projected);
                }
            }
        }

        let selects_all = projector_covered_by(&Projector::new(), selection);
        let coverage = if self.coverage().is_none() && selects_all {
            None
        } else {
            intersect_coverages(Some(selection), self.coverage())
        };
        Ok(Self::from_subdomains(projected_patches)?.with_canonical_coverage(coverage))
    }
}
