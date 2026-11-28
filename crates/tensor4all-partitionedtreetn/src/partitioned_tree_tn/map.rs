//! Per-patch mapping and elementwise conjugation for [`PartitionedTreeTN`].

use std::fmt::Debug;
use std::hash::Hash;

use crate::error::Result;
use crate::partitioned_tree_tn::PartitionedTreeTN;
use crate::subdomain_tree_tn::SubDomainTreeTN;

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Apply `f` to every patch in canonical order and collect the results into
    /// a validated partition.
    ///
    /// Patches are visited in the deterministic order of [`Self::iter`]. The
    /// callback `f` may change each patch's tensor network, its site indices,
    /// and its projector — for example relabeling a projected site or removing
    /// a site by contraction — provided all returned patches share one named
    /// topology, site-index placement, and scalar dtype, and have pairwise
    /// disjoint projectors. `f` must keep projected sites projected, or return
    /// the projector that the returned network actually obeys.
    ///
    /// The collected patches are validated through [`Self::from_subdomains`].
    ///
    /// The coverage contract of [`Self::coverage`] is preserved when `f` returns
    /// every patch with its original projector. If `f` changes any projector the
    /// old coverage cannot be transported in general, so the result carries no
    /// coverage (`None`); re-attach one with [`Self::with_coverage`].
    /// `f` is `Sync` so a lane-parallel variant can reuse the same closure
    /// bound without changing callers. On any error, `self` is left untouched.
    ///
    /// # Errors
    ///
    /// Returns the first error raised by `f`, or the first validation error
    /// from [`Self::from_subdomains`]:
    /// [`PartitionedTreeTNError::TopologyMismatch`](crate::PartitionedTreeTNError::TopologyMismatch),
    /// [`PartitionedTreeTNError::SiteIndexMismatch`](crate::PartitionedTreeTNError::SiteIndexMismatch),
    /// [`PartitionedTreeTNError::DTypeMismatch`](crate::PartitionedTreeTNError::DTypeMismatch),
    /// [`PartitionedTreeTNError::OverlappingProjectors`](crate::PartitionedTreeTNError::OverlappingProjectors),
    /// or a patch validation error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorIndex};
    /// use tensor4all_partitionedtreetn::{
    ///     PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
    /// };
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let new_site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 2.0])?],
    ///     vec![0usize],
    /// )?;
    /// let partition = PartitionedTreeTN::from_subdomains(vec![
    ///     SubDomainTreeTN::new(tree.clone(), Projector::from_pairs([(site.clone(), 0)])?)?,
    ///     SubDomainTreeTN::new(tree, Projector::from_pairs([(site.clone(), 1)])?)?,
    /// ])?;
    ///
    /// let relabeled = partition.try_map_patches(|patch| {
    ///     let data = patch
    ///         .data()
    ///         .replaceind(&site, &new_site)
    ///         .map_err(PartitionedTreeTNError::from)?;
    ///     let val = patch
    ///         .projector()
    ///         .get(&site)
    ///         .ok_or(PartitionedTreeTNError::ProjectorMismatch)?;
    ///     SubDomainTreeTN::new(data, Projector::from_pairs([(new_site.clone(), val)])?)
    /// })?;
    ///
    /// assert_eq!(relabeled.len(), 2);
    /// assert!(relabeled.contains(&Projector::from_pairs([(new_site.clone(), 0)])?));
    /// assert!(relabeled.contains(&Projector::from_pairs([(new_site, 1)])?));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_map_patches<F>(&self, f: F) -> Result<Self>
    where
        F: Fn(&SubDomainTreeTN<V>) -> Result<SubDomainTreeTN<V>> + Sync,
    {
        let mut mapped = Vec::with_capacity(self.len());
        let mut projectors_preserved = true;
        for (projector, patch) in self.iter() {
            let new_patch = f(patch)?;
            projectors_preserved &= new_patch.projector() == projector;
            mapped.push(new_patch);
        }
        let result = Self::from_subdomains(mapped)?;
        // A coverage is stated in terms of projectors, so it can only be carried
        // over when `f` left every projector as it was.
        if projectors_preserved {
            Ok(result.with_canonical_coverage(self.coverage().map(<[_]>::to_vec)))
        } else {
            Ok(result)
        }
    }

    /// Return the elementwise complex conjugate of every patch in the
    /// partition, preserving all projectors and leaving `self` untouched.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::InvalidTopology`](crate::PartitionedTreeTNError::InvalidTopology),
    /// [`PartitionedTreeTNError::DTypeMismatch`](crate::PartitionedTreeTNError::DTypeMismatch),
    /// [`PartitionedTreeTNError::UnsupportedDType`](crate::PartitionedTreeTNError::UnsupportedDType),
    /// [`PartitionedTreeTNError::TensorStorage`](crate::PartitionedTreeTNError::TensorStorage),
    /// [`PartitionedTreeTNError::TensorConstruction`](crate::PartitionedTreeTNError::TensorConstruction),
    /// or [`PartitionedTreeTNError::TreeTN`](crate::PartitionedTreeTNError::TreeTN) if rebuilding a
    /// conjugated patch fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorIndex};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(
    ///         vec![site.clone()],
    ///         vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    ///     )?],
    ///     vec![0usize],
    /// )?;
    /// let partition = PartitionedTreeTN::from_subdomains(vec![
    ///     SubDomainTreeTN::new(tree.clone(), Projector::from_pairs([(site.clone(), 0)])?)?,
    ///     SubDomainTreeTN::new(tree, Projector::from_pairs([(site.clone(), 1)])?)?,
    /// ])?;
    ///
    /// let conj = partition.conj()?;
    /// let dense = conj.to_treetn()?.to_dense()?;
    /// assert_eq!(
    ///     dense.to_vec::<Complex64>()?,
    ///     vec![Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)]
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn conj(&self) -> Result<Self> {
        self.try_map_patches(|patch| {
            let mut data = patch.data().clone();
            for node in data.node_indices() {
                if let Some(tensor) = data.tensor_mut(node) {
                    *tensor = tensor.conj();
                }
            }
            SubDomainTreeTN::new(data, patch.projector().clone())
        })
    }
}
