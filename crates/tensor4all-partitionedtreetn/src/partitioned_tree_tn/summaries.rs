//! Per-patch bond-dimension, norm, and distance summaries of a partition.

use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::AnyScalar;

use super::PartitionedTreeTN;
use crate::error::{PartitionedTreeTNError, Result};
use crate::projector::Projector;
use crate::subdomain_tree_tn::{ensure_same_dtype, ensure_same_tree_structure};

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Return the largest internal bond dimension across all patches, or `0`
    /// for an empty partition.
    ///
    /// This folds [`crate::SubDomainTreeTN::max_bond_dim`] over [`Self::iter`]. A
    /// non-empty partition whose patches are all bondless trees returns `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let empty = PartitionedTreeTN::<usize>::new();
    /// assert_eq!(empty.max_bond_dim(), 0);
    ///
    /// let s0 = DynIndex::new_dyn(2);
    /// let s1 = DynIndex::new_dyn(2);
    /// let link = DynIndex::new_dyn(3);
    /// let t0 = IdxTensor::from_dense(vec![s0.clone(), link.clone()], vec![1.0_f64; 6])?;
    /// let t1 = IdxTensor::from_dense(vec![link, s1], vec![1.0_f64; 6])?;
    /// let tree = TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1usize])?;
    /// let patch = SubDomainTreeTN::new(tree, Projector::from_pairs([(s0, 0)])?)?;
    /// let partition = PartitionedTreeTN::from_subdomain(patch)?;
    /// assert_eq!(partition.max_bond_dim(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn max_bond_dim(&self) -> usize {
        self.iter()
            .fold(0, |acc, (_, patch)| acc.max(patch.max_bond_dim()))
    }

    /// Compute the Frobenius norm of each patch in [`Self::iter`] order.
    ///
    /// Returns the norm `‖A_p‖` (not the squared norm) for every patch,
    /// delegating to [`crate::SubDomainTreeTN::norm`] without materializing a dense
    /// tensor. An empty partition returns an empty vector.
    ///
    /// # Errors
    ///
    /// Returns the validation errors documented by [`Self::norm_squared`],
    /// including [`PartitionedTreeTNError::OverlappingProjectors`],
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`],
    /// [`PartitionedTreeTNError::DTypeMismatch`], and
    /// [`PartitionedTreeTNError::TreeTN`] when a patch norm cannot be
    /// evaluated.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0_f64, 4.0])?],
    ///     vec![0usize],
    /// )?;
    /// let p0 = Projector::from_pairs([(site.clone(), 0)])?;
    /// let p1 = Projector::from_pairs([(site, 1)])?;
    /// let partition = PartitionedTreeTN::from_subdomains(vec![
    ///     SubDomainTreeTN::new(tree.clone(), p1.clone())?,
    ///     SubDomainTreeTN::new(tree, p0.clone())?,
    /// ])?;
    /// let norms = partition.patch_norms()?;
    /// assert_eq!(norms.len(), 2);
    /// assert_eq!(norms[0].0, p0);
    /// assert!((norms[0].1 - 3.0).abs() < 1.0e-12);
    /// assert_eq!(norms[1].0, p1);
    /// assert!((norms[1].1 - 4.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn patch_norms(&self) -> Result<Vec<(Projector, f64)>> {
        self.validate_contents()?;
        self.iter()
            .map(|(projector, patch)| Ok((projector.clone(), patch.norm()?)))
            .collect()
    }

    /// Compute the Frobenius distance `‖A_p - B_p‖` between matching patches
    /// of `self` and `other`, in [`Self::iter`] order.
    ///
    /// Each distance is computed from the exact compressed difference of the
    /// two patch networks without dense materialization. Layout equality means
    /// the two partitions have the exact same set of [`Projector`] keys; when
    /// the key sets differ, this returns
    /// [`PartitionedTreeTNError::LayoutMismatch`] naming the first differing
    /// projector key in canonical order.
    ///
    /// A key present in only one partition is a
    /// [`PartitionedTreeTNError::LayoutMismatch`]; this function never reads an
    /// absent patch as zero. A known-zero patch must be stored under its
    /// projector key.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::LayoutMismatch`] when the two
    /// partitions do not have identical [`Projector`] key sets,
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] when paired patches have
    /// incompatible tree structures or scalar dtypes, the validation errors
    /// documented by [`Self::norm_squared`], or
    /// [`PartitionedTreeTNError::TreeTN`] when forming or taking the norm of a
    /// patch difference fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let make = |vals: Vec<f64>, coord: usize| -> Result<SubDomainTreeTN, Box<dyn std::error::Error>> {
    ///     let tree = TreeTN::from_tensors(
    ///         vec![IdxTensor::from_dense(vec![site.clone()], vals)?],
    ///         vec![0usize],
    ///     )?;
    ///     Ok(SubDomainTreeTN::new(tree, Projector::from_pairs([(site.clone(), coord)])?)?)
    /// };
    /// let left = PartitionedTreeTN::from_subdomains(vec![
    ///     make(vec![3.0, 0.0], 0)?,
    ///     make(vec![0.0, 10.0], 1)?,
    /// ])?;
    /// let right = PartitionedTreeTN::from_subdomains(vec![
    ///     make(vec![1.0, 0.0], 0)?,
    ///     make(vec![0.0, 6.0], 1)?,
    /// ])?;
    /// let distances = left.patch_distances(&right)?;
    /// assert_eq!(distances.len(), 2);
    /// assert!((distances[0].1 - 2.0).abs() < 1.0e-12);
    /// assert!((distances[1].1 - 4.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn patch_distances(&self, other: &Self) -> Result<Vec<(Projector, f64)>> {
        self.validate_contents()?;
        other.validate_contents()?;

        let first_left_only = self
            .iter()
            .find(|(projector, _)| !other.contains(projector));
        let first_right_only = other
            .iter()
            .find(|(projector, _)| !self.contains(projector));
        match (first_left_only, first_right_only) {
            (Some((left_proj, left_patch)), Some((right_proj, right_patch))) => {
                let first = if left_patch.patch_order_key() <= right_patch.patch_order_key() {
                    left_proj.clone()
                } else {
                    right_proj.clone()
                };
                return Err(PartitionedTreeTNError::LayoutMismatch { projector: first });
            }
            (Some((left_proj, _)), None) => {
                return Err(PartitionedTreeTNError::LayoutMismatch {
                    projector: left_proj.clone(),
                });
            }
            (None, Some((right_proj, _))) => {
                return Err(PartitionedTreeTNError::LayoutMismatch {
                    projector: right_proj.clone(),
                });
            }
            (None, None) => {}
        }

        self.iter()
            .map(|(projector, left_patch)| {
                let right_patch =
                    other
                        .get(projector)
                        .ok_or_else(|| PartitionedTreeTNError::LayoutMismatch {
                            projector: projector.clone(),
                        })?;
                ensure_same_tree_structure(left_patch.data(), right_patch.data())?;
                ensure_same_dtype(left_patch.scalar_kind()?, right_patch.scalar_kind()?)?;
                let neg_right = right_patch.data().scale(AnyScalar::new_real(-1.0))?;
                let mut diff = left_patch.data().add(&neg_right)?;
                let distance = diff.norm()?;
                Ok((projector.clone(), distance))
            })
            .collect()
    }
}
