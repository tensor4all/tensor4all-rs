//! Dense materialization of [`SubDomainTreeTN`] and [`PartitionedTreeTN`].

use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{DynIndex, IdxTensor, TensorElement};
use tensor4all_treetn::TreeTN;

use super::PartitionedTreeTN;
use crate::error::{PartitionedTreeTNError, Result};
use crate::subdomain_tree_tn::{ensure_same_dtype, scalar_kind_of, SubDomainTreeTN};

impl<V> SubDomainTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Contract this patch into a dense [`IdxTensor`] over its **free** site
    /// indices only.
    ///
    /// Unlike `self.data().to_dense()`, which retains every projected site axis
    /// and returns a full-domain array that is mostly zeros outside the
    /// subdomain, this method slices each projected site index to its fixed
    /// projector coordinate on the local node tensor *before* contracting the
    /// tree. The result has only the unprojected (free) site indices, ordered by
    /// sorted node name and then by each node tensor's leg order (matching
    /// [`TreeTN::to_dense`]).
    ///
    /// When every site index of the patch is projected, the returned tensor has
    /// rank 0 (a single scalar entry) rather than failing.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::DTypeMismatch`] when the patch's scalar
    /// dtype differs from `T`, [`PartitionedTreeTNError::Empty`] or topology /
    /// projector invariant errors when the patch is invalid, or
    /// [`PartitionedTreeTNError::TensorStorage`] /
    /// [`PartitionedTreeTNError::TensorConstruction`] /
    /// [`PartitionedTreeTNError::TreeTN`] when slicing or contraction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorIndex};
    /// use tensor4all_partitionedtreetn::{Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s0 = DynIndex::new_dyn(2);
    /// let s1 = DynIndex::new_dyn(3);
    /// let tensor = IdxTensor::from_dense(
    ///     vec![s0.clone(), s1.clone()],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    /// )?;
    /// let tree = TreeTN::from_tensors(vec![tensor], vec![0usize])?;
    /// let patch = SubDomainTreeTN::new(tree, Projector::from_pairs([(s1.clone(), 1)])?)?;
    ///
    /// // `to_dense` drops the projected axis `s1` and returns only the free axis `s0`.
    /// let dense = patch.to_dense::<f64>()?;
    /// assert_eq!(dense.external_indices(), vec![s0]);
    /// assert_eq!(dense.to_vec::<f64>()?, vec![3.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_dense<T: TensorElement + Default>(&self) -> Result<IdxTensor> {
        self.validate_invariants()?;
        let expected_kind = scalar_kind_of::<T>()?;
        let actual_kind = self.scalar_kind()?;
        ensure_same_dtype(Some(expected_kind), actual_kind)?;

        let node_names = self.data().node_names();
        let mut sliced_tensors = Vec::with_capacity(node_names.len());
        let mut sliced_names = Vec::with_capacity(node_names.len());

        for name in node_names {
            let node = self.data().node_index(&name).ok_or_else(|| {
                PartitionedTreeTNError::tree("TreeTN node name has no node index")
            })?;
            let tensor = self
                .data()
                .tensor(node)
                .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node has no tensor"))?;

            let mut proj_indices = Vec::new();
            let mut proj_values = Vec::new();
            if let Some(site_indices) = self.data().node_site_indices(&name) {
                for index in site_indices {
                    if let Some(value) = self.projector().get(&index) {
                        proj_indices.push(index);
                        proj_values.push(value);
                    }
                }
            }

            let sliced = if proj_indices.is_empty() {
                tensor.clone()
            } else {
                tensor.select_indices(&proj_indices, &proj_values)?
            };
            sliced_tensors.push(sliced);
            sliced_names.push(name);
        }

        let sliced_tree = TreeTN::from_tensors(sliced_tensors, sliced_names)?;
        sliced_tree.to_dense().map_err(PartitionedTreeTNError::from)
    }
}

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Materialize the partition into a single dense [`IdxTensor`], patch by
    /// patch into each patch's subdomain.
    ///
    /// Each patch is contracted once over its own free site indices via
    /// [`SubDomainTreeTN::to_dense`] and scattered directly into the output
    /// buffer, avoiding any intermediate summed network whose bond dimensions
    /// are the sums of the patches' bond dimensions. Uncovered coordinates are
    /// filled with zero.
    ///
    /// Output axes are ordered by sorted node name and then by each node
    /// tensor's leg order, matching [`TreeTN::to_dense`]. When the partition is
    /// empty, this returns a rank-0 zero tensor of scalar type `T`.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::DTypeMismatch`] when any patch's scalar
    /// dtype differs from `T`, [`PartitionedTreeTNError::VolumeOverflow`] when
    /// the product of site dimensions overflows `usize`, invariant errors when
    /// the partition metadata is invalid, or backend / contraction errors from
    /// [`SubDomainTreeTN::to_dense`] and [`IdxTensor::from_dense`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorIndex};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let x = DynIndex::new_dyn(2);
    /// let y = DynIndex::new_dyn(2);
    /// let full = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![x.clone(), y.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])?],
    ///     vec![0usize],
    /// )?;
    /// let patch = SubDomainTreeTN::new(full, Projector::from_pairs([(y.clone(), 1)])?)?;
    /// let partition = PartitionedTreeTN::from_subdomain(patch)?;
    ///
    /// let dense = partition.to_dense::<f64>()?;
    /// assert_eq!(dense.external_indices(), vec![x, y]);
    /// assert_eq!(dense.to_vec::<f64>()?, vec![0.0, 0.0, 3.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_dense<T: TensorElement + Default>(&self) -> Result<IdxTensor> {
        self.validate_contents()?;
        if self.is_empty() {
            return IdxTensor::from_dense(Vec::new(), vec![T::default()])
                .map_err(PartitionedTreeTNError::from);
        }

        let expected_kind = scalar_kind_of::<T>()?;
        let entries = self.ordered_entries();
        let first_patch = entries[0].1;
        ensure_same_dtype(Some(expected_kind), first_patch.scalar_kind()?)?;

        let mut node_names = first_patch.data().node_names();
        node_names.sort();
        let mut full_indices: Vec<DynIndex> = Vec::new();
        for name in &node_names {
            if let Some(site_indices) = first_patch.data().node_site_indices(name) {
                full_indices.extend(site_indices);
            }
        }

        let mut full_strides = Vec::with_capacity(full_indices.len());
        let mut total_len = 1usize;
        for index in &full_indices {
            full_strides.push(total_len);
            total_len = total_len
                .checked_mul(index.dim)
                .ok_or(PartitionedTreeTNError::VolumeOverflow)?;
        }

        let mut buffer = vec![T::default(); total_len];
        #[cfg(debug_assertions)]
        let mut written = vec![false; total_len];

        for (projector, patch) in entries {
            let patch_dense = patch.to_dense::<T>()?;
            let patch_values = patch_dense.to_vec::<T>()?;

            let mut base_offset = 0usize;
            let mut free_dims = Vec::new();
            let mut free_full_strides = Vec::new();
            for (index, &stride) in full_indices.iter().zip(full_strides.iter()) {
                if let Some(coord) = projector.get(index) {
                    base_offset += coord * stride;
                } else {
                    free_dims.push(index.dim);
                    free_full_strides.push(stride);
                }
            }

            if free_dims.is_empty() {
                debug_assert_eq!(patch_values.len(), 1);
                #[cfg(debug_assertions)]
                {
                    debug_assert!(
                        !written[base_offset],
                        "overlapping patch write at linear index {base_offset}"
                    );
                    written[base_offset] = true;
                }
                buffer[base_offset] = patch_values[0];
                continue;
            }

            let mut coords = vec![0usize; free_dims.len()];
            let mut dest = base_offset;
            for &value in &patch_values {
                #[cfg(debug_assertions)]
                {
                    debug_assert!(
                        !written[dest],
                        "overlapping patch write at linear index {dest}"
                    );
                    written[dest] = true;
                }
                buffer[dest] = value;

                for axis in 0..free_dims.len() {
                    coords[axis] += 1;
                    dest += free_full_strides[axis];
                    if coords[axis] < free_dims[axis] {
                        break;
                    }
                    dest -= coords[axis] * free_full_strides[axis];
                    coords[axis] = 0;
                }
            }
        }

        IdxTensor::from_dense(full_indices, buffer).map_err(PartitionedTreeTNError::from)
    }
}
