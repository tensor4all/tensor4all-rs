//! SubDomainTT: A tensor train with an associated projector
//!
//! A `SubDomainTT` represents a tensor train whose values are only valid
//! within a specific subdomain defined by a projector.

use std::collections::HashSet;

use crate::error::{PartitionedTTError, Result};
use crate::projector::Projector;
use tensor4all_core::{AnyScalar, DynIndex, TensorAccess, TensorDynLen};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

/// A tensor train with an associated projector defining its subdomain.
///
/// The projector specifies which indices are fixed to specific values.
/// The tensor train values are only valid within this projected subdomain.
///
/// # Examples
///
/// ```ignore
/// use tensor4all_partitionedtt::{SubDomainTT, Projector};
/// use tensor4all_itensorlike::TensorTrain;
///
/// // Create a tensor train
/// let tt = TensorTrain::new(tensors).unwrap();
///
/// // Create a projector that fixes some index to value 1
/// let projector = Projector::from_pairs([(some_index, 1)]);
///
/// // Create a SubDomainTT
/// let subdomain_tt = SubDomainTT::new(tt, projector);
/// ```
#[derive(Debug, Clone)]
pub struct SubDomainTT {
    /// The underlying tensor train
    data: TensorTrain,
    /// The projector defining the subdomain
    projector: Projector,
}

impl SubDomainTT {
    /// Create a new SubDomainTT from a tensor train and projector.
    ///
    /// The projector is trimmed to only include indices that exist in the tensor train.
    pub fn new(data: TensorTrain, projector: Projector) -> Self {
        // Trim projector to only include valid indices
        let all_indices = Self::collect_all_indices(&data);
        let trimmed_projector = projector.filter_indices(&all_indices);
        Self {
            data,
            projector: trimmed_projector,
        }
    }

    /// Create a SubDomainTT from a tensor train with an empty projector.
    pub fn from_tt(data: TensorTrain) -> Self {
        Self {
            data,
            projector: Projector::new(),
        }
    }

    /// Collect all site indices from the tensor train.
    fn collect_all_indices(tt: &TensorTrain) -> Vec<DynIndex> {
        tt.siteinds().into_iter().flatten().collect()
    }

    /// Get all site indices (flattened).
    pub fn all_indices(&self) -> Vec<DynIndex> {
        Self::collect_all_indices(&self.data)
    }

    /// Get a reference to the underlying tensor train.
    pub fn data(&self) -> &TensorTrain {
        &self.data
    }

    /// Get a mutable reference to the underlying tensor train.
    pub fn data_mut(&mut self) -> &mut TensorTrain {
        &mut self.data
    }

    /// Get a reference to the projector.
    pub fn projector(&self) -> &Projector {
        &self.projector
    }

    /// Convert to the underlying tensor train, consuming self.
    pub fn into_data(self) -> TensorTrain {
        self.data
    }

    /// Get the number of sites.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor train is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the maximum bond dimension.
    pub fn max_bond_dim(&self) -> usize {
        self.data.maxbonddim()
    }

    /// Get the site indices (nested per site).
    pub fn siteinds(&self) -> Vec<Vec<DynIndex>> {
        self.data.siteinds()
    }

    /// Check if an index is projected.
    pub fn is_projected_at(&self, index: &DynIndex) -> bool {
        self.projector.is_projected_at(index)
    }

    /// Project to a more restrictive projector.
    ///
    /// Returns `None` if the projectors are incompatible (conflicting values).
    /// The resulting SubDomainTT has tensor values zeroed out where the
    /// projection doesn't match.
    pub fn project(&self, projector: &Projector) -> Option<Self> {
        // Check if projectors are compatible
        if !self.projector.has_overlap(projector) {
            return None;
        }

        // Merge projectors
        let merged_projector = self.projector.intersection(projector)?;

        // Project tensor data
        let projected_data = self.project_tensor_data(projector)?;

        Some(Self {
            data: projected_data,
            projector: merged_projector,
        })
    }

    /// Project the tensor data by zeroing out non-matching slices.
    fn project_tensor_data(&self, projector: &Projector) -> Option<TensorTrain> {
        let siteinds = self.data.siteinds();
        let mut new_tensors = Vec::with_capacity(self.data.len());

        for (site, site_indices) in siteinds.iter().enumerate() {
            let tensor = self.data.tensor(site);

            // Check if any site index is projected
            let mut projected_tensor = tensor.clone();
            for idx in site_indices {
                if let Some(projected_value) = projector.get(idx) {
                    projected_tensor =
                        Self::project_tensor_at_index(&projected_tensor, idx, projected_value);
                }
            }
            new_tensors.push(projected_tensor);
        }

        TensorTrain::new(new_tensors).ok()
    }

    /// Project a single tensor by zeroing out all slices except the specified one.
    fn project_tensor_at_index(
        tensor: &TensorDynLen,
        index: &DynIndex,
        projected_value: usize,
    ) -> TensorDynLen {
        use num_complex::Complex64;

        // Find the axis corresponding to this index
        let indices = tensor.indices();
        let axis = indices.iter().position(|i| i == index);

        if let Some(axis) = axis {
            let dim = indices[axis].dim;
            let shape: Vec<usize> = indices.iter().map(|i| i.dim).collect();
            let total_size: usize = shape.iter().product();

            if projected_value >= dim {
                // Invalid projection - zero out entire tensor
                if tensor.is_f64() {
                    return TensorDynLen::zeros_f64(indices.to_vec());
                } else {
                    return TensorDynLen::zeros_c64(indices.to_vec());
                }
            }

            // Helper to convert flat index to multi-index
            let flat_to_multi = |flat_idx: usize| -> Vec<usize> {
                let mut multi_idx = Vec::with_capacity(shape.len());
                let mut remaining = flat_idx;
                for &d in shape.iter().rev() {
                    multi_idx.push(remaining % d);
                    remaining /= d;
                }
                multi_idx.reverse();
                multi_idx
            };

            // Create result tensor based on scalar type
            if tensor.is_f64() {
                let src_data = tensor.as_slice_f64().unwrap_or(&[]);
                let mut result_data = vec![0.0_f64; total_size];

                for flat_idx in 0..total_size {
                    let multi_idx = flat_to_multi(flat_idx);
                    if multi_idx[axis] == projected_value && flat_idx < src_data.len() {
                        result_data[flat_idx] = src_data[flat_idx];
                    }
                }

                TensorDynLen::from_dense_f64(indices.to_vec(), result_data)
            } else {
                let src_data = tensor.as_slice_c64().unwrap_or(&[]);
                let mut result_data = vec![Complex64::new(0.0, 0.0); total_size];

                for flat_idx in 0..total_size {
                    let multi_idx = flat_to_multi(flat_idx);
                    if multi_idx[axis] == projected_value && flat_idx < src_data.len() {
                        result_data[flat_idx] = src_data[flat_idx];
                    }
                }

                TensorDynLen::from_dense_c64(indices.to_vec(), result_data)
            }
        } else {
            // Index not found - return tensor unchanged
            tensor.clone()
        }
    }

    /// Compute the Frobenius norm.
    pub fn norm(&self) -> f64 {
        self.data.norm()
    }

    /// Compute the squared Frobenius norm.
    pub fn norm_squared(&self) -> f64 {
        self.data.norm_squared()
    }

    /// Truncate the tensor train.
    pub fn truncate(&mut self, options: &TruncateOptions) -> Result<()> {
        self.data
            .truncate(options)
            .map_err(|e| PartitionedTTError::TensorTrainError(format!("Truncation failed: {}", e)))
    }

    /// Contract with another SubDomainTT.
    ///
    /// Returns `None` if the projectors are incompatible.
    ///
    /// Before contraction, both inputs are projected to their subdomains
    /// (values outside the subdomain are zeroed out).
    pub fn contract(&self, other: &Self, options: &ContractOptions) -> Result<Option<Self>> {
        // Check if projectors are compatible
        if !self.projector.has_overlap(other.projector()) {
            return Ok(None);
        }

        // Compute the projector after contraction (external indices only)
        let (proj_after, _external_indices) = Self::projector_after_contract(self, other)?;

        // Project both inputs to their subdomains before contraction
        // This ensures values outside the subdomain are zeroed out
        let self_projected = self.apply_projection();
        let other_projected = other.apply_projection();

        // Contract the projected tensor trains
        let contracted_data = self_projected
            .contract(&other_projected, options)
            .map_err(|e| {
                PartitionedTTError::TensorTrainError(format!("Contraction failed: {}", e))
            })?;

        // Create result with the new projector
        let result = Self::new(contracted_data, proj_after);

        Ok(Some(result))
    }

    /// Apply the projector to the tensor data, zeroing out values outside the subdomain.
    ///
    /// Returns the TensorTrain with projection applied.
    fn apply_projection(&self) -> TensorTrain {
        if self.projector.is_empty() {
            return self.data.clone();
        }

        match self.project_tensor_data(&self.projector) {
            Some(tt) => tt,
            None => self.data.clone(),
        }
    }

    /// Compute the projector after contracting two SubDomainTTs.
    ///
    /// Returns (projector, external_indices) where:
    /// - projector contains only projections for external indices
    /// - external_indices are indices that are not contracted away
    fn projector_after_contract(m1: &Self, m2: &Self) -> Result<(Projector, HashSet<DynIndex>)> {
        let indices1: HashSet<_> = m1.all_indices().into_iter().collect();
        let indices2: HashSet<_> = m2.all_indices().into_iter().collect();

        // External indices = (indices1 ∪ indices2) - (indices1 ∩ indices2)
        let common: HashSet<_> = indices1.intersection(&indices2).cloned().collect();
        let all: HashSet<_> = indices1.union(&indices2).cloned().collect();
        let external: HashSet<_> = all.difference(&common).cloned().collect();

        // Build projector for external indices only
        let mut proj_data = Vec::new();
        for idx in &external {
            if let Some(val) = m1.projector.get(idx) {
                proj_data.push((idx.clone(), val));
            } else if let Some(val) = m2.projector.get(idx) {
                proj_data.push((idx.clone(), val));
            }
        }

        Ok((Projector::from_pairs(proj_data), external))
    }

    /// Inner product with another SubDomainTT.
    pub fn inner(&self, other: &Self) -> AnyScalar {
        self.data.inner(other.data())
    }
}

// Conversion from TensorTrain
impl From<TensorTrain> for SubDomainTT {
    fn from(tt: TensorTrain) -> Self {
        Self::from_tt(tt)
    }
}

// Conversion to TensorTrain
impl From<SubDomainTT> for TensorTrain {
    fn from(subdomain: SubDomainTT) -> Self {
        subdomain.into_data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::index::Index;
    use tensor4all_core::StorageScalar;

    fn make_index(size: usize) -> DynIndex {
        Index::new_dyn(size)
    }

    fn make_tensor(indices: Vec<DynIndex>) -> TensorDynLen {
        let dims: Vec<usize> = indices.iter().map(|i| i.dim).collect();
        let size: usize = dims.iter().product();
        let data: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();
        let storage = f64::dense_storage_with_shape(data, &dims);
        TensorDynLen::new(indices, storage)
    }

    fn make_simple_tt() -> (TensorTrain, Vec<DynIndex>, Vec<DynIndex>) {
        // Create a 2-site tensor train
        let s0 = make_index(2); // site 0
        let l01 = make_index(3); // link 0-1
        let s1 = make_index(2); // site 1

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();
        (tt, vec![s0, s1], vec![l01])
    }

    #[test]
    fn test_subdomain_tt_creation() {
        let (tt, site_inds, _) = make_simple_tt();
        let projector = Projector::from_pairs([(site_inds[0].clone(), 1)]);

        let subdomain = SubDomainTT::new(tt, projector);

        assert_eq!(subdomain.len(), 2);
        assert!(subdomain.is_projected_at(&site_inds[0]));
        assert!(!subdomain.is_projected_at(&site_inds[1]));
    }

    #[test]
    fn test_subdomain_tt_from_tt() {
        let (tt, _, _) = make_simple_tt();
        let subdomain = SubDomainTT::from_tt(tt);

        assert_eq!(subdomain.len(), 2);
        assert!(subdomain.projector().is_empty());
    }

    #[test]
    fn test_subdomain_tt_project() {
        let (tt, site_inds, _) = make_simple_tt();
        let subdomain = SubDomainTT::from_tt(tt);

        // Project to fix site 0 to value 1
        let projector = Projector::from_pairs([(site_inds[0].clone(), 1)]);
        let projected = subdomain.project(&projector);

        assert!(projected.is_some());
        let projected = projected.unwrap();
        assert!(projected.is_projected_at(&site_inds[0]));
        assert_eq!(projected.projector().get(&site_inds[0]), Some(1));
    }

    #[test]
    fn test_subdomain_tt_project_incompatible() {
        let (tt, site_inds, _) = make_simple_tt();
        let projector1 = Projector::from_pairs([(site_inds[0].clone(), 0)]);
        let subdomain = SubDomainTT::new(tt, projector1);

        // Try to project with incompatible projector (different value at same site)
        let projector2 = Projector::from_pairs([(site_inds[0].clone(), 1)]);
        let projected = subdomain.project(&projector2);

        assert!(projected.is_none());
    }

    #[test]
    fn test_subdomain_tt_all_indices() {
        let (tt, site_inds, _) = make_simple_tt();
        let subdomain = SubDomainTT::from_tt(tt);

        let all_indices = subdomain.all_indices();
        assert_eq!(all_indices.len(), 2);
        assert!(all_indices.contains(&site_inds[0]));
        assert!(all_indices.contains(&site_inds[1]));
    }

    #[test]
    fn test_subdomain_tt_norm() {
        let (tt, _, _) = make_simple_tt();
        let subdomain = SubDomainTT::from_tt(tt);

        let norm = subdomain.norm();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_subdomain_tt_trim_projector() {
        let (tt, site_inds, _) = make_simple_tt();
        // Projector with an index that doesn't exist in TT
        let fake_index = make_index(5);
        let projector = Projector::from_pairs([(site_inds[0].clone(), 1), (fake_index.clone(), 0)]);

        let subdomain = SubDomainTT::new(tt, projector);

        // Fake index should be trimmed
        assert!(subdomain.is_projected_at(&site_inds[0]));
        assert!(!subdomain.is_projected_at(&fake_index));
        assert_eq!(subdomain.projector().len(), 1);
    }
}
