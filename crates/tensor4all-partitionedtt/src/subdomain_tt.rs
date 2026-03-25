//! SubDomainTT: A tensor train with an associated projector
//!
//! A `SubDomainTT` represents a tensor train whose values are only valid
//! within a specific subdomain defined by a projector.

use std::collections::HashSet;

use crate::error::{PartitionedTTError, Result};
use crate::projector::Projector;
use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

/// A tensor train with an associated projector defining its subdomain.
///
/// The projector specifies which indices are fixed to specific values.
/// The tensor train values are only valid within this projected subdomain.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::{DynIndex, Projector, SubDomainTT, TensorDynLen, TensorTrain};
///
/// let site0 = DynIndex::new_dyn(2);
/// let bond = DynIndex::new_dyn(1);
/// let site1 = DynIndex::new_dyn(2);
///
/// let t0 = TensorDynLen::from_dense(vec![site0.clone(), bond.clone()], vec![1.0, 2.0]).unwrap();
/// let t1 = TensorDynLen::from_dense(vec![bond.clone(), site1.clone()], vec![3.0, 4.0]).unwrap();
/// let tt = TensorTrain::new(vec![t0, t1]).unwrap();
///
/// let projector = Projector::from_pairs([(site0.clone(), 1)]);
/// let subdomain_tt = SubDomainTT::new(tt, projector);
///
/// assert_eq!(subdomain_tt.len(), 2);
/// assert_eq!(subdomain_tt.projector().get(&site0), Some(1));
/// assert_eq!(subdomain_tt.projector().get(&site1), None);
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
        if !self.projector.is_compatible_with(projector) {
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
            let axis_stride = shape[..axis].iter().copied().product::<usize>().max(1);

            if projected_value >= dim {
                // Invalid projection - zero out entire tensor
                if tensor.is_f64() {
                    return TensorDynLen::zeros::<f64>(indices.to_vec()).unwrap();
                } else {
                    return TensorDynLen::zeros::<num_complex::Complex64>(indices.to_vec())
                        .unwrap();
                }
            }

            // Create result tensor based on scalar type
            if tensor.is_f64() {
                let src_data = tensor.to_vec::<f64>().unwrap_or_default();
                let mut result_data = vec![0.0_f64; total_size];

                for flat_idx in 0..total_size {
                    let axis_value = (flat_idx / axis_stride) % dim;
                    if axis_value == projected_value && flat_idx < src_data.len() {
                        result_data[flat_idx] = src_data[flat_idx];
                    }
                }

                TensorDynLen::from_dense(indices.to_vec(), result_data).unwrap()
            } else {
                let src_data = tensor.to_vec::<Complex64>().unwrap_or_default();
                let mut result_data = vec![Complex64::new(0.0, 0.0); total_size];

                for flat_idx in 0..total_size {
                    let axis_value = (flat_idx / axis_stride) % dim;
                    if axis_value == projected_value && flat_idx < src_data.len() {
                        result_data[flat_idx] = src_data[flat_idx];
                    }
                }

                TensorDynLen::from_dense(indices.to_vec(), result_data).unwrap()
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
        if !self.projector.is_compatible_with(other.projector()) {
            return Ok(None);
        }

        // Compute the projector after contraction (external indices only)
        let (proj_after, _external_indices) = Self::projector_after_contract(self, other)?;

        // Project both inputs to their subdomains before contraction
        // This ensures values outside the subdomain are zeroed out
        let self_projected = self.apply_projection();
        let other_projected = other.apply_projection();

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
mod tests;
