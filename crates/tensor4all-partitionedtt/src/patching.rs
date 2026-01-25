//! Adaptive patching algorithms for PartitionedTT
//!
//! This module provides functions for adding SubDomainTTs with automatic
//! splitting when bond dimensions exceed limits.

use crate::error::Result;
use crate::partitioned_tt::PartitionedTT;
use crate::subdomain_tt::SubDomainTT;
use tensor4all_core::DynIndex;

/// Options for patching operations.
#[derive(Debug, Clone)]
pub struct PatchingOptions {
    /// Relative tolerance for truncation
    pub rtol: f64,
    /// Maximum bond dimension before splitting
    pub max_bond_dim: usize,
    /// Order of indices for patching (indices to split first)
    pub patch_order: Vec<DynIndex>,
}

impl Default for PatchingOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-12,
            max_bond_dim: 100,
            patch_order: Vec::new(),
        }
    }
}

/// Add SubDomainTTs with automatic patching.
///
/// If the resulting bond dimension exceeds `max_bond_dim`, the sum is
/// recursively partitioned into smaller patches.
///
/// Note: This is a placeholder implementation. Full patching requires
/// tensor train addition and truncation algorithms.
pub fn add_with_patching(
    subdomains: Vec<SubDomainTT>,
    _options: &PatchingOptions,
) -> Result<PartitionedTT> {
    // Placeholder: just create a PartitionedTT from the subdomains
    // A full implementation would:
    // 1. Attempt to sum subdomains with the same projector
    // 2. If max_bond_dim is exceeded, split into smaller patches
    // 3. Recursively apply patching to the split patches

    PartitionedTT::from_subdomains(subdomains)
}

/// Truncate a PartitionedTT with adaptive weighting.
///
/// Each SubDomainTT is truncated with a cutoff adjusted by its relative norm.
pub fn truncate_adaptive(
    partitioned: &PartitionedTT,
    _rtol: f64,
    _max_bond_dim: usize,
) -> Result<PartitionedTT> {
    // Placeholder: return a clone
    // A full implementation would:
    // 1. Compute norm of each subdomain
    // 2. Adjust cutoff based on relative norms
    // 3. Truncate each subdomain with adjusted cutoff
    // 4. Optionally iterate to refine weights

    Ok(partitioned.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projector::Projector;
    use tensor4all_core::index::Index;
    use tensor4all_core::{StorageScalar, TensorDynLen};
    use tensor4all_itensorlike::TensorTrain;

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

    /// Create shared indices for testing
    fn make_shared_indices() -> (Vec<DynIndex>, DynIndex) {
        let s0 = make_index(2);
        let l01 = make_index(3);
        let s1 = make_index(2);
        (vec![s0, s1], l01)
    }

    /// Create a TT using the provided indices
    fn make_tt_with_indices(site_inds: &[DynIndex], link_ind: &DynIndex) -> TensorTrain {
        let t0 = make_tensor(vec![site_inds[0].clone(), link_ind.clone()]);
        let t1 = make_tensor(vec![link_ind.clone(), site_inds[1].clone()]);
        TensorTrain::new(vec![t0, t1]).unwrap()
    }

    #[test]
    fn test_add_with_patching_simple() {
        let (site_inds, link_ind) = make_shared_indices();

        let tt1 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));

        let tt2 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[0].clone(), 1)]));

        let options = PatchingOptions::default();
        let result = add_with_patching(vec![subdomain1, subdomain2], &options).unwrap();

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_truncate_adaptive() {
        let (site_inds, link_ind) = make_shared_indices();

        let tt1 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));

        let tt2 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[0].clone(), 1)]));

        let partitioned = PartitionedTT::from_subdomains(vec![subdomain1, subdomain2]).unwrap();

        let result = truncate_adaptive(&partitioned, 1e-12, 100).unwrap();
        assert_eq!(result.len(), 2);
    }
}
