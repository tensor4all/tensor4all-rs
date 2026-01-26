//! Adaptive patching algorithms for PartitionedTT
//!
//! This module provides functions for adding SubDomainTTs with automatic
//! splitting when bond dimensions exceed limits.
//!
//! **Note**: These functions are experimental and may change or fail for
//! complex use cases. The core functionality relies on TT addition which
//! is now implemented.

use crate::error::{PartitionedTTError, Result};
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
/// Creates a PartitionedTT from the given subdomains. If subdomains have
/// disjoint projectors, they are kept separate. If any subdomains share
/// the same projector, they are summed using TT addition.
///
/// # Errors
///
/// Returns an error if:
/// - Subdomains have overlapping (but not identical) projectors
/// - TT addition fails due to incompatible structures
///
/// # Note
///
/// The `max_bond_dim` option in PatchingOptions is **not yet implemented**.
/// If you need bond dimension control after summing, use
/// `PartitionedTT::to_tensor_train()` followed by explicit truncation.
pub fn add_with_patching(
    subdomains: Vec<SubDomainTT>,
    options: &PatchingOptions,
) -> Result<PartitionedTT> {
    // Check if any subdomains would require actual patching
    let max_bond = subdomains
        .iter()
        .map(|s| s.max_bond_dim())
        .max()
        .unwrap_or(0);

    if max_bond > options.max_bond_dim && !options.patch_order.is_empty() {
        return Err(PartitionedTTError::NotImplemented(
            "Adaptive patching with bond dimension splitting is not yet implemented. \
             Use PartitionedTT::from_subdomains() for simple cases."
                .to_string(),
        ));
    }

    PartitionedTT::from_subdomains(subdomains)
}

/// Truncate a PartitionedTT with adaptive weighting.
///
/// # Errors
///
/// This function is **not yet implemented** and will return an error.
/// For truncation, access individual SubDomainTTs via `iter()` or `values()`
/// and truncate them directly.
///
/// # Future Implementation
///
/// A full implementation would:
/// 1. Compute norm of each subdomain
/// 2. Adjust cutoff based on relative norms
/// 3. Truncate each subdomain with adjusted cutoff
/// 4. Optionally iterate to refine weights
pub fn truncate_adaptive(
    _partitioned: &PartitionedTT,
    _rtol: f64,
    _max_bond_dim: usize,
) -> Result<PartitionedTT> {
    Err(PartitionedTTError::NotImplemented(
        "truncate_adaptive is not yet implemented. \
         Access subdomains via iter() and truncate them individually."
            .to_string(),
    ))
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
    fn test_add_with_patching_requires_splitting_fails() {
        let (site_inds, link_ind) = make_shared_indices();

        let tt1 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));

        // Options that require splitting (max_bond_dim smaller than actual)
        let options = PatchingOptions {
            rtol: 1e-12,
            max_bond_dim: 1,                         // Force splitting
            patch_order: vec![site_inds[0].clone()], // Non-empty patch order triggers check
        };

        let result = add_with_patching(vec![subdomain1], &options);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PartitionedTTError::NotImplemented(_)
        ));
    }

    #[test]
    fn test_truncate_adaptive_not_implemented() {
        let (site_inds, link_ind) = make_shared_indices();

        let tt1 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));

        let partitioned = PartitionedTT::from_subdomains(vec![subdomain1]).unwrap();

        let result = truncate_adaptive(&partitioned, 1e-12, 100);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PartitionedTTError::NotImplemented(_)
        ));
    }
}
