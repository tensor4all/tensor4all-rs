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
mod tests;
