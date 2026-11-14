//! How the subset-operator preparation applies the operator to each patch.

use tensor4all_treetn::{ApplyOptions, CertifiedApplyError};

use super::ReconstructionTolerance;
use crate::PartitionedTreeTNError;

/// How [`ReconstructionTarget::prepare_subset_images`](super::ReconstructionTarget) applies the operator to
/// each preimage patch.
#[derive(Debug)]
pub(super) enum LeafApplication<'a, V> {
    /// Exact zip-up; the image carries only backend roundoff.
    Exact,
    /// The caller's truncating options, measured against the exact image.
    Measured(&'a ApplyOptions<V>),
    /// Certified truncation: `share` of the absolute allowance at `tolerance`
    /// is split equally over the patches, and each image's proved bound is its
    /// error. The exact image is never formed.
    Certified {
        share: f64,
        memory_limit: usize,
        tolerance: ReconstructionTolerance,
    },
}

// Manual impls: the variants hold only references and plain values, so the
// enum is `Copy` whether or not `V` is.
impl<V> Clone for LeafApplication<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for LeafApplication<'_, V> {}

/// Map a certified-application failure to the scheduler's error kinds.
pub(super) fn certified_failure(error: CertifiedApplyError) -> PartitionedTreeTNError {
    const OPERATION: &str = "certified operator application";
    match error {
        CertifiedApplyError::MemoryLimit {
            estimated_bytes, ..
        } => PartitionedTreeTNError::ResourceLimit {
            operation: OPERATION,
            limit: "memory_limit (bytes)",
            value: estimated_bytes,
        },
        CertifiedApplyError::BudgetUnreachable { .. } => PartitionedTreeTNError::InvalidOptions {
            operation: OPERATION,
            reason: "a patch cannot be certified within its share of the allowance, which is at roundoff level; raise the share or the tolerance",
        },
        CertifiedApplyError::UnsupportedTopology { .. } => PartitionedTreeTNError::InvalidOptions {
            operation: OPERATION,
            reason: "certified application supports chain-shaped states only; use the exact or measured application on other trees",
        },
        other => PartitionedTreeTNError::tree(format!("certified operator application: {other}")),
    }
}
