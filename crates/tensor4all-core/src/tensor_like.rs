//! TensorLike trait for unifying tensor types.
//!
//! This module provides a trait for tensor-like objects that expose external indices
//! and support contraction operations. Currently only `TensorDynLen` implements this trait.
//!
//! Note: This trait works with concrete types (`DynIndex`, `TensorDynLen`) only.
//!
//! # Factorization
//!
//! This module also provides the unified factorization interface via the `factorize` method
//! on the `TensorLike` trait, along with supporting types:
//! - [`FactorizeError`] - Error type for factorize operations
//! - [`FactorizeAlg`] - Factorization algorithm (SVD, QR, LU, CI)
//! - [`Canonical`] - Canonical direction (Left, Right)
//! - [`FactorizeOptions`] - Options for factorization
//! - [`FactorizeResult`] - Result of factorization

use crate::index_like::DynIndex;
use crate::tensor::TensorDynLen;
use anyhow::Result;
use dyn_clone::DynClone;
use std::any::Any;
use std::fmt::Debug;
use thiserror::Error;

// ============================================================================
// Factorization types
// ============================================================================

/// Error type for factorize operations.
#[derive(Debug, Error)]
pub enum FactorizeError {
    #[error("Factorization failed: {0}")]
    ComputationError(#[from] anyhow::Error),
    #[error("Invalid rtol value: {0}. rtol must be finite and non-negative.")]
    InvalidRtol(f64),
    #[error("Unsupported storage type: {0}")]
    UnsupportedStorage(&'static str),
    #[error("Unsupported canonical direction for this algorithm: {0}")]
    UnsupportedCanonical(&'static str),
    #[error("SVD error: {0}")]
    SvdError(#[from] crate::svd::SvdError),
    #[error("QR error: {0}")]
    QrError(#[from] crate::qr::QrError),
}

/// Factorization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FactorizeAlg {
    /// Singular Value Decomposition
    #[default]
    SVD,
    /// QR decomposition
    QR,
    /// Rank-revealing LU decomposition
    LU,
    /// Cross Interpolation (LU-based)
    CI,
}

/// Canonical direction for factorization.
///
/// This determines which factor is "canonical" (orthogonal for SVD/QR,
/// or unit-diagonal for LU/CI).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Canonical {
    /// Left factor is canonical.
    /// - SVD: L=U (orthogonal), R=S*V
    /// - QR: L=Q (orthogonal), R=R
    /// - LU/CI: L has unit diagonal
    #[default]
    Left,
    /// Right factor is canonical.
    /// - SVD: L=U*S, R=V (orthogonal)
    /// - QR: Not supported (would need LQ)
    /// - LU/CI: U has unit diagonal
    Right,
}

/// Options for tensor factorization.
#[derive(Debug, Clone)]
pub struct FactorizeOptions {
    /// Factorization algorithm to use.
    pub alg: FactorizeAlg,
    /// Canonical direction.
    pub canonical: Canonical,
    /// Relative tolerance for truncation.
    /// If `None`, uses the algorithm's default.
    pub rtol: Option<f64>,
    /// Maximum rank for truncation.
    /// If `None`, no rank limit is applied.
    pub max_rank: Option<usize>,
}

impl Default for FactorizeOptions {
    fn default() -> Self {
        Self {
            alg: FactorizeAlg::SVD,
            canonical: Canonical::Left,
            rtol: None,
            max_rank: None,
        }
    }
}

impl FactorizeOptions {
    /// Create options for SVD factorization.
    pub fn svd() -> Self {
        Self {
            alg: FactorizeAlg::SVD,
            ..Default::default()
        }
    }

    /// Create options for QR factorization.
    pub fn qr() -> Self {
        Self {
            alg: FactorizeAlg::QR,
            ..Default::default()
        }
    }

    /// Create options for LU factorization.
    pub fn lu() -> Self {
        Self {
            alg: FactorizeAlg::LU,
            ..Default::default()
        }
    }

    /// Create options for CI factorization.
    pub fn ci() -> Self {
        Self {
            alg: FactorizeAlg::CI,
            ..Default::default()
        }
    }

    /// Set canonical direction.
    pub fn with_canonical(mut self, canonical: Canonical) -> Self {
        self.canonical = canonical;
        self
    }

    /// Set relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = Some(rtol);
        self
    }

    /// Set maximum rank.
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = Some(max_rank);
        self
    }
}

/// Result of tensor factorization.
///
/// Uses concrete types (`DynIndex`, `TensorDynLen`).
#[derive(Debug, Clone)]
pub struct FactorizeResult {
    /// Left factor tensor.
    pub left: TensorDynLen,
    /// Right factor tensor.
    pub right: TensorDynLen,
    /// Bond index connecting left and right factors.
    pub bond_index: DynIndex,
    /// Singular values (only for SVD).
    pub singular_values: Option<Vec<f64>>,
    /// Rank of the factorization.
    pub rank: usize,
}

// ============================================================================
// TensorLike trait
// ============================================================================

/// Trait for tensor-like objects that expose external indices and support contraction.
///
/// This trait provides a common interface for dense tensors (`TensorDynLen`).
///
/// # Design Principles
///
/// - **Minimal interface**: Only external indices and explicit contraction
/// - **Object-safe**: Uses `Vec` returns instead of iterators for trait object compatibility
/// - **Clonable trait objects**: Uses `dyn-clone` for `Box<dyn TensorLike>` cloneability
/// - **Stable ordering**: `external_indices()` returns indices in deterministic order
/// - **Concrete types**: Uses `DynIndex` and `TensorDynLen` directly
///
/// # Example
///
/// ```ignore
/// use tensor4all_core::TensorLike;
///
/// fn contract_external<T: TensorLike>(a: &T, b: &T) -> Result<TensorDynLen> {
///     // Get common external indices and contract
///     let pairs = find_common_indices(a.external_indices(), b.external_indices());
///     a.tensordot(b, &pairs)
/// }
/// ```
pub trait TensorLike: DynClone + Send + Sync + Debug {
    /// Return flattened external indices for this object.
    ///
    /// - For `TensorDynLen`: returns the tensor's indices
    ///
    /// # Ordering
    ///
    /// The ordering MUST be stable (deterministic). Implementations should:
    /// - Sort indices by their `id` field, or
    /// - Use insertion-ordered storage
    ///
    /// This ensures consistent behavior for hashing, serialization, and comparison.
    fn external_indices(&self) -> Vec<DynIndex>;

    /// Number of external indices.
    ///
    /// Default implementation calls `external_indices().len()`, but implementations
    /// SHOULD override this for efficiency when the count can be computed without
    /// allocating the full index list.
    fn num_external_indices(&self) -> usize {
        self.external_indices().len()
    }

    /// Convert this object to a dense tensor.
    ///
    /// - For `TensorDynLen`: returns a clone of self
    ///
    /// This method is required for implementing `tensordot` via trait objects,
    /// since we need access to the underlying tensor data.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails.
    fn to_tensor(&self) -> Result<TensorDynLen>;

    /// Return `self` as `Any` for optional downcasting / runtime type inspection.
    ///
    /// This allows callers to attempt downcasting a trait object back to its
    /// concrete type when needed (similar to C++'s `dynamic_cast`).
    ///
    /// # Implementation
    ///
    /// Implementers should simply return `self`:
    ///
    /// ```ignore
    /// fn as_any(&self) -> &dyn Any { self }
    /// ```
    ///
    /// This requires the concrete type to be `'static` (the usual `Any` constraint).
    fn as_any(&self) -> &dyn Any;

    /// Replace an index in this tensor-like object.
    ///
    /// This replaces the index matching `old_index` by ID with `new_index`.
    /// The storage data is not modified, only the index metadata is changed.
    ///
    /// # Arguments
    ///
    /// * `old_index` - The index to replace (matched by ID)
    /// * `new_index` - The new index to use
    ///
    /// # Returns
    ///
    /// A new `TensorDynLen` with the index replaced.
    ///
    /// # Default Implementation
    ///
    /// The default implementation converts to `TensorDynLen` via `to_tensor()`
    /// and then uses `TensorDynLen::replaceind`.
    fn replaceind(&self, old_index: &DynIndex, new_index: &DynIndex) -> Result<TensorDynLen> {
        let tensor = self.to_tensor()?;
        Ok(tensor.replaceind(old_index, new_index))
    }

    /// Replace multiple indices in this tensor-like object.
    ///
    /// This replaces each index in `old_indices` (matched by ID) with the
    /// corresponding index in `new_indices`. The storage data is not modified.
    ///
    /// # Arguments
    ///
    /// * `old_indices` - The indices to replace (matched by ID)
    /// * `new_indices` - The new indices to use
    ///
    /// # Returns
    ///
    /// A new `TensorDynLen` with the indices replaced.
    ///
    /// # Default Implementation
    ///
    /// The default implementation converts to `TensorDynLen` via `to_tensor()`
    /// and then uses `TensorDynLen::replaceinds`.
    fn replaceinds(
        &self,
        old_indices: &[DynIndex],
        new_indices: &[DynIndex],
    ) -> Result<TensorDynLen> {
        let tensor = self.to_tensor()?;
        Ok(tensor.replaceinds(old_indices, new_indices))
    }

    /// Explicit contraction between two tensor-like objects.
    ///
    /// This performs binary contraction over the specified index pairs.
    /// Each pair `(idx_self, idx_other)` specifies:
    /// - An index from `self` to contract
    /// - An index from `other` to contract with
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor-like object to contract with
    /// * `pairs` - List of (self_index, other_index) pairs to contract
    ///
    /// # Returns
    ///
    /// A new `TensorDynLen` representing the contracted result.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `pairs` is empty
    /// - An index in `pairs` doesn't exist in the corresponding object
    /// - Index dimensions don't match
    /// - There are common indices not in `pairs` (ambiguous contraction)
    ///
    /// # Default Implementation
    ///
    /// The default implementation converts both operands to `TensorDynLen` via
    /// `to_tensor()` and then uses `TensorDynLen::tensordot`.
    fn tensordot(
        &self,
        other: &dyn TensorLike,
        pairs: &[(DynIndex, DynIndex)],
    ) -> Result<TensorDynLen> {
        // Convert both operands to TensorDynLen
        let self_tensor = self.to_tensor()?;
        let other_tensor = other.to_tensor()?;

        // Use TensorDynLen's tensordot
        self_tensor.tensordot(&other_tensor, pairs)
    }

    /// Factorize this tensor into left and right factors.
    ///
    /// This function dispatches to the appropriate algorithm based on `options.alg`:
    /// - `SVD`: Singular Value Decomposition
    /// - `QR`: QR decomposition
    /// - `LU`: Rank-revealing LU decomposition
    /// - `CI`: Cross Interpolation
    ///
    /// The `canonical` option controls which factor is "canonical":
    /// - `Canonical::Left`: Left factor is orthogonal (SVD/QR) or unit-diagonal (LU/CI)
    /// - `Canonical::Right`: Right factor is orthogonal (SVD) or unit-diagonal (LU/CI)
    ///
    /// # Arguments
    /// * `left_inds` - Indices to place on the left side
    /// * `options` - Factorization options
    ///
    /// # Returns
    /// A `FactorizeResult` containing the left and right factors, bond index,
    /// singular values (for SVD), and rank.
    ///
    /// # Errors
    /// Returns `FactorizeError` if:
    /// - The storage type is not supported (only DenseF64 and DenseC64)
    /// - QR is used with `Canonical::Right`
    /// - The underlying algorithm fails
    ///
    /// # Default Implementation
    ///
    /// The default implementation converts to `TensorDynLen` via `to_tensor()`
    /// and then calls the standalone `factorize` function.
    fn factorize(
        &self,
        left_inds: &[DynIndex],
        options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult, FactorizeError> {
        let tensor = self
            .to_tensor()
            .map_err(|e| FactorizeError::ComputationError(e))?;
        crate::factorize::factorize(&tensor, left_inds, options)
    }
}

// Make trait objects cloneable
dyn_clone::clone_trait_object!(TensorLike);

// ============================================================================
// Helper methods on trait objects for downcasting
// ============================================================================

/// Extension trait for downcasting `dyn TensorLike` trait objects.
///
/// This provides convenient methods for runtime type checking and downcasting.
pub trait TensorLikeDowncast {
    /// Check if the underlying type is `T`.
    fn is<T: 'static>(&self) -> bool;

    /// Attempt to downcast to a reference of type `T`.
    fn downcast_ref<T: 'static>(&self) -> Option<&T>;
}

impl TensorLikeDowncast for dyn TensorLike {
    fn is<T: 'static>(&self) -> bool {
        self.as_any().is::<T>()
    }

    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

impl TensorLikeDowncast for dyn TensorLike + Send {
    fn is<T: 'static>(&self) -> bool {
        self.as_any().is::<T>()
    }

    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

impl TensorLikeDowncast for dyn TensorLike + Send + Sync {
    fn is<T: 'static>(&self) -> bool {
        self.as_any().is::<T>()
    }

    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

// ============================================================================
// Implementation for TensorDynLen
// ============================================================================

impl TensorLike for TensorDynLen {
    fn external_indices(&self) -> Vec<DynIndex> {
        // For TensorDynLen, all indices are external.
        self.indices.clone()
    }

    fn num_external_indices(&self) -> usize {
        self.indices.len()
    }

    fn to_tensor(&self) -> Result<TensorDynLen> {
        // TensorDynLen is already a tensor, just clone it
        Ok(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    // Use the default implementation of tensordot which calls to_tensor
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic compile-time check that the trait is object-safe
    fn _assert_object_safe() {
        fn _takes_trait_object(_obj: &dyn TensorLike) {
            // This won't compile if TensorLike is not object-safe
        }
    }
}
