//! TensorLike trait for unifying tensor types.
//!
//! This module provides a fully generic trait for tensor-like objects that expose
//! external indices and support contraction operations.
//!
//! # Design
//!
//! The trait is **fully generic** (monomorphic), meaning:
//! - No trait objects (`dyn TensorLike`)
//! - Uses associated type for `Index`
//! - All methods return `Self` instead of concrete types
//!
//! For heterogeneous tensor collections, use an enum wrapper.

use crate::any_scalar::AnyScalar;
use crate::IndexLike;
use anyhow::Result;
use std::fmt::Debug;

// ============================================================================
// Factorization types (non-generic, algorithm-specific)
// ============================================================================

use thiserror::Error;

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
/// Generic over the tensor type `T`.
#[derive(Debug, Clone)]
pub struct FactorizeResult<T: TensorLike> {
    /// Left factor tensor.
    pub left: T,
    /// Right factor tensor.
    pub right: T,
    /// Bond index connecting left and right factors.
    pub bond_index: T::Index,
    /// Singular values (only for SVD).
    pub singular_values: Option<Vec<f64>>,
    /// Rank of the factorization.
    pub rank: usize,
}

// ============================================================================
// TensorLike trait (fully generic)
// ============================================================================

/// Trait for tensor-like objects that expose external indices and support contraction.
///
/// This trait is **fully generic** (monomorphic), meaning it does not support
/// trait objects (`dyn TensorLike`). For heterogeneous tensor collections,
/// use an enum wrapper instead.
///
/// # Design Principles
///
/// - **Minimal interface**: Only external indices and explicit contraction
/// - **Fully generic**: Uses associated type for `Index`, returns `Self`
/// - **Stable ordering**: `external_indices()` returns indices in deterministic order
/// - **No trait objects**: Requires `Sized`, cannot use `dyn TensorLike`
///
/// # Example
///
/// ```ignore
/// use tensor4all_core::TensorLike;
///
/// fn contract_pair<T: TensorLike>(a: &T, b: &T) -> Result<T> {
///     let pairs = find_common_pairs(&a.external_indices(), &b.external_indices());
///     a.tensordot(b, &pairs)
/// }
/// ```
///
/// # Heterogeneous Collections
///
/// For mixing different tensor types, define an enum:
///
/// ```ignore
/// enum TensorNetwork {
///     Dense(TensorDynLen),
///     MPS(MatrixProductState),
/// }
/// ```
pub trait TensorLike: Sized + Clone + Debug + Send + Sync {
    /// The index type used by this tensor.
    type Index: IndexLike;

    /// Return flattened external indices for this object.
    ///
    /// # Ordering
    ///
    /// The ordering MUST be stable (deterministic). Implementations should:
    /// - Sort indices by their `id` field, or
    /// - Use insertion-ordered storage
    ///
    /// This ensures consistent behavior for hashing, serialization, and comparison.
    fn external_indices(&self) -> Vec<Self::Index>;

    /// Number of external indices.
    ///
    /// Default implementation calls `external_indices().len()`, but implementations
    /// SHOULD override this for efficiency when the count can be computed without
    /// allocating the full index list.
    fn num_external_indices(&self) -> usize {
        self.external_indices().len()
    }

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
    /// A new tensor with the index replaced.
    fn replaceind(&self, old_index: &Self::Index, new_index: &Self::Index) -> Result<Self>;

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
    /// A new tensor with the indices replaced.
    fn replaceinds(
        &self,
        old_indices: &[Self::Index],
        new_indices: &[Self::Index],
    ) -> Result<Self>;

    /// Explicit contraction between two tensor-like objects.
    ///
    /// This performs binary contraction over the specified index pairs.
    /// Each pair `(idx_self, idx_other)` specifies:
    /// - An index from `self` to contract
    /// - An index from `other` to contract with
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to contract with (same type)
    /// * `pairs` - List of (self_index, other_index) pairs to contract
    ///
    /// # Returns
    ///
    /// A new tensor representing the contracted result.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `pairs` is empty
    /// - An index in `pairs` doesn't exist in the corresponding object
    /// - Index dimensions don't match
    /// - There are common indices not in `pairs` (ambiguous contraction)
    fn tensordot(
        &self,
        other: &Self,
        pairs: &[(Self::Index, Self::Index)],
    ) -> Result<Self>;

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
    fn factorize(
        &self,
        left_inds: &[Self::Index],
        options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError>;

    /// Tensor conjugate operation.
    ///
    /// This is a generalized conjugate operation that depends on the tensor type:
    /// - For dense tensors (TensorDynLen): element-wise complex conjugate
    /// - For symmetric tensors: tensor conjugate considering symmetry sectors
    ///
    /// This operation is essential for computing inner products and overlaps
    /// in tensor network algorithms like fitting.
    ///
    /// # Returns
    /// A new tensor representing the tensor conjugate.
    fn conj(&self) -> Self;

    /// Direct sum of two tensors along specified index pairs.
    ///
    /// For tensors A and B with indices to be summed specified as pairs,
    /// creates a new tensor C where each paired index has dimension = dim_A + dim_B.
    /// Non-paired indices must match exactly between A and B (same ID).
    ///
    /// # Arguments
    ///
    /// * `other` - Second tensor
    /// * `pairs` - Pairs of (self_index, other_index) to be summed. Each pair creates
    ///   a new index in the result with dimension = dim(self_index) + dim(other_index).
    ///
    /// # Returns
    ///
    /// A `DirectSumResult` containing the result tensor and new indices created
    /// for the summed dimensions (one per pair).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A has indices [i, j] with dims [2, 3]
    /// // B has indices [i, k] with dims [2, 4]
    /// // If we pair (j, k), result has indices [i, m] with dims [2, 7]
    /// // where m is a new index with dim = 3 + 4 = 7
    /// let result = a.direct_sum(&b, &[(j, k)])?;
    /// ```
    fn direct_sum(
        &self,
        other: &Self,
        pairs: &[(Self::Index, Self::Index)],
    ) -> Result<DirectSumResult<Self>>;

    /// Outer product (tensor product) of two tensors.
    ///
    /// Computes the tensor product of `self` and `other`, resulting in a tensor
    /// with all indices from both tensors. No indices are contracted.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to compute outer product with
    ///
    /// # Returns
    ///
    /// A new tensor representing the outer product.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensors have common indices (by ID).
    /// Use `tensordot` for contraction when indices overlap.
    fn outer_product(&self, other: &Self) -> Result<Self>;

    /// Compute the squared Frobenius norm of the tensor.
    ///
    /// The squared Frobenius norm is defined as the sum of squared absolute values
    /// of all tensor elements: `||T||_F^2 = sum_i |T_i|^2`.
    ///
    /// This is used for computing norms in tensor network algorithms,
    /// convergence checks, and normalization.
    ///
    /// # Returns
    /// The squared Frobenius norm as a non-negative f64.
    fn norm_squared(&self) -> f64;

    /// Permute tensor indices to match the specified order.
    ///
    /// This reorders the tensor's axes to match the order specified by `new_order`.
    /// The indices in `new_order` are matched by ID with the tensor's current indices.
    ///
    /// # Arguments
    ///
    /// * `new_order` - The desired order of indices (matched by ID)
    ///
    /// # Returns
    ///
    /// A new tensor with permuted indices.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of indices doesn't match
    /// - An index ID in `new_order` is not found in the tensor
    fn permuteinds(&self, new_order: &[Self::Index]) -> Result<Self>;

    /// Contract multiple tensors using einsum-style contraction.
    ///
    /// This method contracts 2 or more tensors over their common indices.
    /// Indices appearing in exactly two tensors are contracted.
    /// Indices appearing in only one tensor appear in the output.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensors to contract (must have length >= 1)
    ///
    /// # Returns
    ///
    /// A new tensor representing the contracted result.
    ///
    /// # Behavior by N
    /// - N=0: Error
    /// - N=1: Clone of input
    /// - N=2: Uses `tensordot` over common indices
    /// - N>=3: Recursively contracts pairs
    fn contract_einsum(tensors: &[Self]) -> Result<Self>;

    // ========================================================================
    // Vector space operations (for Krylov solvers)
    // ========================================================================

    /// Compute a linear combination: `a * self + b * other`.
    ///
    /// This is the fundamental vector space operation.
    fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self>;

    /// Scalar multiplication.
    fn scale(&self, scalar: AnyScalar) -> Self;

    /// Inner product (dot product) of two tensors.
    ///
    /// Computes `⟨self, other⟩ = Σ conj(self)_i * other_i`.
    fn inner_product(&self, other: &Self) -> Result<AnyScalar>;

    /// Compute the Frobenius norm of the tensor.
    fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
}

/// Result of direct sum operation.
#[derive(Debug, Clone)]
pub struct DirectSumResult<T: TensorLike> {
    /// The resulting tensor from direct sum.
    pub tensor: T,
    /// New indices created for the summed dimensions (one per pair).
    pub new_indices: Vec<T::Index>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time check that TensorLike requires Sized (no dyn TensorLike)
    fn _assert_sized<T: TensorLike>() {
        // This confirms T: Sized is required
    }
}
