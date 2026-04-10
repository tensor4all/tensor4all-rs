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
use crate::tensor_index::TensorIndex;
use anyhow::Result;
use std::fmt::Debug;

// ============================================================================
// Factorization types (non-generic, algorithm-specific)
// ============================================================================

use thiserror::Error;

/// Error type for factorize operations.
#[derive(Debug, Error)]
pub enum FactorizeError {
    /// Factorization computation failed.
    #[error("Factorization failed: {0}")]
    ComputationError(
        /// The underlying error
        #[from]
        anyhow::Error,
    ),
    /// Invalid relative tolerance value (must be finite and non-negative).
    #[error("Invalid rtol value: {0}. rtol must be finite and non-negative.")]
    InvalidRtol(
        /// The invalid rtol value
        f64,
    ),
    /// The storage type is not supported for this operation.
    #[error("Unsupported storage type: {0}")]
    UnsupportedStorage(
        /// Description of the unsupported storage type
        &'static str,
    ),
    /// The canonical direction is not supported for this algorithm.
    #[error("Unsupported canonical direction for this algorithm: {0}")]
    UnsupportedCanonical(
        /// Description of the unsupported canonical direction
        &'static str,
    ),
    /// Error from SVD operation.
    #[error("SVD error: {0}")]
    SvdError(
        /// The underlying SVD error
        #[from]
        crate::svd::SvdError,
    ),
    /// Error from QR operation.
    #[error("QR error: {0}")]
    QrError(
        /// The underlying QR error
        #[from]
        crate::qr::QrError,
    ),
    /// Error from matrix CI operation.
    #[error("Matrix CI error: {0}")]
    MatrixCIError(
        /// The underlying matrix CI error
        #[from]
        tensor4all_tcicore::MatrixCIError,
    ),
}

/// Factorization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FactorizeAlg {
    /// Singular Value Decomposition.
    #[default]
    SVD,
    /// QR decomposition.
    QR,
    /// Rank-revealing LU decomposition.
    LU,
    /// Cross Interpolation (LU-based).
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
// Contraction types
// ============================================================================

/// Specifies which tensor pairs are allowed to contract.
///
/// This enum controls which tensor pairs can have their indices contracted
/// in multi-tensor contraction operations. This is useful for tensor networks
/// where the graph structure determines which tensors are connected.
///
/// # Example
///
/// ```
/// use tensor4all_core::{AllowedPairs, DynIndex, TensorDynLen, TensorLike};
///
/// # fn main() -> anyhow::Result<()> {
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let k = DynIndex::new_dyn(2);
///
/// let a = TensorDynLen::from_dense(
///     vec![i.clone(), j.clone()],
///     vec![1.0, 0.0, 0.0, 1.0],
/// )?;
/// let b = TensorDynLen::from_dense(
///     vec![j.clone(), k.clone()],
///     vec![1.0, 2.0, 3.0, 4.0],
/// )?;
/// let c = TensorDynLen::from_dense(vec![k.clone()], vec![1.0, 10.0])?;
///
/// let tensor_refs: Vec<&TensorDynLen> = vec![&a, &b, &c];
/// let all = <TensorDynLen as TensorLike>::contract(&tensor_refs, AllowedPairs::All)?;
///
/// let edges = vec![(0, 1), (1, 2)];
/// let specified =
///     <TensorDynLen as TensorLike>::contract(&tensor_refs, AllowedPairs::Specified(&edges))?;
///
/// assert_eq!(all.dims(), vec![2]);
/// assert!(all.sub(&specified)?.maxabs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub enum AllowedPairs<'a> {
    /// All tensor pairs are allowed to contract.
    ///
    /// Indices with matching IDs across any two tensors will be contracted.
    /// This is the default behavior, equivalent to ITensor's `*` operator.
    All,
    /// Only specified tensor pairs are allowed to contract.
    ///
    /// Each pair is `(tensor_idx_a, tensor_idx_b)` into the input tensor slice.
    /// Indices are only contracted if they belong to an allowed pair.
    ///
    /// This is useful for tensor networks where the graph structure
    /// determines which tensors are connected (e.g., TreeTN edges).
    Specified(&'a [(usize, usize)]),
}

/// Linearization order used when fusing or unfusing multiple logical indices
/// into one physical index.
///
/// This matters for exact reshape-style operations such as replacing one fused
/// index with several unfused indices.
///
/// # Examples
///
/// ```
/// use tensor4all_core::LinearizationOrder;
///
/// assert_eq!(LinearizationOrder::ColumnMajor.as_str(), "column-major");
/// assert_eq!(LinearizationOrder::RowMajor.as_str(), "row-major");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearizationOrder {
    /// First index changes fastest.
    ColumnMajor,
    /// Last index changes fastest.
    RowMajor,
}

impl LinearizationOrder {
    /// Return a short human-readable description.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::LinearizationOrder;
    ///
    /// assert_eq!(LinearizationOrder::ColumnMajor.as_str(), "column-major");
    /// ```
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ColumnMajor => "column-major",
            Self::RowMajor => "row-major",
        }
    }
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
/// - **Minimal interface**: Only external indices and automatic contraction
/// - **Fully generic**: Uses associated type for `Index`, returns `Self`
/// - **Stable ordering**: `external_indices()` returns indices in deterministic order
/// - **No trait objects**: Requires `Sized`, cannot use `dyn TensorLike`
///
/// # Example
///
/// ```
/// use tensor4all_core::{AllowedPairs, DynIndex, TensorDynLen, TensorLike};
///
/// fn contract_pair(a: &TensorDynLen, b: &TensorDynLen) -> anyhow::Result<TensorDynLen> {
///     Ok(<TensorDynLen as TensorLike>::contract(&[a, b], AllowedPairs::All)?)
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let a = TensorDynLen::from_dense(
///     vec![i.clone(), j.clone()],
///     vec![1.0, 0.0, 0.0, 1.0],
/// )?;
/// let b = TensorDynLen::from_dense(vec![j.clone()], vec![2.0, 3.0])?;
///
/// let result = contract_pair(&a, &b)?;
/// assert_eq!(result.to_vec::<f64>()?, vec![2.0, 3.0]);
/// # Ok(())
/// # }
/// ```
///
/// # Heterogeneous Collections
///
/// For mixing different tensor types, define an enum:
///
/// ```
/// use tensor4all_core::{block_tensor::BlockTensor, DynIndex, TensorDynLen};
///
/// let i = DynIndex::new_dyn(2);
/// let dense = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
/// let block = BlockTensor::new(vec![dense.clone()], (1, 1));
///
/// enum TensorNetwork {
///     Dense(TensorDynLen),
///     Block(BlockTensor<TensorDynLen>),
/// }
///
/// let network = TensorNetwork::Block(block);
/// assert!(matches!(network, TensorNetwork::Block(_)));
/// ```
///
/// # Supertrait
///
/// `TensorLike` extends `TensorIndex`, which provides:
/// - `external_indices()` - Get all external indices
/// - `num_external_indices()` - Count external indices
/// - `replaceind()` / `replaceinds()` - Replace indices
///
/// This separation allows tensor networks (like `TreeTN`) to implement
/// index operations without implementing contraction/factorization.
pub trait TensorLike: TensorIndex {
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
        left_inds: &[<Self as TensorIndex>::Index],
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
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let j = DynIndex::new_dyn(2);
    /// let k = DynIndex::new_dyn(3);
    ///
    /// let a = TensorDynLen::from_dense(vec![j.clone()], vec![1.0, 2.0])?;
    /// let b = TensorDynLen::from_dense(vec![k.clone()], vec![3.0, 4.0, 5.0])?;
    /// let result = a.direct_sum(&b, &[(j.clone(), k.clone())])?;
    ///
    /// assert_eq!(result.new_indices.len(), 1);
    /// assert_eq!(result.tensor.dims(), vec![5]);
    /// assert_eq!(result.tensor.to_vec::<f64>()?, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// # Ok(())
    /// # }
    /// ```
    fn direct_sum(
        &self,
        other: &Self,
        pairs: &[(<Self as TensorIndex>::Index, <Self as TensorIndex>::Index)],
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
    fn permuteinds(&self, new_order: &[<Self as TensorIndex>::Index]) -> Result<Self>;

    /// Contract multiple tensors over their contractable indices.
    ///
    /// This method contracts 2 or more tensors. Pairs of indices that satisfy
    /// `is_contractable()` (same ID, same dimension, compatible ConjState)
    /// are contracted based on the `allowed` parameter.
    ///
    /// Handles disconnected tensor graphs automatically by:
    /// 1. Finding connected components based on contractable indices
    /// 2. Contracting each connected component separately
    /// 3. Combining results using outer product
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensor references to contract (must have length >= 1)
    /// * `allowed` - Specifies which tensor pairs can have their indices contracted:
    ///   - `AllowedPairs::All`: Contract all contractable index pairs (default behavior)
    ///   - `AllowedPairs::Specified(&[(i, j)])`: Only contract indices between specified tensor pairs
    ///
    /// # Returns
    ///
    /// A new tensor representing the contracted result.
    /// If tensors form disconnected components, they are combined via outer product.
    ///
    /// # Behavior by N
    /// - N=0: Error
    /// - N=1: Clone of input
    /// - N>=2: Contract connected components, combine with outer product
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No tensors are provided
    /// - `AllowedPairs::Specified` contains a pair with no contractable indices
    ///
    /// # Example
    ///
    /// ```
    /// use tensor4all_core::{AllowedPairs, DynIndex, TensorDynLen, TensorLike};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let k = DynIndex::new_dyn(2);
    ///
    /// let a = TensorDynLen::from_dense(
    ///     vec![i.clone(), j.clone()],
    ///     vec![1.0, 0.0, 0.0, 1.0],
    /// )?;
    /// let b = TensorDynLen::from_dense(
    ///     vec![j.clone(), k.clone()],
    ///     vec![1.0, 2.0, 3.0, 4.0],
    /// )?;
    /// let c = TensorDynLen::from_dense(vec![k.clone()], vec![1.0, 10.0])?;
    ///
    /// let all = <TensorDynLen as TensorLike>::contract(&[&a, &b, &c], AllowedPairs::All)?;
    /// let specified = <TensorDynLen as TensorLike>::contract(
    ///     &[&a, &b, &c],
    ///     AllowedPairs::Specified(&[(0, 1), (1, 2)]),
    /// )?;
    ///
    /// assert_eq!(all.dims(), vec![2]);
    /// assert!(all.sub(&specified)?.maxabs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    fn contract(tensors: &[&Self], allowed: AllowedPairs<'_>) -> Result<Self>;

    /// Contract multiple tensors that must form a connected graph.
    ///
    /// This is the core contraction method that requires all tensors to be
    /// connected through contractable indices. Use [`Self::contract`] if you want
    /// automatic handling of disconnected components via outer product.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensor references to contract (must form a connected graph)
    /// * `allowed` - Specifies which tensor pairs can have their indices contracted
    ///
    /// # Returns
    ///
    /// A new tensor representing the contracted result.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No tensors are provided
    /// - The tensors form a disconnected graph
    ///
    /// # Example
    ///
    /// ```
    /// use tensor4all_core::{AllowedPairs, DynIndex, TensorDynLen, TensorLike};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let k = DynIndex::new_dyn(2);
    ///
    /// let a = TensorDynLen::from_dense(
    ///     vec![i.clone(), j.clone()],
    ///     vec![1.0, 0.0, 0.0, 1.0],
    /// )?;
    /// let b = TensorDynLen::from_dense(
    ///     vec![j.clone(), k.clone()],
    ///     vec![1.0, 2.0, 3.0, 4.0],
    /// )?;
    /// let c = TensorDynLen::from_dense(vec![k.clone()], vec![1.0, 10.0])?;
    ///
    /// let result = TensorDynLen::contract_connected(&[&a, &b, &c], AllowedPairs::All)?;
    /// assert_eq!(result.dims(), vec![2]);
    /// # Ok(())
    /// # }
    /// ```
    fn contract_connected(tensors: &[&Self], allowed: AllowedPairs<'_>) -> Result<Self>;

    // ========================================================================
    // Vector space operations (for Krylov solvers)
    // ========================================================================

    /// Compute a linear combination: `a * self + b * other`.
    ///
    /// This is the fundamental vector space operation.
    fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self>;

    /// Scalar multiplication.
    fn scale(&self, scalar: AnyScalar) -> Result<Self>;

    /// Inner product (dot product) of two tensors.
    ///
    /// Computes `⟨self, other⟩ = Σ conj(self)_i * other_i`.
    fn inner_product(&self, other: &Self) -> Result<AnyScalar>;

    /// Compute the Frobenius norm of the tensor.
    fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Maximum absolute value of all elements (L-infinity norm).
    fn maxabs(&self) -> f64;

    /// Element-wise subtraction: `self - other`.
    ///
    /// Indices are automatically permuted to match `self`'s order via `axpby`.
    fn sub(&self, other: &Self) -> Result<Self> {
        self.axpby(AnyScalar::new_real(1.0), other, AnyScalar::new_real(-1.0))
    }

    /// Negate all elements: `-self`.
    fn neg(&self) -> Result<Self> {
        self.scale(AnyScalar::new_real(-1.0))
    }

    /// Approximate equality check (Julia `isapprox` semantics).
    ///
    /// Returns `true` if `||self - other|| <= max(atol, rtol * max(||self||, ||other||))`.
    fn isapprox(&self, other: &Self, atol: f64, rtol: f64) -> bool {
        let diff = match self.sub(other) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let diff_norm = diff.norm();
        diff_norm <= atol.max(rtol * self.norm().max(other.norm()))
    }

    /// Validate structural consistency of this tensor.
    ///
    /// The default implementation does nothing (always succeeds).
    /// Types with internal structure (for example, block-sparse tensors) can override
    /// this to check invariants such as index sharing between blocks.
    fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Create a diagonal (Kronecker delta) tensor for a single index pair.
    ///
    /// Creates a 2D tensor `T[i, o]` where `T[i, o] = δ_{i,o}` (1 if i==o, 0 otherwise).
    ///
    /// # Arguments
    ///
    /// * `input_index` - Input index
    /// * `output_index` - Output index (must have same dimension as input)
    ///
    /// # Returns
    ///
    /// A 2D tensor with shape `[dim, dim]` representing the identity matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    ///
    /// # Example
    ///
    /// For dimension 2:
    /// ```text
    /// diagonal(i, o) = [[1, 0], [0, 1]]
    /// ```
    fn diagonal(
        input_index: &<Self as TensorIndex>::Index,
        output_index: &<Self as TensorIndex>::Index,
    ) -> Result<Self>;

    /// Create a delta (identity) tensor as outer product of diagonals.
    ///
    /// For paired indices `(i1, o1), (i2, o2), ...`, creates a tensor where:
    /// `T[i1, o1, i2, o2, ...] = δ_{i1,o1} × δ_{i2,o2} × ...`
    ///
    /// This is computed as the outer product of individual diagonal tensors.
    ///
    /// # Arguments
    ///
    /// * `input_indices` - Input indices
    /// * `output_indices` - Output indices (must have same length and matching dimensions)
    ///
    /// # Returns
    ///
    /// A tensor representing the identity operator on the given index space.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Number of input and output indices don't match
    /// - Dimensions of paired indices don't match
    ///
    /// # Example
    ///
    /// For a single index pair with dimension 2:
    /// ```text
    /// delta([i], [o]) = [[1, 0], [0, 1]]
    /// ```
    fn delta(
        input_indices: &[<Self as TensorIndex>::Index],
        output_indices: &[<Self as TensorIndex>::Index],
    ) -> Result<Self> {
        // Validate same number of input and output indices
        if input_indices.len() != output_indices.len() {
            return Err(anyhow::anyhow!(
                "Number of input indices ({}) must match output indices ({})",
                input_indices.len(),
                output_indices.len()
            ));
        }

        if input_indices.is_empty() {
            // Return a scalar tensor with value 1.0
            return Self::scalar_one();
        }

        // Build as outer product of diagonal tensors
        let mut result = Self::diagonal(&input_indices[0], &output_indices[0])?;
        for (inp, out) in input_indices[1..].iter().zip(output_indices[1..].iter()) {
            let diag = Self::diagonal(inp, out)?;
            result = result.outer_product(&diag)?;
        }
        Ok(result)
    }

    /// Create a scalar tensor with value 1.0.
    ///
    /// This is used as the identity element for outer products.
    fn scalar_one() -> Result<Self>;

    /// Create a tensor filled with 1.0 for the given indices.
    ///
    /// This is useful for adding indices to tensors via outer product
    /// without changing tensor values (since multiplying by 1 is identity).
    ///
    /// # Example
    /// To add a dummy index `l` to tensor `T`:
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let i = DynIndex::new_dyn(2);
    /// let l = DynIndex::new_dyn(3);
    /// let t = TensorDynLen::from_dense(vec![i.clone()], vec![2.0, 4.0])?;
    ///
    /// let ones = TensorDynLen::ones(&[l.clone()])?;
    /// let t_with_l = t.outer_product(&ones)?;
    ///
    /// assert_eq!(t_with_l.dims(), vec![2, 3]);
    /// assert_eq!(t_with_l.to_vec::<f64>()?, vec![2.0, 4.0, 2.0, 4.0, 2.0, 4.0]);
    /// # Ok(())
    /// # }
    /// ```
    fn ones(indices: &[<Self as TensorIndex>::Index]) -> Result<Self>;

    /// Create a one-hot tensor with value 1.0 at the specified index positions.
    ///
    /// Similar to ITensors.jl's `onehot(i => 1, j => 2)`.
    ///
    /// # Arguments
    /// * `index_vals` - Pairs of (Index, 0-indexed position)
    ///
    /// # Errors
    /// Returns error if any value >= corresponding index dimension.
    fn onehot(index_vals: &[(<Self as TensorIndex>::Index, usize)]) -> Result<Self>;
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
mod tests;
