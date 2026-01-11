use crate::index_like::IndexLike;
use crate::index_ops::{common_ind_positions, prepare_contraction, prepare_contraction_pairs};
use crate::defaults::{DynId, DynIndex, TensorData};
use crate::storage::{
    contract_storage, storage_to_dtensor, AnyScalar, Storage, StorageScalar, SumFromStorage,
};
use anyhow::Result;
use tensor4all_tensorbackend::mdarray::DTensor;
use num_complex::Complex64;
use std::collections::HashSet;
use std::ops::Mul;
use std::sync::Arc;

/// Compute the permutation array from original indices to new indices.
///
/// This function finds the mapping from new indices to original indices by
/// matching index IDs. The result is a permutation array `perm` such that
/// `new_indices[i]` corresponds to `original_indices[perm[i]]`.
///
/// # Arguments
/// * `original_indices` - The original indices in their current order
/// * `new_indices` - The desired new indices order (must be a permutation of original_indices)
///
/// # Returns
/// A `Vec<usize>` representing the permutation: `perm[i]` is the position in
/// `original_indices` of the index that should be at position `i` in `new_indices`.
///
/// # Panics
/// Panics if any index ID in `new_indices` doesn't match an index in `original_indices`,
/// or if there are duplicate indices in `new_indices`.
///
/// # Example
/// ```
/// use tensor4all_core::tensor::compute_permutation_from_indices;
/// use tensor4all_core::DynIndex;
///
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let original = vec![i.clone(), j.clone()];
/// let new_order = vec![j.clone(), i.clone()];
///
/// let perm = compute_permutation_from_indices(&original, &new_order);
/// assert_eq!(perm, vec![1, 0]);  // j is at position 1, i is at position 0
/// ```
pub fn compute_permutation_from_indices(
    original_indices: &[DynIndex],
    new_indices: &[DynIndex],
) -> Vec<usize> {
    assert_eq!(
        new_indices.len(),
        original_indices.len(),
        "new_indices length must match original_indices length"
    );

    let mut perm = Vec::with_capacity(new_indices.len());
    let mut used = std::collections::HashSet::new();

    for new_idx in new_indices {
        // Find the position of this index in the original indices
        // DynIndex implements Eq, so we can compare directly
        let pos = original_indices
            .iter()
            .position(|old_idx| old_idx == new_idx)
            .expect("new_indices must be a permutation of original_indices");

        if used.contains(&pos) {
            panic!("duplicate index in new_indices");
        }
        used.insert(pos);
        perm.push(pos);
    }

    perm
}

/// Trait for accessing tensor fields.
///
/// This trait provides access to the internal fields of tensor types,
/// allowing generic code to work with tensors without knowing the exact type.
pub trait TensorAccess {
    /// Get a reference to the indices.
    fn indices(&self) -> &[DynIndex];

    /// Get a reference to the underlying data (Storage).
    fn data(&self) -> &Storage;
}

/// Tensor with dynamic rank (number of indices) and dynamic scalar type.
///
/// This is a concrete type using `DynIndex` (= `Index<DynId, TagSet>`).
///
/// Internally contains a `TensorData` which supports lazy operations like
/// outer product and permutation.
pub struct TensorDynLen {
    /// Full index information (includes tags and other metadata).
    pub indices: Vec<DynIndex>,
    /// Dimension sizes (same order as indices).
    pub dims: Vec<usize>,
    /// Internal lazy tensor data representation.
    data: TensorData,
}

impl TensorAccess for TensorDynLen {
    fn indices(&self) -> &[DynIndex] {
        &self.indices
    }

    fn data(&self) -> &Storage {
        // For simple tensors, return the underlying storage directly
        // For lazy tensors, this would need materialization
        TensorDynLen::storage(self)
    }
}

impl TensorDynLen {
    /// Create a new tensor with dynamic rank.
    ///
    /// # Panics
    /// Panics if the storage is Diag and not all indices have the same dimension.
    /// Panics if there are duplicate indices.
    pub fn new(indices: Vec<DynIndex>, dims: Vec<usize>, storage: Arc<Storage>) -> Self {
        assert_eq!(
            indices.len(),
            dims.len(),
            "indices and dims must have the same length"
        );

        // Check for duplicate indices using IndexLike's Eq
        {
            let mut seen = HashSet::new();
            for idx in &indices {
                if !seen.insert(idx.clone()) {
                    panic!("Tensor indices must all be unique (no duplicate IDs)");
                }
            }
        }

        // Validate DiagTensor: all indices must have the same dimension
        if storage.as_ref().is_diag() && !dims.is_empty() {
            let first_dim = dims[0];
            for (i, &dim) in dims.iter().enumerate() {
                assert_eq!(
                    dim, first_dim,
                    "DiagTensor requires all indices to have the same dimension, but dims[{}] = {} != dims[0] = {}",
                    i, dim, first_dim
                );
            }
        }

        // Create TensorData from storage
        let index_ids: Vec<DynId> = indices.iter().map(|idx| idx.id().clone()).collect();
        let data = TensorData::new(storage, index_ids, dims.clone());

        Self {
            indices,
            dims,
            data,
        }
    }

    /// Create a new tensor with dynamic rank, automatically computing dimensions from indices.
    ///
    /// This is a convenience constructor that extracts dimensions from indices using `IndexLike::dim()`.
    ///
    /// # Panics
    /// Panics if the storage is Diag and not all indices have the same dimension.
    /// Panics if there are duplicate indices.
    pub fn from_indices(indices: Vec<DynIndex>, storage: Arc<Storage>) -> Self {
        let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
        Self::new(indices, dims, storage)
    }

    /// Check if this tensor is simple (single storage, no lazy operations pending).
    pub fn is_simple(&self) -> bool {
        self.data.is_simple()
    }

    /// Get the storage (for simple tensors only).
    ///
    /// # Panics
    /// Panics if the tensor has pending lazy operations.
    /// Use `try_storage()` or `materialize_storage()` for safe access.
    pub fn storage(&self) -> &Arc<Storage> {
        self.data
            .storage()
            .expect("storage() called on lazy tensor - use try_storage() or materialize_storage()")
    }

    /// Try to get the storage without materializing.
    ///
    /// Returns `None` if the tensor has pending lazy operations.
    /// Use `materialize_storage()` to force materialization.
    pub fn try_storage(&self) -> Option<&Arc<Storage>> {
        self.data.storage()
    }

    /// Get the storage, materializing if necessary.
    ///
    /// For simple tensors, returns the underlying storage without copying.
    /// For lazy tensors, performs any pending operations and returns the result.
    pub fn materialize_storage(&self) -> Result<Arc<Storage>> {
        if self.data.is_simple() {
            Ok(self.data.storage().unwrap().clone())
        } else {
            let (storage, _dims) = self.data.materialize()?;
            Ok(storage)
        }
    }

    /// Get the internal TensorData reference.
    pub fn tensor_data(&self) -> &TensorData {
        &self.data
    }

    /// Create TensorDynLen directly from TensorData and indices.
    ///
    /// This is an internal constructor for building tensors from lazy operations.
    fn from_data(indices: Vec<DynIndex>, dims: Vec<usize>, data: TensorData) -> Self {
        Self { indices, dims, data }
    }

    /// Sum all elements, returning `AnyScalar`.
    pub fn sum(&self) -> AnyScalar {
        AnyScalar::sum_from_storage(self.storage())
    }

    /// Sum all elements as f64.
    pub fn sum_f64(&self) -> f64 {
        f64::sum_from_storage(self.storage())
    }

    /// Extract the scalar value from a 0-dimensional tensor (or 1-element tensor).
    ///
    /// This is similar to Julia's `only()` function.
    ///
    /// # Panics
    ///
    /// Panics if the tensor has more than one element.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor4all_core::{TensorDynLen, Storage, AnyScalar};
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use std::sync::Arc;
    ///
    /// // Create a scalar tensor (0 dimensions, 1 element)
    /// let indices: Vec<Index<DynId>> = vec![];
    /// let dims: Vec<usize> = vec![];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![42.0])));
    /// let tensor: TensorDynLen = TensorDynLen::new(indices, dims, storage);
    ///
    /// assert_eq!(tensor.only().real(), 42.0);
    /// ```
    pub fn only(&self) -> AnyScalar {
        let total_size: usize = self.dims.iter().product();
        assert!(
            total_size == 1 || (self.dims.is_empty()),
            "only() requires a scalar tensor (1 element), got {} elements with dims {:?}",
            if self.dims.is_empty() { 1 } else { total_size },
            self.dims
        );
        self.sum()
    }

    /// Permute the tensor dimensions using the given new indices order.
    ///
    /// This is the main permutation method that takes the desired new indices
    /// and automatically computes the corresponding permutation of dimensions
    /// and data. The new indices must be a permutation of the original indices
    /// (matched by ID).
    ///
    /// # Arguments
    /// * `new_indices` - The desired new indices order. Must be a permutation
    ///   of `self.indices` (matched by ID).
    ///
    /// # Panics
    /// Panics if `new_indices.len() != self.indices.len()`, if any index ID
    /// doesn't match, or if there are duplicate indices.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core::Storage;
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// // Create a 2×3 tensor
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let indices = vec![i.clone(), j.clone()];
    /// let dims = vec![2, 3];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor: TensorDynLen = TensorDynLen::new(indices, dims, storage);
    ///
    /// // Permute to 3×2: swap the two dimensions by providing new indices order
    /// let permuted = tensor.permute_indices(&[j, i]);
    /// assert_eq!(permuted.dims, vec![3, 2]);
    /// ```
    pub fn permute_indices(&self, new_indices: &[DynIndex]) -> Self {
        // Compute permutation by matching IDs
        let perm = compute_permutation_from_indices(&self.indices, new_indices);

        // Compute new dims from new indices
        let new_dims: Vec<usize> = new_indices.iter().map(|idx| idx.dim()).collect();

        // Permute storage data using the computed permutation
        let new_storage = Arc::new(self.storage().permute_storage(&self.dims, &perm));

        Self::new(new_indices.to_vec(), new_dims, new_storage)
    }

    /// Permute the tensor dimensions, returning a new tensor.
    ///
    /// This method reorders the indices, dimensions, and data according to the
    /// given permutation. The permutation specifies which old axis each new
    /// axis corresponds to: `new_axis[i] = old_axis[perm[i]]`.
    ///
    /// # Arguments
    /// * `perm` - The permutation: `perm[i]` is the old axis index for new axis `i`
    ///
    /// # Panics
    /// Panics if `perm.len() != self.dims.len()` or if the permutation is invalid.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core::Storage;
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// // Create a 2×3 tensor
    /// let indices = vec![
    ///     Index::new_dyn(2),
    ///     Index::new_dyn(3),
    /// ];
    /// let dims = vec![2, 3];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor: TensorDynLen = TensorDynLen::new(indices, dims, storage);
    ///
    /// // Permute to 3×2: swap the two dimensions
    /// let permuted = tensor.permute(&[1, 0]);
    /// assert_eq!(permuted.dims, vec![3, 2]);
    /// ```
    pub fn permute(&self, perm: &[usize]) -> Self {
        assert_eq!(
            perm.len(),
            self.dims.len(),
            "permutation length must match tensor rank"
        );

        // Permute indices and dims
        let new_indices: Vec<DynIndex> =
            perm.iter().map(|&i| self.indices[i].clone()).collect();
        let new_dims: Vec<usize> = perm.iter().map(|&i| self.dims[i]).collect();

        // Permute storage data
        let new_storage = Arc::new(self.storage().permute_storage(&self.dims, perm));

        Self::new(new_indices, new_dims, new_storage)
    }

    /// Contract this tensor with another tensor along common indices.
    ///
    /// This method finds common indices between `self` and `other`, then contracts
    /// along those indices. The result tensor contains all non-contracted indices
    /// from both tensors, with indices from `self` appearing first, followed by
    /// indices from `other` that are not common.
    ///
    /// # Arguments
    /// * `other` - The tensor to contract with
    ///
    /// # Returns
    /// A new tensor resulting from the contraction.
    ///
    /// # Panics
    /// Panics if there are no common indices, if common indices have mismatched
    /// dimensions, or if storage types don't match.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core::Storage;
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// // Create two tensors: A[i, j] and B[j, k]
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let k = Index::new_dyn(4);
    ///
    /// let indices_a = vec![i.clone(), j.clone()];
    /// let dims_a = vec![2, 3];
    /// let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor_a: TensorDynLen = TensorDynLen::new(indices_a, dims_a, storage_a);
    ///
    /// let indices_b = vec![j.clone(), k.clone()];
    /// let dims_b = vec![3, 4];
    /// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 12])));
    /// let tensor_b: TensorDynLen = TensorDynLen::new(indices_b, dims_b, storage_b);
    ///
    /// // Contract along j: result is C[i, k]
    /// let result = tensor_a.contract_einsum(&tensor_b);
    /// assert_eq!(result.dims, vec![2, 4]);
    /// ```
    pub fn contract_einsum(&self, other: &Self) -> Self {
        let spec = prepare_contraction(&self.indices, &self.dims, &other.indices, &other.dims)
            .expect("contraction preparation failed");

        let result_storage = Arc::new(contract_storage(
            self.storage(),
            &self.dims,
            &spec.axes_a,
            other.storage(),
            &other.dims,
            &spec.axes_b,
            &spec.result_dims,
        ));

        Self::new(spec.result_indices, spec.result_dims, result_storage)
    }

    /// Contract this tensor with another tensor along explicitly specified index pairs.
    ///
    /// Similar to NumPy's `tensordot`, this method contracts only along the explicitly
    /// specified pairs of indices. Unlike `contract()` which automatically contracts
    /// all common indices, `tensordot` gives you explicit control over which indices
    /// to contract.
    ///
    /// # Arguments
    /// * `other` - The tensor to contract with
    /// * `pairs` - Pairs of indices to contract: `(index_from_self, index_from_other)`
    ///
    /// # Returns
    /// A new tensor resulting from the contraction, or an error if:
    /// - Any specified index is not found in the respective tensor
    /// - Dimensions don't match for any pair
    /// - The same axis is specified multiple times in `self` or `other`
    /// - There are common indices (same ID) that are not in the contraction pairs
    ///   (batch contraction is not yet implemented)
    ///
    /// # Future: Batch Contraction
    /// In a future version, common indices not specified in `pairs` will be treated
    /// as batch dimensions (like batched GEMM). Currently, this case returns an error.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core::Storage;
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// // Create two tensors: A[i, j] and B[k, l] where j and k have same dimension but different IDs
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let k = Index::new_dyn(3);  // Same dimension as j, but different ID
    /// let l = Index::new_dyn(4);
    ///
    /// let indices_a = vec![i.clone(), j.clone()];
    /// let dims_a = vec![2, 3];
    /// let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor_a: TensorDynLen = TensorDynLen::new(indices_a, dims_a, storage_a);
    ///
    /// let indices_b = vec![k.clone(), l.clone()];
    /// let dims_b = vec![3, 4];
    /// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 12])));
    /// let tensor_b: TensorDynLen = TensorDynLen::new(indices_b, dims_b, storage_b);
    ///
    /// // Contract j (from A) with k (from B): result is C[i, l]
    /// let result = tensor_a.tensordot(&tensor_b, &[(j.clone(), k.clone())]).unwrap();
    /// assert_eq!(result.dims, vec![2, 4]);
    /// ```
    pub fn tensordot(
        &self,
        other: &Self,
        pairs: &[(DynIndex, DynIndex)],
    ) -> Result<Self> {
        use crate::index_ops::ContractionError;

        let spec =
            prepare_contraction_pairs(&self.indices, &self.dims, &other.indices, &other.dims, pairs)
                .map_err(|e| match e {
                    ContractionError::NoCommonIndices => {
                        anyhow::anyhow!("tensordot: No pairs specified for contraction")
                    }
                    ContractionError::BatchContractionNotImplemented => anyhow::anyhow!(
                        "tensordot: Common index found but not in contraction pairs. \
                         Batch contraction is not yet implemented."
                    ),
                    ContractionError::IndexNotFound { tensor } => {
                        anyhow::anyhow!("tensordot: Index not found in {} tensor", tensor)
                    }
                    ContractionError::DimensionMismatch {
                        pos_a,
                        pos_b,
                        dim_a,
                        dim_b,
                    } => anyhow::anyhow!(
                        "tensordot: Dimension mismatch: self[{}]={} != other[{}]={}",
                        pos_a,
                        dim_a,
                        pos_b,
                        dim_b
                    ),
                    ContractionError::DuplicateAxis { tensor, pos } => {
                        anyhow::anyhow!("tensordot: Duplicate axis {} in {} tensor", pos, tensor)
                    }
                })?;

        let result_storage = Arc::new(contract_storage(
            self.storage(),
            &self.dims,
            &spec.axes_a,
            other.storage(),
            &other.dims,
            &spec.axes_b,
            &spec.result_dims,
        ));

        Ok(Self::new(spec.result_indices, spec.result_dims, result_storage))
    }

    /// Compute the outer product (tensor product) of two tensors.
    ///
    /// Creates a new tensor whose indices are the concatenation of the indices
    /// from both input tensors. The result has shape `[...self.dims, ...other.dims]`.
    ///
    /// This is equivalent to numpy's `np.outer` or `np.tensordot(a, b, axes=0)`,
    /// or ITensor's `*` operator when there are no common indices.
    ///
    /// # Arguments
    /// * `other` - The other tensor to compute outer product with
    ///
    /// # Returns
    /// A new tensor with indices from both tensors.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use tensor4all_core::Storage;
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let tensor_a: TensorDynLen = TensorDynLen::new(
    ///     vec![i.clone()],
    ///     vec![2],
    ///     Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 2.0]))),
    /// );
    /// let tensor_b: TensorDynLen = TensorDynLen::new(
    ///     vec![j.clone()],
    ///     vec![3],
    ///     Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 2.0, 3.0]))),
    /// );
    ///
    /// // Outer product: C[i, j] = A[i] * B[j]
    /// let result = tensor_a.outer_product(&tensor_b).unwrap();
    /// assert_eq!(result.dims, vec![2, 3]);
    /// ```
    pub fn outer_product(&self, other: &Self) -> Result<Self> {
        use crate::storage::contract_storage;
        use anyhow::Context;

        // Check for common indices - outer product should have none
        let common_positions = common_ind_positions(&self.indices, &other.indices);
        if !common_positions.is_empty() {
            let common_ids: Vec<_> = common_positions
                .iter()
                .map(|(pos_a, _)| self.indices[*pos_a].id())
                .collect();
            return Err(anyhow::anyhow!(
                "outer_product: tensors have common indices {:?}. \
                 Use tensordot to contract common indices, or use sim() to replace \
                 indices with fresh IDs before computing outer product.",
                common_ids
            ))
            .context("outer_product: common indices found");
        }

        // Build result indices and dimensions
        let mut result_indices = self.indices.clone();
        result_indices.extend(other.indices.iter().cloned());

        let mut result_dims = self.dims.clone();
        result_dims.extend(other.dims.iter().cloned());

        // Perform outer product using contract_storage with empty axes
        let result_storage = Arc::new(contract_storage(
            self.storage(),
            &self.dims,
            &[], // No axes to contract from self
            other.storage(),
            &other.dims,
            &[], // No axes to contract from other
            &result_dims,
        ));

        Ok(Self::new(result_indices, result_dims, result_storage))
    }
}

// ============================================================================
// Random tensor generation
// ============================================================================

impl TensorDynLen {
    /// Create a random f64 tensor with values from standard normal distribution.
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `indices` - The indices for the tensor
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use rand::SeedableRng;
    /// use rand_chacha::ChaCha8Rng;
    ///
    /// let mut rng = ChaCha8Rng::seed_from_u64(42);
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let tensor: TensorDynLen = TensorDynLen::random_f64(&mut rng, vec![i, j]);
    /// assert_eq!(tensor.dims, vec![2, 3]);
    /// ```
    pub fn random_f64<R: rand::Rng>(rng: &mut R, indices: Vec<DynIndex>) -> Self {
        let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
        let total_size: usize = dims.iter().product();
        let storage = Arc::new(Storage::DenseF64(crate::storage::DenseStorageF64::random(
            rng, total_size,
        )));
        Self::new(indices, dims, storage)
    }

    /// Create a random Complex64 tensor with values from standard normal distribution.
    ///
    /// Both real and imaginary parts are drawn from standard normal distribution.
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `indices` - The indices for the tensor
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use rand::SeedableRng;
    /// use rand_chacha::ChaCha8Rng;
    ///
    /// let mut rng = ChaCha8Rng::seed_from_u64(42);
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let tensor: TensorDynLen = TensorDynLen::random_c64(&mut rng, vec![i, j]);
    /// assert_eq!(tensor.dims, vec![2, 3]);
    /// ```
    pub fn random_c64<R: rand::Rng>(rng: &mut R, indices: Vec<DynIndex>) -> Self {
        let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
        let total_size: usize = dims.iter().product();
        let storage = Arc::new(Storage::DenseC64(crate::storage::DenseStorageC64::random(
            rng, total_size,
        )));
        Self::new(indices, dims, storage)
    }
}

/// Implement multiplication operator for tensor contraction.
///
/// The `*` operator performs tensor contraction along common indices.
/// This is equivalent to calling the `contract` method.
///
/// # Example
/// ```
/// use tensor4all_core::TensorDynLen;
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::Storage;
/// use tensor4all_core::storage::DenseStorageF64;
/// use std::sync::Arc;
///
/// // Create two tensors: A[i, j] and B[j, k]
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let dims_a = vec![2, 3];
/// let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
/// let tensor_a: TensorDynLen = TensorDynLen::new(indices_a, dims_a, storage_a);
///
/// let indices_b = vec![j.clone(), k.clone()];
/// let dims_b = vec![3, 4];
/// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 12])));
/// let tensor_b: TensorDynLen = TensorDynLen::new(indices_b, dims_b, storage_b);
///
/// // Contract along j using * operator: result is C[i, k]
/// let result = &tensor_a * &tensor_b;
/// assert_eq!(result.dims, vec![2, 4]);
/// ```
impl Mul<&TensorDynLen> for &TensorDynLen {
    type Output = TensorDynLen;

    fn mul(self, other: &TensorDynLen) -> Self::Output {
        self.contract_einsum(other)
    }
}

/// Implement multiplication operator for tensor contraction (owned version).
///
/// This allows using `tensor_a * tensor_b` when both tensors are owned.
impl Mul<TensorDynLen> for TensorDynLen {
    type Output = TensorDynLen;

    fn mul(self, other: TensorDynLen) -> Self::Output {
        self.contract_einsum(&other)
    }
}

/// Implement multiplication operator for tensor contraction (mixed reference/owned).
impl Mul<TensorDynLen> for &TensorDynLen {
    type Output = TensorDynLen;

    fn mul(self, other: TensorDynLen) -> Self::Output {
        self.contract_einsum(&other)
    }
}

/// Implement multiplication operator for tensor contraction (mixed owned/reference).
impl Mul<&TensorDynLen> for TensorDynLen {
    type Output = TensorDynLen;

    fn mul(self, other: &TensorDynLen) -> Self::Output {
        self.contract_einsum(other)
    }
}

/// Check if a tensor is a DiagTensor (has Diag storage).
pub fn is_diag_tensor(tensor: &TensorDynLen) -> bool {
    tensor.storage().as_ref().is_diag()
}

impl TensorDynLen {
    /// Add two tensors element-wise.
    ///
    /// The tensors must have the same index set (matched by ID). If the indices
    /// are in a different order, the other tensor will be permuted to match `self`.
    ///
    /// # Arguments
    /// * `other` - The tensor to add
    ///
    /// # Returns
    /// A new tensor representing `self + other`, or an error if:
    /// - The tensors have different index sets
    /// - The dimensions don't match
    /// - Storage types are incompatible
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core::Storage;
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    ///
    /// let indices_a = vec![i.clone(), j.clone()];
    /// let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a)));
    /// let tensor_a: TensorDynLen = TensorDynLen::new(indices_a, vec![2, 3], storage_a);
    ///
    /// let indices_b = vec![i.clone(), j.clone()];
    /// let data_b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    /// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b)));
    /// let tensor_b: TensorDynLen = TensorDynLen::new(indices_b, vec![2, 3], storage_b);
    ///
    /// let sum = tensor_a.add(&tensor_b).unwrap();
    /// // sum = [[2, 3, 4], [5, 6, 7]]
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self> {
        // Validate that both tensors have the same number of indices
        if self.indices.len() != other.indices.len() {
            return Err(anyhow::anyhow!(
                "Index count mismatch: self has {} indices, other has {}",
                self.indices.len(),
                other.indices.len()
            ));
        }

        // Validate that both tensors have the same set of indices
        let self_set: HashSet<_> = self.indices.iter().collect();
        let other_set: HashSet<_> = other.indices.iter().collect();

        if self_set != other_set {
            return Err(anyhow::anyhow!(
                "Index set mismatch: tensors must have the same indices"
            ));
        }

        // Permute other to match self's index order (no-op if already aligned)
        let other_aligned = other.permute_indices(&self.indices);

        // Validate dimensions match after alignment
        if self.dims != other_aligned.dims {
            return Err(anyhow::anyhow!(
                "Dimension mismatch after alignment: self has dims {:?}, other has {:?}",
                self.dims,
                other_aligned.dims
            ));
        }

        // Get storages (materializes lazy tensors if needed)
        let self_storage = self.materialize_storage()?;
        let other_storage = other_aligned.materialize_storage()?;

        // Add storages using try_add (returns Result instead of panicking)
        let result_storage = self_storage
            .as_ref()
            .try_add(other_storage.as_ref())
            .map_err(|e| anyhow::anyhow!("Storage addition failed: {}", e))?;

        Ok(Self::new(self.indices.clone(), self.dims.clone(), Arc::new(result_storage)))
    }

    /// Compute a linear combination: `a * self + b * other`.
    pub fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self> {
        // Scale self by a
        let scaled_self = self.scale(a)?;
        // Scale other by b
        let scaled_other = other.scale(b)?;
        // Add the two
        scaled_self.add(&scaled_other)
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: AnyScalar) -> Result<Self> {
        let storage = self.materialize_storage()?;
        let scaled_storage = storage.scale(&scalar);
        Ok(Self::new(self.indices.clone(), self.dims.clone(), Arc::new(scaled_storage)))
    }

    /// Inner product (dot product) of two tensors.
    ///
    /// Computes `⟨self, other⟩ = Σ conj(self)_i * other_i`.
    pub fn inner_product(&self, other: &Self) -> Result<AnyScalar> {
        // Contract self.conj() with other over all indices
        let conj_self = self.conj();
        let result = super::contract::contract_multi(&[conj_self, other.clone()])?;
        // Result should be a scalar (no indices)
        Ok(result.sum())
    }
}

impl Clone for TensorDynLen {
    fn clone(&self) -> Self {
        Self {
            indices: self.indices.clone(),
            dims: self.dims.clone(),
            data: self.data.clone(),
        }
    }
}

// ============================================================================
// Index Replacement Methods
// ============================================================================

impl TensorDynLen {
    /// Replace an index in the tensor with a new index.
    ///
    /// This replaces the index matching `old_index` by ID with `new_index`.
    /// The storage data is not modified, only the index metadata is changed.
    ///
    /// # Arguments
    /// * `old_index` - The index to replace (matched by ID)
    /// * `new_index` - The new index to use
    ///
    /// # Returns
    /// A new tensor with the index replaced. If no index matches `old_index`,
    /// returns a clone of the original tensor.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core::Storage;
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let new_i = Index::new_dyn(2);  // Same dimension, different ID
    ///
    /// let indices = vec![i.clone(), j.clone()];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor: TensorDynLen = TensorDynLen::new(indices, vec![2, 3], storage);
    ///
    /// // Replace index i with new_i
    /// let replaced = tensor.replaceind(&i, &new_i);
    /// assert_eq!(replaced.indices[0].id, new_i.id);
    /// assert_eq!(replaced.indices[1].id, j.id);
    /// ```
    pub fn replaceind(&self, old_index: &DynIndex, new_index: &DynIndex) -> Self {
        let new_indices: Vec<_> = self
            .indices
            .iter()
            .map(|idx| {
                if *idx == *old_index {
                    new_index.clone()
                } else {
                    idx.clone()
                }
            })
            .collect();

        // Create new TensorData with updated index IDs
        let new_index_ids: Vec<DynId> = new_indices.iter().map(|idx| idx.id().clone()).collect();
        let new_data = TensorData::new(self.storage().clone(), new_index_ids, self.dims.clone());

        Self::from_data(new_indices, self.dims.clone(), new_data)
    }

    /// Replace multiple indices in the tensor.
    ///
    /// This replaces each index in `old_indices` (matched by ID) with the corresponding
    /// index in `new_indices`. The storage data is not modified.
    ///
    /// # Arguments
    /// * `old_indices` - The indices to replace (matched by ID)
    /// * `new_indices` - The new indices to use
    ///
    /// # Panics
    /// Panics if `old_indices` and `new_indices` have different lengths.
    ///
    /// # Returns
    /// A new tensor with the indices replaced. Indices not found in `old_indices`
    /// are kept unchanged.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core::Storage;
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let new_i = Index::new_dyn(2);
    /// let new_j = Index::new_dyn(3);
    ///
    /// let indices = vec![i.clone(), j.clone()];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor: TensorDynLen = TensorDynLen::new(indices, vec![2, 3], storage);
    ///
    /// // Replace both indices
    /// let replaced = tensor.replaceinds(&[i.clone(), j.clone()], &[new_i.clone(), new_j.clone()]);
    /// assert_eq!(replaced.indices[0].id, new_i.id);
    /// assert_eq!(replaced.indices[1].id, new_j.id);
    /// ```
    pub fn replaceinds(
        &self,
        old_indices: &[DynIndex],
        new_indices: &[DynIndex],
    ) -> Self {
        assert_eq!(
            old_indices.len(),
            new_indices.len(),
            "old_indices and new_indices must have the same length"
        );

        // Build a map from old indices to new indices
        let replacement_map: std::collections::HashMap<_, _> = old_indices
            .iter()
            .zip(new_indices.iter())
            .map(|(old, new)| (old, new))
            .collect();

        let new_indices_vec: Vec<_> = self
            .indices
            .iter()
            .map(|idx| {
                if let Some(new_idx) = replacement_map.get(idx) {
                    (*new_idx).clone()
                } else {
                    idx.clone()
                }
            })
            .collect();

        // Create new TensorData with updated index IDs
        let new_index_ids: Vec<DynId> = new_indices_vec.iter().map(|idx| idx.id().clone()).collect();
        let new_data = TensorData::new(self.storage().clone(), new_index_ids, self.dims.clone());

        Self::from_data(new_indices_vec, self.dims.clone(), new_data)
    }
}

// ============================================================================
// Complex Conjugation
// ============================================================================

impl TensorDynLen {
    /// Complex conjugate of all tensor elements.
    ///
    /// For real (f64) tensors, returns a copy (conjugate of real is identity).
    /// For complex (Complex64) tensors, conjugates each element.
    ///
    /// The indices and dimensions remain unchanged.
    ///
    /// This is inspired by the `conj` operation in ITensorMPS.jl.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::Storage;
    /// use tensor4all_core::storage::DenseStorageC64;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use num_complex::Complex64;
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)];
    /// let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data)));
    /// let tensor: TensorDynLen = TensorDynLen::new(vec![i], vec![2], storage);
    ///
    /// let conj_tensor = tensor.conj();
    /// // Elements are now conjugated: 1-2i, 3+4i
    /// ```
    pub fn conj(&self) -> Self {
        // Conjugate tensor: conjugate storage data and map indices via IndexLike::conj()
        // For default undirected indices, conj() is a no-op, so this is future-proof
        // for QSpace-compatible directed indices where conj() flips Ket <-> Bra
        let new_indices: Vec<DynIndex> = self.indices.iter().map(|idx| idx.conj()).collect();
        let new_storage = Arc::new(self.storage().conj());
        Self::new(new_indices, self.dims.clone(), new_storage)
    }
}

// ============================================================================
// Norm Computation
// ============================================================================

impl TensorDynLen {
    /// Compute the squared Frobenius norm of the tensor: ||T||² = Σ|T_ijk...|²
    ///
    /// For real tensors: sum of squares of all elements.
    /// For complex tensors: sum of |z|² = z * conj(z) for all elements.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::Storage;
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];  // 1² + 2² + ... + 6² = 91
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));
    /// let tensor: TensorDynLen = TensorDynLen::new(vec![i, j], vec![2, 3], storage);
    ///
    /// assert!((tensor.norm_squared() - 91.0).abs() < 1e-10);
    /// ```
    pub fn norm_squared(&self) -> f64 {
        // Special case: scalar tensor (no indices)
        if self.indices.is_empty() {
            // For a scalar, ||T||² = |value|²
            let value = self.sum();
            let abs_val = value.abs();
            return abs_val * abs_val;
        }

        // Contract tensor with its conjugate over all indices → scalar
        // ||T||² = Σ T_ijk... * conj(T_ijk...) = Σ |T_ijk...|²
        let conj = self.conj();
        let scalar = self.contract_einsum(&conj);
        scalar.sum().real() // Result is always real for ||T||²
    }

    /// Compute the Frobenius norm of the tensor: ||T|| = sqrt(Σ|T_ijk...|²)
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::Storage;
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let data = vec![3.0, 4.0];  // sqrt(9 + 16) = 5
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));
    /// let tensor: TensorDynLen = TensorDynLen::new(vec![i], vec![2], storage);
    ///
    /// assert!((tensor.norm() - 5.0).abs() < 1e-10);
    /// ```
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Compute the relative distance between two tensors.
    ///
    /// Returns `||A - B|| / ||A||` (Frobenius norm).
    /// If `||A|| = 0`, returns `||B||` instead to avoid division by zero.
    ///
    /// This is the ITensor-style distance function useful for comparing tensors.
    ///
    /// # Arguments
    /// * `other` - The other tensor to compare with
    ///
    /// # Returns
    /// The relative distance as a f64 value.
    ///
    /// # Note
    /// The indices of both tensors must be permutable to each other.
    /// The result tensor (A - B) uses the index ordering from self.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::Storage;
    /// use tensor4all_core::storage::DenseStorageF64;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let data_a = vec![1.0, 0.0];
    /// let data_b = vec![1.0, 0.0];  // Same tensor
    /// let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a)));
    /// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b)));
    /// let tensor_a: TensorDynLen = TensorDynLen::new(vec![i.clone()], vec![2], storage_a);
    /// let tensor_b: TensorDynLen = TensorDynLen::new(vec![i.clone()], vec![2], storage_b);
    ///
    /// assert!(tensor_a.distance(&tensor_b) < 1e-10);  // Zero distance
    /// ```
    pub fn distance(&self, other: &Self) -> f64 {
        let norm_self = self.norm();

        // Compute A - B = A + (-1) * B
        let neg_other_storage = other.storage().as_ref() * (-1.0_f64);
        let neg_other = Self::new(
            other.indices.clone(),
            other.dims.clone(),
            std::sync::Arc::new(neg_other_storage),
        );
        let diff = self
            .add(&neg_other)
            .expect("distance: tensors must have same indices");
        let norm_diff = diff.norm();

        if norm_self > 0.0 {
            norm_diff / norm_self
        } else {
            norm_diff
        }
    }
}

impl std::fmt::Debug for TensorDynLen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorDynLen")
            .field("indices", &self.indices)
            .field("dims", &self.dims)
            .field("data", &self.data)
            .finish()
    }
}

/// Create a DiagTensor with dynamic rank from diagonal data.
///
/// # Arguments
/// * `indices` - The indices for the tensor (all must have the same dimension)
/// * `diag_data` - The diagonal elements (length must equal the dimension of indices)
///
/// # Panics
/// Panics if indices have different dimensions, or if diag_data length doesn't match.
pub fn diag_tensor_dyn_len(
    indices: Vec<DynIndex>,
    diag_data: Vec<f64>,
) -> TensorDynLen {
    let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
    let first_dim = dims[0];

    // Validate all indices have same dimension
    for (i, &dim) in dims.iter().enumerate() {
        assert_eq!(
            dim, first_dim,
            "DiagTensor requires all indices to have the same dimension, but dims[{}] = {} != dims[0] = {}",
            i, dim, first_dim
        );
    }

    assert_eq!(
        diag_data.len(),
        first_dim,
        "diag_data length ({}) must equal index dimension ({})",
        diag_data.len(),
        first_dim
    );

    let storage = Arc::new(Storage::new_diag_f64(diag_data));
    TensorDynLen::new(indices, dims, storage)
}

/// Create a DiagTensor with dynamic rank from complex diagonal data.
pub fn diag_tensor_dyn_len_c64(
    indices: Vec<DynIndex>,
    diag_data: Vec<Complex64>,
) -> TensorDynLen {
    let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
    let first_dim = dims[0];

    // Validate all indices have same dimension
    for (i, &dim) in dims.iter().enumerate() {
        assert_eq!(
            dim, first_dim,
            "DiagTensor requires all indices to have the same dimension, but dims[{}] = {} != dims[0] = {}",
            i, dim, first_dim
        );
    }

    assert_eq!(
        diag_data.len(),
        first_dim,
        "diag_data length ({}) must equal index dimension ({})",
        diag_data.len(),
        first_dim
    );

    let storage = Arc::new(Storage::new_diag_c64(diag_data));
    TensorDynLen::new(indices, dims, storage)
}

/// Unfold a tensor into a matrix by splitting indices into left and right groups.
///
/// This function validates the split, permutes the tensor so that left indices come first,
/// and returns a 2D matrix tensor (`DTensor<T, 2>`) along with metadata.
///
/// # Arguments
/// * `t` - Input tensor
/// * `left_inds` - Indices to place on the left (row) side of the matrix
///
/// # Returns
/// A tuple `(matrix_tensor, left_len, m, n, left_indices, right_indices)` where:
/// - `matrix_tensor` is a `DTensor<T, 2>` with shape `[m, n]` containing the unfolded data
/// - `left_len` is the number of left indices
/// - `m` is the product of left index dimensions
/// - `n` is the product of right index dimensions
/// - `left_indices` is the vector of left indices (cloned)
/// - `right_indices` is the vector of right indices (cloned)
///
/// # Errors
/// Returns an error if:
/// - The tensor rank is < 2
/// - `left_inds` is empty or contains all indices
/// - `left_inds` contains indices not in the tensor or duplicates
/// - Storage type is not supported (must be DenseF64 or DenseC64)
pub fn unfold_split<T: StorageScalar>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
) -> Result<(
    DTensor<T, 2>,
    usize,
    usize,
    usize,
    Vec<DynIndex>,
    Vec<DynIndex>,
)> {
    let rank = t.dims.len();

    // Validate rank
    anyhow::ensure!(rank >= 2, "Tensor must have rank >= 2, got rank {}", rank);

    let left_len = left_inds.len();

    // Validate split: must be a proper subset
    anyhow::ensure!(
        left_len > 0 && left_len < rank,
        "Left indices must be a non-empty proper subset of tensor indices (0 < left_len < rank), got left_len={}, rank={}",
        left_len,
        rank
    );

    // Validate that all left_inds are in the tensor and there are no duplicates
    let tensor_set: HashSet<_> = t.indices.iter().collect();
    let mut left_set = HashSet::new();

    for left_idx in left_inds {
        anyhow::ensure!(
            tensor_set.contains(left_idx),
            "Index in left_inds not found in tensor"
        );
        anyhow::ensure!(
            left_set.insert(left_idx),
            "Duplicate index in left_inds"
        );
    }

    // Build right_inds: all indices not in left_inds, in original order
    let mut right_inds = Vec::new();
    for idx in &t.indices {
        if !left_set.contains(idx) {
            right_inds.push(idx.clone());
        }
    }

    // Build new_indices: left_inds first, then right_inds
    let mut new_indices = Vec::with_capacity(rank);
    new_indices.extend_from_slice(left_inds);
    new_indices.extend_from_slice(&right_inds);

    // Permute tensor to have left indices first, then right indices
    let unfolded = t.permute_indices(&new_indices);

    // Compute matrix dimensions
    let m: usize = unfolded.dims[..left_len].iter().product();
    let n: usize = unfolded.dims[left_len..].iter().product();

    // Create DTensor directly from storage
    let matrix_tensor = storage_to_dtensor::<T>(unfolded.storage().as_ref(), [m, n])
        .map_err(|e| anyhow::anyhow!("Failed to create DTensor: {}", e))?;

    Ok((
        matrix_tensor,
        left_len,
        m,
        n,
        left_inds.to_vec(),
        right_inds,
    ))
}

// ============================================================================
// TensorLike implementation for TensorDynLen
// ============================================================================

use crate::tensor_like::{FactorizeError, FactorizeOptions, FactorizeResult, TensorLike};

impl TensorLike for TensorDynLen {
    type Index = DynIndex;

    fn external_indices(&self) -> Vec<DynIndex> {
        // For TensorDynLen, all indices are external.
        self.indices.clone()
    }

    fn num_external_indices(&self) -> usize {
        self.indices.len()
    }

    fn replaceind(&self, old_index: &DynIndex, new_index: &DynIndex) -> Result<Self> {
        // Delegate to the inherent method
        Ok(TensorDynLen::replaceind(self, old_index, new_index))
    }

    fn replaceinds(
        &self,
        old_indices: &[DynIndex],
        new_indices: &[DynIndex],
    ) -> Result<Self> {
        // Delegate to the inherent method
        Ok(TensorDynLen::replaceinds(self, old_indices, new_indices))
    }

    fn tensordot(
        &self,
        other: &Self,
        pairs: &[(DynIndex, DynIndex)],
    ) -> Result<Self> {
        // Delegate to the inherent method
        TensorDynLen::tensordot(self, other, pairs)
    }

    fn factorize(
        &self,
        left_inds: &[DynIndex],
        options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        crate::factorize::factorize(self, left_inds, options)
    }

    fn conj(&self) -> Self {
        // Delegate to the inherent method (complex conjugate for dense tensors)
        TensorDynLen::conj(self)
    }

    fn direct_sum(
        &self,
        other: &Self,
        pairs: &[(DynIndex, DynIndex)],
    ) -> Result<crate::tensor_like::DirectSumResult<Self>> {
        let (tensor, new_indices) = crate::direct_sum::direct_sum(self, other, pairs)?;
        Ok(crate::tensor_like::DirectSumResult { tensor, new_indices })
    }

    fn outer_product(&self, other: &Self) -> Result<Self> {
        // Delegate to the inherent method
        TensorDynLen::outer_product(self, other)
    }

    fn norm_squared(&self) -> f64 {
        // Delegate to the inherent method
        TensorDynLen::norm_squared(self)
    }

    fn permuteinds(&self, new_order: &[DynIndex]) -> Result<Self> {
        // Delegate to the inherent method
        Ok(TensorDynLen::permute_indices(self, new_order))
    }

    fn contract_einsum(tensors: &[Self]) -> Result<Self> {
        // Delegate to contract_multi which uses omeco for optimization
        super::contract::contract_multi(tensors)
    }

    fn axpby(&self, a: crate::AnyScalar, other: &Self, b: crate::AnyScalar) -> Result<Self> {
        // Delegate to the inherent method
        TensorDynLen::axpby(self, a, other, b)
    }

    fn scale(&self, scalar: crate::AnyScalar) -> Result<Self> {
        // Delegate to the inherent method
        TensorDynLen::scale(self, scalar)
    }

    fn inner_product(&self, other: &Self) -> Result<crate::AnyScalar> {
        // Delegate to the inherent method
        TensorDynLen::inner_product(self, other)
    }

    fn diagonal(input_index: &DynIndex, output_index: &DynIndex) -> Result<Self> {
        use crate::storage::DenseStorageF64;

        let dim = input_index.dim();
        if dim != output_index.dim() {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: input index has dim {}, output has dim {}",
                dim,
                output_index.dim(),
            ));
        }

        // Build identity matrix
        let mut data = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }

        let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));
        Ok(TensorDynLen::new(
            vec![input_index.clone(), output_index.clone()],
            vec![dim, dim],
            storage,
        ))
    }

    fn scalar_one() -> Result<Self> {
        use crate::storage::DenseStorageF64;
        let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0])));
        Ok(TensorDynLen::new(vec![], vec![], storage))
    }

    // delta() uses the default implementation via diagonal() and outer_product()
}
