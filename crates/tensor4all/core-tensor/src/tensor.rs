use std::sync::Arc;
use std::collections::HashSet;
use std::ops::Mul;
use num_complex::Complex64;
use tensor4all_core_common::index::{Index, NoSymmSpace, Symmetry};
use tensor4all_core_common::index_ops::{common_inds, check_unique_indices};
use crate::storage::{AnyScalar, Storage, StorageScalar, SumFromStorage, contract_storage, storage_to_dtensor};
use anyhow::Result;
use mdarray::DTensor;

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
/// use tensor4all_core_tensor::tensor::compute_permutation_from_indices;
/// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let original = vec![i.clone(), j.clone()];
/// let new_order = vec![j.clone(), i.clone()];
///
/// let perm = compute_permutation_from_indices(&original, &new_order);
/// assert_eq!(perm, vec![1, 0]);  // j is at position 1, i is at position 0
/// ```
pub fn compute_permutation_from_indices<Id, Symm>(
    original_indices: &[Index<Id, Symm>],
    new_indices: &[Index<Id, Symm>],
) -> Vec<usize>
where
    Id: std::hash::Hash + Eq,
{
    assert_eq!(
        new_indices.len(),
        original_indices.len(),
        "new_indices length must match original_indices length"
    );

    let mut perm = Vec::with_capacity(new_indices.len());
    let mut used = std::collections::HashSet::new();

    for new_idx in new_indices {
        // Find the position of this index in the original indices
        let pos = original_indices
            .iter()
            .position(|old_idx| old_idx.id == new_idx.id)
            .expect("new_indices must be a permutation of original_indices");
        
        if used.contains(&pos) {
            panic!("duplicate index in new_indices");
        }
        used.insert(pos);
        perm.push(pos);
    }

    perm
}

/// Trait for extracting type parameters from tensor types.
///
/// This allows extracting `Id` and `Symm` from tensor types like `TensorDynLen<Id, Symm>`
/// without exposing them directly in the type signature.
pub trait TensorType {
    /// Index ID type
    type Id: Clone + std::hash::Hash + Eq;
    /// Symmetry type
    type Symm: Clone + Symmetry;
}

/// Trait for accessing tensor fields.
///
/// This trait provides access to the internal fields of tensor types,
/// allowing generic code to work with tensors without knowing the exact type.
pub trait TensorAccess: TensorType {
    /// Get a reference to the indices.
    fn indices(&self) -> &[Index<Self::Id, Self::Symm>];
    
    /// Get a reference to the storage.
    fn storage(&self) -> &Storage;
}

/// Tensor with dynamic rank (number of indices) and dynamic scalar type.
pub struct TensorDynLen<Id, Symm = NoSymmSpace> {
    pub indices: Vec<Index<Id, Symm>>,
    pub dims: Vec<usize>,
    pub storage: Arc<Storage>,
}

impl<Id, Symm> TensorType for TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    type Id = Id;
    type Symm = Symm;
}

impl<Id, Symm> TensorAccess for TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    fn indices(&self) -> &[Index<Self::Id, Self::Symm>] {
        &self.indices
    }
    
    fn storage(&self) -> &Storage {
        &self.storage
    }
}

impl<Id, Symm> TensorDynLen<Id, Symm> {
    /// Create a new tensor with dynamic rank.
    ///
    /// Dimensions are automatically computed from the indices using `Index::size()`.
    ///
    /// # Panics
    /// Panics if the storage is Diag and not all indices have the same dimension.
    /// Panics if there are duplicate indices (indices with the same ID).
    pub fn new(indices: Vec<Index<Id, Symm>>, dims: Vec<usize>, storage: Arc<Storage>) -> Self
    where
        Id: std::hash::Hash + Eq,
    {
        assert_eq!(
            indices.len(),
            dims.len(),
            "indices and dims must have the same length"
        );
        
        // Check for duplicate indices
        check_unique_indices(&indices).expect("Tensor indices must all be unique (no duplicate IDs)");
        
        // Validate DiagTensor: all indices must have the same dimension
        if storage.as_ref().is_diag() {
            let first_dim = dims[0];
            for (i, &dim) in dims.iter().enumerate() {
                assert_eq!(
                    dim, first_dim,
                    "DiagTensor requires all indices to have the same dimension, but dims[{}] = {} != dims[0] = {}",
                    i, dim, first_dim
                );
            }
        }
        
        Self {
            indices,
            dims,
            storage,
        }
    }

    /// Create a new tensor with dynamic rank, automatically computing dimensions from indices.
    ///
    /// This is a convenience constructor that extracts dimensions from indices using `Index::size()`.
    ///
    /// # Panics
    /// Panics if the storage is Diag and not all indices have the same dimension.
    /// Panics if there are duplicate indices (indices with the same ID).
    pub fn from_indices(indices: Vec<Index<Id, Symm>>, storage: Arc<Storage>) -> Self
    where
        Symm: Symmetry,
        Id: std::hash::Hash + Eq,
    {
        let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();
        Self::new(indices, dims, storage)
    }

    /// Get a mutable reference to storage (COW: clones if shared).
    pub fn storage_mut(&mut self) -> &mut Storage {
        Arc::make_mut(&mut self.storage)
    }

    /// Sum all elements, returning `AnyScalar`.
    pub fn sum(&self) -> AnyScalar {
        AnyScalar::sum_from_storage(&self.storage)
    }

    /// Sum all elements as f64.
    pub fn sum_f64(&self) -> f64 {
        f64::sum_from_storage(&self.storage)
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
    /// use tensor4all_core_tensor::{TensorDynLen, Storage, AnyScalar};
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use std::sync::Arc;
    ///
    /// // Create a scalar tensor (0 dimensions, 1 element)
    /// let indices: Vec<Index<DynId>> = vec![];
    /// let dims: Vec<usize> = vec![];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![42.0])));
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
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
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core_tensor::Storage;
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// // Create a 2×3 tensor
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let indices = vec![i.clone(), j.clone()];
    /// let dims = vec![2, 3];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
    ///
    /// // Permute to 3×2: swap the two dimensions by providing new indices order
    /// let permuted = tensor.permute_indices(&[j, i]);
    /// assert_eq!(permuted.dims, vec![3, 2]);
    /// ```
    pub fn permute_indices(&self, new_indices: &[Index<Id, Symm>]) -> Self
    where
        Id: Clone + std::hash::Hash + Eq,
        Symm: Clone + Symmetry,
    {
        // Compute permutation by matching IDs
        let perm = compute_permutation_from_indices(&self.indices, new_indices);

        // Compute new dims from new indices
        let new_dims: Vec<usize> = new_indices
            .iter()
            .map(|idx| idx.size())
            .collect();

        // Permute storage data using the computed permutation
        let new_storage = Arc::new(self.storage.permute_storage(&self.dims, &perm));

        Self {
            indices: new_indices.to_vec(),
            dims: new_dims,
            storage: new_storage,
        }
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
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core_tensor::Storage;
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// // Create a 2×3 tensor
    /// let indices = vec![
    ///     Index::new_dyn(2),
    ///     Index::new_dyn(3),
    /// ];
    /// let dims = vec![2, 3];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
    ///
    /// // Permute to 3×2: swap the two dimensions
    /// let permuted = tensor.permute(&[1, 0]);
    /// assert_eq!(permuted.dims, vec![3, 2]);
    /// ```
    pub fn permute(&self, perm: &[usize]) -> Self
    where
        Id: Clone,
        Symm: Clone,
    {
        assert_eq!(
            perm.len(),
            self.dims.len(),
            "permutation length must match tensor rank"
        );

        // Permute indices and dims
        let new_indices: Vec<Index<Id, Symm>> = perm
            .iter()
            .map(|&i| self.indices[i].clone())
            .collect();
        let new_dims: Vec<usize> = perm
            .iter()
            .map(|&i| self.dims[i])
            .collect();

        // Permute storage data
        let new_storage = Arc::new(self.storage.permute_storage(&self.dims, perm));

        Self {
            indices: new_indices,
            dims: new_dims,
            storage: new_storage,
        }
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
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core_tensor::Storage;
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
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
    /// let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(indices_a, dims_a, storage_a);
    ///
    /// let indices_b = vec![j.clone(), k.clone()];
    /// let dims_b = vec![3, 4];
    /// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 12])));
    /// let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(indices_b, dims_b, storage_b);
    ///
    /// // Contract along j: result is C[i, k]
    /// let result = tensor_a.contract_einsum(&tensor_b);
    /// assert_eq!(result.dims, vec![2, 4]);
    /// ```
    pub fn contract_einsum(&self, other: &Self) -> Self
    where
        Id: Clone + std::hash::Hash + Eq,
        Symm: Clone + Symmetry,
    {
        // Find common indices
        let common = common_inds(&self.indices, &other.indices);
        if common.is_empty() {
            panic!("No common indices found for contraction");
        }

        // Find positions of common indices in both tensors
        let mut axes_a = Vec::new();
        let mut axes_b = Vec::new();

        for common_idx in &common {
            // Find position in self
            let pos_a = self.indices
                .iter()
                .position(|idx| idx.id == common_idx.id)
                .expect("common index must be in self");
            axes_a.push(pos_a);

            // Find position in other
            let pos_b = other.indices
                .iter()
                .position(|idx| idx.id == common_idx.id)
                .expect("common index must be in other");
            axes_b.push(pos_b);

            // Verify dimensions match
            assert_eq!(
                self.dims[pos_a],
                other.dims[pos_b],
                "Common index dimension mismatch: {} != {}",
                self.dims[pos_a],
                other.dims[pos_b]
            );
        }

        // Get non-contracted indices
        let mut result_indices = Vec::new();
        let mut result_dims = Vec::new();

        // Add non-contracted indices from self
        for (i, idx) in self.indices.iter().enumerate() {
            if !axes_a.contains(&i) {
                result_indices.push(idx.clone());
                result_dims.push(self.dims[i]);
            }
        }

        // Add non-contracted indices from other
        for (i, idx) in other.indices.iter().enumerate() {
            if !axes_b.contains(&i) {
                result_indices.push(idx.clone());
                result_dims.push(other.dims[i]);
            }
        }

        // Perform contraction
        let result_storage = Arc::new(contract_storage(
            &self.storage,
            &self.dims,
            &axes_a,
            &other.storage,
            &other.dims,
            &axes_b,
            &result_dims,
        ));

        Self {
            indices: result_indices,
            dims: result_dims,
            storage: result_storage,
        }
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
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core_tensor::Storage;
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
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
    /// let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(indices_a, dims_a, storage_a);
    ///
    /// let indices_b = vec![k.clone(), l.clone()];
    /// let dims_b = vec![3, 4];
    /// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 12])));
    /// let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(indices_b, dims_b, storage_b);
    ///
    /// // Contract j (from A) with k (from B): result is C[i, l]
    /// let result = tensor_a.tensordot(&tensor_b, &[(j.clone(), k.clone())]).unwrap();
    /// assert_eq!(result.dims, vec![2, 4]);
    /// ```
    pub fn tensordot(
        &self,
        other: &Self,
        pairs: &[(Index<Id, Symm>, Index<Id, Symm>)],
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
        Symm: Clone + Symmetry,
    {
        use anyhow::Context;

        if pairs.is_empty() {
            return Err(anyhow::anyhow!("No pairs specified for contraction"))
                .context("tensordot: at least one pair must be specified");
        }

        // Collect IDs of indices that will be contracted
        let contracted_ids_self: HashSet<_> = pairs.iter().map(|(idx, _)| &idx.id).collect();
        let contracted_ids_other: HashSet<_> = pairs.iter().map(|(_, idx)| &idx.id).collect();

        // Find common indices (same ID in both tensors)
        let common = common_inds(&self.indices, &other.indices);

        // Check if any common index is NOT in the contraction pairs
        for common_idx in &common {
            let in_contracted_self = contracted_ids_self.contains(&common_idx.id);
            let in_contracted_other = contracted_ids_other.contains(&common_idx.id);

            if !in_contracted_self || !in_contracted_other {
                return Err(anyhow::anyhow!(
                    "Common index with id {:?} found but not in contraction pairs. \
                     Batch contraction (treating common indices as batch dimensions) \
                     is not yet implemented.",
                    common_idx.id
                ))
                .context("tensordot: batch contraction not yet implemented");
            }
        }

        // Find positions of indices in both tensors and validate
        let mut axes_a = Vec::new();
        let mut axes_b = Vec::new();

        for (idx_a, idx_b) in pairs {
            // Find position in self
            let pos_a = self.indices
                .iter()
                .position(|idx| idx.id == idx_a.id)
                .ok_or_else(|| anyhow::anyhow!("Index with id matching specified index not found in self tensor"))
                .context("tensordot: index from self not found")?;

            // Find position in other
            let pos_b = other.indices
                .iter()
                .position(|idx| idx.id == idx_b.id)
                .ok_or_else(|| anyhow::anyhow!("Index with id matching specified index not found in other tensor"))
                .context("tensordot: index from other not found")?;

            // Verify dimensions match
            if self.dims[pos_a] != other.dims[pos_b] {
                return Err(anyhow::anyhow!(
                    "Dimension mismatch for pair: self[{}] = {} != other[{}] = {}",
                    pos_a,
                    self.dims[pos_a],
                    pos_b,
                    other.dims[pos_b]
                ))
                .context("tensordot: dimensions must match for each pair");
            }

            // Check for duplicate axes in self
            if axes_a.contains(&pos_a) {
                return Err(anyhow::anyhow!(
                    "Duplicate axis {} in self tensor",
                    pos_a
                ))
                .context("tensordot: each axis can only be contracted once");
            }

            // Check for duplicate axes in other
            if axes_b.contains(&pos_b) {
                return Err(anyhow::anyhow!(
                    "Duplicate axis {} in other tensor",
                    pos_b
                ))
                .context("tensordot: each axis can only be contracted once");
            }

            axes_a.push(pos_a);
            axes_b.push(pos_b);
        }

        // Get non-contracted indices
        let mut result_indices = Vec::new();
        let mut result_dims = Vec::new();

        // Add non-contracted indices from self
        for (i, idx) in self.indices.iter().enumerate() {
            if !axes_a.contains(&i) {
                result_indices.push(idx.clone());
                result_dims.push(self.dims[i]);
            }
        }

        // Add non-contracted indices from other
        for (i, idx) in other.indices.iter().enumerate() {
            if !axes_b.contains(&i) {
                result_indices.push(idx.clone());
                result_dims.push(other.dims[i]);
            }
        }

        // Perform contraction
        let result_storage = Arc::new(contract_storage(
            &self.storage,
            &self.dims,
            &axes_a,
            &other.storage,
            &other.dims,
            &axes_b,
            &result_dims,
        ));

        Ok(Self {
            indices: result_indices,
            dims: result_dims,
            storage: result_storage,
        })
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
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
    /// use tensor4all_core_tensor::Storage;
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(
    ///     vec![i.clone()],
    ///     vec![2],
    ///     Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 2.0]))),
    /// );
    /// let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(
    ///     vec![j.clone()],
    ///     vec![3],
    ///     Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0, 2.0, 3.0]))),
    /// );
    ///
    /// // Outer product: C[i, j] = A[i] * B[j]
    /// let result = tensor_a.outer_product(&tensor_b).unwrap();
    /// assert_eq!(result.dims, vec![2, 3]);
    /// ```
    pub fn outer_product(&self, other: &Self) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
        Symm: Clone + Symmetry,
    {
        use anyhow::Context;
        use crate::storage::contract_storage;

        // Check for common indices - outer product should have none
        let common = common_inds(&self.indices, &other.indices);
        if !common.is_empty() {
            return Err(anyhow::anyhow!(
                "outer_product: tensors have common indices with ids {:?}. \
                 Use tensordot to contract common indices, or use sim() to replace \
                 indices with fresh IDs before computing outer product.",
                common.iter().map(|idx| &idx.id).collect::<Vec<_>>()
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
            &self.storage,
            &self.dims,
            &[],  // No axes to contract from self
            &other.storage,
            &other.dims,
            &[],  // No axes to contract from other
            &result_dims,
        ));

        Ok(Self {
            indices: result_indices,
            dims: result_dims,
            storage: result_storage,
        })
    }
}

// ============================================================================
// Random tensor generation
// ============================================================================

impl<Id, Symm> TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    /// Create a random f64 tensor with values from standard normal distribution.
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `indices` - The indices for the tensor
    ///
    /// # Example
    /// ```
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use rand::SeedableRng;
    /// use rand_chacha::ChaCha8Rng;
    ///
    /// let mut rng = ChaCha8Rng::seed_from_u64(42);
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::random_f64(&mut rng, vec![i, j]);
    /// assert_eq!(tensor.dims, vec![2, 3]);
    /// ```
    pub fn random_f64<R: rand::Rng>(rng: &mut R, indices: Vec<Index<Id, Symm>>) -> Self {
        let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();
        let total_size: usize = dims.iter().product();
        let storage = Arc::new(Storage::DenseF64(
            crate::storage::DenseStorageF64::random(rng, total_size)
        ));
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
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use rand::SeedableRng;
    /// use rand_chacha::ChaCha8Rng;
    ///
    /// let mut rng = ChaCha8Rng::seed_from_u64(42);
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::random_c64(&mut rng, vec![i, j]);
    /// assert_eq!(tensor.dims, vec![2, 3]);
    /// ```
    pub fn random_c64<R: rand::Rng>(rng: &mut R, indices: Vec<Index<Id, Symm>>) -> Self {
        let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();
        let total_size: usize = dims.iter().product();
        let storage = Arc::new(Storage::DenseC64(
            crate::storage::DenseStorageC64::random(rng, total_size)
        ));
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
/// use tensor4all_core_tensor::TensorDynLen;
/// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core_tensor::Storage;
/// use tensor4all_core_tensor::storage::DenseStorageF64;
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
/// let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(indices_a, dims_a, storage_a);
///
/// let indices_b = vec![j.clone(), k.clone()];
/// let dims_b = vec![3, 4];
/// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 12])));
/// let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(indices_b, dims_b, storage_b);
///
/// // Contract along j using * operator: result is C[i, k]
/// let result = &tensor_a * &tensor_b;
/// assert_eq!(result.dims, vec![2, 4]);
/// ```
impl<Id, Symm> Mul<&TensorDynLen<Id, Symm>> for &TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    type Output = TensorDynLen<Id, Symm>;

    fn mul(self, other: &TensorDynLen<Id, Symm>) -> Self::Output {
        self.contract_einsum(other)
    }
}

/// Implement multiplication operator for tensor contraction (owned version).
///
/// This allows using `tensor_a * tensor_b` when both tensors are owned.
impl<Id, Symm> Mul<TensorDynLen<Id, Symm>> for TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    type Output = TensorDynLen<Id, Symm>;

    fn mul(self, other: TensorDynLen<Id, Symm>) -> Self::Output {
        self.contract_einsum(&other)
    }
}

/// Implement multiplication operator for tensor contraction (mixed reference/owned).
impl<Id, Symm> Mul<TensorDynLen<Id, Symm>> for &TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    type Output = TensorDynLen<Id, Symm>;

    fn mul(self, other: TensorDynLen<Id, Symm>) -> Self::Output {
        self.contract_einsum(&other)
    }
}

/// Implement multiplication operator for tensor contraction (mixed owned/reference).
impl<Id, Symm> Mul<&TensorDynLen<Id, Symm>> for TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    type Output = TensorDynLen<Id, Symm>;

    fn mul(self, other: &TensorDynLen<Id, Symm>) -> Self::Output {
        self.contract_einsum(other)
    }
}


/// Check if a tensor is a DiagTensor (has Diag storage).
pub fn is_diag_tensor<Id, Symm>(tensor: &TensorDynLen<Id, Symm>) -> bool {
    tensor.storage.as_ref().is_diag()
}

impl<Id, Symm> TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
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
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core_tensor::Storage;
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    ///
    /// let indices_a = vec![i.clone(), j.clone()];
    /// let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a)));
    /// let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(indices_a, vec![2, 3], storage_a);
    ///
    /// let indices_b = vec![i.clone(), j.clone()];
    /// let data_b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    /// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b)));
    /// let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(indices_b, vec![2, 3], storage_b);
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

        // Validate that both tensors have the same set of index IDs
        let self_ids: HashSet<_> = self.indices.iter().map(|idx| &idx.id).collect();
        let other_ids: HashSet<_> = other.indices.iter().map(|idx| &idx.id).collect();

        if self_ids != other_ids {
            return Err(anyhow::anyhow!(
                "Index set mismatch: tensors must have the same indices (by ID)"
            ));
        }

        // Check if we need to permute other to match self's index order
        let needs_permute = self.indices.iter()
            .zip(other.indices.iter())
            .any(|(a, b)| a.id != b.id);

        let other_aligned = if needs_permute {
            // Permute other to match self's index order
            other.permute_indices(&self.indices)
        } else {
            // No permutation needed; we'll use a reference via clone
            // (cheap due to Arc)
            other.clone()
        };

        // Validate dimensions match after alignment
        if self.dims != other_aligned.dims {
            return Err(anyhow::anyhow!(
                "Dimension mismatch after alignment: self has dims {:?}, other has {:?}",
                self.dims,
                other_aligned.dims
            ));
        }

        // Add storages using try_add (returns Result instead of panicking)
        let result_storage = self.storage.as_ref()
            .try_add(other_aligned.storage.as_ref())
            .map_err(|e| anyhow::anyhow!("Storage addition failed: {}", e))?;

        Ok(Self {
            indices: self.indices.clone(),
            dims: self.dims.clone(),
            storage: Arc::new(result_storage),
        })
    }
}

impl<Id, Symm> Clone for TensorDynLen<Id, Symm>
where
    Id: Clone,
    Symm: Clone,
{
    fn clone(&self) -> Self {
        Self {
            indices: self.indices.clone(),
            dims: self.dims.clone(),
            storage: Arc::clone(&self.storage),
        }
    }
}

// ============================================================================
// Index Replacement Methods
// ============================================================================

impl<Id, Symm> TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone,
{
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
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core_tensor::Storage;
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let new_i = Index::new_dyn(2);  // Same dimension, different ID
    ///
    /// let indices = vec![i.clone(), j.clone()];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, vec![2, 3], storage);
    ///
    /// // Replace index i with new_i
    /// let replaced = tensor.replaceind(&i, &new_i);
    /// assert_eq!(replaced.indices[0].id, new_i.id);
    /// assert_eq!(replaced.indices[1].id, j.id);
    /// ```
    pub fn replaceind(&self, old_index: &Index<Id, Symm>, new_index: &Index<Id, Symm>) -> Self {
        let new_indices: Vec<_> = self
            .indices
            .iter()
            .map(|idx| {
                if idx.id == old_index.id {
                    new_index.clone()
                } else {
                    idx.clone()
                }
            })
            .collect();

        Self {
            indices: new_indices,
            dims: self.dims.clone(),
            storage: self.storage.clone(),
        }
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
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_core_tensor::Storage;
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let new_i = Index::new_dyn(2);
    /// let new_j = Index::new_dyn(3);
    ///
    /// let indices = vec![i.clone(), j.clone()];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, vec![2, 3], storage);
    ///
    /// // Replace both indices
    /// let replaced = tensor.replaceinds(&[i.clone(), j.clone()], &[new_i.clone(), new_j.clone()]);
    /// assert_eq!(replaced.indices[0].id, new_i.id);
    /// assert_eq!(replaced.indices[1].id, new_j.id);
    /// ```
    pub fn replaceinds(&self, old_indices: &[Index<Id, Symm>], new_indices: &[Index<Id, Symm>]) -> Self {
        assert_eq!(
            old_indices.len(),
            new_indices.len(),
            "old_indices and new_indices must have the same length"
        );

        // Build a map from old IDs to new indices
        let replacement_map: std::collections::HashMap<_, _> = old_indices
            .iter()
            .zip(new_indices.iter())
            .map(|(old, new)| (&old.id, new))
            .collect();

        let new_indices_vec: Vec<_> = self
            .indices
            .iter()
            .map(|idx| {
                if let Some(new_idx) = replacement_map.get(&idx.id) {
                    (*new_idx).clone()
                } else {
                    idx.clone()
                }
            })
            .collect();

        Self {
            indices: new_indices_vec,
            dims: self.dims.clone(),
            storage: self.storage.clone(),
        }
    }
}

// ============================================================================
// Complex Conjugation
// ============================================================================

impl<Id, Symm> TensorDynLen<Id, Symm>
where
    Id: Clone,
    Symm: Clone,
{
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
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_tensor::Storage;
    /// use tensor4all_core_tensor::storage::DenseStorageC64;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use num_complex::Complex64;
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)];
    /// let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data)));
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::new(vec![i], vec![2], storage);
    ///
    /// let conj_tensor = tensor.conj();
    /// // Elements are now conjugated: 1-2i, 3+4i
    /// ```
    pub fn conj(&self) -> Self {
        Self {
            indices: self.indices.clone(),
            dims: self.dims.clone(),
            storage: Arc::new(self.storage.conj()),
        }
    }
}

// ============================================================================
// Norm Computation
// ============================================================================

impl<Id, Symm> TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    /// Compute the squared Frobenius norm of the tensor: ||T||² = Σ|T_ijk...|²
    ///
    /// For real tensors: sum of squares of all elements.
    /// For complex tensors: sum of |z|² = z * conj(z) for all elements.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_tensor::Storage;
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];  // 1² + 2² + ... + 6² = 91
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::new(vec![i, j], vec![2, 3], storage);
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
        scalar.sum().real()  // Result is always real for ||T||²
    }

    /// Compute the Frobenius norm of the tensor: ||T|| = sqrt(Σ|T_ijk...|²)
    ///
    /// # Example
    /// ```
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_tensor::Storage;
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let data = vec![3.0, 4.0];  // sqrt(9 + 16) = 5
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::new(vec![i], vec![2], storage);
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
    /// use tensor4all_core_tensor::TensorDynLen;
    /// use tensor4all_core_tensor::Storage;
    /// use tensor4all_core_tensor::storage::DenseStorageF64;
    /// use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
    /// use std::sync::Arc;
    ///
    /// let i = Index::new_dyn(2);
    /// let data_a = vec![1.0, 0.0];
    /// let data_b = vec![1.0, 0.0];  // Same tensor
    /// let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_a)));
    /// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data_b)));
    /// let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone()], vec![2], storage_a);
    /// let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone()], vec![2], storage_b);
    ///
    /// assert!(tensor_a.distance(&tensor_b) < 1e-10);  // Zero distance
    /// ```
    pub fn distance(&self, other: &Self) -> f64
    where
        Id: std::fmt::Debug,
    {
        let norm_self = self.norm();

        // Compute A - B = A + (-1) * B
        let neg_other_storage = other.storage.as_ref() * (-1.0_f64);
        let neg_other = Self {
            indices: other.indices.clone(),
            dims: other.dims.clone(),
            storage: std::sync::Arc::new(neg_other_storage),
        };
        let diff = self.add(&neg_other).expect("distance: tensors must have same indices");
        let norm_diff = diff.norm();

        if norm_self > 0.0 {
            norm_diff / norm_self
        } else {
            norm_diff
        }
    }
}

impl<Id, Symm> std::fmt::Debug for TensorDynLen<Id, Symm>
where
    Id: std::fmt::Debug,
    Symm: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorDynLen")
            .field("indices", &self.indices)
            .field("dims", &self.dims)
            .field("storage", &self.storage)
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
pub fn diag_tensor_dyn_len<Id, Symm>(
    indices: Vec<Index<Id, Symm>>,
    diag_data: Vec<f64>,
) -> TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();
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
pub fn diag_tensor_dyn_len_c64<Id, Symm>(
    indices: Vec<Index<Id, Symm>>,
    diag_data: Vec<Complex64>,
) -> TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();
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
pub fn unfold_split<Id, T, Symm>(
    t: &TensorDynLen<Id, Symm>,
    left_inds: &[Index<Id, Symm>],
) -> Result<(
    DTensor<T, 2>,
    usize,
    usize,
    usize,
    Vec<Index<Id, Symm>>,
    Vec<Index<Id, Symm>>,
)>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    T: StorageScalar,
{
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
    let tensor_id_set: HashSet<_> = t.indices.iter().map(|idx| &idx.id).collect();
    let mut left_id_set = HashSet::new();
    
    for left_idx in left_inds {
        anyhow::ensure!(
            tensor_id_set.contains(&left_idx.id),
            "Index in left_inds not found in tensor"
        );
        anyhow::ensure!(
            left_id_set.insert(&left_idx.id),
            "Duplicate index in left_inds"
        );
    }

    // Build right_inds: all indices not in left_inds, in original order
    let mut right_inds = Vec::new();
    for idx in &t.indices {
        if !left_id_set.contains(&idx.id) {
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
    let matrix_tensor = storage_to_dtensor::<T>(unfolded.storage.as_ref(), [m, n])
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

