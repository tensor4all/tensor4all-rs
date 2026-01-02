use std::sync::Arc;
use tensor4all_index::index::{Index, NoSymmSpace, common_inds, Symmetry};
use crate::storage::{AnyScalar, Storage, SumFromStorage, permute_storage, contract_storage};

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
/// use tensor4all_tensor::tensor::compute_permutation_from_indices;
/// use tensor4all_index::index::{DefaultIndex as Index, DynId};
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

/// Tensor with dynamic rank (number of indices).
pub struct TensorDynLen<Id, T, Symm = NoSymmSpace> {
    pub indices: Vec<Index<Id, Symm>>,
    pub dims: Vec<usize>,
    pub storage: Arc<Storage>,
    _phantom: std::marker::PhantomData<T>,
}

impl<Id, T, Symm> TensorDynLen<Id, T, Symm> {
    /// Create a new tensor with dynamic rank.
    ///
    /// # Panics
    /// Panics if `indices.len() != dims.len()`.
    pub fn new(indices: Vec<Index<Id, Symm>>, dims: Vec<usize>, storage: Arc<Storage>) -> Self {
        assert_eq!(
            indices.len(),
            dims.len(),
            "indices and dims must have the same length"
        );
        Self {
            indices,
            dims,
            storage,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get a mutable reference to storage (COW: clones if shared).
    pub fn storage_mut(&mut self) -> &mut Storage {
        Arc::make_mut(&mut self.storage)
    }

    /// Sum all elements, returning `T`.
    ///
    /// For dynamic element tensors, use `T = AnyScalar`.
    pub fn sum(&self) -> T
    where
        T: SumFromStorage,
    {
        T::sum_from_storage(&self.storage)
    }

    /// Sum all elements as f64.
    pub fn sum_f64(&self) -> f64 {
        f64::sum_from_storage(&self.storage)
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
    /// use tensor4all_tensor::TensorDynLen;
    /// use tensor4all_index::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_tensor::Storage;
    /// use std::sync::Arc;
    ///
    /// // Create a 2×3 tensor
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let indices = vec![i.clone(), j.clone()];
    /// let dims = vec![2, 3];
    /// let storage = Arc::new(Storage::new_dense_f64(6));
    /// let tensor: TensorDynLen<DynId, f64> = TensorDynLen::new(indices, dims, storage);
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
        let new_storage = Arc::new(permute_storage(&self.storage, &self.dims, &perm));

        Self {
            indices: new_indices.to_vec(),
            dims: new_dims,
            storage: new_storage,
            _phantom: std::marker::PhantomData,
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
    /// use tensor4all_tensor::TensorDynLen;
    /// use tensor4all_index::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_tensor::Storage;
    /// use std::sync::Arc;
    ///
    /// // Create a 2×3 tensor
    /// let indices = vec![
    ///     Index::new_dyn(2),
    ///     Index::new_dyn(3),
    /// ];
    /// let dims = vec![2, 3];
    /// let storage = Arc::new(Storage::new_dense_f64(6));
    /// let tensor: TensorDynLen<DynId, f64> = TensorDynLen::new(indices, dims, storage);
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
        let new_storage = Arc::new(permute_storage(&self.storage, &self.dims, perm));

        Self {
            indices: new_indices,
            dims: new_dims,
            storage: new_storage,
            _phantom: std::marker::PhantomData,
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
    /// use tensor4all_tensor::TensorDynLen;
    /// use tensor4all_index::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_tensor::Storage;
    /// use std::sync::Arc;
    ///
    /// // Create two tensors: A[i, j] and B[j, k]
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let k = Index::new_dyn(4);
    ///
    /// let indices_a = vec![i.clone(), j.clone()];
    /// let dims_a = vec![2, 3];
    /// let storage_a = Arc::new(Storage::new_dense_f64(6));
    /// let tensor_a: TensorDynLen<DynId, f64> = TensorDynLen::new(indices_a, dims_a, storage_a);
    ///
    /// let indices_b = vec![j.clone(), k.clone()];
    /// let dims_b = vec![3, 4];
    /// let storage_b = Arc::new(Storage::new_dense_f64(12));
    /// let tensor_b: TensorDynLen<DynId, f64> = TensorDynLen::new(indices_b, dims_b, storage_b);
    ///
    /// // Contract along j: result is C[i, k]
    /// let result = tensor_a.contract(&tensor_b);
    /// assert_eq!(result.dims, vec![2, 4]);
    /// ```
    pub fn contract(&self, other: &Self) -> Self
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
        ));

        Self {
            indices: result_indices,
            dims: result_dims,
            storage: result_storage,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Tensor with static rank `N` (number of indices).
pub struct TensorStaticLen<const N: usize, Id, T, Symm = NoSymmSpace> {
    pub indices: [Index<Id, Symm>; N],
    pub dims: [usize; N],
    pub storage: Arc<Storage>,
    _phantom: std::marker::PhantomData<T>,
}

impl<const N: usize, Id, T, Symm> TensorStaticLen<N, Id, T, Symm> {
    /// Create a new tensor with static rank `N`.
    pub fn new(indices: [Index<Id, Symm>; N], dims: [usize; N], storage: Arc<Storage>) -> Self {
        Self {
            indices,
            dims,
            storage,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get a mutable reference to storage (COW: clones if shared).
    pub fn storage_mut(&mut self) -> &mut Storage {
        Arc::make_mut(&mut self.storage)
    }

    /// Sum all elements, returning `T`.
    ///
    /// For dynamic element tensors, use `T = AnyScalar`.
    pub fn sum(&self) -> T
    where
        T: SumFromStorage,
    {
        T::sum_from_storage(&self.storage)
    }

    /// Sum all elements as f64.
    pub fn sum_f64(&self) -> f64 {
        f64::sum_from_storage(&self.storage)
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
    /// Panics if `new_indices.len() != N`, if any index ID doesn't match, or
    /// if there are duplicate indices.
    ///
    /// # Example
    /// ```
    /// use tensor4all_tensor::TensorStaticLen;
    /// use tensor4all_index::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_tensor::Storage;
    /// use std::sync::Arc;
    ///
    /// // Create a 2×3 tensor
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let indices = [i.clone(), j.clone()];
    /// let dims = [2, 3];
    /// let storage = Arc::new(Storage::new_dense_f64(6));
    /// let tensor: TensorStaticLen<2, DynId, f64> = TensorStaticLen::new(indices, dims, storage);
    ///
    /// // Permute to 3×2: swap the two dimensions by providing new indices order
    /// let permuted = tensor.permute_indices(&[j, i]);
    /// assert_eq!(permuted.dims, [3, 2]);
    /// ```
    pub fn permute_indices(&self, new_indices: &[Index<Id, Symm>; N]) -> Self
    where
        Id: Clone + std::hash::Hash + Eq,
        Symm: Clone + Symmetry,
    {
        // Compute permutation by matching IDs
        let perm = compute_permutation_from_indices(&self.indices, new_indices);

        // Compute new dims from new indices
        let new_dims: [usize; N] = std::array::from_fn(|i| new_indices[i].size());

        // Convert dims to slice for permute_storage
        let dims_slice: &[usize] = &self.dims;

        // Permute storage data using the computed permutation
        let new_storage = Arc::new(permute_storage(&self.storage, dims_slice, &perm));

        Self {
            indices: std::array::from_fn(|i| new_indices[i].clone()),
            dims: new_dims,
            storage: new_storage,
            _phantom: std::marker::PhantomData,
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
    /// Panics if `perm.len() != N` or if the permutation is invalid.
    ///
    /// # Example
    /// ```
    /// use tensor4all_tensor::TensorStaticLen;
    /// use tensor4all_index::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_tensor::Storage;
    /// use std::sync::Arc;
    ///
    /// // Create a 2×3 tensor
    /// let indices = [
    ///     Index::new_dyn(2),
    ///     Index::new_dyn(3),
    /// ];
    /// let dims = [2, 3];
    /// let storage = Arc::new(Storage::new_dense_f64(6));
    /// let tensor: TensorStaticLen<2, DynId, f64> = TensorStaticLen::new(indices, dims, storage);
    ///
    /// // Permute to 3×2: swap the two dimensions
    /// let permuted = tensor.permute(&[1, 0]);
    /// assert_eq!(permuted.dims, [3, 2]);
    /// ```
    pub fn permute(&self, perm: &[usize]) -> Self
    where
        Id: Clone,
        Symm: Clone,
    {
        assert_eq!(
            perm.len(),
            N,
            "permutation length must match tensor rank"
        );

        // Permute indices and dims
        let new_indices: [Index<Id, Symm>; N] = std::array::from_fn(|i| {
            self.indices[perm[i]].clone()
        });
        let new_dims: [usize; N] = std::array::from_fn(|i| {
            self.dims[perm[i]]
        });

        // Convert dims to slice for permute_storage
        let dims_slice: &[usize] = &self.dims;

        // Permute storage data
        let new_storage = Arc::new(permute_storage(&self.storage, dims_slice, perm));

        Self {
            indices: new_indices,
            dims: new_dims,
            storage: new_storage,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Contract this tensor with another tensor along common indices.
    ///
    /// This method finds common indices between `self` and `other`, then contracts
    /// along those indices. The result tensor contains all non-contracted indices
    /// from both tensors, with indices from `self` appearing first, followed by
    /// indices from `other` that are not common.
    ///
    /// Note: The result is always a `TensorDynLen` because the rank of the result
    /// may differ from the input ranks.
    ///
    /// # Arguments
    /// * `other` - The tensor to contract with (can be `TensorStaticLen` or `TensorDynLen`)
    ///
    /// # Returns
    /// A new `TensorDynLen` resulting from the contraction.
    ///
    /// # Panics
    /// Panics if there are no common indices, if common indices have mismatched
    /// dimensions, or if storage types don't match.
    ///
    /// # Example
    /// ```
    /// use tensor4all_tensor::{TensorStaticLen, TensorDynLen};
    /// use tensor4all_index::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_tensor::Storage;
    /// use std::sync::Arc;
    ///
    /// // Create two tensors: A[i, j] and B[j, k]
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let k = Index::new_dyn(4);
    ///
    /// let indices_a = [i.clone(), j.clone()];
    /// let dims_a = [2, 3];
    /// let storage_a = Arc::new(Storage::new_dense_f64(6));
    /// let tensor_a: TensorStaticLen<2, DynId, f64> = TensorStaticLen::new(indices_a, dims_a, storage_a);
    ///
    /// let indices_b = [j.clone(), k.clone()];
    /// let dims_b = [3, 4];
    /// let storage_b = Arc::new(Storage::new_dense_f64(12));
    /// let tensor_b: TensorStaticLen<2, DynId, f64> = TensorStaticLen::new(indices_b, dims_b, storage_b);
    ///
    /// // Contract along j: result is C[i, k]
    /// let result = tensor_a.contract(&tensor_b);
    /// assert_eq!(result.dims, vec![2, 4]);
    /// ```
    pub fn contract<const M: usize>(&self, other: &TensorStaticLen<M, Id, T, Symm>) -> TensorDynLen<Id, T, Symm>
    where
        Id: Clone + std::hash::Hash + Eq,
        Symm: Clone + Symmetry,
    {
        // Convert static indices to slices for common_inds
        let self_indices_slice: &[Index<Id, Symm>] = &self.indices;
        let other_indices_slice: &[Index<Id, Symm>] = &other.indices;

        // Find common indices
        let common = common_inds(self_indices_slice, other_indices_slice);
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

        // Convert dims to slices for contract_storage
        let dims_a_slice: &[usize] = &self.dims;
        let dims_b_slice: &[usize] = &other.dims;

        // Perform contraction
        let result_storage = Arc::new(contract_storage(
            &self.storage,
            dims_a_slice,
            &axes_a,
            &other.storage,
            dims_b_slice,
            &axes_b,
        ));

        TensorDynLen {
            indices: result_indices,
            dims: result_dims,
            storage: result_storage,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Convenience alias for dynamic element type tensors with dynamic rank.
pub type AnyTensorDynLen<Id, Symm = NoSymmSpace> = TensorDynLen<Id, AnyScalar, Symm>;

/// Convenience alias for dynamic element type tensors with static rank.
pub type AnyTensorStaticLen<const N: usize, Id, Symm = NoSymmSpace> = TensorStaticLen<N, Id, AnyScalar, Symm>;

