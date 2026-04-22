use crate::defaults::DynIndex;
use crate::index_like::IndexLike;
use crate::index_ops::{common_ind_positions, prepare_contraction, prepare_contraction_pairs};
use crate::tensor_like::LinearizationOrder;
use crate::{storage::Storage, AnyScalar};
use anyhow::Result;
use num_complex::Complex64;
use num_traits::Zero;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use std::collections::HashSet;
use std::ops::{Mul, Neg, Sub};
use std::sync::{Arc, OnceLock};
use tenferro::eager_einsum::eager_einsum_ad;
use tenferro::{CpuBackend, DType, EagerTensor, Tensor as NativeTensor};
use tensor4all_tensorbackend::{
    axpby_native_tensor, contract_native_tensor, default_eager_ctx,
    dense_native_tensor_from_col_major, diag_native_tensor_from_col_major,
    native_tensor_primal_to_dense_col_major, native_tensor_primal_to_diag_c64,
    native_tensor_primal_to_diag_f64, native_tensor_primal_to_storage,
    reshape_col_major_native_tensor, scale_native_tensor, storage_to_native_tensor, TensorElement,
};

/// Trait for scalar types that can generate random values from a standard
/// normal distribution.
///
/// This enables the generic [`TensorDynLen::random`] constructor.
pub trait RandomScalar: TensorElement {
    /// Generate a random value from the standard normal distribution.
    fn random_value<R: Rng>(rng: &mut R) -> Self;
}

impl RandomScalar for f64 {
    fn random_value<R: Rng>(rng: &mut R) -> Self {
        StandardNormal.sample(rng)
    }
}

impl RandomScalar for Complex64 {
    fn random_value<R: Rng>(rng: &mut R) -> Self {
        Complex64::new(StandardNormal.sample(rng), StandardNormal.sample(rng))
    }
}

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

/// Trait for accessing tensor index metadata.
pub trait TensorAccess {
    /// Get a reference to the indices.
    fn indices(&self) -> &[DynIndex];
}

/// Dynamic-rank dense tensor -- the central data type of tensor4all.
///
/// `TensorDynLen` stores a multi-dimensional array of `f64` or `Complex64`
/// values together with a list of [`DynIndex`] labels. The indices carry
/// unique identities (UUIDs) so that contraction, addition, and other
/// binary operations can automatically match legs by identity rather than
/// position.
///
/// # Key Operations
///
/// | Operation | Method |
/// |-----------|--------|
/// | Create from data | [`from_dense`](Self::from_dense), [`from_diag`](Self::from_diag), [`zeros`](Self::zeros) |
/// | Extract data | [`to_vec`](Self::to_vec), [`sum`](Self::sum), [`only`](Self::only) |
/// | Contraction | [`contract`](Self::contract), `*` operator |
/// | Arithmetic | [`add`](Self::add), [`scale`](Self::scale), [`axpby`](Self::axpby), `-` operator |
/// | Factorization | via [`TensorLike::factorize`] |
/// | Norms | [`norm`](Self::norm), [`norm_squared`](Self::norm_squared), [`maxabs`](Self::maxabs) |
/// | Index ops | [`replaceind`](Self::replaceind), [`permute_indices`](Self::permute_indices) |
///
/// # Data Layout
///
/// Data is stored in **column-major** order (first index varies fastest),
/// matching Fortran, Julia, and ITensors.jl conventions.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{TensorDynLen, DynIndex};
///
/// // Create a 2x3 real tensor
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let t = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// assert_eq!(t.dims(), vec![2, 3]);
/// assert!(t.is_f64());
///
/// // Sum all elements: 1+2+3+4+5+6 = 21
/// let s = t.sum();
/// assert!((s.real() - 21.0).abs() < 1e-12);
///
/// // Extract data back out
/// let data_out = t.to_vec::<f64>().unwrap();
/// assert_eq!(data_out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
#[derive(Clone)]
pub struct TensorDynLen {
    /// Full index information (includes tags and other metadata).
    pub indices: Vec<DynIndex>,
    /// Authoritative compact payload storage.
    pub(crate) storage: Arc<Storage>,
    /// Lazily materialized eager payload for native execution and AD.
    pub(crate) eager_cache: Arc<OnceLock<Arc<EagerTensor<CpuBackend>>>>,
}

impl TensorAccess for TensorDynLen {
    fn indices(&self) -> &[DynIndex] {
        &self.indices
    }
}

impl TensorDynLen {
    const EINSUM_LABELS: &'static [u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    fn dense_axis_classes(rank: usize) -> Vec<usize> {
        (0..rank).collect()
    }

    fn diag_axis_classes(rank: usize) -> Vec<usize> {
        if rank == 0 {
            vec![]
        } else {
            vec![0; rank]
        }
    }

    fn canonicalize_axis_classes(axis_classes: &[usize]) -> Vec<usize> {
        let mut map = std::collections::HashMap::new();
        let mut next = 0usize;
        axis_classes
            .iter()
            .map(|&class_id| {
                *map.entry(class_id).or_insert_with(|| {
                    let canonical = next;
                    next += 1;
                    canonical
                })
            })
            .collect()
    }

    fn permute_axis_classes(&self, perm: &[usize]) -> Vec<usize> {
        let axis_classes = self.storage.axis_classes();
        let permuted: Vec<usize> = perm.iter().map(|&index| axis_classes[index]).collect();
        Self::canonicalize_axis_classes(&permuted)
    }

    fn is_diag_axis_classes(axis_classes: &[usize]) -> bool {
        axis_classes.len() >= 2 && axis_classes.iter().all(|&class_id| class_id == 0)
    }

    fn einsum_labels(ids: &[usize]) -> Result<String> {
        let mut out = String::with_capacity(ids.len());
        for &id in ids {
            let label = Self::EINSUM_LABELS.get(id).ok_or_else(|| {
                anyhow::anyhow!("einsum label {id} exceeds supported label range")
            })?;
            out.push(char::from(*label));
        }
        Ok(out)
    }

    fn build_binary_einsum_subscripts(
        lhs_rank: usize,
        axes_a: &[usize],
        rhs_rank: usize,
        axes_b: &[usize],
    ) -> Result<String> {
        anyhow::ensure!(
            axes_a.len() == axes_b.len(),
            "contract axis length mismatch: lhs {:?}, rhs {:?}",
            axes_a,
            axes_b
        );

        let mut lhs_ids = vec![usize::MAX; lhs_rank];
        let mut rhs_ids = vec![usize::MAX; rhs_rank];
        let mut next_id = 0usize;

        let mut seen_lhs = vec![false; lhs_rank];
        let mut seen_rhs = vec![false; rhs_rank];

        for (&lhs_axis, &rhs_axis) in axes_a.iter().zip(axes_b.iter()) {
            anyhow::ensure!(
                lhs_axis < lhs_rank,
                "lhs contract axis {lhs_axis} out of range"
            );
            anyhow::ensure!(
                rhs_axis < rhs_rank,
                "rhs contract axis {rhs_axis} out of range"
            );
            anyhow::ensure!(
                !seen_lhs[lhs_axis],
                "duplicate lhs contract axis {lhs_axis}"
            );
            anyhow::ensure!(
                !seen_rhs[rhs_axis],
                "duplicate rhs contract axis {rhs_axis}"
            );
            seen_lhs[lhs_axis] = true;
            seen_rhs[rhs_axis] = true;
            lhs_ids[lhs_axis] = next_id;
            rhs_ids[rhs_axis] = next_id;
            next_id += 1;
        }

        let mut output_ids = Vec::with_capacity(lhs_rank + rhs_rank - 2 * axes_a.len());
        for id in &mut lhs_ids {
            if *id == usize::MAX {
                *id = next_id;
                output_ids.push(next_id);
                next_id += 1;
            }
        }
        for id in &mut rhs_ids {
            if *id == usize::MAX {
                *id = next_id;
                output_ids.push(next_id);
                next_id += 1;
            }
        }

        Ok(format!(
            "{},{}->{}",
            Self::einsum_labels(&lhs_ids)?,
            Self::einsum_labels(&rhs_ids)?,
            Self::einsum_labels(&output_ids)?,
        ))
    }

    fn scale_subscripts(rank: usize) -> Result<String> {
        if rank == 0 {
            Ok("->".to_string())
        } else {
            let ids: Vec<usize> = (0..rank).collect();
            let labels = Self::einsum_labels(&ids)?;
            Ok(format!("{labels},->{labels}"))
        }
    }

    fn validate_indices(indices: &[DynIndex]) {
        let mut seen = HashSet::new();
        for idx in indices {
            assert!(
                seen.insert(idx.clone()),
                "Tensor indices must all be unique (no duplicate IDs)"
            );
        }
    }

    fn validate_diag_dims(dims: &[usize]) -> Result<()> {
        if !dims.is_empty() {
            let first_dim = dims[0];
            for (i, &dim) in dims.iter().enumerate() {
                anyhow::ensure!(
                    dim == first_dim,
                    "DiagTensor requires all indices to have the same dimension, but dims[{i}] = {dim} != dims[0] = {first_dim}"
                );
            }
        }
        Ok(())
    }

    fn seed_native_payload(storage: &Storage, dims: &[usize]) -> Result<NativeTensor> {
        storage_to_native_tensor(storage, dims)
    }

    fn empty_eager_cache() -> Arc<OnceLock<Arc<EagerTensor<CpuBackend>>>> {
        Arc::new(OnceLock::new())
    }

    fn eager_cache_with(
        inner: EagerTensor<CpuBackend>,
    ) -> Arc<OnceLock<Arc<EagerTensor<CpuBackend>>>> {
        let cache = Arc::new(OnceLock::new());
        let _ = cache.set(Arc::new(inner));
        cache
    }

    fn storage_from_native_with_axis_classes(
        native: &NativeTensor,
        axis_classes: &[usize],
        logical_rank: usize,
    ) -> Result<Storage> {
        if Self::is_diag_axis_classes(axis_classes) {
            match native.dtype() {
                DType::F32 | DType::F64 => Storage::from_diag_col_major(
                    native_tensor_primal_to_diag_f64(native)?,
                    logical_rank,
                ),
                DType::C32 | DType::C64 => Storage::from_diag_col_major(
                    native_tensor_primal_to_diag_c64(native)?,
                    logical_rank,
                ),
            }
        } else {
            native_tensor_primal_to_storage(native)
        }
    }

    fn validate_storage_matches_indices(indices: &[DynIndex], storage: &Storage) -> Result<()> {
        let dims = Self::expected_dims_from_indices(indices);
        let storage_dims = storage.logical_dims();
        if storage_dims != dims {
            return Err(anyhow::anyhow!(
                "storage logical dims {:?} do not match indices dims {:?}",
                storage_dims,
                dims
            ));
        }
        if storage.is_diag() {
            Self::validate_diag_dims(&dims)?;
        }
        Ok(())
    }

    fn materialized_inner(&self) -> &EagerTensor<CpuBackend> {
        self.eager_cache
            .get_or_init(|| {
                let native = Self::seed_native_payload(self.storage.as_ref(), &self.dims())
                    .unwrap_or_else(|err| panic!("TensorDynLen materialization failed: {err}"));
                Arc::new(EagerTensor::from_tensor_in(native, default_eager_ctx()))
            })
            .as_ref()
    }

    pub(crate) fn as_inner(&self) -> &EagerTensor<CpuBackend> {
        self.materialized_inner()
    }

    /// Compute dims from `indices` order.
    #[inline]
    fn expected_dims_from_indices(indices: &[DynIndex]) -> Vec<usize> {
        indices.iter().map(|idx| idx.dim()).collect()
    }

    /// Get dims in the current `indices` order.
    ///
    /// This is computed on-demand from `indices` (single source of truth).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(3);
    /// let k = DynIndex::new_dyn(4);
    /// let t = TensorDynLen::from_dense(
    ///     vec![i, j, k],
    ///     vec![0.0; 24],
    /// ).unwrap();
    /// assert_eq!(t.dims(), vec![2, 3, 4]);
    /// ```
    pub fn dims(&self) -> Vec<usize> {
        Self::expected_dims_from_indices(&self.indices)
    }

    /// Create a new tensor with dynamic rank.
    ///
    /// # Panics
    /// Panics if the storage is Diag and not all indices have the same dimension.
    /// Panics if there are duplicate indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen, Storage};
    /// use std::sync::Arc;
    ///
    /// let i = DynIndex::new_dyn(3);
    /// let storage = Arc::new(Storage::new_dense::<f64>(3));
    /// let t = TensorDynLen::new(vec![i], storage);
    /// assert_eq!(t.dims(), vec![3]);
    /// ```
    pub fn new(indices: Vec<DynIndex>, storage: Arc<Storage>) -> Self {
        match Self::from_storage(indices, storage) {
            Ok(tensor) => tensor,
            Err(err) => panic!("TensorDynLen::new failed: {err}"),
        }
    }

    /// Create a new tensor with dynamic rank, automatically computing dimensions from indices.
    ///
    /// This is a convenience constructor that extracts dimensions from indices using `IndexLike::dim()`.
    ///
    /// # Panics
    /// Panics if the storage is Diag and not all indices have the same dimension.
    /// Panics if there are duplicate indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen, Storage};
    /// use std::sync::Arc;
    ///
    /// let i = DynIndex::new_dyn(4);
    /// let storage = Arc::new(Storage::new_dense::<f64>(4));
    /// let t = TensorDynLen::from_indices(vec![i], storage);
    /// assert_eq!(t.dims(), vec![4]);
    /// ```
    pub fn from_indices(indices: Vec<DynIndex>, storage: Arc<Storage>) -> Self {
        Self::new(indices, storage)
    }

    /// Create a tensor from explicit compact storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen, Storage};
    /// use std::sync::Arc;
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let storage = Arc::new(Storage::new_diag(vec![1.0_f64, 2.0]));
    /// let t = TensorDynLen::from_storage(vec![i, j], storage).unwrap();
    /// assert_eq!(t.dims(), vec![2, 2]);
    /// ```
    pub fn from_storage(indices: Vec<DynIndex>, storage: Arc<Storage>) -> Result<Self> {
        Self::validate_indices(&indices);
        Self::validate_storage_matches_indices(&indices, storage.as_ref())?;
        Ok(Self {
            indices,
            storage,
            eager_cache: Self::empty_eager_cache(),
        })
    }

    /// Create a tensor from explicit structured storage.
    ///
    /// This is an alias for [`TensorDynLen::from_storage`] with a name that
    /// emphasizes that compact structured metadata is preserved.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage logical dimensions do not match the
    /// supplied indices, or if duplicate indices are provided.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tensor4all_core::{DynIndex, Storage, StorageKind, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let storage = Arc::new(Storage::from_diag_col_major(vec![1.0_f64, 2.0], 2).unwrap());
    /// let tensor = TensorDynLen::from_structured_storage(vec![i, j], storage).unwrap();
    /// assert_eq!(tensor.storage().storage_kind(), StorageKind::Diagonal);
    /// ```
    pub fn from_structured_storage(indices: Vec<DynIndex>, storage: Arc<Storage>) -> Result<Self> {
        Self::from_storage(indices, storage)
    }

    /// Create a tensor from a native tenferro payload.
    pub(crate) fn from_native(indices: Vec<DynIndex>, native: NativeTensor) -> Result<Self> {
        let axis_classes = Self::dense_axis_classes(indices.len());
        Self::from_native_with_axis_classes(indices, native, axis_classes)
    }

    pub(crate) fn from_native_with_axis_classes(
        indices: Vec<DynIndex>,
        native: NativeTensor,
        axis_classes: Vec<usize>,
    ) -> Result<Self> {
        Self::from_inner_with_axis_classes(
            indices,
            EagerTensor::from_tensor_in(native, default_eager_ctx()),
            axis_classes,
        )
    }

    pub(crate) fn from_inner(
        indices: Vec<DynIndex>,
        inner: EagerTensor<CpuBackend>,
    ) -> Result<Self> {
        let axis_classes = Self::dense_axis_classes(indices.len());
        Self::from_inner_with_axis_classes(indices, inner, axis_classes)
    }

    pub(crate) fn from_inner_with_axis_classes(
        indices: Vec<DynIndex>,
        inner: EagerTensor<CpuBackend>,
        axis_classes: Vec<usize>,
    ) -> Result<Self> {
        let dims = Self::expected_dims_from_indices(&indices);
        Self::validate_indices(&indices);
        if dims != inner.data().shape() {
            return Err(anyhow::anyhow!(
                "native payload dims {:?} do not match indices dims {:?}",
                inner.data().shape(),
                dims
            ));
        }
        if Self::is_diag_axis_classes(&axis_classes) {
            Self::validate_diag_dims(&dims)?;
        }
        let storage = Self::storage_from_native_with_axis_classes(
            inner.data(),
            &axis_classes,
            indices.len(),
        )?;
        Ok(Self {
            indices,
            storage: Arc::new(storage),
            eager_cache: Self::eager_cache_with(inner),
        })
    }

    /// Borrow the indices.
    pub fn indices(&self) -> &[DynIndex] {
        &self.indices
    }

    /// Borrow the native payload.
    pub(crate) fn as_native(&self) -> &NativeTensor {
        self.materialized_inner().data()
    }

    /// Enable reverse-mode AD tracking on this tensor by creating a tracked leaf.
    pub fn enable_grad(self) -> Self {
        let native = self.as_native().clone();
        Self {
            indices: self.indices,
            storage: self.storage,
            eager_cache: Self::eager_cache_with(EagerTensor::requires_grad_in(
                native,
                default_eager_ctx(),
            )),
        }
    }

    /// Report whether this tensor participates in gradient tracking.
    pub fn tracks_grad(&self) -> bool {
        self.eager_cache
            .get()
            .is_some_and(|inner| inner.tracks_grad())
    }

    /// Return the accumulated gradient, if one has been stored.
    pub fn grad(&self) -> Result<Option<Self>> {
        self.materialized_inner()
            .grad()
            .map(|grad| {
                Self::from_native_with_axis_classes(
                    self.indices.clone(),
                    grad.as_ref().clone(),
                    self.storage.axis_classes().to_vec(),
                )
            })
            .transpose()
    }

    /// Clear the accumulated gradient stored for this tensor.
    pub fn clear_grad(&self) -> Result<()> {
        if let Some(inner) = self.eager_cache.get() {
            inner.clear_grad();
        }
        Ok(())
    }

    /// Run reverse-mode autodiff from this scalar tensor.
    pub fn backward(&self) -> Result<()> {
        self.materialized_inner()
            .backward()
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("TensorDynLen::backward failed: {e}"))
    }

    /// Detach this tensor from the reverse graph.
    pub fn detach(&self) -> Self {
        Self::from_inner_with_axis_classes(
            self.indices.clone(),
            self.materialized_inner().detach(),
            self.storage.axis_classes().to_vec(),
        )
        .expect("TensorDynLen::detach returned invalid tensor")
    }

    /// Check if this tensor is already in canonical form.
    pub fn is_simple(&self) -> bool {
        true
    }

    /// Materialize the primal snapshot as storage.
    pub fn to_storage(&self) -> Result<Arc<Storage>> {
        Ok(Arc::clone(&self.storage))
    }

    /// Returns the authoritative compact storage.
    pub fn storage(&self) -> Arc<Storage> {
        Arc::clone(&self.storage)
    }

    /// Sum all elements, returning `AnyScalar`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(3);
    /// let t = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0, 3.0]).unwrap();
    /// let s = t.sum();
    /// assert!((s.real() - 6.0).abs() < 1e-12);
    /// ```
    pub fn sum(&self) -> AnyScalar {
        if self.indices.is_empty() {
            return AnyScalar::from_tensor_unchecked(self.clone());
        }
        let axes: Vec<usize> = (0..self.indices.len()).collect();
        let reduced = self
            .materialized_inner()
            .reduce_sum(&axes)
            .unwrap_or_else(|e| panic!("TensorDynLen::sum failed: {e}"));
        AnyScalar::from_tensor_unchecked(
            Self::from_inner(Vec::new(), reduced)
                .unwrap_or_else(|e| panic!("TensorDynLen::sum returned invalid scalar: {e}")),
        )
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
    /// use tensor4all_core::{TensorDynLen, AnyScalar};
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// // Create a scalar tensor (0 dimensions, 1 element)
    /// let indices: Vec<Index<DynId>> = vec![];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(indices, vec![42.0]).unwrap();
    ///
    /// assert_eq!(tensor.only().real(), 42.0);
    /// ```
    pub fn only(&self) -> AnyScalar {
        let dims = self.dims();
        let total_size: usize = dims.iter().product();
        assert!(
            total_size == 1 || dims.is_empty(),
            "only() requires a scalar tensor (1 element), got {} elements with dims {:?}",
            if dims.is_empty() { 1 } else { total_size },
            dims
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
    ///
    /// // Create a 2×3 tensor
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let indices = vec![i.clone(), j.clone()];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(indices, vec![0.0; 6]).unwrap();
    ///
    /// // Permute to 3×2: swap the two dimensions by providing new indices order
    /// let permuted = tensor.permute_indices(&[j, i]);
    /// assert_eq!(permuted.dims(), vec![3, 2]);
    /// ```
    pub fn permute_indices(&self, new_indices: &[DynIndex]) -> Self {
        // Compute permutation by matching IDs
        let perm = compute_permutation_from_indices(&self.indices, new_indices);

        let permuted = self
            .materialized_inner()
            .transpose(&perm)
            .unwrap_or_else(|e| panic!("TensorDynLen::permute_indices failed: {e}"));
        let axis_classes = self.permute_axis_classes(&perm);
        Self::from_inner_with_axis_classes(new_indices.to_vec(), permuted, axis_classes)
            .expect("TensorDynLen::permute_indices returned invalid tensor")
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
    /// Panics if `perm.len() != self.indices.len()` or if the permutation is invalid.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// // Create a 2×3 tensor
    /// let indices = vec![
    ///     Index::new_dyn(2),
    ///     Index::new_dyn(3),
    /// ];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(indices, vec![0.0; 6]).unwrap();
    ///
    /// // Permute to 3×2: swap the two dimensions
    /// let permuted = tensor.permute(&[1, 0]);
    /// assert_eq!(permuted.dims(), vec![3, 2]);
    /// ```
    pub fn permute(&self, perm: &[usize]) -> Self {
        assert_eq!(
            perm.len(),
            self.indices.len(),
            "permutation length must match tensor rank"
        );

        // Permute indices
        let new_indices: Vec<DynIndex> = perm.iter().map(|&i| self.indices[i].clone()).collect();
        let permuted = self
            .materialized_inner()
            .transpose(perm)
            .unwrap_or_else(|e| panic!("TensorDynLen::permute failed: {e}"));
        let axis_classes = self.permute_axis_classes(perm);
        Self::from_inner_with_axis_classes(new_indices, permuted, axis_classes)
            .expect("TensorDynLen::permute returned invalid tensor")
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
    ///
    /// // Create two tensors: A[i, j] and B[j, k]
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let k = Index::new_dyn(4);
    ///
    /// let indices_a = vec![i.clone(), j.clone()];
    /// let tensor_a: TensorDynLen = TensorDynLen::from_dense(indices_a, vec![0.0; 6]).unwrap();
    ///
    /// let indices_b = vec![j.clone(), k.clone()];
    /// let tensor_b: TensorDynLen = TensorDynLen::from_dense(indices_b, vec![0.0; 12]).unwrap();
    ///
    /// // Contract along j: result is C[i, k]
    /// let result = tensor_a.contract(&tensor_b);
    /// assert_eq!(result.dims(), vec![2, 4]);
    /// ```
    pub fn contract(&self, other: &Self) -> Self {
        let self_dims = Self::expected_dims_from_indices(&self.indices);
        let other_dims = Self::expected_dims_from_indices(&other.indices);
        let spec = prepare_contraction(&self.indices, &self_dims, &other.indices, &other_dims)
            .expect("contraction preparation failed");

        if self.indices.is_empty() && other.indices.is_empty() {
            let result = self
                .materialized_inner()
                .mul(other.materialized_inner())
                .unwrap_or_else(|e| panic!("TensorDynLen::contract scalar multiply failed: {e}"));
            return Self::from_inner(spec.result_indices, result)
                .expect("TensorDynLen::contract returned invalid scalar");
        }

        if self.as_native().dtype() != other.as_native().dtype() {
            let result_native = contract_native_tensor(
                self.as_native(),
                &spec.axes_a,
                other.as_native(),
                &spec.axes_b,
            )
            .unwrap_or_else(|e| panic!("TensorDynLen::contract native fallback failed: {e}"));
            return Self::from_native(spec.result_indices, result_native)
                .expect("TensorDynLen::contract native fallback returned invalid tensor");
        }

        let subscripts = Self::build_binary_einsum_subscripts(
            self.indices.len(),
            &spec.axes_a,
            other.indices.len(),
            &spec.axes_b,
        )
        .expect("TensorDynLen::contract failed to build einsum subscripts");
        let result = eager_einsum_ad(
            &[self.materialized_inner(), other.materialized_inner()],
            &subscripts,
        )
        .unwrap_or_else(|e| panic!("TensorDynLen::contract failed: {e}"));
        Self::from_inner(spec.result_indices, result)
            .expect("TensorDynLen::contract returned invalid tensor")
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
    ///
    /// // Create two tensors: A[i, j] and B[k, l] where j and k have same dimension but different IDs
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let k = Index::new_dyn(3);  // Same dimension as j, but different ID
    /// let l = Index::new_dyn(4);
    ///
    /// let indices_a = vec![i.clone(), j.clone()];
    /// let tensor_a: TensorDynLen = TensorDynLen::from_dense(indices_a, vec![0.0; 6]).unwrap();
    ///
    /// let indices_b = vec![k.clone(), l.clone()];
    /// let tensor_b: TensorDynLen = TensorDynLen::from_dense(indices_b, vec![0.0; 12]).unwrap();
    ///
    /// // Contract j (from A) with k (from B): result is C[i, l]
    /// let result = tensor_a.tensordot(&tensor_b, &[(j.clone(), k.clone())]).unwrap();
    /// assert_eq!(result.dims(), vec![2, 4]);
    /// ```
    pub fn tensordot(&self, other: &Self, pairs: &[(DynIndex, DynIndex)]) -> Result<Self> {
        use crate::index_ops::ContractionError;

        let self_dims = Self::expected_dims_from_indices(&self.indices);
        let other_dims = Self::expected_dims_from_indices(&other.indices);
        let spec = prepare_contraction_pairs(
            &self.indices,
            &self_dims,
            &other.indices,
            &other_dims,
            pairs,
        )
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

        if self.indices.is_empty() && other.indices.is_empty() {
            let result = self
                .materialized_inner()
                .mul(other.materialized_inner())
                .map_err(|e| anyhow::anyhow!("tensordot scalar multiply failed: {e}"))?;
            return Self::from_inner(spec.result_indices, result);
        }

        if self.as_native().dtype() != other.as_native().dtype() {
            let result_native = contract_native_tensor(
                self.as_native(),
                &spec.axes_a,
                other.as_native(),
                &spec.axes_b,
            )?;
            return Self::from_native(spec.result_indices, result_native);
        }

        let subscripts = Self::build_binary_einsum_subscripts(
            self.indices.len(),
            &spec.axes_a,
            other.indices.len(),
            &spec.axes_b,
        )?;
        let result = eager_einsum_ad(
            &[self.materialized_inner(), other.materialized_inner()],
            &subscripts,
        )
        .map_err(|e| anyhow::anyhow!("tensordot failed: {e}"))?;
        Self::from_inner(spec.result_indices, result)
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
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let tensor_a: TensorDynLen = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    /// let tensor_b: TensorDynLen =
    ///     TensorDynLen::from_dense(vec![j.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    ///
    /// // Outer product: C[i, j] = A[i] * B[j]
    /// let result = tensor_a.outer_product(&tensor_b).unwrap();
    /// assert_eq!(result.dims(), vec![2, 3]);
    /// ```
    pub fn outer_product(&self, other: &Self) -> Result<Self> {
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
        if self.as_native().dtype() != other.as_native().dtype() {
            let result_native =
                contract_native_tensor(self.as_native(), &[], other.as_native(), &[])?;
            return Self::from_native(result_indices, result_native);
        }

        let subscripts = Self::build_binary_einsum_subscripts(
            self.indices.len(),
            &[],
            other.indices.len(),
            &[],
        )?;
        let result = eager_einsum_ad(
            &[self.materialized_inner(), other.materialized_inner()],
            &subscripts,
        )
        .map_err(|e| anyhow::anyhow!("outer_product failed: {e}"))?;
        Self::from_inner(result_indices, result)
    }
}

// ============================================================================
// Random tensor generation
// ============================================================================

impl TensorDynLen {
    /// Create a random tensor with values from standard normal distribution (generic over scalar type).
    ///
    /// For `f64`, each element is drawn from the standard normal distribution.
    /// For `Complex64`, both real and imaginary parts are drawn independently.
    ///
    /// # Type Parameters
    /// * `T` - The scalar element type (must implement [`RandomScalar`])
    /// * `R` - The random number generator type
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
    /// let tensor: TensorDynLen = TensorDynLen::random::<f64, _>(&mut rng, vec![i, j]);
    /// assert_eq!(tensor.dims(), vec![2, 3]);
    /// ```
    pub fn random<T: RandomScalar, R: Rng>(rng: &mut R, indices: Vec<DynIndex>) -> Self {
        let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
        let size: usize = dims.iter().product();
        let data: Vec<T> = (0..size).map(|_| T::random_value(rng)).collect();
        Self::from_dense(indices, data).expect("TensorDynLen::random failed")
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
///
/// // Create two tensors: A[i, j] and B[j, k]
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let tensor_a: TensorDynLen = TensorDynLen::from_dense(indices_a, vec![0.0; 6]).unwrap();
///
/// let indices_b = vec![j.clone(), k.clone()];
/// let tensor_b: TensorDynLen = TensorDynLen::from_dense(indices_b, vec![0.0; 12]).unwrap();
///
/// // Contract along j using * operator: result is C[i, k]
/// let result = &tensor_a * &tensor_b;
/// assert_eq!(result.dims(), vec![2, 4]);
/// ```
impl Mul<&TensorDynLen> for &TensorDynLen {
    type Output = TensorDynLen;

    fn mul(self, other: &TensorDynLen) -> Self::Output {
        self.contract(other)
    }
}

/// Implement multiplication operator for tensor contraction (owned version).
///
/// This allows using `tensor_a * tensor_b` when both tensors are owned.
impl Mul<TensorDynLen> for TensorDynLen {
    type Output = TensorDynLen;

    fn mul(self, other: TensorDynLen) -> Self::Output {
        self.contract(&other)
    }
}

/// Implement multiplication operator for tensor contraction (mixed reference/owned).
impl Mul<TensorDynLen> for &TensorDynLen {
    type Output = TensorDynLen;

    fn mul(self, other: TensorDynLen) -> Self::Output {
        self.contract(&other)
    }
}

/// Implement multiplication operator for tensor contraction (mixed owned/reference).
impl Mul<&TensorDynLen> for TensorDynLen {
    type Output = TensorDynLen;

    fn mul(self, other: &TensorDynLen) -> Self::Output {
        self.contract(other)
    }
}

impl Sub<&TensorDynLen> for &TensorDynLen {
    type Output = TensorDynLen;

    fn sub(self, other: &TensorDynLen) -> Self::Output {
        TensorDynLen::axpby(
            self,
            AnyScalar::new_real(1.0),
            other,
            AnyScalar::new_real(-1.0),
        )
        .expect("tensor subtraction failed")
    }
}

impl Sub<TensorDynLen> for TensorDynLen {
    type Output = TensorDynLen;

    fn sub(self, other: TensorDynLen) -> Self::Output {
        Sub::sub(&self, &other)
    }
}

impl Sub<TensorDynLen> for &TensorDynLen {
    type Output = TensorDynLen;

    fn sub(self, other: TensorDynLen) -> Self::Output {
        Sub::sub(self, &other)
    }
}

impl Sub<&TensorDynLen> for TensorDynLen {
    type Output = TensorDynLen;

    fn sub(self, other: &TensorDynLen) -> Self::Output {
        Sub::sub(&self, other)
    }
}

impl Neg for &TensorDynLen {
    type Output = TensorDynLen;

    fn neg(self) -> Self::Output {
        TensorDynLen::scale(self, AnyScalar::new_real(-1.0)).expect("tensor negation failed")
    }
}

impl Neg for TensorDynLen {
    type Output = TensorDynLen;

    fn neg(self) -> Self::Output {
        Neg::neg(&self)
    }
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
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    ///
    /// let indices_a = vec![i.clone(), j.clone()];
    /// let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor_a: TensorDynLen = TensorDynLen::from_dense(indices_a, data_a).unwrap();
    ///
    /// let indices_b = vec![i.clone(), j.clone()];
    /// let data_b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    /// let tensor_b: TensorDynLen = TensorDynLen::from_dense(indices_b, data_b).unwrap();
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
        let self_expected_dims = Self::expected_dims_from_indices(&self.indices);
        let other_expected_dims = Self::expected_dims_from_indices(&other_aligned.indices);
        if self_expected_dims != other_expected_dims {
            use crate::TagSetLike;
            let fmt = |indices: &[DynIndex]| -> Vec<String> {
                indices
                    .iter()
                    .map(|idx| {
                        let tags: Vec<String> = idx.tags().iter().collect();
                        format!("{:?}(dim={},tags={:?})", idx.id(), idx.dim(), tags)
                    })
                    .collect()
            };
            return Err(anyhow::anyhow!(
                "Dimension mismatch after alignment.\n\
                 self: dims={:?}, indices(order)={:?}\n\
                 other_aligned: dims={:?}, indices(order)={:?}",
                self_expected_dims,
                fmt(&self.indices),
                other_expected_dims,
                fmt(&other_aligned.indices)
            ));
        }

        self.axpby(
            AnyScalar::new_real(1.0),
            &other_aligned,
            AnyScalar::new_real(1.0),
        )
    }

    /// Compute a linear combination: `a * self + b * other`.
    ///
    /// Both tensors must have the same set of indices (matched by ID).
    /// If indices are in a different order, `other` is automatically permuted
    /// to match `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    /// let b = TensorDynLen::from_dense(vec![i.clone()], vec![3.0, 4.0]).unwrap();
    ///
    /// // 2*a + 3*b = [2+9, 4+12] = [11, 16]
    /// let result = a.axpby(AnyScalar::new_real(2.0), &b, AnyScalar::new_real(3.0)).unwrap();
    /// let data = result.to_vec::<f64>().unwrap();
    /// assert!((data[0] - 11.0).abs() < 1e-12);
    /// assert!((data[1] - 16.0).abs() < 1e-12);
    /// ```
    pub fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self> {
        // Validate that both tensors have the same number of indices.
        if self.indices.len() != other.indices.len() {
            return Err(anyhow::anyhow!(
                "Index count mismatch: self has {} indices, other has {}",
                self.indices.len(),
                other.indices.len()
            ));
        }

        // Validate that both tensors have the same set of indices.
        let self_set: HashSet<_> = self.indices.iter().collect();
        let other_set: HashSet<_> = other.indices.iter().collect();
        if self_set != other_set {
            return Err(anyhow::anyhow!(
                "Index set mismatch: tensors must have the same indices"
            ));
        }

        // Align other tensor axis order to self.
        let other_aligned = other.permute_indices(&self.indices);

        // Validate dimensions match after alignment.
        let self_expected_dims = Self::expected_dims_from_indices(&self.indices);
        let other_expected_dims = Self::expected_dims_from_indices(&other_aligned.indices);
        if self_expected_dims != other_expected_dims {
            return Err(anyhow::anyhow!(
                "Dimension mismatch after alignment: self={:?}, other_aligned={:?}",
                self_expected_dims,
                other_expected_dims
            ));
        }

        let axis_classes = if self.storage.axis_classes() == other_aligned.storage.axis_classes() {
            self.storage.axis_classes().to_vec()
        } else {
            Self::dense_axis_classes(self.indices.len())
        };

        if self.as_native().dtype() != other_aligned.as_native().dtype()
            || self.as_native().dtype() != a.as_tensor().as_native().dtype()
            || other_aligned.as_native().dtype() != b.as_tensor().as_native().dtype()
        {
            let combined = axpby_native_tensor(
                self.as_native(),
                &a.to_backend_scalar(),
                other_aligned.as_native(),
                &b.to_backend_scalar(),
            )?;
            return Self::from_native_with_axis_classes(
                self.indices.clone(),
                combined,
                axis_classes,
            );
        }

        let lhs = self.scale(a)?;
        let rhs = other_aligned.scale(b)?;
        let combined = lhs
            .materialized_inner()
            .add(rhs.materialized_inner())
            .map_err(|e| anyhow::anyhow!("tensor addition failed: {e}"))?;
        Self::from_inner_with_axis_classes(self.indices.clone(), combined, axis_classes)
    }

    /// Scalar multiplication.
    ///
    /// Multiplies every element by `scalar`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(3);
    /// let t = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0, 3.0]).unwrap();
    /// let scaled = t.scale(AnyScalar::new_real(2.0)).unwrap();
    /// assert_eq!(scaled.to_vec::<f64>().unwrap(), vec![2.0, 4.0, 6.0]);
    /// ```
    pub fn scale(&self, scalar: AnyScalar) -> Result<Self> {
        if self.as_native().dtype() != scalar.as_tensor().as_native().dtype() {
            let scaled = scale_native_tensor(self.as_native(), &scalar.to_backend_scalar())?;
            return Self::from_native_with_axis_classes(
                self.indices.clone(),
                scaled,
                self.storage.axis_classes().to_vec(),
            );
        }

        let scaled = if self.indices.is_empty() {
            self.materialized_inner()
                .mul(scalar.as_tensor().materialized_inner())
                .map_err(|e| anyhow::anyhow!("scalar multiplication failed: {e}"))?
        } else {
            let subscripts = Self::scale_subscripts(self.indices.len())?;
            eager_einsum_ad(
                &[
                    self.materialized_inner(),
                    scalar.as_tensor().materialized_inner(),
                ],
                &subscripts,
            )
            .map_err(|e| anyhow::anyhow!("tensor scaling failed: {e}"))?
        };
        Self::from_inner_with_axis_classes(
            self.indices.clone(),
            scaled,
            self.storage.axis_classes().to_vec(),
        )
    }

    /// Inner product (dot product) of two tensors.
    ///
    /// Computes `⟨self, other⟩ = Σ conj(self)_i * other_i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(3);
    /// let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    /// let b = TensorDynLen::from_dense(vec![i.clone()], vec![4.0, 5.0, 6.0]).unwrap();
    ///
    /// // <a, b> = 1*4 + 2*5 + 3*6 = 32
    /// let ip = a.inner_product(&b).unwrap();
    /// assert!((ip.real() - 32.0).abs() < 1e-12);
    /// ```
    pub fn inner_product(&self, other: &Self) -> Result<AnyScalar> {
        if self.indices.len() == other.indices.len() {
            let self_set: HashSet<_> = self.indices.iter().collect();
            let other_set: HashSet<_> = other.indices.iter().collect();
            if self_set == other_set {
                let other_aligned = other.permute_indices(&self.indices);
                let result = self.conj().contract(&other_aligned);
                return Ok(result.sum());
            }
        }

        // Contract self.conj() with other over all indices
        let conj_self = self.conj();
        let result =
            super::contract::contract_multi(&[&conj_self, other], crate::AllowedPairs::All)?;
        // Result should be a scalar (no indices)
        Ok(result.sum())
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
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let new_i = Index::new_dyn(2);  // Same dimension, different ID
    ///
    /// let indices = vec![i.clone(), j.clone()];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(indices, vec![0.0; 6]).unwrap();
    ///
    /// // Replace index i with new_i
    /// let replaced = tensor.replaceind(&i, &new_i);
    /// assert_eq!(replaced.indices[0].id, new_i.id);
    /// assert_eq!(replaced.indices[1].id, j.id);
    /// ```
    pub fn replaceind(&self, old_index: &DynIndex, new_index: &DynIndex) -> Self {
        // Validate dimension match
        if old_index.dim() != new_index.dim() {
            panic!(
                "Index space mismatch: cannot replace index with dimension {} with index of dimension {}",
                old_index.dim(),
                new_index.dim()
            );
        }

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

        Self {
            indices: new_indices,
            storage: Arc::clone(&self.storage),
            eager_cache: Arc::clone(&self.eager_cache),
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
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let new_i = Index::new_dyn(2);
    /// let new_j = Index::new_dyn(3);
    ///
    /// let indices = vec![i.clone(), j.clone()];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(indices, vec![0.0; 6]).unwrap();
    ///
    /// // Replace both indices
    /// let replaced = tensor.replaceinds(&[i.clone(), j.clone()], &[new_i.clone(), new_j.clone()]);
    /// assert_eq!(replaced.indices[0].id, new_i.id);
    /// assert_eq!(replaced.indices[1].id, new_j.id);
    /// ```
    pub fn replaceinds(&self, old_indices: &[DynIndex], new_indices: &[DynIndex]) -> Self {
        assert_eq!(
            old_indices.len(),
            new_indices.len(),
            "old_indices and new_indices must have the same length"
        );

        // Validate dimension matches for all replacements
        for (old, new) in old_indices.iter().zip(new_indices.iter()) {
            if old.dim() != new.dim() {
                panic!(
                    "Index space mismatch: cannot replace index with dimension {} with index of dimension {}",
                    old.dim(),
                    new.dim()
                );
            }
        }

        // Build a map from old indices to new indices
        let replacement_map: std::collections::HashMap<_, _> =
            old_indices.iter().zip(new_indices.iter()).collect();

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

        Self {
            indices: new_indices_vec,
            storage: Arc::clone(&self.storage),
            eager_cache: Arc::clone(&self.eager_cache),
        }
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
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use num_complex::Complex64;
    ///
    /// let i = Index::new_dyn(2);
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(vec![i], data).unwrap();
    ///
    /// let conj_tensor = tensor.conj();
    /// // Elements are now conjugated: 1-2i, 3+4i
    /// ```
    pub fn conj(&self) -> Self {
        // Conjugate tensor: conjugate storage data and map indices via IndexLike::conj()
        // For default undirected indices, conj() is a no-op, so this is future-proof
        // for QSpace-compatible directed indices where conj() flips Ket <-> Bra
        let new_indices: Vec<DynIndex> = self.indices.iter().map(|idx| idx.conj()).collect();
        let conjugated = self
            .materialized_inner()
            .conj()
            .unwrap_or_else(|e| panic!("TensorDynLen::conj failed: {e}"));
        Self::from_inner_with_axis_classes(
            new_indices,
            conjugated,
            self.storage.axis_classes().to_vec(),
        )
        .expect("TensorDynLen::conj returned invalid tensor")
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
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];  // 1² + 2² + ... + 6² = 91
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(vec![i, j], data).unwrap();
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
        let scalar = self.contract(&conj);
        // The mathematical result is nonnegative and real. Clamp tiny negative
        // roundoff so downstream `sqrt` stays well-defined for complex tensors.
        scalar.sum().real().max(0.0)
    }

    /// Compute the Frobenius norm of the tensor: ||T|| = sqrt(Σ|T_ijk...|²)
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let data = vec![3.0, 4.0];  // sqrt(9 + 16) = 5
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(vec![i], data).unwrap();
    ///
    /// assert!((tensor.norm() - 5.0).abs() < 1e-10);
    /// ```
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Maximum absolute value of all elements (L-infinity norm).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(4);
    /// let t = TensorDynLen::from_dense(vec![i], vec![-5.0, 1.0, 3.0, -2.0]).unwrap();
    /// assert!((t.maxabs() - 5.0).abs() < 1e-12);
    /// ```
    pub fn maxabs(&self) -> f64 {
        self.to_storage()
            .map(|storage| storage.max_abs())
            .unwrap_or(0.0)
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
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let data_a = vec![1.0, 0.0];
    /// let data_b = vec![1.0, 0.0];  // Same tensor
    /// let tensor_a: TensorDynLen = TensorDynLen::from_dense(vec![i.clone()], data_a).unwrap();
    /// let tensor_b: TensorDynLen = TensorDynLen::from_dense(vec![i.clone()], data_b).unwrap();
    ///
    /// assert!(tensor_a.distance(&tensor_b) < 1e-10);  // Zero distance
    /// ```
    pub fn distance(&self, other: &Self) -> f64 {
        let norm_self = self.norm();

        // Compute A - B = A + (-1) * B
        let neg_other = other
            .scale(AnyScalar::new_real(-1.0))
            .expect("distance: tensor scaling failed");
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
            .field("dims", &self.dims())
            .field("is_diag", &self.is_diag())
            .finish()
    }
}

/// Create a diagonal tensor with dynamic rank from diagonal data.
///
/// # Arguments
/// * `indices` - The indices for the tensor (all must have the same dimension)
/// * `diag_data` - The diagonal elements (length must equal the dimension of indices)
///
/// The public native bridge currently materializes diagonal payloads densely, so
/// the returned tensor is mathematically diagonal but may not report
/// [`TensorDynLen::is_diag`] at the native-storage level.
///
/// # Panics
/// Panics if indices have different dimensions, or if diag_data length doesn't match.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, diag_tensor_dyn_len};
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(3);
/// let t = diag_tensor_dyn_len(vec![i, j], vec![1.0, 2.0, 3.0]);
/// assert_eq!(t.dims(), vec![3, 3]);
/// ```
pub fn diag_tensor_dyn_len(indices: Vec<DynIndex>, diag_data: Vec<f64>) -> TensorDynLen {
    TensorDynLen::from_diag(indices, diag_data)
        .unwrap_or_else(|err| panic!("diag_tensor_dyn_len failed: {err}"))
}

/// Unfold a tensor into a matrix by splitting indices into left and right groups.
///
/// This function validates the split, permutes the tensor so that left indices
/// come first, and returns a rank-2 native tenferro tensor along with metadata.
///
/// # Arguments
/// * `t` - Input tensor
/// * `left_inds` - Indices to place on the left (row) side of the matrix
///
/// # Returns
/// A tuple `(matrix_tensor, left_len, m, n, left_indices, right_indices)` where:
/// - `matrix_tensor` is a rank-2 `tenferro::Tensor` with shape `[m, n]`
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
/// - Native reshape fails
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen, unfold_split};
///
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// // 2x3 dense tensor with data [1..6]
/// let t = TensorDynLen::from_dense(
///     vec![i.clone(), j.clone()],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ).unwrap();
///
/// let (matrix, left_len, m, n, left_indices, right_indices) =
///     unfold_split(&t, &[i]).unwrap();
/// assert_eq!(left_len, 1);
/// assert_eq!(m, 2);
/// assert_eq!(n, 3);
/// assert_eq!(left_indices.len(), 1);
/// assert_eq!(right_indices.len(), 1);
/// ```
#[allow(clippy::type_complexity)]
pub fn unfold_split(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
) -> Result<(
    NativeTensor,
    usize,
    usize,
    usize,
    Vec<DynIndex>,
    Vec<DynIndex>,
)> {
    let rank = t.indices.len();

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
        anyhow::ensure!(left_set.insert(left_idx), "Duplicate index in left_inds");
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
    let unfolded_dims = unfolded.dims();
    let m: usize = unfolded_dims[..left_len].iter().product();
    let n: usize = unfolded_dims[left_len..].iter().product();

    let matrix_tensor = reshape_col_major_native_tensor(unfolded.as_native(), &[m, n])?;

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
// TensorIndex implementation for TensorDynLen
// ============================================================================

use crate::tensor_index::TensorIndex;

impl TensorIndex for TensorDynLen {
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

    fn replaceinds(&self, old_indices: &[DynIndex], new_indices: &[DynIndex]) -> Result<Self> {
        // Delegate to the inherent method
        Ok(TensorDynLen::replaceinds(self, old_indices, new_indices))
    }
}

// ============================================================================
// TensorLike implementation for TensorDynLen
// ============================================================================

use crate::tensor_like::{FactorizeError, FactorizeOptions, FactorizeResult, TensorLike};

impl TensorLike for TensorDynLen {
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
        Ok(crate::tensor_like::DirectSumResult {
            tensor,
            new_indices,
        })
    }

    fn outer_product(&self, other: &Self) -> Result<Self> {
        // Delegate to the inherent method
        TensorDynLen::outer_product(self, other)
    }

    fn norm_squared(&self) -> f64 {
        // Delegate to the inherent method
        TensorDynLen::norm_squared(self)
    }

    fn maxabs(&self) -> f64 {
        TensorDynLen::maxabs(self)
    }

    fn permuteinds(&self, new_order: &[DynIndex]) -> Result<Self> {
        // Delegate to the inherent method
        Ok(TensorDynLen::permute_indices(self, new_order))
    }

    fn contract(tensors: &[&Self], allowed: crate::AllowedPairs<'_>) -> Result<Self> {
        // Delegate to contract_multi which handles disconnected components
        super::contract::contract_multi(tensors, allowed)
    }

    fn contract_connected(tensors: &[&Self], allowed: crate::AllowedPairs<'_>) -> Result<Self> {
        // Delegate to contract_connected which requires connected graph
        super::contract::contract_connected(tensors, allowed)
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

        TensorDynLen::from_dense(vec![input_index.clone(), output_index.clone()], data)
    }

    fn scalar_one() -> Result<Self> {
        TensorDynLen::from_dense(vec![], vec![1.0_f64])
    }

    fn ones(indices: &[DynIndex]) -> Result<Self> {
        if indices.is_empty() {
            return Self::scalar_one();
        }
        let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();
        let total_size = checked_total_size(&dims)?;
        TensorDynLen::from_dense(indices.to_vec(), vec![1.0_f64; total_size])
    }

    fn onehot(index_vals: &[(DynIndex, usize)]) -> Result<Self> {
        if index_vals.is_empty() {
            return Self::scalar_one();
        }
        let indices: Vec<DynIndex> = index_vals.iter().map(|(idx, _)| idx.clone()).collect();
        let vals: Vec<usize> = index_vals.iter().map(|(_, v)| *v).collect();
        let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();

        for (k, (&v, &d)) in vals.iter().zip(dims.iter()).enumerate() {
            if v >= d {
                return Err(anyhow::anyhow!(
                    "onehot: value {} at position {} is >= dimension {}",
                    v,
                    k,
                    d
                ));
            }
        }

        let total_size = checked_total_size(&dims)?;
        let mut data = vec![0.0_f64; total_size];

        let offset = column_major_offset(&dims, &vals)?;
        data[offset] = 1.0;

        Self::from_dense(indices, data)
    }

    // delta() uses the default implementation via diagonal() and outer_product()
}

fn checked_total_size(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1_usize, |acc, &d| {
        if d == 0 {
            return Err(anyhow::anyhow!("invalid dimension 0"));
        }
        acc.checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("tensor size overflow"))
    })
}

fn column_major_offset(dims: &[usize], vals: &[usize]) -> Result<usize> {
    if dims.len() != vals.len() {
        return Err(anyhow::anyhow!(
            "column_major_offset: dims.len() != vals.len()"
        ));
    }
    checked_total_size(dims)?;

    let mut offset = 0usize;
    let mut stride = 1usize;
    for (k, (&v, &d)) in vals.iter().zip(dims.iter()).enumerate() {
        if d == 0 {
            return Err(anyhow::anyhow!("invalid dimension 0 at position {}", k));
        }
        if v >= d {
            return Err(anyhow::anyhow!(
                "column_major_offset: value {} at position {} is >= dimension {}",
                v,
                k,
                d
            ));
        }
        let term = v
            .checked_mul(stride)
            .ok_or_else(|| anyhow::anyhow!("column_major_offset: overflow"))?;
        offset = offset
            .checked_add(term)
            .ok_or_else(|| anyhow::anyhow!("column_major_offset: overflow"))?;
        stride = stride
            .checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("column_major_offset: overflow"))?;
    }
    Ok(offset)
}

// ============================================================================
// High-level API for tensor construction (avoids direct Storage access)
// ============================================================================

impl TensorDynLen {
    fn any_scalar_payload_to_complex(data: Vec<AnyScalar>) -> Vec<Complex64> {
        data.into_iter()
            .map(|value| {
                value
                    .as_c64()
                    .unwrap_or_else(|| Complex64::new(value.real(), 0.0))
            })
            .collect()
    }

    fn any_scalar_payload_to_real(data: Vec<AnyScalar>) -> Vec<f64> {
        data.into_iter().map(|value| value.real()).collect()
    }

    fn validate_dense_payload_len(data_len: usize, dims: &[usize]) -> Result<()> {
        let expected_len = checked_total_size(dims)?;
        anyhow::ensure!(
            data_len == expected_len,
            "dense payload length {} does not match dims {:?} (expected {})",
            data_len,
            dims,
            expected_len
        );
        Ok(())
    }

    fn validate_diag_payload_len(data_len: usize, dims: &[usize]) -> Result<()> {
        anyhow::ensure!(
            !dims.is_empty(),
            "diagonal tensor construction requires at least one index"
        );
        Self::validate_diag_dims(dims)?;
        anyhow::ensure!(
            data_len == dims[0],
            "diagonal payload length {} does not match diagonal dimension {}",
            data_len,
            dims[0]
        );
        Ok(())
    }

    /// Create a tensor from dense data with explicit indices.
    ///
    /// This is the recommended high-level API for creating tensors from raw data.
    /// It avoids direct access to `Storage` internals.
    ///
    /// # Type Parameters
    /// * `T` - Scalar type (`f64` or `Complex64`)
    ///
    /// # Arguments
    /// * `indices` - Vector of indices for the tensor
    /// * `data` - Tensor data in column-major order
    ///
    /// # Panics
    /// Panics if data length doesn't match the product of index dimensions.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor: TensorDynLen = TensorDynLen::from_dense(vec![i, j], data).unwrap();
    /// assert_eq!(tensor.dims(), vec![2, 3]);
    /// ```
    pub fn from_dense<T: TensorElement>(indices: Vec<DynIndex>, data: Vec<T>) -> Result<Self> {
        let dims = Self::expected_dims_from_indices(&indices);
        Self::validate_indices(&indices);
        Self::validate_dense_payload_len(data.len(), &dims)?;
        let native = dense_native_tensor_from_col_major(&data, &dims)?;
        Self::from_native(indices, native)
    }

    /// Create a tensor from dense payload data provided as [`AnyScalar`] values.
    ///
    /// This is the preferred public API when the caller only knows the scalar
    /// type at runtime.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{AnyScalar, TensorDynLen};
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense_any(
    ///     vec![i, j],
    ///     vec![
    ///         AnyScalar::new_real(1.0),
    ///         AnyScalar::new_complex(0.0, 1.0),
    ///         AnyScalar::new_real(2.0),
    ///         AnyScalar::new_real(3.0),
    ///     ],
    /// ).unwrap();
    ///
    /// assert!(tensor.is_complex());
    /// assert_eq!(tensor.dims(), vec![2, 2]);
    /// ```
    pub fn from_dense_any(indices: Vec<DynIndex>, data: Vec<AnyScalar>) -> Result<Self> {
        if data.iter().any(AnyScalar::is_complex) {
            Self::from_dense(indices, Self::any_scalar_payload_to_complex(data))
        } else {
            Self::from_dense(indices, Self::any_scalar_payload_to_real(data))
        }
    }

    /// Create a diagonal tensor from diagonal payload data with explicit indices.
    ///
    /// All indices must have the same dimension, and `data.len()` must equal
    /// that dimension. The resulting tensor has nonzero entries only on
    /// the multi-index diagonal (`T[i,i,...,i] = data[i]`).
    ///
    /// The public native bridge currently materializes diagonal payloads densely, so
    /// the returned tensor is mathematically diagonal but may not report
    /// [`TensorDynLen::is_diag`] at the native-storage level.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(3);
    /// let j = DynIndex::new_dyn(3);
    /// let diag = TensorDynLen::from_diag(vec![i, j], vec![1.0, 2.0, 3.0]).unwrap();
    ///
    /// let data = diag.to_vec::<f64>().unwrap();
    /// // 3x3 identity-like: [1,0,0, 0,2,0, 0,0,3] in column-major
    /// assert!((data[0] - 1.0).abs() < 1e-12);
    /// assert!((data[4] - 2.0).abs() < 1e-12);
    /// assert!((data[8] - 3.0).abs() < 1e-12);
    /// assert!((data[1]).abs() < 1e-12);  // off-diagonal is zero
    /// ```
    pub fn from_diag<T: TensorElement>(indices: Vec<DynIndex>, data: Vec<T>) -> Result<Self> {
        let dims = Self::expected_dims_from_indices(&indices);
        Self::validate_indices(&indices);
        Self::validate_diag_payload_len(data.len(), &dims)?;
        let native = diag_native_tensor_from_col_major(&data, dims.len())?;
        Self::from_native_with_axis_classes(indices, native, Self::diag_axis_classes(dims.len()))
    }

    /// Create a diagonal tensor from diagonal payload data provided as
    /// [`AnyScalar`] values.
    ///
    /// This is the preferred public API when the caller only knows the scalar
    /// type at runtime.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{AnyScalar, TensorDynLen};
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(2);
    /// let tensor = TensorDynLen::from_diag_any(
    ///     vec![i, j],
    ///     vec![AnyScalar::new_real(1.0), AnyScalar::new_complex(2.0, -1.0)],
    /// ).unwrap();
    ///
    /// assert!(tensor.is_complex());
    /// assert_eq!(tensor.dims(), vec![2, 2]);
    /// ```
    pub fn from_diag_any(indices: Vec<DynIndex>, data: Vec<AnyScalar>) -> Result<Self> {
        if data.iter().any(AnyScalar::is_complex) {
            Self::from_diag(indices, Self::any_scalar_payload_to_complex(data))
        } else {
            Self::from_diag(indices, Self::any_scalar_payload_to_real(data))
        }
    }

    /// Create a copy tensor whose nonzero entries are `value` on the diagonal.
    ///
    /// For indices `[i, j, k]`, the returned tensor satisfies
    /// `T[i, j, k] = value` when `i = j = k`, and zero otherwise.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{AnyScalar, TensorDynLen};
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(2);
    /// let k = Index::new_dyn(2);
    /// let tensor = TensorDynLen::copy_tensor(
    ///     vec![i, j, k],
    ///     AnyScalar::new_real(1.0),
    /// ).unwrap();
    ///
    /// assert_eq!(tensor.dims(), vec![2, 2, 2]);
    /// ```
    pub fn copy_tensor(indices: Vec<DynIndex>, value: AnyScalar) -> Result<Self> {
        if indices.is_empty() {
            return Self::from_dense_any(vec![], vec![value]);
        }
        let dim = indices[0].dim();
        let data = vec![value; dim];
        Self::from_diag_any(indices, data)
    }

    /// Replace one fused index with multiple indices using an exact reshape.
    ///
    /// The caller must specify how the old fused index should be decoded into
    /// the new indices via `order`.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, LinearizationOrder, TensorDynLen, TensorLike};
    ///
    /// let fused = DynIndex::new_dyn(4);
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![fused.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    ///
    /// let unfused = tensor
    ///     .unfuse_index(&fused, &[i.clone(), j.clone()], LinearizationOrder::ColumnMajor)
    ///     .unwrap();
    ///
    /// let expected = TensorDynLen::from_dense(vec![i, j], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// assert!(unfused.isapprox(&expected, 1e-12, 0.0));
    /// ```
    pub fn unfuse_index(
        &self,
        old_index: &DynIndex,
        new_indices: &[DynIndex],
        order: LinearizationOrder,
    ) -> Result<Self> {
        anyhow::ensure!(
            !new_indices.is_empty(),
            "unfuse_index requires at least one replacement index"
        );

        let axis = self
            .indices
            .iter()
            .position(|idx| idx.id() == old_index.id())
            .ok_or_else(|| anyhow::anyhow!("index {:?} not found in tensor", old_index.id()))?;

        let replacement_dims: Vec<usize> = new_indices.iter().map(DynIndex::dim).collect();
        let replacement_product = checked_product(&replacement_dims)?;
        anyhow::ensure!(
            replacement_product == old_index.dim(),
            "product of new index dimensions must match the replaced index dimension"
        );

        let mut result_indices =
            Vec::with_capacity(self.indices.len() - 1usize + new_indices.len());
        result_indices.extend_from_slice(&self.indices[..axis]);
        result_indices.extend(new_indices.iter().cloned());
        result_indices.extend_from_slice(&self.indices[axis + 1..]);
        Self::validate_indices(&result_indices);

        let old_dims = self.dims();
        let mut new_dims = Vec::with_capacity(old_dims.len() - 1usize + replacement_dims.len());
        new_dims.extend_from_slice(&old_dims[..axis]);
        new_dims.extend_from_slice(&replacement_dims);
        new_dims.extend_from_slice(&old_dims[axis + 1..]);

        let old_data = self.to_vec_any()?;
        let mut new_data = vec![AnyScalar::new_real(0.0); old_data.len()];
        for (old_linear, value) in old_data.into_iter().enumerate() {
            let old_multi = decode_col_major_linear(old_linear, &old_dims)?;
            let split_multi = decode_linear_with_order(old_multi[axis], &replacement_dims, order)?;
            let mut new_multi = Vec::with_capacity(new_dims.len());
            new_multi.extend_from_slice(&old_multi[..axis]);
            new_multi.extend_from_slice(&split_multi);
            new_multi.extend_from_slice(&old_multi[axis + 1..]);
            let new_linear = encode_col_major_linear(&new_multi, &new_dims)?;
            new_data[new_linear] = value;
        }

        Self::from_dense_any(result_indices, new_data)
    }

    /// Create a scalar (0-dimensional) tensor from a supported element value.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    ///
    /// let scalar = TensorDynLen::scalar(42.0).unwrap();
    /// assert_eq!(scalar.dims(), Vec::<usize>::new());
    /// assert_eq!(scalar.only().real(), 42.0);
    /// ```
    pub fn scalar<T: TensorElement>(value: T) -> Result<Self> {
        Self::from_dense(vec![], vec![value])
    }

    /// Create a tensor filled with zeros of a supported element type.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let tensor = TensorDynLen::zeros::<f64>(vec![i, j]).unwrap();
    /// assert_eq!(tensor.dims(), vec![2, 3]);
    /// ```
    pub fn zeros<T: TensorElement + Zero + Clone>(indices: Vec<DynIndex>) -> Result<Self> {
        let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
        let size: usize = dims.iter().product();
        Self::from_dense(indices, vec![T::zero(); size])
    }
}

// ============================================================================
// High-level API for data extraction (avoids direct .storage() access)
// ============================================================================

impl TensorDynLen {
    /// Extract tensor data as a column-major `Vec<T>`.
    ///
    /// # Type Parameters
    /// * `T` - The scalar element type (`f64` or `Complex64`).
    ///
    /// # Returns
    /// A vector of the tensor data in column-major order.
    ///
    /// # Errors
    /// Returns an error if the tensor's scalar type does not match `T`.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0]).unwrap();
    /// let data = tensor.to_vec::<f64>().unwrap();
    /// assert_eq!(data, &[1.0, 2.0]);
    /// ```
    pub fn to_vec<T: TensorElement>(&self) -> Result<Vec<T>> {
        native_tensor_primal_to_dense_col_major(self.as_native())
    }

    fn to_vec_any(&self) -> Result<Vec<AnyScalar>> {
        if self.is_complex() {
            self.to_vec::<Complex64>().map(|data| {
                data.into_iter()
                    .map(|value| AnyScalar::new_complex(value.re, value.im))
                    .collect()
            })
        } else {
            self.to_vec::<f64>()
                .map(|data| data.into_iter().map(AnyScalar::new_real).collect())
        }
    }

    /// Extract tensor data as a column-major `Vec<f64>`.
    ///
    /// Prefer the generic [`to_vec::<f64>()`](Self::to_vec) method.
    /// This wrapper is kept for C API compatibility.
    pub fn as_slice_f64(&self) -> Result<Vec<f64>> {
        self.to_vec::<f64>()
    }

    /// Extract tensor data as a column-major `Vec<Complex64>`.
    ///
    /// Prefer the generic [`to_vec::<Complex64>()`](Self::to_vec) method.
    /// This wrapper is kept for C API compatibility.
    pub fn as_slice_c64(&self) -> Result<Vec<Complex64>> {
        self.to_vec::<Complex64>()
    }

    /// Check if the tensor has f64 storage.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    ///
    /// let i = Index::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0]).unwrap();
    /// assert!(tensor.is_f64());
    /// assert!(!tensor.is_complex());
    /// ```
    pub fn is_f64(&self) -> bool {
        self.storage.is_f64()
    }

    /// Check whether the tensor carries diagonal logical axis metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, Storage, TensorDynLen};
    ///
    /// // Tensors from `from_dense` use dense storage
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let dense = TensorDynLen::from_dense(vec![i, j], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    /// assert!(!dense.is_diag());
    ///
    /// // Diagonal metadata is preserved when constructing from diagonal storage.
    /// let k = DynIndex::new_dyn(2);
    /// let l = DynIndex::new_dyn(2);
    /// let diag = TensorDynLen::from_storage(
    ///     vec![k, l],
    ///     Storage::from_diag_col_major(vec![1.0, 2.0], 2)
    ///         .map(std::sync::Arc::new)
    ///         .unwrap(),
    /// )
    /// .unwrap();
    /// assert!(diag.is_diag());
    /// ```
    pub fn is_diag(&self) -> bool {
        self.storage.is_diag()
    }

    /// Check if the tensor has complex storage (C64).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use num_complex::Complex64;
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let real_t = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    /// assert!(!real_t.is_complex());
    ///
    /// let complex_t = TensorDynLen::from_dense(
    ///     vec![i],
    ///     vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
    /// ).unwrap();
    /// assert!(complex_t.is_complex());
    /// ```
    pub fn is_complex(&self) -> bool {
        self.storage.is_complex()
    }
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("dimension product overflow"))
    })
}

fn decode_col_major_linear(linear: usize, dims: &[usize]) -> Result<Vec<usize>> {
    let total = checked_product(dims)?;
    anyhow::ensure!(
        linear < total,
        "linear offset {} out of bounds for dims {:?}",
        linear,
        dims
    );
    let mut remaining = linear;
    let mut out = Vec::with_capacity(dims.len());
    for &dim in dims {
        out.push(remaining % dim);
        remaining /= dim;
    }
    Ok(out)
}

fn encode_col_major_linear(indices: &[usize], dims: &[usize]) -> Result<usize> {
    anyhow::ensure!(
        indices.len() == dims.len(),
        "index rank {} does not match dims {:?}",
        indices.len(),
        dims
    );
    let mut linear = 0usize;
    let mut stride = 1usize;
    for (&index, &dim) in indices.iter().zip(dims.iter()) {
        anyhow::ensure!(
            index < dim,
            "index {} out of bounds for dimension {}",
            index,
            dim
        );
        linear += index * stride;
        stride = stride
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("stride overflow"))?;
    }
    Ok(linear)
}

fn decode_linear_with_order(
    linear: usize,
    dims: &[usize],
    order: LinearizationOrder,
) -> Result<Vec<usize>> {
    let total = checked_product(dims)?;
    anyhow::ensure!(
        linear < total,
        "linear offset {} out of bounds for dims {:?}",
        linear,
        dims
    );

    let mut remaining = linear;
    let mut out = vec![0usize; dims.len()];
    match order {
        LinearizationOrder::ColumnMajor => {
            for (slot, &dim) in out.iter_mut().zip(dims.iter()) {
                *slot = remaining % dim;
                remaining /= dim;
            }
        }
        LinearizationOrder::RowMajor => {
            for (slot, &dim) in out.iter_mut().rev().zip(dims.iter().rev()) {
                *slot = remaining % dim;
                remaining /= dim;
            }
        }
    }
    Ok(out)
}
