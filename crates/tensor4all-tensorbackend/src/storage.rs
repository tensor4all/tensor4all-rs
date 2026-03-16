use mdarray::{DynRank, Shape, Tensor};
use num_complex::{Complex64, ComplexFloat};
use num_traits::{One, Zero};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Deref, DerefMut, Mul};
use std::sync::Arc;

use crate::layout::storage_strides;
use crate::tensor_element::TensorElement;

/// Trait for scalar types that can be used in dense storage.
///
/// This trait defines the requirements for types that can be stored in `DenseStorage<T>`.
pub trait DenseScalar:
    Clone
    + Copy
    + Debug
    + Default
    + Zero
    + One
    + Add<Output = Self>
    + Mul<Output = Self>
    + AddAssign
    + ComplexFloat
    + Send
    + Sync
    + 'static
{
}

impl DenseScalar for f64 {}
impl DenseScalar for Complex64 {}

/// Dense storage for tensor elements, wrapping mdarray's Tensor with dynamic rank.
///
/// This type provides shape-aware storage using `Tensor<T, DynRank>` internally.
/// Shape information is stored within the tensor, eliminating the need to pass
/// dimensions separately to operations like `permute` and `contract`.
#[derive(Debug, Clone)]
pub struct DenseStorage<T>(Tensor<T, DynRank>);

impl<T> DenseStorage<T> {
    /// Create a new DenseStorage from a Vec with explicit shape.
    ///
    /// # Panics
    /// Panics if the product of dims doesn't match vec.len().
    pub fn from_vec_with_shape(vec: Vec<T>, dims: &[usize]) -> Self {
        let expected_len: usize = dims.iter().product();
        assert_eq!(
            vec.len(),
            expected_len,
            "Vec length {} doesn't match shape {:?} (product {})",
            vec.len(),
            dims,
            expected_len
        );
        let tensor = Tensor::from(vec).into_shape(DynRank::from_dims(dims));
        Self(tensor)
    }

    /// Create a scalar (0-dimensional) storage from a single value.
    pub fn from_scalar(val: T) -> Self {
        let tensor = Tensor::from(vec![val]).into_shape(DynRank::from_dims(&[]));
        Self(tensor)
    }

    /// Get the shape (dimensions) of the storage.
    pub fn dims(&self) -> Vec<usize> {
        self.0.shape().with_dims(|d| d.to_vec())
    }

    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Get underlying data as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.0[..]
    }

    /// Get underlying data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0[..]
    }

    /// Convert to Vec, consuming the storage.
    pub fn into_vec(self) -> Vec<T> {
        self.0.into_vec()
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get a reference to the underlying tensor.
    pub fn tensor(&self) -> &Tensor<T, DynRank> {
        &self.0
    }

    /// Get a mutable reference to the underlying tensor.
    pub fn tensor_mut(&mut self) -> &mut Tensor<T, DynRank> {
        &mut self.0
    }

    /// Consume and return the underlying tensor.
    pub fn into_tensor(self) -> Tensor<T, DynRank> {
        self.0
    }

    /// Create from an existing tensor.
    pub fn from_tensor(tensor: Tensor<T, DynRank>) -> Self {
        Self(tensor)
    }

    /// Iterate over elements.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }
}

impl<T: Clone> DenseStorage<T> {
    /// Get element at linear index.
    pub fn get(&self, i: usize) -> T {
        self.0[i].clone()
    }
}

impl<T: Copy> DenseStorage<T> {
    /// Set element at linear index.
    pub fn set(&mut self, i: usize, val: T) {
        self.0[i] = val;
    }
}

impl<T> Deref for DenseStorage<T> {
    type Target = Tensor<T, DynRank>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for DenseStorage<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: DenseScalar> DenseStorage<T> {
    /// Permute the dense storage data according to the given permutation.
    ///
    /// Uses the internal shape information - no external dims parameter needed.
    pub fn permute(&self, perm: &[usize]) -> Self {
        assert_eq!(
            perm.len(),
            self.rank(),
            "permutation length {} must match rank {}",
            perm.len(),
            self.rank()
        );

        // Use mdarray's permute functionality directly
        let permuted = self.0.permute(perm).to_tensor();
        Self(permuted)
    }

    /// Contract this dense storage with another dense storage.
    ///
    /// This method handles non-contiguous contracted axes by permuting the tensors
    /// to make the contracted axes contiguous before GEMM-style contraction.
    ///
    /// Uses internal shape information - no external dims parameters needed.
    pub fn contract(&self, axes: &[usize], other: &Self, other_axes: &[usize]) -> Self {
        let dims = self.dims();
        let other_dims = other.dims();

        // For self: move contracted axes to end.
        // For other: move contracted axes to front.
        let (perm_self, new_axes_self, new_dims_self) =
            compute_contraction_permutation(&dims, axes, false);
        let (perm_other, new_axes_other, new_dims_other) =
            compute_contraction_permutation(&other_dims, other_axes, true);

        let storage_self = if perm_self.iter().enumerate().all(|(i, &p)| i == p) {
            Cow::Borrowed(self)
        } else {
            Cow::Owned(self.permute(&perm_self))
        };
        let storage_other = if perm_other.iter().enumerate().all(|(i, &p)| i == p) {
            Cow::Borrowed(other)
        } else {
            Cow::Owned(other.permute(&perm_other))
        };

        let result_vec = contract_via_gemm(
            storage_self.as_slice(),
            &new_dims_self,
            &new_axes_self,
            storage_other.as_slice(),
            &new_dims_other,
            &new_axes_other,
        );

        let result_dims = compute_result_dims(&dims, axes, &other_dims, other_axes);
        Self::from_vec_with_shape(result_vec, &result_dims)
    }
}

// Random generation for f64
impl DenseStorage<f64> {
    /// Create storage with random values from standard normal distribution.
    ///
    /// Creates a 1D storage with the given size.
    pub fn random_1d<R: Rng>(rng: &mut R, size: usize) -> Self {
        let data: Vec<f64> = (0..size).map(|_| StandardNormal.sample(rng)).collect();
        Self::from_vec_with_shape(data, &[size])
    }

    /// Create storage with random values with explicit shape.
    pub fn random<R: Rng>(rng: &mut R, dims: &[usize]) -> Self {
        let size: usize = dims.iter().product();
        let data: Vec<f64> = (0..size).map(|_| StandardNormal.sample(rng)).collect();
        Self::from_vec_with_shape(data, dims)
    }
}

// Random generation for Complex64
impl DenseStorage<Complex64> {
    /// Create storage with random complex values (re, im both from standard normal).
    ///
    /// Creates a 1D storage with the given size.
    pub fn random_1d<R: Rng>(rng: &mut R, size: usize) -> Self {
        let data: Vec<Complex64> = (0..size)
            .map(|_| Complex64::new(StandardNormal.sample(rng), StandardNormal.sample(rng)))
            .collect();
        Self::from_vec_with_shape(data, &[size])
    }

    /// Create storage with random complex values with explicit shape.
    pub fn random<R: Rng>(rng: &mut R, dims: &[usize]) -> Self {
        let size: usize = dims.iter().product();
        let data: Vec<Complex64> = (0..size)
            .map(|_| Complex64::new(StandardNormal.sample(rng), StandardNormal.sample(rng)))
            .collect();
        Self::from_vec_with_shape(data, dims)
    }
}

/// Type alias for f64 dense storage (for backward compatibility).
pub type DenseStorageF64 = DenseStorage<f64>;

/// Type alias for Complex64 dense storage (for backward compatibility).
pub type DenseStorageC64 = DenseStorage<Complex64>;

/// Compute the result dimensions after contraction.
fn compute_result_dims(
    dims_a: &[usize],
    axes_a: &[usize],
    dims_b: &[usize],
    axes_b: &[usize],
) -> Vec<usize> {
    let mut result_dims = Vec::new();
    for (axis, &dim) in dims_a.iter().enumerate() {
        if !axes_a.contains(&axis) {
            result_dims.push(dim);
        }
    }
    for (axis, &dim) in dims_b.iter().enumerate() {
        if !axes_b.contains(&axis) {
            result_dims.push(dim);
        }
    }
    result_dims
}

/// Contract two tensors via GEMM-style matrix multiplication.
///
/// This function assumes contracted axes are already contiguous:
/// - For `a`: contracted axes are at the end.
/// - For `b`: contracted axes are at the front.
fn contract_via_gemm<T: DenseScalar>(
    a: &[T],
    dims_a: &[usize],
    axes_a: &[usize],
    b: &[T],
    dims_b: &[usize],
    axes_b: &[usize],
) -> Vec<T> {
    let naxes = axes_a.len();
    assert_eq!(naxes, axes_b.len(), "Number of contracted axes must match");

    let ndim_a = dims_a.len();

    let m: usize = dims_a.iter().take(ndim_a - naxes).product::<usize>().max(1);
    let k: usize = dims_a.iter().skip(ndim_a - naxes).product::<usize>().max(1);
    let n: usize = dims_b.iter().skip(naxes).product::<usize>().max(1);

    let k_b: usize = dims_b.iter().take(naxes).product::<usize>().max(1);
    assert_eq!(
        k, k_b,
        "Contracted dimension sizes must match: {} vs {}",
        k, k_b
    );

    let mut c = vec![T::zero(); m * n];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_il * b[l * n + j];
            }
        }
    }
    c
}

/// Compute permutation to make contracted axes contiguous.
///
/// If `axes_at_front` is true, contracted axes are moved to the front.
/// Otherwise, contracted axes are moved to the end.
///
/// Returns `(permutation, new_axes, new_dims)`.
fn compute_contraction_permutation(
    dims: &[usize],
    axes: &[usize],
    axes_at_front: bool,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let ndim = dims.len();
    let naxes = axes.len();
    let non_contracted: Vec<usize> = (0..ndim).filter(|i| !axes.contains(i)).collect();

    let perm: Vec<usize> = if axes_at_front {
        axes.iter().chain(non_contracted.iter()).copied().collect()
    } else {
        non_contracted.iter().chain(axes.iter()).copied().collect()
    };
    let new_dims: Vec<usize> = perm.iter().map(|&i| dims[i]).collect();
    let new_axes: Vec<usize> = if axes_at_front {
        (0..naxes).collect()
    } else {
        (ndim - naxes..ndim).collect()
    };
    (perm, new_axes, new_dims)
}

/// Diagonal storage for tensor elements, generic over scalar type.
#[derive(Debug, Clone)]
pub struct DiagStorage<T>(Vec<T>);

impl<T> DiagStorage<T> {
    /// Create a new diagonal storage from a vector of diagonal elements.
    pub fn from_vec(vec: Vec<T>) -> Self {
        Self(vec)
    }

    /// Get a slice of the diagonal elements.
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Get a mutable slice of the diagonal elements.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    /// Consume the storage and return the underlying vector.
    pub fn into_vec(self) -> Vec<T> {
        self.0
    }

    /// Return the number of diagonal elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return true if the storage has no elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<T: Clone> DiagStorage<T> {
    /// Get a clone of the diagonal element at index `i`.
    pub fn get(&self, i: usize) -> T {
        self.0[i].clone()
    }
}

impl<T: Copy> DiagStorage<T> {
    /// Set the diagonal element at index `i` to `val`.
    pub fn set(&mut self, i: usize, val: T) {
        self.0[i] = val;
    }
}

impl<T: DenseScalar> DiagStorage<T> {
    /// Convert diagonal storage to a dense vector representation.
    /// Creates a dense vector with diagonal elements set and off-diagonal elements as zero.
    pub fn to_dense_vec(&self, dims: &[usize]) -> Vec<T> {
        let total_size: usize = dims.iter().product();
        let mut dense_vec = vec![T::zero(); total_size];
        let mindim_val = mindim(dims);

        // Set diagonal elements
        // For a tensor with indices [i, j, k, ...] where all have dimension d,
        // the diagonal elements are at positions where i == j == k == ...
        // The linear index for position (i, i, i, ...) is computed as:
        // i * (1 + stride_1 + stride_2 + ...)
        // For a tensor with dimensions [d, d, d, ...], the stride for dimension k is d^(rank - k - 1)
        let rank = dims.len();
        if rank == 0 {
            return vec![self.0[0]];
        }

        // Compute stride for each dimension
        let mut strides = vec![1; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        // Total stride for diagonal: sum of all strides
        let diag_stride: usize = strides.iter().sum();

        // Set diagonal elements
        for i in 0..mindim_val.min(self.0.len()) {
            let linear_idx = i * diag_stride;
            if linear_idx < total_size {
                dense_vec[linear_idx] = self.0[i];
            }
        }

        dense_vec
    }

    /// Contract this diagonal storage with another diagonal storage of the same type.
    /// Returns either a scalar (Dense with one element) or a diagonal storage.
    pub fn contract_diag_diag(
        &self,
        dims: &[usize],
        other: &Self,
        other_dims: &[usize],
        result_dims: &[usize],
        make_dense: impl FnOnce(Vec<T>) -> Storage,
        make_diag: impl FnOnce(Vec<T>) -> Storage,
    ) -> Storage {
        let mindim_a = mindim(dims);
        let mindim_b = mindim(other_dims);
        let min_len = mindim_a.min(mindim_b).min(self.0.len()).min(other.0.len());

        if result_dims.is_empty() {
            // All indices contracted: compute inner product (scalar result)
            let scalar: T = (0..min_len).fold(T::zero(), |acc, i| acc + self.0[i] * other.0[i]);
            make_dense(vec![scalar])
        } else {
            // Some indices remain: element-wise product (DiagTensor result)
            let result_diag: Vec<T> = (0..min_len).map(|i| self.0[i] * other.0[i]).collect();
            make_diag(result_diag)
        }
    }

    /// Contract this diagonal storage with a dense storage.
    ///
    /// # Arguments
    /// * `diag_dims` - Dimensions of the diagonal tensor (all must be equal)
    /// * `axes_diag` - Axes of the diagonal tensor to contract
    /// * `dense` - The dense storage to contract with
    /// * `dense_dims` - Dimensions of the dense tensor
    /// * `axes_dense` - Axes of the dense tensor to contract (must match axes_diag in length)
    /// * `result_dims` - Dimensions of the result tensor
    ///
    /// # Returns
    /// The contracted storage (always Dense, since Diag structure is generally lost)
    #[allow(clippy::too_many_arguments)]
    pub fn contract_diag_dense(
        &self,
        diag_dims: &[usize],
        axes_diag: &[usize],
        dense: &DenseStorage<T>,
        dense_dims: &[usize],
        axes_dense: &[usize],
        result_dims: &[usize],
        make_storage: impl FnOnce(Vec<T>) -> Storage,
    ) -> Storage {
        contract_diag_dense_impl(
            self.as_slice(),
            diag_dims,
            axes_diag,
            dense.as_slice(),
            dense_dims,
            axes_dense,
            result_dims,
            make_storage,
        )
    }
}

/// Type alias for f64 diagonal storage (for backward compatibility).
pub type DiagStorageF64 = DiagStorage<f64>;

/// Type alias for Complex64 diagonal storage (for backward compatibility).
pub type DiagStorageC64 = DiagStorage<Complex64>;

/// Generic implementation of Diag × Dense contraction.
///
/// This function exploits the diagonal structure: for a diagonal tensor,
/// all indices have the same value t (0 <= t < d). When contracting with
/// a dense tensor, we only need to access the "diagonal slices" of the dense tensor.
///
/// Algorithm:
/// 1. For each diagonal index t = 0..d:
///    - Get diag[t]
///    - Extract the slice of dense where all axes_dense have value t
///    - Multiply and accumulate into result
#[allow(clippy::too_many_arguments)]
fn contract_diag_dense_impl<T: DenseScalar>(
    diag: &[T],
    diag_dims: &[usize],
    axes_diag: &[usize],
    dense: &[T],
    dense_dims: &[usize],
    axes_dense: &[usize],
    result_dims: &[usize],
    make_storage: impl FnOnce(Vec<T>) -> Storage,
) -> Storage {
    let diag_rank = diag_dims.len();
    let dense_rank = dense_dims.len();
    let _num_contracted = axes_diag.len();

    // The diagonal dimension (all diag_dims should be equal)
    let d = if diag_dims.is_empty() {
        1
    } else {
        diag_dims[0]
    };

    // Compute the non-contracted axes for the result
    let diag_non_contracted: Vec<usize> =
        (0..diag_rank).filter(|i| !axes_diag.contains(i)).collect();
    let dense_non_contracted: Vec<usize> = (0..dense_rank)
        .filter(|i| !axes_dense.contains(i))
        .collect();

    // Result size
    let result_size: usize = result_dims.iter().product();
    let result_size = if result_size == 0 { 1 } else { result_size };

    // Compute strides for dense tensor (row-major)
    let dense_axis_strides = storage_strides(dense_dims);

    // Compute the size of the non-contracted part of dense
    let dense_non_contracted_dims: Vec<usize> = dense_non_contracted
        .iter()
        .map(|&i| dense_dims[i])
        .collect();
    let dense_slice_size: usize = dense_non_contracted_dims.iter().product();
    let dense_slice_size = if dense_slice_size == 0 {
        1
    } else {
        dense_slice_size
    };

    // Compute strides for the non-contracted dense dimensions
    let dense_non_contracted_strides = storage_strides(&dense_non_contracted_dims);

    // If all diag axes are contracted, result has shape = dense_non_contracted_dims
    // If some diag axes remain, result has shape = [d, d, ...] (diag non-contracted) + dense_non_contracted_dims
    let diag_non_contracted_count = diag_non_contracted.len();

    if diag_non_contracted_count == 0 {
        // All diagonal indices contracted: result is purely from dense's non-contracted part
        // For each t, we accumulate diag[t] * dense_slice[t] into result
        let mut result = vec![T::zero(); result_size];

        for (t, &diag_val) in diag.iter().enumerate().take(d.min(diag.len())) {
            // Compute base offset in dense for this t (all contracted axes = t)
            let base_offset: usize = axes_dense
                .iter()
                .map(|&axis| t * dense_axis_strides[axis])
                .sum();

            // Iterate over all non-contracted positions in dense
            for (flat_idx, result_item) in result.iter_mut().enumerate().take(dense_slice_size) {
                // Convert flat_idx to multi-index for non-contracted dims
                let mut offset = base_offset;
                let mut remaining = flat_idx;
                for (local_axis, &global_axis) in dense_non_contracted.iter().enumerate() {
                    let idx = remaining / dense_non_contracted_strides[local_axis];
                    remaining %= dense_non_contracted_strides[local_axis];
                    offset += idx * dense_axis_strides[global_axis];
                }

                *result_item += diag_val * dense[offset];
            }
        }

        make_storage(result)
    } else {
        // Some diagonal indices remain: result has diagonal structure in those indices
        // Result shape: [d, d, ...] (diag_non_contracted_count times) + dense_non_contracted_dims
        // But since the diagonal indices must all equal, only the diagonal elements are non-zero
        // Result is effectively: for each t, result[t,t,...,dense_indices] = diag[t] * dense_slice

        // Compute result strides
        let result_strides = storage_strides(result_dims);

        let mut result = vec![T::zero(); result_size];

        for (t, &diag_val) in diag.iter().enumerate().take(d.min(diag.len())) {
            // Compute base offset in dense for this t
            let base_offset_dense: usize = axes_dense
                .iter()
                .map(|&axis| t * dense_axis_strides[axis])
                .sum();

            // Compute base offset in result for diagonal position (t, t, ...)
            let base_offset_result: usize = (0..diag_non_contracted_count)
                .map(|i| t * result_strides[i])
                .sum();

            // Iterate over all non-contracted positions in dense
            #[allow(clippy::needless_range_loop)]
            for flat_idx in 0..dense_slice_size {
                // Convert flat_idx to multi-index for non-contracted dims
                let mut offset_dense = base_offset_dense;
                let mut offset_result = base_offset_result;
                let mut remaining = flat_idx;

                for (local_axis, &global_axis) in dense_non_contracted.iter().enumerate() {
                    let idx = remaining / dense_non_contracted_strides[local_axis];
                    remaining %= dense_non_contracted_strides[local_axis];
                    offset_dense += idx * dense_axis_strides[global_axis];
                    // In result, these come after the diagonal indices
                    offset_result += idx * result_strides[diag_non_contracted_count + local_axis];
                }

                result[offset_result] = diag_val * dense[offset_dense];
            }
        }

        make_storage(result)
    }
}

/// Helper for Dense × Diag contraction: compute as Diag × Dense and permute result.
///
/// This function handles the case where Dense appears first in the contraction.
/// It computes the contraction using `contract_diag_dense_impl` (which assumes Diag first),
/// then permutes the result to match the expected dimension order.
#[allow(clippy::too_many_arguments)]
fn contract_dense_diag_impl<T: DenseScalar>(
    dense: &DenseStorage<T>,
    dense_dims: &[usize],
    axes_dense: &[usize],
    diag: &DiagStorage<T>,
    diag_dims: &[usize],
    axes_diag: &[usize],
    _result_dims: &[usize],
    make_storage: impl FnOnce(Vec<T>, &[usize]) -> Storage,
    make_permuted: impl FnOnce(Storage, &[usize]) -> Storage,
) -> Storage {
    let diag_rank = diag_dims.len();
    let dense_rank = dense_dims.len();
    let diag_non_contracted_count = diag_rank - axes_diag.len();
    let dense_non_contracted_count = dense_rank - axes_dense.len();

    // Build reordered result_dims: [diag_non_contracted..., dense_non_contracted...]
    let diag_non_contracted_dims: Vec<usize> = (0..diag_rank)
        .filter(|i| !axes_diag.contains(i))
        .map(|i| diag_dims[i])
        .collect();
    let dense_non_contracted_dims: Vec<usize> = (0..dense_rank)
        .filter(|i| !axes_dense.contains(i))
        .map(|i| dense_dims[i])
        .collect();
    let swapped_result_dims: Vec<usize> = diag_non_contracted_dims
        .iter()
        .chain(dense_non_contracted_dims.iter())
        .copied()
        .collect();

    let intermediate = contract_diag_dense_impl(
        diag.as_slice(),
        diag_dims,
        axes_diag,
        dense.as_slice(),
        dense_dims,
        axes_dense,
        &swapped_result_dims,
        |v| make_storage(v, &swapped_result_dims),
    );

    // Permute to get [dense_non_contracted..., diag_non_contracted...]
    if diag_non_contracted_count > 0 && dense_non_contracted_count > 0 {
        // Build permutation: move diag dims to end
        let mut perm: Vec<usize> = (diag_non_contracted_count
            ..(diag_non_contracted_count + dense_non_contracted_count))
            .collect();
        perm.extend(0..diag_non_contracted_count);
        make_permuted(intermediate, &perm)
    } else {
        intermediate
    }
}

/// Storage backend for tensor data.
/// Supports Dense and Diag storage for f64 and Complex64 element types.
#[derive(Debug, Clone)]
pub enum Storage {
    /// Dense storage with f64 elements.
    DenseF64(DenseStorageF64),
    /// Dense storage with Complex64 elements.
    DenseC64(DenseStorageC64),
    /// Diagonal storage with f64 elements.
    DiagF64(DiagStorageF64),
    /// Diagonal storage with Complex64 elements.
    DiagC64(DiagStorageC64),
}

/// Type-driven constructor for `Storage`.
///
/// This enables `<T as DenseStorageFactory>::new_dense(size)` which creates
/// a 1D zero-initialized DenseStorage with the given size.
pub trait DenseStorageFactory {
    /// Create a 1D zero-initialized storage with the given size.
    fn new_dense(size: usize) -> Storage;

    /// Create storage with the given shape, zero-initialized.
    fn new_dense_with_shape(dims: &[usize]) -> Storage;
}

impl DenseStorageFactory for f64 {
    fn new_dense(size: usize) -> Storage {
        Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            vec![0.0; size],
            &[size],
        ))
    }

    fn new_dense_with_shape(dims: &[usize]) -> Storage {
        let size: usize = dims.iter().product();
        Storage::DenseF64(DenseStorageF64::from_vec_with_shape(vec![0.0; size], dims))
    }
}

impl DenseStorageFactory for Complex64 {
    fn new_dense(size: usize) -> Storage {
        Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![Complex64::new(0.0, 0.0); size],
            &[size],
        ))
    }

    fn new_dense_with_shape(dims: &[usize]) -> Storage {
        let size: usize = dims.iter().product();
        Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![Complex64::new(0.0, 0.0); size],
            dims,
        ))
    }
}

/// Types that can be computed as the result of a reduction over `Storage`.
///
/// This lets callers write `let s: T = tensor.sum();` without matching on storage.
pub trait SumFromStorage: Sized {
    /// Compute the sum of all elements in the storage.
    fn sum_from_storage(storage: &Storage) -> Self;
}

impl SumFromStorage for f64 {
    fn sum_from_storage(storage: &Storage) -> Self {
        match storage {
            Storage::DenseF64(v) => v.as_slice().iter().copied().sum(),
            Storage::DenseC64(v) => v.as_slice().iter().map(|z| z.re).sum(),
            Storage::DiagF64(v) => v.as_slice().iter().copied().sum(),
            Storage::DiagC64(v) => v.as_slice().iter().map(|z| z.re).sum(),
        }
    }
}

impl SumFromStorage for Complex64 {
    fn sum_from_storage(storage: &Storage) -> Self {
        match storage {
            Storage::DenseF64(v) => Complex64::new(v.as_slice().iter().copied().sum(), 0.0),
            Storage::DenseC64(v) => v.as_slice().iter().copied().sum(),
            Storage::DiagF64(v) => Complex64::new(v.as_slice().iter().copied().sum(), 0.0),
            Storage::DiagC64(v) => v.as_slice().iter().copied().sum(),
        }
    }
}

// AnyScalar is now in its own module
pub use crate::any_scalar::AnyScalar;

impl Storage {
    /// Create a new 1D zero-initialized DenseF64 storage with the given size.
    pub fn new_dense_f64(size: usize) -> Self {
        Self::DenseF64(DenseStorageF64::from_vec_with_shape(
            vec![0.0; size],
            &[size],
        ))
    }

    /// Create a new 1D zero-initialized DenseC64 storage with the given size.
    pub fn new_dense_c64(size: usize) -> Self {
        Self::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![Complex64::new(0.0, 0.0); size],
            &[size],
        ))
    }

    /// Create a new DiagF64 storage with the given diagonal data.
    pub fn new_diag_f64(diag_data: Vec<f64>) -> Self {
        Self::DiagF64(DiagStorageF64::from_vec(diag_data))
    }

    /// Create a new DiagC64 storage with the given diagonal data.
    pub fn new_diag_c64(diag_data: Vec<Complex64>) -> Self {
        Self::DiagC64(DiagStorageC64::from_vec(diag_data))
    }

    /// Check if this storage is a Diag storage type.
    pub fn is_diag(&self) -> bool {
        matches!(self, Self::DiagF64(_) | Self::DiagC64(_))
    }

    /// Check if this storage uses f64 scalar type.
    pub fn is_f64(&self) -> bool {
        matches!(self, Self::DenseF64(_) | Self::DiagF64(_))
    }

    /// Check if this storage uses Complex64 scalar type.
    pub fn is_c64(&self) -> bool {
        matches!(self, Self::DenseC64(_) | Self::DiagC64(_))
    }

    /// Check if this storage uses complex scalar type.
    ///
    /// This is an alias for `is_c64()`.
    pub fn is_complex(&self) -> bool {
        self.is_c64()
    }

    /// Get the length of the storage (number of elements).
    pub fn len(&self) -> usize {
        match self {
            Self::DenseF64(v) => v.len(),
            Self::DenseC64(v) => v.len(),
            Self::DiagF64(v) => v.len(),
            Self::DiagC64(v) => v.len(),
        }
    }

    /// Check if the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sum all elements as f64.
    pub fn sum_f64(&self) -> f64 {
        f64::sum_from_storage(self)
    }

    /// Sum all elements as Complex64.
    pub fn sum_c64(&self) -> Complex64 {
        Complex64::sum_from_storage(self)
    }

    /// Maximum absolute value over all stored elements.
    ///
    /// For real storage this is `max(|x|)`, and for complex storage this is
    /// `max(norm(z))`.
    pub fn max_abs(&self) -> f64 {
        match self {
            Storage::DenseF64(v) => v.as_slice().iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
            Storage::DiagF64(v) => v.as_slice().iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
            Storage::DenseC64(v) => v
                .as_slice()
                .iter()
                .map(|z| z.norm())
                .fold(0.0_f64, f64::max),
            Storage::DiagC64(v) => v
                .as_slice()
                .iter()
                .map(|z| z.norm())
                .fold(0.0_f64, f64::max),
        }
    }

    /// Convert this storage to dense storage.
    /// For Diag storage, creates a Dense storage with diagonal elements set
    /// and off-diagonal elements as zero.
    /// For Dense storage, returns a copy (clone).
    pub fn to_dense_storage(&self, dims: &[usize]) -> Storage {
        match self {
            Storage::DenseF64(v) => {
                // Clone preserves shape
                Storage::DenseF64(v.clone())
            }
            Storage::DenseC64(v) => {
                // Clone preserves shape
                Storage::DenseC64(v.clone())
            }
            Storage::DiagF64(d) => Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                d.to_dense_vec(dims),
                dims,
            )),
            Storage::DiagC64(d) => Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
                d.to_dense_vec(dims),
                dims,
            )),
        }
    }

    /// Permute the storage data according to the given permutation.
    ///
    /// For DenseStorage, uses internal shape information.
    /// The `dims` parameter is ignored for Dense (kept for DiagStorage compatibility).
    pub fn permute_storage(&self, _dims: &[usize], perm: &[usize]) -> Storage {
        match self {
            Storage::DenseF64(v) => Storage::DenseF64(v.permute(perm)),
            Storage::DenseC64(v) => Storage::DenseC64(v.permute(perm)),
            // For Diag storage, permute is trivial: data doesn't change, only index order changes
            Storage::DiagF64(v) => Storage::DiagF64(v.clone()),
            Storage::DiagC64(v) => Storage::DiagC64(v.clone()),
        }
    }

    /// Extract real part from Complex64 storage as f64 storage.
    /// For f64 storage, returns a copy (clone).
    pub fn extract_real_part(&self) -> Storage {
        match self {
            Storage::DenseF64(v) => {
                // Clone preserves shape
                Storage::DenseF64(v.clone())
            }
            Storage::DiagF64(d) => {
                Storage::DiagF64(DiagStorageF64::from_vec(d.as_slice().to_vec()))
            }
            Storage::DenseC64(v) => {
                let dims = v.dims();
                let real_vec: Vec<f64> = v.as_slice().iter().map(|z| z.re).collect();
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(real_vec, &dims))
            }
            Storage::DiagC64(d) => {
                let real_vec: Vec<f64> = d.as_slice().iter().map(|z| z.re).collect();
                Storage::DiagF64(DiagStorageF64::from_vec(real_vec))
            }
        }
    }

    /// Extract imaginary part from Complex64 storage as f64 storage.
    /// For f64 storage, returns zero storage (will be resized appropriately).
    pub fn extract_imag_part(&self, dims: &[usize]) -> Storage {
        match self {
            Storage::DenseF64(v) => {
                // For real storage, imaginary part is zero, preserve shape
                let d = v.dims();
                let total_size: usize = d.iter().product();
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                    vec![0.0; total_size],
                    &d,
                ))
            }
            Storage::DiagF64(_) => {
                // For real diagonal storage, imaginary part is zero
                let mindim_val = mindim(dims);
                Storage::DiagF64(DiagStorageF64::from_vec(vec![0.0; mindim_val]))
            }
            Storage::DenseC64(v) => {
                let d = v.dims();
                let imag_vec: Vec<f64> = v.as_slice().iter().map(|z| z.im).collect();
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(imag_vec, &d))
            }
            Storage::DiagC64(d) => {
                let imag_vec: Vec<f64> = d.as_slice().iter().map(|z| z.im).collect();
                Storage::DiagF64(DiagStorageF64::from_vec(imag_vec))
            }
        }
    }

    /// Convert f64 storage to Complex64 storage (real part only, imaginary part is zero).
    /// For Complex64 storage, returns a clone.
    pub fn to_complex_storage(&self) -> Storage {
        match self {
            Storage::DenseF64(v) => {
                let dims = v.dims();
                let c64_vec: Vec<Complex64> = v
                    .as_slice()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0))
                    .collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(c64_vec, &dims))
            }
            Storage::DiagF64(d) => {
                let c64_vec: Vec<Complex64> = d
                    .as_slice()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0))
                    .collect();
                Storage::DiagC64(DiagStorageC64::from_vec(c64_vec))
            }
            Storage::DenseC64(v) => {
                // Clone preserves shape
                Storage::DenseC64(v.clone())
            }
            Storage::DiagC64(d) => {
                Storage::DiagC64(DiagStorageC64::from_vec(d.as_slice().to_vec()))
            }
        }
    }

    /// Complex conjugate of all elements.
    ///
    /// For real (f64) storage, returns a clone (conjugate of real is identity).
    /// For complex (Complex64) storage, conjugates each element.
    ///
    /// This is inspired by the `conj` operation in ITensorMPS.jl.
    ///
    /// # Example
    /// ```
    /// use tensor4all_tensorbackend::{Storage, DenseStorageC64};
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)];
    /// let storage = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(data, &[2]));
    /// let conj_storage = storage.conj();
    ///
    /// // conj(1+2i) = 1-2i, conj(3-4i) = 3+4i
    /// ```
    pub fn conj(&self) -> Self {
        match self {
            Storage::DenseF64(v) => {
                // Real numbers: conj(x) = x, clone preserves shape
                Storage::DenseF64(v.clone())
            }
            Storage::DenseC64(v) => {
                let dims = v.dims();
                let conj_vec: Vec<Complex64> = v.as_slice().iter().map(|z| z.conj()).collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(conj_vec, &dims))
            }
            Storage::DiagF64(d) => {
                // Real numbers: conj(x) = x
                Storage::DiagF64(DiagStorageF64::from_vec(d.as_slice().to_vec()))
            }
            Storage::DiagC64(d) => {
                let conj_vec: Vec<Complex64> = d.as_slice().iter().map(|z| z.conj()).collect();
                Storage::DiagC64(DiagStorageC64::from_vec(conj_vec))
            }
        }
    }

    /// Combine two f64 storages into Complex64 storage.
    /// real_storage becomes the real part, imag_storage becomes the imaginary part.
    /// Formula: real + i * imag
    pub fn combine_to_complex(real_storage: &Storage, imag_storage: &Storage) -> Storage {
        match (real_storage, imag_storage) {
            (Storage::DenseF64(real), Storage::DenseF64(imag)) => {
                assert_eq!(real.len(), imag.len(), "Storage lengths must match");
                let dims = real.dims();
                let complex_vec: Vec<Complex64> = real
                    .as_slice()
                    .iter()
                    .zip(imag.as_slice().iter())
                    .map(|(&r, &i)| Complex64::new(r, i))
                    .collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(complex_vec, &dims))
            }
            (Storage::DiagF64(real), Storage::DiagF64(imag)) => {
                assert_eq!(real.len(), imag.len(), "Storage lengths must match");
                let complex_vec: Vec<Complex64> = real
                    .as_slice()
                    .iter()
                    .zip(imag.as_slice().iter())
                    .map(|(&r, &i)| Complex64::new(r, i))
                    .collect();
                Storage::DiagC64(DiagStorageC64::from_vec(complex_vec))
            }
            _ => panic!("Both storages must be the same type (DenseF64 or DiagF64)"),
        }
    }

    /// Add two storages element-wise, returning `Result` on error instead of panicking.
    ///
    /// Both storages must have the same type and length.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Storage types don't match
    /// - Storage lengths don't match
    pub fn try_add(&self, other: &Storage) -> Result<Storage, String> {
        match (self, other) {
            (Storage::DenseF64(a), Storage::DenseF64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for addition: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let dims = a.dims();
                let sum_vec: Vec<f64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Ok(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                    sum_vec, &dims,
                )))
            }
            (Storage::DenseC64(a), Storage::DenseC64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for addition: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let dims = a.dims();
                let sum_vec: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Ok(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
                    sum_vec, &dims,
                )))
            }
            (Storage::DiagF64(a), Storage::DiagF64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for addition: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let sum_vec: Vec<f64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Ok(Storage::DiagF64(DiagStorageF64::from_vec(sum_vec)))
            }
            (Storage::DiagC64(a), Storage::DiagC64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for addition: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let sum_vec: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Ok(Storage::DiagC64(DiagStorageC64::from_vec(sum_vec)))
            }
            _ => Err(format!(
                "Storage types must match for addition: {:?} vs {:?}",
                std::mem::discriminant(self),
                std::mem::discriminant(other)
            )),
        }
    }

    /// Try to subtract two storages element-wise.
    ///
    /// Returns an error if the storages have different types or lengths.
    pub fn try_sub(&self, other: &Storage) -> Result<Storage, String> {
        match (self, other) {
            (Storage::DenseF64(a), Storage::DenseF64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for subtraction: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let dims = a.dims();
                let diff_vec: Vec<f64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x - y)
                    .collect();
                Ok(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                    diff_vec, &dims,
                )))
            }
            (Storage::DenseC64(a), Storage::DenseC64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for subtraction: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let dims = a.dims();
                let diff_vec: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x - y)
                    .collect();
                Ok(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
                    diff_vec, &dims,
                )))
            }
            (Storage::DiagF64(a), Storage::DiagF64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for subtraction: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let diff_vec: Vec<f64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x - y)
                    .collect();
                Ok(Storage::DiagF64(DiagStorageF64::from_vec(diff_vec)))
            }
            (Storage::DiagC64(a), Storage::DiagC64(b)) => {
                if a.len() != b.len() {
                    return Err(format!(
                        "Storage lengths must match for subtraction: {} != {}",
                        a.len(),
                        b.len()
                    ));
                }
                let diff_vec: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x - y)
                    .collect();
                Ok(Storage::DiagC64(DiagStorageC64::from_vec(diff_vec)))
            }
            _ => Err(format!(
                "Storage types must match for subtraction: {:?} vs {:?}",
                std::mem::discriminant(self),
                std::mem::discriminant(other)
            )),
        }
    }

    /// Scale storage by a scalar value.
    ///
    /// If the scalar is complex but the storage is real, the storage is promoted to complex.
    pub fn scale(&self, scalar: &crate::AnyScalar) -> Storage {
        self * scalar.clone()
    }

    /// Compute linear combination: `a * self + b * other`.
    ///
    /// Returns an error if the storages have different types or lengths.
    /// If any scalar is complex, the result is promoted to complex.
    pub fn axpby(
        &self,
        a: &crate::AnyScalar,
        other: &Storage,
        b: &crate::AnyScalar,
    ) -> Result<Storage, String> {
        // First check lengths match
        if self.len() != other.len() {
            return Err(format!(
                "Storage lengths must match for axpby: {} != {}",
                self.len(),
                other.len()
            ));
        }

        // Determine if we need complex output
        let needs_complex = a.is_complex()
            || b.is_complex()
            || matches!(self, Storage::DenseC64(_) | Storage::DiagC64(_))
            || matches!(other, Storage::DenseC64(_) | Storage::DiagC64(_));

        if needs_complex {
            // Promote everything to complex
            let a_c: Complex64 = a.clone().into();
            let b_c: Complex64 = b.clone().into();

            let (result, dims): (Vec<Complex64>, Vec<usize>) = match (self, other) {
                (Storage::DenseF64(x), Storage::DenseF64(y)) => (
                    x.as_slice()
                        .iter()
                        .zip(y.as_slice().iter())
                        .map(|(&xi, &yi)| {
                            a_c * Complex64::new(xi, 0.0) + b_c * Complex64::new(yi, 0.0)
                        })
                        .collect(),
                    x.dims(),
                ),
                (Storage::DenseF64(x), Storage::DenseC64(y)) => (
                    x.as_slice()
                        .iter()
                        .zip(y.as_slice().iter())
                        .map(|(&xi, &yi)| a_c * Complex64::new(xi, 0.0) + b_c * yi)
                        .collect(),
                    x.dims(),
                ),
                (Storage::DenseC64(x), Storage::DenseF64(y)) => (
                    x.as_slice()
                        .iter()
                        .zip(y.as_slice().iter())
                        .map(|(&xi, &yi)| a_c * xi + b_c * Complex64::new(yi, 0.0))
                        .collect(),
                    x.dims(),
                ),
                (Storage::DenseC64(x), Storage::DenseC64(y)) => (
                    x.as_slice()
                        .iter()
                        .zip(y.as_slice().iter())
                        .map(|(&xi, &yi)| a_c * xi + b_c * yi)
                        .collect(),
                    x.dims(),
                ),
                _ => {
                    return Err(format!(
                        "axpby not supported for storage types: {:?} vs {:?}",
                        std::mem::discriminant(self),
                        std::mem::discriminant(other)
                    ))
                }
            };
            Ok(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
                result, &dims,
            )))
        } else {
            // All real
            if !a.is_real() || !b.is_real() {
                return Err(format!(
                    "expected real scalars in real axpby branch: a={a}, b={b}"
                ));
            }
            let a_f = a.real();
            let b_f = b.real();

            match (self, other) {
                (Storage::DenseF64(x), Storage::DenseF64(y)) => {
                    let dims = x.dims();
                    let result: Vec<f64> = x
                        .as_slice()
                        .iter()
                        .zip(y.as_slice().iter())
                        .map(|(&xi, &yi)| a_f * xi + b_f * yi)
                        .collect();
                    Ok(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                        result, &dims,
                    )))
                }
                (Storage::DiagF64(x), Storage::DiagF64(y)) => {
                    let result: Vec<f64> = x
                        .as_slice()
                        .iter()
                        .zip(y.as_slice().iter())
                        .map(|(&xi, &yi)| a_f * xi + b_f * yi)
                        .collect();
                    Ok(Storage::DiagF64(DiagStorageF64::from_vec(result)))
                }
                _ => Err(format!(
                    "axpby not supported for storage types: {:?} vs {:?}",
                    std::mem::discriminant(self),
                    std::mem::discriminant(other)
                )),
            }
        }
    }
}

/// Helper to get a mutable reference to storage, cloning if needed (COW).
pub fn make_mut_storage(arc: &mut Arc<Storage>) -> &mut Storage {
    Arc::make_mut(arc)
}

/// Get the minimum dimension from a slice of dimensions.
/// This is used for DiagTensor where all indices must have the same dimension.
pub fn mindim(dims: &[usize]) -> usize {
    dims.iter().copied().min().unwrap_or(1)
}

/// Contract two storage tensors along specified axes.
///
/// This is an internal helper function that contracts two `Storage` tensors.
/// For Dense tensors, uses dense contraction kernels.
/// For Diag tensors, implements specialized diagonal contraction.
///
/// # Arguments
/// * `storage_a` - First tensor storage
/// * `dims_a` - Dimensions of the first tensor
/// * `axes_a` - Axes of the first tensor to contract
/// * `storage_b` - Second tensor storage
/// * `dims_b` - Dimensions of the second tensor
/// * `axes_b` - Axes of the second tensor to contract
/// * `result_dims` - Dimensions of the result tensor (empty for scalar result)
///
/// # Returns
/// A new `Storage` containing the contracted result.
///
/// # Panics
/// Panics if the contracted dimensions don't match, or if the storage types
/// are incompatible.
pub fn contract_storage(
    storage_a: &Storage,
    dims_a: &[usize],
    axes_a: &[usize],
    storage_b: &Storage,
    dims_b: &[usize],
    axes_b: &[usize],
    result_dims: &[usize],
) -> Storage {
    // Verify that contracted dimensions match
    for (a_axis, b_axis) in axes_a.iter().zip(axes_b.iter()) {
        assert_eq!(
            dims_a[*a_axis], dims_b[*b_axis],
            "Contracted dimensions must match: dims_a[{}] = {} != dims_b[{}] = {}",
            a_axis, dims_a[*a_axis], b_axis, dims_b[*b_axis]
        );
    }

    match (storage_a, storage_b) {
        // Same type cases: DenseStorage has internal shape, use dense contraction.
        (Storage::DenseF64(a), Storage::DenseF64(b)) => {
            Storage::DenseF64(a.contract(axes_a, b, axes_b))
        }
        (Storage::DenseC64(a), Storage::DenseC64(b)) => {
            Storage::DenseC64(a.contract(axes_a, b, axes_b))
        }
        (Storage::DenseF64(a), Storage::DenseC64(b)) => {
            let promoted_a = promote_dense_to_c64(a);
            Storage::DenseC64(promoted_a.contract(axes_a, b, axes_b))
        }
        (Storage::DenseC64(a), Storage::DenseF64(b)) => {
            let promoted_b = promote_dense_to_c64(b);
            Storage::DenseC64(a.contract(axes_a, &promoted_b, axes_b))
        }
        // DiagTensor × DiagTensor contraction
        (Storage::DiagF64(a), Storage::DiagF64(b)) => a.contract_diag_diag(
            dims_a,
            b,
            dims_b,
            result_dims,
            |v| Storage::DenseF64(DenseStorage::from_vec_with_shape(v, result_dims)),
            |v| Storage::DiagF64(DiagStorage::from_vec(v)),
        ),
        (Storage::DiagC64(a), Storage::DiagC64(b)) => a.contract_diag_diag(
            dims_a,
            b,
            dims_b,
            result_dims,
            |v| Storage::DenseC64(DenseStorage::from_vec_with_shape(v, result_dims)),
            |v| Storage::DiagC64(DiagStorage::from_vec(v)),
        ),

        // Mixed Diag types: promote both to complex and recurse once.
        (Storage::DiagF64(_), Storage::DiagC64(_)) | (Storage::DiagC64(_), Storage::DiagF64(_)) => {
            let promoted_a = storage_a.to_complex_storage();
            let promoted_b = storage_b.to_complex_storage();
            contract_storage(
                &promoted_a,
                dims_a,
                axes_a,
                &promoted_b,
                dims_b,
                axes_b,
                result_dims,
            )
        }

        // DiagTensor × DenseTensor: use optimized contract_diag_dense
        (Storage::DiagF64(diag), Storage::DenseF64(dense)) => {
            diag.contract_diag_dense(dims_a, axes_a, dense, dims_b, axes_b, result_dims, |v| {
                Storage::DenseF64(DenseStorage::from_vec_with_shape(v, result_dims))
            })
        }
        (Storage::DiagC64(diag), Storage::DenseC64(dense)) => {
            diag.contract_diag_dense(dims_a, axes_a, dense, dims_b, axes_b, result_dims, |v| {
                Storage::DenseC64(DenseStorage::from_vec_with_shape(v, result_dims))
            })
        }

        // DenseTensor × DiagTensor: use generic helper
        (Storage::DenseF64(dense), Storage::DiagF64(diag)) => contract_dense_diag_impl(
            dense,
            dims_a,
            axes_a,
            diag,
            dims_b,
            axes_b,
            result_dims,
            |v, dims| Storage::DenseF64(DenseStorage::from_vec_with_shape(v, dims)),
            |s, perm| s.permute_storage(&[], perm),
        ),
        (Storage::DenseC64(dense), Storage::DiagC64(diag)) => contract_dense_diag_impl(
            dense,
            dims_a,
            axes_a,
            diag,
            dims_b,
            axes_b,
            result_dims,
            |v, dims| Storage::DenseC64(DenseStorage::from_vec_with_shape(v, dims)),
            |s, perm| s.permute_storage(&[], perm),
        ),

        // Mixed Diag/Dense with type promotion: promote f64 to Complex64
        (Storage::DiagF64(diag_f64), Storage::DenseC64(dense_c64)) => {
            // Diag<f64> × Dense<C64>: promote Diag to C64
            let diag_c64 = promote_diag_to_c64(diag_f64);
            diag_c64.contract_diag_dense(
                dims_a,
                axes_a,
                dense_c64,
                dims_b,
                axes_b,
                result_dims,
                |v| Storage::DenseC64(DenseStorage::from_vec_with_shape(v, result_dims)),
            )
        }
        (Storage::DenseC64(dense_c64), Storage::DiagF64(diag_f64)) => {
            // Dense<C64> × Diag<f64>: promote Diag to C64, use helper
            let diag_c64 = promote_diag_to_c64(diag_f64);
            contract_dense_diag_impl(
                dense_c64,
                dims_a,
                axes_a,
                &diag_c64,
                dims_b,
                axes_b,
                result_dims,
                |v, dims| Storage::DenseC64(DenseStorage::from_vec_with_shape(v, dims)),
                |s, perm| s.permute_storage(&[], perm),
            )
        }
        (Storage::DiagC64(diag_c64), Storage::DenseF64(dense_f64)) => {
            // Diag<C64> × Dense<f64>: promote Dense to C64
            let dense_c64 = promote_dense_to_c64(dense_f64);
            diag_c64.contract_diag_dense(
                dims_a,
                axes_a,
                &dense_c64,
                dims_b,
                axes_b,
                result_dims,
                |v| Storage::DenseC64(DenseStorage::from_vec_with_shape(v, result_dims)),
            )
        }
        (Storage::DenseF64(dense_f64), Storage::DiagC64(diag_c64)) => {
            // Dense<f64> × Diag<C64>: promote Dense to C64, use helper
            let dense_c64 = promote_dense_to_c64(dense_f64);
            contract_dense_diag_impl(
                &dense_c64,
                dims_a,
                axes_a,
                diag_c64,
                dims_b,
                axes_b,
                result_dims,
                |v, dims| Storage::DenseC64(DenseStorage::from_vec_with_shape(v, dims)),
                |s, perm| s.permute_storage(&[], perm),
            )
        }
    }
}

/// Promote Diag<f64> to Diag<Complex64>
fn promote_diag_to_c64(diag: &DiagStorage<f64>) -> DiagStorage<Complex64> {
    DiagStorage::from_vec(
        diag.as_slice()
            .iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect(),
    )
}

/// Promote Dense<f64> to Dense<Complex64>
fn promote_dense_to_c64(dense: &DenseStorage<f64>) -> DenseStorage<Complex64> {
    let dims = dense.dims();
    DenseStorage::from_vec_with_shape(
        dense
            .as_slice()
            .iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect(),
        &dims,
    )
}

/// Scalar types that can be extracted from and stored in `Storage`.
///
/// This trait provides conversion methods between scalar types and `Storage`,
/// supporting both view-based (borrowed) and owned operations.
///
/// # View-based operations
/// - `extract_dense_view`: Returns a borrowed slice (no copy)
/// - `extract_dense_cow`: Returns `Cow` (borrowed if possible, owned if needed)
///
/// # Owned operations
/// - `extract_dense`: Returns owned `Vec` (always copies)
/// - `dense_storage`: Creates `Storage` from owned `Vec`
pub trait StorageScalar: TensorElement + Copy + 'static {
    /// Extract a borrowed view of dense storage data (no copy).
    ///
    /// Returns an error if the storage is not the matching dense type.
    fn extract_dense_view(storage: &Storage) -> Result<&[Self], String>;

    /// Extract dense storage data as `Cow` (borrowed if possible, owned if needed).
    ///
    /// For dense storage, returns `Cow::Borrowed` (no copy).
    /// For other storage types, may need to convert to dense first (copy).
    fn extract_dense_cow<'a>(storage: &'a Storage) -> Result<Cow<'a, [Self]>, String> {
        Self::extract_dense_view(storage).map(Cow::Borrowed)
    }

    /// Extract dense storage data as owned `Vec` (always copies).
    ///
    /// This is a convenience method that calls `extract_dense_cow` and converts to owned.
    fn extract_dense(storage: &Storage) -> Result<Vec<Self>, String> {
        Ok(Self::extract_dense_cow(storage)?.into_owned())
    }

    /// Create `Storage` from owned dense data with explicit shape.
    fn dense_storage_with_shape(data: Vec<Self>, dims: &[usize]) -> Arc<Storage>;

    /// Create `Storage` from owned dense data (1D shape, for backward compatibility).
    fn dense_storage(data: Vec<Self>) -> Arc<Storage> {
        let len = data.len();
        Self::dense_storage_with_shape(data, &[len])
    }
}

impl StorageScalar for f64 {
    fn extract_dense_view(storage: &Storage) -> Result<&[Self], String> {
        match storage {
            Storage::DenseF64(ds) => Ok(ds.as_slice()),
            _ => Err(format!("Expected DenseF64 storage, got {:?}", storage)),
        }
    }

    fn dense_storage_with_shape(data: Vec<Self>, dims: &[usize]) -> Arc<Storage> {
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
            data, dims,
        )))
    }
}

impl StorageScalar for Complex64 {
    fn extract_dense_view(storage: &Storage) -> Result<&[Self], String> {
        match storage {
            Storage::DenseC64(ds) => Ok(ds.as_slice()),
            _ => Err(format!("Expected DenseC64 storage, got {:?}", storage)),
        }
    }

    fn dense_storage_with_shape(data: Vec<Self>, dims: &[usize]) -> Arc<Storage> {
        Arc::new(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            data, dims,
        )))
    }
}

/// Add two storages element-wise.
/// Both storages must have the same type and length.
///
/// # Panics
///
/// Panics if storage types don't match or lengths differ.
///
/// # Note
///
/// **Prefer using [`Storage::try_add`]** which returns a `Result` instead of panicking.
/// This trait implementation is kept for convenience but may panic on invalid inputs.
impl Add<&Storage> for &Storage {
    type Output = Storage;

    fn add(self, rhs: &Storage) -> Self::Output {
        match (self, rhs) {
            (Storage::DenseF64(a), Storage::DenseF64(b)) => {
                assert_eq!(a.len(), b.len(), "Storage lengths must match for addition");
                let dims = a.dims();
                let sum_vec: Vec<f64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(sum_vec, &dims))
            }
            (Storage::DenseC64(a), Storage::DenseC64(b)) => {
                assert_eq!(a.len(), b.len(), "Storage lengths must match for addition");
                let dims = a.dims();
                let sum_vec: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(sum_vec, &dims))
            }
            (Storage::DiagF64(a), Storage::DiagF64(b)) => {
                assert_eq!(a.len(), b.len(), "Storage lengths must match for addition");
                let sum_vec: Vec<f64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Storage::DiagF64(DiagStorageF64::from_vec(sum_vec))
            }
            (Storage::DiagC64(a), Storage::DiagC64(b)) => {
                assert_eq!(a.len(), b.len(), "Storage lengths must match for addition");
                let sum_vec: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Storage::DiagC64(DiagStorageC64::from_vec(sum_vec))
            }
            _ => panic!("Storage types must match for addition"),
        }
    }
}

/// Multiply storage by a scalar (f64).
/// For Complex64 storage, multiplies each element by the scalar (treated as real).
impl Mul<f64> for &Storage {
    type Output = Storage;

    fn mul(self, scalar: f64) -> Self::Output {
        match self {
            Storage::DenseF64(v) => {
                let dims = v.dims();
                let scaled_vec: Vec<f64> = v.as_slice().iter().map(|&x| x * scalar).collect();
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(scaled_vec, &dims))
            }
            Storage::DenseC64(v) => {
                let dims = v.dims();
                let scaled_vec: Vec<Complex64> = v
                    .as_slice()
                    .iter()
                    .map(|&z| z * Complex64::new(scalar, 0.0))
                    .collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled_vec, &dims))
            }
            Storage::DiagF64(d) => {
                let scaled_vec: Vec<f64> = d.as_slice().iter().map(|&x| x * scalar).collect();
                Storage::DiagF64(DiagStorageF64::from_vec(scaled_vec))
            }
            Storage::DiagC64(d) => {
                let scaled_vec: Vec<Complex64> = d
                    .as_slice()
                    .iter()
                    .map(|&z| z * Complex64::new(scalar, 0.0))
                    .collect();
                Storage::DiagC64(DiagStorageC64::from_vec(scaled_vec))
            }
        }
    }
}

/// Multiply storage by a scalar (Complex64).
impl Mul<Complex64> for &Storage {
    type Output = Storage;

    fn mul(self, scalar: Complex64) -> Self::Output {
        match self {
            Storage::DenseF64(v) => {
                // Promote f64 to Complex64
                let dims = v.dims();
                let scaled_vec: Vec<Complex64> = v
                    .as_slice()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0) * scalar)
                    .collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled_vec, &dims))
            }
            Storage::DenseC64(v) => {
                let dims = v.dims();
                let scaled_vec: Vec<Complex64> = v.as_slice().iter().map(|&z| z * scalar).collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled_vec, &dims))
            }
            Storage::DiagF64(d) => {
                // Promote f64 to Complex64
                let scaled_vec: Vec<Complex64> = d
                    .as_slice()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0) * scalar)
                    .collect();
                Storage::DiagC64(DiagStorageC64::from_vec(scaled_vec))
            }
            Storage::DiagC64(d) => {
                let scaled_vec: Vec<Complex64> = d.as_slice().iter().map(|&z| z * scalar).collect();
                Storage::DiagC64(DiagStorageC64::from_vec(scaled_vec))
            }
        }
    }
}

/// Multiply storage by a scalar (AnyScalar).
/// May promote f64 storage to Complex64 when scalar is complex.
impl Mul<AnyScalar> for &Storage {
    type Output = Storage;

    fn mul(self, scalar: AnyScalar) -> Self::Output {
        if scalar.is_complex() {
            let z: Complex64 = scalar.into();
            self * z
        } else {
            self * scalar.real()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::storage_strides;
    use crate::{dense_linear_multi_index, dense_linear_offset};

    /// Helper to extract f64 data from storage
    fn extract_f64(storage: &Storage) -> Vec<f64> {
        match storage {
            Storage::DenseF64(ds) => ds.as_slice().to_vec(),
            _ => panic!("Expected DenseF64"),
        }
    }

    /// Helper to extract Complex64 data from storage
    fn extract_c64(storage: &Storage) -> Vec<Complex64> {
        match storage {
            Storage::DenseC64(ds) => ds.as_slice().to_vec(),
            _ => panic!("Expected DenseC64"),
        }
    }

    // ===== DiagStorage<T> generic tests =====

    #[test]
    fn test_diag_storage_generic_f64() {
        let diag: DiagStorage<f64> = DiagStorage::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(diag.len(), 3);
        assert_eq!(diag.get(0), 1.0);
        assert_eq!(diag.get(1), 2.0);
        assert_eq!(diag.get(2), 3.0);
    }

    #[test]
    fn test_diag_storage_generic_c64() {
        let diag: DiagStorage<Complex64> =
            DiagStorage::from_vec(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);
        assert_eq!(diag.len(), 2);
        assert_eq!(diag.get(0), Complex64::new(1.0, 2.0));
        assert_eq!(diag.get(1), Complex64::new(3.0, 4.0));
    }

    #[test]
    fn test_diag_to_dense_vec_2d() {
        // 2D diagonal tensor [3, 3] with diag = [1, 2, 3]
        let diag: DiagStorage<f64> = DiagStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let dense = diag.to_dense_vec(&[3, 3]);
        // Expected: [[1,0,0], [0,2,0], [0,0,3]] in row-major
        assert_eq!(dense, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_diag_to_dense_vec_3d() {
        // 3D diagonal tensor [2, 2, 2] with diag = [1, 2]
        let diag: DiagStorage<f64> = DiagStorage::from_vec(vec![1.0, 2.0]);
        let dense = diag.to_dense_vec(&[2, 2, 2]);
        // Position (0,0,0) = 1.0, position (1,1,1) = 2.0, others = 0
        // Row-major: [[[1,0],[0,0]], [[0,0],[0,2]]]
        // Linear: 1,0,0,0,0,0,0,2
        assert_eq!(dense, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
    }

    // ===== Diag × Dense contraction tests =====

    #[test]
    fn test_contract_diag_dense_2d_all_contracted() {
        // Diag tensor [3, 3] with diag = [1, 2, 3]
        // Dense tensor [3, 3] with all 1s
        // Contract all axes: result = sum_t diag[t] * dense[t, t] = 1*1 + 2*1 + 3*1 = 6
        let diag = Storage::DiagF64(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
        let dense = Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![1.0; 9], &[3, 3]));

        let result = contract_storage(&diag, &[3, 3], &[0, 1], &dense, &[3, 3], &[0, 1], &[]);

        let data = extract_f64(&result);
        assert_eq!(data.len(), 1);
        assert!((data[0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_contract_diag_dense_2d_one_axis() {
        // Diag tensor [3, 3] with diag = [1, 2, 3]
        // Dense tensor [3, 2]
        // Contract axis 1 of diag with axis 0 of dense
        // Result[i, j] = diag[i, i] * dense[i, j] (since diag is only non-zero when i=k)
        //              = diag[i] * dense[i, j]
        let diag = Storage::DiagF64(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
        // Dense = [[1,2], [3,4], [5,6]] in row-major
        let dense = Storage::DenseF64(DenseStorage::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
        ));

        let result = contract_storage(&diag, &[3, 3], &[1], &dense, &[3, 2], &[0], &[3, 2]);

        let data = extract_f64(&result);
        // Result should be:
        // [diag[0]*dense[0,:], diag[1]*dense[1,:], diag[2]*dense[2,:]]
        // = [1*[1,2], 2*[3,4], 3*[5,6]]
        // = [[1,2], [6,8], [15,18]]
        // Row-major: [1, 2, 6, 8, 15, 18]
        assert_eq!(data.len(), 6);
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[1] - 2.0).abs() < 1e-10);
        assert!((data[2] - 6.0).abs() < 1e-10);
        assert!((data[3] - 8.0).abs() < 1e-10);
        assert!((data[4] - 15.0).abs() < 1e-10);
        assert!((data[5] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_contract_dense_diag_2d_one_axis() {
        // Dense × Diag (reversed order)
        // Dense tensor [2, 3]
        // Diag tensor [3, 3] with diag = [1, 2, 3]
        // Contract axis 1 of dense with axis 0 of diag
        let dense = Storage::DenseF64(DenseStorage::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let diag = Storage::DiagF64(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));

        let result = contract_storage(&dense, &[2, 3], &[1], &diag, &[3, 3], &[0], &[2, 3]);

        let data = extract_f64(&result);
        // Result[i, j] = dense[i, k] * diag[k, j] summed over k
        // But diag is only non-zero when k=j, so:
        // Result[i, j] = dense[i, j] * diag[j]
        // = [[1*1, 2*2, 3*3], [4*1, 5*2, 6*3]]
        // = [[1, 4, 9], [4, 10, 18]]
        // Row-major: [1, 4, 9, 4, 10, 18]
        assert_eq!(data.len(), 6);
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[1] - 4.0).abs() < 1e-10);
        assert!((data[2] - 9.0).abs() < 1e-10);
        assert!((data[3] - 4.0).abs() < 1e-10);
        assert!((data[4] - 10.0).abs() < 1e-10);
        assert!((data[5] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_contract_diag_dense_3d() {
        // Diag tensor [2, 2, 2] with diag = [1, 2]
        // Dense tensor [2, 3]
        // Contract axis 2 of diag with axis 0 of dense
        // Result has shape [2, 2, 3] but only diagonal in first two indices is non-zero
        let diag = Storage::DiagF64(DiagStorage::from_vec(vec![1.0, 2.0]));
        let dense = Storage::DenseF64(DenseStorage::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));

        let result = contract_storage(&diag, &[2, 2, 2], &[2], &dense, &[2, 3], &[0], &[2, 2, 3]);

        let data = extract_f64(&result);
        assert_eq!(data.len(), 12);
        // Result shape [2, 2, 3]
        // diag only non-zero at (t, t, t), so result[i, j, k] = diag[i] * dense[i, k] if i==j, else 0
        // Result[0, 0, :] = diag[0] * dense[0, :] = 1 * [1, 2, 3] = [1, 2, 3]
        // Result[0, 1, :] = 0 (diag is zero when i != j)
        // Result[1, 0, :] = 0
        // Result[1, 1, :] = diag[1] * dense[1, :] = 2 * [4, 5, 6] = [8, 10, 12]
        // Row-major: [1,2,3, 0,0,0, 0,0,0, 8,10,12]
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[1] - 2.0).abs() < 1e-10);
        assert!((data[2] - 3.0).abs() < 1e-10);
        assert!((data[3] - 0.0).abs() < 1e-10);
        assert!((data[4] - 0.0).abs() < 1e-10);
        assert!((data[5] - 0.0).abs() < 1e-10);
        assert!((data[6] - 0.0).abs() < 1e-10);
        assert!((data[7] - 0.0).abs() < 1e-10);
        assert!((data[8] - 0.0).abs() < 1e-10);
        assert!((data[9] - 8.0).abs() < 1e-10);
        assert!((data[10] - 10.0).abs() < 1e-10);
        assert!((data[11] - 12.0).abs() < 1e-10);
    }

    // ===== Type promotion tests =====

    #[test]
    fn test_contract_diag_f64_dense_c64() {
        // Diag<f64> × Dense<Complex64> should produce Dense<Complex64>
        let diag = Storage::DiagF64(DiagStorage::from_vec(vec![1.0, 2.0]));
        let dense = Storage::DenseC64(DenseStorage::from_vec_with_shape(
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, 2.0),
                Complex64::new(3.0, 3.0),
                Complex64::new(4.0, 4.0),
            ],
            &[2, 2],
        ));

        let result = contract_storage(&diag, &[2, 2], &[1], &dense, &[2, 2], &[0], &[2, 2]);

        let data = extract_c64(&result);
        assert_eq!(data.len(), 4);
        // Result[i, j] = diag[i] * dense[i, j]
        // Result[0, 0] = 1 * (1+1i) = 1+1i
        // Result[0, 1] = 1 * (2+2i) = 2+2i
        // Result[1, 0] = 2 * (3+3i) = 6+6i
        // Result[1, 1] = 2 * (4+4i) = 8+8i
        assert!((data[0] - Complex64::new(1.0, 1.0)).norm() < 1e-10);
        assert!((data[1] - Complex64::new(2.0, 2.0)).norm() < 1e-10);
        assert!((data[2] - Complex64::new(6.0, 6.0)).norm() < 1e-10);
        assert!((data[3] - Complex64::new(8.0, 8.0)).norm() < 1e-10);
    }

    #[test]
    fn test_contract_diag_c64_dense_f64() {
        // Diag<Complex64> × Dense<f64> should produce Dense<Complex64>
        let diag = Storage::DiagC64(DiagStorage::from_vec(vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
        ]));
        let dense = Storage::DenseF64(DenseStorage::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0],
            &[2, 2],
        ));

        let result = contract_storage(&diag, &[2, 2], &[1], &dense, &[2, 2], &[0], &[2, 2]);

        let data = extract_c64(&result);
        assert_eq!(data.len(), 4);
        // Result[i, j] = diag[i] * dense[i, j]
        // Result[0, 0] = (1+1i) * 1 = 1+1i
        // Result[0, 1] = (1+1i) * 2 = 2+2i
        // Result[1, 0] = (2+2i) * 3 = 6+6i
        // Result[1, 1] = (2+2i) * 4 = 8+8i
        assert!((data[0] - Complex64::new(1.0, 1.0)).norm() < 1e-10);
        assert!((data[1] - Complex64::new(2.0, 2.0)).norm() < 1e-10);
        assert!((data[2] - Complex64::new(6.0, 6.0)).norm() < 1e-10);
        assert!((data[3] - Complex64::new(8.0, 8.0)).norm() < 1e-10);
    }

    #[test]
    fn test_contract_dense_f64_diag_c64() {
        // Dense<f64> × Diag<Complex64> should produce Dense<Complex64>
        let dense = Storage::DenseF64(DenseStorage::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0],
            &[2, 2],
        ));
        let diag = Storage::DiagC64(DiagStorage::from_vec(vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
        ]));

        let result = contract_storage(&dense, &[2, 2], &[1], &diag, &[2, 2], &[0], &[2, 2]);

        let data = extract_c64(&result);
        assert_eq!(data.len(), 4);
        // Result[i, j] = dense[i, j] * diag[j]
        // Result[0, 0] = 1 * (1+1i) = 1+1i
        // Result[0, 1] = 2 * (2+2i) = 4+4i
        // Result[1, 0] = 3 * (1+1i) = 3+3i
        // Result[1, 1] = 4 * (2+2i) = 8+8i
        assert!((data[0] - Complex64::new(1.0, 1.0)).norm() < 1e-10);
        assert!((data[1] - Complex64::new(4.0, 4.0)).norm() < 1e-10);
        assert!((data[2] - Complex64::new(3.0, 3.0)).norm() < 1e-10);
        assert!((data[3] - Complex64::new(8.0, 8.0)).norm() < 1e-10);
    }

    #[test]
    fn test_contract_dense_c64_diag_f64() {
        // Dense<Complex64> × Diag<f64> should produce Dense<Complex64>
        let dense = Storage::DenseC64(DenseStorage::from_vec_with_shape(
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, 2.0),
                Complex64::new(3.0, 3.0),
                Complex64::new(4.0, 4.0),
            ],
            &[2, 2],
        ));
        let diag = Storage::DiagF64(DiagStorage::from_vec(vec![1.0, 2.0]));

        let result = contract_storage(&dense, &[2, 2], &[1], &diag, &[2, 2], &[0], &[2, 2]);

        let data = extract_c64(&result);
        assert_eq!(data.len(), 4);
        // Result[i, j] = dense[i, j] * diag[j]
        // Result[0, 0] = (1+1i) * 1 = 1+1i
        // Result[0, 1] = (2+2i) * 2 = 4+4i
        // Result[1, 0] = (3+3i) * 1 = 3+3i
        // Result[1, 1] = (4+4i) * 2 = 8+8i
        assert!((data[0] - Complex64::new(1.0, 1.0)).norm() < 1e-10);
        assert!((data[1] - Complex64::new(4.0, 4.0)).norm() < 1e-10);
        assert!((data[2] - Complex64::new(3.0, 3.0)).norm() < 1e-10);
        assert!((data[3] - Complex64::new(8.0, 8.0)).norm() < 1e-10);
    }

    // ===== Diag × Diag contraction tests =====

    #[test]
    fn test_contract_diag_diag_all_contracted() {
        // Diag [3, 3] × Diag [3, 3] with all indices contracted
        // Result = sum_t diag1[t] * diag2[t] (inner product)
        let diag1 = Storage::DiagF64(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
        let diag2 = Storage::DiagF64(DiagStorage::from_vec(vec![4.0, 5.0, 6.0]));

        let result = contract_storage(&diag1, &[3, 3], &[0, 1], &diag2, &[3, 3], &[0, 1], &[]);

        let data = extract_f64(&result);
        assert_eq!(data.len(), 1);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((data[0] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_contract_diag_diag_partial() {
        // Diag [3, 3] × Diag [3, 3] with one axis contracted
        // Result is a diagonal tensor
        let diag1 = Storage::DiagF64(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
        let diag2 = Storage::DiagF64(DiagStorage::from_vec(vec![4.0, 5.0, 6.0]));

        let result = contract_storage(&diag1, &[3, 3], &[1], &diag2, &[3, 3], &[0], &[3, 3]);

        // Result is element-wise product: [1*4, 2*5, 3*6] = [4, 10, 18]
        match &result {
            Storage::DiagF64(d) => {
                assert_eq!(d.as_slice(), &[4.0, 10.0, 18.0]);
            }
            _ => panic!("Expected DiagF64"),
        }
    }

    // ===== Type inspection tests =====

    #[test]
    fn test_is_f64() {
        let dense_f64 = Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
        let dense_c64 = Storage::DenseC64(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 0.0)],
            &[1],
        ));
        let diag_f64 = Storage::DiagF64(DiagStorage::from_vec(vec![1.0]));
        let diag_c64 = Storage::DiagC64(DiagStorage::from_vec(vec![Complex64::new(1.0, 0.0)]));

        assert!(dense_f64.is_f64());
        assert!(!dense_c64.is_f64());
        assert!(diag_f64.is_f64());
        assert!(!diag_c64.is_f64());
    }

    #[test]
    fn test_is_c64() {
        let dense_f64 = Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
        let dense_c64 = Storage::DenseC64(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 0.0)],
            &[1],
        ));
        let diag_f64 = Storage::DiagF64(DiagStorage::from_vec(vec![1.0]));
        let diag_c64 = Storage::DiagC64(DiagStorage::from_vec(vec![Complex64::new(1.0, 0.0)]));

        assert!(!dense_f64.is_c64());
        assert!(dense_c64.is_c64());
        assert!(!diag_f64.is_c64());
        assert!(diag_c64.is_c64());
    }

    #[test]
    fn test_is_complex() {
        let dense_f64 = Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
        let dense_c64 = Storage::DenseC64(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 0.0)],
            &[1],
        ));
        let diag_f64 = Storage::DiagF64(DiagStorage::from_vec(vec![1.0]));
        let diag_c64 = Storage::DiagC64(DiagStorage::from_vec(vec![Complex64::new(1.0, 0.0)]));

        // is_complex is an alias for is_c64
        assert!(!dense_f64.is_complex());
        assert!(dense_c64.is_complex());
        assert!(!diag_f64.is_complex());
        assert!(diag_c64.is_complex());
    }

    // ===== DenseStorage basic operations tests =====

    #[test]
    fn test_dense_from_scalar() {
        let ds = DenseStorage::from_scalar(42.0_f64);
        assert_eq!(ds.rank(), 0);
        assert_eq!(ds.len(), 1);
        assert!(!ds.is_empty());
        assert_eq!(ds.dims(), Vec::<usize>::new());
        assert_eq!(ds.as_slice(), &[42.0]);
    }

    #[test]
    fn test_dense_from_scalar_c64() {
        let val = Complex64::new(1.0, 2.0);
        let ds = DenseStorage::from_scalar(val);
        assert_eq!(ds.rank(), 0);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.as_slice(), &[val]);
    }

    #[test]
    fn test_dense_from_tensor_into_tensor_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ds = DenseStorage::from_vec_with_shape(data.clone(), &[2, 3]);
        let tensor = ds.into_tensor();
        assert_eq!(tensor.len(), 6);
        let ds2 = DenseStorage::from_tensor(tensor);
        assert_eq!(ds2.dims(), vec![2, 3]);
        assert_eq!(ds2.as_slice(), &data[..]);
    }

    #[test]
    fn test_dense_get_set() {
        let mut ds = DenseStorage::from_vec_with_shape(vec![10.0, 20.0, 30.0], &[3]);
        assert_eq!(ds.get(0), 10.0);
        assert_eq!(ds.get(1), 20.0);
        assert_eq!(ds.get(2), 30.0);

        ds.set(1, 99.0);
        assert_eq!(ds.get(1), 99.0);
    }

    #[test]
    fn test_dense_len_is_empty_dims_rank() {
        let ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(ds.len(), 4);
        assert!(!ds.is_empty());
        assert_eq!(ds.dims(), vec![2, 2]);
        assert_eq!(ds.rank(), 2);
    }

    #[test]
    fn test_dense_iter() {
        let ds = DenseStorage::from_vec_with_shape(vec![5.0, 10.0, 15.0], &[3]);
        let collected: Vec<f64> = ds.iter().copied().collect();
        assert_eq!(collected, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_dense_as_slice_as_mut_slice() {
        let mut ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]);
        assert_eq!(ds.as_slice(), &[1.0, 2.0, 3.0]);

        let slice = ds.as_mut_slice();
        slice[0] = 100.0;
        assert_eq!(ds.as_slice(), &[100.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dense_into_vec() {
        let ds = DenseStorage::from_vec_with_shape(vec![7.0, 8.0, 9.0], &[3]);
        let v = ds.into_vec();
        assert_eq!(v, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_dense_permute() {
        // 2x3 tensor, permute to 3x2
        let ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let permuted = ds.permute(&[1, 0]);
        assert_eq!(permuted.dims(), vec![3, 2]);
        // Original row-major [2,3]: [[1,2,3],[4,5,6]]
        // Transposed row-major [3,2]: [[1,4],[2,5],[3,6]]
        assert_eq!(permuted.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_dense_contract_matrix_multiply() {
        // Matrix multiply: [2,3] x [3,2] -> [2,2]
        let a = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = DenseStorage::from_vec_with_shape(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let result = a.contract(&[1], &b, &[0]);
        assert_eq!(result.dims(), vec![2, 2]);
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
        let data = result.as_slice();
        assert!((data[0] - 58.0).abs() < 1e-10);
        assert!((data[1] - 64.0).abs() < 1e-10);
        assert!((data[2] - 139.0).abs() < 1e-10);
        assert!((data[3] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense_contract_inner_product() {
        // Inner product: [3] x [3] -> scalar
        let a = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]);
        let b = DenseStorage::from_vec_with_shape(vec![4.0, 5.0, 6.0], &[3]);
        let result = a.contract(&[0], &b, &[0]);
        // 1*4 + 2*5 + 3*6 = 4+10+18 = 32
        assert_eq!(result.len(), 1);
        assert!((result.as_slice()[0] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense_random_f64() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        let ds = DenseStorage::<f64>::random(&mut rng, &[3, 4]);
        assert_eq!(ds.dims(), vec![3, 4]);
        assert_eq!(ds.len(), 12);
        // Values should not all be zero (with overwhelming probability)
        let nonzero = ds.as_slice().iter().any(|&x| x.abs() > 1e-10);
        assert!(nonzero);
    }

    #[test]
    fn test_dense_random_1d_f64() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(123);
        let ds = DenseStorage::<f64>::random_1d(&mut rng, 5);
        assert_eq!(ds.dims(), vec![5]);
        assert_eq!(ds.len(), 5);
    }

    #[test]
    fn test_dense_random_c64() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        let ds = DenseStorage::<Complex64>::random(&mut rng, &[2, 3]);
        assert_eq!(ds.dims(), vec![2, 3]);
        assert_eq!(ds.len(), 6);
        // At least one element should have nonzero imaginary part
        let has_imag = ds.as_slice().iter().any(|z| z.im.abs() > 1e-10);
        assert!(has_imag);
    }

    #[test]
    fn test_dense_random_1d_c64() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(99);
        let ds = DenseStorage::<Complex64>::random_1d(&mut rng, 4);
        assert_eq!(ds.dims(), vec![4]);
        assert_eq!(ds.len(), 4);
    }

    // ===== DiagStorage additional tests =====

    #[test]
    fn test_diag_as_slice_as_mut_slice() {
        let mut diag = DiagStorage::from_vec(vec![10.0, 20.0, 30.0]);
        assert_eq!(diag.as_slice(), &[10.0, 20.0, 30.0]);

        let slice = diag.as_mut_slice();
        slice[1] = 99.0;
        assert_eq!(diag.as_slice(), &[10.0, 99.0, 30.0]);
    }

    #[test]
    fn test_diag_into_vec() {
        let diag = DiagStorage::from_vec(vec![5.0, 6.0, 7.0]);
        let v = diag.into_vec();
        assert_eq!(v, vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_diag_is_empty() {
        let empty: DiagStorage<f64> = DiagStorage::from_vec(vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let nonempty = DiagStorage::from_vec(vec![1.0]);
        assert!(!nonempty.is_empty());
    }

    #[test]
    fn test_diag_set() {
        let mut diag = DiagStorage::from_vec(vec![1.0, 2.0, 3.0]);
        diag.set(0, 100.0);
        diag.set(2, 300.0);
        assert_eq!(diag.get(0), 100.0);
        assert_eq!(diag.get(1), 2.0);
        assert_eq!(diag.get(2), 300.0);
    }

    #[test]
    fn test_diag_to_dense_vec_1d() {
        // 1D diagonal tensor [3] with diag = [1, 2, 3]
        // This is just the vector itself
        let diag = DiagStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let dense = diag.to_dense_vec(&[3]);
        assert_eq!(dense, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_diag_to_dense_vec_c64() {
        let diag = DiagStorage::from_vec(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);
        let dense = diag.to_dense_vec(&[2, 2]);
        // [[1+2i, 0], [0, 3+4i]] in row-major
        assert_eq!(dense[0], Complex64::new(1.0, 2.0));
        assert_eq!(dense[1], Complex64::zero());
        assert_eq!(dense[2], Complex64::zero());
        assert_eq!(dense[3], Complex64::new(3.0, 4.0));
    }

    #[test]
    fn test_storage_strides_freeze_current_mdarray_backing_order() {
        assert_eq!(storage_strides(&[]), Vec::<usize>::new());
        assert_eq!(storage_strides(&[4]), vec![1]);
        assert_eq!(storage_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(storage_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(storage_strides(&[2, 1, 3]), vec![3, 3, 1]);
    }

    #[test]
    fn test_compute_strides_matches_current_column_major_linearization() {
        let dims = [2, 3, 4];
        assert_eq!(dense_linear_offset(&dims, &[0, 0, 0]).unwrap(), 0);
        assert_eq!(dense_linear_offset(&dims, &[1, 0, 0]).unwrap(), 1);
        assert_eq!(dense_linear_offset(&dims, &[0, 1, 0]).unwrap(), 2);
        assert_eq!(dense_linear_offset(&dims, &[0, 0, 1]).unwrap(), 6);
        assert_eq!(dense_linear_offset(&dims, &[1, 2, 3]).unwrap(), 23);
    }

    #[test]
    fn test_dense_linear_multi_index_matches_current_column_major_linearization() {
        let dims = [2, 3, 4];
        assert_eq!(dense_linear_multi_index(&dims, 0).unwrap(), vec![0, 0, 0]);
        assert_eq!(dense_linear_multi_index(&dims, 1).unwrap(), vec![1, 0, 0]);
        assert_eq!(dense_linear_multi_index(&dims, 2).unwrap(), vec![0, 1, 0]);
        assert_eq!(dense_linear_multi_index(&dims, 6).unwrap(), vec![0, 0, 1]);
        assert_eq!(dense_linear_multi_index(&dims, 23).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_diag_contract_diag_diag_scalar_result() {
        // All indices contracted: inner product
        let d1 = DiagStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let d2 = DiagStorage::from_vec(vec![4.0, 5.0, 6.0]);
        let result = d1.contract_diag_diag(
            &[3, 3],
            &d2,
            &[3, 3],
            &[],
            |v| Storage::DenseF64(DenseStorage::from_vec_with_shape(v, &[])),
            |v| Storage::DiagF64(DiagStorage::from_vec(v)),
        );
        let data = extract_f64(&result);
        assert_eq!(data.len(), 1);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((data[0] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_diag_contract_diag_diag_diag_result() {
        // Partial contraction: element-wise product
        let d1 = DiagStorage::from_vec(vec![2.0, 3.0]);
        let d2 = DiagStorage::from_vec(vec![5.0, 7.0]);
        let result = d1.contract_diag_diag(
            &[2, 2],
            &d2,
            &[2, 2],
            &[2, 2],
            |v| Storage::DenseF64(DenseStorage::from_vec_with_shape(v, &[2, 2])),
            |v| Storage::DiagF64(DiagStorage::from_vec(v)),
        );
        match &result {
            Storage::DiagF64(d) => {
                assert_eq!(d.as_slice(), &[10.0, 21.0]);
            }
            _ => panic!("Expected DiagF64"),
        }
    }

    #[test]
    fn test_diag_contract_diag_dense_basic() {
        // Diag [2,2] diag=[1,2], Dense [2,3] = [[1,2,3],[4,5,6]]
        // Contract axis 1 of diag with axis 0 of dense
        // Result[i,j] = diag[i] * dense[i,j]
        let diag = DiagStorage::from_vec(vec![1.0, 2.0]);
        let dense = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = diag.contract_diag_dense(&[2, 2], &[1], &dense, &[2, 3], &[0], &[2, 3], |v| {
            Storage::DenseF64(DenseStorage::from_vec_with_shape(v, &[2, 3]))
        });
        let data = extract_f64(&result);
        assert_eq!(data.len(), 6);
        // Result = [[1*1, 1*2, 1*3], [2*4, 2*5, 2*6]] = [[1,2,3],[8,10,12]]
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[1] - 2.0).abs() < 1e-10);
        assert!((data[2] - 3.0).abs() < 1e-10);
        assert!((data[3] - 8.0).abs() < 1e-10);
        assert!((data[4] - 10.0).abs() < 1e-10);
        assert!((data[5] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense_tensor_ref_and_mut() {
        let mut ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]);
        // Test tensor() for read access
        assert_eq!(ds.tensor().len(), 3);
        // Test tensor_mut() for write access
        ds.tensor_mut()[0] = 99.0;
        assert_eq!(ds.get(0), 99.0);
    }

    #[test]
    fn test_dense_deref() {
        let ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0], &[2]);
        // Deref gives access to tensor methods
        let _len = ds.len();
        assert_eq!(_len, 2);
    }

    #[test]
    fn test_dense_contract_c64() {
        // Verify contraction works for Complex64 too
        let a = DenseStorage::from_vec_with_shape(
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, 0.0),
            ],
            &[2, 2],
        );
        let b = DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
            &[2],
        );
        let result = a.contract(&[1], &b, &[0]);
        assert_eq!(result.dims(), vec![2]);
        // C[0] = (1+0i)*(1+0i) + (0+1i)*(0+1i) = 1 + (-1) = 0
        // C[1] = (1+1i)*(1+0i) + (2+0i)*(0+1i) = 1+1i + 0+2i = 1+3i
        assert!((result.as_slice()[0] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((result.as_slice()[1] - Complex64::new(1.0, 3.0)).norm() < 1e-10);
    }

    #[test]
    fn test_dense_permute_3d() {
        // 3D tensor [2, 3, 1], permute to [3, 1, 2]
        let ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
        let permuted = ds.permute(&[1, 2, 0]);
        assert_eq!(permuted.dims(), vec![3, 1, 2]);
        assert_eq!(permuted.len(), 6);
    }

    // ===== Storage-level tests for is_diag =====

    #[test]
    fn test_storage_is_diag() {
        let dense = Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
        let diag = Storage::DiagF64(DiagStorage::from_vec(vec![1.0]));
        assert!(!dense.is_diag());
        assert!(diag.is_diag());
    }

    // ===== Storage len / is_empty =====

    #[test]
    fn test_storage_len_is_empty() {
        let dense = Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![1.0, 2.0], &[2]));
        assert_eq!(dense.len(), 2);
        assert!(!dense.is_empty());

        let diag = Storage::DiagF64(DiagStorage::from_vec(vec![]));
        assert_eq!(diag.len(), 0);
        assert!(diag.is_empty());
    }

    // ===== DenseStorageFactory tests =====

    #[test]
    fn test_dense_storage_factory_f64() {
        let s = <f64 as DenseStorageFactory>::new_dense(3);
        assert_eq!(s.len(), 3);
        assert!(s.is_f64());
        let data = extract_f64(&s);
        assert_eq!(data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dense_storage_factory_c64() {
        let s = <Complex64 as DenseStorageFactory>::new_dense(2);
        assert_eq!(s.len(), 2);
        assert!(s.is_c64());
    }

    #[test]
    fn test_dense_storage_factory_with_shape_f64() {
        let s = <f64 as DenseStorageFactory>::new_dense_with_shape(&[2, 3]);
        assert_eq!(s.len(), 6);
    }

    #[test]
    fn test_dense_storage_factory_with_shape_c64() {
        let s = <Complex64 as DenseStorageFactory>::new_dense_with_shape(&[3, 2]);
        assert_eq!(s.len(), 6);
    }

    // ===== SumFromStorage tests =====

    #[test]
    fn test_sum_from_storage_f64() {
        let s = Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]));
        let sum: f64 = f64::sum_from_storage(&s);
        assert!((sum - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_sum_from_storage_diag_f64() {
        let s = Storage::DiagF64(DiagStorage::from_vec(vec![10.0, 20.0]));
        let sum: f64 = f64::sum_from_storage(&s);
        assert!((sum - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_storage_sum_f64_method() {
        let s = Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]));
        assert!((s.sum_f64() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_storage_sum_c64_method() {
        let s = Storage::DenseC64(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
            &[2],
        ));
        let sum = s.sum_c64();
        assert!((sum - Complex64::new(4.0, 6.0)).norm() < 1e-10);
    }

    #[test]
    fn test_storage_max_abs_and_to_dense_storage_cover_complex_and_diag() {
        let dense_c64 = Storage::DenseC64(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, -1.0)],
            &[2],
        ));
        assert!((dense_c64.max_abs() - 5.0).abs() < 1e-10);
        match dense_c64.to_dense_storage(&[2]) {
            Storage::DenseC64(ds) => assert_eq!(
                ds.as_slice(),
                &[Complex64::new(3.0, 4.0), Complex64::new(1.0, -1.0)]
            ),
            other => panic!("expected DenseC64, got {other:?}"),
        }

        let diag_c64 = Storage::DiagC64(DiagStorage::from_vec(vec![
            Complex64::new(0.0, 2.0),
            Complex64::new(3.0, 4.0),
        ]));
        assert!((diag_c64.max_abs() - 5.0).abs() < 1e-10);
        match diag_c64.to_dense_storage(&[2, 2]) {
            Storage::DenseC64(ds) => {
                assert_eq!(
                    ds.as_slice(),
                    &[
                        Complex64::new(0.0, 2.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(3.0, 4.0),
                    ]
                );
            }
            other => panic!("expected DenseC64, got {other:?}"),
        }
    }

    #[test]
    fn test_storage_projection_promotion_and_conjugation_helpers() {
        let dense_c64 = Storage::DenseC64(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)],
            &[2],
        ));
        match dense_c64.extract_real_part() {
            Storage::DenseF64(ds) => assert_eq!(ds.as_slice(), &[1.0, 3.0]),
            other => panic!("expected DenseF64, got {other:?}"),
        }
        match dense_c64.extract_imag_part(&[2]) {
            Storage::DenseF64(ds) => assert_eq!(ds.as_slice(), &[-2.0, 4.0]),
            other => panic!("expected DenseF64, got {other:?}"),
        }
        match dense_c64.conj() {
            Storage::DenseC64(ds) => {
                assert_eq!(
                    ds.as_slice(),
                    &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)]
                )
            }
            other => panic!("expected DenseC64, got {other:?}"),
        }

        let diag_f64 = Storage::DiagF64(DiagStorage::from_vec(vec![2.0, -1.0]));
        match diag_f64.extract_imag_part(&[2, 2]) {
            Storage::DiagF64(ds) => assert_eq!(ds.as_slice(), &[0.0, 0.0]),
            other => panic!("expected DiagF64, got {other:?}"),
        }
        match diag_f64.to_complex_storage() {
            Storage::DiagC64(ds) => {
                assert_eq!(
                    ds.as_slice(),
                    &[Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.0)]
                )
            }
            other => panic!("expected DiagC64, got {other:?}"),
        }
        let real = Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![1.0, 2.0], &[2]));
        let imag = Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![0.5, -1.5], &[2]));
        match Storage::combine_to_complex(&real, &imag) {
            Storage::DenseC64(ds) => {
                assert_eq!(
                    ds.as_slice(),
                    &[Complex64::new(1.0, 0.5), Complex64::new(2.0, -1.5)]
                )
            }
            other => panic!("expected DenseC64, got {other:?}"),
        }
    }

    #[test]
    fn test_storage_try_add_and_try_sub_cover_all_variants_and_errors() {
        let dense_f64_a =
            Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![1.0, 2.0], &[2]));
        let dense_f64_b =
            Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![3.0, -1.0], &[2]));
        match dense_f64_a.try_add(&dense_f64_b).unwrap() {
            Storage::DenseF64(ds) => assert_eq!(ds.as_slice(), &[4.0, 1.0]),
            other => panic!("expected DenseF64, got {other:?}"),
        }
        match dense_f64_a.try_sub(&dense_f64_b).unwrap() {
            Storage::DenseF64(ds) => assert_eq!(ds.as_slice(), &[-2.0, 3.0]),
            other => panic!("expected DenseF64, got {other:?}"),
        }

        let dense_c64_a = Storage::DenseC64(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, -2.0)],
            &[2],
        ));
        let dense_c64_b = Storage::DenseC64(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(-1.0, 0.5), Complex64::new(3.0, 1.0)],
            &[2],
        ));
        assert!(matches!(
            dense_c64_a.try_add(&dense_c64_b).unwrap(),
            Storage::DenseC64(_)
        ));
        assert!(matches!(
            dense_c64_a.try_sub(&dense_c64_b).unwrap(),
            Storage::DenseC64(_)
        ));

        let diag_f64_a = Storage::DiagF64(DiagStorage::from_vec(vec![1.0, 2.0]));
        let diag_f64_b = Storage::DiagF64(DiagStorage::from_vec(vec![0.5, -3.0]));
        assert!(matches!(
            diag_f64_a.try_add(&diag_f64_b).unwrap(),
            Storage::DiagF64(_)
        ));
        assert!(matches!(
            diag_f64_a.try_sub(&diag_f64_b).unwrap(),
            Storage::DiagF64(_)
        ));

        let diag_c64_a = Storage::DiagC64(DiagStorage::from_vec(vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(0.0, 2.0),
        ]));
        let diag_c64_b = Storage::DiagC64(DiagStorage::from_vec(vec![
            Complex64::new(0.5, 0.5),
            Complex64::new(-3.0, 1.0),
        ]));
        assert!(matches!(
            diag_c64_a.try_add(&diag_c64_b).unwrap(),
            Storage::DiagC64(_)
        ));
        assert!(matches!(
            diag_c64_a.try_sub(&diag_c64_b).unwrap(),
            Storage::DiagC64(_)
        ));

        let mismatched_len = Storage::DenseF64(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
        let err = dense_f64_a.try_add(&mismatched_len).unwrap_err();
        assert!(err.contains("Storage lengths must match for addition"));
        let err = dense_f64_a.try_sub(&mismatched_len).unwrap_err();
        assert!(err.contains("Storage lengths must match for subtraction"));

        let mismatched_type = Storage::DenseC64(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            &[2],
        ));
        let err = dense_f64_a.try_add(&mismatched_type).unwrap_err();
        assert!(err.contains("Storage types must match for addition"));
        let err = dense_f64_a.try_sub(&mismatched_type).unwrap_err();
        assert!(err.contains("Storage types must match for subtraction"));
    }
}
