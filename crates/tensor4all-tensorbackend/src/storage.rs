use faer::linalg::matmul::matmul as faer_matmul;
use faer::{Accum, Par};
use faer_traits::ComplexField;
use mdarray::{DTensor, DynRank, Rank, Shape, Tensor};
use num_complex::Complex64;
use num_traits::{One, Zero};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Deref, DerefMut, Mul};
use std::sync::Arc;

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
    + ComplexField
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
    /// to make the contracted axes contiguous before calling GEMM-based contraction.
    ///
    /// Uses internal shape information - no external dims parameters needed.
    pub fn contract(&self, axes: &[usize], other: &Self, other_axes: &[usize]) -> Self {
        let dims = self.dims();
        let other_dims = other.dims();

        // Check if axes are contiguous (need to permute if not)
        // For self: move contracted axes to end
        // For other: move contracted axes to front
        let (perm_self, new_axes_self, new_dims_self) =
            compute_contraction_permutation(&dims, axes, false);
        let (perm_other, new_axes_other, new_dims_other) =
            compute_contraction_permutation(&other_dims, other_axes, true);

        // Permute self if needed
        let storage_self = if perm_self.iter().enumerate().all(|(i, &p)| i == p) {
            std::borrow::Cow::Borrowed(self)
        } else {
            std::borrow::Cow::Owned(self.permute(&perm_self))
        };

        // Permute other if needed
        let storage_other = if perm_other.iter().enumerate().all(|(i, &p)| i == p) {
            std::borrow::Cow::Borrowed(other)
        } else {
            std::borrow::Cow::Owned(other.permute(&perm_other))
        };

        // Use manual GEMM-based contraction for robustness
        // This handles non-contiguous axes and edge cases that mdarray-linalg may not support
        let result_vec = contract_via_gemm(
            storage_self.as_slice(),
            &new_dims_self,
            &new_axes_self,
            storage_other.as_slice(),
            &new_dims_other,
            &new_axes_other,
        );

        // Compute result dimensions
        let result_dims = compute_result_dims(&dims, axes, &other_dims, other_axes);
        Self::from_vec_with_shape(result_vec, &result_dims)
    }
}

/// Compute the result dimensions after contraction.
fn compute_result_dims(
    dims_a: &[usize],
    axes_a: &[usize],
    dims_b: &[usize],
    axes_b: &[usize],
) -> Vec<usize> {
    let mut result_dims = Vec::new();
    for (i, &d) in dims_a.iter().enumerate() {
        if !axes_a.contains(&i) {
            result_dims.push(d);
        }
    }
    for (i, &d) in dims_b.iter().enumerate() {
        if !axes_b.contains(&i) {
            result_dims.push(d);
        }
    }
    result_dims
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

/// Contract two tensors via GEMM (matrix multiplication).
///
/// This function assumes that contracted axes are already contiguous:
/// - For `a`: contracted axes are at the END (axes_a are the last naxes positions)
/// - For `b`: contracted axes are at the FRONT (axes_b are the first naxes positions)
///
/// The contraction is equivalent to:
/// ```text
/// A[m..., k...] @ B[k..., n...] = C[m..., n...]
/// ```
///
/// where `k...` are the contracted dimensions.
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

    // Compute M, K, N for GEMM
    // M = product of non-contracted dimensions in A (first part after permutation)
    // K = product of contracted dimensions (should match in both)
    // N = product of non-contracted dimensions in B (last part after permutation)

    let ndim_a = dims_a.len();

    // Axes should be contiguous at end of A and front of B
    // For A: non-contracted are positions 0..ndim_a-naxes, contracted are ndim_a-naxes..ndim_a
    // For B: contracted are positions 0..naxes, non-contracted are naxes..ndim_b

    let m: usize = dims_a.iter().take(ndim_a - naxes).product();
    let m = if m == 0 { 1 } else { m }; // Handle case with no non-contracted dims

    let k: usize = dims_a.iter().skip(ndim_a - naxes).product();
    let k = if k == 0 { 1 } else { k };

    let n: usize = dims_b.iter().skip(naxes).product();
    let n = if n == 0 { 1 } else { n };

    // Verify K matches in both tensors
    let k_b: usize = dims_b.iter().take(naxes).product();
    let k_b = if k_b == 0 { 1 } else { k_b };
    assert_eq!(
        k, k_b,
        "Contracted dimension sizes must match: {} vs {}",
        k, k_b
    );

    // Perform GEMM: C[m, n] = A[m, k] @ B[k, n]
    // Data is row-major, so A[i, l] = a[i * k + l], B[l, j] = b[l * n + j]
    //
    // Create faer MatRef views with row-major strides:
    // - row_stride = number of columns (k for A, n for B)
    // - col_stride = 1
    let a_mat = unsafe { faer::MatRef::from_raw_parts(a.as_ptr(), m, k, k as isize, 1) };
    let b_mat = unsafe { faer::MatRef::from_raw_parts(b.as_ptr(), k, n, n as isize, 1) };

    let mut c = vec![T::zero(); m * n];
    let mut c_mat =
        unsafe { faer::MatMut::from_raw_parts_mut(c.as_mut_ptr(), m, n, n as isize, 1) };

    // Use faer GEMM: C = 1.0 * A * B + 0.0 * C
    faer_matmul(&mut c_mat, Accum::Replace, a_mat, b_mat, T::one(), Par::Seq);

    c
}

/// Compute permutation to make contracted axes contiguous.
///
/// If `axes_at_front` is true, contracted axes are moved to the front (maintaining original order).
/// If false, contracted axes are moved to the end (maintaining original order).
///
/// Returns (permutation, new_axes, new_dims).
///
/// IMPORTANT: The new_axes are returned in the SAME ORDER as the input axes,
/// so that the correspondence with the other tensor is preserved.
fn compute_contraction_permutation(
    dims: &[usize],
    axes: &[usize],
    axes_at_front: bool,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let ndim = dims.len();
    let naxes = axes.len();

    // Build permutation, preserving the order of contracted axes
    let non_contracted: Vec<usize> = (0..ndim).filter(|i| !axes.contains(i)).collect();

    let perm: Vec<usize> = if axes_at_front {
        // Contracted axes at front (in original order): [axes..., non_contracted...]
        axes.iter().chain(non_contracted.iter()).cloned().collect()
    } else {
        // Contracted axes at end (in original order): [non_contracted..., axes...]
        non_contracted.iter().chain(axes.iter()).cloned().collect()
    };

    // Compute new dims after permutation
    let new_dims: Vec<usize> = perm.iter().map(|&i| dims[i]).collect();

    // Compute new axes positions after permutation
    // The axes are now contiguous, but in the same order as the input
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
    pub fn from_vec(vec: Vec<T>) -> Self {
        Self(vec)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    pub fn into_vec(self) -> Vec<T> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<T: Clone> DiagStorage<T> {
    pub fn get(&self, i: usize) -> T {
        self.0[i].clone()
    }
}

impl<T: Copy> DiagStorage<T> {
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
    let dense_strides = compute_strides(dense_dims);

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
    let dense_non_contracted_strides = compute_strides(&dense_non_contracted_dims);

    // If all diag axes are contracted, result has shape = dense_non_contracted_dims
    // If some diag axes remain, result has shape = [d, d, ...] (diag non-contracted) + dense_non_contracted_dims
    let diag_non_contracted_count = diag_non_contracted.len();

    if diag_non_contracted_count == 0 {
        // All diagonal indices contracted: result is purely from dense's non-contracted part
        // For each t, we accumulate diag[t] * dense_slice[t] into result
        let mut result = vec![T::zero(); result_size];

        for (t, &diag_val) in diag.iter().enumerate().take(d.min(diag.len())) {
            // Compute base offset in dense for this t (all contracted axes = t)
            let base_offset: usize = axes_dense.iter().map(|&axis| t * dense_strides[axis]).sum();

            // Iterate over all non-contracted positions in dense
            for (flat_idx, result_item) in result.iter_mut().enumerate().take(dense_slice_size) {
                // Convert flat_idx to multi-index for non-contracted dims
                let mut offset = base_offset;
                let mut remaining = flat_idx;
                for (local_axis, &global_axis) in dense_non_contracted.iter().enumerate() {
                    let idx = remaining / dense_non_contracted_strides[local_axis];
                    remaining %= dense_non_contracted_strides[local_axis];
                    offset += idx * dense_strides[global_axis];
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
        let result_strides = compute_strides(result_dims);

        let mut result = vec![T::zero(); result_size];

        for (t, &diag_val) in diag.iter().enumerate().take(d.min(diag.len())) {
            // Compute base offset in dense for this t
            let base_offset_dense: usize =
                axes_dense.iter().map(|&axis| t * dense_strides[axis]).sum();

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
                    offset_dense += idx * dense_strides[global_axis];
                    // In result, these come after the diagonal indices
                    offset_result += idx * result_strides[diag_non_contracted_count + local_axis];
                }

                result[offset_result] = diag_val * dense[offset_dense];
            }
        }

        make_storage(result)
    }
}

/// Compute row-major strides for given dimensions.
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
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
/// When `backend-libtorch` is enabled, also supports Torch storage for autograd.
#[derive(Debug, Clone)]
pub enum Storage {
    DenseF64(DenseStorageF64),
    DenseC64(DenseStorageC64),
    DiagF64(DiagStorageF64),
    DiagC64(DiagStorageC64),
    /// Torch tensor storage for f64 (requires `backend-libtorch` feature)
    #[cfg(feature = "backend-libtorch")]
    TorchF64(crate::torch::TorchStorage<f64>),
    /// Torch tensor storage for Complex64 (requires `backend-libtorch` feature)
    #[cfg(feature = "backend-libtorch")]
    TorchC64(crate::torch::TorchStorage<Complex64>),
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
    fn sum_from_storage(storage: &Storage) -> Self;
}

impl SumFromStorage for f64 {
    fn sum_from_storage(storage: &Storage) -> Self {
        match storage {
            Storage::DenseF64(v) => v.as_slice().iter().copied().sum(),
            Storage::DenseC64(v) => v.as_slice().iter().map(|z| z.re).sum(),
            Storage::DiagF64(v) => v.as_slice().iter().copied().sum(),
            Storage::DiagC64(v) => v.as_slice().iter().map(|z| z.re).sum(),
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchF64(v) => v.to_vec().iter().copied().sum(),
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchC64(v) => v.to_vec().iter().map(|z| z.re).sum(),
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
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchF64(v) => Complex64::new(v.to_vec().iter().copied().sum(), 0.0),
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchC64(v) => v.to_vec().iter().copied().sum(),
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

    /// Check if this storage is a Torch storage type.
    #[cfg(feature = "backend-libtorch")]
    pub fn is_torch(&self) -> bool {
        matches!(self, Self::TorchF64(_) | Self::TorchC64(_))
    }

    /// Get the length of the storage (number of elements).
    pub fn len(&self) -> usize {
        match self {
            Self::DenseF64(v) => v.len(),
            Self::DenseC64(v) => v.len(),
            Self::DiagF64(v) => v.len(),
            Self::DiagC64(v) => v.len(),
            #[cfg(feature = "backend-libtorch")]
            Self::TorchF64(v) => v.len(),
            #[cfg(feature = "backend-libtorch")]
            Self::TorchC64(v) => v.len(),
        }
    }

    /// Check if the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ========================================================================
    // Autograd methods (only meaningful for Torch storage)
    // ========================================================================

    /// Check if this storage requires gradient computation.
    ///
    /// Returns `false` for non-Torch storage types.
    #[cfg(feature = "backend-libtorch")]
    pub fn requires_grad(&self) -> bool {
        match self {
            Self::TorchF64(v) => v.requires_grad(),
            Self::TorchC64(v) => v.requires_grad(),
            _ => false,
        }
    }

    /// Set whether this storage requires gradient computation.
    ///
    /// # Errors
    /// Returns an error for non-Torch storage types.
    #[cfg(feature = "backend-libtorch")]
    pub fn set_requires_grad(&mut self, requires_grad: bool) -> anyhow::Result<()> {
        match self {
            Self::TorchF64(v) => {
                v.set_requires_grad(requires_grad);
                Ok(())
            }
            Self::TorchC64(v) => {
                v.set_requires_grad(requires_grad);
                Ok(())
            }
            _ => anyhow::bail!("set_requires_grad is only supported for Torch storage"),
        }
    }

    /// Get the gradient of this storage, if it exists.
    ///
    /// Returns `None` for non-Torch storage or if no gradient has been computed.
    #[cfg(feature = "backend-libtorch")]
    pub fn grad(&self) -> Option<Storage> {
        match self {
            Self::TorchF64(v) => v.grad().map(Storage::TorchF64),
            Self::TorchC64(v) => v.grad().map(Storage::TorchC64),
            _ => None,
        }
    }

    /// Compute gradients by backpropagating from this storage.
    ///
    /// The storage must contain a scalar (single element).
    ///
    /// # Errors
    /// Returns an error for non-Torch storage types or if the storage is not a scalar.
    #[cfg(feature = "backend-libtorch")]
    pub fn backward(&self) -> anyhow::Result<()> {
        match self {
            Self::TorchF64(v) => v.backward(),
            Self::TorchC64(v) => v.backward(),
            _ => anyhow::bail!("backward is only supported for Torch storage"),
        }
    }

    /// Detach this storage from the computation graph.
    ///
    /// Returns a new storage that shares data but doesn't track gradients.
    /// For non-Torch storage, returns a clone.
    #[cfg(feature = "backend-libtorch")]
    pub fn detach(&self) -> Storage {
        match self {
            Self::TorchF64(v) => Self::TorchF64(v.detach()),
            Self::TorchC64(v) => Self::TorchC64(v.detach()),
            _ => self.clone(),
        }
    }

    /// Sum all elements as f64.
    pub fn sum_f64(&self) -> f64 {
        f64::sum_from_storage(self)
    }

    /// Sum all elements as Complex64.
    pub fn sum_c64(&self) -> Complex64 {
        Complex64::sum_from_storage(self)
    }

    /// Convert this storage to dense storage.
    /// For Diag storage, creates a Dense storage with diagonal elements set
    /// and off-diagonal elements as zero.
    /// For Dense storage, returns a copy (clone).
    /// For Torch storage, converts to mdarray Dense storage.
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
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchF64(v) => {
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(v.to_vec(), &v.dims()))
            }
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchC64(v) => {
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(v.to_vec(), &v.dims()))
            }
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
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchF64(v) => Storage::TorchF64(v.permute(perm)),
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchC64(v) => Storage::TorchC64(v.permute(perm)),
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
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchF64(v) => {
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(v.to_vec(), &v.dims()))
            }
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchC64(v) => {
                let real_vec: Vec<f64> = v.to_vec().iter().map(|z| z.re).collect();
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(real_vec, &v.dims()))
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
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchF64(v) => {
                // For real storage, imaginary part is zero
                let d = v.dims();
                let total_size: usize = d.iter().product();
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                    vec![0.0; total_size],
                    &d,
                ))
            }
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchC64(v) => {
                let d = v.dims();
                let imag_vec: Vec<f64> = v.to_vec().iter().map(|z| z.im).collect();
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(imag_vec, &d))
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
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchF64(v) => Storage::TorchC64(v.to_complex()),
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchC64(v) => Storage::TorchC64(v.clone()),
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
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchF64(v) => Storage::TorchF64(v.clone()),
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchC64(v) => {
                // Use native torch conj() for efficiency and autograd compatibility
                let conj_tensor = v.tensor().conj();
                Storage::TorchC64(crate::torch::TorchStorage::from_tensor(conj_tensor))
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
        use crate::AnyScalar;
        match (self, scalar) {
            // Real storage with real scalar
            (Storage::DenseF64(a), AnyScalar::F64(s)) => {
                let dims = a.dims();
                let scaled: Vec<f64> = a.as_slice().iter().map(|&x| x * s).collect();
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(scaled, &dims))
            }
            // Real storage with complex scalar -> promote to complex
            (Storage::DenseF64(a), AnyScalar::C64(s)) => {
                let dims = a.dims();
                let scaled: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0) * s)
                    .collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled, &dims))
            }
            // Complex storage with real scalar
            (Storage::DenseC64(a), AnyScalar::F64(s)) => {
                let dims = a.dims();
                let scaled: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .map(|&x| x * Complex64::new(*s, 0.0))
                    .collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled, &dims))
            }
            // Complex storage with complex scalar
            (Storage::DenseC64(a), AnyScalar::C64(s)) => {
                let dims = a.dims();
                let scaled: Vec<Complex64> = a.as_slice().iter().map(|&x| x * s).collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled, &dims))
            }
            // Diagonal variants
            (Storage::DiagF64(a), AnyScalar::F64(s)) => {
                let scaled: Vec<f64> = a.as_slice().iter().map(|&x| x * s).collect();
                Storage::DiagF64(DiagStorageF64::from_vec(scaled))
            }
            (Storage::DiagF64(a), AnyScalar::C64(s)) => {
                let scaled: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0) * s)
                    .collect();
                Storage::DiagC64(DiagStorageC64::from_vec(scaled))
            }
            (Storage::DiagC64(a), AnyScalar::F64(s)) => {
                let scaled: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .map(|&x| x * Complex64::new(*s, 0.0))
                    .collect();
                Storage::DiagC64(DiagStorageC64::from_vec(scaled))
            }
            (Storage::DiagC64(a), AnyScalar::C64(s)) => {
                let scaled: Vec<Complex64> = a.as_slice().iter().map(|&x| x * s).collect();
                Storage::DiagC64(DiagStorageC64::from_vec(scaled))
            }
            // Torch variants - convert to mdarray, scale, keep as mdarray for now
            #[cfg(feature = "backend-libtorch")]
            (Storage::TorchF64(a), AnyScalar::F64(s)) => {
                let dims = a.dims();
                let scaled: Vec<f64> = a.to_vec().iter().map(|&x| x * s).collect();
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(scaled, &dims))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::TorchF64(a), AnyScalar::C64(s)) => {
                let dims = a.dims();
                let scaled: Vec<Complex64> = a
                    .to_vec()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0) * s)
                    .collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled, &dims))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::TorchC64(a), AnyScalar::F64(s)) => {
                let dims = a.dims();
                let scaled: Vec<Complex64> = a
                    .to_vec()
                    .iter()
                    .map(|&x| x * Complex64::new(*s, 0.0))
                    .collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled, &dims))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::TorchC64(a), AnyScalar::C64(s)) => {
                let dims = a.dims();
                let scaled: Vec<Complex64> = a.to_vec().iter().map(|&x| x * s).collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled, &dims))
            }
            // Torch scalar variants - use native torch operations to preserve autograd
            #[cfg(feature = "backend-libtorch")]
            (Storage::DenseF64(a), AnyScalar::TorchF64(s)) => {
                let tensor = crate::torch::TorchStorage::<f64>::from_vec_with_shape(
                    a.as_slice().to_vec(),
                    &a.dims(),
                );
                let result = tensor.tensor() * s;
                Storage::TorchF64(crate::torch::TorchStorage::from_tensor(result))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::DenseF64(a), AnyScalar::TorchC64(s)) => {
                let data: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0))
                    .collect();
                let tensor =
                    crate::torch::TorchStorage::<Complex64>::from_vec_with_shape(data, &a.dims());
                let result = tensor.tensor() * s;
                Storage::TorchC64(crate::torch::TorchStorage::from_tensor(result))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::DenseC64(a), AnyScalar::TorchF64(s)) => {
                let tensor = crate::torch::TorchStorage::<Complex64>::from_vec_with_shape(
                    a.as_slice().to_vec(),
                    &a.dims(),
                );
                let result = tensor.tensor() * s;
                Storage::TorchC64(crate::torch::TorchStorage::from_tensor(result))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::DenseC64(a), AnyScalar::TorchC64(s)) => {
                let tensor = crate::torch::TorchStorage::<Complex64>::from_vec_with_shape(
                    a.as_slice().to_vec(),
                    &a.dims(),
                );
                let result = tensor.tensor() * s;
                Storage::TorchC64(crate::torch::TorchStorage::from_tensor(result))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::DiagF64(a), AnyScalar::TorchF64(s)) => {
                // Convert diag to dense first, then scale
                let dense = a.to_dense_vec(&[a.len(), a.len()]);
                let tensor = crate::torch::TorchStorage::<f64>::from_vec_with_shape(
                    dense,
                    &[a.len(), a.len()],
                );
                let result = tensor.tensor() * s;
                Storage::TorchF64(crate::torch::TorchStorage::from_tensor(result))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::DiagF64(a), AnyScalar::TorchC64(s)) => {
                let dense: Vec<Complex64> = a
                    .to_dense_vec(&[a.len(), a.len()])
                    .into_iter()
                    .map(|x| Complex64::new(x, 0.0))
                    .collect();
                let tensor = crate::torch::TorchStorage::<Complex64>::from_vec_with_shape(
                    dense,
                    &[a.len(), a.len()],
                );
                let result = tensor.tensor() * s;
                Storage::TorchC64(crate::torch::TorchStorage::from_tensor(result))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::DiagC64(a), AnyScalar::TorchF64(s)) => {
                let dense = a.to_dense_vec(&[a.len(), a.len()]);
                let tensor = crate::torch::TorchStorage::<Complex64>::from_vec_with_shape(
                    dense,
                    &[a.len(), a.len()],
                );
                let result = tensor.tensor() * s;
                Storage::TorchC64(crate::torch::TorchStorage::from_tensor(result))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::DiagC64(a), AnyScalar::TorchC64(s)) => {
                let dense = a.to_dense_vec(&[a.len(), a.len()]);
                let tensor = crate::torch::TorchStorage::<Complex64>::from_vec_with_shape(
                    dense,
                    &[a.len(), a.len()],
                );
                let result = tensor.tensor() * s;
                Storage::TorchC64(crate::torch::TorchStorage::from_tensor(result))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::TorchF64(a), AnyScalar::TorchF64(s)) => {
                let result = a.tensor() * s;
                Storage::TorchF64(crate::torch::TorchStorage::from_tensor(result))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::TorchF64(a), AnyScalar::TorchC64(s)) => {
                let complex = a.to_complex();
                let result = complex.tensor() * s;
                Storage::TorchC64(crate::torch::TorchStorage::from_tensor(result))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::TorchC64(a), AnyScalar::TorchF64(s)) => {
                let result = a.tensor() * s;
                Storage::TorchC64(crate::torch::TorchStorage::from_tensor(result))
            }
            #[cfg(feature = "backend-libtorch")]
            (Storage::TorchC64(a), AnyScalar::TorchC64(s)) => {
                let result = a.tensor() * s;
                Storage::TorchC64(crate::torch::TorchStorage::from_tensor(result))
            }
        }
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
        use crate::AnyScalar;

        // First check lengths match
        if self.len() != other.len() {
            return Err(format!(
                "Storage lengths must match for axpby: {} != {}",
                self.len(),
                other.len()
            ));
        }

        // Determine if we need complex output
        #[cfg(not(feature = "backend-libtorch"))]
        let needs_complex = matches!(a, AnyScalar::C64(_))
            || matches!(b, AnyScalar::C64(_))
            || matches!(self, Storage::DenseC64(_) | Storage::DiagC64(_))
            || matches!(other, Storage::DenseC64(_) | Storage::DiagC64(_));

        #[cfg(feature = "backend-libtorch")]
        let needs_complex = matches!(a, AnyScalar::C64(_))
            || matches!(b, AnyScalar::C64(_))
            || matches!(
                self,
                Storage::DenseC64(_) | Storage::DiagC64(_) | Storage::TorchC64(_)
            )
            || matches!(
                other,
                Storage::DenseC64(_) | Storage::DiagC64(_) | Storage::TorchC64(_)
            );

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
            let a_f = match a {
                AnyScalar::F64(v) => *v,
                AnyScalar::C64(_) => unreachable!(),
                #[cfg(feature = "backend-libtorch")]
                AnyScalar::TorchF64(t) => t.double_value(&[]),
                #[cfg(feature = "backend-libtorch")]
                AnyScalar::TorchC64(_) => unreachable!(),
            };
            let b_f = match b {
                AnyScalar::F64(v) => *v,
                AnyScalar::C64(_) => unreachable!(),
                #[cfg(feature = "backend-libtorch")]
                AnyScalar::TorchF64(t) => t.double_value(&[]),
                #[cfg(feature = "backend-libtorch")]
                AnyScalar::TorchC64(_) => unreachable!(),
            };

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
/// For Dense tensors, uses mdarray-linalg's contract method.
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
        // Same type cases: DenseStorage has internal shape, use new API
        (Storage::DenseF64(a), Storage::DenseF64(b)) => {
            Storage::DenseF64(a.contract(axes_a, b, axes_b))
        }
        (Storage::DenseC64(a), Storage::DenseC64(b)) => {
            Storage::DenseC64(a.contract(axes_a, b, axes_b))
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

        // Mixed types: f64 × Complex64 (use real/imaginary separation)
        (Storage::DenseF64(_), Storage::DenseC64(_))
        | (Storage::DiagF64(_), Storage::DiagC64(_)) => {
            // Extract real and imaginary parts from Complex64
            let c64_real = storage_b.extract_real_part();
            let c64_imag = storage_b.extract_imag_part(dims_b);

            // Contract f64 with real part
            let result_real = contract_storage(
                storage_a,
                dims_a,
                axes_a,
                &c64_real,
                dims_b,
                axes_b,
                result_dims,
            );

            // Contract f64 with imaginary part
            let result_imag = contract_storage(
                storage_a,
                dims_a,
                axes_a,
                &c64_imag,
                dims_b,
                axes_b,
                result_dims,
            );

            // Combine: result_real + i * result_imag
            // Convert real part to Complex64, multiply imag by i (which promotes to Complex64), then add
            let result_real_c64 = result_real.to_complex_storage();
            let result_imag_scaled = &result_imag * Complex64::new(0.0, 1.0);
            &result_real_c64 + &result_imag_scaled
        }

        // Mixed types: Complex64 × f64 (use real/imaginary separation)
        (Storage::DenseC64(_), Storage::DenseF64(_))
        | (Storage::DiagC64(_), Storage::DiagF64(_)) => {
            // Extract real and imaginary parts from Complex64
            let c64_real = storage_a.extract_real_part();
            let c64_imag = storage_a.extract_imag_part(dims_a);

            // Contract real part with f64
            let result_real = contract_storage(
                &c64_real,
                dims_a,
                axes_a,
                storage_b,
                dims_b,
                axes_b,
                result_dims,
            );

            // Contract imaginary part with f64
            let result_imag = contract_storage(
                &c64_imag,
                dims_a,
                axes_a,
                storage_b,
                dims_b,
                axes_b,
                result_dims,
            );

            // Combine: result_real + i * result_imag
            let result_real_c64 = result_real.to_complex_storage();
            let result_imag_scaled = &result_imag * Complex64::new(0.0, 1.0);
            let result_imag_c64 = result_imag_scaled.to_complex_storage();
            &result_real_c64 + &result_imag_c64
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

        // Torch storage: convert to dense and contract
        #[cfg(feature = "backend-libtorch")]
        (Storage::TorchF64(_), _)
        | (Storage::TorchC64(_), _)
        | (_, Storage::TorchF64(_))
        | (_, Storage::TorchC64(_)) => {
            // Convert both to dense storage and contract
            let dense_a = storage_a.to_dense_storage(dims_a);
            let dense_b = storage_b.to_dense_storage(dims_b);
            contract_storage(
                &dense_a,
                dims_a,
                axes_a,
                &dense_b,
                dims_b,
                axes_b,
                result_dims,
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
pub trait StorageScalar: Copy + 'static {
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

/// Convert dense storage to a DTensor with rank 2.
///
/// This function extracts data from dense storage and reshapes it into a `DTensor<T, 2>`
/// with the specified shape `[m, n]`. The data length must match `m * n`.
///
/// # Arguments
/// * `storage` - Dense storage (DenseF64 or DenseC64)
/// * `shape` - Shape array `[m, n]`
///
/// # Returns
/// A `DTensor<T, 2>` with the specified shape
///
/// # Errors
/// Returns an error if:
/// - Storage type doesn't match T
/// - Data length doesn't match `m * n`
pub fn storage_to_dtensor<T: StorageScalar>(
    storage: &Storage,
    shape: [usize; 2],
) -> Result<DTensor<T, 2>, String> {
    // Extract data
    let data = T::extract_dense(storage)?;

    // Validate length
    let expected_len: usize = shape[0] * shape[1];
    if data.len() != expected_len {
        return Err(format!(
            "Data length {} doesn't match shape product {}",
            data.len(),
            expected_len
        ));
    }

    // Create 1D tensor, then reshape to 2D
    let tensor_1d = mdarray::Tensor::<T, Rank<1>>::from(data);
    Ok(tensor_1d.into_shape(shape))
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
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchF64(v) => {
                let dims = v.dims();
                let scaled_vec: Vec<f64> = v.to_vec().iter().map(|&x| x * scalar).collect();
                Storage::DenseF64(DenseStorageF64::from_vec_with_shape(scaled_vec, &dims))
            }
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchC64(v) => {
                let dims = v.dims();
                let scaled_vec: Vec<Complex64> = v
                    .to_vec()
                    .iter()
                    .map(|&z| z * Complex64::new(scalar, 0.0))
                    .collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled_vec, &dims))
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
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchF64(v) => {
                let dims = v.dims();
                let scaled_vec: Vec<Complex64> = v
                    .to_vec()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0) * scalar)
                    .collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled_vec, &dims))
            }
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchC64(v) => {
                let dims = v.dims();
                let scaled_vec: Vec<Complex64> = v.to_vec().iter().map(|&z| z * scalar).collect();
                Storage::DenseC64(DenseStorageC64::from_vec_with_shape(scaled_vec, &dims))
            }
        }
    }
}

/// Multiply storage by a scalar (AnyScalar).
/// May promote f64 storage to Complex64 when scalar is complex.
impl Mul<AnyScalar> for &Storage {
    type Output = Storage;

    fn mul(self, scalar: AnyScalar) -> Self::Output {
        match scalar {
            AnyScalar::F64(x) => self * x,
            AnyScalar::C64(z) => self * z,
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(_) | AnyScalar::TorchC64(_) => self.scale(&scalar),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
