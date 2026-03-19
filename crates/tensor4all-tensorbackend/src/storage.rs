use anyhow::{anyhow, ensure, Result};
use mdarray::{DynRank, Shape, Tensor};
use num_complex::{Complex64, ComplexFloat};
use num_traits::{One, Zero};
#[cfg(test)]
use rand::Rng;
#[cfg(test)]
use rand_distr::{Distribution, StandardNormal};
use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Deref, DerefMut, Mul};
use std::sync::Arc;

/// Trait for scalar types supported by legacy dense/diagonal kernels.
pub(crate) trait DenseScalar:
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

/// Legacy dense kernel storage backed by mdarray's Tensor with dynamic rank.
///
/// This internal type keeps row-major physical kernels alive while higher-level
/// callers move to `StructuredStorage`.
#[derive(Debug, Clone)]
pub(crate) struct DenseStorage<T>(Tensor<T, DynRank>);

impl<T> DenseStorage<T> {
    /// Create a new legacy dense kernel storage from a Vec with explicit shape.
    ///
    /// # Panics
    /// Panics if the product of dims doesn't match vec.len().
    pub(crate) fn from_vec_with_shape(vec: Vec<T>, dims: &[usize]) -> Self {
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
    #[cfg(test)]
    pub(crate) fn from_scalar(val: T) -> Self {
        let tensor = Tensor::from(vec![val]).into_shape(DynRank::from_dims(&[]));
        Self(tensor)
    }

    /// Get the shape (dimensions) of the storage.
    pub(crate) fn dims(&self) -> Vec<usize> {
        self.0.shape().with_dims(|d| d.to_vec())
    }

    /// Get the rank (number of dimensions).
    pub(crate) fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Get underlying data as a slice.
    pub(crate) fn as_slice(&self) -> &[T] {
        &self.0[..]
    }

    /// Get underlying data as a mutable slice.
    #[cfg(test)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0[..]
    }

    /// Convert to Vec, consuming the storage.
    #[cfg(test)]
    pub(crate) fn into_vec(self) -> Vec<T> {
        self.0.into_vec()
    }

    /// Get the total number of elements.
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the storage is empty.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get a reference to the underlying tensor.
    #[cfg(test)]
    pub(crate) fn tensor(&self) -> &Tensor<T, DynRank> {
        &self.0
    }

    /// Get a mutable reference to the underlying tensor.
    #[cfg(test)]
    pub(crate) fn tensor_mut(&mut self) -> &mut Tensor<T, DynRank> {
        &mut self.0
    }

    /// Consume and return the underlying tensor.
    #[cfg(test)]
    pub(crate) fn into_tensor(self) -> Tensor<T, DynRank> {
        self.0
    }

    /// Create from an existing tensor.
    #[cfg(test)]
    pub(crate) fn from_tensor(tensor: Tensor<T, DynRank>) -> Self {
        Self(tensor)
    }

    /// Iterate over elements.
    #[cfg(test)]
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }
}

impl<T: Clone> DenseStorage<T> {
    /// Get element at linear index.
    #[cfg(test)]
    pub(crate) fn get(&self, i: usize) -> T {
        self.0[i].clone()
    }
}

impl<T: Copy> DenseStorage<T> {
    /// Set element at linear index.
    #[cfg(test)]
    pub(crate) fn set(&mut self, i: usize, val: T) {
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
    pub(crate) fn permute(&self, perm: &[usize]) -> Self {
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
    pub(crate) fn contract(&self, axes: &[usize], other: &Self, other_axes: &[usize]) -> Self {
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
    #[cfg(test)]
    pub(crate) fn random_1d<R: Rng>(rng: &mut R, size: usize) -> Self {
        let data: Vec<f64> = (0..size).map(|_| StandardNormal.sample(rng)).collect();
        Self::from_vec_with_shape(data, &[size])
    }

    /// Create storage with random values with explicit shape.
    #[cfg(test)]
    pub(crate) fn random<R: Rng>(rng: &mut R, dims: &[usize]) -> Self {
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
    #[cfg(test)]
    pub(crate) fn random_1d<R: Rng>(rng: &mut R, size: usize) -> Self {
        let data: Vec<Complex64> = (0..size)
            .map(|_| Complex64::new(StandardNormal.sample(rng), StandardNormal.sample(rng)))
            .collect();
        Self::from_vec_with_shape(data, &[size])
    }

    /// Create storage with random complex values with explicit shape.
    #[cfg(test)]
    pub(crate) fn random<R: Rng>(rng: &mut R, dims: &[usize]) -> Self {
        let size: usize = dims.iter().product();
        let data: Vec<Complex64> = (0..size)
            .map(|_| Complex64::new(StandardNormal.sample(rng), StandardNormal.sample(rng)))
            .collect();
        Self::from_vec_with_shape(data, dims)
    }
}

/// Type alias for f64 dense storage (for backward compatibility).
#[doc(hidden)]
pub(crate) type DenseStorageF64 = DenseStorage<f64>;

/// Type alias for Complex64 dense storage (for backward compatibility).
#[doc(hidden)]
pub(crate) type DenseStorageC64 = DenseStorage<Complex64>;

pub(crate) fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1isize;
    for &dim in dims {
        strides.push(stride);
        stride = stride
            .checked_mul(dim as isize)
            .unwrap_or_else(|| panic!("column-major stride overflow for dims {dims:?}"));
    }
    strides
}

fn validate_canonical_axis_classes(axis_classes: &[usize]) -> Result<()> {
    let mut next_class = 0usize;
    for &class_id in axis_classes {
        ensure!(
            class_id <= next_class,
            "axis_classes must be canonical first-appearance labels, got {axis_classes:?}"
        );
        if class_id == next_class {
            next_class += 1;
        }
    }
    Ok(())
}

fn required_storage_len(dims: &[usize], strides: &[isize]) -> Result<usize> {
    if dims.is_empty() {
        return Ok(1);
    }
    if dims.contains(&0) {
        return Ok(0);
    }
    ensure!(
        dims.len() == strides.len(),
        "payload dims {:?} and strides {:?} must have the same rank",
        dims,
        strides
    );

    let mut max_offset = 0usize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        ensure!(
            stride >= 0,
            "negative strides are not supported in StructuredStorage: {strides:?}"
        );
        if dim > 1 {
            max_offset = max_offset
                .checked_add((dim - 1) * usize::try_from(stride).unwrap_or(usize::MAX))
                .ok_or_else(|| anyhow!("payload stride overflow for dims {dims:?}"))?;
        }
    }
    Ok(max_offset + 1)
}

fn logical_dims_from_axis_classes(payload_dims: &[usize], axis_classes: &[usize]) -> Vec<usize> {
    axis_classes
        .iter()
        .map(|&class_id| payload_dims[class_id])
        .collect()
}

fn col_major_multi_index(mut linear: usize, dims: &[usize]) -> Vec<usize> {
    let mut index = Vec::with_capacity(dims.len());
    for &dim in dims {
        if dim == 0 {
            index.push(0);
        } else {
            index.push(linear % dim);
            linear /= dim;
        }
    }
    index
}

fn row_major_to_col_major_values<T: Clone>(data: &[T], dims: &[usize]) -> Vec<T> {
    let total_len: usize = dims.iter().product();
    if total_len == 0 {
        return Vec::new();
    }

    let row_major_strides = compute_strides(dims);
    (0..total_len)
        .map(|linear| {
            let index = col_major_multi_index(linear, dims);
            let offset: usize = index
                .iter()
                .zip(row_major_strides.iter())
                .map(|(&value, &stride)| value * stride)
                .sum();
            data[offset].clone()
        })
        .collect()
}

fn offset_from_strides(index: &[usize], strides: &[isize]) -> usize {
    index
        .iter()
        .zip(strides.iter())
        .map(|(&value, &stride)| value * usize::try_from(stride).unwrap_or(usize::MAX))
        .sum()
}

/// Structured tensor snapshot storage.
///
/// `data` and `strides` describe the payload tensor, while `axis_classes`
/// describes how logical axes map onto payload axes. Logical flat-buffer
/// semantics are column-major.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredStorage<T> {
    data: Vec<T>,
    payload_dims: Vec<usize>,
    strides: Vec<isize>,
    axis_classes: Vec<usize>,
}

impl<T> StructuredStorage<T> {
    /// Creates a structured payload snapshot from explicit payload metadata.
    ///
    /// `payload_dims` and `strides` describe the compressed payload tensor,
    /// while `axis_classes` maps logical axes onto payload axes in canonical
    /// first-appearance order.
    pub fn new(
        data: Vec<T>,
        payload_dims: Vec<usize>,
        strides: Vec<isize>,
        axis_classes: Vec<usize>,
    ) -> Result<Self> {
        validate_canonical_axis_classes(&axis_classes)?;
        ensure!(
            payload_dims.len()
                == axis_classes
                    .iter()
                    .copied()
                    .max()
                    .map(|value| value + 1)
                    .unwrap_or(0),
            "payload rank {} does not match axis_classes {:?}",
            payload_dims.len(),
            axis_classes
        );
        ensure!(
            strides.len() == payload_dims.len(),
            "payload dims {:?} and strides {:?} must have the same rank",
            payload_dims,
            strides
        );
        let required_len = required_storage_len(&payload_dims, &strides)?;
        ensure!(
            data.len() == required_len,
            "payload storage len {} does not match required len {} for dims {:?} and strides {:?}",
            data.len(),
            required_len,
            payload_dims,
            strides
        );
        Ok(Self {
            data,
            payload_dims,
            strides,
            axis_classes,
        })
    }

    /// Creates a dense structured snapshot from column-major logical data.
    pub fn from_dense_col_major(data: Vec<T>, logical_dims: &[usize]) -> Self {
        let payload_dims = logical_dims.to_vec();
        let strides = col_major_strides(&payload_dims);
        let axis_classes = (0..logical_dims.len()).collect();
        Self::new(data, payload_dims, strides, axis_classes)
            .unwrap_or_else(|err| panic!("StructuredStorage::from_dense_col_major failed: {err}"))
    }

    /// Creates a diagonal structured snapshot from column-major diagonal data.
    pub fn from_diag_col_major(diag_data: Vec<T>, logical_rank: usize) -> Self {
        let payload_dims = if logical_rank == 0 {
            vec![]
        } else {
            vec![diag_data.len()]
        };
        let strides = col_major_strides(&payload_dims);
        let axis_classes = vec![0; logical_rank];
        Self::new(diag_data, payload_dims, strides, axis_classes)
            .unwrap_or_else(|err| panic!("StructuredStorage::from_diag_col_major failed: {err}"))
    }

    /// Returns the owned payload buffer.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns the payload tensor dimensions.
    pub fn payload_dims(&self) -> &[usize] {
        &self.payload_dims
    }

    /// Returns the payload tensor strides.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Returns the canonical logical-to-payload axis classes.
    pub fn axis_classes(&self) -> &[usize] {
        &self.axis_classes
    }

    /// Returns the logical dimensions derived from `payload_dims` and `axis_classes`.
    pub fn logical_dims(&self) -> Vec<usize> {
        logical_dims_from_axis_classes(&self.payload_dims, &self.axis_classes)
    }

    /// Returns the logical rank.
    pub fn logical_rank(&self) -> usize {
        self.axis_classes.len()
    }

    /// Returns `true` when the logical tensor is dense.
    pub fn is_dense(&self) -> bool {
        self.axis_classes
            .iter()
            .copied()
            .eq(0..self.axis_classes.len())
    }

    /// Returns `true` when the logical tensor is diagonal.
    pub fn is_diag(&self) -> bool {
        self.logical_rank() >= 2 && self.axis_classes.iter().all(|&class_id| class_id == 0)
    }

    /// Returns the payload buffer length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the payload buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a borrowed view when the logical tensor is dense and the
    /// payload is already stored contiguously in column-major order.
    pub fn dense_col_major_view_if_contiguous(&self) -> Option<&[T]> {
        if self.is_dense() && self.strides == col_major_strides(&self.payload_dims) {
            Some(&self.data)
        } else {
            None
        }
    }
}

impl<T: Clone> StructuredStorage<T> {
    /// Materializes the payload tensor as a contiguous column-major buffer.
    pub fn payload_col_major_vec(&self) -> Vec<T> {
        let payload_len: usize = self.payload_dims.iter().product();
        if payload_len == 0 {
            return Vec::new();
        }
        if self.strides == col_major_strides(&self.payload_dims) {
            return self.data.clone();
        }

        (0..payload_len)
            .map(|linear| {
                let index = col_major_multi_index(linear, &self.payload_dims);
                let offset = offset_from_strides(&index, &self.strides);
                self.data[offset].clone()
            })
            .collect()
    }

    /// Returns a copy of the storage with logical axes permuted.
    pub fn permute_logical_axes(&self, perm: &[usize]) -> Self {
        assert_eq!(
            perm.len(),
            self.axis_classes.len(),
            "logical permutation length {} must match logical rank {}",
            perm.len(),
            self.axis_classes.len()
        );
        let axis_classes = perm.iter().map(|&index| self.axis_classes[index]).collect();
        Self::new(
            self.data.clone(),
            self.payload_dims.clone(),
            self.strides.clone(),
            axis_classes,
        )
        .unwrap_or_else(|err| panic!("StructuredStorage::permute_logical_axes failed: {err}"))
    }
}

impl<T: Copy> StructuredStorage<T> {
    /// Maps payload elements while preserving payload metadata and axis classes.
    pub fn map_copy<U>(&self, mut f: impl FnMut(T) -> U) -> StructuredStorage<U> {
        StructuredStorage::new(
            self.data.iter().copied().map(&mut f).collect(),
            self.payload_dims.clone(),
            self.strides.clone(),
            self.axis_classes.clone(),
        )
        .unwrap_or_else(|err| panic!("StructuredStorage::map_copy failed: {err}"))
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
pub(crate) struct DiagStorage<T>(Vec<T>);

impl<T> DiagStorage<T> {
    /// Create a new diagonal storage from a vector of diagonal elements.
    pub(crate) fn from_vec(vec: Vec<T>) -> Self {
        Self(vec)
    }

    /// Get a slice of the diagonal elements.
    pub(crate) fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Get a mutable slice of the diagonal elements.
    #[cfg(test)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    /// Consume the storage and return the underlying vector.
    #[cfg(test)]
    pub(crate) fn into_vec(self) -> Vec<T> {
        self.0
    }

    /// Return the number of diagonal elements.
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// Return true if the storage has no elements.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<T: Clone> DiagStorage<T> {
    /// Get a clone of the diagonal element at index `i`.
    #[cfg(test)]
    pub(crate) fn get(&self, i: usize) -> T {
        self.0[i].clone()
    }
}

impl<T: Copy> DiagStorage<T> {
    /// Set the diagonal element at index `i` to `val`.
    #[cfg(test)]
    pub(crate) fn set(&mut self, i: usize, val: T) {
        self.0[i] = val;
    }
}

impl<T: DenseScalar> DiagStorage<T> {
    /// Convert diagonal storage to a dense vector representation.
    /// Creates a dense vector with diagonal elements set and off-diagonal elements as zero.
    pub(crate) fn to_dense_vec(&self, dims: &[usize]) -> Vec<T> {
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
    pub(crate) fn contract_diag_diag(
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
    pub(crate) fn contract_diag_dense(
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
#[doc(hidden)]
pub(crate) type DiagStorageF64 = DiagStorage<f64>;

/// Type alias for Complex64 diagonal storage (for backward compatibility).
#[doc(hidden)]
pub(crate) type DiagStorageC64 = DiagStorage<Complex64>;

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
///
/// Public callers interact with this opaque wrapper through constructors and
/// high-level query/materialization methods. Temporary dense/diagonal kernel
/// variants stay crate-private during the structured-storage migration.
#[derive(Debug, Clone)]
pub struct Storage(pub(crate) StorageRepr);

#[derive(Debug, Clone)]
pub(crate) struct NativePayload<T> {
    pub(crate) data: Vec<T>,
    pub(crate) payload_dims: Vec<usize>,
    pub(crate) axis_classes: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub(crate) enum StorageRepr {
    /// Dense storage with f64 elements.
    #[doc(hidden)]
    DenseF64(DenseStorageF64),
    /// Dense storage with Complex64 elements.
    #[doc(hidden)]
    DenseC64(DenseStorageC64),
    /// Diagonal storage with f64 elements.
    #[doc(hidden)]
    DiagF64(DiagStorageF64),
    /// Diagonal storage with Complex64 elements.
    #[doc(hidden)]
    DiagC64(DiagStorageC64),
    /// General structured storage with f64 elements.
    StructuredF64(StructuredStorage<f64>),
    /// General structured storage with Complex64 elements.
    StructuredC64(StructuredStorage<Complex64>),
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
        match &storage.0 {
            StorageRepr::DenseF64(v) => v.as_slice().iter().copied().sum(),
            StorageRepr::DenseC64(v) => v.as_slice().iter().map(|z| z.re).sum(),
            StorageRepr::DiagF64(v) => v.as_slice().iter().copied().sum(),
            StorageRepr::DiagC64(v) => v.as_slice().iter().map(|z| z.re).sum(),
            StorageRepr::StructuredF64(v) => v.data().iter().copied().sum(),
            StorageRepr::StructuredC64(v) => v.data().iter().map(|z| z.re).sum(),
        }
    }
}

impl SumFromStorage for Complex64 {
    fn sum_from_storage(storage: &Storage) -> Self {
        match &storage.0 {
            StorageRepr::DenseF64(v) => Complex64::new(v.as_slice().iter().copied().sum(), 0.0),
            StorageRepr::DenseC64(v) => v.as_slice().iter().copied().sum(),
            StorageRepr::DiagF64(v) => Complex64::new(v.as_slice().iter().copied().sum(), 0.0),
            StorageRepr::DiagC64(v) => v.as_slice().iter().copied().sum(),
            StorageRepr::StructuredF64(v) => Complex64::new(v.data().iter().copied().sum(), 0.0),
            StorageRepr::StructuredC64(v) => v.data().iter().copied().sum(),
        }
    }
}

// AnyScalar is now in its own module
pub use crate::any_scalar::AnyScalar;

impl Storage {
    pub(crate) fn from_repr(repr: StorageRepr) -> Self {
        Self(repr)
    }

    #[cfg(test)]
    pub(crate) fn repr(&self) -> &StorageRepr {
        &self.0
    }

    pub(crate) fn dense_f64_legacy(value: DenseStorageF64) -> Self {
        Self(StorageRepr::DenseF64(value))
    }

    pub(crate) fn dense_c64_legacy(value: DenseStorageC64) -> Self {
        Self(StorageRepr::DenseC64(value))
    }

    pub(crate) fn diag_f64_legacy(value: DiagStorageF64) -> Self {
        Self(StorageRepr::DiagF64(value))
    }

    pub(crate) fn diag_c64_legacy(value: DiagStorageC64) -> Self {
        Self(StorageRepr::DiagC64(value))
    }

    pub(crate) fn structured_f64(value: StructuredStorage<f64>) -> Self {
        Self(StorageRepr::StructuredF64(value))
    }

    pub(crate) fn structured_c64(value: StructuredStorage<Complex64>) -> Self {
        Self(StorageRepr::StructuredC64(value))
    }

    fn validate_dense_len<T>(data: &[T], logical_dims: &[usize], label: &str) -> Result<()> {
        let expected_len: usize = logical_dims.iter().product();
        ensure!(
            data.len() == expected_len,
            "{label} len {} does not match logical dims {:?} (expected {})",
            data.len(),
            logical_dims,
            expected_len
        );
        Ok(())
    }

    fn validate_diag_dims(payload_len: usize, logical_dims: &[usize], label: &str) -> Result<()> {
        ensure!(
            logical_dims.iter().all(|&dim| dim == payload_len),
            "{label} payload len {payload_len} does not match logical dims {:?}",
            logical_dims
        );
        Ok(())
    }

    pub(crate) fn native_payload_f64(&self, logical_dims: &[usize]) -> Result<NativePayload<f64>> {
        match &self.0 {
            StorageRepr::DenseF64(value) => {
                Self::validate_dense_len(value.as_slice(), logical_dims, "dense f64 payload")?;
                Ok(NativePayload {
                    data: row_major_to_col_major_values(value.as_slice(), logical_dims),
                    payload_dims: logical_dims.to_vec(),
                    axis_classes: None,
                })
            }
            StorageRepr::DiagF64(value) => {
                Self::validate_diag_dims(value.len(), logical_dims, "diag f64")?;
                Ok(NativePayload {
                    data: value.as_slice().to_vec(),
                    payload_dims: vec![value.len()],
                    axis_classes: Some(vec![0; logical_dims.len()]),
                })
            }
            StorageRepr::StructuredF64(value) => {
                ensure!(
                    value.logical_dims() == logical_dims,
                    "logical dims {:?} do not match structured f64 logical dims {:?}",
                    logical_dims,
                    value.logical_dims()
                );
                let axis_classes = if value.is_dense() {
                    None
                } else {
                    Some(value.axis_classes().to_vec())
                };
                Ok(NativePayload {
                    data: value.payload_col_major_vec(),
                    payload_dims: value.payload_dims().to_vec(),
                    axis_classes,
                })
            }
            StorageRepr::DenseC64(_) | StorageRepr::DiagC64(_) | StorageRepr::StructuredC64(_) => {
                Err(anyhow!(
                    "complex storage cannot be converted to f64 native payload"
                ))
            }
        }
    }

    pub(crate) fn native_payload_c64(
        &self,
        logical_dims: &[usize],
    ) -> Result<NativePayload<Complex64>> {
        match &self.0 {
            StorageRepr::DenseC64(value) => {
                Self::validate_dense_len(value.as_slice(), logical_dims, "dense c64 payload")?;
                Ok(NativePayload {
                    data: row_major_to_col_major_values(value.as_slice(), logical_dims),
                    payload_dims: logical_dims.to_vec(),
                    axis_classes: None,
                })
            }
            StorageRepr::DiagC64(value) => {
                Self::validate_diag_dims(value.len(), logical_dims, "diag c64")?;
                Ok(NativePayload {
                    data: value.as_slice().to_vec(),
                    payload_dims: vec![value.len()],
                    axis_classes: Some(vec![0; logical_dims.len()]),
                })
            }
            StorageRepr::DenseF64(value) => {
                Self::validate_dense_len(value.as_slice(), logical_dims, "dense f64 payload")?;
                Ok(NativePayload {
                    data: row_major_to_col_major_values(value.as_slice(), logical_dims)
                        .into_iter()
                        .map(|entry| Complex64::new(entry, 0.0))
                        .collect(),
                    payload_dims: logical_dims.to_vec(),
                    axis_classes: None,
                })
            }
            StorageRepr::DiagF64(value) => {
                Self::validate_diag_dims(value.len(), logical_dims, "diag f64")?;
                Ok(NativePayload {
                    data: value
                        .as_slice()
                        .iter()
                        .copied()
                        .map(|entry| Complex64::new(entry, 0.0))
                        .collect(),
                    payload_dims: vec![value.len()],
                    axis_classes: Some(vec![0; logical_dims.len()]),
                })
            }
            StorageRepr::StructuredC64(value) => {
                ensure!(
                    value.logical_dims() == logical_dims,
                    "logical dims {:?} do not match structured c64 logical dims {:?}",
                    logical_dims,
                    value.logical_dims()
                );
                let axis_classes = if value.is_dense() {
                    None
                } else {
                    Some(value.axis_classes().to_vec())
                };
                Ok(NativePayload {
                    data: value.payload_col_major_vec(),
                    payload_dims: value.payload_dims().to_vec(),
                    axis_classes,
                })
            }
            StorageRepr::StructuredF64(value) => {
                ensure!(
                    value.logical_dims() == logical_dims,
                    "logical dims {:?} do not match structured f64 logical dims {:?} for promotion",
                    logical_dims,
                    value.logical_dims()
                );
                let axis_classes = if value.is_dense() {
                    None
                } else {
                    Some(value.axis_classes().to_vec())
                };
                Ok(NativePayload {
                    data: value
                        .payload_col_major_vec()
                        .into_iter()
                        .map(|entry| Complex64::new(entry, 0.0))
                        .collect(),
                    payload_dims: value.payload_dims().to_vec(),
                    axis_classes,
                })
            }
        }
    }

    /// Create a new 1D zero-initialized DenseF64 storage with the given size.
    pub fn new_dense_f64(size: usize) -> Self {
        Self::from_dense_f64_col_major(vec![0.0; size], &[size])
            .unwrap_or_else(|err| panic!("Storage::new_dense_f64 failed: {err}"))
    }

    /// Create a new 1D zero-initialized DenseC64 storage with the given size.
    pub fn new_dense_c64(size: usize) -> Self {
        Self::from_dense_c64_col_major(vec![Complex64::new(0.0, 0.0); size], &[size])
            .unwrap_or_else(|err| panic!("Storage::new_dense_c64 failed: {err}"))
    }

    /// Create a new DiagF64 storage with the given diagonal data.
    pub fn new_diag_f64(diag_data: Vec<f64>) -> Self {
        Self::from_diag_f64_col_major(diag_data, 2)
            .unwrap_or_else(|err| panic!("Storage::new_diag_f64 failed: {err}"))
    }

    /// Create a new DiagC64 storage with the given diagonal data.
    pub fn new_diag_c64(diag_data: Vec<Complex64>) -> Self {
        Self::from_diag_c64_col_major(diag_data, 2)
            .unwrap_or_else(|err| panic!("Storage::new_diag_c64 failed: {err}"))
    }

    /// Create dense f64 storage from column-major logical values.
    pub fn from_dense_f64_col_major(data: Vec<f64>, logical_dims: &[usize]) -> Result<Self> {
        Self::validate_dense_len(&data, logical_dims, "dense f64 payload")?;
        Ok(Self::from_repr(StorageRepr::StructuredF64(
            StructuredStorage::from_dense_col_major(data, logical_dims),
        )))
    }

    /// Create dense Complex64 storage from column-major logical values.
    pub fn from_dense_c64_col_major(data: Vec<Complex64>, logical_dims: &[usize]) -> Result<Self> {
        Self::validate_dense_len(&data, logical_dims, "dense c64 payload")?;
        Ok(Self::from_repr(StorageRepr::StructuredC64(
            StructuredStorage::from_dense_col_major(data, logical_dims),
        )))
    }

    /// Create diagonal f64 storage from column-major diagonal payload values.
    pub fn from_diag_f64_col_major(diag_data: Vec<f64>, logical_rank: usize) -> Result<Self> {
        Ok(Self::from_repr(StorageRepr::StructuredF64(
            StructuredStorage::from_diag_col_major(diag_data, logical_rank),
        )))
    }

    /// Create diagonal Complex64 storage from column-major diagonal payload values.
    pub fn from_diag_c64_col_major(diag_data: Vec<Complex64>, logical_rank: usize) -> Result<Self> {
        Ok(Self::from_repr(StorageRepr::StructuredC64(
            StructuredStorage::from_diag_col_major(diag_data, logical_rank),
        )))
    }

    /// Create a new structured f64 storage.
    pub fn new_structured_f64(
        data: Vec<f64>,
        payload_dims: Vec<usize>,
        strides: Vec<isize>,
        axis_classes: Vec<usize>,
    ) -> Result<Self> {
        Ok(Self::from_repr(StorageRepr::StructuredF64(
            StructuredStorage::new(data, payload_dims, strides, axis_classes)?,
        )))
    }

    /// Create a new structured Complex64 storage.
    pub fn new_structured_c64(
        data: Vec<Complex64>,
        payload_dims: Vec<usize>,
        strides: Vec<isize>,
        axis_classes: Vec<usize>,
    ) -> Result<Self> {
        Ok(Self::from_repr(StorageRepr::StructuredC64(
            StructuredStorage::new(data, payload_dims, strides, axis_classes)?,
        )))
    }

    /// Check if this storage is logically dense.
    pub fn is_dense(&self) -> bool {
        match &self.0 {
            StorageRepr::DenseF64(_) | StorageRepr::DenseC64(_) => true,
            StorageRepr::DiagF64(_) | StorageRepr::DiagC64(_) => false,
            StorageRepr::StructuredF64(value) => value.is_dense(),
            StorageRepr::StructuredC64(value) => value.is_dense(),
        }
    }

    /// Check if this storage is a Diag storage type.
    pub fn is_diag(&self) -> bool {
        match &self.0 {
            StorageRepr::DiagF64(_) | StorageRepr::DiagC64(_) => true,
            StorageRepr::DenseF64(_) | StorageRepr::DenseC64(_) => false,
            StorageRepr::StructuredF64(value) => value.is_diag(),
            StorageRepr::StructuredC64(value) => value.is_diag(),
        }
    }

    /// Check if this storage uses f64 scalar type.
    pub fn is_f64(&self) -> bool {
        matches!(
            &self.0,
            StorageRepr::DenseF64(_) | StorageRepr::DiagF64(_) | StorageRepr::StructuredF64(_)
        )
    }

    /// Check if this storage uses Complex64 scalar type.
    pub fn is_c64(&self) -> bool {
        matches!(
            &self.0,
            StorageRepr::DenseC64(_) | StorageRepr::DiagC64(_) | StorageRepr::StructuredC64(_)
        )
    }

    /// Check if this storage uses complex scalar type.
    ///
    /// This is an alias for `is_c64()`.
    pub fn is_complex(&self) -> bool {
        self.is_c64()
    }

    /// Get the length of the storage (number of elements).
    pub fn len(&self) -> usize {
        match &self.0 {
            StorageRepr::DenseF64(v) => v.len(),
            StorageRepr::DenseC64(v) => v.len(),
            StorageRepr::DiagF64(v) => v.len(),
            StorageRepr::DiagC64(v) => v.len(),
            StorageRepr::StructuredF64(v) => v.len(),
            StorageRepr::StructuredC64(v) => v.len(),
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
        match &self.0 {
            StorageRepr::DenseF64(v) => {
                v.as_slice().iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
            }
            StorageRepr::DiagF64(v) => v.as_slice().iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
            StorageRepr::DenseC64(v) => v
                .as_slice()
                .iter()
                .map(|z| z.norm())
                .fold(0.0_f64, f64::max),
            StorageRepr::DiagC64(v) => v
                .as_slice()
                .iter()
                .map(|z| z.norm())
                .fold(0.0_f64, f64::max),
            StorageRepr::StructuredF64(v) => {
                v.data().iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
            }
            StorageRepr::StructuredC64(v) => {
                v.data().iter().map(|z| z.norm()).fold(0.0_f64, f64::max)
            }
        }
    }

    /// Materialize dense logical values as a column-major `f64` buffer.
    pub fn to_dense_f64_col_major_vec(&self, logical_dims: &[usize]) -> Result<Vec<f64>, String> {
        match &self.0 {
            StorageRepr::DenseF64(v) => {
                let expected_len: usize = logical_dims.iter().product();
                if expected_len != v.len() {
                    return Err(format!(
                        "logical dims {:?} (len={expected_len}) do not match DenseF64 len {}",
                        logical_dims,
                        v.len()
                    ));
                }
                Ok(row_major_to_col_major_values(v.as_slice(), logical_dims))
            }
            StorageRepr::DiagF64(v) => Ok(row_major_to_col_major_values(
                &v.to_dense_vec(logical_dims),
                logical_dims,
            )),
            StorageRepr::StructuredF64(v) => {
                let structured_dims = v.logical_dims();
                if structured_dims != logical_dims {
                    return Err(format!(
                        "logical dims {:?} do not match StructuredF64 logical dims {:?}",
                        logical_dims, structured_dims
                    ));
                }
                if let Some(view) = v.dense_col_major_view_if_contiguous() {
                    Ok(view.to_vec())
                } else if v.is_dense() {
                    Ok(v.payload_col_major_vec())
                } else {
                    let native =
                        crate::tenferro_bridge::storage_to_native_tensor(self, logical_dims)
                            .map_err(|err| err.to_string())?;
                    crate::tenferro_bridge::native_tensor_primal_to_dense_f64_col_major(&native)
                        .map_err(|err| err.to_string())
                }
            }
            StorageRepr::DenseC64(_) | StorageRepr::DiagC64(_) | StorageRepr::StructuredC64(_) => {
                Err("expected f64 storage when materializing dense f64 values".to_string())
            }
        }
    }

    /// Materialize dense logical values as a column-major `Complex64` buffer.
    pub fn to_dense_c64_col_major_vec(
        &self,
        logical_dims: &[usize],
    ) -> Result<Vec<Complex64>, String> {
        match &self.0 {
            StorageRepr::DenseC64(v) => {
                let expected_len: usize = logical_dims.iter().product();
                if expected_len != v.len() {
                    return Err(format!(
                        "logical dims {:?} (len={expected_len}) do not match DenseC64 len {}",
                        logical_dims,
                        v.len()
                    ));
                }
                Ok(row_major_to_col_major_values(v.as_slice(), logical_dims))
            }
            StorageRepr::DiagC64(v) => Ok(row_major_to_col_major_values(
                &v.to_dense_vec(logical_dims),
                logical_dims,
            )),
            StorageRepr::StructuredC64(v) => {
                let structured_dims = v.logical_dims();
                if structured_dims != logical_dims {
                    return Err(format!(
                        "logical dims {:?} do not match StructuredC64 logical dims {:?}",
                        logical_dims, structured_dims
                    ));
                }
                if let Some(view) = v.dense_col_major_view_if_contiguous() {
                    Ok(view.to_vec())
                } else if v.is_dense() {
                    Ok(v.payload_col_major_vec())
                } else {
                    let native =
                        crate::tenferro_bridge::storage_to_native_tensor(self, logical_dims)
                            .map_err(|err| err.to_string())?;
                    crate::tenferro_bridge::native_tensor_primal_to_dense_c64_col_major(&native)
                        .map_err(|err| err.to_string())
                }
            }
            StorageRepr::DenseF64(_) | StorageRepr::DiagF64(_) | StorageRepr::StructuredF64(_) => {
                Err("expected Complex64 storage when materializing dense c64 values".to_string())
            }
        }
    }

    /// Convert this storage to dense storage.
    /// For Diag storage, creates a Dense storage with diagonal elements set
    /// and off-diagonal elements as zero.
    /// For Dense storage, returns a copy (clone).
    pub fn to_dense_storage(&self, dims: &[usize]) -> Storage {
        if self.is_f64() {
            let values = self
                .to_dense_f64_col_major_vec(dims)
                .unwrap_or_else(|err| panic!("Storage::to_dense_storage failed: {err}"));
            Storage::from_dense_f64_col_major(values, dims)
                .unwrap_or_else(|err| panic!("Storage::to_dense_storage failed: {err}"))
        } else {
            let values = self
                .to_dense_c64_col_major_vec(dims)
                .unwrap_or_else(|err| panic!("Storage::to_dense_storage failed: {err}"));
            Storage::from_dense_c64_col_major(values, dims)
                .unwrap_or_else(|err| panic!("Storage::to_dense_storage failed: {err}"))
        }
    }

    /// Permute the storage data according to the given permutation.
    ///
    /// Legacy dense kernels use their internal physical shape directly. The
    /// `dims` parameter remains for diagonal payload compatibility.
    pub fn permute_storage(&self, _dims: &[usize], perm: &[usize]) -> Storage {
        match &self.0 {
            StorageRepr::DenseF64(v) => Storage::dense_f64_legacy(v.permute(perm)),
            StorageRepr::DenseC64(v) => Storage::dense_c64_legacy(v.permute(perm)),
            // For Diag storage, permute is trivial: data doesn't change, only index order changes
            StorageRepr::DiagF64(v) => Storage::diag_f64_legacy(v.clone()),
            StorageRepr::DiagC64(v) => Storage::diag_c64_legacy(v.clone()),
            StorageRepr::StructuredF64(v) => Storage::structured_f64(v.permute_logical_axes(perm)),
            StorageRepr::StructuredC64(v) => Storage::structured_c64(v.permute_logical_axes(perm)),
        }
    }

    /// Extract real part from Complex64 storage as f64 storage.
    /// For f64 storage, returns a copy (clone).
    pub fn extract_real_part(&self) -> Storage {
        match &self.0 {
            StorageRepr::DenseF64(v) => {
                // Clone preserves shape
                Storage::dense_f64_legacy(v.clone())
            }
            StorageRepr::DiagF64(d) => {
                Storage::diag_f64_legacy(DiagStorageF64::from_vec(d.as_slice().to_vec()))
            }
            StorageRepr::DenseC64(v) => {
                let dims = v.dims();
                let real_vec: Vec<f64> = v.as_slice().iter().map(|z| z.re).collect();
                Storage::dense_f64_legacy(DenseStorageF64::from_vec_with_shape(real_vec, &dims))
            }
            StorageRepr::DiagC64(d) => {
                let real_vec: Vec<f64> = d.as_slice().iter().map(|z| z.re).collect();
                Storage::diag_f64_legacy(DiagStorageF64::from_vec(real_vec))
            }
            StorageRepr::StructuredF64(v) => Storage::structured_f64(v.clone()),
            StorageRepr::StructuredC64(v) => Storage::structured_f64(v.map_copy(|z| z.re)),
        }
    }

    /// Extract imaginary part from Complex64 storage as f64 storage.
    /// For f64 storage, returns zero storage (will be resized appropriately).
    pub fn extract_imag_part(&self, dims: &[usize]) -> Storage {
        match &self.0 {
            StorageRepr::DenseF64(v) => {
                // For real storage, imaginary part is zero, preserve shape
                let d = v.dims();
                let total_size: usize = d.iter().product();
                Storage::dense_f64_legacy(DenseStorageF64::from_vec_with_shape(
                    vec![0.0; total_size],
                    &d,
                ))
            }
            StorageRepr::DiagF64(_) => {
                // For real diagonal storage, imaginary part is zero
                let mindim_val = mindim(dims);
                Storage::diag_f64_legacy(DiagStorageF64::from_vec(vec![0.0; mindim_val]))
            }
            StorageRepr::DenseC64(v) => {
                let d = v.dims();
                let imag_vec: Vec<f64> = v.as_slice().iter().map(|z| z.im).collect();
                Storage::dense_f64_legacy(DenseStorageF64::from_vec_with_shape(imag_vec, &d))
            }
            StorageRepr::DiagC64(d) => {
                let imag_vec: Vec<f64> = d.as_slice().iter().map(|z| z.im).collect();
                Storage::diag_f64_legacy(DiagStorageF64::from_vec(imag_vec))
            }
            StorageRepr::StructuredF64(v) => Storage::structured_f64(v.map_copy(|_| 0.0)),
            StorageRepr::StructuredC64(v) => Storage::structured_f64(v.map_copy(|z| z.im)),
        }
    }

    /// Convert f64 storage to Complex64 storage (real part only, imaginary part is zero).
    /// For Complex64 storage, returns a clone.
    pub fn to_complex_storage(&self) -> Storage {
        match &self.0 {
            StorageRepr::DenseF64(v) => {
                let dims = v.dims();
                let c64_vec: Vec<Complex64> = v
                    .as_slice()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0))
                    .collect();
                Storage::dense_c64_legacy(DenseStorageC64::from_vec_with_shape(c64_vec, &dims))
            }
            StorageRepr::DiagF64(d) => {
                let c64_vec: Vec<Complex64> = d
                    .as_slice()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0))
                    .collect();
                Storage::diag_c64_legacy(DiagStorageC64::from_vec(c64_vec))
            }
            StorageRepr::DenseC64(v) => {
                // Clone preserves shape
                Storage::dense_c64_legacy(v.clone())
            }
            StorageRepr::DiagC64(d) => {
                Storage::diag_c64_legacy(DiagStorageC64::from_vec(d.as_slice().to_vec()))
            }
            StorageRepr::StructuredF64(v) => {
                Storage::structured_c64(v.map_copy(|x| Complex64::new(x, 0.0)))
            }
            StorageRepr::StructuredC64(v) => Storage::structured_c64(v.clone()),
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
    /// use tensor4all_tensorbackend::Storage;
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)];
    /// let storage = Storage::from_dense_c64_col_major(data, &[2]).unwrap();
    /// let conj_storage = storage.conj();
    ///
    /// // conj(1+2i) = 1-2i, conj(3-4i) = 3+4i
    /// ```
    pub fn conj(&self) -> Self {
        match &self.0 {
            StorageRepr::DenseF64(v) => {
                // Real numbers: conj(x) = x, clone preserves shape
                Storage::dense_f64_legacy(v.clone())
            }
            StorageRepr::DenseC64(v) => {
                let dims = v.dims();
                let conj_vec: Vec<Complex64> = v.as_slice().iter().map(|z| z.conj()).collect();
                Storage::dense_c64_legacy(DenseStorageC64::from_vec_with_shape(conj_vec, &dims))
            }
            StorageRepr::DiagF64(d) => {
                // Real numbers: conj(x) = x
                Storage::diag_f64_legacy(DiagStorageF64::from_vec(d.as_slice().to_vec()))
            }
            StorageRepr::DiagC64(d) => {
                let conj_vec: Vec<Complex64> = d.as_slice().iter().map(|z| z.conj()).collect();
                Storage::diag_c64_legacy(DiagStorageC64::from_vec(conj_vec))
            }
            StorageRepr::StructuredF64(v) => Storage::structured_f64(v.clone()),
            StorageRepr::StructuredC64(v) => Storage::structured_c64(v.map_copy(|z| z.conj())),
        }
    }

    /// Combine two f64 storages into Complex64 storage.
    /// real_storage becomes the real part, imag_storage becomes the imaginary part.
    /// Formula: real + i * imag
    pub fn combine_to_complex(real_storage: &Storage, imag_storage: &Storage) -> Storage {
        match (&real_storage.0, &imag_storage.0) {
            (StorageRepr::DenseF64(real), StorageRepr::DenseF64(imag)) => {
                assert_eq!(real.len(), imag.len(), "Storage lengths must match");
                let dims = real.dims();
                let complex_vec: Vec<Complex64> = real
                    .as_slice()
                    .iter()
                    .zip(imag.as_slice().iter())
                    .map(|(&r, &i)| Complex64::new(r, i))
                    .collect();
                Storage::dense_c64_legacy(DenseStorageC64::from_vec_with_shape(complex_vec, &dims))
            }
            (StorageRepr::DiagF64(real), StorageRepr::DiagF64(imag)) => {
                assert_eq!(real.len(), imag.len(), "Storage lengths must match");
                let complex_vec: Vec<Complex64> = real
                    .as_slice()
                    .iter()
                    .zip(imag.as_slice().iter())
                    .map(|(&r, &i)| Complex64::new(r, i))
                    .collect();
                Storage::diag_c64_legacy(DiagStorageC64::from_vec(complex_vec))
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
        match (&self.0, &other.0) {
            (StorageRepr::DenseF64(a), StorageRepr::DenseF64(b)) => {
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
                Ok(Storage::dense_f64_legacy(
                    DenseStorageF64::from_vec_with_shape(sum_vec, &dims),
                ))
            }
            (StorageRepr::DenseC64(a), StorageRepr::DenseC64(b)) => {
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
                Ok(Storage::dense_c64_legacy(
                    DenseStorageC64::from_vec_with_shape(sum_vec, &dims),
                ))
            }
            (StorageRepr::DiagF64(a), StorageRepr::DiagF64(b)) => {
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
                Ok(Storage::diag_f64_legacy(DiagStorageF64::from_vec(sum_vec)))
            }
            (StorageRepr::DiagC64(a), StorageRepr::DiagC64(b)) => {
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
                Ok(Storage::diag_c64_legacy(DiagStorageC64::from_vec(sum_vec)))
            }
            _ => Err(format!(
                "Storage types must match for addition: {:?} vs {:?}",
                std::mem::discriminant(&self.0),
                std::mem::discriminant(&other.0)
            )),
        }
    }

    /// Try to subtract two storages element-wise.
    ///
    /// Returns an error if the storages have different types or lengths.
    pub fn try_sub(&self, other: &Storage) -> Result<Storage, String> {
        match (&self.0, &other.0) {
            (StorageRepr::DenseF64(a), StorageRepr::DenseF64(b)) => {
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
                Ok(Storage::dense_f64_legacy(
                    DenseStorageF64::from_vec_with_shape(diff_vec, &dims),
                ))
            }
            (StorageRepr::DenseC64(a), StorageRepr::DenseC64(b)) => {
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
                Ok(Storage::dense_c64_legacy(
                    DenseStorageC64::from_vec_with_shape(diff_vec, &dims),
                ))
            }
            (StorageRepr::DiagF64(a), StorageRepr::DiagF64(b)) => {
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
                Ok(Storage::diag_f64_legacy(DiagStorageF64::from_vec(diff_vec)))
            }
            (StorageRepr::DiagC64(a), StorageRepr::DiagC64(b)) => {
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
                Ok(Storage::diag_c64_legacy(DiagStorageC64::from_vec(diff_vec)))
            }
            _ => Err(format!(
                "Storage types must match for subtraction: {:?} vs {:?}",
                std::mem::discriminant(&self.0),
                std::mem::discriminant(&other.0)
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
            || matches!(&self.0, StorageRepr::DenseC64(_) | StorageRepr::DiagC64(_))
            || matches!(&other.0, StorageRepr::DenseC64(_) | StorageRepr::DiagC64(_));

        if needs_complex {
            // Promote everything to complex
            let a_c: Complex64 = a.clone().into();
            let b_c: Complex64 = b.clone().into();

            let (result, dims): (Vec<Complex64>, Vec<usize>) = match (&self.0, &other.0) {
                (StorageRepr::DenseF64(x), StorageRepr::DenseF64(y)) => (
                    x.as_slice()
                        .iter()
                        .zip(y.as_slice().iter())
                        .map(|(&xi, &yi)| {
                            a_c * Complex64::new(xi, 0.0) + b_c * Complex64::new(yi, 0.0)
                        })
                        .collect(),
                    x.dims(),
                ),
                (StorageRepr::DenseF64(x), StorageRepr::DenseC64(y)) => (
                    x.as_slice()
                        .iter()
                        .zip(y.as_slice().iter())
                        .map(|(&xi, &yi)| a_c * Complex64::new(xi, 0.0) + b_c * yi)
                        .collect(),
                    x.dims(),
                ),
                (StorageRepr::DenseC64(x), StorageRepr::DenseF64(y)) => (
                    x.as_slice()
                        .iter()
                        .zip(y.as_slice().iter())
                        .map(|(&xi, &yi)| a_c * xi + b_c * Complex64::new(yi, 0.0))
                        .collect(),
                    x.dims(),
                ),
                (StorageRepr::DenseC64(x), StorageRepr::DenseC64(y)) => (
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
                        std::mem::discriminant(&self.0),
                        std::mem::discriminant(&other.0)
                    ))
                }
            };
            Ok(Storage::dense_c64_legacy(
                DenseStorageC64::from_vec_with_shape(result, &dims),
            ))
        } else {
            // All real
            if !a.is_real() || !b.is_real() {
                return Err(format!(
                    "expected real scalars in real axpby branch: a={a}, b={b}"
                ));
            }
            let a_f = a.real();
            let b_f = b.real();

            match (&self.0, &other.0) {
                (StorageRepr::DenseF64(x), StorageRepr::DenseF64(y)) => {
                    let dims = x.dims();
                    let result: Vec<f64> = x
                        .as_slice()
                        .iter()
                        .zip(y.as_slice().iter())
                        .map(|(&xi, &yi)| a_f * xi + b_f * yi)
                        .collect();
                    Ok(Storage::dense_f64_legacy(
                        DenseStorageF64::from_vec_with_shape(result, &dims),
                    ))
                }
                (StorageRepr::DiagF64(x), StorageRepr::DiagF64(y)) => {
                    let result: Vec<f64> = x
                        .as_slice()
                        .iter()
                        .zip(y.as_slice().iter())
                        .map(|(&xi, &yi)| a_f * xi + b_f * yi)
                        .collect();
                    Ok(Storage::diag_f64_legacy(DiagStorageF64::from_vec(result)))
                }
                _ => Err(format!(
                    "axpby not supported for storage types: {:?} vs {:?}",
                    std::mem::discriminant(&self.0),
                    std::mem::discriminant(&other.0)
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

    if matches!(
        &storage_a.0,
        StorageRepr::StructuredF64(_) | StorageRepr::StructuredC64(_)
    ) || matches!(
        &storage_b.0,
        StorageRepr::StructuredF64(_) | StorageRepr::StructuredC64(_)
    ) {
        return crate::tenferro_bridge::contract_storage_native(
            storage_a,
            dims_a,
            axes_a,
            storage_b,
            dims_b,
            axes_b,
            result_dims,
        )
        .unwrap_or_else(|err| panic!("contract_storage structured fallback failed: {err}"));
    }

    match (&storage_a.0, &storage_b.0) {
        // Same-type legacy dense kernels carry their physical shape internally.
        (StorageRepr::DenseF64(a), StorageRepr::DenseF64(b)) => {
            Storage::dense_f64_legacy(a.contract(axes_a, b, axes_b))
        }
        (StorageRepr::DenseC64(a), StorageRepr::DenseC64(b)) => {
            Storage::dense_c64_legacy(a.contract(axes_a, b, axes_b))
        }
        (StorageRepr::DenseF64(a), StorageRepr::DenseC64(b)) => {
            let promoted_a = promote_dense_to_c64(a);
            Storage::dense_c64_legacy(promoted_a.contract(axes_a, b, axes_b))
        }
        (StorageRepr::DenseC64(a), StorageRepr::DenseF64(b)) => {
            let promoted_b = promote_dense_to_c64(b);
            Storage::dense_c64_legacy(a.contract(axes_a, &promoted_b, axes_b))
        }
        // DiagTensor × DiagTensor contraction
        (StorageRepr::DiagF64(a), StorageRepr::DiagF64(b)) => a.contract_diag_diag(
            dims_a,
            b,
            dims_b,
            result_dims,
            |v| Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(v, result_dims)),
            |v| Storage::diag_f64_legacy(DiagStorage::from_vec(v)),
        ),
        (StorageRepr::DiagC64(a), StorageRepr::DiagC64(b)) => a.contract_diag_diag(
            dims_a,
            b,
            dims_b,
            result_dims,
            |v| Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(v, result_dims)),
            |v| Storage::diag_c64_legacy(DiagStorage::from_vec(v)),
        ),

        // Mixed Diag types: promote both to complex and recurse once.
        (StorageRepr::DiagF64(_), StorageRepr::DiagC64(_))
        | (StorageRepr::DiagC64(_), StorageRepr::DiagF64(_)) => {
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
        (StorageRepr::DiagF64(diag), StorageRepr::DenseF64(dense)) => {
            diag.contract_diag_dense(dims_a, axes_a, dense, dims_b, axes_b, result_dims, |v| {
                Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(v, result_dims))
            })
        }
        (StorageRepr::DiagC64(diag), StorageRepr::DenseC64(dense)) => {
            diag.contract_diag_dense(dims_a, axes_a, dense, dims_b, axes_b, result_dims, |v| {
                Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(v, result_dims))
            })
        }

        // DenseTensor × DiagTensor: use generic helper
        (StorageRepr::DenseF64(dense), StorageRepr::DiagF64(diag)) => contract_dense_diag_impl(
            dense,
            dims_a,
            axes_a,
            diag,
            dims_b,
            axes_b,
            result_dims,
            |v, dims| Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(v, dims)),
            |s, perm| s.permute_storage(&[], perm),
        ),
        (StorageRepr::DenseC64(dense), StorageRepr::DiagC64(diag)) => contract_dense_diag_impl(
            dense,
            dims_a,
            axes_a,
            diag,
            dims_b,
            axes_b,
            result_dims,
            |v, dims| Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(v, dims)),
            |s, perm| s.permute_storage(&[], perm),
        ),

        // Mixed Diag/Dense with type promotion: promote f64 to Complex64
        (StorageRepr::DiagF64(diag_f64), StorageRepr::DenseC64(dense_c64)) => {
            // Diag<f64> × Dense<C64>: promote Diag to C64
            let diag_c64 = promote_diag_to_c64(diag_f64);
            diag_c64.contract_diag_dense(
                dims_a,
                axes_a,
                dense_c64,
                dims_b,
                axes_b,
                result_dims,
                |v| Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(v, result_dims)),
            )
        }
        (StorageRepr::DenseC64(dense_c64), StorageRepr::DiagF64(diag_f64)) => {
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
                |v, dims| Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(v, dims)),
                |s, perm| s.permute_storage(&[], perm),
            )
        }
        (StorageRepr::DiagC64(diag_c64), StorageRepr::DenseF64(dense_f64)) => {
            // Diag<C64> × Dense<f64>: promote Dense to C64
            let dense_c64 = promote_dense_to_c64(dense_f64);
            diag_c64.contract_diag_dense(
                dims_a,
                axes_a,
                &dense_c64,
                dims_b,
                axes_b,
                result_dims,
                |v| Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(v, result_dims)),
            )
        }
        (StorageRepr::DenseF64(dense_f64), StorageRepr::DiagC64(diag_c64)) => {
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
                |v, dims| Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(v, dims)),
                |s, perm| s.permute_storage(&[], perm),
            )
        }
        (StorageRepr::StructuredF64(_), _)
        | (StorageRepr::StructuredC64(_), _)
        | (_, StorageRepr::StructuredF64(_))
        | (_, StorageRepr::StructuredC64(_)) => {
            unreachable!("structured storage cases are handled by the native fallback above")
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
        match (&self.0, &rhs.0) {
            (StorageRepr::DenseF64(a), StorageRepr::DenseF64(b)) => {
                assert_eq!(a.len(), b.len(), "Storage lengths must match for addition");
                let dims = a.dims();
                let sum_vec: Vec<f64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Storage::dense_f64_legacy(DenseStorageF64::from_vec_with_shape(sum_vec, &dims))
            }
            (StorageRepr::DenseC64(a), StorageRepr::DenseC64(b)) => {
                assert_eq!(a.len(), b.len(), "Storage lengths must match for addition");
                let dims = a.dims();
                let sum_vec: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Storage::dense_c64_legacy(DenseStorageC64::from_vec_with_shape(sum_vec, &dims))
            }
            (StorageRepr::DiagF64(a), StorageRepr::DiagF64(b)) => {
                assert_eq!(a.len(), b.len(), "Storage lengths must match for addition");
                let sum_vec: Vec<f64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Storage::diag_f64_legacy(DiagStorageF64::from_vec(sum_vec))
            }
            (StorageRepr::DiagC64(a), StorageRepr::DiagC64(b)) => {
                assert_eq!(a.len(), b.len(), "Storage lengths must match for addition");
                let sum_vec: Vec<Complex64> = a
                    .as_slice()
                    .iter()
                    .zip(b.as_slice().iter())
                    .map(|(&x, &y)| x + y)
                    .collect();
                Storage::diag_c64_legacy(DiagStorageC64::from_vec(sum_vec))
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
        match &self.0 {
            StorageRepr::DenseF64(v) => {
                let dims = v.dims();
                let scaled_vec: Vec<f64> = v.as_slice().iter().map(|&x| x * scalar).collect();
                Storage::dense_f64_legacy(DenseStorageF64::from_vec_with_shape(scaled_vec, &dims))
            }
            StorageRepr::DenseC64(v) => {
                let dims = v.dims();
                let scaled_vec: Vec<Complex64> = v
                    .as_slice()
                    .iter()
                    .map(|&z| z * Complex64::new(scalar, 0.0))
                    .collect();
                Storage::dense_c64_legacy(DenseStorageC64::from_vec_with_shape(scaled_vec, &dims))
            }
            StorageRepr::DiagF64(d) => {
                let scaled_vec: Vec<f64> = d.as_slice().iter().map(|&x| x * scalar).collect();
                Storage::diag_f64_legacy(DiagStorageF64::from_vec(scaled_vec))
            }
            StorageRepr::DiagC64(d) => {
                let scaled_vec: Vec<Complex64> = d
                    .as_slice()
                    .iter()
                    .map(|&z| z * Complex64::new(scalar, 0.0))
                    .collect();
                Storage::diag_c64_legacy(DiagStorageC64::from_vec(scaled_vec))
            }
            StorageRepr::StructuredF64(v) => Storage::structured_f64(v.map_copy(|x| x * scalar)),
            StorageRepr::StructuredC64(v) => {
                Storage::structured_c64(v.map_copy(|z| z * Complex64::new(scalar, 0.0)))
            }
        }
    }
}

/// Multiply storage by a scalar (Complex64).
impl Mul<Complex64> for &Storage {
    type Output = Storage;

    fn mul(self, scalar: Complex64) -> Self::Output {
        match &self.0 {
            StorageRepr::DenseF64(v) => {
                // Promote f64 to Complex64
                let dims = v.dims();
                let scaled_vec: Vec<Complex64> = v
                    .as_slice()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0) * scalar)
                    .collect();
                Storage::dense_c64_legacy(DenseStorageC64::from_vec_with_shape(scaled_vec, &dims))
            }
            StorageRepr::DenseC64(v) => {
                let dims = v.dims();
                let scaled_vec: Vec<Complex64> = v.as_slice().iter().map(|&z| z * scalar).collect();
                Storage::dense_c64_legacy(DenseStorageC64::from_vec_with_shape(scaled_vec, &dims))
            }
            StorageRepr::DiagF64(d) => {
                // Promote f64 to Complex64
                let scaled_vec: Vec<Complex64> = d
                    .as_slice()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0) * scalar)
                    .collect();
                Storage::diag_c64_legacy(DiagStorageC64::from_vec(scaled_vec))
            }
            StorageRepr::DiagC64(d) => {
                let scaled_vec: Vec<Complex64> = d.as_slice().iter().map(|&z| z * scalar).collect();
                Storage::diag_c64_legacy(DiagStorageC64::from_vec(scaled_vec))
            }
            StorageRepr::StructuredF64(v) => {
                Storage::structured_c64(v.map_copy(|x| Complex64::new(x, 0.0) * scalar))
            }
            StorageRepr::StructuredC64(v) => Storage::structured_c64(v.map_copy(|z| z * scalar)),
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
mod tests;
