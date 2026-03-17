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
mod tests {
    use super::*;

    /// Helper to extract f64 data from storage
    fn extract_f64(storage: &Storage) -> Vec<f64> {
        match storage.repr() {
            StorageRepr::DenseF64(ds) => ds.as_slice().to_vec(),
            StorageRepr::StructuredF64(ds) => ds.data().to_vec(),
            _ => panic!("Expected f64 dense-compatible storage"),
        }
    }

    /// Helper to extract Complex64 data from storage
    fn extract_c64(storage: &Storage) -> Vec<Complex64> {
        match storage.repr() {
            StorageRepr::DenseC64(ds) => ds.as_slice().to_vec(),
            StorageRepr::StructuredC64(ds) => ds.data().to_vec(),
            _ => panic!("Expected c64 dense-compatible storage"),
        }
    }

    // ===== Legacy diagonal kernel generic tests =====

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
        let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
        let dense =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0; 9], &[3, 3]));

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
        let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
        // Dense = [[1,2], [3,4], [5,6]] in row-major
        let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(
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
        let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));

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
        let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0]));
        let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(
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
        let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0]));
        let dense = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
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
        let diag = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
        ]));
        let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(
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
        let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0],
            &[2, 2],
        ));
        let diag = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![
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
        let dense = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, 2.0),
                Complex64::new(3.0, 3.0),
                Complex64::new(4.0, 4.0),
            ],
            &[2, 2],
        ));
        let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0]));

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
        let diag1 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
        let diag2 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![4.0, 5.0, 6.0]));

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
        let diag1 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
        let diag2 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![4.0, 5.0, 6.0]));

        let result = contract_storage(&diag1, &[3, 3], &[1], &diag2, &[3, 3], &[0], &[3, 3]);

        // Result is element-wise product: [1*4, 2*5, 3*6] = [4, 10, 18]
        match result.repr() {
            StorageRepr::DiagF64(d) => {
                assert_eq!(d.as_slice(), &[4.0, 10.0, 18.0]);
            }
            _ => panic!("Expected DiagF64"),
        }
    }

    // ===== Type inspection tests =====

    #[test]
    fn test_is_f64() {
        let dense_f64 =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
        let dense_c64 = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 0.0)],
            &[1],
        ));
        let diag_f64 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0]));
        let diag_c64 =
            Storage::diag_c64_legacy(DiagStorage::from_vec(vec![Complex64::new(1.0, 0.0)]));

        assert!(dense_f64.is_f64());
        assert!(!dense_c64.is_f64());
        assert!(diag_f64.is_f64());
        assert!(!diag_c64.is_f64());
    }

    #[test]
    fn test_is_c64() {
        let dense_f64 =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
        let dense_c64 = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 0.0)],
            &[1],
        ));
        let diag_f64 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0]));
        let diag_c64 =
            Storage::diag_c64_legacy(DiagStorage::from_vec(vec![Complex64::new(1.0, 0.0)]));

        assert!(!dense_f64.is_c64());
        assert!(dense_c64.is_c64());
        assert!(!diag_f64.is_c64());
        assert!(diag_c64.is_c64());
    }

    #[test]
    fn test_is_complex() {
        let dense_f64 =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
        let dense_c64 = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 0.0)],
            &[1],
        ));
        let diag_f64 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0]));
        let diag_c64 =
            Storage::diag_c64_legacy(DiagStorage::from_vec(vec![Complex64::new(1.0, 0.0)]));

        // is_complex is an alias for is_c64
        assert!(!dense_f64.is_complex());
        assert!(dense_c64.is_complex());
        assert!(!diag_f64.is_complex());
        assert!(diag_c64.is_complex());
    }

    // ===== Legacy dense kernel tests =====

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

    // ===== Legacy diagonal kernel tests =====

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
    fn test_diag_contract_diag_diag_scalar_result() {
        // All indices contracted: inner product
        let d1 = DiagStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let d2 = DiagStorage::from_vec(vec![4.0, 5.0, 6.0]);
        let result = d1.contract_diag_diag(
            &[3, 3],
            &d2,
            &[3, 3],
            &[],
            |v| Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(v, &[])),
            |v| Storage::diag_f64_legacy(DiagStorage::from_vec(v)),
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
            |v| Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(v, &[2, 2])),
            |v| Storage::diag_f64_legacy(DiagStorage::from_vec(v)),
        );
        match result.repr() {
            StorageRepr::DiagF64(d) => {
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
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(v, &[2, 3]))
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
        let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
        let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0]));
        assert!(!dense.is_diag());
        assert!(diag.is_diag());
    }

    #[test]
    fn structured_storage_rejects_noncanonical_axis_classes() {
        let err = StructuredStorage::<f64>::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            vec![1, 2],
            vec![1, 0, 0],
        )
        .unwrap_err();

        assert!(err.to_string().contains("canonical"));
    }

    #[test]
    fn structured_storage_column_major_helpers_cover_contiguous_padded_and_empty_payloads() {
        let dense =
            StructuredStorage::from_dense_col_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(dense.logical_dims(), vec![2, 3]);
        assert!(dense.is_dense());
        assert!(!dense.is_diag());
        assert_eq!(
            dense.dense_col_major_view_if_contiguous().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            dense.payload_col_major_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        let padded = StructuredStorage::new(
            vec![10.0, 20.0, -1.0, 30.0, 40.0, -1.0, 50.0, 60.0],
            vec![2, 3],
            vec![1, 3],
            vec![0, 1],
        )
        .unwrap();
        assert!(padded.dense_col_major_view_if_contiguous().is_none());
        assert_eq!(
            padded.payload_col_major_vec(),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        );

        let empty = StructuredStorage::from_dense_col_major(Vec::<f64>::new(), &[0, 3]);
        assert!(empty.is_empty());
        assert_eq!(empty.payload_col_major_vec(), Vec::<f64>::new());
    }

    #[test]
    fn structured_storage_permute_and_map_copy_preserve_metadata() {
        let storage = StructuredStorage::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![1, 2],
            vec![0, 1, 0],
        )
        .unwrap();
        assert_eq!(storage.logical_dims(), vec![2, 3, 2]);

        let permuted = storage.permute_logical_axes(&[0, 2, 1]);
        assert_eq!(permuted.axis_classes(), &[0, 0, 1]);
        assert_eq!(permuted.logical_dims(), vec![2, 2, 3]);

        let mapped = permuted.map_copy(|x| x * 10.0);
        assert_eq!(mapped.data(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        assert_eq!(mapped.payload_dims(), &[2, 3]);
        assert_eq!(mapped.strides(), &[1, 2]);
        assert_eq!(mapped.axis_classes(), &[0, 0, 1]);
    }

    #[test]
    fn structured_storage_validates_payload_rank_and_required_len() {
        let rank_err = StructuredStorage::<f64>::new(vec![1.0, 2.0], vec![2], vec![1], vec![0, 1])
            .unwrap_err();
        assert!(rank_err.to_string().contains("payload rank"));

        let len_err =
            StructuredStorage::<f64>::new(vec![1.0, 2.0], vec![2, 2], vec![1, 3], vec![0, 1])
                .unwrap_err();
        assert!(len_err.to_string().contains("required len"));

        let scalar_diag = StructuredStorage::from_diag_col_major(vec![42.0], 0);
        assert_eq!(scalar_diag.payload_dims(), &[] as &[usize]);
        assert_eq!(scalar_diag.logical_rank(), 0);
        assert!(scalar_diag.is_dense());
        assert!(!scalar_diag.is_diag());
        assert_eq!(scalar_diag.payload_col_major_vec(), vec![42.0]);
    }

    // ===== Storage len / is_empty =====

    #[test]
    fn test_storage_len_is_empty() {
        let dense =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0, 2.0], &[2]));
        assert_eq!(dense.len(), 2);
        assert!(!dense.is_empty());

        let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![]));
        assert_eq!(diag.len(), 0);
        assert!(diag.is_empty());
    }

    // ===== Storage zero-constructor tests =====

    #[test]
    fn test_storage_new_dense_f64() {
        let s = Storage::new_dense_f64(3);
        assert_eq!(s.len(), 3);
        assert!(s.is_f64());
        let data = extract_f64(&s);
        assert_eq!(data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_storage_new_dense_c64() {
        let s = Storage::new_dense_c64(2);
        assert_eq!(s.len(), 2);
        assert!(s.is_c64());
    }

    #[test]
    fn test_storage_from_dense_f64_col_major_zeros_with_shape() {
        let s = Storage::from_dense_f64_col_major(vec![0.0; 6], &[2, 3]).unwrap();
        assert_eq!(s.len(), 6);
    }

    #[test]
    fn test_storage_from_dense_c64_col_major_zeros_with_shape() {
        let s =
            Storage::from_dense_c64_col_major(vec![Complex64::new(0.0, 0.0); 6], &[3, 2]).unwrap();
        assert_eq!(s.len(), 6);
    }

    // ===== SumFromStorage tests =====

    #[test]
    fn test_sum_from_storage_f64() {
        let s =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]));
        let sum: f64 = f64::sum_from_storage(&s);
        assert!((sum - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_sum_from_storage_diag_f64() {
        let s = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![10.0, 20.0]));
        let sum: f64 = f64::sum_from_storage(&s);
        assert!((sum - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_storage_sum_f64_method() {
        let s =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]));
        assert!((s.sum_f64() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_storage_sum_c64_method() {
        let s = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
            &[2],
        ));
        let sum = s.sum_c64();
        assert!((sum - Complex64::new(4.0, 6.0)).norm() < 1e-10);
    }

    #[test]
    fn test_storage_max_abs_and_to_dense_storage_cover_complex_and_diag() {
        let dense_c64 = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, -1.0)],
            &[2],
        ));
        assert!((dense_c64.max_abs() - 5.0).abs() < 1e-10);
        match dense_c64.to_dense_storage(&[2]).repr() {
            StorageRepr::StructuredC64(ds) => assert_eq!(
                ds.payload_col_major_vec().as_slice(),
                &[Complex64::new(3.0, 4.0), Complex64::new(1.0, -1.0)]
            ),
            StorageRepr::DenseC64(ds) => assert_eq!(
                ds.as_slice(),
                &[Complex64::new(3.0, 4.0), Complex64::new(1.0, -1.0)]
            ),
            other => panic!("expected DenseC64, got {other:?}"),
        }

        let diag_c64 = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![
            Complex64::new(0.0, 2.0),
            Complex64::new(3.0, 4.0),
        ]));
        assert!((diag_c64.max_abs() - 5.0).abs() < 1e-10);
        match diag_c64.to_dense_storage(&[2, 2]).repr() {
            StorageRepr::StructuredC64(ds) => {
                assert_eq!(
                    ds.payload_col_major_vec().as_slice(),
                    &[
                        Complex64::new(0.0, 2.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(3.0, 4.0),
                    ]
                );
            }
            StorageRepr::DenseC64(ds) => {
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
        let dense_c64 = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)],
            &[2],
        ));
        match dense_c64.extract_real_part().repr() {
            StorageRepr::DenseF64(ds) => assert_eq!(ds.as_slice(), &[1.0, 3.0]),
            StorageRepr::StructuredF64(ds) => {
                assert_eq!(ds.payload_col_major_vec().as_slice(), &[1.0, 3.0])
            }
            other => panic!("expected DenseF64, got {other:?}"),
        }
        match dense_c64.extract_imag_part(&[2]).repr() {
            StorageRepr::DenseF64(ds) => assert_eq!(ds.as_slice(), &[-2.0, 4.0]),
            StorageRepr::StructuredF64(ds) => {
                assert_eq!(ds.payload_col_major_vec().as_slice(), &[-2.0, 4.0])
            }
            other => panic!("expected DenseF64, got {other:?}"),
        }
        match dense_c64.conj().repr() {
            StorageRepr::DenseC64(ds) => {
                assert_eq!(
                    ds.as_slice(),
                    &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)]
                )
            }
            StorageRepr::StructuredC64(ds) => {
                assert_eq!(
                    ds.payload_col_major_vec().as_slice(),
                    &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)]
                )
            }
            other => panic!("expected DenseC64, got {other:?}"),
        }

        let diag_f64 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![2.0, -1.0]));
        match diag_f64.extract_imag_part(&[2, 2]).repr() {
            StorageRepr::DiagF64(ds) => assert_eq!(ds.as_slice(), &[0.0, 0.0]),
            StorageRepr::StructuredF64(ds) => {
                assert_eq!(ds.payload_col_major_vec().as_slice(), &[0.0, 0.0])
            }
            other => panic!("expected DiagF64, got {other:?}"),
        }
        match diag_f64.to_complex_storage().repr() {
            StorageRepr::DiagC64(ds) => {
                assert_eq!(
                    ds.as_slice(),
                    &[Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.0)]
                )
            }
            StorageRepr::StructuredC64(ds) => {
                assert_eq!(
                    ds.payload_col_major_vec().as_slice(),
                    &[Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.0)]
                )
            }
            other => panic!("expected DiagC64, got {other:?}"),
        }
        let real =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0, 2.0], &[2]));
        let imag =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![0.5, -1.5], &[2]));
        match Storage::combine_to_complex(&real, &imag).repr() {
            StorageRepr::DenseC64(ds) => {
                assert_eq!(
                    ds.as_slice(),
                    &[Complex64::new(1.0, 0.5), Complex64::new(2.0, -1.5)]
                )
            }
            StorageRepr::StructuredC64(ds) => {
                assert_eq!(
                    ds.payload_col_major_vec().as_slice(),
                    &[Complex64::new(1.0, 0.5), Complex64::new(2.0, -1.5)]
                )
            }
            other => panic!("expected DenseC64, got {other:?}"),
        }
    }

    #[test]
    fn test_storage_try_add_and_try_sub_cover_all_variants_and_errors() {
        let dense_f64_a =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0, 2.0], &[2]));
        let dense_f64_b =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![3.0, -1.0], &[2]));
        match dense_f64_a.try_add(&dense_f64_b).unwrap().repr() {
            StorageRepr::DenseF64(ds) => assert_eq!(ds.as_slice(), &[4.0, 1.0]),
            other => panic!("expected DenseF64, got {other:?}"),
        }
        match dense_f64_a.try_sub(&dense_f64_b).unwrap().repr() {
            StorageRepr::DenseF64(ds) => assert_eq!(ds.as_slice(), &[-2.0, 3.0]),
            other => panic!("expected DenseF64, got {other:?}"),
        }

        let dense_c64_a = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, -2.0)],
            &[2],
        ));
        let dense_c64_b = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(-1.0, 0.5), Complex64::new(3.0, 1.0)],
            &[2],
        ));
        assert!(matches!(
            dense_c64_a.try_add(&dense_c64_b).unwrap().repr(),
            StorageRepr::DenseC64(_)
        ));
        assert!(matches!(
            dense_c64_a.try_sub(&dense_c64_b).unwrap().repr(),
            StorageRepr::DenseC64(_)
        ));

        let diag_f64_a = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0]));
        let diag_f64_b = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![0.5, -3.0]));
        assert!(matches!(
            diag_f64_a.try_add(&diag_f64_b).unwrap().repr(),
            StorageRepr::DiagF64(_)
        ));
        assert!(matches!(
            diag_f64_a.try_sub(&diag_f64_b).unwrap().repr(),
            StorageRepr::DiagF64(_)
        ));

        let diag_c64_a = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(0.0, 2.0),
        ]));
        let diag_c64_b = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![
            Complex64::new(0.5, 0.5),
            Complex64::new(-3.0, 1.0),
        ]));
        assert!(matches!(
            diag_c64_a.try_add(&diag_c64_b).unwrap().repr(),
            StorageRepr::DiagC64(_)
        ));
        assert!(matches!(
            diag_c64_a.try_sub(&diag_c64_b).unwrap().repr(),
            StorageRepr::DiagC64(_)
        ));

        let mismatched_len =
            Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
        let err = dense_f64_a.try_add(&mismatched_len).unwrap_err();
        assert!(err.contains("Storage lengths must match for addition"));
        let err = dense_f64_a.try_sub(&mismatched_len).unwrap_err();
        assert!(err.contains("Storage lengths must match for subtraction"));

        let mismatched_type = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            &[2],
        ));
        let err = dense_f64_a.try_add(&mismatched_type).unwrap_err();
        assert!(err.contains("Storage types must match for addition"));
        let err = dense_f64_a.try_sub(&mismatched_type).unwrap_err();
        assert!(err.contains("Storage types must match for subtraction"));
    }
}
