//! Fixed-rank tensor type backed by a flat `Vec<T>` with row-major layout.
//!
//! This module provides a simple tensor type that replaces the `mdarray::DTensor`
//! dependency for the simplett crate.

use std::ops::{Index, IndexMut};

/// Rank-N tensor backed by a flat `Vec<T>` with row-major layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T, const N: usize> {
    data: Vec<T>,
    dims: [usize; N],
    /// Precomputed strides for row-major layout.
    strides: [usize; N],
}

/// 2D tensor (matrix).
pub type Tensor2<T> = Tensor<T, 2>;

/// 3D tensor.
pub type Tensor3<T> = Tensor<T, 3>;

/// 4D tensor.
pub type Tensor4<T> = Tensor<T, 4>;

/// Compute row-major strides from dimensions.
fn compute_strides<const N: usize>(dims: &[usize; N]) -> [usize; N] {
    let mut strides = [0usize; N];
    if N == 0 {
        return strides;
    }
    strides[N - 1] = 1;
    for i in (0..N - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

impl<T, const N: usize> Tensor<T, N> {
    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty (zero elements).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Dimension along `axis`.
    pub fn dim(&self, axis: usize) -> usize {
        self.dims[axis]
    }

    /// All dimensions.
    pub fn dims(&self) -> &[usize; N] {
        &self.dims
    }

    /// View the underlying data as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// View the underlying data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Convert a multi-index to a flat offset (row-major).
    #[inline]
    fn offset(&self, idx: &[usize; N]) -> usize {
        let mut offset = 0;
        for ((&i, &d), &s) in idx.iter().zip(self.dims.iter()).zip(self.strides.iter()) {
            debug_assert!(i < d, "index out of bounds");
            offset += i * s;
        }
        offset
    }

    /// Iterate over all elements in row-major order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Iterate mutably over all elements in row-major order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<T: Clone, const N: usize> Tensor<T, N> {
    /// Create a tensor filled with `value`.
    pub fn from_elem(dims: [usize; N], value: T) -> Self {
        let total: usize = dims.iter().product();
        let strides = compute_strides(&dims);
        Self {
            data: vec![value; total],
            dims,
            strides,
        }
    }
}

impl<T, const N: usize> Tensor<T, N> {
    /// Create a tensor by applying `f` to each multi-index (row-major order).
    pub fn from_fn(dims: [usize; N], mut f: impl FnMut([usize; N]) -> T) -> Self {
        let total: usize = dims.iter().product();
        let strides = compute_strides(&dims);
        let mut data = Vec::with_capacity(total);

        // Iterate over all multi-indices in row-major order.
        let mut idx = [0usize; N];
        for _ in 0..total {
            data.push(f(idx));
            // Increment the multi-index (rightmost index increments first).
            for k in (0..N).rev() {
                idx[k] += 1;
                if idx[k] < dims[k] {
                    break;
                }
                idx[k] = 0;
            }
        }

        Self {
            data,
            dims,
            strides,
        }
    }
}

impl<T, const N: usize> Index<[usize; N]> for Tensor<T, N> {
    type Output = T;

    fn index(&self, idx: [usize; N]) -> &T {
        let offset = self.offset(&idx);
        &self.data[offset]
    }
}

impl<T, const N: usize> IndexMut<[usize; N]> for Tensor<T, N> {
    fn index_mut(&mut self, idx: [usize; N]) -> &mut T {
        let offset = self.offset(&idx);
        &mut self.data[offset]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor2_from_elem() {
        let t: Tensor2<f64> = Tensor2::from_elem([3, 4], 0.0);
        assert_eq!(t.len(), 12);
        assert_eq!(t.dim(0), 3);
        assert_eq!(t.dim(1), 4);
        assert_eq!(t[[0, 0]], 0.0);
    }

    #[test]
    fn test_tensor2_indexing() {
        let mut t: Tensor2<f64> = Tensor2::from_elem([2, 3], 0.0);
        t[[0, 0]] = 1.0;
        t[[0, 1]] = 2.0;
        t[[0, 2]] = 3.0;
        t[[1, 0]] = 4.0;
        t[[1, 1]] = 5.0;
        t[[1, 2]] = 6.0;
        assert_eq!(t[[0, 0]], 1.0);
        assert_eq!(t[[1, 2]], 6.0);
        // Row-major: data layout is [1, 2, 3, 4, 5, 6]
        assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor3_from_fn() {
        let t: Tensor3<usize> = Tensor3::from_fn([2, 3, 4], |[i, j, k]| i * 100 + j * 10 + k);
        assert_eq!(t[[0, 0, 0]], 0);
        assert_eq!(t[[1, 2, 3]], 123);
        assert_eq!(t[[0, 1, 2]], 12);
        assert_eq!(t.dim(0), 2);
        assert_eq!(t.dim(1), 3);
        assert_eq!(t.dim(2), 4);
        assert_eq!(t.len(), 24);
    }

    #[test]
    fn test_tensor4_from_fn() {
        let t: Tensor4<usize> =
            Tensor4::from_fn([2, 3, 4, 5], |[i, j, k, l]| i * 1000 + j * 100 + k * 10 + l);
        assert_eq!(t[[1, 2, 3, 4]], 1234);
        assert_eq!(t.dim(0), 2);
        assert_eq!(t.dim(1), 3);
        assert_eq!(t.dim(2), 4);
        assert_eq!(t.dim(3), 5);
        assert_eq!(t.len(), 120);
    }

    #[test]
    fn test_iter() {
        let t: Tensor2<i32> = Tensor2::from_fn([2, 3], |[i, j]| (i * 3 + j) as i32);
        let collected: Vec<i32> = t.iter().copied().collect();
        assert_eq!(collected, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_dims() {
        let t: Tensor3<f64> = Tensor3::from_elem([2, 3, 4], 1.0);
        assert_eq!(t.dims(), &[2, 3, 4]);
    }
}
