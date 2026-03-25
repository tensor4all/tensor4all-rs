//! Simple dense multi-dimensional array with row-major storage.
//!
//! This is a lightweight local helper used internally to replace `mdarray::DTensor`.
//! It supports arbitrary element types and dynamic rank with row-major indexing.

use std::ops::{Index, IndexMut};

/// A dense multi-dimensional array stored in row-major (C) order.
///
/// `DenseArray<T>` stores data in a flat `Vec<T>` with row-major strides.
/// It supports arbitrary element types (no trait bounds on `T` for the struct itself).
#[derive(Clone, Debug)]
pub struct DenseArray<T> {
    data: Vec<T>,
    dims: Vec<usize>,
    /// Row-major strides: strides[i] = product of dims[i+1..].
    strides: Vec<usize>,
}

impl<T: Clone> DenseArray<T> {
    /// Create a new array filled with `value`.
    pub fn from_elem(dims: &[usize], value: T) -> Self {
        let total: usize = dims.iter().product();
        let strides = compute_row_major_strides(dims);
        Self {
            data: vec![value; total],
            dims: dims.to_vec(),
            strides,
        }
    }
}

impl<T> DenseArray<T> {
    /// Return the dimensions (shape) of the array.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Compute the flat index for a given multi-dimensional index.
    fn flat_index(&self, idx: &[usize]) -> usize {
        debug_assert_eq!(idx.len(), self.dims.len());
        idx.iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum()
    }
}

impl<T, const N: usize> Index<[usize; N]> for DenseArray<T> {
    type Output = T;
    fn index(&self, idx: [usize; N]) -> &T {
        let flat = self.flat_index(&idx);
        &self.data[flat]
    }
}

impl<T, const N: usize> IndexMut<[usize; N]> for DenseArray<T> {
    fn index_mut(&mut self, idx: [usize; N]) -> &mut T {
        let flat = self.flat_index(&idx);
        &mut self.data[flat]
    }
}

/// Compute row-major strides for given dimensions.
fn compute_row_major_strides(dims: &[usize]) -> Vec<usize> {
    let rank = dims.len();
    let mut strides = vec![0usize; rank];
    if rank > 0 {
        strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    }
    strides
}
