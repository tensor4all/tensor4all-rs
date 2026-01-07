//! Block data types using mdarray.
//!
//! This module provides type aliases and helper functions for block data,
//! built on top of mdarray's Tensor and Slice types.

use std::marker::PhantomData;
use std::sync::Arc;

use mdarray::{DSlice, DTensor, Dense, Strided};

use crate::scalar::Scalar;

/// Owned 2D block data.
///
/// Uses mdarray's DTensor for efficient storage and operations.
/// Data is stored in row-major (C) order.
pub type BlockTensor2<T> = DTensor<T, 2>;

/// View over 2D block data (dense layout).
pub type BlockSlice2<'a, T> = &'a DSlice<T, 2, Dense>;

/// View over 2D block data (strided layout, for permuted views).
pub type BlockSliceStrided2<'a, T> = &'a DSlice<T, 2, Strided>;

/// Owned block data wrapped with Arc for sharing.
#[derive(Debug, Clone)]
pub struct BlockData<T: Scalar> {
    tensor: Arc<BlockTensor2<T>>,
    _marker: PhantomData<T>,
}

impl<T: Scalar> BlockData<T> {
    /// Create a new block from data and shape.
    ///
    /// # Arguments
    /// * `data` - Row-major data
    /// * `shape` - [rows, cols]
    ///
    /// # Panics
    /// Panics if data length doesn't match shape.
    pub fn new(data: Vec<T>, shape: [usize; 2]) -> Self {
        let expected_len = shape[0] * shape[1];
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        // Create tensor using from_fn to ensure row-major order
        let tensor: BlockTensor2<T> = DTensor::<T, 2>::from_fn(shape, |idx| {
            let linear = idx[0] * shape[1] + idx[1];
            data[linear]
        });

        Self {
            tensor: Arc::new(tensor),
            _marker: PhantomData,
        }
    }

    /// Create a scalar block (1x1).
    pub fn scalar(value: T) -> Self {
        Self::new(vec![value], [1, 1])
    }

    /// Create a block from an existing DTensor.
    pub fn from_tensor(tensor: BlockTensor2<T>) -> Self {
        Self {
            tensor: Arc::new(tensor),
            _marker: PhantomData,
        }
    }

    /// Get the shape of this block.
    pub fn shape(&self) -> [usize; 2] {
        let s = self.tensor.shape();
        [s.0, s.1]
    }

    /// Get the number of rows.
    pub fn nrows(&self) -> usize {
        self.tensor.shape().0
    }

    /// Get the number of columns.
    pub fn ncols(&self) -> usize {
        self.tensor.shape().1
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.nrows() * self.ncols()
    }

    /// Check if block is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a view (slice) of this block.
    pub fn as_slice(&self) -> &DSlice<T, 2, Dense> {
        self.tensor.as_ref()
    }

    /// Get the underlying tensor reference.
    pub fn as_tensor(&self) -> &BlockTensor2<T> {
        &self.tensor
    }

    /// Get element at [i, j].
    pub fn get(&self, idx: [usize; 2]) -> T {
        self.tensor[idx]
    }

    /// Create a transposed view (no data copy).
    ///
    /// Returns an owned BlockData with transposed data.
    /// For zero-copy transpose, use `as_slice().t()` directly.
    pub fn transpose(&self) -> BlockData<T> {
        let [m, n] = self.shape();
        let transposed: BlockTensor2<T> =
            DTensor::<T, 2>::from_fn([n, m], |idx| self.tensor[[idx[1], idx[0]]]);
        Self::from_tensor(transposed)
    }

    /// Convert to Vec in row-major order.
    pub fn to_vec(&self) -> Vec<T> {
        let [m, n] = self.shape();
        let mut result = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                result.push(self.tensor[[i, j]]);
            }
        }
        result
    }

    /// Add another block to this one (element-wise).
    ///
    /// # Panics
    /// Panics if shapes don't match.
    pub fn add(&self, other: &BlockData<T>) -> BlockData<T> {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch in add");
        let [m, n] = self.shape();
        let result: BlockTensor2<T> =
            DTensor::<T, 2>::from_fn([m, n], |idx| self.tensor[idx] + other.tensor[idx]);
        Self::from_tensor(result)
    }
}

// For convenient creation from Vec with shape
impl<T: Scalar> From<(Vec<T>, [usize; 2])> for BlockData<T> {
    fn from((data, shape): (Vec<T>, [usize; 2])) -> Self {
        Self::new(data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn test_block_data_new_generic<T: Scalar>() {
        let data: Vec<T> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(T::from_f64)
            .collect();
        let block = BlockData::<T>::new(data, [2, 3]);

        assert_eq!(block.shape(), [2, 3]);
        assert_eq!(block.nrows(), 2);
        assert_eq!(block.ncols(), 3);
        assert_eq!(block.len(), 6);
    }

    #[test]
    fn test_block_data_new_f64() {
        test_block_data_new_generic::<f64>();
    }

    #[test]
    fn test_block_data_new_c64() {
        test_block_data_new_generic::<Complex64>();
    }

    fn test_block_data_scalar_generic<T: Scalar>() {
        let block = BlockData::<T>::scalar(T::from_f64(42.0));

        assert_eq!(block.shape(), [1, 1]);
        assert_eq!(block.len(), 1);
        assert_eq!(block.get([0, 0]), T::from_f64(42.0));
    }

    #[test]
    fn test_block_data_scalar_f64() {
        test_block_data_scalar_generic::<f64>();
    }

    #[test]
    fn test_block_data_scalar_c64() {
        test_block_data_scalar_generic::<Complex64>();
    }

    fn test_block_data_get_generic<T: Scalar>() {
        // Row-major: [[1, 2, 3], [4, 5, 6]]
        let data: Vec<T> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(T::from_f64)
            .collect();
        let block = BlockData::<T>::new(data, [2, 3]);

        assert_eq!(block.get([0, 0]), T::from_f64(1.0));
        assert_eq!(block.get([0, 2]), T::from_f64(3.0));
        assert_eq!(block.get([1, 0]), T::from_f64(4.0));
        assert_eq!(block.get([1, 2]), T::from_f64(6.0));
    }

    #[test]
    fn test_block_data_get_f64() {
        test_block_data_get_generic::<f64>();
    }

    #[test]
    fn test_block_data_get_c64() {
        test_block_data_get_generic::<Complex64>();
    }

    fn test_block_data_transpose_generic<T: Scalar>() {
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        let data: Vec<T> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(T::from_f64)
            .collect();
        let block = BlockData::<T>::new(data, [2, 3]);
        let transposed = block.transpose();

        assert_eq!(transposed.shape(), [3, 2]);
        assert_eq!(transposed.get([0, 0]), T::from_f64(1.0));
        assert_eq!(transposed.get([0, 1]), T::from_f64(4.0));
        assert_eq!(transposed.get([1, 0]), T::from_f64(2.0));
        assert_eq!(transposed.get([2, 1]), T::from_f64(6.0));
    }

    #[test]
    fn test_block_data_transpose_f64() {
        test_block_data_transpose_generic::<f64>();
    }

    #[test]
    fn test_block_data_transpose_c64() {
        test_block_data_transpose_generic::<Complex64>();
    }

    fn test_block_data_add_generic<T: Scalar>() {
        let a = BlockData::<T>::new(
            vec![1.0, 2.0, 3.0, 4.0]
                .into_iter()
                .map(T::from_f64)
                .collect(),
            [2, 2],
        );
        let b = BlockData::<T>::new(
            vec![10.0, 20.0, 30.0, 40.0]
                .into_iter()
                .map(T::from_f64)
                .collect(),
            [2, 2],
        );
        let c = a.add(&b);

        assert_eq!(c.get([0, 0]), T::from_f64(11.0));
        assert_eq!(c.get([0, 1]), T::from_f64(22.0));
        assert_eq!(c.get([1, 0]), T::from_f64(33.0));
        assert_eq!(c.get([1, 1]), T::from_f64(44.0));
    }

    #[test]
    fn test_block_data_add_f64() {
        test_block_data_add_generic::<f64>();
    }

    #[test]
    fn test_block_data_add_c64() {
        test_block_data_add_generic::<Complex64>();
    }

    fn test_block_data_to_vec_generic<T: Scalar>() {
        let data: Vec<T> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(T::from_f64)
            .collect();
        let block = BlockData::<T>::new(data.clone(), [2, 3]);

        assert_eq!(block.to_vec(), data);
    }

    #[test]
    fn test_block_data_to_vec_f64() {
        test_block_data_to_vec_generic::<f64>();
    }

    #[test]
    fn test_block_data_to_vec_c64() {
        test_block_data_to_vec_generic::<Complex64>();
    }

    fn test_as_slice_generic<T: Scalar>() {
        let block = BlockData::<T>::new(
            vec![1.0, 2.0, 3.0, 4.0]
                .into_iter()
                .map(T::from_f64)
                .collect(),
            [2, 2],
        );
        let slice = block.as_slice();

        // Access via mdarray slice
        assert_eq!(slice[[0, 0]], T::from_f64(1.0));
        assert_eq!(slice[[1, 1]], T::from_f64(4.0));
    }

    #[test]
    fn test_as_slice_f64() {
        test_as_slice_generic::<f64>();
    }

    #[test]
    fn test_as_slice_c64() {
        test_as_slice_generic::<Complex64>();
    }

    #[test]
    fn test_complex_specific() {
        // Test complex-specific behavior
        let z1 = Complex64::new(1.0, 2.0);
        let z2 = Complex64::new(3.0, 4.0);
        let block = BlockData::<Complex64>::new(vec![z1, z2], [1, 2]);

        assert_eq!(block.get([0, 0]), z1);
        assert_eq!(block.get([0, 1]), z2);
    }
}
