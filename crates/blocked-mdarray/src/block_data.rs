//! Block data types using mdarray.
//!
//! This module provides type aliases and helper functions for block data,
//! built on top of mdarray's Tensor and Slice types.

use std::sync::Arc;

use mdarray::{DTensor, DSlice, Dense, Strided};

/// Owned 2D block data.
///
/// Uses mdarray's DTensor for efficient storage and operations.
/// Data is stored in row-major (C) order.
pub type BlockTensor2 = DTensor<f64, 2>;

/// View over 2D block data (dense layout).
pub type BlockSlice2<'a> = &'a DSlice<f64, 2, Dense>;

/// View over 2D block data (strided layout, for permuted views).
pub type BlockSliceStrided2<'a> = &'a DSlice<f64, 2, Strided>;

/// Owned block data wrapped with Arc for sharing.
#[derive(Debug, Clone)]
pub struct BlockData {
    tensor: Arc<BlockTensor2>,
}

impl BlockData {
    /// Create a new block from data and shape.
    ///
    /// # Arguments
    /// * `data` - Row-major data
    /// * `shape` - [rows, cols]
    ///
    /// # Panics
    /// Panics if data length doesn't match shape.
    pub fn new(data: Vec<f64>, shape: [usize; 2]) -> Self {
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
        let tensor: BlockTensor2 = DTensor::<f64, 2>::from_fn(shape, |idx| {
            let linear = idx[0] * shape[1] + idx[1];
            data[linear]
        });

        Self {
            tensor: Arc::new(tensor),
        }
    }

    /// Create a scalar block (1×1).
    pub fn scalar(value: f64) -> Self {
        Self::new(vec![value], [1, 1])
    }

    /// Create a block from an existing DTensor.
    pub fn from_tensor(tensor: BlockTensor2) -> Self {
        Self {
            tensor: Arc::new(tensor),
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
    pub fn as_slice(&self) -> &DSlice<f64, 2, Dense> {
        self.tensor.as_ref()
    }

    /// Get the underlying tensor reference.
    pub fn as_tensor(&self) -> &BlockTensor2 {
        &self.tensor
    }

    /// Get element at [i, j].
    pub fn get(&self, idx: [usize; 2]) -> f64 {
        self.tensor[idx]
    }

    /// Create a transposed view (no data copy).
    ///
    /// Returns an owned BlockData with transposed data.
    /// For zero-copy transpose, use `as_slice().t()` directly.
    pub fn transpose(&self) -> BlockData {
        let [m, n] = self.shape();
        let transposed: BlockTensor2 = DTensor::<f64, 2>::from_fn([n, m], |idx| self.tensor[[idx[1], idx[0]]]);
        Self::from_tensor(transposed)
    }

    /// Convert to Vec in row-major order.
    pub fn to_vec(&self) -> Vec<f64> {
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
    pub fn add(&self, other: &BlockData) -> BlockData {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch in add");
        let [m, n] = self.shape();
        let result: BlockTensor2 = DTensor::<f64, 2>::from_fn([m, n], |idx| {
            self.tensor[idx] + other.tensor[idx]
        });
        Self::from_tensor(result)
    }
}

// For convenient creation from Vec with shape
impl From<(Vec<f64>, [usize; 2])> for BlockData {
    fn from((data, shape): (Vec<f64>, [usize; 2])) -> Self {
        Self::new(data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_data_new() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let block = BlockData::new(data, [2, 3]);

        assert_eq!(block.shape(), [2, 3]);
        assert_eq!(block.nrows(), 2);
        assert_eq!(block.ncols(), 3);
        assert_eq!(block.len(), 6);
    }

    #[test]
    fn test_block_data_scalar() {
        let block = BlockData::scalar(42.0);

        assert_eq!(block.shape(), [1, 1]);
        assert_eq!(block.len(), 1);
        assert_eq!(block.get([0, 0]), 42.0);
    }

    #[test]
    fn test_block_data_get() {
        // Row-major: [[1, 2, 3], [4, 5, 6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let block = BlockData::new(data, [2, 3]);

        assert_eq!(block.get([0, 0]), 1.0);
        assert_eq!(block.get([0, 2]), 3.0);
        assert_eq!(block.get([1, 0]), 4.0);
        assert_eq!(block.get([1, 2]), 6.0);
    }

    #[test]
    fn test_block_data_transpose() {
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let block = BlockData::new(data, [2, 3]);
        let transposed = block.transpose();

        assert_eq!(transposed.shape(), [3, 2]);
        assert_eq!(transposed.get([0, 0]), 1.0);
        assert_eq!(transposed.get([0, 1]), 4.0);
        assert_eq!(transposed.get([1, 0]), 2.0);
        assert_eq!(transposed.get([2, 1]), 6.0);
    }

    #[test]
    fn test_block_data_add() {
        let a = BlockData::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b = BlockData::new(vec![10.0, 20.0, 30.0, 40.0], [2, 2]);
        let c = a.add(&b);

        assert_eq!(c.get([0, 0]), 11.0);
        assert_eq!(c.get([0, 1]), 22.0);
        assert_eq!(c.get([1, 0]), 33.0);
        assert_eq!(c.get([1, 1]), 44.0);
    }

    #[test]
    fn test_block_data_to_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let block = BlockData::new(data.clone(), [2, 3]);

        assert_eq!(block.to_vec(), data);
    }

    #[test]
    fn test_as_slice() {
        let block = BlockData::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let slice = block.as_slice();

        // Access via mdarray slice
        assert_eq!(slice[[0, 0]], 1.0);
        assert_eq!(slice[[1, 1]], 4.0);
    }
}
