//! Blocked array and view types.

use std::collections::HashMap;
use std::marker::PhantomData;

use mdarray::{DSlice, Dense};

use crate::block_data::BlockData;
use crate::partition::{block_linear_index, block_multi_index, BlockIndex, BlockPartition};
use crate::scalar::Scalar;

/// A blocked multi-dimensional array (owns data).
///
/// The array is partitioned along each axis, creating a grid of blocks.
/// Only non-zero blocks are stored in memory (sparse representation).
#[derive(Debug, Clone)]
pub struct BlockedArray<T: Scalar> {
    /// Block partition for each axis.
    partitions: Vec<BlockPartition>,
    /// Non-zero blocks stored in a HashMap.
    /// Key: linear block index, Value: block data.
    blocks: HashMap<usize, BlockData<T>>,
    _marker: PhantomData<T>,
}

impl<T: Scalar> BlockedArray<T> {
    /// Create an empty blocked array with given partitions.
    pub fn new(partitions: Vec<BlockPartition>) -> Self {
        Self {
            partitions,
            blocks: HashMap::new(),
            _marker: PhantomData,
        }
    }

    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.partitions.len()
    }

    /// Get the total shape (sum of block sizes per axis).
    pub fn shape(&self) -> Vec<usize> {
        self.partitions.iter().map(|p| p.total_dim()).collect()
    }

    /// Get the partitions.
    pub fn partitions(&self) -> &[BlockPartition] {
        &self.partitions
    }

    /// Get the number of blocks per axis.
    pub fn num_blocks(&self) -> Vec<usize> {
        self.partitions.iter().map(|p| p.num_blocks()).collect()
    }

    /// Get the total number of blocks (including zero blocks).
    pub fn total_blocks(&self) -> usize {
        self.num_blocks().iter().product()
    }

    /// Get the number of non-zero (stored) blocks.
    pub fn num_nonzero_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the shape of a specific block.
    pub fn block_shape(&self, block_idx: &BlockIndex) -> [usize; 2] {
        assert_eq!(block_idx.len(), 2, "Only 2D blocks are supported");
        [
            self.partitions[0].block_size(block_idx[0]),
            self.partitions[1].block_size(block_idx[1]),
        ]
    }

    /// Get a block reference (returns None for zero blocks).
    pub fn get_block(&self, block_idx: &BlockIndex) -> Option<&BlockData<T>> {
        let linear = block_linear_index(block_idx, &self.num_blocks());
        self.blocks.get(&linear)
    }

    /// Get a block as mdarray slice (returns None for zero blocks).
    pub fn get_block_slice(&self, block_idx: &BlockIndex) -> Option<&DSlice<T, 2, Dense>> {
        self.get_block(block_idx).map(|b| b.as_slice())
    }

    /// Set a block.
    pub fn set_block(&mut self, block_idx: BlockIndex, data: BlockData<T>) {
        let expected_shape = self.block_shape(&block_idx);
        assert_eq!(
            data.shape(),
            expected_shape,
            "Block shape {:?} does not match expected {:?}",
            data.shape(),
            expected_shape
        );

        let linear = block_linear_index(&block_idx, &self.num_blocks());
        self.blocks.insert(linear, data);
    }

    /// Accumulate into a block (add to existing or create new).
    ///
    /// Used in matrix multiplication to accumulate partial results.
    pub fn accumulate_block(&mut self, block_idx: BlockIndex, data: BlockData<T>) {
        let linear = block_linear_index(&block_idx, &self.num_blocks());

        if let Some(existing) = self.blocks.get(&linear) {
            // Add to existing block
            let sum = existing.add(&data);
            self.blocks.insert(linear, sum);
        } else {
            // Insert new block
            let expected_shape = self.block_shape(&block_idx);
            assert_eq!(
                data.shape(),
                expected_shape,
                "Block shape {:?} does not match expected {:?}",
                data.shape(),
                expected_shape
            );
            self.blocks.insert(linear, data);
        }
    }

    /// Remove a block (make it zero).
    pub fn remove_block(&mut self, block_idx: &BlockIndex) -> Option<BlockData<T>> {
        let linear = block_linear_index(block_idx, &self.num_blocks());
        self.blocks.remove(&linear)
    }

    /// Iterate over non-zero blocks.
    pub fn iter_blocks(&self) -> impl Iterator<Item = (BlockIndex, &BlockData<T>)> {
        let num_blocks = self.num_blocks();
        self.blocks.iter().map(move |(&linear, data)| {
            let block_idx = block_multi_index(linear, &num_blocks);
            (block_idx, data)
        })
    }

    /// Create a view over this array.
    pub fn view(&self) -> BlockedView<'_, T> {
        BlockedView {
            source: self,
            transposed: false,
        }
    }

    /// Create a transposed view (for 2D arrays).
    pub fn transpose(&self) -> BlockedView<'_, T> {
        assert_eq!(self.rank(), 2, "Transpose requires 2D array");
        BlockedView {
            source: self,
            transposed: true,
        }
    }

    /// Compute sparsity ratio (stored elements / total elements).
    pub fn sparsity(&self) -> f64 {
        let stored: usize = self.blocks.values().map(|b| b.len()).sum();
        let total: usize = self.shape().iter().product();
        if total == 0 {
            0.0
        } else {
            stored as f64 / total as f64
        }
    }
}

/// View over a BlockedArray (borrowed, possibly transposed).
#[derive(Debug, Clone)]
pub struct BlockedView<'a, T: Scalar> {
    /// Reference to the source array.
    source: &'a BlockedArray<T>,
    /// Whether the view is transposed.
    transposed: bool,
}

impl<'a, T: Scalar> BlockedView<'a, T> {
    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.source.rank()
    }

    /// Get the effective shape (after transposition).
    pub fn shape(&self) -> Vec<usize> {
        let orig_shape = self.source.shape();
        if self.transposed {
            vec![orig_shape[1], orig_shape[0]]
        } else {
            orig_shape
        }
    }

    /// Get the effective partitions (after transposition).
    pub fn partitions(&self) -> Vec<BlockPartition> {
        let orig_parts = self.source.partitions();
        if self.transposed {
            vec![orig_parts[1].clone(), orig_parts[0].clone()]
        } else {
            orig_parts.to_vec()
        }
    }

    /// Get the number of blocks per axis (after transposition).
    pub fn num_blocks(&self) -> Vec<usize> {
        let orig_num = self.source.num_blocks();
        if self.transposed {
            vec![orig_num[1], orig_num[0]]
        } else {
            orig_num
        }
    }

    /// Get a block with transposition applied.
    ///
    /// The block index is in the transposed coordinate system.
    /// Returns the block data (transposed if the view is transposed).
    pub fn get_block(&self, block_idx: &BlockIndex) -> Option<BlockData<T>> {
        let source_idx = if self.transposed {
            vec![block_idx[1], block_idx[0]]
        } else {
            block_idx.clone()
        };

        self.source.get_block(&source_idx).map(|b| {
            if self.transposed {
                b.transpose()
            } else {
                b.clone()
            }
        })
    }

    /// Iterate over non-zero blocks (with transposition applied).
    pub fn iter_blocks(&self) -> impl Iterator<Item = (BlockIndex, BlockData<T>)> + '_ {
        self.source.iter_blocks().map(move |(orig_idx, data)| {
            if self.transposed {
                let transposed_idx = vec![orig_idx[1], orig_idx[0]];
                (transposed_idx, data.transpose())
            } else {
                (orig_idx, data.clone())
            }
        })
    }

    /// Further transpose the view.
    pub fn transpose(&self) -> BlockedView<'a, T> {
        assert_eq!(self.rank(), 2, "Transpose requires 2D view");
        BlockedView {
            source: self.source,
            transposed: !self.transposed,
        }
    }

    /// Materialize to owned BlockedArray.
    pub fn to_owned(&self) -> BlockedArray<T> {
        if !self.transposed {
            return self.source.clone();
        }

        let new_partitions = self.partitions();
        let mut result = BlockedArray::new(new_partitions);

        for (idx, data) in self.iter_blocks() {
            result.set_block(idx, data);
        }

        result
    }
}

/// Trait for types that can act as blocked arrays (owned or view).
pub trait BlockedArrayLike<T: Scalar> {
    /// Get the rank.
    fn rank(&self) -> usize;

    /// Get the shape.
    fn shape(&self) -> Vec<usize>;

    /// Get the partitions.
    fn partitions(&self) -> Vec<BlockPartition>;

    /// Get the number of blocks per axis.
    fn num_blocks(&self) -> Vec<usize>;

    /// Get a block (returns owned BlockData).
    fn get_block(&self, block_idx: &BlockIndex) -> Option<BlockData<T>>;

    /// Iterate over non-zero blocks, returning (index, data) pairs.
    ///
    /// Returns a Vec to maintain trait object safety.
    fn iter_nonzero_blocks(&self) -> Vec<(BlockIndex, BlockData<T>)>;
}

impl<T: Scalar> BlockedArrayLike<T> for BlockedArray<T> {
    fn rank(&self) -> usize {
        self.rank()
    }

    fn shape(&self) -> Vec<usize> {
        self.shape()
    }

    fn partitions(&self) -> Vec<BlockPartition> {
        self.partitions.clone()
    }

    fn num_blocks(&self) -> Vec<usize> {
        self.num_blocks()
    }

    fn get_block(&self, block_idx: &BlockIndex) -> Option<BlockData<T>> {
        BlockedArray::get_block(self, block_idx).cloned()
    }

    fn iter_nonzero_blocks(&self) -> Vec<(BlockIndex, BlockData<T>)> {
        self.iter_blocks()
            .map(|(idx, data)| (idx, data.clone()))
            .collect()
    }
}

impl<'a, T: Scalar> BlockedArrayLike<T> for BlockedView<'a, T> {
    fn rank(&self) -> usize {
        self.rank()
    }

    fn shape(&self) -> Vec<usize> {
        self.shape()
    }

    fn partitions(&self) -> Vec<BlockPartition> {
        self.partitions()
    }

    fn num_blocks(&self) -> Vec<usize> {
        self.num_blocks()
    }

    fn get_block(&self, block_idx: &BlockIndex) -> Option<BlockData<T>> {
        BlockedView::get_block(self, block_idx)
    }

    fn iter_nonzero_blocks(&self) -> Vec<(BlockIndex, BlockData<T>)> {
        self.iter_blocks().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn test_blocked_array_new_generic<T: Scalar>() {
        let partitions = vec![
            BlockPartition::new(vec![2, 3]),
            BlockPartition::new(vec![4, 5]),
        ];
        let arr = BlockedArray::<T>::new(partitions);

        assert_eq!(arr.rank(), 2);
        assert_eq!(arr.shape(), vec![5, 9]);
        assert_eq!(arr.num_blocks(), vec![2, 2]);
        assert_eq!(arr.total_blocks(), 4);
        assert_eq!(arr.num_nonzero_blocks(), 0);
    }

    #[test]
    fn test_blocked_array_new_f64() {
        test_blocked_array_new_generic::<f64>();
    }

    #[test]
    fn test_blocked_array_new_c64() {
        test_blocked_array_new_generic::<Complex64>();
    }

    fn test_blocked_array_set_get_generic<T: Scalar>() {
        let partitions = vec![
            BlockPartition::new(vec![2, 3]),
            BlockPartition::new(vec![4, 5]),
        ];
        let mut arr = BlockedArray::<T>::new(partitions);

        // Set block [0, 0] (shape 2x4)
        let data = BlockData::<T>::new(vec![T::from_f64(1.0); 8], [2, 4]);
        arr.set_block(vec![0, 0], data);

        // Set block [1, 1] (shape 3x5)
        let data = BlockData::<T>::new(vec![T::from_f64(2.0); 15], [3, 5]);
        arr.set_block(vec![1, 1], data);

        assert_eq!(arr.num_nonzero_blocks(), 2);

        // Get blocks
        let block00 = arr.get_block(&vec![0, 0]).unwrap();
        assert_eq!(block00.shape(), [2, 4]);
        assert_eq!(block00.get([0, 0]), T::from_f64(1.0));

        let block11 = arr.get_block(&vec![1, 1]).unwrap();
        assert_eq!(block11.shape(), [3, 5]);
        assert_eq!(block11.get([0, 0]), T::from_f64(2.0));

        // Zero block
        assert!(arr.get_block(&vec![0, 1]).is_none());
    }

    #[test]
    fn test_blocked_array_set_get_f64() {
        test_blocked_array_set_get_generic::<f64>();
    }

    #[test]
    fn test_blocked_array_set_get_c64() {
        test_blocked_array_set_get_generic::<Complex64>();
    }

    fn test_blocked_view_transpose_generic<T: Scalar>() {
        let partitions = vec![
            BlockPartition::new(vec![2, 3]),
            BlockPartition::new(vec![4, 5]),
        ];
        let mut arr = BlockedArray::<T>::new(partitions);

        // Set block [0, 1] (shape 2x5)
        let data = BlockData::<T>::new(
            (0..10).map(|i| T::from_f64(i as f64)).collect(),
            [2, 5],
        );
        arr.set_block(vec![0, 1], data);

        // Create transposed view
        let view = arr.transpose();
        assert_eq!(view.shape(), vec![9, 5]);
        assert_eq!(view.num_blocks(), vec![2, 2]);

        // In transposed view, block [1, 0] corresponds to original [0, 1]
        let block = view.get_block(&vec![1, 0]).unwrap();
        assert_eq!(block.shape(), [5, 2]); // Transposed shape

        // Check transposed element access
        // Original [0, 1][0, 0] = 0.0 -> Transposed [1, 0][0, 0]
        assert_eq!(block.get([0, 0]), T::from_f64(0.0));
        // Original [0, 1][0, 1] = 1.0 -> Transposed [1, 0][1, 0]
        assert_eq!(block.get([1, 0]), T::from_f64(1.0));
    }

    #[test]
    fn test_blocked_view_transpose_f64() {
        test_blocked_view_transpose_generic::<f64>();
    }

    #[test]
    fn test_blocked_view_transpose_c64() {
        test_blocked_view_transpose_generic::<Complex64>();
    }

    fn test_blocked_view_to_owned_generic<T: Scalar>() {
        let partitions = vec![
            BlockPartition::new(vec![2, 3]),
            BlockPartition::new(vec![4, 5]),
        ];
        let mut arr = BlockedArray::<T>::new(partitions);

        let data = BlockData::<T>::new(
            (0..10).map(|i| T::from_f64(i as f64)).collect(),
            [2, 5],
        );
        arr.set_block(vec![0, 1], data);

        // Materialize transposed view
        let transposed = arr.transpose().to_owned();
        assert_eq!(transposed.shape(), vec![9, 5]);

        // Block [1, 0] should now have transposed data
        let block = transposed.get_block(&vec![1, 0]).unwrap();
        assert_eq!(block.shape(), [5, 2]);
    }

    #[test]
    fn test_blocked_view_to_owned_f64() {
        test_blocked_view_to_owned_generic::<f64>();
    }

    #[test]
    fn test_blocked_view_to_owned_c64() {
        test_blocked_view_to_owned_generic::<Complex64>();
    }

    fn test_accumulate_block_generic<T: Scalar>() {
        let partitions = vec![
            BlockPartition::uniform(2, 2),
            BlockPartition::uniform(2, 2),
        ];
        let mut arr = BlockedArray::<T>::new(partitions);

        // First accumulation
        let data1 = BlockData::<T>::new(
            vec![1.0, 2.0, 3.0, 4.0]
                .into_iter()
                .map(T::from_f64)
                .collect(),
            [2, 2],
        );
        arr.accumulate_block(vec![0, 0], data1);

        // Second accumulation
        let data2 = BlockData::<T>::new(
            vec![10.0, 20.0, 30.0, 40.0]
                .into_iter()
                .map(T::from_f64)
                .collect(),
            [2, 2],
        );
        arr.accumulate_block(vec![0, 0], data2);

        let block = arr.get_block(&vec![0, 0]).unwrap();
        assert_eq!(block.get([0, 0]), T::from_f64(11.0));
        assert_eq!(block.get([0, 1]), T::from_f64(22.0));
        assert_eq!(block.get([1, 0]), T::from_f64(33.0));
        assert_eq!(block.get([1, 1]), T::from_f64(44.0));
    }

    #[test]
    fn test_accumulate_block_f64() {
        test_accumulate_block_generic::<f64>();
    }

    #[test]
    fn test_accumulate_block_c64() {
        test_accumulate_block_generic::<Complex64>();
    }
}
