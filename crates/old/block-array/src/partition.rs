//! Block partition for array axes.

use std::ops::Range;

/// Partition of an axis into blocks.
///
/// Each axis of a blocked array has a partition that defines how
/// the axis is divided into blocks. For example, an axis of size 10
/// could be partitioned into blocks of sizes [3, 4, 3].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockPartition {
    /// Size of each block.
    block_sizes: Vec<usize>,
    /// Cumulative offsets: [0, s0, s0+s1, ..., total_dim].
    offsets: Vec<usize>,
}

impl BlockPartition {
    /// Create a new partition from block sizes.
    ///
    /// # Example
    /// ```
    /// use block_array::BlockPartition;
    /// let p = BlockPartition::new(vec![3, 4, 3]);
    /// assert_eq!(p.num_blocks(), 3);
    /// assert_eq!(p.total_dim(), 10);
    /// ```
    pub fn new(block_sizes: Vec<usize>) -> Self {
        let mut offsets = Vec::with_capacity(block_sizes.len() + 1);
        offsets.push(0);
        let mut cumsum = 0;
        for &size in &block_sizes {
            cumsum += size;
            offsets.push(cumsum);
        }
        Self {
            block_sizes,
            offsets,
        }
    }

    /// Create a uniform partition with equal block sizes.
    ///
    /// # Example
    /// ```
    /// use block_array::BlockPartition;
    /// let p = BlockPartition::uniform(4, 3);  // 3 blocks of size 4
    /// assert_eq!(p.num_blocks(), 3);
    /// assert_eq!(p.total_dim(), 12);
    /// ```
    pub fn uniform(block_size: usize, num_blocks: usize) -> Self {
        Self::new(vec![block_size; num_blocks])
    }

    /// Create a trivial partition (single block containing the entire axis).
    ///
    /// # Example
    /// ```
    /// use block_array::BlockPartition;
    /// let p = BlockPartition::trivial(10);
    /// assert_eq!(p.num_blocks(), 1);
    /// assert_eq!(p.block_size(0), 10);
    /// ```
    pub fn trivial(total_dim: usize) -> Self {
        Self::new(vec![total_dim])
    }

    /// Get the number of blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.block_sizes.len()
    }

    /// Get the total dimension (sum of all block sizes).
    #[inline]
    pub fn total_dim(&self) -> usize {
        *self.offsets.last().unwrap_or(&0)
    }

    /// Get the size of a specific block.
    ///
    /// # Panics
    /// Panics if `block_idx >= num_blocks()`.
    #[inline]
    pub fn block_size(&self, block_idx: usize) -> usize {
        self.block_sizes[block_idx]
    }

    /// Get the range of indices for a specific block.
    ///
    /// # Panics
    /// Panics if `block_idx >= num_blocks()`.
    #[inline]
    pub fn block_range(&self, block_idx: usize) -> Range<usize> {
        self.offsets[block_idx]..self.offsets[block_idx + 1]
    }

    /// Get the starting offset of a specific block.
    #[inline]
    pub fn block_offset(&self, block_idx: usize) -> usize {
        self.offsets[block_idx]
    }

    /// Get a reference to the block sizes.
    #[inline]
    pub fn block_sizes(&self) -> &[usize] {
        &self.block_sizes
    }

    /// Get a reference to the offsets.
    #[inline]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }
}

/// Multi-dimensional block index.
pub type BlockIndex = Vec<usize>;

/// Compute linear block index from multi-dimensional block index.
///
/// Uses row-major (C-order) indexing.
pub fn block_linear_index(block_idx: &BlockIndex, num_blocks: &[usize]) -> usize {
    let mut linear = 0;
    let mut stride = 1;
    for i in (0..block_idx.len()).rev() {
        linear += block_idx[i] * stride;
        stride *= num_blocks[i];
    }
    linear
}

/// Compute multi-dimensional block index from linear index.
///
/// Uses row-major (C-order) indexing.
pub fn block_multi_index(linear_idx: usize, num_blocks: &[usize]) -> BlockIndex {
    let mut result = vec![0; num_blocks.len()];
    let mut remaining = linear_idx;
    for i in (0..num_blocks.len()).rev() {
        result[i] = remaining % num_blocks[i];
        remaining /= num_blocks[i];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_new() {
        let p = BlockPartition::new(vec![3, 4, 3]);
        assert_eq!(p.num_blocks(), 3);
        assert_eq!(p.total_dim(), 10);
        assert_eq!(p.block_size(0), 3);
        assert_eq!(p.block_size(1), 4);
        assert_eq!(p.block_size(2), 3);
    }

    #[test]
    fn test_partition_uniform() {
        let p = BlockPartition::uniform(5, 4);
        assert_eq!(p.num_blocks(), 4);
        assert_eq!(p.total_dim(), 20);
        for i in 0..4 {
            assert_eq!(p.block_size(i), 5);
        }
    }

    #[test]
    fn test_partition_trivial() {
        let p = BlockPartition::trivial(10);
        assert_eq!(p.num_blocks(), 1);
        assert_eq!(p.total_dim(), 10);
        assert_eq!(p.block_size(0), 10);
    }

    #[test]
    fn test_block_range() {
        let p = BlockPartition::new(vec![3, 4, 3]);
        assert_eq!(p.block_range(0), 0..3);
        assert_eq!(p.block_range(1), 3..7);
        assert_eq!(p.block_range(2), 7..10);
    }

    #[test]
    fn test_linear_index() {
        let num_blocks = vec![2, 3, 4];

        // Test some known values
        assert_eq!(block_linear_index(&vec![0, 0, 0], &num_blocks), 0);
        assert_eq!(block_linear_index(&vec![0, 0, 1], &num_blocks), 1);
        assert_eq!(block_linear_index(&vec![0, 1, 0], &num_blocks), 4);
        assert_eq!(block_linear_index(&vec![1, 0, 0], &num_blocks), 12);

        // Round-trip test
        for linear in 0..24 {
            let multi = block_multi_index(linear, &num_blocks);
            assert_eq!(block_linear_index(&multi, &num_blocks), linear);
        }
    }
}
