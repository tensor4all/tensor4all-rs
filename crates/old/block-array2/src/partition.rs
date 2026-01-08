//! Block partition for array axes.

use std::ops::Range;

/// Partition of an axis into blocks.
///
/// Each axis of a blocked array has a partition that defines how
/// the axis is divided into blocks. For example, an axis of size 10
/// could be partitioned into blocks of sizes `[3, 4, 3]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockPartition {
    /// Size of each block.
    block_sizes: Vec<usize>,
    /// Cumulative offsets: `[0, s0, s0+s1, ..., total_dim]`.
    offsets: Vec<usize>,
}

impl BlockPartition {
    /// Create a new partition from block sizes.
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
    pub fn uniform(block_size: usize, num_blocks: usize) -> Self {
        Self::new(vec![block_size; num_blocks])
    }

    /// Create a trivial partition (single block containing the entire axis).
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
    #[inline]
    pub fn block_size(&self, block_idx: usize) -> usize {
        self.block_sizes[block_idx]
    }

    /// Get the range of indices for a specific block.
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
pub fn block_linear_index(block_idx: &BlockIndex, num_blocks: &[usize]) -> usize {
    assert_eq!(
        block_idx.len(),
        num_blocks.len(),
        "Block index rank mismatch"
    );
    let mut linear = 0;
    let mut stride = 1;
    for (i, &idx) in block_idx.iter().rev().enumerate() {
        let axis = num_blocks.len() - 1 - i;
        linear += idx * stride;
        stride *= num_blocks[axis];
    }
    linear
}

/// Compute multi-dimensional block index from linear block index.
pub fn block_multi_index(mut linear: usize, num_blocks: &[usize]) -> BlockIndex {
    let mut idx = vec![0; num_blocks.len()];
    for axis in (0..num_blocks.len()).rev() {
        let nb = num_blocks[axis];
        idx[axis] = linear % nb;
        linear /= nb;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_trivial() {
        let p = BlockPartition::trivial(10);
        assert_eq!(p.num_blocks(), 1);
        assert_eq!(p.total_dim(), 10);
        assert_eq!(p.block_size(0), 10);
        assert_eq!(p.block_range(0), 0..10);
    }

    #[test]
    fn test_partition_uniform() {
        let p = BlockPartition::uniform(4, 3);
        assert_eq!(p.num_blocks(), 3);
        assert_eq!(p.total_dim(), 12);
        assert_eq!(p.block_size(2), 4);
        assert_eq!(p.block_range(1), 4..8);
    }

    #[test]
    fn test_linear_multi_round_trip() {
        let nb = vec![2, 3, 4];
        let idx = vec![1, 2, 3];
        let lin = block_linear_index(&idx, &nb);
        let idx2 = block_multi_index(lin, &nb);
        assert_eq!(idx, idx2);
    }
}
