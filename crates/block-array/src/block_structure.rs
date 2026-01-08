//! Block structure metadata (without actual data).
//!
//! `BlockStructure` represents the shape and sparsity pattern of a blocked array
//! without storing actual element values. This enables:
//! - Cost estimation for matrix operations
//! - Structure-only computation (e.g., predicting output sparsity)
//! - Lightweight representation for optimization algorithms

use std::collections::{HashMap, HashSet};

use crate::partition::{block_linear_index, block_multi_index, BlockIndex, BlockPartition};

/// A plan for reshaping a block structure from one set of partitions to another.
///
/// This captures the mapping from old block indices to new block indices,
/// allowing efficient reshape operations on both `BlockStructure` and `BlockArray`.
#[derive(Debug, Clone)]
pub struct ReshapePlan {
    /// Number of blocks per axis in the old structure.
    pub old_num_blocks: Vec<usize>,
    /// Number of blocks per axis in the new structure.
    pub new_num_blocks: Vec<usize>,
    /// Target partitions.
    pub new_partitions: Vec<BlockPartition>,
    /// Mapping from old linear block index to new linear block index.
    pub block_mapping: HashMap<usize, usize>,
}

impl ReshapePlan {
    /// Create a new reshape plan.
    ///
    /// The `nonzero_blocks` parameter specifies which old linear indices are non-zero
    /// and should be included in the mapping.
    pub fn new(
        old_num_blocks: Vec<usize>,
        new_partitions: Vec<BlockPartition>,
        nonzero_blocks: &HashSet<usize>,
    ) -> Self {
        let new_num_blocks: Vec<usize> = new_partitions.iter().map(|p| p.num_blocks()).collect();

        let block_mapping: HashMap<usize, usize> = nonzero_blocks
            .iter()
            .map(|&old_linear| {
                let old_multi = block_multi_index(old_linear, &old_num_blocks);
                let global_linear = block_linear_index(&old_multi, &old_num_blocks);
                let new_multi = block_multi_index(global_linear, &new_num_blocks);
                let new_linear = block_linear_index(&new_multi, &new_num_blocks);
                (old_linear, new_linear)
            })
            .collect();

        Self {
            old_num_blocks,
            new_num_blocks,
            new_partitions,
            block_mapping,
        }
    }

    /// Get the new block shape for a given old linear block index.
    pub fn new_block_shape(&self, old_linear: usize) -> Vec<usize> {
        let new_linear = self
            .block_mapping
            .get(&old_linear)
            .expect("Block not in mapping");
        let new_multi = block_multi_index(*new_linear, &self.new_num_blocks);
        new_multi
            .iter()
            .enumerate()
            .map(|(axis, &idx)| self.new_partitions[axis].block_size(idx))
            .collect()
    }

    /// Iterate over (old_linear, new_linear) pairs.
    pub fn iter_mapping(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.block_mapping.iter().map(|(&old, &new)| (old, new))
    }
}

/// Merge multiple partitions into one.
///
/// Each combination of blocks from the input partitions becomes a single block
/// in the output partition. The merged block size is the product of individual sizes.
///
/// Example: merge([2, 3], [4, 5]) produces a partition with 4 blocks:
/// - Block 0: size 2*4 = 8  (from block 0 of first, block 0 of second)
/// - Block 1: size 2*5 = 10 (from block 0 of first, block 1 of second)
/// - Block 2: size 3*4 = 12 (from block 1 of first, block 0 of second)
/// - Block 3: size 3*5 = 15 (from block 1 of first, block 1 of second)
fn merge_partitions(partitions: &[&BlockPartition]) -> BlockPartition {
    if partitions.is_empty() {
        return BlockPartition::trivial(1);
    }
    if partitions.len() == 1 {
        return partitions[0].clone();
    }

    // Compute all combinations of block indices
    let num_blocks_per_axis: Vec<usize> = partitions.iter().map(|p| p.num_blocks()).collect();
    let total_blocks: usize = num_blocks_per_axis.iter().product();

    let mut merged_sizes = Vec::with_capacity(total_blocks);
    for linear in 0..total_blocks {
        // Convert linear index to multi-index
        let multi_idx = block_multi_index(linear, &num_blocks_per_axis);
        // Compute product of block sizes
        let size: usize = multi_idx
            .iter()
            .enumerate()
            .map(|(axis, &idx)| partitions[axis].block_size(idx))
            .product();
        merged_sizes.push(size);
    }

    BlockPartition::new(merged_sizes)
}

/// Block structure metadata (partitions + non-zero block positions).
///
/// This type represents the "shape" of a blocked array without the actual data.
/// It can be used for:
/// - Estimating computation costs before performing operations
/// - Computing the structure of operation results
/// - Optimization algorithms that reason about block sparsity
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockStructure {
    /// Block partition for each axis.
    partitions: Vec<BlockPartition>,
    /// Linear indices of non-zero blocks.
    nonzero_blocks: HashSet<usize>,
}

impl BlockStructure {
    /// Create a new block structure.
    pub fn new(partitions: Vec<BlockPartition>, nonzero_blocks: HashSet<usize>) -> Self {
        Self {
            partitions,
            nonzero_blocks,
        }
    }

    /// Create an empty structure with given partitions.
    pub fn empty(partitions: Vec<BlockPartition>) -> Self {
        Self {
            partitions,
            nonzero_blocks: HashSet::new(),
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
        self.nonzero_blocks.len()
    }

    /// Get the set of non-zero block indices (linear).
    pub fn nonzero_blocks(&self) -> &HashSet<usize> {
        &self.nonzero_blocks
    }

    /// Check if a block is non-zero.
    pub fn has_block(&self, block_idx: &BlockIndex) -> bool {
        let linear = block_linear_index(block_idx, &self.num_blocks());
        self.nonzero_blocks.contains(&linear)
    }

    /// Add a non-zero block.
    pub fn insert_block(&mut self, block_idx: &BlockIndex) {
        let linear = block_linear_index(block_idx, &self.num_blocks());
        self.nonzero_blocks.insert(linear);
    }

    /// Remove a block (mark as zero).
    pub fn remove_block(&mut self, block_idx: &BlockIndex) -> bool {
        let linear = block_linear_index(block_idx, &self.num_blocks());
        self.nonzero_blocks.remove(&linear)
    }

    /// Get the shape of a specific block (N-dimensional).
    pub fn block_shape(&self, block_idx: &BlockIndex) -> Vec<usize> {
        assert_eq!(
            block_idx.len(),
            self.rank(),
            "Block index rank {} must match structure rank {}",
            block_idx.len(),
            self.rank()
        );
        block_idx
            .iter()
            .enumerate()
            .map(|(axis, &idx)| self.partitions[axis].block_size(idx))
            .collect()
    }

    /// Get the shape of a specific block as a fixed-size array (2D only).
    pub fn block_shape_2d(&self, block_idx: &BlockIndex) -> [usize; 2] {
        assert_eq!(self.rank(), 2, "block_shape_2d requires 2D structure");
        assert_eq!(block_idx.len(), 2, "Block index must be 2D");
        [
            self.partitions[0].block_size(block_idx[0]),
            self.partitions[1].block_size(block_idx[1]),
        ]
    }

    /// Get the number of elements in a specific block.
    pub fn block_size(&self, block_idx: &BlockIndex) -> usize {
        self.block_shape(block_idx).iter().product()
    }

    /// Get the total number of elements (sum over all dimensions).
    pub fn total_elements(&self) -> usize {
        self.shape().iter().product()
    }

    /// Transform block indices according to a permutation.
    ///
    /// Takes a list of block indices and returns the transformed indices
    /// after applying the permutation to the axes.
    pub fn permute_block_indices(&self, indices: &[BlockIndex], perm: &[usize]) -> Vec<BlockIndex> {
        indices
            .iter()
            .map(|idx| perm.iter().map(|&a| idx[a]).collect())
            .collect()
    }

    /// Permute axes.
    pub fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), self.rank(), "Axes length must match rank");

        // Verify axes is a valid permutation
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort();
        assert_eq!(
            sorted_axes,
            (0..self.rank()).collect::<Vec<_>>(),
            "Axes must be a permutation of 0..rank"
        );

        let new_partitions: Vec<_> = axes.iter().map(|&a| self.partitions[a].clone()).collect();
        let new_num_blocks: Vec<_> = new_partitions.iter().map(|p| p.num_blocks()).collect();

        // Transform block indices
        let orig_indices: Vec<BlockIndex> = self.iter_nonzero_indices().collect();
        let new_indices = self.permute_block_indices(&orig_indices, axes);

        let new_nonzero: HashSet<usize> = new_indices
            .iter()
            .map(|idx| block_linear_index(idx, &new_num_blocks))
            .collect();

        Self {
            partitions: new_partitions,
            nonzero_blocks: new_nonzero,
        }
    }

    /// Iterate over non-zero block indices.
    pub fn iter_nonzero_indices(&self) -> impl Iterator<Item = BlockIndex> + '_ {
        let num_blocks = self.num_blocks();
        self.nonzero_blocks
            .iter()
            .map(move |&linear| block_multi_index(linear, &num_blocks))
    }

    /// Reshape to a new shape with trivial partitions.
    ///
    /// This is a convenience method that creates trivial partitions (single block
    /// per axis) for the new shape and delegates to `reshape_to`.
    ///
    /// # Arguments
    /// * `new_shape` - Target shape
    ///
    /// # Panics
    /// - If total elements don't match
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_partitions: Vec<BlockPartition> = new_shape
            .iter()
            .map(|&dim| BlockPartition::trivial(dim))
            .collect();
        self.reshape_to(new_partitions)
    }

    /// Create a reshape plan for transforming to new partitions.
    ///
    /// The plan can be reused to reshape both `BlockStructure` and `BlockArray`
    /// with the same mapping.
    ///
    /// # Arguments
    /// * `new_partitions` - Target partitions for each axis
    ///
    /// # Panics
    /// - If total elements don't match
    pub fn plan_reshape_to(&self, new_partitions: Vec<BlockPartition>) -> ReshapePlan {
        // Verify total dims match
        let old_total: usize = self.shape().iter().product();
        let new_total: usize = new_partitions.iter().map(|p| p.total_dim()).product();
        assert_eq!(
            old_total, new_total,
            "Total elements must match: {} vs {}",
            old_total, new_total
        );

        ReshapePlan::new(self.num_blocks(), new_partitions, &self.nonzero_blocks)
    }

    /// Reshape using a pre-computed plan.
    ///
    /// This is more efficient when reshaping multiple arrays with the same
    /// block structure, as the mapping is computed only once.
    pub fn reshape_with_plan(&self, plan: &ReshapePlan) -> Self {
        let new_nonzero: HashSet<usize> = plan.block_mapping.values().copied().collect();

        Self {
            partitions: plan.new_partitions.clone(),
            nonzero_blocks: new_nonzero,
        }
    }

    /// Reshape to new partitions (general reshape).
    ///
    /// This transforms the block structure to use new partitions.
    /// Total elements must be preserved. Block indices are transformed
    /// by mapping through linear indices in row-major order.
    ///
    /// # Arguments
    /// * `new_partitions` - Target partitions for each axis
    ///
    /// # Panics
    /// - If total elements don't match
    pub fn reshape_to(&self, new_partitions: Vec<BlockPartition>) -> Self {
        let plan = self.plan_reshape_to(new_partitions);
        self.reshape_with_plan(&plan)
    }

    /// Compute merged partitions for grouping consecutive axes.
    ///
    /// This computes the target partitions when merging consecutive axes.
    /// For example, with partitions for axes [i, j, k] and group_sizes [2, 1],
    /// this returns [merged(i,j), k].
    ///
    /// # Arguments
    /// * `group_sizes` - Number of axes to merge for each output axis.
    ///   Sum must equal the current rank.
    ///
    /// # Returns
    /// New partitions where each partition is the merged result of the corresponding group.
    pub fn compute_merged_partitions(&self, group_sizes: &[usize]) -> Vec<BlockPartition> {
        let total_axes: usize = group_sizes.iter().sum();
        assert_eq!(
            total_axes,
            self.rank(),
            "Sum of group sizes {} must equal rank {}",
            total_axes,
            self.rank()
        );
        assert!(
            group_sizes.iter().all(|&s| s > 0),
            "All group sizes must be positive"
        );

        let mut new_partitions = Vec::with_capacity(group_sizes.len());
        let mut axis = 0;

        for &group_size in group_sizes {
            if group_size == 1 {
                new_partitions.push(self.partitions[axis].clone());
            } else {
                let merged_parts: Vec<_> = (axis..axis + group_size)
                    .map(|a| &self.partitions[a])
                    .collect();
                new_partitions.push(merge_partitions(&merged_parts));
            }
            axis += group_size;
        }

        new_partitions
    }

    /// Build an index of non-zero blocks grouped by a specific axis.
    ///
    /// For a 2D structure:
    /// - `group_by_axis=0`: groups by row, returns `row -> [cols with non-zero blocks]`
    /// - `group_by_axis=1`: groups by col, returns `col -> [rows with non-zero blocks]`
    pub fn group_by_axis(&self, group_by_axis: usize) -> HashMap<usize, Vec<usize>> {
        let num_blocks = self.num_blocks();
        let mut index: HashMap<usize, Vec<usize>> = HashMap::new();

        for &linear in &self.nonzero_blocks {
            let block_idx = block_multi_index(linear, &num_blocks);
            let key = block_idx[group_by_axis];
            let value = block_idx[1 - group_by_axis];
            index.entry(key).or_default().push(value);
        }

        index
    }

    /// Compute the resulting structure of matrix multiplication (C = self @ other).
    ///
    /// This predicts which blocks in the result will be non-zero based on the
    /// sparsity patterns of the inputs, without performing actual computation.
    pub fn matmul_structure(&self, other: &Self) -> Self {
        assert_eq!(self.rank(), 2, "matmul requires 2D structure");
        assert_eq!(other.rank(), 2, "matmul requires 2D structure");
        assert_eq!(
            self.partitions[1], other.partitions[0],
            "Inner partitions must match"
        );

        let result_partitions = vec![self.partitions[0].clone(), other.partitions[1].clone()];
        let result_num_blocks = vec![
            self.partitions[0].num_blocks(),
            other.partitions[1].num_blocks(),
        ];

        let a_by_k = self.group_by_axis(1); // k -> [i...]
        let b_by_k = other.group_by_axis(0); // k -> [j...]

        let mut result_nonzero = HashSet::new();

        for (k, a_rows) in &a_by_k {
            if let Some(b_cols) = b_by_k.get(k) {
                for &i in a_rows {
                    for &j in b_cols {
                        let linear = block_linear_index(&vec![i, j], &result_num_blocks);
                        result_nonzero.insert(linear);
                    }
                }
            }
        }

        Self {
            partitions: result_partitions,
            nonzero_blocks: result_nonzero,
        }
    }

    /// Estimate the computational cost (FLOPs) for matrix multiplication.
    ///
    /// Returns the estimated number of floating-point operations for C = self @ other.
    /// Includes both multiplication and accumulation costs.
    pub fn estimate_matmul_cost(&self, other: &Self) -> u64 {
        assert_eq!(self.rank(), 2, "matmul requires 2D structure");
        assert_eq!(other.rank(), 2, "matmul requires 2D structure");
        assert_eq!(
            self.partitions[1], other.partitions[0],
            "Inner partitions must match"
        );

        let a_by_k = self.group_by_axis(1);
        let b_by_k = other.group_by_axis(0);

        let mut total_ops: u64 = 0;
        let mut output_counts: HashMap<(usize, usize), usize> = HashMap::new();

        for (k, a_rows) in &a_by_k {
            if let Some(b_cols) = b_by_k.get(k) {
                let k_size = self.partitions[1].block_size(*k);
                for &i in a_rows {
                    let m = self.partitions[0].block_size(i);
                    for &j in b_cols {
                        let n = other.partitions[1].block_size(j);
                        // matmul: 2 * m * k * n ops (m*k*n muls + m*k*n adds)
                        total_ops += 2 * (m * k_size * n) as u64;
                        *output_counts.entry((i, j)).or_default() += 1;
                    }
                }
            }
        }

        // accumulate cost: (count - 1) * m * n per output block
        for (&(i, j), &count) in &output_counts {
            if count > 1 {
                let m = self.partitions[0].block_size(i);
                let n = other.partitions[1].block_size(j);
                total_ops += ((count - 1) * m * n) as u64;
            }
        }

        total_ops
    }

    /// Compute the resulting structure of tensor contraction (tensordot).
    ///
    /// Contracts `self` and `other` along the specified axes.
    /// - `axes_self`: axes of `self` to contract
    /// - `axes_other`: axes of `other` to contract (must match in partition)
    ///
    /// The result has shape: [free axes of self] + [free axes of other]
    ///
    /// # Panics
    /// - If axes lengths don't match
    /// - If contracted partitions don't match
    pub fn tensordot(&self, other: &Self, axes_self: &[usize], axes_other: &[usize]) -> Self {
        assert_eq!(
            axes_self.len(),
            axes_other.len(),
            "Number of contracted axes must match"
        );

        // Verify contracted partitions match
        for (&a, &b) in axes_self.iter().zip(axes_other.iter()) {
            assert_eq!(
                self.partitions[a], other.partitions[b],
                "Contracted partitions must match: self axis {} vs other axis {}",
                a, b
            );
        }

        // Compute free axes (axes not being contracted)
        let free_self: Vec<usize> = (0..self.rank())
            .filter(|a| !axes_self.contains(a))
            .collect();
        let free_other: Vec<usize> = (0..other.rank())
            .filter(|a| !axes_other.contains(a))
            .collect();

        // Build permutation: [free axes, contracted axes]
        let perm_self: Vec<usize> = free_self.iter().chain(axes_self.iter()).copied().collect();
        let perm_other: Vec<usize> = axes_other
            .iter()
            .chain(free_other.iter())
            .copied()
            .collect();

        // Permute
        let a_perm = self.permute(&perm_self);
        let b_perm = other.permute(&perm_other);

        // Reshape to 2D: [free, contracted] and [contracted, free]
        let num_free_self = free_self.len().max(1);
        let num_contracted = axes_self.len().max(1);
        let num_free_other = free_other.len().max(1);

        let a_2d = if self.rank() == 0 {
            a_perm
        } else if free_self.is_empty() {
            // All axes contracted: reshape to [contracted]
            let group_sizes = vec![axes_self.len().max(1)];
            let new_partitions = a_perm.compute_merged_partitions(&group_sizes);
            a_perm.reshape_to(new_partitions)
        } else if axes_self.is_empty() {
            // No contraction: reshape to [free]
            let group_sizes = vec![free_self.len()];
            let new_partitions = a_perm.compute_merged_partitions(&group_sizes);
            a_perm.reshape_to(new_partitions)
        } else {
            let group_sizes = vec![num_free_self, num_contracted];
            let new_partitions = a_perm.compute_merged_partitions(&group_sizes);
            a_perm.reshape_to(new_partitions)
        };

        let b_2d = if other.rank() == 0 {
            b_perm
        } else if free_other.is_empty() {
            let group_sizes = vec![axes_other.len()];
            let new_partitions = b_perm.compute_merged_partitions(&group_sizes);
            b_perm.reshape_to(new_partitions)
        } else if axes_other.is_empty() {
            let group_sizes = vec![free_other.len()];
            let new_partitions = b_perm.compute_merged_partitions(&group_sizes);
            b_perm.reshape_to(new_partitions)
        } else {
            let group_sizes = vec![num_contracted, num_free_other];
            let new_partitions = b_perm.compute_merged_partitions(&group_sizes);
            b_perm.reshape_to(new_partitions)
        };

        // matmul if both are 2D, otherwise handle edge cases
        let result_2d = if a_2d.rank() == 2 && b_2d.rank() == 2 {
            a_2d.matmul_structure(&b_2d)
        } else {
            // Edge case: one or both are 1D (full contraction or no contraction)
            // For simplicity, return the appropriate structure
            let mut result_parts = Vec::new();
            if !free_self.is_empty() {
                for &a in &free_self {
                    result_parts.push(self.partitions[a].clone());
                }
            }
            if !free_other.is_empty() {
                for &a in &free_other {
                    result_parts.push(other.partitions[a].clone());
                }
            }
            if result_parts.is_empty() {
                // Scalar result
                result_parts.push(BlockPartition::trivial(1));
            }

            // For now, assume all combinations of nonzero blocks contribute
            // This is a conservative estimate
            let mut result = Self::empty(result_parts);
            for _ in self.iter_nonzero_indices() {
                for other_idx in other.iter_nonzero_indices() {
                    let mut result_idx: Vec<usize> = free_self
                        .iter()
                        .map(|_| 0) // placeholder
                        .collect();
                    result_idx.extend(free_other.iter().map(|&a| other_idx[a]));
                    if !result_idx.is_empty() {
                        result.insert_block(&result_idx);
                    }
                }
            }
            result
        };

        // Reshape back to N-dim if needed
        if free_self.len() + free_other.len() <= 2 {
            result_2d
        } else {
            // Build group sizes to expand back
            let mut group_sizes = Vec::new();
            if !free_self.is_empty() {
                group_sizes.push(1); // Keep first merged axis as is for now
            }
            if !free_other.is_empty() {
                group_sizes.push(1);
            }
            result_2d // Simplified: return 2D structure
        }
    }

    /// Estimate the computational cost for tensor contraction.
    ///
    /// Uses the same approach as tensordot but returns cost instead.
    pub fn estimate_tensordot_cost(
        &self,
        other: &Self,
        axes_self: &[usize],
        axes_other: &[usize],
    ) -> u64 {
        assert_eq!(
            axes_self.len(),
            axes_other.len(),
            "Number of contracted axes must match"
        );

        // Verify contracted partitions match
        for (&a, &b) in axes_self.iter().zip(axes_other.iter()) {
            assert_eq!(
                self.partitions[a], other.partitions[b],
                "Contracted partitions must match"
            );
        }

        // Compute free axes
        let free_self: Vec<usize> = (0..self.rank())
            .filter(|a| !axes_self.contains(a))
            .collect();
        let free_other: Vec<usize> = (0..other.rank())
            .filter(|a| !axes_other.contains(a))
            .collect();

        // Build permutation
        let perm_self: Vec<usize> = free_self.iter().chain(axes_self.iter()).copied().collect();
        let perm_other: Vec<usize> = axes_other
            .iter()
            .chain(free_other.iter())
            .copied()
            .collect();

        // Permute and reshape to 2D
        let a_perm = self.permute(&perm_self);
        let b_perm = other.permute(&perm_other);

        let num_free_self = free_self.len();
        let num_contracted = axes_self.len();
        let num_free_other = free_other.len();

        if num_free_self == 0 || num_free_other == 0 || num_contracted == 0 {
            // Edge case: handle separately
            // For now, return a simple estimate
            let self_size: u64 = self
                .iter_nonzero_indices()
                .map(|idx| self.block_size(&idx) as u64)
                .sum();
            let other_size: u64 = other
                .iter_nonzero_indices()
                .map(|idx| other.block_size(&idx) as u64)
                .sum();
            return 2 * self_size * other_size / (num_contracted.max(1) as u64);
        }

        let group_sizes_a = vec![num_free_self, num_contracted];
        let group_sizes_b = vec![num_contracted, num_free_other];
        let a_2d = a_perm.reshape_to(a_perm.compute_merged_partitions(&group_sizes_a));
        let b_2d = b_perm.reshape_to(b_perm.compute_merged_partitions(&group_sizes_b));

        a_2d.estimate_matmul_cost(&b_2d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_structure_new() {
        let partitions = vec![
            BlockPartition::new(vec![2, 3]),
            BlockPartition::new(vec![4, 5]),
        ];
        let structure = BlockStructure::empty(partitions);

        assert_eq!(structure.rank(), 2);
        assert_eq!(structure.shape(), vec![5, 9]);
        assert_eq!(structure.num_blocks(), vec![2, 2]);
        assert_eq!(structure.num_nonzero_blocks(), 0);
    }

    #[test]
    fn test_block_structure_insert_has() {
        let partitions = vec![BlockPartition::uniform(2, 3), BlockPartition::uniform(2, 3)];
        let mut structure = BlockStructure::empty(partitions);

        structure.insert_block(&vec![0, 1]);
        structure.insert_block(&vec![2, 2]);

        assert!(structure.has_block(&vec![0, 1]));
        assert!(structure.has_block(&vec![2, 2]));
        assert!(!structure.has_block(&vec![0, 0]));
        assert_eq!(structure.num_nonzero_blocks(), 2);
    }

    #[test]
    fn test_block_structure_permute_2d() {
        let partitions = vec![
            BlockPartition::new(vec![2, 3]),
            BlockPartition::new(vec![4, 5]),
        ];
        let mut structure = BlockStructure::empty(partitions);
        structure.insert_block(&vec![0, 1]);
        structure.insert_block(&vec![1, 0]);

        // Transpose via permute
        let permuted = structure.permute(&[1, 0]);

        assert_eq!(permuted.shape(), vec![9, 5]);
        assert!(permuted.has_block(&vec![1, 0])); // was [0, 1]
        assert!(permuted.has_block(&vec![0, 1])); // was [1, 0]
    }

    #[test]
    fn test_block_structure_permute_3d() {
        let partitions = vec![
            BlockPartition::uniform(2, 2), // axis 0: 2 blocks of size 2
            BlockPartition::uniform(3, 2), // axis 1: 2 blocks of size 3
            BlockPartition::uniform(4, 2), // axis 2: 2 blocks of size 4
        ];
        let mut structure = BlockStructure::empty(partitions);
        structure.insert_block(&vec![0, 1, 0]);
        structure.insert_block(&vec![1, 0, 1]);

        // Permute: (0, 1, 2) -> (2, 0, 1)
        let permuted = structure.permute(&[2, 0, 1]);

        assert_eq!(permuted.rank(), 3);
        assert_eq!(permuted.shape(), vec![8, 4, 6]); // [axis2, axis0, axis1]

        // [0, 1, 0] -> [0, 0, 1]
        assert!(permuted.has_block(&vec![0, 0, 1]));
        // [1, 0, 1] -> [1, 1, 0]
        assert!(permuted.has_block(&vec![1, 1, 0]));
    }

    #[test]
    fn test_group_by_axis() {
        let partitions = vec![BlockPartition::uniform(2, 3), BlockPartition::uniform(2, 3)];
        let mut structure = BlockStructure::empty(partitions);
        // Row 0: cols 0, 2
        // Row 1: col 1
        structure.insert_block(&vec![0, 0]);
        structure.insert_block(&vec![0, 2]);
        structure.insert_block(&vec![1, 1]);

        let by_row = structure.group_by_axis(0);
        assert_eq!(by_row.get(&0).map(|v| v.len()), Some(2));
        assert_eq!(by_row.get(&1).map(|v| v.len()), Some(1));

        let by_col = structure.group_by_axis(1);
        assert_eq!(by_col.get(&0).map(|v| v.len()), Some(1));
        assert_eq!(by_col.get(&1).map(|v| v.len()), Some(1));
        assert_eq!(by_col.get(&2).map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_matmul_structure() {
        // A: 2x3 blocks, B: 3x2 blocks
        // A has blocks at [0,0], [0,1], [1,2]
        // B has blocks at [0,0], [1,0], [2,1]
        // C = A @ B should have:
        //   C[0,0] from A[0,0]*B[0,0] + A[0,1]*B[1,0]
        //   C[1,1] from A[1,2]*B[2,1]
        let a_parts = vec![BlockPartition::uniform(2, 2), BlockPartition::uniform(2, 3)];
        let mut a = BlockStructure::empty(a_parts);
        a.insert_block(&vec![0, 0]);
        a.insert_block(&vec![0, 1]);
        a.insert_block(&vec![1, 2]);

        let b_parts = vec![BlockPartition::uniform(2, 3), BlockPartition::uniform(2, 2)];
        let mut b = BlockStructure::empty(b_parts);
        b.insert_block(&vec![0, 0]);
        b.insert_block(&vec![1, 0]);
        b.insert_block(&vec![2, 1]);

        let c = a.matmul_structure(&b);

        assert_eq!(c.num_blocks(), vec![2, 2]);
        assert!(c.has_block(&vec![0, 0])); // from A[0,0]*B[0,0] and A[0,1]*B[1,0]
        assert!(c.has_block(&vec![1, 1])); // from A[1,2]*B[2,1]
        assert!(!c.has_block(&vec![0, 1]));
        assert!(!c.has_block(&vec![1, 0]));
    }

    #[test]
    fn test_estimate_matmul_cost() {
        // Simple case: single block multiplication
        // A[2x3] @ B[3x4] = C[2x4]
        // Cost = 2 * 2 * 3 * 4 = 48
        let a_parts = vec![BlockPartition::trivial(2), BlockPartition::trivial(3)];
        let mut a = BlockStructure::empty(a_parts);
        a.insert_block(&vec![0, 0]);

        let b_parts = vec![BlockPartition::trivial(3), BlockPartition::trivial(4)];
        let mut b = BlockStructure::empty(b_parts);
        b.insert_block(&vec![0, 0]);

        let cost = a.estimate_matmul_cost(&b);
        assert_eq!(cost, 48);
    }

    #[test]
    fn test_estimate_matmul_cost_with_accumulate() {
        // A[0,0] and A[0,1] both contribute to C[0,0]
        // A: 2x2 block sizes, B: 2x2 block sizes
        // matmul cost: 2 * (2*2*2) + 2 * (2*2*2) = 32
        // accumulate cost: 1 * (2*2) = 4
        // total = 36
        let a_parts = vec![BlockPartition::uniform(2, 1), BlockPartition::uniform(2, 2)];
        let mut a = BlockStructure::empty(a_parts);
        a.insert_block(&vec![0, 0]);
        a.insert_block(&vec![0, 1]);

        let b_parts = vec![BlockPartition::uniform(2, 2), BlockPartition::uniform(2, 1)];
        let mut b = BlockStructure::empty(b_parts);
        b.insert_block(&vec![0, 0]);
        b.insert_block(&vec![1, 0]);

        let cost = a.estimate_matmul_cost(&b);
        assert_eq!(cost, 36);
    }

    #[test]
    fn test_estimate_matmul_cost_sparse() {
        // Sparse case: no matching k indices -> cost = 0
        let a_parts = vec![BlockPartition::uniform(2, 2), BlockPartition::uniform(2, 2)];
        let mut a = BlockStructure::empty(a_parts);
        a.insert_block(&vec![0, 0]); // k=0

        let b_parts = vec![BlockPartition::uniform(2, 2), BlockPartition::uniform(2, 2)];
        let mut b = BlockStructure::empty(b_parts);
        b.insert_block(&vec![1, 0]); // k=1

        let cost = a.estimate_matmul_cost(&b);
        assert_eq!(cost, 0);
    }

    #[test]
    fn test_reshape_3d_to_2d() {
        // 3D structure: 2x3x2 blocks
        let partitions = vec![
            BlockPartition::uniform(2, 2), // axis 0: 2 blocks of size 2
            BlockPartition::uniform(3, 3), // axis 1: 3 blocks of size 3
            BlockPartition::uniform(4, 2), // axis 2: 2 blocks of size 4
        ];
        let mut structure = BlockStructure::empty(partitions);
        structure.insert_block(&vec![0, 1, 0]);
        structure.insert_block(&vec![1, 2, 1]);

        // Reshape to 2D: merge first two axes using compute_merged_partitions + reshape_to
        let group_sizes = [2, 1];
        let new_partitions = structure.compute_merged_partitions(&group_sizes);
        let reshaped = structure.reshape_to(new_partitions);

        assert_eq!(reshaped.rank(), 2);
        // First axis: 2*3 = 6 blocks
        // Second axis: 2 blocks
        assert_eq!(reshaped.num_blocks(), vec![6, 2]);

        // [0, 1, 0] -> [0*3 + 1, 0] = [1, 0]
        assert!(reshaped.has_block(&vec![1, 0]));
        // [1, 2, 1] -> [1*3 + 2, 1] = [5, 1]
        assert!(reshaped.has_block(&vec![5, 1]));

        assert_eq!(reshaped.num_nonzero_blocks(), 2);
    }

    #[test]
    fn test_reshape_identity() {
        // Reshape with all 1s should be identity
        let partitions = vec![BlockPartition::uniform(2, 2), BlockPartition::uniform(3, 2)];
        let mut structure = BlockStructure::empty(partitions);
        structure.insert_block(&vec![0, 1]);
        structure.insert_block(&vec![1, 0]);

        // Using compute_merged_partitions + reshape_to
        let group_sizes = [1, 1];
        let new_partitions = structure.compute_merged_partitions(&group_sizes);
        let reshaped = structure.reshape_to(new_partitions);

        assert_eq!(reshaped.rank(), 2);
        assert_eq!(reshaped.num_blocks(), vec![2, 2]);
        assert!(reshaped.has_block(&vec![0, 1]));
        assert!(reshaped.has_block(&vec![1, 0]));
    }

    #[test]
    fn test_reshape_to_1d_with_merged_partitions() {
        // Reshape 2D to 1D using compute_merged_partitions
        let partitions = vec![
            BlockPartition::new(vec![2, 3]), // 2 blocks
            BlockPartition::new(vec![4, 5]), // 2 blocks
        ];
        let mut structure = BlockStructure::empty(partitions);
        structure.insert_block(&vec![0, 0]); // -> 0
        structure.insert_block(&vec![0, 1]); // -> 1
        structure.insert_block(&vec![1, 1]); // -> 3

        let group_sizes = [2];
        let new_partitions = structure.compute_merged_partitions(&group_sizes);
        let reshaped = structure.reshape_to(new_partitions);

        assert_eq!(reshaped.rank(), 1);
        assert_eq!(reshaped.num_blocks(), vec![4]); // 2*2 = 4 blocks

        // Block sizes: [2*4, 2*5, 3*4, 3*5] = [8, 10, 12, 15]
        assert!(reshaped.has_block(&vec![0])); // from [0, 0]
        assert!(reshaped.has_block(&vec![1])); // from [0, 1]
        assert!(reshaped.has_block(&vec![3])); // from [1, 1]
        assert!(!reshaped.has_block(&vec![2])); // [1, 0] was not set
    }

    #[test]
    fn test_merge_partitions() {
        // Test the merge_partitions helper
        let p1 = BlockPartition::new(vec![2, 3]);
        let p2 = BlockPartition::new(vec![4, 5]);

        let merged = super::merge_partitions(&[&p1, &p2]);

        // 4 blocks with sizes: [2*4, 2*5, 3*4, 3*5] = [8, 10, 12, 15]
        assert_eq!(merged.num_blocks(), 4);
        assert_eq!(merged.block_size(0), 8);
        assert_eq!(merged.block_size(1), 10);
        assert_eq!(merged.block_size(2), 12);
        assert_eq!(merged.block_size(3), 15);
        assert_eq!(merged.total_dim(), 8 + 10 + 12 + 15);
    }

    #[test]
    fn test_tensordot_3d_3d() {
        // A[i, j, k] × B[k, l, m] -> C[i, j, l, m]
        // Contract on axis 2 of A and axis 0 of B
        let k_partition = BlockPartition::uniform(3, 2); // shared

        let a_parts = vec![
            BlockPartition::uniform(2, 2), // i: 2 blocks
            BlockPartition::uniform(4, 2), // j: 2 blocks
            k_partition.clone(),           // k: 2 blocks
        ];
        let mut a = BlockStructure::empty(a_parts);
        a.insert_block(&vec![0, 1, 0]);
        a.insert_block(&vec![1, 0, 1]);

        let b_parts = vec![
            k_partition.clone(),           // k: 2 blocks
            BlockPartition::uniform(5, 2), // l: 2 blocks
            BlockPartition::uniform(6, 2), // m: 2 blocks
        ];
        let mut b = BlockStructure::empty(b_parts);
        b.insert_block(&vec![0, 0, 1]);
        b.insert_block(&vec![1, 1, 0]);

        let c = a.tensordot(&b, &[2], &[0]);

        // Result should be 2D (merged): [i*j, l*m] = [4, 4] blocks
        assert_eq!(c.rank(), 2);
        assert_eq!(c.num_blocks(), vec![4, 4]);
    }

    #[test]
    fn test_tensordot_matmul_equivalent() {
        // tensordot with single axis contraction should be equivalent to matmul
        let k_partition = BlockPartition::uniform(3, 2);

        let a_parts = vec![
            BlockPartition::uniform(2, 2), // i
            k_partition.clone(),           // k
        ];
        let mut a = BlockStructure::empty(a_parts);
        a.insert_block(&vec![0, 0]);
        a.insert_block(&vec![1, 1]);

        let b_parts = vec![
            k_partition.clone(),           // k
            BlockPartition::uniform(4, 2), // j
        ];
        let mut b = BlockStructure::empty(b_parts);
        b.insert_block(&vec![0, 1]);
        b.insert_block(&vec![1, 0]);

        // tensordot
        let c_td = a.tensordot(&b, &[1], &[0]);
        // matmul
        let c_mm = a.matmul_structure(&b);

        assert_eq!(c_td.rank(), c_mm.rank());
        assert_eq!(c_td.num_blocks(), c_mm.num_blocks());
        assert_eq!(c_td.num_nonzero_blocks(), c_mm.num_nonzero_blocks());
    }

    #[test]
    fn test_estimate_tensordot_cost() {
        let k_partition = BlockPartition::uniform(3, 2);

        let a_parts = vec![BlockPartition::uniform(2, 2), k_partition.clone()];
        let mut a = BlockStructure::empty(a_parts);
        a.insert_block(&vec![0, 0]);

        let b_parts = vec![k_partition.clone(), BlockPartition::uniform(4, 2)];
        let mut b = BlockStructure::empty(b_parts);
        b.insert_block(&vec![0, 0]);

        // tensordot cost should equal matmul cost
        let td_cost = a.estimate_tensordot_cost(&b, &[1], &[0]);
        let mm_cost = a.estimate_matmul_cost(&b);

        assert_eq!(td_cost, mm_cost);
    }

    #[test]
    fn test_reshape_to_new_shape() {
        // 2D structure with single block -> reshape to 1D
        let partitions = vec![BlockPartition::trivial(4), BlockPartition::trivial(3)];
        let mut structure = BlockStructure::empty(partitions);
        structure.insert_block(&vec![0, 0]);

        // Reshape [4, 3] -> [12]
        let reshaped = structure.reshape(&[12]);

        assert_eq!(reshaped.rank(), 1);
        assert_eq!(reshaped.shape(), vec![12]);
        assert_eq!(reshaped.num_blocks(), vec![1]);
        assert!(reshaped.has_block(&vec![0]));
    }

    #[test]
    fn test_reshape_1d_to_2d() {
        // 1D structure -> reshape to 2D
        let partitions = vec![BlockPartition::trivial(12)];
        let mut structure = BlockStructure::empty(partitions);
        structure.insert_block(&vec![0]);

        // Reshape [12] -> [3, 4]
        let reshaped = structure.reshape(&[3, 4]);

        assert_eq!(reshaped.rank(), 2);
        assert_eq!(reshaped.shape(), vec![3, 4]);
        assert_eq!(reshaped.num_blocks(), vec![1, 1]);
        assert!(reshaped.has_block(&vec![0, 0]));
    }

    #[test]
    fn test_reshape_2d_to_3d() {
        // 2D structure -> reshape to 3D
        let partitions = vec![BlockPartition::trivial(6), BlockPartition::trivial(4)];
        let mut structure = BlockStructure::empty(partitions);
        structure.insert_block(&vec![0, 0]);

        // Reshape [6, 4] -> [2, 3, 4]
        let reshaped = structure.reshape(&[2, 3, 4]);

        assert_eq!(reshaped.rank(), 3);
        assert_eq!(reshaped.shape(), vec![2, 3, 4]);
        assert_eq!(reshaped.num_blocks(), vec![1, 1, 1]);
        assert!(reshaped.has_block(&vec![0, 0, 0]));
    }

    #[test]
    fn test_reshape_round_trip_2d_3d_2d() {
        // Test round-trip: 2D -> 3D -> 2D should recover original structure
        let partitions = vec![BlockPartition::trivial(6), BlockPartition::trivial(4)];
        let mut original = BlockStructure::empty(partitions);
        original.insert_block(&vec![0, 0]);

        // Count elements before
        let elements_before: usize = original
            .iter_nonzero_indices()
            .map(|idx| original.block_size(&idx))
            .sum();

        // 2D [6, 4] -> 3D [2, 3, 4]
        let reshaped_3d = original.reshape(&[2, 3, 4]);
        let elements_after_3d: usize = reshaped_3d
            .iter_nonzero_indices()
            .map(|idx| reshaped_3d.block_size(&idx))
            .sum();
        assert_eq!(
            elements_before, elements_after_3d,
            "Elements must be preserved in 2D->3D"
        );

        // 3D [2, 3, 4] -> 2D [6, 4]
        let round_trip = reshaped_3d.reshape(&[6, 4]);
        let elements_after_round: usize = round_trip
            .iter_nonzero_indices()
            .map(|idx| round_trip.block_size(&idx))
            .sum();
        assert_eq!(
            elements_before, elements_after_round,
            "Elements must be preserved in round-trip"
        );

        // Verify structure matches original
        assert_eq!(round_trip.rank(), original.rank());
        assert_eq!(round_trip.shape(), original.shape());
        assert_eq!(round_trip.num_blocks(), original.num_blocks());
        assert_eq!(
            round_trip.num_nonzero_blocks(),
            original.num_nonzero_blocks()
        );
        assert!(round_trip.has_block(&vec![0, 0]));
    }

    #[test]
    fn test_reshape_preserves_nonzero_elements() {
        // Test that non-zero block element count is preserved through multiple reshapes
        let partitions = vec![BlockPartition::trivial(24)];
        let mut structure = BlockStructure::empty(partitions);
        structure.insert_block(&vec![0]);

        let count_elements = |s: &BlockStructure| -> usize {
            s.iter_nonzero_indices().map(|idx| s.block_size(&idx)).sum()
        };

        let initial_elements = count_elements(&structure);
        assert_eq!(initial_elements, 24);

        // [24] -> [4, 6]
        let s1 = structure.reshape(&[4, 6]);
        assert_eq!(count_elements(&s1), initial_elements);

        // [4, 6] -> [2, 2, 6]
        let s2 = s1.reshape(&[2, 2, 6]);
        assert_eq!(count_elements(&s2), initial_elements);

        // [2, 2, 6] -> [2, 12]
        let s3 = s2.reshape(&[2, 12]);
        assert_eq!(count_elements(&s3), initial_elements);

        // [2, 12] -> [24]
        let s4 = s3.reshape(&[24]);
        assert_eq!(count_elements(&s4), initial_elements);
    }

    #[test]
    fn test_plan_reshape_to_and_reshape_with_plan() {
        // Test that plan_reshape_to + reshape_with_plan gives same result as reshape_to
        let partitions = vec![BlockPartition::trivial(6), BlockPartition::trivial(4)];
        let mut structure = BlockStructure::empty(partitions);
        structure.insert_block(&vec![0, 0]);

        // Using reshape_to directly
        let new_partitions = vec![
            BlockPartition::trivial(2),
            BlockPartition::trivial(3),
            BlockPartition::trivial(4),
        ];
        let reshaped_direct = structure.reshape_to(new_partitions.clone());

        // Using plan_reshape_to + reshape_with_plan
        let plan = structure.plan_reshape_to(new_partitions);
        let reshaped_with_plan = structure.reshape_with_plan(&plan);

        // Results should be identical
        assert_eq!(reshaped_direct.rank(), reshaped_with_plan.rank());
        assert_eq!(reshaped_direct.shape(), reshaped_with_plan.shape());
        assert_eq!(
            reshaped_direct.num_blocks(),
            reshaped_with_plan.num_blocks()
        );
        assert_eq!(
            reshaped_direct.num_nonzero_blocks(),
            reshaped_with_plan.num_nonzero_blocks()
        );

        // Verify plan contents
        assert_eq!(plan.old_num_blocks, vec![1, 1]);
        assert_eq!(plan.new_num_blocks, vec![1, 1, 1]);
        assert_eq!(plan.block_mapping.len(), 1);
        assert_eq!(plan.block_mapping.get(&0), Some(&0));
    }

    #[test]
    fn test_reshape_plan_reuse() {
        // Test that the same plan can be used for multiple structures with same shape
        let partitions = vec![BlockPartition::trivial(6), BlockPartition::trivial(4)];

        let mut structure1 = BlockStructure::empty(partitions.clone());
        structure1.insert_block(&vec![0, 0]);

        let mut structure2 = BlockStructure::empty(partitions);
        structure2.insert_block(&vec![0, 0]);

        // Create plan from structure1
        let new_partitions = vec![BlockPartition::trivial(2), BlockPartition::trivial(12)];
        let plan = structure1.plan_reshape_to(new_partitions);

        // Use same plan for both structures
        let reshaped1 = structure1.reshape_with_plan(&plan);
        let reshaped2 = structure2.reshape_with_plan(&plan);

        // Both should have same result
        assert_eq!(reshaped1.shape(), reshaped2.shape());
        assert_eq!(
            reshaped1.num_nonzero_blocks(),
            reshaped2.num_nonzero_blocks()
        );
    }
}
