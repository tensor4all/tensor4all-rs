//! Block array types.

use std::collections::HashMap;

use crate::block_data::{BlockData, BlockDataLike};
use crate::block_structure::{BlockStructure, ReshapePlan};
use crate::partition::{block_linear_index, block_multi_index, BlockIndex, BlockPartition};
use crate::scalar::Scalar;

/// A blocked multi-dimensional array (owns data).
///
/// The array is partitioned along each axis, creating a grid of blocks.
/// Only non-zero blocks are stored in memory (sparse representation).
///
/// Internally wraps a `BlockStructure` for metadata and a HashMap for actual data.
#[derive(Debug, Clone)]
pub struct BlockArray<T: Scalar> {
    /// Block structure (partitions + non-zero block indices).
    structure: BlockStructure,
    /// Non-zero blocks stored in a HashMap.
    /// Key: linear block index, Value: block data.
    data: HashMap<usize, BlockData<T>>,
}

impl<T: Scalar> BlockArray<T> {
    /// Create an empty blocked array with given partitions.
    pub fn new(partitions: Vec<BlockPartition>) -> Self {
        Self {
            structure: BlockStructure::empty(partitions),
            data: HashMap::new(),
        }
    }

    /// Get a reference to the block structure (metadata only).
    pub fn structure(&self) -> &BlockStructure {
        &self.structure
    }

    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.structure.rank()
    }

    /// Get the total shape (sum of block sizes per axis).
    pub fn shape(&self) -> Vec<usize> {
        self.structure.shape()
    }

    /// Get the partitions.
    pub fn partitions(&self) -> &[BlockPartition] {
        self.structure.partitions()
    }

    /// Get the number of blocks per axis.
    pub fn num_blocks(&self) -> Vec<usize> {
        self.structure.num_blocks()
    }

    /// Get the total number of blocks (including zero blocks).
    pub fn total_blocks(&self) -> usize {
        self.structure.total_blocks()
    }

    /// Get the number of non-zero (stored) blocks.
    pub fn num_nonzero_blocks(&self) -> usize {
        self.structure.num_nonzero_blocks()
    }

    /// Get the shape of a specific block (N-dimensional).
    pub fn block_shape(&self, block_idx: &BlockIndex) -> Vec<usize> {
        self.structure.block_shape(block_idx)
    }

    /// Get a block reference (returns None for zero blocks).
    pub fn get_block(&self, block_idx: &BlockIndex) -> Option<&BlockData<T>> {
        let linear = block_linear_index(block_idx, &self.num_blocks());
        self.data.get(&linear)
    }

    /// Set a block.
    pub fn set_block(&mut self, block_idx: BlockIndex, data: BlockData<T>) {
        let linear = block_linear_index(&block_idx, &self.num_blocks());
        self.structure.insert_block(&block_idx);
        self.data.insert(linear, data);
    }

    /// Accumulate (add) data to a block.
    ///
    /// If the block exists, adds the data element-wise.
    /// If the block doesn't exist, sets it to the given data.
    pub fn accumulate_block(&mut self, block_idx: BlockIndex, data: BlockData<T>) {
        let linear = block_linear_index(&block_idx, &self.num_blocks());
        if let Some(existing) = self.data.get(&linear) {
            let summed = existing.add(&data);
            self.data.insert(linear, summed);
        } else {
            self.structure.insert_block(&block_idx);
            self.data.insert(linear, data);
        }
    }

    /// Remove a block (make it zero).
    pub fn remove_block(&mut self, block_idx: &BlockIndex) -> Option<BlockData<T>> {
        let linear = block_linear_index(block_idx, &self.num_blocks());
        self.structure.remove_block(block_idx);
        self.data.remove(&linear)
    }

    /// Iterate over non-zero blocks.
    pub fn iter_blocks(&self) -> impl Iterator<Item = (BlockIndex, &BlockData<T>)> {
        let num_blocks = self.num_blocks();
        self.data.iter().map(move |(&linear, data)| {
            let block_idx = block_multi_index(linear, &num_blocks);
            (block_idx, data)
        })
    }

    /// Permute axes, returning a new owned BlockArray.
    pub fn permute(&self, perm: &[usize]) -> Self {
        // Collect original indices
        let orig_num_blocks = self.num_blocks();
        let orig_indices: Vec<BlockIndex> = self
            .data
            .keys()
            .map(|&linear| block_multi_index(linear, &orig_num_blocks))
            .collect();

        // Transform indices
        let new_indices = self.structure.permute_block_indices(&orig_indices, perm);

        // Build new structure and data
        let new_structure = self.structure.permute(perm);
        let new_num_blocks = new_structure.num_blocks();

        let new_data: HashMap<usize, BlockData<T>> = self
            .data
            .iter()
            .zip(new_indices.iter())
            .map(|((&_orig_linear, block_data), new_idx)| {
                let new_linear = block_linear_index(new_idx, &new_num_blocks);
                let new_block = block_data.permute(perm);
                (new_linear, new_block)
            })
            .collect();

        Self {
            structure: new_structure,
            data: new_data,
        }
    }

    /// Create a reshape plan for transforming to new partitions.
    ///
    /// The plan can be reused to reshape both structure and data efficiently.
    ///
    /// # Arguments
    /// * `new_partitions` - Target partitions for each axis
    ///
    /// # Panics
    /// - If total elements don't match
    pub fn plan_reshape_to(&self, new_partitions: Vec<BlockPartition>) -> ReshapePlan {
        self.structure.plan_reshape_to(new_partitions)
    }

    /// Reshape using a pre-computed plan.
    ///
    /// This is more efficient when reshaping multiple arrays with the same
    /// block structure, as the mapping is computed only once.
    pub fn reshape_with_plan(&self, plan: &ReshapePlan) -> Self {
        let new_structure = self.structure.reshape_with_plan(plan);

        let new_data: HashMap<usize, BlockData<T>> = self
            .data
            .iter()
            .map(|(&old_linear, block_data)| {
                let new_linear = *plan
                    .block_mapping
                    .get(&old_linear)
                    .expect("Block not in reshape plan");
                let new_block_shape = plan.new_block_shape(old_linear);
                let new_block = block_data.reshape(&new_block_shape);
                (new_linear, new_block)
            })
            .collect();

        Self {
            structure: new_structure,
            data: new_data,
        }
    }

    /// Reshape to new partitions (general reshape).
    ///
    /// This transforms the blocked array to use new partitions.
    /// Total elements must be preserved. Block indices and data are transformed
    /// to match the new partition structure.
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

    /// Tensor contraction (tensordot).
    ///
    /// Contracts `self` and `other` along the specified axes.
    /// - `axes_self`: axes of `self` to contract
    /// - `axes_other`: axes of `other` to contract (must match in partition)
    ///
    /// The result has shape: [free axes of self] + [free axes of other]
    pub fn tensordot(
        &self,
        other: &BlockArray<T>,
        axes_self: &[usize],
        axes_other: &[usize],
    ) -> BlockArray<T> {
        // Step 1: Permute self so contracted axes are at the end
        // free_self = axes not in axes_self
        let rank_self = self.rank();
        let free_self: Vec<usize> = (0..rank_self).filter(|a| !axes_self.contains(a)).collect();
        let perm_self: Vec<usize> = free_self.iter().chain(axes_self.iter()).copied().collect();

        // Step 2: Permute other so contracted axes are at the beginning
        let rank_other = other.rank();
        let free_other: Vec<usize> = (0..rank_other)
            .filter(|a| !axes_other.contains(a))
            .collect();
        let perm_other: Vec<usize> = axes_other
            .iter()
            .chain(free_other.iter())
            .copied()
            .collect();

        // Apply permutations
        let a_perm = self.permute(&perm_self);
        let b_perm = other.permute(&perm_other);

        // Save original free axis partitions for final reshape
        // After permutation: a_perm has [free_self axes, contracted axes]
        //                    b_perm has [contracted axes, free_other axes]
        let free_self_partitions: Vec<BlockPartition> = free_self
            .iter()
            .map(|&orig_axis| self.partitions()[orig_axis].clone())
            .collect();
        let free_other_partitions: Vec<BlockPartition> = free_other
            .iter()
            .map(|&orig_axis| other.partitions()[orig_axis].clone())
            .collect();

        // Step 3: Reshape to 2D matrices using compute_merged_partitions + reshape_to
        // A: [free_self..., contracted...] -> [prod(free_self), prod(contracted)]
        // B: [contracted..., free_other...] -> [prod(contracted), prod(free_other)]
        let num_free_self = free_self.len();
        let num_contracted = axes_self.len();
        let num_free_other = free_other.len();

        let group_sizes_a = if num_free_self == 0 {
            vec![num_contracted]
        } else if num_contracted == 0 {
            vec![num_free_self]
        } else {
            vec![num_free_self, num_contracted]
        };

        let group_sizes_b = if num_contracted == 0 {
            vec![num_free_other]
        } else if num_free_other == 0 {
            vec![num_contracted]
        } else {
            vec![num_contracted, num_free_other]
        };

        let partitions_a = a_perm.structure.compute_merged_partitions(&group_sizes_a);
        let partitions_b = b_perm.structure.compute_merged_partitions(&group_sizes_b);
        let a_2d = a_perm.reshape_to(partitions_a);
        let b_2d = b_perm.reshape_to(partitions_b);

        // Step 4: Sparse blocked matrix multiplication
        // Build indices: group A blocks by column (k), B blocks by row (k)
        let mut a_by_k: HashMap<usize, Vec<usize>> = HashMap::new();
        for (block_idx, _) in a_2d.iter_blocks() {
            if block_idx.len() == 2 {
                let k = block_idx[1];
                let i = block_idx[0];
                a_by_k.entry(k).or_default().push(i);
            } else if block_idx.len() == 1 {
                // Edge case: only contracted axes (scalar result on self side)
                a_by_k.entry(block_idx[0]).or_default().push(0);
            }
        }

        let mut b_by_k: HashMap<usize, Vec<usize>> = HashMap::new();
        for (block_idx, _) in b_2d.iter_blocks() {
            if block_idx.len() == 2 {
                let k = block_idx[0];
                let j = block_idx[1];
                b_by_k.entry(k).or_default().push(j);
            } else if block_idx.len() == 1 {
                // Edge case: only contracted axes (scalar result on other side)
                b_by_k.entry(block_idx[0]).or_default().push(0);
            }
        }

        // Handle full contraction (both 1D -> scalar result) as a special case
        if a_2d.rank() == 1 && b_2d.rank() == 1 {
            // Full contraction: result is a scalar (0D tensor)
            // Both a and b are 1D with the same contracted dimension
            // Result = sum over k of dot(a[k], b[k])
            let mut result_scalar = T::zero();

            for (k, _) in &a_by_k {
                if b_by_k.contains_key(k) {
                    let a_block = a_2d.get_block(&vec![*k]).unwrap();
                    let b_block = b_2d.get_block(&vec![*k]).unwrap();
                    result_scalar = result_scalar + a_block.dot(&b_block);
                }
            }

            // Create scalar result (0D blocked array with trivial structure)
            let mut nonzero = std::collections::HashSet::new();
            nonzero.insert(0);
            let scalar_structure = BlockStructure::new(vec![], nonzero);
            let scalar_data =
                BlockData::from_tensor(mdarray::Tensor::from_fn(&[], |_| result_scalar));
            let mut data = HashMap::new();
            data.insert(0, scalar_data);
            let scalar_result = BlockArray {
                structure: scalar_structure,
                data,
            };

            return scalar_result;
        }

        // Build 2D result structure for intermediate result
        let c_2d_structure = a_2d
            .structure
            .tensordot(&b_2d.structure, &[a_2d.rank() - 1], &[0]);

        // Result array (2D intermediate)
        let mut c_2d = BlockArray {
            structure: c_2d_structure,
            data: HashMap::new(),
        };

        // Outer product formulation
        for (k, a_rows) in &a_by_k {
            let b_cols = match b_by_k.get(k) {
                Some(cols) => cols,
                None => continue,
            };

            for &i in a_rows {
                let a_idx = if a_2d.rank() == 2 {
                    vec![i, *k]
                } else {
                    vec![*k]
                };
                let a_block = a_2d.get_block(&a_idx).unwrap();

                for &j in b_cols {
                    let b_idx = if b_2d.rank() == 2 {
                        vec![*k, j]
                    } else {
                        vec![*k]
                    };
                    let b_block = b_2d.get_block(&b_idx).unwrap();

                    // Compute A[i,k] @ B[k,j]
                    let product = a_block.matmul(b_block);

                    // Accumulate into C[i,j]
                    c_2d.accumulate_block(vec![i, j], product);
                }
            }
        }

        // Step 5: Reshape 2D result back to multi-dimensional
        // C_2d: [prod(free_self), prod(free_other)] -> [free_self..., free_other...]
        // Use reshape_to with the original free axis partitions to split merged axes
        if num_free_self + num_free_other == 0 {
            // Scalar result
            c_2d
        } else if num_free_self <= 1 && num_free_other <= 1 {
            // Already in correct shape (2D or less)
            c_2d
        } else {
            // Need to reshape back: combine the original partitions
            let result_partitions: Vec<BlockPartition> = free_self_partitions
                .into_iter()
                .chain(free_other_partitions)
                .collect();
            c_2d.reshape_to(result_partitions)
        }
    }
}

/// Trait for types that can act as blocked arrays (owned or view).
pub trait BlockArrayLike<T: Scalar> {
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

    /// Get the block structure.
    fn structure(&self) -> BlockStructure;
}

impl<T: Scalar> BlockArrayLike<T> for BlockArray<T> {
    fn rank(&self) -> usize {
        self.rank()
    }

    fn shape(&self) -> Vec<usize> {
        self.shape()
    }

    fn partitions(&self) -> Vec<BlockPartition> {
        self.partitions().to_vec()
    }

    fn num_blocks(&self) -> Vec<usize> {
        self.num_blocks()
    }

    fn get_block(&self, block_idx: &BlockIndex) -> Option<BlockData<T>> {
        BlockArray::get_block(self, block_idx).cloned()
    }

    fn iter_nonzero_blocks(&self) -> Vec<(BlockIndex, BlockData<T>)> {
        self.iter_blocks()
            .map(|(idx, data)| (idx, data.clone()))
            .collect()
    }

    fn structure(&self) -> BlockStructure {
        self.structure.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mdarray::Tensor;

    fn make_block_data(shape: &[usize], start_val: f64) -> BlockData<f64> {
        let tensor = Tensor::from_fn(shape, |idx| {
            let linear: usize = idx.iter().enumerate().fold(0, |acc, (i, &x)| {
                let stride: usize = shape[i + 1..].iter().product();
                acc + x * stride.max(1)
            });
            start_val + linear as f64
        });
        BlockData::from_tensor(tensor)
    }

    fn tensor_from_slice(shape: &[usize], data: &[f64]) -> Tensor<f64> {
        let mut iter = data.iter();
        Tensor::from_fn(shape, |_| *iter.next().unwrap())
    }

    #[test]
    fn test_tensordot_matrix_multiply() {
        // Test: A(2x3) @ B(3x4) = C(2x4)
        // Single block case (trivial partitions)
        let part_2 = BlockPartition::trivial(2);
        let part_3 = BlockPartition::trivial(3);
        let part_4 = BlockPartition::trivial(4);

        // A: shape (2, 3)
        let mut a = BlockArray::new(vec![part_2.clone(), part_3.clone()]);
        // [[1, 2, 3], [4, 5, 6]]
        let a_data =
            BlockData::from_tensor(tensor_from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        a.set_block(vec![0, 0], a_data);

        // B: shape (3, 4)
        let mut b = BlockArray::new(vec![part_3, part_4]);
        // [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        let b_data = BlockData::from_tensor(tensor_from_slice(
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        ));
        b.set_block(vec![0, 0], b_data);

        // C = A @ B
        let c = a.tensordot(&b, &[1], &[0]);

        assert_eq!(c.shape(), vec![2, 4]);
        assert_eq!(c.num_nonzero_blocks(), 1);

        // Expected: [[38, 44, 50, 56], [83, 98, 113, 128]]
        let c_block = c.get_block(&vec![0, 0]).unwrap();
        let c_shape = c_block.shape();
        assert_eq!(c_shape, vec![2, 4]);
    }

    #[test]
    #[ignore] // TODO: Fix test - uniform(4, 2) creates blocks of size 4, not 2
    fn test_tensordot_blocked_matmul() {
        // Test: 2x2 blocked matrix multiplication
        // A: 2 blocks per axis (total 4 blocks), B: 2 blocks per axis
        let part_a = BlockPartition::uniform(4, 2); // 2 blocks of size 2
        let part_b = BlockPartition::uniform(4, 2);
        let part_c = BlockPartition::uniform(4, 2);

        // A: (4, 4) with 2x2 block structure
        let mut a = BlockArray::new(vec![part_a.clone(), part_b.clone()]);
        // Set blocks A[0,0] and A[1,1] (diagonal)
        a.set_block(vec![0, 0], make_block_data(&[2, 2], 1.0));
        a.set_block(vec![1, 1], make_block_data(&[2, 2], 5.0));

        // B: (4, 4) with 2x2 block structure
        let mut b = BlockArray::new(vec![part_b, part_c]);
        // Set blocks B[0,0] and B[1,0]
        b.set_block(vec![0, 0], make_block_data(&[2, 2], 1.0));
        b.set_block(vec![1, 0], make_block_data(&[2, 2], 1.0));

        // C = A @ B (contract axis 1 of A with axis 0 of B)
        let c = a.tensordot(&b, &[1], &[0]);

        assert_eq!(c.shape(), vec![4, 4]);
        // A[0,0] @ B[0,0] -> C[0,0]
        // A[1,1] @ B[1,0] -> C[1,0]
        assert_eq!(c.num_nonzero_blocks(), 2);

        // Verify block indices
        assert!(c.get_block(&vec![0, 0]).is_some());
        assert!(c.get_block(&vec![1, 0]).is_some());
        assert!(c.get_block(&vec![0, 1]).is_none());
        assert!(c.get_block(&vec![1, 1]).is_none());
    }

    #[test]
    #[ignore] // TODO: Fix test - uniform(4, 2) creates blocks of size 4, not 2
    fn test_tensordot_accumulation() {
        // Test: Multiple contributions to same output block
        // A[0,0], A[0,1] present; B[0,0], B[1,0] present
        // C[0,0] = A[0,0]@B[0,0] + A[0,1]@B[1,0]
        let part = BlockPartition::uniform(4, 2);

        let mut a = BlockArray::new(vec![part.clone(), part.clone()]);
        a.set_block(vec![0, 0], make_block_data(&[2, 2], 1.0));
        a.set_block(vec![0, 1], make_block_data(&[2, 2], 1.0));

        let mut b = BlockArray::new(vec![part.clone(), part]);
        b.set_block(vec![0, 0], make_block_data(&[2, 2], 1.0));
        b.set_block(vec![1, 0], make_block_data(&[2, 2], 1.0));

        let c = a.tensordot(&b, &[1], &[0]);

        assert_eq!(c.shape(), vec![4, 4]);
        // Only C[0,0] should be non-zero (from accumulation)
        assert_eq!(c.num_nonzero_blocks(), 1);
        assert!(c.get_block(&vec![0, 0]).is_some());
    }

    #[test]
    fn test_tensordot_3d_contract_middle() {
        // A: (2, 3, 4), B: (3, 5)
        // Contract axis 1 of A with axis 0 of B
        // Result: (2, 4, 5)
        let part_2 = BlockPartition::trivial(2);
        let part_3 = BlockPartition::trivial(3);
        let part_4 = BlockPartition::trivial(4);
        let part_5 = BlockPartition::trivial(5);

        let mut a = BlockArray::new(vec![part_2.clone(), part_3.clone(), part_4.clone()]);
        a.set_block(vec![0, 0, 0], make_block_data(&[2, 3, 4], 1.0));

        let mut b = BlockArray::new(vec![part_3, part_5]);
        b.set_block(vec![0, 0], make_block_data(&[3, 5], 1.0));

        let c = a.tensordot(&b, &[1], &[0]);

        assert_eq!(c.shape(), vec![2, 4, 5]);
        assert_eq!(c.num_nonzero_blocks(), 1);
    }

    #[test]
    fn test_reshape_2d_to_1d() {
        // 2D array [2, 3] -> 1D array [6]
        let part_2 = BlockPartition::trivial(2);
        let part_3 = BlockPartition::trivial(3);

        let mut a = BlockArray::new(vec![part_2, part_3]);
        // [[1, 2, 3], [4, 5, 6]]
        let data =
            BlockData::from_tensor(tensor_from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        a.set_block(vec![0, 0], data);

        let b = a.reshape(&[6]);

        assert_eq!(b.rank(), 1);
        assert_eq!(b.shape(), vec![6]);
        assert_eq!(b.num_nonzero_blocks(), 1);

        let b_block = b.get_block(&vec![0]).unwrap();
        assert_eq!(b_block.shape(), vec![6]);
    }

    #[test]
    fn test_reshape_1d_to_2d() {
        // 1D array [6] -> 2D array [2, 3]
        let part_6 = BlockPartition::trivial(6);

        let mut a = BlockArray::new(vec![part_6]);
        let data = BlockData::from_tensor(tensor_from_slice(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        a.set_block(vec![0], data);

        let b = a.reshape(&[2, 3]);

        assert_eq!(b.rank(), 2);
        assert_eq!(b.shape(), vec![2, 3]);
        assert_eq!(b.num_nonzero_blocks(), 1);

        let b_block = b.get_block(&vec![0, 0]).unwrap();
        assert_eq!(b_block.shape(), vec![2, 3]);
    }

    #[test]
    fn test_reshape_2d_to_3d() {
        // 2D array [4, 6] -> 3D array [2, 2, 6]
        let part_4 = BlockPartition::trivial(4);
        let part_6 = BlockPartition::trivial(6);

        let mut a = BlockArray::new(vec![part_4, part_6]);
        let data = make_block_data(&[4, 6], 1.0);
        a.set_block(vec![0, 0], data);

        let b = a.reshape(&[2, 2, 6]);

        assert_eq!(b.rank(), 3);
        assert_eq!(b.shape(), vec![2, 2, 6]);
        assert_eq!(b.num_nonzero_blocks(), 1);

        let b_block = b.get_block(&vec![0, 0, 0]).unwrap();
        assert_eq!(b_block.shape(), vec![2, 2, 6]);
    }

    #[test]
    fn test_reshape_preserves_elements() {
        // Verify that reshape preserves element count
        let part = BlockPartition::trivial(12);

        let mut a = BlockArray::new(vec![part]);
        let data = BlockData::from_tensor(tensor_from_slice(
            &[12],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        ));
        a.set_block(vec![0], data);

        // [12] -> [3, 4] -> [2, 6] -> [12]
        let b = a.reshape(&[3, 4]);
        let c = b.reshape(&[2, 6]);
        let d = c.reshape(&[12]);

        assert_eq!(d.shape(), vec![12]);
        assert_eq!(d.num_nonzero_blocks(), 1);
    }

    #[test]
    fn test_reshape_round_trip_2d_3d_2d() {
        // Test round-trip: 2D -> 3D -> 2D should recover original structure
        let part_6 = BlockPartition::trivial(6);
        let part_4 = BlockPartition::trivial(4);

        let mut original = BlockArray::new(vec![part_6, part_4]);
        let data = make_block_data(&[6, 4], 1.0);
        original.set_block(vec![0, 0], data);

        // Helper to count elements in non-zero blocks
        let count_elements = |arr: &BlockArray<f64>| -> usize {
            arr.iter_blocks().map(|(_, block)| block.len()).sum()
        };

        let elements_before = count_elements(&original);
        assert_eq!(elements_before, 24);

        // 2D [6, 4] -> 3D [2, 3, 4]
        let reshaped_3d = original.reshape(&[2, 3, 4]);
        assert_eq!(
            count_elements(&reshaped_3d),
            elements_before,
            "Elements must be preserved in 2D->3D"
        );
        assert_eq!(reshaped_3d.shape(), vec![2, 3, 4]);

        // 3D [2, 3, 4] -> 2D [6, 4]
        let round_trip = reshaped_3d.reshape(&[6, 4]);
        assert_eq!(
            count_elements(&round_trip),
            elements_before,
            "Elements must be preserved in round-trip"
        );

        // Verify structure matches original
        assert_eq!(round_trip.rank(), original.rank());
        assert_eq!(round_trip.shape(), original.shape());
        assert_eq!(
            round_trip.num_nonzero_blocks(),
            original.num_nonzero_blocks()
        );
    }

    #[test]
    fn test_reshape_preserves_nonzero_elements_chain() {
        // Test that element count is preserved through multiple reshapes
        let part = BlockPartition::trivial(24);

        let mut arr = BlockArray::new(vec![part]);
        let data = make_block_data(&[24], 1.0);
        arr.set_block(vec![0], data);

        let count_elements =
            |a: &BlockArray<f64>| -> usize { a.iter_blocks().map(|(_, block)| block.len()).sum() };

        let initial_elements = count_elements(&arr);
        assert_eq!(initial_elements, 24);

        // [24] -> [4, 6]
        let s1 = arr.reshape(&[4, 6]);
        assert_eq!(count_elements(&s1), initial_elements);
        assert_eq!(s1.shape(), vec![4, 6]);

        // [4, 6] -> [2, 2, 6]
        let s2 = s1.reshape(&[2, 2, 6]);
        assert_eq!(count_elements(&s2), initial_elements);
        assert_eq!(s2.shape(), vec![2, 2, 6]);

        // [2, 2, 6] -> [2, 12]
        let s3 = s2.reshape(&[2, 12]);
        assert_eq!(count_elements(&s3), initial_elements);
        assert_eq!(s3.shape(), vec![2, 12]);

        // [2, 12] -> [24]
        let s4 = s3.reshape(&[24]);
        assert_eq!(count_elements(&s4), initial_elements);
        assert_eq!(s4.shape(), vec![24]);
    }

    #[test]
    fn test_plan_reshape_to_and_reshape_with_plan() {
        // Test that plan_reshape_to + reshape_with_plan gives same result as reshape_to
        let part_6 = BlockPartition::trivial(6);
        let part_4 = BlockPartition::trivial(4);

        let mut arr = BlockArray::new(vec![part_6, part_4]);
        let data = make_block_data(&[6, 4], 1.0);
        arr.set_block(vec![0, 0], data);

        // Using reshape_to directly
        let new_partitions = vec![
            BlockPartition::trivial(2),
            BlockPartition::trivial(3),
            BlockPartition::trivial(4),
        ];
        let reshaped_direct = arr.reshape_to(new_partitions.clone());

        // Using plan_reshape_to + reshape_with_plan
        let plan = arr.plan_reshape_to(new_partitions);
        let reshaped_with_plan = arr.reshape_with_plan(&plan);

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

        // Verify block shapes match
        let block_direct = reshaped_direct.get_block(&vec![0, 0, 0]).unwrap();
        let block_with_plan = reshaped_with_plan.get_block(&vec![0, 0, 0]).unwrap();
        assert_eq!(block_direct.shape(), block_with_plan.shape());
    }

    #[test]
    fn test_reshape_plan_reuse_for_arrays() {
        // Test that the same plan can be used for multiple arrays with same structure
        let part_6 = BlockPartition::trivial(6);
        let part_4 = BlockPartition::trivial(4);

        let mut arr1 = BlockArray::new(vec![part_6.clone(), part_4.clone()]);
        arr1.set_block(vec![0, 0], make_block_data(&[6, 4], 1.0));

        let mut arr2 = BlockArray::new(vec![part_6, part_4]);
        arr2.set_block(vec![0, 0], make_block_data(&[6, 4], 100.0)); // Different values

        // Create plan from arr1
        let new_partitions = vec![BlockPartition::trivial(2), BlockPartition::trivial(12)];
        let plan = arr1.plan_reshape_to(new_partitions);

        // Use same plan for both arrays
        let reshaped1 = arr1.reshape_with_plan(&plan);
        let reshaped2 = arr2.reshape_with_plan(&plan);

        // Both should have same structure
        assert_eq!(reshaped1.shape(), reshaped2.shape());
        assert_eq!(
            reshaped1.num_nonzero_blocks(),
            reshaped2.num_nonzero_blocks()
        );

        // But different data
        let block1 = reshaped1.get_block(&vec![0, 0]).unwrap();
        let block2 = reshaped2.get_block(&vec![0, 0]).unwrap();
        assert_eq!(block1.shape(), block2.shape());
    }
}
