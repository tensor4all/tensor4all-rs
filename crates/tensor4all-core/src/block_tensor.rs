//! Block tensor type for GMRES with block matrices.
//!
//! This module provides [`BlockTensor`], a collection of tensors organized
//! in a block structure. It implements [`TensorLike`] for the vector space
//! operations required by GMRES, allowing block matrix linear equations
//! `Ax = b` to be solved using the existing GMRES implementation.
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_core::block_tensor::BlockTensor;
//! use tensor4all_core::krylov::{gmres, GmresOptions};
//!
//! // Create 2x1 block vectors
//! let b = BlockTensor::new(vec![b1, b2], (2, 1));
//! let x0 = BlockTensor::new(vec![zero1, zero2], (2, 1));
//!
//! // Define block matrix operator
//! let apply_a = |x: &BlockTensor<T>| { /* ... */ };
//!
//! let result = gmres(apply_a, &b, &x0, &GmresOptions::default())?;
//! ```

use crate::any_scalar::AnyScalar;
use crate::tensor_index::TensorIndex;
use crate::tensor_like::{
    AllowedPairs, DirectSumResult, FactorizeError, FactorizeOptions, FactorizeResult, TensorLike,
};
use anyhow::Result;

/// A collection of tensors organized in a block structure.
///
/// Each block is a tensor of type `T` implementing [`TensorLike`].
/// The blocks are stored in row-major order (for 2D block matrices).
///
/// # Type Parameters
///
/// * `T` - The tensor type for each block, must implement `TensorLike`
#[derive(Debug, Clone)]
pub struct BlockTensor<T: TensorLike> {
    /// Blocks stored in row-major order
    blocks: Vec<T>,
    /// Block structure (rows, cols)
    shape: (usize, usize),
}

impl<T: TensorLike> BlockTensor<T> {
    /// Create a new block tensor with validation.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Vector of blocks in row-major order
    /// * `shape` - Block structure as (rows, cols)
    ///
    /// # Errors
    ///
    /// Returns an error if `rows * cols != blocks.len()`.
    pub fn try_new(blocks: Vec<T>, shape: (usize, usize)) -> Result<Self> {
        let (rows, cols) = shape;
        anyhow::ensure!(
            rows * cols == blocks.len(),
            "Block count mismatch: shape ({}, {}) requires {} blocks, but got {}",
            rows,
            cols,
            rows * cols,
            blocks.len()
        );
        Ok(Self { blocks, shape })
    }

    /// Create a new block tensor.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Vector of blocks in row-major order
    /// * `shape` - Block structure as (rows, cols)
    ///
    /// # Panics
    ///
    /// Panics if `rows * cols != blocks.len()`.
    pub fn new(blocks: Vec<T>, shape: (usize, usize)) -> Self {
        Self::try_new(blocks, shape).expect("Invalid block tensor shape")
    }

    /// Get the block structure (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the total number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get a reference to the block at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    pub fn get(&self, row: usize, col: usize) -> &T {
        let (rows, cols) = self.shape;
        assert!(row < rows && col < cols, "Block index out of bounds");
        &self.blocks[row * cols + col]
    }

    /// Get a mutable reference to the block at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        let (rows, cols) = self.shape;
        assert!(row < rows && col < cols, "Block index out of bounds");
        &mut self.blocks[row * cols + col]
    }

    /// Get all blocks as a slice.
    pub fn blocks(&self) -> &[T] {
        &self.blocks
    }

    /// Get all blocks as a mutable slice.
    pub fn blocks_mut(&mut self) -> &mut [T] {
        &mut self.blocks
    }

    /// Consume self and return the blocks.
    pub fn into_blocks(self) -> Vec<T> {
        self.blocks
    }
}

// ============================================================================
// TensorIndex implementation
// ============================================================================

impl<T: TensorLike> TensorIndex for BlockTensor<T> {
    type Index = T::Index;

    fn external_indices(&self) -> Vec<Self::Index> {
        // BlockTensor as a whole doesn't expose external indices
        // Each block has its own indices, but they are internal to the block structure
        vec![]
    }

    fn num_external_indices(&self) -> usize {
        0
    }

    fn replaceind(&self, old_index: &Self::Index, new_index: &Self::Index) -> Result<Self> {
        let replaced: Result<Vec<T>> = self
            .blocks
            .iter()
            .map(|b| b.replaceind(old_index, new_index))
            .collect();
        Ok(Self {
            blocks: replaced?,
            shape: self.shape,
        })
    }

    fn replaceinds(
        &self,
        old_indices: &[Self::Index],
        new_indices: &[Self::Index],
    ) -> Result<Self> {
        let replaced: Result<Vec<T>> = self
            .blocks
            .iter()
            .map(|b| b.replaceinds(old_indices, new_indices))
            .collect();
        Ok(Self {
            blocks: replaced?,
            shape: self.shape,
        })
    }
}

// ============================================================================
// TensorLike implementation
// ============================================================================

impl<T: TensorLike> TensorLike for BlockTensor<T> {
    // ------------------------------------------------------------------------
    // Vector space operations (required for GMRES)
    // ------------------------------------------------------------------------

    fn norm_squared(&self) -> f64 {
        self.blocks.iter().map(|b| b.norm_squared()).sum()
    }

    fn scale(&self, scalar: AnyScalar) -> Result<Self> {
        let scaled: Result<Vec<T>> = self
            .blocks
            .iter()
            .map(|b| b.scale(scalar.clone()))
            .collect();
        Ok(Self {
            blocks: scaled?,
            shape: self.shape,
        })
    }

    fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self> {
        anyhow::ensure!(
            self.shape == other.shape,
            "Block shapes must match: {:?} vs {:?}",
            self.shape,
            other.shape
        );
        let result: Result<Vec<T>> = self
            .blocks
            .iter()
            .zip(other.blocks.iter())
            .map(|(s, o)| s.axpby(a.clone(), o, b.clone()))
            .collect();
        Ok(Self {
            blocks: result?,
            shape: self.shape,
        })
    }

    fn inner_product(&self, other: &Self) -> Result<AnyScalar> {
        anyhow::ensure!(
            self.shape == other.shape,
            "Block shapes must match for inner product: {:?} vs {:?}",
            self.shape,
            other.shape
        );
        let mut sum = AnyScalar::F64(0.0);
        for (s, o) in self.blocks.iter().zip(other.blocks.iter()) {
            sum = sum + s.inner_product(o)?;
        }
        Ok(sum)
    }

    fn conj(&self) -> Self {
        let conjugated: Vec<T> = self.blocks.iter().map(|b| b.conj()).collect();
        Self {
            blocks: conjugated,
            shape: self.shape,
        }
    }

    // ------------------------------------------------------------------------
    // Operations not supported for BlockTensor (return error, don't panic)
    // ------------------------------------------------------------------------

    fn factorize(
        &self,
        _left_inds: &[<Self as TensorIndex>::Index],
        _options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        Err(FactorizeError::ComputationError(anyhow::anyhow!(
            "BlockTensor does not support factorize"
        )))
    }

    fn direct_sum(
        &self,
        _other: &Self,
        _pairs: &[(<Self as TensorIndex>::Index, <Self as TensorIndex>::Index)],
    ) -> Result<DirectSumResult<Self>> {
        anyhow::bail!("BlockTensor does not support direct_sum")
    }

    fn outer_product(&self, _other: &Self) -> Result<Self> {
        anyhow::bail!("BlockTensor does not support outer_product")
    }

    fn permuteinds(&self, _new_order: &[<Self as TensorIndex>::Index]) -> Result<Self> {
        anyhow::bail!("BlockTensor does not support permuteinds")
    }

    fn contract(_tensors: &[&Self], _allowed: AllowedPairs<'_>) -> Result<Self> {
        anyhow::bail!("BlockTensor does not support contract")
    }

    fn contract_connected(_tensors: &[&Self], _allowed: AllowedPairs<'_>) -> Result<Self> {
        anyhow::bail!("BlockTensor does not support contract_connected")
    }

    fn diagonal(
        _input_index: &<Self as TensorIndex>::Index,
        _output_index: &<Self as TensorIndex>::Index,
    ) -> Result<Self> {
        anyhow::bail!("BlockTensor does not support diagonal")
    }

    fn scalar_one() -> Result<Self> {
        anyhow::bail!("BlockTensor does not support scalar_one")
    }

    fn ones(_indices: &[<Self as TensorIndex>::Index]) -> Result<Self> {
        anyhow::bail!("BlockTensor does not support ones")
    }

    fn onehot(_index_vals: &[(<Self as TensorIndex>::Index, usize)]) -> Result<Self> {
        anyhow::bail!("BlockTensor does not support onehot")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::tensordynlen::TensorDynLen;
    use crate::defaults::DynIndex;
    use crate::krylov::{gmres, GmresOptions};
    use crate::storage::{DenseStorageF64, Storage};
    use std::sync::Arc;

    /// Helper to create a 1D tensor (vector) with given data and shared index.
    fn make_vector_with_index(data: Vec<f64>, idx: &DynIndex) -> TensorDynLen {
        let n = data.len();
        TensorDynLen::new(
            vec![idx.clone()],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                data,
                &[n],
            ))),
        )
    }

    // ========================================================================
    // Test 0: BlockTensor invariants
    // ========================================================================

    #[test]
    fn test_try_new_valid() {
        let idx = DynIndex::new_dyn(2);
        let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
        let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);

        let block = BlockTensor::try_new(vec![b1, b2], (2, 1));
        assert!(block.is_ok());
        let block = block.unwrap();
        assert_eq!(block.shape(), (2, 1));
        assert_eq!(block.num_blocks(), 2);
    }

    #[test]
    fn test_try_new_invalid_shape() {
        let idx = DynIndex::new_dyn(2);
        let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
        let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);

        // 2 blocks but shape says 3x1 = 3 blocks
        let result = BlockTensor::try_new(vec![b1, b2], (3, 1));
        assert!(result.is_err());
    }

    #[test]
    fn test_axpby_shape_mismatch() {
        let idx = DynIndex::new_dyn(2);
        let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
        let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
        let b3 = make_vector_with_index(vec![5.0, 6.0], &idx);

        let block_2x1 = BlockTensor::new(vec![b1.clone(), b2], (2, 1));
        let block_1x1 = BlockTensor::new(vec![b3], (1, 1));

        let result = block_2x1.axpby(AnyScalar::F64(1.0), &block_1x1, AnyScalar::F64(1.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_inner_product_shape_mismatch() {
        let idx = DynIndex::new_dyn(2);
        let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
        let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
        let b3 = make_vector_with_index(vec![5.0, 6.0], &idx);

        let block_2x1 = BlockTensor::new(vec![b1.clone(), b2], (2, 1));
        let block_1x1 = BlockTensor::new(vec![b3], (1, 1));

        let result = block_2x1.inner_product(&block_1x1);
        assert!(result.is_err());
    }

    // ========================================================================
    // Test 1: Identity operator GMRES
    // ========================================================================

    #[test]
    fn test_gmres_identity_block() {
        // Solve Ax = b where A = I (identity)
        // 2x1 block structure, each block is a 2-element vector
        let idx = DynIndex::new_dyn(2);

        let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
        let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
        let b = BlockTensor::new(vec![b1, b2], (2, 1));

        let zero1 = make_vector_with_index(vec![0.0, 0.0], &idx);
        let zero2 = make_vector_with_index(vec![0.0, 0.0], &idx);
        let x0 = BlockTensor::new(vec![zero1, zero2], (2, 1));

        // Identity operator: A x = x
        let apply_a =
            |x: &BlockTensor<TensorDynLen>| -> Result<BlockTensor<TensorDynLen>> { Ok(x.clone()) };

        let options = GmresOptions {
            max_iter: 10,
            rtol: 1e-10,
            max_restarts: 1,
            verbose: false,
            check_true_residual: false,
        };

        let result = gmres(apply_a, &b, &x0, &options).unwrap();

        assert!(result.converged, "GMRES should converge for identity");
        assert!(
            result.residual_norm < 1e-10,
            "Residual should be small: {}",
            result.residual_norm
        );

        // Check solution matches b
        let diff = result
            .solution
            .axpby(AnyScalar::F64(1.0), &b, AnyScalar::F64(-1.0))
            .unwrap();
        assert!(diff.norm() < 1e-10, "Solution should equal b");
    }

    // ========================================================================
    // Test 2: Diagonal block matrix GMRES
    // ========================================================================

    #[test]
    fn test_gmres_diagonal_block() {
        // A = [[D1, 0], [0, D2]] where D1 = diag(2, 3), D2 = diag(4, 5)
        // b = [[2, 3]^T, [4, 5]^T] -> x = [[1, 1]^T, [1, 1]^T]
        let idx = DynIndex::new_dyn(2);

        let b1 = make_vector_with_index(vec![2.0, 3.0], &idx);
        let b2 = make_vector_with_index(vec![4.0, 5.0], &idx);
        let b = BlockTensor::new(vec![b1, b2], (2, 1));

        let zero1 = make_vector_with_index(vec![0.0, 0.0], &idx);
        let zero2 = make_vector_with_index(vec![0.0, 0.0], &idx);
        let x0 = BlockTensor::new(vec![zero1, zero2], (2, 1));

        let expected1 = make_vector_with_index(vec![1.0, 1.0], &idx);
        let expected2 = make_vector_with_index(vec![1.0, 1.0], &idx);
        let expected = BlockTensor::new(vec![expected1, expected2], (2, 1));

        // Diagonal block operator
        let diag1 = [2.0, 3.0];
        let diag2 = [4.0, 5.0];

        let apply_a = move |x: &BlockTensor<TensorDynLen>| -> Result<BlockTensor<TensorDynLen>> {
            let x1 = x.get(0, 0);
            let x2 = x.get(1, 0);

            // Apply D1 to x1
            let x1_data = match x1.storage().as_ref() {
                Storage::DenseF64(d) => d.as_slice().to_vec(),
                _ => anyhow::bail!("Expected DenseF64"),
            };
            let y1_data: Vec<f64> = x1_data
                .iter()
                .zip(diag1.iter())
                .map(|(&xi, &di)| xi * di)
                .collect();
            let y1 = TensorDynLen::new(
                x1.indices.clone(),
                Arc::new(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                    y1_data,
                    &x1.dims(),
                ))),
            );

            // Apply D2 to x2
            let x2_data = match x2.storage().as_ref() {
                Storage::DenseF64(d) => d.as_slice().to_vec(),
                _ => anyhow::bail!("Expected DenseF64"),
            };
            let y2_data: Vec<f64> = x2_data
                .iter()
                .zip(diag2.iter())
                .map(|(&xi, &di)| xi * di)
                .collect();
            let y2 = TensorDynLen::new(
                x2.indices.clone(),
                Arc::new(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                    y2_data,
                    &x2.dims(),
                ))),
            );

            Ok(BlockTensor::new(vec![y1, y2], (2, 1)))
        };

        let options = GmresOptions {
            max_iter: 10,
            rtol: 1e-10,
            max_restarts: 1,
            verbose: false,
            check_true_residual: false,
        };

        let result = gmres(apply_a, &b, &x0, &options).unwrap();

        assert!(result.converged, "GMRES should converge");

        // Check solution
        let diff = result
            .solution
            .axpby(AnyScalar::F64(1.0), &expected, AnyScalar::F64(-1.0))
            .unwrap();
        assert!(
            diff.norm() < 1e-8,
            "Solution error too large: {}",
            diff.norm()
        );
    }

    // ========================================================================
    // Test 3: Upper triangular block matrix GMRES
    // ========================================================================

    #[test]
    fn test_gmres_upper_triangular_block() {
        // A = [[I, B], [0, I]] where B = I (identity)
        // Ax = [x1 + x2, x2]^T
        // b = [[2, 3]^T, [1, 1]^T]
        // x2 = [1, 1]^T, x1 = [2, 3]^T - [1, 1]^T = [1, 2]^T
        let idx = DynIndex::new_dyn(2);

        let b1 = make_vector_with_index(vec![2.0, 3.0], &idx);
        let b2 = make_vector_with_index(vec![1.0, 1.0], &idx);
        let b = BlockTensor::new(vec![b1, b2], (2, 1));

        let zero1 = make_vector_with_index(vec![0.0, 0.0], &idx);
        let zero2 = make_vector_with_index(vec![0.0, 0.0], &idx);
        let x0 = BlockTensor::new(vec![zero1, zero2], (2, 1));

        let expected1 = make_vector_with_index(vec![1.0, 2.0], &idx);
        let expected2 = make_vector_with_index(vec![1.0, 1.0], &idx);
        let expected = BlockTensor::new(vec![expected1, expected2], (2, 1));

        // Upper triangular block operator: A = [[I, I], [0, I]]
        let apply_a = |x: &BlockTensor<TensorDynLen>| -> Result<BlockTensor<TensorDynLen>> {
            let x1 = x.get(0, 0);
            let x2 = x.get(1, 0);

            // y1 = x1 + x2
            let y1 = x1.axpby(AnyScalar::F64(1.0), x2, AnyScalar::F64(1.0))?;
            // y2 = x2
            let y2 = x2.clone();

            Ok(BlockTensor::new(vec![y1, y2], (2, 1)))
        };

        let options = GmresOptions {
            max_iter: 10,
            rtol: 1e-10,
            max_restarts: 3,
            verbose: false,
            check_true_residual: false,
        };

        let result = gmres(apply_a, &b, &x0, &options).unwrap();

        assert!(result.converged, "GMRES should converge");

        // Check solution
        let diff = result
            .solution
            .axpby(AnyScalar::F64(1.0), &expected, AnyScalar::F64(-1.0))
            .unwrap();
        assert!(
            diff.norm() < 1e-8,
            "Solution error too large: {}",
            diff.norm()
        );
    }

    // ========================================================================
    // Test: Basic vector space operations
    // ========================================================================

    #[test]
    fn test_norm_squared() {
        let idx = DynIndex::new_dyn(2);
        let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
        let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
        let block = BlockTensor::new(vec![b1, b2], (2, 1));

        // norm_squared = 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
        let norm_sq = block.norm_squared();
        assert!((norm_sq - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale() {
        let idx = DynIndex::new_dyn(2);
        let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
        let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
        let block = BlockTensor::new(vec![b1, b2], (2, 1));

        let scaled = block.scale(AnyScalar::F64(2.0)).unwrap();

        // Check scaled values
        let expected1 = make_vector_with_index(vec![2.0, 4.0], &idx);
        let expected2 = make_vector_with_index(vec![6.0, 8.0], &idx);
        let expected = BlockTensor::new(vec![expected1, expected2], (2, 1));

        let diff = scaled
            .axpby(AnyScalar::F64(1.0), &expected, AnyScalar::F64(-1.0))
            .unwrap();
        assert!(diff.norm() < 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let idx = DynIndex::new_dyn(2);
        let a1 = make_vector_with_index(vec![1.0, 2.0], &idx);
        let a2 = make_vector_with_index(vec![3.0, 4.0], &idx);
        let a = BlockTensor::new(vec![a1, a2], (2, 1));

        let b1 = make_vector_with_index(vec![5.0, 6.0], &idx);
        let b2 = make_vector_with_index(vec![7.0, 8.0], &idx);
        let b = BlockTensor::new(vec![b1, b2], (2, 1));

        // inner_product = (1*5 + 2*6) + (3*7 + 4*8) = 17 + 53 = 70
        let ip = a.inner_product(&b).unwrap();
        assert!((ip.real() - 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_conj() {
        let idx = DynIndex::new_dyn(2);
        let b1 = make_vector_with_index(vec![1.0, 2.0], &idx);
        let b2 = make_vector_with_index(vec![3.0, 4.0], &idx);
        let block = BlockTensor::new(vec![b1, b2], (2, 1));

        // For real tensors, conj should be identity
        let conjugated = block.conj();
        let diff = conjugated
            .axpby(AnyScalar::F64(1.0), &block, AnyScalar::F64(-1.0))
            .unwrap();
        assert!(diff.norm() < 1e-10);
    }
}
