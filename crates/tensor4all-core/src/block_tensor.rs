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

use std::collections::HashSet;

use crate::any_scalar::AnyScalar;
use crate::index_like::IndexLike;
use crate::tensor_index::TensorIndex;
use crate::tensor_like::{
    AllowedPairs, DirectSumResult, FactorizeError, FactorizeOptions, FactorizeResult, TensorLike,
};
use anyhow::Result;

/// A collection of tensors organized in a block structure.
///
/// Each block is a tensor of type `T` implementing [`TensorLike`].
/// The flattened block list is ordered row-by-row:
/// `(0, 0), (0, 1), ..., (1, 0), (1, 1), ...`.
///
/// # Type Parameters
///
/// * `T` - The tensor type for each block, must implement `TensorLike`
#[derive(Debug, Clone)]
pub struct BlockTensor<T: TensorLike> {
    /// Blocks flattened row-by-row in block-matrix order
    blocks: Vec<T>,
    /// Block structure (rows, cols)
    shape: (usize, usize),
}

impl<T: TensorLike> BlockTensor<T> {
    /// Create a new block tensor with validation.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Vector of blocks flattened row-by-row
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
    /// * `blocks` - Vector of blocks flattened row-by-row
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

    /// Validate that blocks share external indices consistently.
    ///
    /// For column vectors (cols=1), no index sharing is required between
    /// blocks. Different rows can have independent physical indices
    /// (the operator determines their relationship).
    ///
    /// For matrices (rows x cols), checks that:
    /// - All blocks have the same number of external indices.
    /// - Blocks in the same row share some common index IDs (output indices).
    /// - Blocks in the same column share some common index IDs (input indices).
    pub fn validate_indices(&self) -> Result<()> {
        let (rows, cols) = self.shape;

        if cols <= 1 {
            // Column vector: blocks in different rows can have independent indices.
            // The operator determines the relationship between blocks.
            return Ok(());
        }

        // Matrix: check all blocks have the same number of external indices
        let first_count = self.blocks[0].num_external_indices();
        for (i, block) in self.blocks.iter().enumerate().skip(1) {
            let n = block.num_external_indices();
            anyhow::ensure!(
                n == first_count,
                "Block {} has {} external indices, but block 0 has {}",
                i,
                n,
                first_count
            );
        }

        // Same row: blocks should share some common index IDs (output indices)
        for row in 0..rows {
            let ref_ids: HashSet<_> = self
                .get(row, 0)
                .external_indices()
                .iter()
                .map(|idx| idx.id().clone())
                .collect();
            for col in 1..cols {
                let ids: HashSet<_> = self
                    .get(row, col)
                    .external_indices()
                    .iter()
                    .map(|idx| idx.id().clone())
                    .collect();
                let common_count = ref_ids.intersection(&ids).count();
                anyhow::ensure!(
                    common_count > 0,
                    "Matrix row {}: blocks ({},{}) and ({},{}) share no index IDs",
                    row,
                    row,
                    0,
                    row,
                    col
                );
            }
        }

        // Same column: blocks should share some common index IDs (input indices)
        for col in 0..cols {
            let ref_ids: HashSet<_> = self
                .get(0, col)
                .external_indices()
                .iter()
                .map(|idx| idx.id().clone())
                .collect();
            for row in 1..rows {
                let ids: HashSet<_> = self
                    .get(row, col)
                    .external_indices()
                    .iter()
                    .map(|idx| idx.id().clone())
                    .collect();
                let common_count = ref_ids.intersection(&ids).count();
                anyhow::ensure!(
                    common_count > 0,
                    "Matrix col {}: blocks ({},{}) and ({},{}) share no index IDs",
                    col,
                    0,
                    col,
                    row,
                    col
                );
            }
        }

        Ok(())
    }
}

// ============================================================================
// TensorIndex implementation
// ============================================================================

impl<T: TensorLike> TensorIndex for BlockTensor<T> {
    type Index = T::Index;

    fn external_indices(&self) -> Vec<Self::Index> {
        // Collect unique external indices across all blocks (deduplicated by ID).
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for block in &self.blocks {
            for idx in block.external_indices() {
                if seen.insert(idx.id().clone()) {
                    result.push(idx);
                }
            }
        }
        result
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

    fn maxabs(&self) -> f64 {
        self.blocks
            .iter()
            .map(|b| b.maxabs())
            .fold(0.0_f64, f64::max)
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
        let mut sum = AnyScalar::new_real(0.0);
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

    fn validate(&self) -> Result<()> {
        self.validate_indices()
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
mod tests;
