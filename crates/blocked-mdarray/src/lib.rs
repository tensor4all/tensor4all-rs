//! Blocked multi-dimensional arrays with sparse-matrix-style operations.
//!
//! This crate provides efficient data structures for block-sparse tensors,
//! where only certain blocks contain non-zero data. This is particularly
//! useful for tensors with symmetry constraints (e.g., quantum number conservation).
//!
//! # Design
//!
//! The design follows mdarray's view-based pattern:
//! - Built on top of mdarray's `DTensor` and `DSlice` types
//! - Transposition returns views (no data copy)
//! - Data is only copied when materialized via `to_owned()`
//! - Blocks are stored in a sparse HashMap
//!
//! # Core Types
//!
//! - [`BlockPartition`]: Defines how an axis is divided into blocks
//! - [`BlockData`]: Owned 2D block data (wraps mdarray's `DTensor<f64, 2>`)
//! - [`BlockedArray`]: Owned blocked array
//! - [`BlockedView`]: Borrowed view (supports lazy transposition)
//!
//! # Example
//!
//! ```
//! use blocked_mdarray::{BlockPartition, BlockedArray, BlockData, blocked_matmul};
//!
//! // Create a 4x4 matrix split into 2x2 blocks
//! let parts = vec![
//!     BlockPartition::uniform(2, 2),
//!     BlockPartition::uniform(2, 2),
//! ];
//! let mut matrix = BlockedArray::new(parts);
//!
//! // Set only diagonal blocks (block-diagonal matrix)
//! matrix.set_block(vec![0, 0], BlockData::new(vec![1.0, 0.0, 0.0, 1.0], [2, 2]));
//! matrix.set_block(vec![1, 1], BlockData::new(vec![2.0, 0.0, 0.0, 2.0], [2, 2]));
//!
//! // Sparse multiplication: only 2 block multiplications instead of 4
//! let result = blocked_matmul(&matrix, &matrix).unwrap();
//! assert_eq!(result.num_nonzero_blocks(), 2);
//! ```

mod block_data;
mod blocked_array;
mod error;
mod matmul;
mod partition;

pub use block_data::{BlockData, BlockSlice2, BlockSliceStrided2, BlockTensor2};
pub use blocked_array::{BlockedArray, BlockedArrayLike, BlockedView};
pub use error::{BlockedArrayError, Result};
pub use matmul::blocked_matmul;
pub use partition::{block_linear_index, block_multi_index, BlockIndex, BlockPartition};
