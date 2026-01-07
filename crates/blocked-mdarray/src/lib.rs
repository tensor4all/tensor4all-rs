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
//! - Generic over scalar type `T` (f64, Complex64, etc.)
//!
//! # Core Types
//!
//! - [`Scalar`]: Trait for scalar types (f64, Complex64)
//! - [`BlockPartition`]: Defines how an axis is divided into blocks
//! - [`BlockData`]: Owned 2D block data (wraps mdarray's `DTensor<T, 2>`)
//! - [`BlockStructure`]: Block structure metadata (partitions + sparsity pattern)
//! - [`BlockedArray`]: Owned blocked array (wraps `BlockStructure` + data)
//! - [`BlockedView`]: Borrowed view (supports lazy transposition)
//!
//! # Example
//!
//! ```
//! use blocked_mdarray::{BlockPartition, BlockedArray, BlockData, blocked_matmul};
//!
//! // Create a 4x4 matrix split into 2x2 blocks (f64)
//! let parts = vec![
//!     BlockPartition::uniform(2, 2),
//!     BlockPartition::uniform(2, 2),
//! ];
//! let mut matrix = BlockedArray::<f64>::new(parts);
//!
//! // Set only diagonal blocks (block-diagonal matrix)
//! matrix.set_block(vec![0, 0], BlockData::new(vec![1.0, 0.0, 0.0, 1.0], [2, 2]));
//! matrix.set_block(vec![1, 1], BlockData::new(vec![2.0, 0.0, 0.0, 2.0], [2, 2]));
//!
//! // Sparse multiplication: only 2 block multiplications instead of 4
//! let result = blocked_matmul(&matrix, &matrix).unwrap();
//! assert_eq!(result.num_nonzero_blocks(), 2);
//! ```
//!
//! # Complex64 Example
//!
//! ```
//! use blocked_mdarray::{BlockPartition, BlockedArray, BlockData, blocked_matmul};
//! use num_complex::Complex64;
//!
//! // Create a complex blocked array
//! let parts = vec![
//!     BlockPartition::trivial(2),
//!     BlockPartition::trivial(2),
//! ];
//! let mut matrix = BlockedArray::<Complex64>::new(parts);
//!
//! // Set with complex values
//! matrix.set_block(
//!     vec![0, 0],
//!     BlockData::new(
//!         vec![
//!             Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0),
//!             Complex64::new(0.0, -1.0), Complex64::new(1.0, 0.0),
//!         ],
//!         [2, 2],
//!     ),
//! );
//! ```

mod block_data;
mod block_structure;
mod blocked_array;
mod blocked_data;
mod error;
mod matmul;
mod partition;
mod scalar;

pub use block_data::{BlockData, BlockSlice2, BlockSliceStrided2, BlockTensor2};
pub use block_structure::BlockStructure;
pub use blocked_array::{BlockedArray, BlockedArrayLike, BlockedView};
pub use blocked_data::{linear_to_multi, multi_to_linear, BlockedData};
pub use error::{BlockedArrayError, Result};
pub use matmul::blocked_matmul;
pub use partition::{block_linear_index, block_multi_index, BlockIndex, BlockPartition};
pub use scalar::Scalar;
