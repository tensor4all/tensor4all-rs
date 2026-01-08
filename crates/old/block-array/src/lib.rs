//! Block multi-dimensional arrays with sparse-matrix-style operations.
//!
//! This crate provides efficient data structures for block-sparse tensors,
//! where only certain blocks contain non-zero data. This is particularly
//! useful for tensors with symmetry constraints (e.g., quantum number conservation).
//!
//! # Design
//!
//! The design follows mdarray's view-based pattern:
//! - Built on top of mdarray's `Tensor` types
//! - Blocks are stored in a sparse HashMap
//! - Generic over scalar type `T` (f64, Complex64, etc.)
//!
//! # Core Types
//!
//! - [`BlockPartition`]: Defines how an axis is divided into blocks
//! - [`BlockData`]: Owned N-D block data (wraps mdarray's `Tensor<T>`)
//! - [`BlockStructure`]: Block structure metadata (partitions + sparsity pattern)
//! - [`BlockArray`]: Owned blocked array (wraps `BlockStructure` + data)
//!
//! # Example
//!
//! ```
//! use block_array::{BlockData, BlockArray, BlockPartition};
//!
//! // Create a 4x4 matrix split into 2x2 blocks (f64)
//! let parts = vec![
//!     BlockPartition::uniform(2, 2),
//!     BlockPartition::uniform(2, 2),
//! ];
//! let matrix = BlockArray::<BlockData<f64>>::new(parts);
//!
//! assert_eq!(matrix.shape(), vec![4, 4]);
//! assert_eq!(matrix.num_blocks(), vec![2, 2]);
//! assert_eq!(matrix.num_nonzero_blocks(), 0);
//! ```

mod block_array;
mod block_data;
mod block_structure;
mod error;
mod partition;
mod tensor_network;

pub use block_array::{BlockArray, BlockArrayLike};
pub use block_data::{BlockData, BlockDataLike};
pub use block_structure::{BlockStructure, ReshapePlan};
pub use error::{BlockArrayError, Result};
pub use partition::{block_linear_index, block_multi_index, BlockIndex, BlockPartition};
pub use tensor_network::{EdgeInfo, EdgeLabel, TensorId, TensorNetwork};
