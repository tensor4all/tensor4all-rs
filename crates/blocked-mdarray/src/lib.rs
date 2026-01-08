//! Blocked multi-dimensional arrays with sparse-matrix-style operations.
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
//! - [`Scalar`]: Trait for scalar types (f64, Complex64)
//! - [`BlockPartition`]: Defines how an axis is divided into blocks
//! - [`BlockedData`]: Owned N-D block data (wraps mdarray's `Tensor<T>`)
//! - [`BlockStructure`]: Block structure metadata (partitions + sparsity pattern)
//! - [`BlockedArray`]: Owned blocked array (wraps `BlockStructure` + data)
//!
//! # Example
//!
//! ```
//! use blocked_mdarray::{BlockPartition, BlockedArray};
//!
//! // Create a 4x4 matrix split into 2x2 blocks (f64)
//! let parts = vec![
//!     BlockPartition::uniform(2, 2),
//!     BlockPartition::uniform(2, 2),
//! ];
//! let matrix = BlockedArray::<f64>::new(parts);
//!
//! assert_eq!(matrix.shape(), vec![4, 4]);
//! assert_eq!(matrix.num_blocks(), vec![2, 2]);
//! assert_eq!(matrix.num_nonzero_blocks(), 0);
//! ```

mod block_structure;
mod blocked_array;
mod blocked_data;
mod error;
mod partition;
mod scalar;
mod tensor_network;

pub use block_structure::{BlockStructure, ReshapePlan};
pub use blocked_array::{BlockedArray, BlockedArrayLike};
pub use blocked_data::{BlockedData, BlockedDataLike};
pub use error::{BlockedArrayError, Result};
pub use partition::{block_linear_index, block_multi_index, BlockIndex, BlockPartition};
pub use scalar::Scalar;
pub use tensor_network::{EdgeInfo, EdgeLabel, TensorId, TensorNetwork};
