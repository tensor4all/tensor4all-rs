//! Error types for einsum operations.

use thiserror::Error;

/// Error types for einsum operations.
#[derive(Debug, Error)]
pub enum EinsumError {
    /// Dimension mismatch between operands for the same axis.
    #[error("Dimension mismatch: axis {axis_id:?} has size {size_a} in operand {op_a} but size {size_b} in operand {op_b}")]
    DimensionMismatch {
        /// The axis identifier where the mismatch occurred.
        axis_id: String,
        /// The size of the axis in the first operand.
        size_a: usize,
        /// The size of the axis in the second operand.
        size_b: usize,
        /// The index of the first operand.
        op_a: usize,
        /// The index of the second operand.
        op_b: usize,
    },

    /// An output axis was not found in any input operand.
    #[error("Output axis {axis_id:?} not found in any input operand")]
    OutputAxisNotFound {
        /// The axis identifier that was not found.
        axis_id: String,
    },

    /// The operand list is empty.
    #[error("Empty operand list")]
    EmptyOperands,

    /// An invalid contraction path was specified.
    #[error("Invalid contraction path: index {index} out of bounds for {num_ops} operands")]
    InvalidPath {
        /// The invalid index in the contraction path.
        index: usize,
        /// The number of operands available.
        num_ops: usize,
    },
}
