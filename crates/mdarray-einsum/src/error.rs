//! Error types for einsum operations.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EinsumError {
    #[error("Dimension mismatch: axis {axis_id:?} has size {size_a} in operand {op_a} but size {size_b} in operand {op_b}")]
    DimensionMismatch {
        axis_id: String,
        size_a: usize,
        size_b: usize,
        op_a: usize,
        op_b: usize,
    },

    #[error("Output axis {axis_id:?} not found in any input operand")]
    OutputAxisNotFound { axis_id: String },

    #[error("Empty operand list")]
    EmptyOperands,

    #[error("Invalid contraction path: index {index} out of bounds for {num_ops} operands")]
    InvalidPath { index: usize, num_ops: usize },
}
