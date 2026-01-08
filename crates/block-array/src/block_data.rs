//! N-dimensional block data storage.

use std::fmt::Debug;

use mdarray::expr::Expression;
use mdarray::Tensor;
use mdarray_linalg::matmul::{ContractBuilder, MatMul};
use mdarray_linalg_faer::Faer;

use crate::scalar::Scalar;

/// Trait for block data storage backends.
///
/// This trait abstracts over different storage backends (CPU, GPU, etc.)
/// allowing `BlockArray` to work with any compatible implementation.
pub trait BlockDataLike<T: Scalar>: Debug + Clone {
    /// Get the rank (number of dimensions).
    fn rank(&self) -> usize;

    /// Get the shape.
    fn shape(&self) -> Vec<usize>;

    /// Get the total number of elements.
    fn len(&self) -> usize;

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Permute axes, returning a new owned instance.
    fn permute(&self, perm: &[usize]) -> Self;

    /// Reshape, returning a new owned instance.
    fn reshape(&self, shape: &[usize]) -> Self;
}

/// N-dimensional block data storage using mdarray.
///
/// Wraps `mdarray::Tensor<T>` (dynamic rank) for block operations.
#[derive(Debug, Clone)]
pub struct BlockData<T: Scalar> {
    tensor: Tensor<T>,
}

impl<T: Scalar> BlockData<T> {
    /// Create a new BlockData from a Tensor.
    pub fn from_tensor(tensor: Tensor<T>) -> Self {
        Self { tensor }
    }

    /// Matrix multiplication for 2D tensors.
    ///
    /// Computes C = A * B where A is (m, k) and B is (k, n).
    /// Uses Faer backend for efficient BLAS-like operations.
    ///
    /// # Panics
    /// - If either tensor is not 2D
    /// - If inner dimensions don't match
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.rank(), 2, "matmul requires 2D tensors");
        assert_eq!(other.rank(), 2, "matmul requires 2D tensors");
        assert_eq!(
            self.shape()[1],
            other.shape()[0],
            "Inner dimensions must match"
        );

        // Use Faer's contract_n for DynRank tensors (contracts last 1 axis of A with first 1 axis of B)
        let tensor = Faer.contract_n(&self.tensor, &other.tensor, 1).eval();

        Self { tensor }
    }

    /// Dot product for 1D tensors (vectors).
    ///
    /// Computes sum(a * b) and returns a scalar value.
    ///
    /// # Panics
    /// - If either tensor is not 1D
    /// - If lengths don't match
    pub fn dot(&self, other: &Self) -> T {
        assert_eq!(self.rank(), 1, "dot requires 1D tensors");
        assert_eq!(other.rank(), 1, "dot requires 1D tensors");
        assert_eq!(
            self.shape()[0],
            other.shape()[0],
            "Vector lengths must match for dot product"
        );

        // Compute dot product: sum of element-wise products
        self.tensor
            .expr()
            .zip(other.tensor.expr())
            .map(|(a, b)| *a * *b)
            .fold(T::zero(), |acc, x| acc + x)
    }

    /// Element-wise addition.
    ///
    /// Uses mdarray's expression system for efficient element-wise operations.
    ///
    /// # Panics
    /// - If shapes don't match
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Shapes must match for addition"
        );

        // Use mdarray's zip expression for element-wise addition
        let tensor = self
            .tensor
            .expr()
            .zip(other.tensor.expr())
            .map(|(a, b)| *a + *b)
            .eval();

        Self { tensor }
    }
}

impl<T: Scalar> BlockDataLike<T> for BlockData<T> {
    fn rank(&self) -> usize {
        self.tensor.rank()
    }

    fn shape(&self) -> Vec<usize> {
        self.tensor.dims().to_vec()
    }

    fn len(&self) -> usize {
        self.tensor.len()
    }

    fn permute(&self, perm: &[usize]) -> Self {
        let view = self.tensor.permute(perm);
        let tensor = view.cloned().eval();
        Self { tensor }
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        let view = self.tensor.reshape(shape);
        let tensor = view.cloned().eval();
        Self { tensor }
    }
}
