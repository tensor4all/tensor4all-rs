//! Block data storage wrapping mdarray Tensor.
//!
//! `BlockedData` wraps mdarray's `Tensor<T, DynRank>` for N-dimensional block storage.
//! All operations are delegated to mdarray.

use std::sync::Arc;

use mdarray::{DynRank, Tensor};

use crate::scalar::Scalar;

/// Block data storage wrapping mdarray Tensor with dynamic rank.
///
/// This is a thin wrapper around mdarray's Tensor. All arithmetic and
/// reshape operations are delegated to mdarray.
#[derive(Debug, Clone)]
pub struct BlockedData<T: Scalar>(Arc<Tensor<T, DynRank>>);

impl<T: Scalar> BlockedData<T> {
    /// Create from mdarray Tensor.
    pub fn new(tensor: Tensor<T, DynRank>) -> Self {
        Self(Arc::new(tensor))
    }

    /// Get reference to underlying mdarray Tensor.
    pub fn as_tensor(&self) -> &Tensor<T, DynRank> {
        &self.0
    }

    /// Get Arc reference for sharing.
    pub fn arc(&self) -> Arc<Tensor<T, DynRank>> {
        Arc::clone(&self.0)
    }
}

/// Helper to convert multi-dimensional index to linear index (row-major).
pub fn multi_to_linear(idx: &[usize], shape: &[usize]) -> usize {
    let mut linear = 0;
    let mut stride = 1;
    for i in (0..idx.len()).rev() {
        linear += idx[i] * stride;
        stride *= shape[i];
    }
    linear
}

/// Helper to convert linear index to multi-dimensional index (row-major).
pub fn linear_to_multi(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        idx[i] = linear % shape[i];
        linear /= shape[i];
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_to_linear() {
        // 2x3 array: [[0,1,2], [3,4,5]]
        let shape = vec![2, 3];
        assert_eq!(multi_to_linear(&[0, 0], &shape), 0);
        assert_eq!(multi_to_linear(&[0, 2], &shape), 2);
        assert_eq!(multi_to_linear(&[1, 0], &shape), 3);
        assert_eq!(multi_to_linear(&[1, 2], &shape), 5);
    }

    #[test]
    fn test_linear_to_multi() {
        let shape = vec![2, 3];
        assert_eq!(linear_to_multi(0, &shape), vec![0, 0]);
        assert_eq!(linear_to_multi(2, &shape), vec![0, 2]);
        assert_eq!(linear_to_multi(3, &shape), vec![1, 0]);
        assert_eq!(linear_to_multi(5, &shape), vec![1, 2]);
    }

    #[test]
    fn test_3d_indexing() {
        // 2x3x4 array
        let shape = vec![2, 3, 4];

        // [1, 2, 3] -> 1*12 + 2*4 + 3 = 23
        assert_eq!(multi_to_linear(&[1, 2, 3], &shape), 23);
        assert_eq!(linear_to_multi(23, &shape), vec![1, 2, 3]);
    }
}
