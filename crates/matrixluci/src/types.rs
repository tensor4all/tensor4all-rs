//! Core data types for matrixluci.

use crate::scalar::Scalar;
use std::ops::{Index, IndexMut};

/// Simple owned dense matrix in column-major layout.
#[derive(Debug, Clone)]
pub struct DenseOwnedMatrix<T: Scalar> {
    pub(crate) data: Vec<T>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

/// Pivot-kernel options.
#[derive(Debug, Clone)]
pub struct PivotKernelOptions {
    /// Relative tolerance.
    pub rel_tol: f64,
    /// Absolute tolerance.
    pub abs_tol: f64,
    /// Maximum rank.
    pub max_rank: usize,
    /// Whether the left factor is unit-diagonal.
    pub left_orthogonal: bool,
}

/// Pivot-only output of a kernel invocation.
#[derive(Debug, Clone)]
pub struct PivotSelectionCore {
    /// Selected row indices.
    pub row_indices: Vec<usize>,
    /// Selected column indices.
    pub col_indices: Vec<usize>,
    /// Pivot error history.
    pub pivot_errors: Vec<f64>,
    /// Selected rank.
    pub rank: usize,
}

impl<T: Scalar> DenseOwnedMatrix<T> {
    /// Create a matrix from a column-major buffer.
    pub fn from_column_major(data: Vec<T>, nrows: usize, ncols: usize) -> Self {
        assert_eq!(data.len(), nrows * ncols);
        Self { data, nrows, ncols }
    }

    /// Create a zero-filled matrix.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![T::zero(); nrows * ncols],
            nrows,
            ncols,
        }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Borrow the underlying column-major data.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrow the underlying column-major data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Scalar> Index<[usize; 2]> for DenseOwnedMatrix<T> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[index[0] + self.nrows * index[1]]
    }
}

impl<T: Scalar> IndexMut<[usize; 2]> for DenseOwnedMatrix<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.data[index[0] + self.nrows * index[1]]
    }
}

impl PivotKernelOptions {
    /// Canonical options for dense no-truncation behavior.
    pub fn no_truncation() -> Self {
        Self {
            rel_tol: 0.0,
            abs_tol: 0.0,
            max_rank: usize::MAX,
            left_orthogonal: true,
        }
    }
}

impl Default for PivotKernelOptions {
    fn default() -> Self {
        Self {
            rel_tol: 1e-14,
            abs_tol: 0.0,
            max_rank: usize::MAX,
            left_orthogonal: true,
        }
    }
}
