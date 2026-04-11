//! Candidate matrix sources for pivot-kernel factorization.
//!
//! Provides the [`CandidateMatrixSource`] trait and two built-in
//! implementations: [`DenseMatrixSource`] (borrowed column-major data) and
//! [`LazyMatrixSource`] (callback-backed block evaluation).

use crate::matrixluci::scalar::Scalar;
use crate::matrixluci::types::DenseOwnedMatrix;
use std::marker::PhantomData;

/// Abstraction for accessing a matrix that will be cross-interpolated.
///
/// Implementors provide block-level access (filling column-major sub-blocks)
/// so that kernels can select pivots without materializing the full matrix.
pub trait CandidateMatrixSource<T: Scalar> {
    /// Number of rows.
    fn nrows(&self) -> usize;

    /// Number of columns.
    fn ncols(&self) -> usize;

    /// Fill `out` with the cross-product A[rows, cols] in column-major order.
    fn get_block(&self, rows: &[usize], cols: &[usize], out: &mut [T]);

    /// Borrow the whole matrix in column-major layout when available.
    fn dense_column_major_slice(&self) -> Option<&[T]> {
        None
    }

    /// Read a single matrix entry.
    fn get(&self, row: usize, col: usize) -> T {
        let mut out = [T::zero(); 1];
        self.get_block(&[row], &[col], &mut out);
        out[0]
    }
}

/// Borrowed dense matrix source with column-major layout.
///
/// Wraps a column-major data slice for use with pivot kernels.
pub struct DenseMatrixSource<'a, T: Scalar> {
    data: &'a [T],
    nrows: usize,
    ncols: usize,
}

/// Callback-backed lazy matrix source.
///
/// Evaluates matrix blocks on demand via a user-supplied closure,
/// avoiding full materialization. The closure fills a column-major
/// output buffer for a given set of rows and columns.
pub struct LazyMatrixSource<T: Scalar, F> {
    nrows: usize,
    ncols: usize,
    fill_block: F,
    _marker: PhantomData<T>,
}

impl<'a, T: Scalar> DenseMatrixSource<'a, T> {
    /// Create a dense source from a column-major slice.
    pub fn from_column_major(data: &'a [T], nrows: usize, ncols: usize) -> Self {
        assert_eq!(data.len(), nrows * ncols);
        Self { data, nrows, ncols }
    }
}

impl<T: Scalar, F> LazyMatrixSource<T, F>
where
    F: Fn(&[usize], &[usize], &mut [T]),
{
    /// Create a lazy source from a block-fill callback.
    pub fn new(nrows: usize, ncols: usize, fill_block: F) -> Self {
        Self {
            nrows,
            ncols,
            fill_block,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar> CandidateMatrixSource<T> for DenseMatrixSource<'_, T> {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn get_block(&self, rows: &[usize], cols: &[usize], out: &mut [T]) {
        assert_eq!(out.len(), rows.len() * cols.len());
        for (j, &col) in cols.iter().enumerate() {
            for (i, &row) in rows.iter().enumerate() {
                out[i + rows.len() * j] = self.data[row + self.nrows * col];
            }
        }
    }

    fn dense_column_major_slice(&self) -> Option<&[T]> {
        Some(self.data)
    }
}

impl<T: Scalar, F> CandidateMatrixSource<T> for LazyMatrixSource<T, F>
where
    F: Fn(&[usize], &[usize], &mut [T]),
{
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn get_block(&self, rows: &[usize], cols: &[usize], out: &mut [T]) {
        assert_eq!(out.len(), rows.len() * cols.len());
        (self.fill_block)(rows, cols, out);
    }
}

pub(crate) fn materialize_source<T: Scalar, S: CandidateMatrixSource<T>>(
    source: &S,
) -> DenseOwnedMatrix<T> {
    let nrows = source.nrows();
    let ncols = source.ncols();
    let rows: Vec<usize> = (0..nrows).collect();
    let cols: Vec<usize> = (0..ncols).collect();
    let mut out = vec![T::zero(); nrows * ncols];
    source.get_block(&rows, &cols, &mut out);
    DenseOwnedMatrix::from_column_major(out, nrows, ncols)
}

#[cfg(test)]
mod tests;
