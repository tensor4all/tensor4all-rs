//! Candidate matrix sources.

use crate::scalar::Scalar;
use crate::types::DenseOwnedMatrix;

/// Candidate matrix access abstraction.
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
pub struct DenseMatrixSource<'a, T: Scalar> {
    data: &'a [T],
    nrows: usize,
    ncols: usize,
}

impl<'a, T: Scalar> DenseMatrixSource<'a, T> {
    /// Create a dense source from a column-major slice.
    pub fn from_column_major(data: &'a [T], nrows: usize, ncols: usize) -> Self {
        assert_eq!(data.len(), nrows * ncols);
        Self { data, nrows, ncols }
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
