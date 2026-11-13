//! Scalar capability for MatrixLUCI implementations.

use num_complex::{Complex32, Complex64};

use crate::error::Result;
use crate::matrix_luci::MatrixLuciFactors;
use crate::matrixlu::RrLUOptions;
use tensor4all_tensorbackend::{
    explicit::Session, BackendLinalgError, BackendLinalgScalar, Matrix, MatrixSolveScalar,
    MatrixTriangularSolveScalar,
};

/// Scalar types supported by MatrixLUCI factorization.
///
/// Common arithmetic comes from [`crate::Scalar`]; this trait adds only the
/// backend solve capabilities and MatrixLUCI dispatch used by the factorizer.
pub trait MatrixLuciScalar:
    crate::Scalar + BackendLinalgScalar + MatrixSolveScalar + MatrixTriangularSolveScalar
{
    #[doc(hidden)]
    fn matrix_luci_factors_from_matrix(
        a: &Matrix<Self>,
        options: RrLUOptions,
    ) -> Result<MatrixLuciFactors<Self>>
    where
        Self: Sized;

    #[doc(hidden)]
    fn matrix_luci_factors_from_blocks<F>(
        nrows: usize,
        ncols: usize,
        fill_block: F,
        options: RrLUOptions,
    ) -> Result<MatrixLuciFactors<Self>>
    where
        F: Fn(&[usize], &[usize], &mut [Self]),
        Self: Sized;

    /// [`Self::matrix_luci_factors_from_blocks`] with the previous pivots of
    /// the same candidate matrix as the rook search's starting points.
    ///
    /// The default ignores `seeds` and forwards to
    /// [`Self::matrix_luci_factors_from_blocks`], so adding this method does
    /// not require existing implementations to change. Seeds only move where
    /// rook walks start and carry no error guarantee, so ignoring them is a
    /// valid, unseeded factorization. Core's implementations override it.
    #[doc(hidden)]
    fn matrix_luci_factors_from_blocks_seeded<F>(
        nrows: usize,
        ncols: usize,
        fill_block: F,
        seeds: &[(usize, usize)],
        options: RrLUOptions,
    ) -> Result<MatrixLuciFactors<Self>>
    where
        F: Fn(&[usize], &[usize], &mut [Self]),
        Self: Sized,
    {
        let _ = seeds;
        Self::matrix_luci_factors_from_blocks(nrows, ncols, fill_block, options)
    }

    /// Triangular solve on an explicit session (non-unit diagonal, no transpose).
    #[doc(hidden)]
    fn matrix_luci_triangular_solve_in(
        session: &mut Session<'_>,
        a: &Matrix<Self>,
        b: &Matrix<Self>,
        left_side: bool,
        lower: bool,
    ) -> std::result::Result<Matrix<Self>, BackendLinalgError>
    where
        Self: Sized;
}

macro_rules! impl_matrix_luci_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl MatrixLuciScalar for $ty {
                fn matrix_luci_factors_from_matrix(
                    a: &Matrix<Self>,
                    options: RrLUOptions,
                ) -> Result<MatrixLuciFactors<Self>> {
                    crate::matrix_luci::dense_matrix_luci_factors_from_matrix(a, options)
                }

                fn matrix_luci_factors_from_blocks<F>(
                    nrows: usize,
                    ncols: usize,
                    fill_block: F,
                    options: RrLUOptions,
                ) -> Result<MatrixLuciFactors<Self>>
                where
                    F: Fn(&[usize], &[usize], &mut [Self]),
                {
                    crate::matrix_luci::lazy_matrix_luci_factors_from_blocks(
                        nrows, ncols, fill_block, options,
                    )
                }

                fn matrix_luci_factors_from_blocks_seeded<F>(
                    nrows: usize,
                    ncols: usize,
                    fill_block: F,
                    seeds: &[(usize, usize)],
                    options: RrLUOptions,
                ) -> Result<MatrixLuciFactors<Self>>
                where
                    F: Fn(&[usize], &[usize], &mut [Self]),
                {
                    crate::matrix_luci::lazy_matrix_luci_factors_from_blocks_seeded(
                        nrows, ncols, fill_block, seeds, options,
                    )
                }

                fn matrix_luci_triangular_solve_in(
                    session: &mut Session<'_>,
                    a: &Matrix<Self>,
                    b: &Matrix<Self>,
                    left_side: bool,
                    lower: bool,
                ) -> std::result::Result<Matrix<Self>, BackendLinalgError> {
                    session.triangular_solve_matrix(a, b, left_side, lower, false, false)
                }
            }
        )*
    };
}

impl_matrix_luci_scalar!(f64, f32, Complex64, Complex32);

#[cfg(test)]
mod tests;
