//! Backend dispatch helpers for linear algebra operations.
//!
//! This module exposes typed wrappers around `tenferro-linalg` while keeping
//! tensor4all's canonical compute object in the dynamic `tenferro::Tensor`
//! frontend.

use anyhow::{anyhow, Result};
use num_complex::ComplexFloat;
use tenferro_algebra::Scalar as TfScalar;
use tenferro_linalg::{qr as tenferro_qr, svd as tenferro_svd, KernelLinalgScalar, LinalgScalar};
use tenferro_tensor::{KeepCountScalar, Tensor as TypedTensor};

use crate::tenferro_bridge::with_tenferro_ctx;

/// Result of SVD decomposition `A = U * diag(S) * Vt`.
///
/// Contains the three factors computed by [`svd_backend`]. The singular values
/// `s` are stored as a 1-D tensor of real values, even when the input matrix
/// is complex.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{svd_backend, SvdResult};
/// use tenferro_tensor::{MemoryOrder, Tensor as TypedTensor};
///
/// // 2x3 real matrix (column-major)
/// let a = TypedTensor::<f64>::from_slice(
///     &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
///     &[2, 3],
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let result = svd_backend(&a).unwrap();
/// assert_eq!(result.u.dims(), &[2, 2]);
/// assert_eq!(result.s.dims(), &[2]);
/// assert_eq!(result.vt.dims(), &[2, 3]);
/// ```
#[derive(Debug, Clone)]
pub struct SvdResult<T>
where
    T: LinalgScalar,
{
    /// Left singular vectors (m x k matrix, where k = min(m, n)).
    pub u: TypedTensor<T>,
    /// Singular values (1-D tensor of length k).
    pub s: TypedTensor<<T as LinalgScalar>::Real>,
    /// Right singular vectors transposed (k x n matrix).
    pub vt: TypedTensor<T>,
}

/// Scalar constraint for tensor4all linalg backend dispatch.
///
/// This is a convenience bound combining [`LinalgScalar`] (SVD/QR trait
/// requirements) with [`KernelLinalgScalar`] (CPU kernel dispatch).
/// Implemented automatically for any type satisfying both bounds, including
/// `f64` and `Complex64`.
pub trait BackendLinalgScalar: LinalgScalar + KernelLinalgScalar {}

impl<T> BackendLinalgScalar for T where T: LinalgScalar + KernelLinalgScalar {}

/// Compute SVD decomposition via tenferro backend.
///
/// Decomposes an m x n matrix `A` into `U * diag(S) * Vt`.
///
/// # Errors
///
/// Returns an error if the SVD computation fails (e.g., non-convergence).
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::svd_backend;
/// use tenferro_tensor::{MemoryOrder, Tensor as TypedTensor};
///
/// let a = TypedTensor::<f64>::from_slice(
///     &[1.0, 0.0, 0.0, 1.0],
///     &[2, 2],
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let result = svd_backend(&a).unwrap();
/// assert_eq!(result.u.dims(), &[2, 2]);
/// assert_eq!(result.s.dims(), &[2]);  // 2 singular values
/// assert_eq!(result.vt.dims(), &[2, 2]);
/// ```
pub fn svd_backend<T>(a: &TypedTensor<T>) -> Result<SvdResult<T>>
where
    T: ComplexFloat + BackendLinalgScalar + TfScalar + Copy + 'static,
    <T as LinalgScalar>::Real: TfScalar + Copy + KeepCountScalar,
{
    let decomp = with_tenferro_ctx("svd", |ctx| {
        tenferro_svd(ctx, a, None)
            .map_err(|e| anyhow!("SVD computation failed via tenferro-linalg: {e}"))
    })?;

    Ok(SvdResult {
        u: decomp.u,
        s: decomp.s,
        vt: decomp.vt,
    })
}

/// Compute QR decomposition via tenferro backend.
///
/// Decomposes an m x n matrix `A` into an orthogonal/unitary matrix `Q` and
/// an upper triangular matrix `R` such that `A = Q * R`.
///
/// # Errors
///
/// Returns an error if the QR computation fails.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::qr_backend;
/// use tenferro_tensor::{MemoryOrder, Tensor as TypedTensor};
///
/// let a = TypedTensor::<f64>::from_slice(
///     &[1.0, 0.0, 0.0, 1.0],
///     &[2, 2],
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let (q, r) = qr_backend(&a).unwrap();
/// assert_eq!(q.dims(), &[2, 2]);
/// assert_eq!(r.dims(), &[2, 2]);
/// ```
pub fn qr_backend<T>(a: &TypedTensor<T>) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: ComplexFloat + BackendLinalgScalar + TfScalar + Copy + 'static,
{
    with_tenferro_ctx("qr", |ctx| {
        let decomp = tenferro_qr(ctx, a)
            .map_err(|e| anyhow!("QR computation failed via tenferro-linalg: {e}"))?;
        Ok((decomp.q, decomp.r))
    })
}

#[cfg(test)]
mod tests;
