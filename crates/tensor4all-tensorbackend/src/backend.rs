//! Backend dispatch helpers for linear algebra operations.
//!
//! This module exposes typed wrappers around `tenferro-linalg` while keeping
//! tensor4all's canonical compute object in the dynamic `tenferro::Tensor`
//! frontend.

use anyhow::{anyhow, Result};
use num_complex::ComplexFloat;
use tenferro_algebra::Scalar as TfScalar;
use tenferro_linalg::{qr as tenferro_qr, svd as tenferro_svd, KernelLinalgScalar, LinalgScalar};
use tenferro_tensor::Tensor as TypedTensor;

use crate::tenferro_bridge::with_tenferro_ctx;

/// Result of SVD decomposition.
#[derive(Debug, Clone)]
pub struct SvdResult<T>
where
    T: LinalgScalar,
{
    /// Left singular vectors.
    pub u: TypedTensor<T>,
    /// Singular values.
    pub s: TypedTensor<<T as LinalgScalar>::Real>,
    /// Right singular vectors transposed.
    pub vt: TypedTensor<T>,
}

/// Scalar constraint for tensor4all linalg backend dispatch.
pub trait BackendLinalgScalar: LinalgScalar + KernelLinalgScalar {}

impl<T> BackendLinalgScalar for T where T: LinalgScalar + KernelLinalgScalar {}

/// Compute SVD decomposition via tenferro backend.
pub fn svd_backend<T>(a: &TypedTensor<T>) -> Result<SvdResult<T>>
where
    T: ComplexFloat + BackendLinalgScalar + TfScalar + Copy + 'static,
    <T as LinalgScalar>::Real: TfScalar + Copy,
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
mod tests {
    use super::*;
    use num_complex::Complex64;
    use tenferro_tensor::MemoryOrder;

    #[test]
    fn qr_backend_reconstructs_real_matrix() {
        let input =
            TypedTensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::RowMajor)
                .unwrap();

        let (q, r) = qr_backend(&input).unwrap();
        assert_eq!(q.dims(), &[2, 2]);
        assert_eq!(r.dims(), &[2, 2]);
        assert_eq!(q.ndim(), input.ndim());
        assert_eq!(r.ndim(), input.ndim());
    }

    #[test]
    fn svd_backend_reconstructs_complex_matrix() {
        let input = TypedTensor::from_slice(
            &[
                Complex64::new(1.0, -0.5),
                Complex64::new(2.0, 1.5),
                Complex64::new(-3.0, 0.25),
                Complex64::new(4.0, -2.0),
            ],
            &[2, 2],
            MemoryOrder::RowMajor,
        )
        .unwrap();

        let decomp = svd_backend(&input).unwrap();
        assert_eq!(decomp.u.dims(), &[2, 2]);
        assert_eq!(decomp.s.dims(), &[2]);
        assert_eq!(decomp.vt.dims(), &[2, 2]);
    }
}
