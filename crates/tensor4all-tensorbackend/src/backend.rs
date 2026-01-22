//! Backend dispatch helpers for linear algebra operations.
//!
//! This module provides centralized backend selection for SVD, QR, and MatMul operations,
//! reducing code duplication between decomposition implementations.
//!
//! All mdarray-linalg types are wrapped to avoid exposing upstream API changes
//! to downstream crates.

use anyhow::Result;
use mdarray::{DSlice, DTensor};
use mdarray_linalg::qr::QR;
use mdarray_linalg::svd::SVD;

#[cfg(feature = "backend-faer")]
use mdarray_linalg_faer::Faer;

#[cfg(feature = "backend-lapack")]
use mdarray_linalg_lapack::Lapack;

// Re-export MatMul trait for einsum backend
pub use mdarray_linalg::matmul::MatMul;

/// Result of SVD decomposition.
///
/// Contains U, S, and Vt matrices as DTensor.
/// This struct wraps mdarray-linalg's SVDDecomp to avoid exposing upstream types.
///
/// For an m×n matrix A:
/// - `u`: m×m unitary matrix (left singular vectors)
/// - `s`: singular values stored in first row as s\[\[0, i\]\] (LAPACK convention)
/// - `vt`: n×n unitary matrix (right singular vectors, transposed/conjugate-transposed)
///
/// The decomposition satisfies: A = U × S × Vt
#[derive(Debug, Clone)]
pub struct SvdResult<T> {
    /// Left singular vectors (m×m matrix)
    pub u: DTensor<T, 2>,
    /// Singular values stored in first row: s\[\[0, i\]\] (LAPACK convention)
    pub s: DTensor<T, 2>,
    /// Right singular vectors (n×n matrix, transposed)
    pub vt: DTensor<T, 2>,
}

/// Convert mdarray Tensor<T, (usize, usize)> to DTensor<T, 2>
fn tensor_to_dtensor<T: Clone>(tensor: &mdarray::Tensor<T, (usize, usize)>) -> DTensor<T, 2> {
    // Tensor<T, (usize, usize)> has shape() method that returns &(usize, usize)
    let rows = tensor.dim(0);
    let cols = tensor.dim(1);
    DTensor::<T, 2>::from_fn([rows, cols], |idx| tensor[[idx[0], idx[1]]].clone())
}

// ============================================================================
// FAER Backend Implementation
// ============================================================================

#[cfg(feature = "backend-faer")]
mod faer_impl {
    use super::*;

    /// Compute SVD decomposition using FAER backend.
    pub fn svd_backend<T>(a: &mut DSlice<T, 2>) -> Result<SvdResult<T>>
    where
        T: num_complex::ComplexFloat
            + faer_traits::ComplexField
            + Default
            + From<<T as num_complex::ComplexFloat>::Real>
            + 'static,
        <T as num_complex::ComplexFloat>::Real: Into<f64>,
    {
        let bd = Faer;
        let decomp = bd
            .svd(a)
            .map_err(|e| anyhow::anyhow!("SVD computation failed: {}", e))?;

        let u = tensor_to_dtensor(&decomp.u);
        let s = tensor_to_dtensor(&decomp.s);
        let vt = tensor_to_dtensor(&decomp.vt);

        Ok(SvdResult { u, s, vt })
    }

    /// Compute QR decomposition using FAER backend.
    pub fn qr_backend<T>(a: &mut DSlice<T, 2>) -> (DTensor<T, 2>, DTensor<T, 2>)
    where
        T: num_complex::ComplexFloat
            + Default
            + faer_traits::ComplexField
            + From<<T as num_complex::ComplexFloat>::Real>
            + 'static,
        <T as num_complex::ComplexFloat>::Real: Into<f64>,
    {
        let bd = Faer;
        bd.qr(a)
    }
}

// ============================================================================
// LAPACK Backend Implementation
// ============================================================================

// LAPACK backend uses concrete type implementations because the internal
// LapackScalar trait is not publicly exported by mdarray-linalg-lapack.
// The SVD and QR traits are implemented for f32, f64, Complex<f32>, Complex<f64>.

#[cfg(feature = "backend-lapack")]
mod lapack_impl {
    use super::*;
    use num_complex::Complex;

    /// Helper macro to implement SVD for concrete types.
    macro_rules! impl_svd_for_type {
        ($t:ty) => {
            impl SvdBackendImpl for $t {
                fn svd_impl(a: &mut DSlice<$t, 2>) -> Result<SvdResult<$t>> {
                    let bd = Lapack::new();
                    let decomp = bd
                        .svd(a)
                        .map_err(|e| anyhow::anyhow!("SVD computation failed: {}", e))?;

                    let u = tensor_to_dtensor(&decomp.u);
                    let s = tensor_to_dtensor(&decomp.s);
                    let vt = tensor_to_dtensor(&decomp.vt);

                    Ok(SvdResult { u, s, vt })
                }
            }
        };
    }

    /// Helper macro to implement QR for concrete types.
    macro_rules! impl_qr_for_type {
        ($t:ty) => {
            impl QrBackendImpl for $t {
                fn qr_impl(a: &mut DSlice<$t, 2>) -> (DTensor<$t, 2>, DTensor<$t, 2>) {
                    let bd = Lapack::new();
                    bd.qr(a)
                }
            }
        };
    }

    /// Internal trait for SVD backend dispatch.
    pub trait SvdBackendImpl: Sized + Clone {
        fn svd_impl(a: &mut DSlice<Self, 2>) -> Result<SvdResult<Self>>;
    }

    /// Internal trait for QR backend dispatch.
    pub trait QrBackendImpl: Sized + Clone {
        fn qr_impl(a: &mut DSlice<Self, 2>) -> (DTensor<Self, 2>, DTensor<Self, 2>);
    }

    // Implement for concrete types supported by LAPACK
    impl_svd_for_type!(f32);
    impl_svd_for_type!(f64);
    impl_svd_for_type!(Complex<f32>);
    impl_svd_for_type!(Complex<f64>);

    impl_qr_for_type!(f32);
    impl_qr_for_type!(f64);
    impl_qr_for_type!(Complex<f32>);
    impl_qr_for_type!(Complex<f64>);

    /// Compute SVD decomposition using LAPACK backend.
    pub fn svd_backend<T: SvdBackendImpl>(a: &mut DSlice<T, 2>) -> Result<SvdResult<T>> {
        T::svd_impl(a)
    }

    /// Compute QR decomposition using LAPACK backend.
    pub fn qr_backend<T: QrBackendImpl>(a: &mut DSlice<T, 2>) -> (DTensor<T, 2>, DTensor<T, 2>) {
        T::qr_impl(a)
    }
}

// ============================================================================
// Public API - Re-export from selected backend
// ============================================================================

#[cfg(all(feature = "backend-faer", not(feature = "backend-lapack")))]
pub use faer_impl::{qr_backend, svd_backend};

#[cfg(all(feature = "backend-lapack", not(feature = "backend-faer")))]
pub use lapack_impl::{qr_backend, svd_backend};

// When both are enabled, prefer FAER (it's the default)
#[cfg(all(feature = "backend-faer", feature = "backend-lapack"))]
pub use faer_impl::{qr_backend, svd_backend};

#[cfg(not(any(feature = "backend-faer", feature = "backend-lapack")))]
compile_error!("At least one backend feature must be enabled (backend-faer or backend-lapack)");

// ============================================================================
// MatMul Backend for einsum operations
// ============================================================================

/// Returns the MatMul backend for einsum operations.
///
/// This function returns the appropriate backend based on enabled features:
/// - `backend-faer` (default): Returns `Faer` backend
/// - `backend-lapack`: Returns `Faer` backend (LAPACK doesn't provide MatMul)
///
/// Note: LAPACK provides SVD/QR but not general matrix multiplication traits.
/// For MatMul, we always use Faer when backend-faer is enabled.
#[cfg(feature = "backend-faer")]
pub fn matmul_backend() -> Faer {
    Faer
}

/// Fallback: use Naive backend when faer is not available.
/// This should rarely happen as backend-faer is the default.
#[cfg(not(feature = "backend-faer"))]
pub fn matmul_backend() -> mdarray_linalg::Naive {
    mdarray_linalg::Naive
}
