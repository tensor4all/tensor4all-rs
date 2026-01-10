//! Backend dispatch helpers for linear algebra operations.
//!
//! This module provides centralized backend selection for SVD and QR operations,
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

/// Compute SVD decomposition using the selected backend.
///
/// # Arguments
/// * `a` - Input matrix slice (mutable, may be modified by backend)
///
/// # Returns
/// SVD decomposition result wrapped in [`SvdResult`]
pub fn svd_backend<T>(a: &mut DSlice<T, 2>) -> Result<SvdResult<T>>
where
    T: num_complex::ComplexFloat
        + faer_traits::ComplexField
        + Default
        + From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    let decomp = {
        #[cfg(feature = "backend-faer")]
        {
            let bd = Faer;
            bd.svd(a)
        }
        #[cfg(feature = "backend-lapack")]
        {
            let bd = Lapack::new();
            bd.svd(a)
        }
        #[cfg(not(any(feature = "backend-faer", feature = "backend-lapack")))]
        {
            compile_error!(
                "At least one backend feature must be enabled (backend-faer or backend-lapack)"
            );
        }
    }
    .map_err(|e| anyhow::anyhow!("SVD computation failed: {}", e))?;

    // Convert from mdarray-linalg's Tensor<T, (usize, usize)> to DTensor<T, 2>
    // by copying the data. This isolates downstream code from upstream type changes.
    let u = tensor_to_dtensor(&decomp.u);
    let s = tensor_to_dtensor(&decomp.s);
    let vt = tensor_to_dtensor(&decomp.vt);

    Ok(SvdResult { u, s, vt })
}

/// Convert mdarray Tensor<T, (usize, usize)> to DTensor<T, 2>
fn tensor_to_dtensor<T: Clone>(tensor: &mdarray::Tensor<T, (usize, usize)>) -> DTensor<T, 2> {
    // Tensor<T, (usize, usize)> has shape() method that returns &(usize, usize)
    let rows = tensor.dim(0);
    let cols = tensor.dim(1);
    DTensor::<T, 2>::from_fn([rows, cols], |idx| tensor[[idx[0], idx[1]]].clone())
}

/// Compute QR decomposition using the selected backend.
///
/// # Arguments
/// * `a` - Input matrix slice (mutable, may be modified by backend)
///
/// # Returns
/// Tuple `(Q, R)` where Q is m×m and R is m×n (full QR)
pub fn qr_backend<T>(a: &mut DSlice<T, 2>) -> (DTensor<T, 2>, DTensor<T, 2>)
where
    T: num_complex::ComplexFloat
        + Default
        + faer_traits::ComplexField
        + From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    #[cfg(feature = "backend-faer")]
    {
        let bd = Faer;
        bd.qr(a)
    }
    #[cfg(feature = "backend-lapack")]
    {
        let bd = Lapack::new();
        bd.qr(a)
    }
    #[cfg(not(any(feature = "backend-faer", feature = "backend-lapack")))]
    {
        compile_error!(
            "At least one backend feature must be enabled (backend-faer or backend-lapack)"
        );
    }
}
