//! Backend dispatch helpers for linear algebra operations.
//!
//! This module exposes tensor4all-local wrappers while delegating execution to
//! `tenferro-linalg`.

use anyhow::{anyhow, Result};
use mdarray::{DSlice, DTensor};
use num_complex::ComplexFloat;
use num_traits::NumCast;
use tenferro_algebra::Scalar as TfScalar;
use tenferro_linalg::{qr as tenferro_qr, svd as tenferro_svd, LinalgScalar};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::tenferro_bridge::with_tenferro_ctx;

/// Result of SVD decomposition.
///
/// Contains U, S, and Vt matrices as DTensor.
/// Singular values are stored in the first row of `s` (`s[[0, i]]`),
/// matching existing tensor4all expectations.
#[derive(Debug, Clone)]
pub struct SvdResult<T> {
    /// Left singular vectors (m×k matrix for thin SVD)
    pub u: DTensor<T, 2>,
    /// Singular values in first row (`1×k`)
    pub s: DTensor<T, 2>,
    /// Right singular vectors (k×n matrix, conjugate-transposed)
    pub vt: DTensor<T, 2>,
}

/// Scalar constraint for tensor4all linalg backend dispatch.
///
/// This currently maps to tenferro's CPU linalg scalar support and provides
/// an abstraction boundary for future GPU backend expansion.
pub trait BackendLinalgScalar: LinalgScalar + tenferro_linalg::backend::CpuLinalgScalar {}

impl<T> BackendLinalgScalar for T where T: LinalgScalar + tenferro_linalg::backend::CpuLinalgScalar {}

fn dslice_to_tenferro_tensor<T>(a: &DSlice<T, 2>) -> Result<Tensor<T>>
where
    T: TfScalar + Copy,
{
    let m = a.dim(0);
    let n = a.dim(1);
    let mut col_major = vec![T::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            col_major[i + m * j] = a[[i, j]];
        }
    }
    Tensor::from_slice(&col_major, &[m, n], MemoryOrder::ColumnMajor).map_err(|e| {
        anyhow!(
            "failed to convert mdarray matrix to tenferro tensor (shape=[{}, {}]): {}",
            m,
            n,
            e
        )
    })
}

fn tenferro_tensor_to_dtensor<T>(tensor: &Tensor<T>) -> Result<DTensor<T, 2>>
where
    T: TfScalar + Copy,
{
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(anyhow!(
            "expected 2D tensor, got ndim={} (dims={:?})",
            dims.len(),
            dims
        ));
    }
    let rows = dims[0];
    let cols = dims[1];
    let row_major = tensor.contiguous(MemoryOrder::RowMajor);
    let data = row_major
        .buffer()
        .as_slice()
        .ok_or_else(|| anyhow!("expected host-accessible contiguous tensor data"))?;

    Ok(DTensor::<T, 2>::from_fn([rows, cols], |idx| {
        data[idx[0] * cols + idx[1]]
    }))
}

fn singular_values_to_dtensor<T>(
    tensor: &Tensor<<T as LinalgScalar>::Real>,
) -> Result<DTensor<T, 2>>
where
    T: ComplexFloat + LinalgScalar + TfScalar + NumCast + Copy,
    <T as LinalgScalar>::Real: TfScalar + Copy,
{
    let dims = tensor.dims();
    if dims.len() != 1 {
        return Err(anyhow!(
            "expected 1D singular-value tensor, got ndim={} (dims={:?})",
            dims.len(),
            dims
        ));
    }
    let k = dims[0];
    let row_major = tensor.contiguous(MemoryOrder::RowMajor);
    let data = row_major
        .buffer()
        .as_slice()
        .ok_or_else(|| anyhow!("expected host-accessible singular-value data"))?;

    Ok(DTensor::<T, 2>::from_fn([1, k], |idx| {
        <T as NumCast>::from(data[idx[1]])
            .unwrap_or_else(|| panic!("failed to cast singular value to target scalar type"))
    }))
}

/// Compute SVD decomposition via tenferro backend.
pub fn svd_backend<T>(a: &mut DSlice<T, 2>) -> Result<SvdResult<T>>
where
    T: ComplexFloat + BackendLinalgScalar + TfScalar + NumCast + Copy + 'static,
    <T as LinalgScalar>::Real: TfScalar + Copy,
{
    let tf_tensor = dslice_to_tenferro_tensor(a)?;
    let decomp = with_tenferro_ctx("svd", |ctx| {
        tenferro_svd(ctx, &tf_tensor, None)
            .map_err(|e| anyhow!("SVD computation failed via tenferro-linalg: {}", e))
    })?;

    let u = tenferro_tensor_to_dtensor(&decomp.u)?;
    let s = singular_values_to_dtensor::<T>(&decomp.s)?;
    let vt = tenferro_tensor_to_dtensor(&decomp.vt)?;

    Ok(SvdResult { u, s, vt })
}

/// Compute QR decomposition via tenferro backend.
///
/// Kept infallible for compatibility with existing call sites.
pub fn qr_backend<T>(a: &mut DSlice<T, 2>) -> (DTensor<T, 2>, DTensor<T, 2>)
where
    T: ComplexFloat + BackendLinalgScalar + TfScalar + Copy + 'static,
{
    let tf_tensor =
        dslice_to_tenferro_tensor(a).unwrap_or_else(|e| panic!("QR input conversion failed: {e}"));
    with_tenferro_ctx("qr", |ctx| {
        let decomp = tenferro_qr(ctx, &tf_tensor)
            .map_err(|e| anyhow!("QR computation failed via tenferro-linalg: {}", e))?;
        let q = tenferro_tensor_to_dtensor(&decomp.q)
            .map_err(|e| anyhow!("QR output conversion (Q) failed: {}", e))?;
        let r = tenferro_tensor_to_dtensor(&decomp.r)
            .map_err(|e| anyhow!("QR output conversion (R) failed: {}", e))?;
        Ok((q, r))
    })
    .unwrap_or_else(|e| panic!("QR backend failed: {e}"))
}
