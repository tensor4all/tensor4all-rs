//! Backend dispatch helpers for linear algebra operations.
//!
//! This module keeps tensor4all's typed factorization entry points thin while
//! routing the actual work through the thread-local tenferro CPU backend.

use anyhow::{anyhow, Result};
use tenferro::{DType, Tensor, TensorBackend, TensorScalar, TypedTensor};

use crate::context::with_default_backend;

/// Result of SVD decomposition `A = U * diag(S) * Vt`.
///
/// The singular values are stored in a real-valued typed tensor, even when the
/// input matrix is complex.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::svd_backend;
/// use tenferro::TypedTensor;
///
/// let a = TypedTensor::<f64>::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]);
/// let result = svd_backend(&a).unwrap();
///
/// assert_eq!(result.u.shape, vec![2, 2]);
/// assert_eq!(result.s.shape, vec![2]);
/// assert_eq!(result.vt.shape, vec![2, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct SvdResult<T: TensorScalar> {
    /// Left singular vectors.
    pub u: TypedTensor<T>,
    /// Singular values.
    pub s: TypedTensor<T::Real>,
    /// Right singular vectors transposed.
    pub vt: TypedTensor<T>,
}

/// Scalar bound accepted by tensor4all's typed linalg wrappers.
pub trait BackendLinalgScalar: TensorScalar {}

impl<T: TensorScalar> BackendLinalgScalar for T {}

fn tensor_scalar_dtype<T: TensorScalar>() -> DType {
    T::into_tensor(vec![0], Vec::new()).dtype()
}

fn try_into_typed_result<T: TensorScalar>(
    op: &'static str,
    tensor: Tensor,
) -> Result<TypedTensor<T>> {
    let actual = tensor.dtype();
    T::try_into_typed(tensor).ok_or_else(|| {
        anyhow!(
            "{op}: dtype mismatch lhs={actual:?} rhs={:?}",
            tensor_scalar_dtype::<T>()
        )
    })
}

fn convert_for_typed<T: TensorScalar>(op: &'static str, tensor: Tensor) -> Result<TypedTensor<T>> {
    let expected = tensor_scalar_dtype::<T>();
    let tensor = if tensor.dtype() == expected {
        tensor
    } else {
        with_default_backend(|backend| {
            backend.with_exec_session(|exec| exec.convert(&tensor, expected))
        })
        .map_err(|e| anyhow!("{op}: dtype conversion to {expected:?} failed: {e}"))?
    };
    try_into_typed_result::<T>(op, tensor)
}

/// Compute a thin/economy SVD on a typed tensor.
///
/// # Errors
///
/// Returns an error if the backend rejects the input or the decomposition
/// fails to converge.
pub fn svd_backend<T>(a: &TypedTensor<T>) -> Result<SvdResult<T>>
where
    T: BackendLinalgScalar,
{
    let tensor = T::into_tensor(a.shape.clone(), a.host_data().to_vec());
    let (u, s, vt) = with_default_backend(|backend| tensor.svd(backend))
        .map_err(|e| anyhow!("SVD computation failed via tenferro-tensor: {e}"))?;
    Ok(SvdResult {
        u: convert_for_typed::<T>("svd", u)?,
        s: convert_for_typed::<T::Real>("svd", s)?,
        vt: convert_for_typed::<T>("svd", vt)?,
    })
}

/// Compute a thin/economy QR decomposition on a typed tensor.
///
/// # Errors
///
/// Returns an error if the backend rejects the input or the decomposition
/// fails.
pub fn qr_backend<T>(a: &TypedTensor<T>) -> Result<(TypedTensor<T>, TypedTensor<T>)>
where
    T: BackendLinalgScalar,
{
    with_default_backend(|backend| a.qr(backend))
        .map_err(|e| anyhow!("QR computation failed via tenferro-tensor: {e}"))
}

#[cfg(test)]
mod tests;
