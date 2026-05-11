//! Backend dispatch helpers for linear algebra operations.
//!
//! This module keeps tensor4all's typed factorization entry points thin while
//! routing the actual work through the shared tenferro CPU backend.

use anyhow::{anyhow, Result};
use num_complex::{Complex32, Complex64};
use tenferro::{DType, Tensor, TensorBackend, TensorScalar, TypedTensor};

use crate::context::with_default_backend;
use crate::matrix::Matrix;

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

/// Result of complete-pivoting LU decomposition `P A Q^T = L U`.
///
/// The parity output from tenferro is intentionally omitted because current
/// tensor4all callers only need the permutation matrices and the upper
/// triangular factor for pivot selection.
#[derive(Debug, Clone)]
pub struct FullPivLuResult<T: TensorScalar> {
    /// Left permutation matrix.
    pub p: TypedTensor<T>,
    /// Lower triangular factor.
    pub l: TypedTensor<T>,
    /// Upper triangular factor.
    pub u: TypedTensor<T>,
    /// Right permutation matrix.
    pub q: TypedTensor<T>,
}

/// Result of complete-pivoting LU decomposition on [`Matrix`] values.
///
/// This is the matrix-shaped counterpart of [`FullPivLuResult`]. It exists so
/// downstream crates can use backend linalg without hand-writing
/// `TypedTensor` conversion code.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, full_piv_lu_matrix};
///
/// let matrix = from_vec2d(vec![vec![0.0_f64, 1.0], vec![2.0, 3.0]]);
/// let factors = full_piv_lu_matrix(&matrix).unwrap();
/// assert_eq!(factors.u.nrows(), 2);
/// assert_eq!(factors.u.ncols(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct FullPivLuMatrixResult<T> {
    /// Left permutation matrix.
    pub p: Matrix<T>,
    /// Lower triangular factor.
    pub l: Matrix<T>,
    /// Upper triangular factor.
    pub u: Matrix<T>,
    /// Right permutation matrix.
    pub q: Matrix<T>,
}

/// Scalar bound accepted by tensor4all's typed linalg wrappers.
pub trait BackendLinalgScalar: TensorScalar {}

impl<T: TensorScalar> BackendLinalgScalar for T {}

/// Scalar types supported by [`solve_matrix`].
///
/// `f64` and `Complex64` are solved directly. `f32` and `Complex32` are
/// promoted to the corresponding 64-bit dtype for the backend solve and then
/// converted back, because the current tenferro CPU LU solve is double
/// precision only.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, solve_matrix};
///
/// let a = from_vec2d(vec![vec![2.0_f32, 1.0], vec![1.0, 2.0]]);
/// let b = from_vec2d(vec![vec![1.0_f32], vec![0.0]]);
/// let x = solve_matrix(&a, &b).unwrap();
/// assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-6);
/// ```
pub trait MatrixSolveScalar: BackendLinalgScalar + crate::matrix::MatrixScalar {
    #[doc(hidden)]
    fn solve_matrix_impl(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>>;
}

fn solve_matrix_direct<T>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>>
where
    T: BackendLinalgScalar + Copy,
{
    let a_tensor = matrix_to_typed_tensor(a);
    let b_tensor = matrix_to_typed_tensor(b);
    let x = solve_backend(&a_tensor, &b_tensor)?;
    typed_tensor_to_matrix("solve", x)
}

impl MatrixSolveScalar for f64 {
    fn solve_matrix_impl(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>> {
        solve_matrix_direct(a, b)
    }
}

impl MatrixSolveScalar for Complex64 {
    fn solve_matrix_impl(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>> {
        solve_matrix_direct(a, b)
    }
}

impl MatrixSolveScalar for f32 {
    fn solve_matrix_impl(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>> {
        let a64 = Matrix::from_col_major_vec(
            a.nrows(),
            a.ncols(),
            a.as_col_major_slice()
                .iter()
                .map(|&value| value as f64)
                .collect(),
        );
        let b64 = Matrix::from_col_major_vec(
            b.nrows(),
            b.ncols(),
            b.as_col_major_slice()
                .iter()
                .map(|&value| value as f64)
                .collect(),
        );
        let x64 = solve_matrix_direct(&a64, &b64)?;
        Ok(Matrix::from_col_major_vec(
            x64.nrows(),
            x64.ncols(),
            x64.as_col_major_slice()
                .iter()
                .map(|&value| value as f32)
                .collect(),
        ))
    }
}

impl MatrixSolveScalar for Complex32 {
    fn solve_matrix_impl(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>> {
        let a64 = Matrix::from_col_major_vec(
            a.nrows(),
            a.ncols(),
            a.as_col_major_slice()
                .iter()
                .map(|&value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        );
        let b64 = Matrix::from_col_major_vec(
            b.nrows(),
            b.ncols(),
            b.as_col_major_slice()
                .iter()
                .map(|&value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        );
        let x64 = solve_matrix_direct(&a64, &b64)?;
        Ok(Matrix::from_col_major_vec(
            x64.nrows(),
            x64.ncols(),
            x64.as_col_major_slice()
                .iter()
                .map(|&value| Complex32::new(value.re as f32, value.im as f32))
                .collect(),
        ))
    }
}

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

fn matrix_to_typed_tensor<T>(matrix: &Matrix<T>) -> TypedTensor<T>
where
    T: TensorScalar + Copy,
{
    TypedTensor::from_vec(
        vec![matrix.nrows(), matrix.ncols()],
        matrix.as_col_major_slice().to_vec(),
    )
}

fn typed_tensor_to_matrix<T>(op: &'static str, tensor: TypedTensor<T>) -> Result<Matrix<T>>
where
    T: TensorScalar + Copy,
{
    if tensor.shape.len() != 2 {
        return Err(anyhow!(
            "{op}: expected a rank-2 tensor, got shape {:?}",
            tensor.shape
        ));
    }
    Ok(Matrix::from_col_major_vec(
        tensor.shape[0],
        tensor.shape[1],
        tensor.as_slice().to_vec(),
    ))
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

/// Solve `A X = B` with the configured tenferro backend.
///
/// # Errors
///
/// Returns an error if the backend rejects the input shapes, the scalar dtype,
/// or the coefficient matrix is singular.
pub fn solve_backend<T>(a: &TypedTensor<T>, b: &TypedTensor<T>) -> Result<TypedTensor<T>>
where
    T: BackendLinalgScalar,
{
    let a_tensor = T::into_tensor(a.shape.clone(), a.host_data().to_vec());
    let b_tensor = T::into_tensor(b.shape.clone(), b.host_data().to_vec());
    let result = with_default_backend(|backend| a_tensor.solve(&b_tensor, backend))
        .map_err(|e| anyhow!("linear solve failed via tenferro-tensor: {e}"))?;
    try_into_typed_result::<T>("solve", result)
}

/// Solve `A X = B` for column-major [`Matrix`] values.
///
/// This routes the operation through the configured tenferro backend and keeps
/// matrix-to-tensor conversion centralized in `tensor4all-tensorbackend`.
///
/// # Errors
///
/// Returns an error if the backend rejects the input shapes, scalar dtype, or
/// coefficient matrix.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, solve_matrix};
///
/// let a = from_vec2d(vec![vec![2.0_f64, 1.0], vec![1.0, 2.0]]);
/// let b = from_vec2d(vec![vec![1.0_f64], vec![0.0]]);
/// let x = solve_matrix(&a, &b).unwrap();
/// assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-12);
/// assert!((x[[1, 0]] + 1.0 / 3.0).abs() < 1.0e-12);
/// ```
pub fn solve_matrix<T>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>>
where
    T: MatrixSolveScalar,
{
    T::solve_matrix_impl(a, b)
}

/// Compute complete-pivoting LU with the configured tenferro backend.
///
/// # Errors
///
/// Returns an error if the backend does not support the input dtype or if the
/// factorization fails.
pub fn full_piv_lu_backend<T>(a: &TypedTensor<T>) -> Result<FullPivLuResult<T>>
where
    T: BackendLinalgScalar,
{
    let tensor = T::into_tensor(a.shape.clone(), a.host_data().to_vec());
    let (p, l, u, q, _parity) = with_default_backend(|backend| tensor.full_piv_lu(backend))
        .map_err(|e| anyhow!("complete-pivoting LU failed via tenferro-tensor: {e}"))?;
    Ok(FullPivLuResult {
        p: convert_for_typed::<T>("full_piv_lu", p)?,
        l: convert_for_typed::<T>("full_piv_lu", l)?,
        u: convert_for_typed::<T>("full_piv_lu", u)?,
        q: convert_for_typed::<T>("full_piv_lu", q)?,
    })
}

/// Compute complete-pivoting LU for a column-major [`Matrix`].
///
/// This is a convenience wrapper over [`full_piv_lu_backend`] for callers that
/// use [`Matrix`] as their dense boundary type.
///
/// # Errors
///
/// Returns an error if the backend does not support the input dtype or if the
/// factorization fails.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, full_piv_lu_matrix};
///
/// let matrix = from_vec2d(vec![vec![0.0_f64, 1.0], vec![2.0, 3.0]]);
/// let factors = full_piv_lu_matrix(&matrix).unwrap();
/// assert_eq!(factors.p.nrows(), 2);
/// assert_eq!(factors.q.ncols(), 2);
/// ```
pub fn full_piv_lu_matrix<T>(a: &Matrix<T>) -> Result<FullPivLuMatrixResult<T>>
where
    T: BackendLinalgScalar + Copy,
{
    let tensor = matrix_to_typed_tensor(a);
    let decomp = full_piv_lu_backend(&tensor)?;
    Ok(FullPivLuMatrixResult {
        p: typed_tensor_to_matrix("full_piv_lu", decomp.p)?,
        l: typed_tensor_to_matrix("full_piv_lu", decomp.l)?,
        u: typed_tensor_to_matrix("full_piv_lu", decomp.u)?,
        q: typed_tensor_to_matrix("full_piv_lu", decomp.q)?,
    })
}

#[cfg(test)]
mod tests;
