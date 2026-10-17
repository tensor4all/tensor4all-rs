//! QR decomposition for tensors.
//!
//! This module works with concrete types (`DynIndex`, `IdxTensor`) only.

use crate::defaults::idx_tensor::unfold_split_inner;
use crate::defaults::DynIndex;
use crate::global_default::GlobalDefault;
use crate::{ExecutionContext, IdxTensor};
use num_complex::{Complex64, ComplexFloat};
use tenferro::DType;
use tenferro_ad::EagerTensor;
use tenferro_linalg::EagerSessionLinalgExt;
use thiserror::Error;

/// Error type for QR operations in tensor4all-linalg.
#[derive(Debug, Error)]
pub enum QrError {
    /// QR computation failed.
    #[error("QR computation failed: {0}")]
    ComputationError(#[from] anyhow::Error),
    /// Invalid relative tolerance value (must be finite and non-negative).
    #[error("Invalid rtol value: {0}. rtol must be finite and non-negative.")]
    InvalidRtol(f64),
}

/// Options for QR decomposition with truncation control.
///
/// # Examples
///
/// ```
/// use tensor4all_core::qr::{QrOptions, qr_with};
/// use tensor4all_core::{DynIndex, TensorContractionLike, IdxTensor};
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(3);
/// let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
/// let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// let opts = QrOptions::new().with_rtol(1e-10);
/// let (q, r) = qr_with::<f64>(&tensor, &[i], &opts).unwrap();
///
/// // Q * R recovers the original tensor
/// let recovered = q.contract_pair(&r).unwrap();
/// assert!(tensor.distance(&recovered).unwrap() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QrOptions {
    /// Relative tolerance for QR row-norm truncation.
    /// If `None`, uses the global default.
    pub rtol: Option<f64>,
    truncate: bool,
}

impl Default for QrOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl QrOptions {
    /// Create new QR options with no overrides.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rtol: None,
            truncate: true,
        }
    }

    /// Set the QR truncation tolerance.
    #[must_use]
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = Some(rtol);
        self
    }

    pub(crate) fn full_rank() -> Self {
        Self {
            rtol: None,
            truncate: false,
        }
    }
}

// Global default rtol using the unified GlobalDefault type
// Default value: 1e-15 (very strict, near machine precision)
static DEFAULT_QR_RTOL: GlobalDefault = GlobalDefault::new(1e-15);

/// Get the global default rtol for QR truncation.
///
/// The default value is 1e-15 (very strict, near machine precision).
pub fn default_qr_rtol() -> f64 {
    DEFAULT_QR_RTOL.get()
}

/// Set the global default rtol for QR truncation.
///
/// # Arguments
/// * `rtol` - Relative tolerance (must be finite and non-negative)
///
/// # Errors
/// Returns `QrError::InvalidRtol` if rtol is not finite or is negative.
pub fn set_default_qr_rtol(rtol: f64) -> Result<(), QrError> {
    DEFAULT_QR_RTOL
        .set(rtol)
        .map_err(|e| QrError::InvalidRtol(e.0))
}

fn compute_retained_rows_qr_from_dense<T>(
    r_full: &[T],
    k: usize,
    n: usize,
    rtol: f64,
) -> Result<Vec<usize>, QrError>
where
    T: ComplexFloat,
    <T as ComplexFloat>::Real: Into<f64>,
{
    if k == 0 || n == 0 {
        return Ok(vec![0]);
    }

    let max_diag = k.min(n);

    // Compute row norms of R (upper triangular: row i has entries from column i..n).
    // Use relative comparison against the maximum row norm, matching
    // compute_retained_rank_qr. The previous implementation compared diagonal
    // elements absolutely and broke at the first small value, which is incorrect
    // for non-pivoted QR where diagonal elements are not necessarily in
    // decreasing order. Neither are the row norms: a non-pivoted QR of a
    // rank-deficient matrix can leave a negligible row before a large one, so
    // the retained rows are selected individually, never as a prefix.
    let row_norms: Vec<f64> = (0..max_diag)
        .map(|i| {
            let mut norm_sq: f64 = 0.0;
            for j in i..n {
                let val: f64 = r_full[i + j * k].abs().into();
                norm_sq += val * val;
            }
            norm_sq.sqrt()
        })
        .collect();

    Ok(retained_rows_from_row_norms(&row_norms, rtol))
}

/// Select the retained rows of the upper-triangular QR factor from its row norms.
///
/// Returns the increasing indices of the rows whose norm is at least
/// `rtol * max_row_norm`; at least one row (row 0 for an all-zero factor) is kept.
/// Dropping row `i` of `R` together with column `i` of `Q` changes `Q R` by at most
/// that row's norm, wherever the row sits.
///
/// Shared by the host dense path and the device path (which reduces the norms
/// on-device and reads back only this `k`-element decision vector).
fn retained_rows_from_row_norms(row_norms: &[f64], rtol: f64) -> Vec<usize> {
    let max_row_norm = row_norms.iter().cloned().fold(0.0_f64, f64::max);
    if max_row_norm == 0.0 {
        return vec![0];
    }

    let threshold = rtol * max_row_norm;
    row_norms
        .iter()
        .enumerate()
        .filter(|&(_, &norm)| norm >= threshold)
        .map(|(row, _)| row)
        .collect()
}

/// Read the `k` upper-triangular row norms of a resident `R` factor.
///
/// All arithmetic stays in the operand's owning runtime; only the `k`-element
/// decision vector crosses the explicit readback boundary through `context`.
#[cfg(feature = "tenferro-cuda")]
fn resident_row_norms(
    r_inner: &EagerTensor,
    k: usize,
    context: &ExecutionContext,
) -> Result<Vec<f64>, QrError> {
    // The resident factor's own runtime runs the reduction in one eager session, as the neighbouring CUDA-only
    // paths do; only the decision vector crosses back through `context`.
    let sum_squares = r_inner
        .runtime()
        .with_eager_session(|session| {
            let upper = session.triu(r_inner, 0)?;
            let magnitudes = session.abs(&upper)?;
            session.reduce_sum_squares(&magnitudes, Some(&[1]))
        })
        .map_err(|error| QrError::ComputationError(anyhow::Error::new(error)))?;
    if sum_squares.shape() != [k] {
        return Err(QrError::ComputationError(anyhow::anyhow!(
            "device QR row-norm reduction returned shape {:?}, expected [{k}]",
            sum_squares.shape()
        )));
    }
    let decision = IdxTensor::from_inner(vec![DynIndex::new_dyn(k)], sum_squares)
        .map_err(QrError::ComputationError)?;
    let values = decision
        .read_decision_data(context)
        .map_err(|error| QrError::ComputationError(anyhow::Error::new(error)))?;
    Ok(values.into_iter().map(|value| value.sqrt()).collect())
}

/// Compute QR decomposition of a tensor with arbitrary rank, returning (Q, R).
///
/// This function uses the global default rtol for truncation.
/// See `qr_with` for per-call rtol control.
///
/// This function computes the thin QR decomposition, where for an unfolded matrix A (m×n),
/// we return Q (m×k) and R (k×n) with k = min(m, n).
///
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
///
/// Truncation is performed based on R's row norms: rows whose norm is below
/// `rtol * max_row_norm` are discarded.
///
/// For the mathematical convention:
/// \[ A = Q * R \]
/// where Q is orthogonal (or unitary for complex) and R is upper triangular.
///
/// # Arguments
/// * `t` - Input tensor with DenseF64 or DenseC64 storage
/// * `left_inds` - Indices to place on the left (row) side of the unfolded matrix
///
/// # Returns
/// A tuple `(Q, R)` where:
/// - `Q` is a tensor with indices `[left_inds..., bond_index]` and dimensions `[left_dims..., r]`
/// - `R` is a tensor with indices `[bond_index, right_inds...]` and dimensions `[r, right_dims...]`
///
///   where `r` is the retained rank (≤ min(m, n)) determined by rtol truncation.
///
/// # Errors
/// Returns `QrError` if:
/// - The tensor rank is < 2
/// - Storage is not DenseF64 or DenseC64
/// - `left_inds` is empty or contains all indices
/// - `left_inds` contains indices not in the tensor or duplicates
/// - The QR computation fails
///
/// # Examples
///
/// ```
/// use tensor4all_core::{IdxTensor, DynIndex, qr};
///
/// // Create a 4x3 matrix
/// let i = DynIndex::new_dyn(4);
/// let j = DynIndex::new_dyn(3);
/// // Identity-like data (4x3 column-major)
/// let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
/// let t = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// let (q, r) = qr::<f64>(&t, &[i.clone()]).unwrap();
///
/// // Q has shape (4, bond) and R has shape (bond, 3)
/// assert_eq!(q.dims()[0], 4);
/// assert_eq!(r.dims()[r.dims().len() - 1], 3);
/// ```
pub fn qr<T>(t: &IdxTensor, left_inds: &[DynIndex]) -> Result<(IdxTensor, IdxTensor), QrError> {
    qr_with::<T>(t, left_inds, &QrOptions::default())
}

/// Compute QR decomposition of a tensor with arbitrary rank, returning (Q, R).
///
/// This function allows per-call control of the truncation tolerance via `QrOptions`.
/// If `options.rtol` is `None`, uses the global default rtol.
///
/// This function computes the thin QR decomposition, where for an unfolded matrix A (m×n),
/// we return Q (m×k) and R (k×n) with k = min(m, n).
///
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
///
/// Truncation is performed based on R's row norms: rows whose norm is below
/// `rtol * max_row_norm` are discarded.
///
/// For the mathematical convention:
/// \[ A = Q * R \]
/// where Q is orthogonal (or unitary for complex) and R is upper triangular.
///
/// # Arguments
/// * `t` - Input tensor with DenseF64 or DenseC64 storage
/// * `left_inds` - Indices to place on the left (row) side of the unfolded matrix
/// * `options` - QR options including rtol for truncation control
///
/// # Returns
/// A tuple `(Q, R)` where:
/// - `Q` is a tensor with indices `[left_inds..., bond_index]` and dimensions `[left_dims..., r]`
/// - `R` is a tensor with indices `[bond_index, right_inds...]` and dimensions `[r, right_dims...]`
///
///   where `r` is the retained rank (≤ min(m, n)) determined by rtol truncation.
///
/// # Errors
/// Returns `QrError` if:
/// - The tensor rank is < 2
/// - Storage is not DenseF64 or DenseC64
/// - `left_inds` is empty or contains all indices
/// - `left_inds` contains indices not in the tensor or duplicates
/// - The QR computation fails
/// - `options.rtol` is invalid (not finite or negative)
pub fn qr_with<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &QrOptions,
) -> Result<(IdxTensor, IdxTensor), QrError> {
    if t.is_cuda_resident() {
        return Err(QrError::ComputationError(anyhow::anyhow!(
            "CUDA-resident input requires qr_with_in with the owning execution context"
        )));
    }
    // Unfold tensor into an eager rank-2 tensor so linalg AD nodes stay connected.
    let (matrix_inner, _, m, n, left_indices, right_indices) = unfold_split_inner(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
        .map_err(QrError::ComputationError)?;
    let k = m.min(n);
    let (q_inner, r_inner) = matrix_inner
        .runtime()
        .with_eager_session(|session| session.qr(&matrix_inner))
        .map_err(|e| QrError::ComputationError(anyhow::anyhow!("{e}")))?;

    let keep = qr_retained_rows(&r_inner, k, n, options, None)?;
    qr_assemble(q_inner, r_inner, k, keep, left_indices, right_indices)
}

/// Compute QR decomposition in a caller-owned execution context.
///
/// The factorization, truncation slicing, and result assembly all execute in
/// `context`. With a CUDA context the retained-rank decision reads back only
/// the `k`-element row-norm vector through [`IdxTensor::read_decision_data`];
/// with a CPU context the rank decision uses the same host read as [`qr_with`].
///
/// # Arguments
///
/// * `t` - Input tensor, which must belong to `context`.
/// * `left_inds` - Indices to place on the left (row) side of the unfolded matrix.
/// * `options` - QR options including rtol for truncation control.
/// * `context` - Caller-owned execution context owning the input and results.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use tensor4all_core::qr::{QrOptions, qr_with_in};
/// use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor, TensorContractionLike};
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = ExecutionContext::Cpu(Arc::new(
///     CpuExecutionContext::from_backend(CpuBackend::new()),
/// ));
/// let i = DynIndex::new_dyn(4);
/// let j = DynIndex::new_dyn(3);
/// let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
/// let t = IdxTensor::from_dense_in(&context, vec![i.clone(), j.clone()], data)?;
/// let (q, r) = qr_with_in::<f64>(&t, &[i.clone()], &QrOptions::default(), &context)?;
/// let recovered = q.contract_pair(&r)?;
/// let expected = t.to_vec::<f64>()?;
/// let actual = recovered.to_vec::<f64>()?;
/// let residual = expected
///     .iter()
///     .zip(actual.iter())
///     .map(|(a, b)| (a - b).abs())
///     .fold(0.0_f64, f64::max);
/// assert!(residual < 1e-12, "QR reconstruction residual {residual}");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns `QrError` when the tensor does not belong to `context`, when the
/// indices, storage, or options are invalid, or when the factorization or
/// explicit decision readback fails. When this build cannot execute in
/// `context`, the rejection carries
/// [`IdxTensorError::UnsupportedExecutionContext`](crate::IdxTensorError::UnsupportedExecutionContext)
/// as its source.
pub fn qr_with_in<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &QrOptions,
    context: &ExecutionContext,
) -> Result<(IdxTensor, IdxTensor), QrError> {
    t.validate_context(context)
        .map_err(|error| QrError::ComputationError(anyhow::Error::new(error)))?;
    let (matrix_inner, _, m, n, left_indices, right_indices) = unfold_split_inner(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
        .map_err(QrError::ComputationError)?;
    let k = m.min(n);
    let (q_inner, r_inner) = matrix_inner
        .runtime()
        .with_eager_session(|session| session.qr(&matrix_inner))
        .map_err(|e| QrError::ComputationError(anyhow::anyhow!("{e}")))?;

    let keep = qr_retained_rows(&r_inner, k, n, options, Some(context))?;
    qr_assemble(q_inner, r_inner, k, keep, left_indices, right_indices)
}

/// Select the retained rows of `R`, reading the decision payload through `context`
/// for device-resident factors and directly from host memory otherwise.
fn qr_retained_rows(
    r_inner: &EagerTensor,
    k: usize,
    n: usize,
    options: &QrOptions,
    context: Option<&ExecutionContext>,
) -> Result<Vec<usize>, QrError> {
    if !options.truncate {
        return Ok((0..k).collect());
    }
    // Determine rtol to use
    let rtol = options.rtol.unwrap_or(default_qr_rtol());
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(QrError::InvalidRtol(rtol));
    }

    #[cfg(feature = "tenferro-cuda")]
    if let Some(context) = context {
        if matches!(context, ExecutionContext::Cuda(_)) {
            let row_norms = resident_row_norms(r_inner, k, context)?;
            return Ok(retained_rows_from_row_norms(&row_norms, rtol));
        }
    }
    #[cfg(not(feature = "tenferro-cuda"))]
    let _ = context;

    let value = r_inner
        .value()
        .map_err(|source| QrError::ComputationError(anyhow::Error::new(source)))?;
    match r_inner.dtype() {
        DType::F64 => {
            let values = value
                .as_slice::<f64>()
                .map_err(|source| QrError::ComputationError(anyhow::Error::new(source)))?;
            compute_retained_rows_qr_from_dense(values, k, n, rtol)
        }
        DType::C64 => {
            let values = value
                .as_slice::<Complex64>()
                .map_err(|source| QrError::ComputationError(anyhow::Error::new(source)))?;
            compute_retained_rows_qr_from_dense(values, k, n, rtol)
        }
        other => Err(QrError::ComputationError(anyhow::anyhow!(
            "native QR returned unsupported scalar type {other:?}"
        ))),
    }
}

/// Slice the thin factors to the retained rows of `R` (and the matching columns
/// of `Q`) and wrap them as [`IdxTensor`]s.
fn qr_assemble(
    mut q_inner: EagerTensor,
    mut r_inner: EagerTensor,
    k: usize,
    keep: Vec<usize>,
    left_indices: Vec<DynIndex>,
    right_indices: Vec<DynIndex>,
) -> Result<(IdxTensor, IdxTensor), QrError> {
    let r = keep.len();
    if r < k {
        q_inner = q_inner
            .runtime()
            .with_eager_session(|session| session.take_axis(&q_inner, 1, &keep))
            .map_err(|e| QrError::ComputationError(anyhow::anyhow!("{e}")))?;
        r_inner = r_inner
            .runtime()
            .with_eager_session(|session| session.take_axis(&r_inner, 0, &keep))
            .map_err(|e| QrError::ComputationError(anyhow::anyhow!("{e}")))?;
    }

    let bond_index = DynIndex::new_bond(r)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))
        .map_err(QrError::ComputationError)?;

    let mut q_indices = left_indices.clone();
    q_indices.push(bond_index.clone());
    let q_dims: Vec<usize> = q_indices.iter().map(|idx| idx.dim).collect();
    let q_reshaped = q_inner
        .runtime()
        .with_eager_session(|session| session.reshape(&q_inner, &q_dims))
        .map_err(|e| {
            QrError::ComputationError(anyhow::anyhow!("eager QR Q reshape failed: {e}"))
        })?;
    let q = IdxTensor::from_inner(q_indices, q_reshaped).map_err(QrError::ComputationError)?;

    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(&right_indices);
    let r_dims: Vec<usize> = r_indices.iter().map(|idx| idx.dim).collect();
    let r_reshaped = r_inner
        .runtime()
        .with_eager_session(|session| session.reshape(&r_inner, &r_dims))
        .map_err(|e| {
            QrError::ComputationError(anyhow::anyhow!("eager QR R reshape failed: {e}"))
        })?;
    let r = IdxTensor::from_inner(r_indices, r_reshaped).map_err(QrError::ComputationError)?;

    Ok((q, r))
}

#[cfg(test)]
mod tests;
