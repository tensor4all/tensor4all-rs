//! QR decomposition for tensors.
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.

use crate::defaults::DynIndex;
use crate::global_default::GlobalDefault;
use crate::truncation::TruncationParams;
use crate::{unfold_split, TensorDynLen};
use num_complex::ComplexFloat;
use tensor4all_tensorbackend::{
    dense_native_tensor_from_col_major, native_tensor_primal_to_dense_c64_col_major,
    native_tensor_primal_to_dense_f64_col_major, qr_native_tensor, reshape_col_major_native_tensor,
    TensorElement,
};
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
#[derive(Debug, Clone, Copy, Default)]
pub struct QrOptions {
    /// Truncation parameters (rtol only for QR).
    pub truncation: TruncationParams,
}

impl QrOptions {
    /// Create new QR options with the specified rtol.
    pub fn with_rtol(rtol: f64) -> Self {
        Self {
            truncation: TruncationParams::new().with_rtol(rtol),
        }
    }

    /// Get rtol from options (for backwards compatibility).
    pub fn rtol(&self) -> Option<f64> {
        self.truncation.rtol
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

fn compute_retained_rank_qr_from_dense<T>(
    r_full: &[T],
    k: usize,
    n: usize,
    rtol: f64,
) -> Result<usize, QrError>
where
    T: ComplexFloat,
    <T as ComplexFloat>::Real: Into<f64>,
{
    if k == 0 || n == 0 {
        return Ok(1);
    }

    let max_diag = k.min(n);

    // Compute row norms of R (upper triangular: row i has entries from column i..n).
    // Use relative comparison against the maximum row norm, matching
    // compute_retained_rank_qr. The previous implementation compared diagonal
    // elements absolutely and broke at the first small value, which is incorrect
    // for non-pivoted QR where diagonal elements are not necessarily in
    // decreasing order.
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

    let max_row_norm = row_norms.iter().cloned().fold(0.0_f64, f64::max);
    if max_row_norm == 0.0 {
        return Ok(1);
    }

    let threshold = rtol * max_row_norm;
    let r = row_norms.iter().filter(|&&norm| norm >= threshold).count();
    Ok(r.max(1))
}

fn truncate_matrix_cols<T: TensorElement>(
    data: &[T],
    rows: usize,
    keep_cols: usize,
) -> anyhow::Result<tenferro::Tensor> {
    dense_native_tensor_from_col_major(&data[..rows * keep_cols], &[rows, keep_cols])
}

fn truncate_matrix_rows<T: TensorElement>(
    data: &[T],
    rows: usize,
    cols: usize,
    keep_rows: usize,
) -> anyhow::Result<tenferro::Tensor> {
    let mut truncated = Vec::with_capacity(keep_rows * cols);
    for col in 0..cols {
        let start = col * rows;
        truncated.extend_from_slice(&data[start..start + keep_rows]);
    }
    dense_native_tensor_from_col_major(&truncated, &[keep_rows, cols])
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
///   where `r` is the retained rank (≤ min(m, n)) determined by rtol truncation.
///
/// # Errors
/// Returns `QrError` if:
/// - The tensor rank is < 2
/// - Storage is not DenseF64 or DenseC64
/// - `left_inds` is empty or contains all indices
/// - `left_inds` contains indices not in the tensor or duplicates
/// - The QR computation fails
pub fn qr<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
) -> Result<(TensorDynLen, TensorDynLen), QrError> {
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
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &QrOptions,
) -> Result<(TensorDynLen, TensorDynLen), QrError> {
    // Determine rtol to use
    let rtol = options.truncation.effective_rtol(default_qr_rtol());
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(QrError::InvalidRtol(rtol));
    }

    // Unfold tensor into a native rank-2 tensor.
    let (matrix_native, _, m, n, left_indices, right_indices) = unfold_split(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
        .map_err(QrError::ComputationError)?;
    let k = m.min(n);
    let (mut q_native, mut r_native) =
        qr_native_tensor(&matrix_native).map_err(QrError::ComputationError)?;
    let r = match r_native.scalar_type() {
        tenferro::ScalarType::F64 => {
            let values = native_tensor_primal_to_dense_f64_col_major(&r_native)
                .map_err(QrError::ComputationError)?;
            compute_retained_rank_qr_from_dense(&values, k, n, rtol)?
        }
        tenferro::ScalarType::C64 => {
            let values = native_tensor_primal_to_dense_c64_col_major(&r_native)
                .map_err(QrError::ComputationError)?;
            compute_retained_rank_qr_from_dense(&values, k, n, rtol)?
        }
        other => {
            return Err(QrError::ComputationError(anyhow::anyhow!(
                "native QR returned unsupported scalar type {other:?}"
            )));
        }
    };
    if r < k {
        match q_native.scalar_type() {
            tenferro::ScalarType::F64 => {
                let q_values = native_tensor_primal_to_dense_f64_col_major(&q_native)
                    .map_err(QrError::ComputationError)?;
                let r_values = native_tensor_primal_to_dense_f64_col_major(&r_native)
                    .map_err(QrError::ComputationError)?;
                q_native =
                    truncate_matrix_cols(&q_values, m, r).map_err(QrError::ComputationError)?;
                r_native =
                    truncate_matrix_rows(&r_values, k, n, r).map_err(QrError::ComputationError)?;
            }
            tenferro::ScalarType::C64 => {
                let q_values = native_tensor_primal_to_dense_c64_col_major(&q_native)
                    .map_err(QrError::ComputationError)?;
                let r_values = native_tensor_primal_to_dense_c64_col_major(&r_native)
                    .map_err(QrError::ComputationError)?;
                q_native =
                    truncate_matrix_cols(&q_values, m, r).map_err(QrError::ComputationError)?;
                r_native =
                    truncate_matrix_rows(&r_values, k, n, r).map_err(QrError::ComputationError)?;
            }
            other => {
                return Err(QrError::ComputationError(anyhow::anyhow!(
                    "native QR returned unsupported scalar type {other:?}"
                )));
            }
        }
    }

    let bond_index = DynIndex::new_bond(r)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))
        .map_err(QrError::ComputationError)?;

    let mut q_indices = left_indices.clone();
    q_indices.push(bond_index.clone());
    let q_dims: Vec<usize> = q_indices.iter().map(|idx| idx.dim).collect();
    let q_reshaped = reshape_col_major_native_tensor(&q_native, &q_dims).map_err(|e| {
        QrError::ComputationError(anyhow::anyhow!("native QR Q reshape failed: {e}"))
    })?;
    let q = TensorDynLen::from_native(q_indices, q_reshaped).map_err(QrError::ComputationError)?;

    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(&right_indices);
    let r_dims: Vec<usize> = r_indices.iter().map(|idx| idx.dim).collect();
    let r_reshaped = reshape_col_major_native_tensor(&r_native, &r_dims).map_err(|e| {
        QrError::ComputationError(anyhow::anyhow!("native QR R reshape failed: {e}"))
    })?;
    let r = TensorDynLen::from_native(r_indices, r_reshaped).map_err(QrError::ComputationError)?;

    Ok((q, r))
}

#[cfg(test)]
mod tests;
