//! SVD decomposition for tensors.
//!
//! Provides [`svd`] and [`svd_with`] for computing truncated SVD of
//! [`TensorDynLen`] values. The tensor is unfolded into a matrix by
//! splitting its indices into left and right groups, then the standard
//! matrix SVD is computed and truncated according to [`SvdOptions`].
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.

use crate::defaults::DynIndex;
use crate::global_default::GlobalDefault;
use crate::index_like::IndexLike;
use crate::truncation::{HasTruncationParams, TruncationParams};
use crate::{unfold_split, TensorDynLen};
use tenferro::DType;
use tensor4all_tensorbackend::{
    dense_native_tensor_from_col_major, diag_native_tensor_from_col_major,
    native_tensor_primal_to_dense_c64_col_major, native_tensor_primal_to_dense_f64_col_major,
    reshape_col_major_native_tensor, svd_native_tensor, TensorElement,
};
use thiserror::Error;

/// Error type for SVD operations in tensor4all-linalg.
#[derive(Debug, Error)]
pub enum SvdError {
    /// SVD computation failed.
    #[error("SVD computation failed: {0}")]
    ComputationError(#[from] anyhow::Error),
    /// Invalid relative tolerance value (must be finite and non-negative).
    #[error("Invalid rtol value: {0}. rtol must be finite and non-negative.")]
    InvalidRtol(f64),
}

/// Options for SVD decomposition with truncation control.
///
/// # Examples
///
/// ```
/// use tensor4all_core::svd::{SvdOptions, svd_with};
/// use tensor4all_core::{DynIndex, TensorDynLen};
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(3);
/// let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
/// let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// let opts = SvdOptions::with_rtol(1e-10);
/// let (u, s, v) = svd_with::<f64>(&tensor, &[i.clone()], &opts).unwrap();
///
/// // U has left index + bond, S is diagonal bond x bond, V has right index + bond
/// assert_eq!(u.dims()[0], 3);
/// assert_eq!(s.dims().len(), 2);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SvdOptions {
    /// Truncation parameters (rtol, max_rank).
    pub truncation: TruncationParams,
}

impl SvdOptions {
    /// Create new SVD options with the specified rtol.
    pub fn with_rtol(rtol: f64) -> Self {
        Self {
            truncation: TruncationParams::new().with_rtol(rtol),
        }
    }

    /// Create new SVD options with the specified max_rank.
    pub fn with_max_rank(max_rank: usize) -> Self {
        Self {
            truncation: TruncationParams::new().with_max_rank(max_rank),
        }
    }

    /// Get rtol from options (for backwards compatibility).
    pub fn rtol(&self) -> Option<f64> {
        self.truncation.rtol
    }

    /// Get max_rank from options (for backwards compatibility).
    pub fn max_rank(&self) -> Option<usize> {
        self.truncation.max_rank
    }
}

impl HasTruncationParams for SvdOptions {
    fn truncation_params(&self) -> &TruncationParams {
        &self.truncation
    }

    fn truncation_params_mut(&mut self) -> &mut TruncationParams {
        &mut self.truncation
    }
}

// Global default rtol using the unified GlobalDefault type
// Default value: 1e-12 (near machine precision)
static DEFAULT_SVD_RTOL: GlobalDefault = GlobalDefault::new(1e-12);

/// Get the global default rtol for SVD truncation.
///
/// The default value is 1e-12 (near machine precision).
pub fn default_svd_rtol() -> f64 {
    DEFAULT_SVD_RTOL.get()
}

/// Set the global default rtol for SVD truncation.
///
/// # Arguments
/// * `rtol` - Relative Frobenius error tolerance (must be finite and non-negative)
///
/// # Errors
/// Returns `SvdError::InvalidRtol` if rtol is not finite or is negative.
pub fn set_default_svd_rtol(rtol: f64) -> Result<(), SvdError> {
    DEFAULT_SVD_RTOL
        .set(rtol)
        .map_err(|e| SvdError::InvalidRtol(e.0))
}

/// Compute the retained rank based on rtol (TSVD truncation).
///
/// This implements the truncation criterion:
///   sum_{i>r} σ_i² / sum_i σ_i² <= rtol²
fn compute_retained_rank(s_vec: &[f64], rtol: f64) -> usize {
    if s_vec.is_empty() {
        return 1;
    }

    let total_sq_norm: f64 = s_vec.iter().map(|&s| s * s).sum();
    if total_sq_norm == 0.0 {
        return 1;
    }

    let threshold = rtol * rtol * total_sq_norm;
    let mut discarded_sq_norm = 0.0;
    let mut r = s_vec.len();
    for i in (0..s_vec.len()).rev() {
        let s_sq = s_vec[i] * s_vec[i];
        if discarded_sq_norm + s_sq <= threshold {
            discarded_sq_norm += s_sq;
            r = i;
        } else {
            break;
        }
    }
    r.max(1)
}

fn singular_values_from_native(tensor: &tenferro::Tensor) -> Result<Vec<f64>, SvdError> {
    match tensor.dtype() {
        DType::F64 => {
            native_tensor_primal_to_dense_f64_col_major(tensor).map_err(SvdError::ComputationError)
        }
        DType::C64 => native_tensor_primal_to_dense_c64_col_major(tensor)
            .map(|values| values.into_iter().map(|value| value.re).collect())
            .map_err(SvdError::ComputationError),
        other => Err(SvdError::ComputationError(anyhow::anyhow!(
            "native SVD returned unsupported singular-value scalar type {other:?}"
        ))),
    }
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

type SvdTruncatedNativeResult = (
    tenferro::Tensor,
    tenferro::Tensor,
    tenferro::Tensor,
    Vec<f64>,
    DynIndex,
    Vec<DynIndex>,
    Vec<DynIndex>,
);

fn svd_truncated_native(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<SvdTruncatedNativeResult, SvdError> {
    let rtol = options.truncation.effective_rtol(default_svd_rtol());
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(SvdError::InvalidRtol(rtol));
    }

    let (matrix_native, _, m, n, left_indices, right_indices) = unfold_split(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
        .map_err(SvdError::ComputationError)?;
    let k = m.min(n);

    let (mut u_native, mut s_native, mut vt_native) =
        svd_native_tensor(&matrix_native).map_err(SvdError::ComputationError)?;
    let s_full = singular_values_from_native(&s_native)?;
    let mut r = compute_retained_rank(&s_full, rtol);
    if let Some(max_rank) = options.truncation.max_rank {
        r = r.min(max_rank);
    }
    if r < k {
        match u_native.dtype() {
            DType::F64 => {
                let u_values = native_tensor_primal_to_dense_f64_col_major(&u_native)
                    .map_err(SvdError::ComputationError)?;
                let vt_values = native_tensor_primal_to_dense_f64_col_major(&vt_native)
                    .map_err(SvdError::ComputationError)?;
                u_native =
                    truncate_matrix_cols(&u_values, m, r).map_err(SvdError::ComputationError)?;
                vt_native = truncate_matrix_rows(&vt_values, k, n, r)
                    .map_err(SvdError::ComputationError)?;
            }
            DType::C64 => {
                let u_values = native_tensor_primal_to_dense_c64_col_major(&u_native)
                    .map_err(SvdError::ComputationError)?;
                let vt_values = native_tensor_primal_to_dense_c64_col_major(&vt_native)
                    .map_err(SvdError::ComputationError)?;
                u_native =
                    truncate_matrix_cols(&u_values, m, r).map_err(SvdError::ComputationError)?;
                vt_native = truncate_matrix_rows(&vt_values, k, n, r)
                    .map_err(SvdError::ComputationError)?;
            }
            other => {
                return Err(SvdError::ComputationError(anyhow::anyhow!(
                    "native SVD returned unsupported singular-vector scalar type {other:?}"
                )));
            }
        }
        s_native = dense_native_tensor_from_col_major(&s_full[..r], &[r])
            .map_err(SvdError::ComputationError)?;
    }

    let bond_index = DynIndex::new_bond(r)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))
        .map_err(SvdError::ComputationError)?;
    let singular_values = s_full[..r].to_vec();

    Ok((
        u_native,
        s_native,
        vt_native,
        singular_values,
        bond_index,
        left_indices,
        right_indices,
    ))
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
///
/// # Examples
///
/// ```
/// use tensor4all_core::{TensorDynLen, DynIndex, svd};
///
/// // Create a 2x3 matrix (rank-1 outer product: all-ones)
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let data = vec![1.0_f64; 6]; // all-ones 2x3 matrix
/// let t = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// let (u, s, v) = svd::<f64>(&t, &[i.clone()]).unwrap();
///
/// // U: shape (left_dim, bond) = (2, bond)
/// assert_eq!(u.dims()[0], 2);
/// // V: shape (right_dim, bond) = (3, bond)
/// assert_eq!(v.dims()[0], 3);
/// // S is a diagonal matrix (bond × bond)
/// assert_eq!(s.dims().len(), 2);
/// ```
pub fn svd<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
) -> Result<(TensorDynLen, TensorDynLen, TensorDynLen), SvdError> {
    svd_with::<T>(t, left_inds, &SvdOptions::default())
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
///
/// This function allows per-call control of the truncation tolerance via `SvdOptions`.
/// If `options.rtol` is `None`, uses the global default rtol.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_core::svd::{SvdOptions, svd_with};
///
/// let i = DynIndex::new_dyn(4);
/// let j = DynIndex::new_dyn(4);
/// // Rank-1 matrix
/// let mut data = vec![0.0_f64; 16];
/// data[0] = 1.0;
/// let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// // Truncate with tight tolerance => rank 1
/// let opts = SvdOptions::with_rtol(1e-10);
/// let (u, s, _v) = svd_with::<f64>(&tensor, &[i.clone()], &opts).unwrap();
/// assert_eq!(s.dims()[0], 1);  // rank-1
///
/// // Truncate with max_rank => capped
/// let opts = SvdOptions::with_max_rank(2);
/// let (_u, s, _v) = svd_with::<f64>(&tensor, &[i.clone()], &opts).unwrap();
/// assert!(s.dims()[0] <= 2);
/// ```
pub fn svd_with<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<(TensorDynLen, TensorDynLen, TensorDynLen), SvdError> {
    let (u_native, s_native, vt_native, _singular_values, bond_index, left_indices, right_indices) =
        svd_truncated_native(t, left_inds, options)?;

    let mut u_indices = left_indices;
    u_indices.push(bond_index.clone());
    let u_dims: Vec<usize> = u_indices.iter().map(|idx| idx.dim).collect();
    let u_reshaped = reshape_col_major_native_tensor(&u_native, &u_dims).map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("native SVD U reshape failed: {e}"))
    })?;
    let u = TensorDynLen::from_native(u_indices, u_reshaped).map_err(SvdError::ComputationError)?;

    let s_indices = vec![bond_index.clone(), bond_index.sim()];
    let s_diag = diag_native_tensor_from_col_major(&singular_values_from_native(&s_native)?, 2)
        .map_err(SvdError::ComputationError)?;
    let s = TensorDynLen::from_native(s_indices, s_diag).map_err(SvdError::ComputationError)?;

    let mut vh_indices = vec![bond_index.clone()];
    vh_indices.extend(right_indices);
    let vh_dims: Vec<usize> = vh_indices.iter().map(|idx| idx.dim).collect();
    let vt_reshaped = reshape_col_major_native_tensor(&vt_native, &vh_dims).map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("native SVD V^T reshape failed: {e}"))
    })?;
    let vh =
        TensorDynLen::from_native(vh_indices, vt_reshaped).map_err(SvdError::ComputationError)?;
    let perm: Vec<usize> = (1..vh.indices.len()).chain(std::iter::once(0)).collect();
    let v = vh.conj().permute(&perm);

    Ok((u, s, v))
}

/// SVD result for factorization, returning `V^H` directly.
pub(crate) struct SvdFactorizeResult {
    pub u: TensorDynLen,
    pub s: TensorDynLen,
    pub vh: TensorDynLen,
    pub bond_index: DynIndex,
    pub singular_values: Vec<f64>,
    pub rank: usize,
}

/// Compute truncated SVD for factorization, returning `V^H` instead of `V`.
pub(crate) fn svd_for_factorize(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<SvdFactorizeResult, SvdError> {
    let (u_native, s_native, vt_native, singular_values, bond_index, left_indices, right_indices) =
        svd_truncated_native(t, left_inds, options)?;
    let rank = singular_values.len();

    let mut u_indices = left_indices;
    u_indices.push(bond_index.clone());
    let u_dims: Vec<usize> = u_indices.iter().map(|idx| idx.dim).collect();
    let u_reshaped = reshape_col_major_native_tensor(&u_native, &u_dims).map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("native SVD U reshape failed: {e}"))
    })?;
    let u = TensorDynLen::from_native(u_indices, u_reshaped).map_err(SvdError::ComputationError)?;

    let s_indices = vec![bond_index.clone(), bond_index.sim()];
    let s_diag = diag_native_tensor_from_col_major(&singular_values_from_native(&s_native)?, 2)
        .map_err(SvdError::ComputationError)?;
    let s = TensorDynLen::from_native(s_indices, s_diag).map_err(SvdError::ComputationError)?;

    let mut vh_indices = vec![bond_index.clone()];
    vh_indices.extend(right_indices);
    let vh_dims: Vec<usize> = vh_indices.iter().map(|idx| idx.dim).collect();
    let vt_reshaped = reshape_col_major_native_tensor(&vt_native, &vh_dims).map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("native SVD V^T reshape failed: {e}"))
    })?;
    let vh =
        TensorDynLen::from_native(vh_indices, vt_reshaped).map_err(SvdError::ComputationError)?;

    Ok(SvdFactorizeResult {
        u,
        s,
        vh,
        bond_index,
        singular_values,
        rank,
    })
}

#[cfg(test)]
mod tests;
