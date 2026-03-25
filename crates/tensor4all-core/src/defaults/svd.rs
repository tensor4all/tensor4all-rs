//! SVD decomposition for tensors.
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.

use crate::defaults::DynIndex;
use crate::global_default::GlobalDefault;
use crate::index_like::IndexLike;
use crate::truncation::{HasTruncationParams, TruncationParams};
use crate::{unfold_split, TensorDynLen};
use tensor4all_tensorbackend::{
    native_tensor_primal_to_dense_c64_col_major, native_tensor_primal_to_dense_f64_col_major,
    reshape_col_major_native_tensor, svd_native_tensor,
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
    match tensor.scalar_type() {
        tenferro::ScalarType::F64 => {
            native_tensor_primal_to_dense_f64_col_major(tensor).map_err(SvdError::ComputationError)
        }
        tenferro::ScalarType::C64 => native_tensor_primal_to_dense_c64_col_major(tensor)
            .map(|values| values.into_iter().map(|value| value.re).collect())
            .map_err(SvdError::ComputationError),
        other => Err(SvdError::ComputationError(anyhow::anyhow!(
            "native SVD returned unsupported singular-value scalar type {other:?}"
        ))),
    }
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
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
pub fn svd_with<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<(TensorDynLen, TensorDynLen, TensorDynLen), SvdError> {
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
        u_native = u_native.take_prefix(1, r).map_err(|e| {
            SvdError::ComputationError(anyhow::anyhow!("native SVD truncation on U failed: {e}"))
        })?;
        s_native = s_native.take_prefix(0, r).map_err(|e| {
            SvdError::ComputationError(anyhow::anyhow!(
                "native SVD truncation on singular values failed: {e}"
            ))
        })?;
        vt_native = vt_native.take_prefix(0, r).map_err(|e| {
            SvdError::ComputationError(anyhow::anyhow!("native SVD V^T truncation failed: {e}"))
        })?;
    }

    let bond_index = DynIndex::new_bond(r)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))
        .map_err(SvdError::ComputationError)?;

    let mut u_indices = left_indices;
    u_indices.push(bond_index.clone());
    let u_dims: Vec<usize> = u_indices.iter().map(|idx| idx.dim).collect();
    let u_reshaped = reshape_col_major_native_tensor(&u_native, &u_dims).map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("native SVD U reshape failed: {e}"))
    })?;
    let u = TensorDynLen::from_native(u_indices, u_reshaped).map_err(SvdError::ComputationError)?;

    let s_indices = vec![bond_index.clone(), bond_index.sim()];
    let s_diag = s_native.diag_embed(2).map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("native SVD diagonal embedding failed: {e}"))
    })?;
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

#[cfg(test)]
mod tests;
