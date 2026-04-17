//! SVD decomposition for tensors.
//!
//! Provides [`svd`] and [`svd_with`] for computing truncated SVD of
//! [`TensorDynLen`] values. The tensor is unfolded into a matrix by
//! splitting its indices into left and right groups, then the standard
//! matrix SVD is computed and truncated according to [`SvdOptions`].
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.

use crate::defaults::DynIndex;
use crate::index_like::IndexLike;
use crate::truncation::{
    validate_svd_truncation_policy, SingularValueMeasure, SvdTruncationPolicy, ThresholdScale,
    TruncationRule,
};
use crate::{unfold_split, TensorDynLen};
use std::sync::Mutex;
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
    /// Invalid truncation threshold value (must be finite and non-negative).
    #[error("Invalid SVD truncation threshold: {0}. Threshold must be finite and non-negative.")]
    InvalidThreshold(f64),
}

/// Options for SVD decomposition with truncation control.
///
/// # Examples
///
/// ```
/// use tensor4all_core::svd::{SvdOptions, svd_with};
/// use tensor4all_core::{DynIndex, SvdTruncationPolicy, TensorDynLen};
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(3);
/// let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
/// let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// let opts = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-10));
/// let (u, s, v) = svd_with::<f64>(&tensor, &[i.clone()], &opts).unwrap();
///
/// // U has left index + bond, S is diagonal bond x bond, V has right index + bond
/// assert_eq!(u.dims()[0], 3);
/// assert_eq!(s.dims().len(), 2);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SvdOptions {
    /// Maximum retained rank after policy-based truncation.
    pub max_rank: Option<usize>,
    /// Per-call SVD truncation policy.
    /// If `None`, the global default policy is used.
    pub policy: Option<SvdTruncationPolicy>,
}

impl SvdOptions {
    /// Create new SVD options with no overrides.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum retained rank.
    #[must_use]
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = Some(max_rank);
        self
    }

    /// Set the SVD truncation policy override.
    #[must_use]
    pub fn with_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.policy = Some(policy);
        self
    }
}

fn default_policy_guard() -> std::sync::MutexGuard<'static, SvdTruncationPolicy> {
    match DEFAULT_SVD_TRUNCATION_POLICY.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

// Default value: relative per-value threshold 1e-12.
static DEFAULT_SVD_TRUNCATION_POLICY: Mutex<SvdTruncationPolicy> =
    Mutex::new(SvdTruncationPolicy::new(1e-12));

/// Get the global default truncation policy for SVD.
///
/// The default policy is `SvdTruncationPolicy::new(1e-12)`.
#[must_use]
pub fn default_svd_truncation_policy() -> SvdTruncationPolicy {
    *default_policy_guard()
}

/// Set the global default truncation policy for SVD.
///
/// # Arguments
/// * `policy` - SVD truncation policy to use when `SvdOptions::policy` is `None`
///
/// # Errors
/// Returns `SvdError::InvalidThreshold` if `policy.threshold` is invalid.
pub fn set_default_svd_truncation_policy(policy: SvdTruncationPolicy) -> Result<(), SvdError> {
    validate_svd_truncation_policy(policy).map_err(|e| SvdError::InvalidThreshold(e.0))?;
    *default_policy_guard() = policy;
    Ok(())
}

fn singular_value_measure(value: f64, measure: SingularValueMeasure) -> f64 {
    match measure {
        SingularValueMeasure::Value => value,
        SingularValueMeasure::SquaredValue => value * value,
    }
}

/// Compute the retained rank based on an explicit SVD truncation policy.
fn compute_retained_rank(s_vec: &[f64], policy: &SvdTruncationPolicy) -> usize {
    if s_vec.is_empty() {
        return 1;
    }

    let measured: Vec<f64> = s_vec
        .iter()
        .map(|&value| singular_value_measure(value, policy.measure))
        .collect();
    if measured.iter().all(|&value| value == 0.0) {
        return 1;
    }

    let retained = match (policy.scale, policy.rule) {
        (ThresholdScale::Relative, TruncationRule::PerValue) => {
            let reference = measured.iter().copied().fold(0.0_f64, f64::max);
            measured
                .iter()
                .take_while(|&&value| reference > 0.0 && value / reference > policy.threshold)
                .count()
        }
        (ThresholdScale::Absolute, TruncationRule::PerValue) => measured
            .iter()
            .take_while(|&&value| value > policy.threshold)
            .count(),
        (ThresholdScale::Relative, TruncationRule::DiscardedTailSum) => {
            let total: f64 = measured.iter().sum();
            if total == 0.0 {
                1
            } else {
                let mut discarded = 0.0;
                let mut keep = measured.len();
                for (i, value) in measured.iter().enumerate().rev() {
                    if (discarded + value) / total <= policy.threshold {
                        discarded += value;
                        keep = i;
                    } else {
                        break;
                    }
                }
                keep
            }
        }
        (ThresholdScale::Absolute, TruncationRule::DiscardedTailSum) => {
            let mut discarded = 0.0;
            let mut keep = measured.len();
            for (i, value) in measured.iter().enumerate().rev() {
                if discarded + value <= policy.threshold {
                    discarded += value;
                    keep = i;
                } else {
                    break;
                }
            }
            keep
        }
    };

    retained.max(1)
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
    let policy = options.policy.unwrap_or_else(default_svd_truncation_policy);
    validate_svd_truncation_policy(policy).map_err(|e| SvdError::InvalidThreshold(e.0))?;

    let (matrix_native, _, m, n, left_indices, right_indices) = unfold_split(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
        .map_err(SvdError::ComputationError)?;
    let k = m.min(n);

    let (mut u_native, mut s_native, mut vt_native) =
        svd_native_tensor(&matrix_native).map_err(SvdError::ComputationError)?;
    let s_full = singular_values_from_native(&s_native)?;
    let mut r = compute_retained_rank(&s_full, &policy);
    if let Some(max_rank) = options.max_rank {
        r = r.min(max_rank);
    }
    r = r.max(1);
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
/// This function allows per-call control of the truncation policy via `SvdOptions`.
/// If `options.policy` is `None`, it uses the global default policy.
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
/// use tensor4all_core::SvdTruncationPolicy;
///
/// // Truncate with a relative per-value threshold => rank 1
/// let opts = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-10));
/// let (u, s, _v) = svd_with::<f64>(&tensor, &[i.clone()], &opts).unwrap();
/// assert_eq!(s.dims()[0], 1);  // rank-1
///
/// // Truncate with max_rank => capped
/// let opts = SvdOptions::new().with_max_rank(2);
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
