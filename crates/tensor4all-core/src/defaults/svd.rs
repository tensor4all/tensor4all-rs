//! SVD decomposition for tensors.
//!
//! Provides [`svd`] and [`svd_with`] for computing truncated SVD of
//! [`IdxTensor`] values. The tensor is unfolded into a matrix by
//! splitting its indices into left and right groups, then the standard
//! matrix SVD is computed and truncated according to [`SvdOptions`].
//!
//! This module works with concrete types (`DynIndex`, `IdxTensor`) only.

use super::eager_session::with_eager_session;
use crate::defaults::idx_tensor::{unfold_index_sets, unfold_split_inner, unfold_split_inner_in};
use crate::defaults::DynIndex;
use crate::index_like::IndexLike;
use crate::truncation::{
    validate_svd_truncation_options, validate_svd_truncation_policy, SingularValueMeasure,
    SvdTruncationOptionsError, SvdTruncationPolicy, ThresholdScale, TruncationRule,
};
use crate::{ExecutionContext, IdxTensor, TensorElement};
use num_complex::{Complex32, Complex64, ComplexFloat};
use std::sync::Mutex;
use tenferro::{DType, TensorScalar};
use tenferro_ad::EagerTensor;
use tenferro_linalg::EagerSessionLinalgExt;
#[cfg(test)]
use tensor4all_tensorbackend::native_tensor_primal_to_dense_col_major;
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
    /// `max_bond_dim` is set to zero.
    #[error("max_bond_dim must be at least 1 when set")]
    InvalidMaxBondDim,
    /// The operand storage cannot enter the runtime-free explicit-session route.
    ///
    /// The session routes publish plain host factors, so an AD-tracked operand would lose its graph;
    /// they refuse it instead.
    #[error("Unsupported storage type: {0}")]
    UnsupportedStorage(
        /// Description of the unsupported operand storage
        &'static str,
    ),
}

/// Options for SVD decomposition with truncation control.
///
/// # Examples
///
/// ```
/// use tensor4all_core::svd::{SvdOptions, svd_with};
/// use tensor4all_core::{DynIndex, SvdTruncationPolicy, IdxTensor};
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(3);
/// let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
/// let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// let opts = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-10));
/// let (u, s, v) = svd_with::<f64>(&tensor, &[i.clone()], &opts).unwrap();
///
/// // U has left index + bond, S is diagonal bond x bond, V has right index + bond
/// assert_eq!(u.dims()[0], 3);
/// assert_eq!(s.dims().len(), 2);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SvdOptions {
    /// Maximum retained rank after policy-based truncation.
    pub max_bond_dim: Option<usize>,
    /// Per-call SVD truncation policy.
    /// If `None`, the global default policy is used.
    pub policy: Option<SvdTruncationPolicy>,
    truncate: bool,
}

impl Default for SvdOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl SvdOptions {
    /// Create new SVD options with no overrides.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_bond_dim: None,
            policy: None,
            truncate: true,
        }
    }

    /// Set the maximum retained rank.
    #[must_use]
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    /// Set the SVD truncation policy override.
    #[must_use]
    pub fn with_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    pub(crate) fn full_rank() -> Self {
        Self {
            max_bond_dim: None,
            policy: None,
            truncate: false,
        }
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
pub(crate) fn compute_retained_rank(s_vec: &[f64], policy: &SvdTruncationPolicy) -> usize {
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

#[cfg(test)]
fn singular_values_from_native(tensor: &tenferro::Tensor) -> Result<Vec<f64>, SvdError> {
    match tensor.dtype() {
        DType::F32 => native_tensor_primal_to_dense_col_major::<f32>(tensor)
            .map(|values| values.into_iter().map(f64::from).collect())
            .map_err(|e| SvdError::ComputationError(e.source)),
        DType::F64 => native_tensor_primal_to_dense_col_major::<f64>(tensor)
            .map_err(|e| SvdError::ComputationError(e.source)),
        DType::C32 => native_tensor_primal_to_dense_col_major::<Complex32>(tensor)
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| f64::from(value.re))
                    .collect()
            })
            .map_err(|e| SvdError::ComputationError(e.source)),
        DType::C64 => native_tensor_primal_to_dense_col_major::<Complex64>(tensor)
            .map(|values| values.into_iter().map(|value| value.re).collect())
            .map_err(|e| SvdError::ComputationError(e.source)),
        other => Err(SvdError::ComputationError(anyhow::anyhow!(
            "native SVD returned unsupported singular-value scalar type {other:?}"
        ))),
    }
}

fn singular_values_from_eager(tensor: &EagerTensor) -> Result<Vec<f64>, SvdError> {
    let value = tensor
        .value()
        .map_err(|source| SvdError::ComputationError(anyhow::Error::new(source)))?;
    match tensor.dtype() {
        DType::F32 => value
            .as_slice::<f32>()
            .map(|values| values.iter().map(|&value| f64::from(value)).collect())
            .map_err(|source| SvdError::ComputationError(anyhow::Error::new(source))),
        DType::F64 => value
            .as_slice::<f64>()
            .map(|values| values.to_vec())
            .map_err(|source| SvdError::ComputationError(anyhow::Error::new(source))),
        DType::C32 => value
            .as_slice::<Complex32>()
            .map(|values| values.iter().map(|value| f64::from(value.re)).collect())
            .map_err(|source| SvdError::ComputationError(anyhow::Error::new(source))),
        DType::C64 => value
            .as_slice::<Complex64>()
            .map(|values| values.iter().map(|value| value.re).collect())
            .map_err(|source| SvdError::ComputationError(anyhow::Error::new(source))),
        other => Err(SvdError::ComputationError(anyhow::anyhow!(
            "eager SVD returned unsupported singular-value scalar type {other:?}"
        ))),
    }
}

/// Read the singular-value decision vector of a resident factor.
///
/// The factor stays resident; only its values cross the explicit readback
/// boundary through `context`.
#[cfg(feature = "tenferro-cuda")]
fn resident_singular_values(
    s_inner: &EagerTensor,
    context: &ExecutionContext,
) -> Result<Vec<f64>, SvdError> {
    let indices: Vec<DynIndex> = s_inner
        .shape()
        .iter()
        .map(|&dim| DynIndex::new_dyn(dim))
        .collect();
    let decision =
        IdxTensor::from_inner(indices, s_inner.clone()).map_err(SvdError::ComputationError)?;
    decision
        .read_decision_data(context)
        .map_err(|error| SvdError::ComputationError(anyhow::Error::new(error)))
}

type SvdTruncatedEagerResult = (
    EagerTensor,
    EagerTensor,
    EagerTensor,
    Vec<f64>,
    DynIndex,
    Vec<DynIndex>,
    Vec<DynIndex>,
);

fn svd_truncated_inner(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<SvdTruncatedEagerResult, SvdError> {
    if t.is_cuda_resident() {
        return Err(SvdError::ComputationError(anyhow::anyhow!(
            "CUDA-resident input requires svd_with_in with the owning execution context"
        )));
    }
    svd_truncated_inner_scoped(t, left_inds, options, None)
}

/// Compute truncated SVD in a caller-owned execution context.
///
/// The factorization and truncation slicing execute in `context`. With a CUDA
/// context only the singular-value decision vector crosses the explicit
/// readback boundary (via [`IdxTensor::read_decision_data`); with a CPU
/// context the decision uses the same host read as [`svd_with`].
///
/// # Errors
///
/// Returns `SvdError` when the tensor does not belong to `context`, when the
/// indices or options are invalid, or when the factorization or explicit
/// decision readback fails.
pub(crate) fn svd_truncated_inner_in(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &SvdOptions,
    context: &ExecutionContext,
) -> Result<SvdTruncatedEagerResult, SvdError> {
    t.validate_context(context)
        .map_err(|error| SvdError::ComputationError(anyhow::Error::new(error)))?;
    svd_truncated_inner_scoped(t, left_inds, options, Some(context))
}

fn svd_truncated_inner_scoped(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &SvdOptions,
    context: Option<&ExecutionContext>,
) -> Result<SvdTruncatedEagerResult, SvdError> {
    let (matrix_inner, _, m, n, left_indices, right_indices) = match context {
        Some(context) => unfold_split_inner_in(t, left_inds, context),
        None => unfold_split_inner(t, left_inds),
    }
    .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
    .map_err(SvdError::ComputationError)?;
    let k = m.min(n);

    let (mut u_inner, mut s_inner, mut vt_inner) =
        with_eager_session(matrix_inner.runtime(), context, |session| {
            session.svd(&matrix_inner)
        })
        .map_err(|e| SvdError::ComputationError(anyhow::anyhow!("{e}")))?;
    #[cfg(feature = "tenferro-cuda")]
    let s_full = if let Some(context) = context {
        if matches!(context, ExecutionContext::Cuda(_)) {
            resident_singular_values(&s_inner, context)?
        } else {
            singular_values_from_eager(&s_inner)?
        }
    } else {
        singular_values_from_eager(&s_inner)?
    };
    #[cfg(not(feature = "tenferro-cuda"))]
    let s_full = {
        let _ = context;
        singular_values_from_eager(&s_inner)?
    };
    let mut r = retained_svd_rank(&s_full, k, options)?;
    r = r.min(s_full.len());
    if r < k {
        let keep: Vec<usize> = (0..r).collect();
        u_inner = with_eager_session(u_inner.runtime(), context, |session| {
            session.take_axis(&u_inner, 1, &keep)
        })
        .map_err(|e| SvdError::ComputationError(anyhow::anyhow!("{e}")))?;
        s_inner = with_eager_session(s_inner.runtime(), context, |session| {
            session.take_axis(&s_inner, 0, &keep)
        })
        .map_err(|e| SvdError::ComputationError(anyhow::anyhow!("{e}")))?;
        vt_inner = with_eager_session(vt_inner.runtime(), context, |session| {
            session.take_axis(&vt_inner, 0, &keep)
        })
        .map_err(|e| SvdError::ComputationError(anyhow::anyhow!("{e}")))?;
    }

    let bond_index = DynIndex::new_bond(r)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))
        .map_err(SvdError::ComputationError)?;
    let singular_values = s_full[..r].to_vec();

    Ok((
        u_inner,
        s_inner,
        vt_inner,
        singular_values,
        bond_index,
        left_indices,
        right_indices,
    ))
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
///
/// # Errors
///
/// Returns [`SvdError::ComputationError`] when `left_inds` is not a non-empty
/// proper subset of the tensor's indices, when the input is CUDA-resident (use
/// [`svd_with_in`]), or when the backend SVD fails.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{IdxTensor, DynIndex, svd};
///
/// // Create a 2x3 matrix (rank-1 outer product: all-ones)
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let data = vec![1.0_f64; 6]; // all-ones 2x3 matrix
/// let t = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
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
    t: &IdxTensor,
    left_inds: &[DynIndex],
) -> Result<(IdxTensor, IdxTensor, IdxTensor), SvdError> {
    svd_with::<T>(t, left_inds, &SvdOptions::default())
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
///
/// This function allows per-call control of the truncation policy via `SvdOptions`.
/// If `options.policy` is `None`, it uses the global default policy.
///
/// # Errors
///
/// Returns [`SvdError::InvalidMaxBondDim`] when `options.max_bond_dim` is
/// `Some(0)`, [`SvdError::InvalidThreshold`] when the truncation threshold is
/// negative or non-finite, and [`SvdError::ComputationError`] when `left_inds`
/// is not a non-empty proper subset of the tensor's indices, when the input is
/// CUDA-resident, or when the backend SVD fails.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_core::svd::{SvdOptions, svd_with};
///
/// let i = DynIndex::new_dyn(4);
/// let j = DynIndex::new_dyn(4);
/// // Rank-1 matrix
/// let mut data = vec![0.0_f64; 16];
/// data[0] = 1.0;
/// let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// use tensor4all_core::SvdTruncationPolicy;
///
/// // Truncate with a relative per-value threshold => rank 1
/// let opts = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-10));
/// let (u, s, _v) = svd_with::<f64>(&tensor, &[i.clone()], &opts).unwrap();
/// assert_eq!(s.dims()[0], 1);  // rank-1
///
/// // Truncate with max_bond_dim => capped
/// let opts = SvdOptions::new().with_max_bond_dim(2);
/// let (_u, s, _v) = svd_with::<f64>(&tensor, &[i.clone()], &opts).unwrap();
/// assert!(s.dims()[0] <= 2);
/// ```
pub fn svd_with<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<(IdxTensor, IdxTensor, IdxTensor), SvdError> {
    let (u_inner, s_inner, vt_inner, _singular_values, bond_index, left_indices, right_indices) =
        svd_truncated_inner(t, left_inds, options)?;

    svd_assemble(
        u_inner,
        s_inner,
        vt_inner,
        bond_index,
        left_indices,
        right_indices,
        None,
    )
}

/// Compute SVD decomposition in a caller-owned execution context.
///
/// The factorization, truncation slicing, and result assembly all execute in
/// `context`. With a CUDA context only the singular-value decision vector
/// crosses the explicit readback boundary; with a CPU context the decision
/// uses the same host read as [`svd_with`].
///
/// # Arguments
///
/// * `t` - Input tensor, which must belong to `context`.
/// * `left_inds` - Indices to place on the left (row) side of the unfolded matrix.
/// * `options` - SVD options including truncation policy and bond cap.
/// * `context` - Caller-owned execution context owning the input and results.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use tensor4all_core::svd::{SvdOptions, svd_with_in};
/// use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor, SvdTruncationPolicy};
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tensor4all_tensorbackend::CpuBackend;
///
/// let context = ExecutionContext::Cpu(Arc::new(
///     CpuExecutionContext::from_backend(CpuBackend::new()),
/// ));
/// let i = DynIndex::new_dyn(4);
/// let j = DynIndex::new_dyn(4);
/// let mut data = vec![0.0_f64; 16];
/// data[0] = 1.0;
/// let tensor = IdxTensor::from_dense_in(&context, vec![i.clone(), j.clone()], data)?;
/// let opts = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-10));
/// let (u, s, _v) = svd_with_in::<f64>(&tensor, &[i.clone()], &opts, &context)?;
/// assert_eq!(s.dims()[0], 1);
/// assert_eq!(u.dims()[0], 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns `SvdError` when the tensor does not belong to `context`, when the
/// indices or options are invalid, or when the factorization or explicit
/// decision readback fails. When this build cannot execute in `context`, the
/// rejection carries
/// [`IdxTensorError::UnsupportedExecutionContext`](crate::IdxTensorError::UnsupportedExecutionContext)
/// as its source.
pub fn svd_with_in<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &SvdOptions,
    context: &ExecutionContext,
) -> Result<(IdxTensor, IdxTensor, IdxTensor), SvdError> {
    let (u_inner, s_inner, vt_inner, _singular_values, bond_index, left_indices, right_indices) =
        svd_truncated_inner_in(t, left_inds, options, context)?;

    svd_assemble(
        u_inner,
        s_inner,
        vt_inner,
        bond_index,
        left_indices,
        right_indices,
        Some(context),
    )
}

/// Wrap the truncated eager factors as [`IdxTensor`]s.
fn svd_assemble(
    u_inner: EagerTensor,
    s_inner: EagerTensor,
    vt_inner: EagerTensor,
    bond_index: DynIndex,
    left_indices: Vec<DynIndex>,
    right_indices: Vec<DynIndex>,
    context: Option<&ExecutionContext>,
) -> Result<(IdxTensor, IdxTensor, IdxTensor), SvdError> {
    let mut u_indices = left_indices;
    u_indices.push(bond_index.clone());
    let u_dims: Vec<usize> = u_indices.iter().map(|idx| idx.dim).collect();
    let u_reshaped = with_eager_session(u_inner.runtime(), context, |session| {
        session.reshape(&u_inner, &u_dims)
    })
    .map_err(|e| SvdError::ComputationError(anyhow::anyhow!("eager SVD U reshape failed: {e}")))?;
    let u = IdxTensor::from_inner(u_indices, u_reshaped).map_err(SvdError::ComputationError)?;

    // S carries a fresh `sim` leg (ITensors convention S: [l, l'], V: l');
    // the SAME sim instance is shared with V^H so the returned triple
    // reconstructs under plain contraction:
    // U [left..., bond] · S [bond, sim] · V^H [sim, right...].
    let sim_bond_index = bond_index.sim();
    let s_indices = vec![bond_index.clone(), sim_bond_index.clone()];
    let s = IdxTensor::from_diag_inner(s_indices, s_inner).map_err(SvdError::ComputationError)?;

    let mut vh_indices = vec![sim_bond_index];
    vh_indices.extend(right_indices);
    let vh_dims: Vec<usize> = vh_indices.iter().map(|idx| idx.dim).collect();
    let vt_reshaped = with_eager_session(vt_inner.runtime(), context, |session| {
        session.reshape(&vt_inner, &vh_dims)
    })
    .map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("eager SVD V^T reshape failed: {e}"))
    })?;
    let vh = IdxTensor::from_inner(vh_indices, vt_reshaped).map_err(SvdError::ComputationError)?;
    let perm: Vec<usize> = (1..vh.indices.len()).chain(std::iter::once(0)).collect();
    let v = vh
        .conj()
        .permute(&perm)
        .map_err(|e| SvdError::ComputationError(anyhow::Error::new(e)))?;

    Ok((u, s, v))
}

/// SVD result for factorization, returning `V^H` directly.
pub(crate) struct SvdFactorizeResult {
    pub u: IdxTensor,
    pub s: IdxTensor,
    pub vh: IdxTensor,
    pub bond_index: DynIndex,
    pub singular_values: Vec<f64>,
    pub rank: usize,
}

/// The retained rank of one SVD, from its singular values.
///
/// This is the one place the SVD truncation policy is applied: the eager, context-scoped and
/// explicit-session routes all funnel their singular values through it, so a policy change cannot
/// diverge between the frontends.
fn retained_svd_rank(
    singular_values: &[f64],
    k: usize,
    options: &SvdOptions,
) -> Result<usize, SvdError> {
    if !options.truncate {
        return Ok(k.max(1));
    }
    // Shared root guard: reject `max_bond_dim == Some(0)` and invalid
    // explicit policies before any truncation, so the caller cannot
    // silently clamp every factorization to rank one.
    validate_svd_truncation_options(options.max_bond_dim, options.policy).map_err(|error| {
        match error {
            SvdTruncationOptionsError::ZeroMaxBondDim => SvdError::InvalidMaxBondDim,
            SvdTruncationOptionsError::InvalidThreshold(threshold) => {
                SvdError::InvalidThreshold(threshold)
            }
        }
    })?;
    let policy = options.policy.unwrap_or_else(default_svd_truncation_policy);
    validate_svd_truncation_policy(policy).map_err(|e| SvdError::InvalidThreshold(e.0))?;
    let mut retained = compute_retained_rank(singular_values, &policy);
    if let Some(max_bond_dim) = options.max_bond_dim {
        retained = retained.min(max_bond_dim);
    }
    Ok(retained.max(1).min(k))
}

/// Truncated SVD for factorization on a caller-held explicit session.
///
/// The runtime-free counterpart of [`svd_for_factorize`]: the index bookkeeping, the truncation
/// policy and the factor layout are shared with the eager route, while the decomposition runs on
/// `session` and the factors are published as plain host-storage [`IdxTensor`]s. No session or eager
/// runtime is entered here.
///
/// # Errors
///
/// Returns [`SvdError::UnsupportedStorage`] for an AD-tracked operand (the plain host factors would
/// drop its graph), [`SvdError::InvalidThreshold`]/[`SvdError::InvalidMaxBondDim`] for invalid
/// truncation options, and [`SvdError::ComputationError`] when the split, the session decomposition or
/// a factor's publication fails.
pub(crate) fn svd_for_factorize_on_session<T>(
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<SvdFactorizeResult, SvdError>
where
    T: TensorElement + ComplexFloat + tensor4all_tensorbackend::StorageScalar,
    <T as ComplexFloat>::Real: Into<f64>,
    <T as TensorScalar>::Real: Into<f64>,
    T: num_traits::NumCast,
{
    if t.tracks_grad() {
        return Err(SvdError::UnsupportedStorage(
            crate::defaults::idx_tensor::TRACKED_SESSION_OPERAND_REFUSAL,
        ));
    }
    let (left_indices, right_indices, order) =
        unfold_index_sets(t, left_inds).map_err(|error| {
            SvdError::ComputationError(anyhow::anyhow!("Failed to unfold tensor: {error}"))
        })?;
    // Read and reorder the operand's own host payload: entering the eager
    // runtime for a transpose would be a second session entry on this thread.
    let (dims, data) = t
        .host_dense_permuted::<T>(&order)
        .map_err(|error| SvdError::ComputationError(anyhow::Error::new(error)))?;
    let left_len = left_indices.len();
    let m = crate::defaults::idx_tensor::checked_product(&dims[..left_len])
        .map_err(SvdError::ComputationError)?;
    let n = crate::defaults::idx_tensor::checked_product(&dims[left_len..])
        .map_err(SvdError::ComputationError)?;
    let k = m.min(n);
    let matrix = tenferro::Tensor::from_vec_col_major(vec![m, n], data)
        .map_err(|error| SvdError::ComputationError(anyhow::Error::new(error)))?;
    let (u_value, s_value, vt_value) = session
        .svd(&matrix)
        .map_err(|error| SvdError::ComputationError(anyhow::anyhow!("{error}")))?;
    // The session returns singular values in the operand's own real dtype, so a
    // narrow operand must not be read as `f64`.
    let s_full: Vec<f64> = s_value
        .as_slice::<<T as TensorScalar>::Real>()
        .map_err(|error| SvdError::ComputationError(anyhow::Error::new(error)))?
        .iter()
        .map(|&value| value.into())
        .collect();
    let rank = retained_svd_rank(&s_full, k, options)?;
    let u_payload = u_value
        .as_slice::<T>()
        .map_err(|error| SvdError::ComputationError(anyhow::Error::new(error)))?;
    let vt_payload = vt_value
        .as_slice::<T>()
        .map_err(|error| SvdError::ComputationError(anyhow::Error::new(error)))?;

    // `U` is column-major `[m, k]`: kept columns are the bond axis of the left factor.
    let mut u_data = Vec::with_capacity(m * rank);
    for column in 0..rank {
        let start = column.checked_mul(m).ok_or_else(|| {
            SvdError::ComputationError(anyhow::anyhow!("SVD U column offset overflows usize"))
        })?;
        let end = start.checked_add(m).ok_or_else(|| {
            SvdError::ComputationError(anyhow::anyhow!("SVD U column end overflows usize"))
        })?;
        u_data.extend_from_slice(u_payload.get(start..end).ok_or_else(|| {
            SvdError::ComputationError(anyhow::anyhow!("SVD U column {column} is out of bounds"))
        })?);
    }
    // `S` is real; it takes the factor dtype so the shared assembly keeps its fast contraction path.
    let mut s_data = Vec::with_capacity(rank);
    for &value in s_full.iter().take(rank) {
        s_data.push(<T as num_traits::NumCast>::from(value).ok_or_else(|| {
            SvdError::ComputationError(anyhow::anyhow!(
                "singular value {value} does not fit the factor scalar type"
            ))
        })?);
    }
    // `V^H` is column-major `[k, n]`: kept rows are the bond axis of the right factor.
    let mut vt_data = Vec::with_capacity(rank * n);
    for column in 0..n {
        let base = column.checked_mul(k).ok_or_else(|| {
            SvdError::ComputationError(anyhow::anyhow!("SVD V^H column offset overflows usize"))
        })?;
        for row in 0..rank {
            let value = vt_payload.get(base + row).ok_or_else(|| {
                SvdError::ComputationError(anyhow::anyhow!("SVD V^H row {row} is out of bounds"))
            })?;
            vt_data.push(*value);
        }
    }

    let bond_index = DynIndex::new_bond(rank).map_err(|error| {
        SvdError::ComputationError(anyhow::anyhow!("Failed to create Link index: {error:?}"))
    })?;
    let mut u_indices = left_indices;
    u_indices.push(bond_index.clone());
    let u = IdxTensor::from_host_dense(u_indices, u_data)
        .map_err(|error| SvdError::ComputationError(anyhow::Error::new(error)))?;
    // The session route publishes S as a dense `[bond, sim]` diagonal matrix: the compact diagonal
    // representation is an eager construction, and the session contraction route serves dense host
    // operands only. The rank-squared payload is negligible next to the factors.
    let mut s_dense = vec![
        <T as num_traits::Zero>::zero();
        rank.checked_mul(rank).ok_or_else(|| {
            SvdError::ComputationError(anyhow::anyhow!("SVD S payload size overflows usize"))
        })?
    ];
    for (position, value) in s_data.iter().enumerate() {
        s_dense[position * (rank + 1)] = *value;
    }
    let s = IdxTensor::from_host_dense(vec![bond_index.clone(), bond_index.sim()], s_dense)
        .map_err(|error| SvdError::ComputationError(anyhow::Error::new(error)))?;
    let mut vh_indices = vec![bond_index.clone()];
    vh_indices.extend(right_indices);
    let vh = IdxTensor::from_host_dense(vh_indices, vt_data)
        .map_err(|error| SvdError::ComputationError(anyhow::Error::new(error)))?;

    Ok(SvdFactorizeResult {
        u,
        s,
        vh,
        bond_index,
        singular_values: s_full[..rank].to_vec(),
        rank,
    })
}

/// Compute truncated SVD for factorization, returning `V^H` instead of `V`.
pub(crate) fn svd_for_factorize(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<SvdFactorizeResult, SvdError> {
    let (u_inner, s_inner, vt_inner, singular_values, bond_index, left_indices, right_indices) =
        svd_truncated_inner(t, left_inds, options)?;
    svd_factorize_assemble(
        (
            u_inner,
            s_inner,
            vt_inner,
            singular_values,
            bond_index,
            left_indices,
            right_indices,
        ),
        None,
    )
}

/// Compute truncated SVD for factorization in a caller-owned context.
///
/// Context-scoped counterpart of [`svd_for_factorize`]: the factorization and
/// truncation slicing execute in `context`, with only the singular-value
/// decision vector crossing the explicit readback boundary on CUDA.
///
/// # Errors
///
/// Returns `SvdError` when the tensor does not belong to `context`, when the
/// indices or options are invalid, or when the factorization or explicit
/// decision readback fails.
pub(crate) fn svd_for_factorize_in(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &SvdOptions,
    context: &ExecutionContext,
) -> Result<SvdFactorizeResult, SvdError> {
    let (u_inner, s_inner, vt_inner, singular_values, bond_index, left_indices, right_indices) =
        svd_truncated_inner_in(t, left_inds, options, context)?;
    svd_factorize_assemble(
        (
            u_inner,
            s_inner,
            vt_inner,
            singular_values,
            bond_index,
            left_indices,
            right_indices,
        ),
        Some(context),
    )
}

/// Wrap the truncated eager factors as [`IdxTensor`]s, returning `V^H` directly.
fn svd_factorize_assemble(
    (
        u_inner,
        s_inner,
        vt_inner,
        singular_values,
        bond_index,
        left_indices,
        right_indices,
    ): SvdTruncatedEagerResult,
    context: Option<&ExecutionContext>,
) -> Result<SvdFactorizeResult, SvdError> {
    let rank = singular_values.len();
    let mut u_indices = left_indices;
    u_indices.push(bond_index.clone());
    let u_dims: Vec<usize> = u_indices.iter().map(|idx| idx.dim).collect();
    let u_reshaped = with_eager_session(u_inner.runtime(), context, |session| {
        session.reshape(&u_inner, &u_dims)
    })
    .map_err(|e| SvdError::ComputationError(anyhow::anyhow!("eager SVD U reshape failed: {e}")))?;
    let u = IdxTensor::from_inner(u_indices, u_reshaped).map_err(SvdError::ComputationError)?;

    // The singular values are real. Contracting a real `S` with a complex
    // factor in `assemble_svd_factors` takes the mixed-dtype einsum path,
    // which traces and compiles a program on every call; giving `S` the
    // factors' dtype keeps that contraction on the cached path.
    let s_inner = if s_inner.dtype() != vt_inner.dtype() {
        with_eager_session(s_inner.runtime(), context, |session| {
            session.convert(&s_inner, vt_inner.dtype())
        })
        .map_err(|e| {
            SvdError::ComputationError(anyhow::anyhow!("eager SVD S dtype conversion failed: {e}"))
        })?
    } else {
        s_inner
    };
    let s_indices = vec![bond_index.clone(), bond_index.sim()];
    // The context-scoped factorization assembles its factors on the context's
    // session, which serves dense operands. Build `S` as a dense diagonal on
    // that engine; the implicit route keeps the compact diagonal storage.
    let s = match context {
        Some(context) => {
            let dense = with_eager_session(s_inner.runtime(), Some(context), |session| {
                session.embed_diag(&s_inner, 0, 1)
            })
            .map_err(|error| {
                SvdError::ComputationError(
                    anyhow::Error::new(error).context("eager SVD S diagonal embedding failed"),
                )
            })?;
            IdxTensor::from_inner(s_indices, dense).map_err(SvdError::ComputationError)?
        }
        None => {
            IdxTensor::from_diag_inner(s_indices, s_inner).map_err(SvdError::ComputationError)?
        }
    };

    let mut vh_indices = vec![bond_index.clone()];
    vh_indices.extend(right_indices);
    let vh_dims: Vec<usize> = vh_indices.iter().map(|idx| idx.dim).collect();
    let vt_reshaped = with_eager_session(vt_inner.runtime(), context, |session| {
        session.reshape(&vt_inner, &vh_dims)
    })
    .map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("eager SVD V^T reshape failed: {e}"))
    })?;
    let vh = IdxTensor::from_inner(vh_indices, vt_reshaped).map_err(SvdError::ComputationError)?;

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
