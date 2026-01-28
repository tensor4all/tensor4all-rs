//! SVD decomposition for tensors.
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.

use crate::defaults::DynIndex;
use crate::global_default::GlobalDefault;
use crate::index_like::IndexLike;
use crate::truncation::{HasTruncationParams, TruncationParams};
use crate::{unfold_split, Storage, StorageScalar, TensorDynLen};
use num_complex::{Complex64, ComplexFloat};
use std::sync::Arc;
use tensor4all_tensorbackend::faer_traits::ComplexField;
use tensor4all_tensorbackend::mdarray::DSlice;
use tensor4all_tensorbackend::{svd_backend, SvdResult};
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
///
/// # Arguments
/// * `s_vec` - Singular values in descending order (non-negative)
/// * `rtol` - Relative Frobenius error tolerance
///
/// # Returns
/// The retained rank `r` (at least 1, at most s_vec.len())
fn compute_retained_rank(s_vec: &[f64], rtol: f64) -> usize {
    if s_vec.is_empty() {
        return 1;
    }

    // Compute total squared norm: sum_i σ_i²
    let total_sq_norm: f64 = s_vec.iter().map(|&s| s * s).sum();

    // Edge case: if total norm is zero, keep rank 1
    if total_sq_norm == 0.0 {
        return 1;
    }

    // Compute cumulative discarded weight from the end
    // We want: sum_{i>r} σ_i² / sum_i σ_i² <= rtol²
    // So: sum_{i>r} σ_i² <= rtol² * sum_i σ_i²
    let rtol_sq = rtol * rtol;
    let threshold = rtol_sq * total_sq_norm;

    // Start from the end and accumulate discarded weight
    let mut discarded_sq_norm = 0.0;
    let mut r = s_vec.len();

    // Iterate backwards, adding singular values until threshold is exceeded
    for i in (0..s_vec.len()).rev() {
        let s_sq = s_vec[i] * s_vec[i];
        if discarded_sq_norm + s_sq <= threshold {
            discarded_sq_norm += s_sq;
            r = i;
        } else {
            break;
        }
    }

    // Ensure at least rank 1 is kept
    r.max(1)
}

/// Extract U, S, V^H from tensorbackend's SvdResult.
///
/// This helper function converts the backend's SVD result to our desired format:
/// - Extracts singular values from the diagonal view (first row)
/// - Converts U from m×m to m×k (takes first k columns)
/// - Extracts V^H as k×n (takes first k rows of the backend's vt)
///
/// # Arguments
/// * `decomp` - SVD decomposition from tensorbackend
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `k` - Bond dimension (min(m, n))
///
/// # Returns
/// A tuple `(u_vec, s_vec, vh_vec)` where:
/// - `u_vec` is a vector of length `m * k` containing U matrix data (row-major)
/// - `s_vec` is a vector of length `k` containing singular values (real, f64)
/// - `vh_vec` is a vector of length `k * n` containing V^H matrix data (row-major)
fn extract_usvh_from_svd_result<T>(
    decomp: SvdResult<T>,
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<T>, Vec<f64>, Vec<T>)
where
    T: ComplexFloat + Default + From<<T as ComplexFloat>::Real>,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    let SvdResult { s, u, vt } = decomp;

    // Extract singular values and convert to real type.
    //
    // NOTE:
    // `mdarray-linalg-faer` writes singular values into a diagonal view created by
    // `into_faer_diag_mut`, which (by design) treats the **first row** as the
    // singular-value buffer (LAPACK-style convention). Therefore, the values live at
    // `s[0, i]`, not necessarily at `s[i, i]`.
    //
    // Singular values are always real (f64), even for complex matrices.
    let mut s_vec: Vec<f64> = Vec::with_capacity(k);
    for i in 0..k {
        let s_val = s[[0, i]];
        // <T as ComplexFloat>::Real is f64 for both f64 and Complex64
        let real_val: <T as ComplexFloat>::Real = s_val.re();
        // Convert to f64 using Into trait
        s_vec.push(real_val.into());
    }

    // Convert U from m×m to m×k (take first k columns)
    let mut u_vec = Vec::with_capacity(m * k);
    for i in 0..m {
        for j in 0..k {
            u_vec.push(u[[i, j]]);
        }
    }

    // Extract V^H as k×n (first k rows of the backend's vt).
    //
    // `mdarray-linalg` returns `vt` as V^T for real types or V^H for complex types.
    // We keep V^H directly (no conjugate-transpose) so that:
    // - `svd_with` can derive V from V^H
    // - `factorize_svd` can use V^H directly for correct reconstruction A = U * S * V^H
    let mut vh_vec = Vec::with_capacity(k * n);
    for i in 0..k {
        for j in 0..n {
            vh_vec.push(vt[[i, j]]);
        }
    }

    (u_vec, s_vec, vh_vec)
}

/// Derive V (n×k) from V^H (k×n) via conjugate-transpose.
fn vh_to_v<T>(vh_vec: &[T], n: usize, k: usize) -> Vec<T>
where
    T: ComplexFloat,
{
    let mut v_vec = Vec::with_capacity(n * k);
    for j in 0..n {
        for i in 0..k {
            // V[j][i] = conj(V^H[i][j])
            v_vec.push(vh_vec[i * n + j].conj());
        }
    }
    v_vec
}

type SvdTruncatedUsvhResult<T> = (
    Vec<T>,
    Vec<f64>,
    Vec<T>,
    DynIndex,
    Vec<DynIndex>,
    Vec<DynIndex>,
    usize,
);

/// Internal helper: compute truncated U, singular values, and V^H in matrix form.
///
/// Returns:
/// - `u_vec`: m×r row-major
/// - `singular_values`: length r
/// - `vh_vec`: r×n row-major (V^H)
/// - `bond_index`: dimension r
/// - `left_indices`, `right_indices`: tensor indices corresponding to unfolding split
fn svd_truncated_usvh<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<SvdTruncatedUsvhResult<T>, SvdError>
where
    T: StorageScalar + ComplexFloat + ComplexField + Default + From<<T as ComplexFloat>::Real>,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Determine rtol to use
    let rtol = options.truncation.effective_rtol(default_svd_rtol());
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(SvdError::InvalidRtol(rtol));
    }

    // Unfold tensor into matrix (returns DTensor<T, 2>)
    let (mut a_tensor, _, m, n, left_indices, right_indices) = unfold_split::<T>(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
        .map_err(SvdError::ComputationError)?;
    let k = m.min(n);

    // Call SVD using selected backend
    let a_slice: &mut DSlice<T, 2> = a_tensor.as_mut();
    let decomp = svd_backend(a_slice).map_err(SvdError::ComputationError)?;

    // Extract U, S, V^H from the decomposition (full rank k)
    let (u_vec_full, s_vec_full, vh_vec_full) = extract_usvh_from_svd_result(decomp, m, n, k);

    // Compute retained rank based on rtol truncation
    let mut r = compute_retained_rank(&s_vec_full, rtol);
    if let Some(max_rank) = options.truncation.max_rank {
        r = r.min(max_rank);
    }

    let singular_values: Vec<f64> = s_vec_full[..r].to_vec();

    // Truncate U: m×r
    let mut u_vec = Vec::with_capacity(m * r);
    for i in 0..m {
        for j in 0..r {
            u_vec.push(u_vec_full[i * k + j]);
        }
    }

    // Truncate V^H: r×n
    let mut vh_vec = Vec::with_capacity(r * n);
    for i in 0..r {
        for j in 0..n {
            vh_vec.push(vh_vec_full[i * n + j]);
        }
    }

    // Create bond index with "Link" tag (dimension r, not k)
    let bond_index = DynIndex::new_bond(r)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))
        .map_err(SvdError::ComputationError)?;

    Ok((
        u_vec,
        singular_values,
        vh_vec,
        bond_index,
        left_indices,
        right_indices,
        n,
    ))
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
///
/// This function uses the global default rtol for truncation.
/// See `svd_with` for per-call rtol control.
///
/// This function mimics ITensor's SVD API, returning U, S, and V (not Vt).
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
///
/// Truncation is performed based on the relative Frobenius error tolerance (rtol):
/// The truncation guarantees: ||A - A_approx||_F / ||A||_F <= rtol.
///
/// For complex-valued matrices, the mathematical convention is:
/// \[ A = U * Σ * V^H \]
/// where \(V^H\) is the conjugate-transpose of \(V\).
///
/// `mdarray-linalg` returns `vt` (conceptually \(V^T\) / \(V^H\) depending on scalar type),
/// and we return **V** (not Vt), so we build V by (conjugate-)transposing the leading k rows.
///
/// # Arguments
/// * `t` - Input tensor with DenseF64 or DenseC64 storage
/// * `left_inds` - Indices to place on the left (row) side of the unfolded matrix
///
/// # Returns
/// A tuple `(U, S, V)` where:
/// - `U` is a tensor with indices `[left_inds..., bond_index]` and dimensions `[left_dims..., r]`
/// - `S` is a r×r diagonal tensor with indices `[bond_index, bond_index]` (singular values are real)
/// - `V` is a tensor with indices `[right_inds..., bond_index]` and dimensions `[right_dims..., r]`
///   where `r` is the retained rank (≤ min(m, n)) determined by rtol truncation.
///
/// Note: Singular values `S` are always real, even for complex input tensors.
///
/// # Errors
/// Returns `SvdError` if:
/// - The tensor rank is < 2
/// - Storage is not DenseF64 or DenseC64
/// - `left_inds` is empty or contains all indices
/// - `left_inds` contains indices not in the tensor or duplicates
/// - The SVD computation fails
#[allow(private_bounds)]
pub fn svd<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
) -> Result<(TensorDynLen, TensorDynLen, TensorDynLen), SvdError>
where
    T: StorageScalar + ComplexFloat + ComplexField + Default + From<<T as ComplexFloat>::Real>,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    svd_with::<T>(t, left_inds, &SvdOptions::default())
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
///
/// This function allows per-call control of the truncation tolerance via `SvdOptions`.
/// If `options.rtol` is `None`, uses the global default rtol.
///
/// This function mimics ITensor's SVD API, returning U, S, and V (not Vt).
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
///
/// Truncation is performed based on the relative Frobenius error tolerance (rtol):
/// The truncation guarantees: ||A - A_approx||_F / ||A||_F <= rtol.
///
/// For complex-valued matrices, the mathematical convention is:
/// \[ A = U * Σ * V^H \]
/// where \(V^H\) is the conjugate-transpose of \(V\).
///
/// `mdarray-linalg` returns `vt` (conceptually \(V^T\) / \(V^H\) depending on scalar type),
/// and we return **V** (not Vt), so we build V by (conjugate-)transposing the leading k rows.
///
/// # Arguments
/// * `t` - Input tensor with DenseF64 or DenseC64 storage
/// * `left_inds` - Indices to place on the left (row) side of the unfolded matrix
/// * `options` - SVD options including rtol for truncation control
///
/// # Returns
/// A tuple `(U, S, V)` where:
/// - `U` is a tensor with indices `[left_inds..., bond_index]` and dimensions `[left_dims..., r]`
/// - `S` is a r×r diagonal tensor with indices `[bond_index, bond_index]` (singular values are real)
/// - `V` is a tensor with indices `[right_inds..., bond_index]` and dimensions `[right_dims..., r]`
///   where `r` is the retained rank (≤ min(m, n)) determined by rtol truncation.
///
/// Note: Singular values `S` are always real, even for complex input tensors.
///
/// # Errors
/// Returns `SvdError` if:
/// - The tensor rank is < 2
/// - Storage is not DenseF64 or DenseC64
/// - `left_inds` is empty or contains all indices
/// - `left_inds` contains indices not in the tensor or duplicates
/// - The SVD computation fails
/// - `options.rtol` is invalid (not finite or negative)
#[allow(private_bounds)]
pub fn svd_with<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<(TensorDynLen, TensorDynLen, TensorDynLen), SvdError>
where
    T: StorageScalar + ComplexFloat + ComplexField + Default + From<<T as ComplexFloat>::Real>,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    let (u_vec, s_vec, vh_vec, bond_index, left_indices, right_indices, n) =
        svd_truncated_usvh::<T>(t, left_inds, options)?;
    let r = s_vec.len();
    let v_vec = vh_to_v(&vh_vec, n, r);

    // Create U tensor: [left_inds..., bond_index]
    let mut u_indices = left_indices;
    u_indices.push(bond_index.clone());
    let u_dims: Vec<usize> = u_indices.iter().map(|idx| idx.dim).collect();
    let u_storage = T::dense_storage_with_shape(u_vec, &u_dims);
    let u = TensorDynLen::from_indices(u_indices, u_storage);

    // Create S tensor: [bond_index, bond_index.sim()] (diagonal)
    // Singular values are always real (f64), even for complex input
    // Use sim() to create a similar index with a new ID to avoid duplicate index IDs
    let s_indices = vec![bond_index.clone(), bond_index.sim()];
    let s_storage = Arc::new(Storage::new_diag_f64(s_vec));
    let s = TensorDynLen::from_indices(s_indices, s_storage);

    // Create V tensor: [right_inds..., bond_index]
    let mut v_indices = right_indices.clone();
    v_indices.push(bond_index.clone());
    let v_dims: Vec<usize> = v_indices.iter().map(|idx| idx.dim).collect();
    let v_storage = T::dense_storage_with_shape(v_vec, &v_dims);
    let v = TensorDynLen::from_indices(v_indices, v_storage);

    Ok((u, s, v))
}

/// Compute SVD decomposition of a complex tensor with arbitrary rank, returning (U, S, V).
///
/// This is a convenience wrapper around the generic `svd` function for `Complex64` tensors.
///
/// For complex-valued matrices, the mathematical convention is:
/// \[ A = U * Σ * V^H \]
/// where \(V^H\) is the conjugate-transpose of \(V\).
///
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
///
/// `mdarray-linalg` returns `vt` (conceptually \(V^T\) / \(V^H\) depending on scalar type),
/// and we return **V** (not Vt), so we build V by conjugate-transposing the leading k rows.
///
/// Note: Singular values `S` are always real (f64), even for complex input tensors.
#[inline]
pub fn svd_c64(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
) -> Result<(TensorDynLen, TensorDynLen, TensorDynLen), SvdError> {
    svd::<Complex64>(t, left_inds)
}

/// SVD result for factorization: returns (U, singular_values, V^H) as tensors.
///
/// Unlike `svd_with`, this returns V^H (not V) to avoid the need for tensor-level
/// conjugation in factorize. V^H has indices `[bond_index, right_inds...]`.
///
/// This is `pub(crate)` because it is only used by the factorization module.
#[allow(private_bounds)]
pub(crate) fn svd_for_factorize<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<SvdFactorizeResult, SvdError>
where
    T: StorageScalar + ComplexFloat + ComplexField + Default + From<<T as ComplexFloat>::Real>,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    let (u_vec, singular_values, vh_vec, bond_index, left_indices, right_indices, _n) =
        svd_truncated_usvh::<T>(t, left_inds, options)?;
    let r = singular_values.len();

    // U tensor: [left_inds..., bond_index]
    let mut u_indices = left_indices;
    u_indices.push(bond_index.clone());
    let u_dims: Vec<usize> = u_indices.iter().map(|idx| idx.dim).collect();
    let u_storage = T::dense_storage_with_shape(u_vec, &u_dims);
    let u = TensorDynLen::from_indices(u_indices, u_storage);

    // V^H tensor: [bond_index, right_inds...]
    let mut vh_indices = vec![bond_index.clone()];
    vh_indices.extend(right_indices);
    let vh_dims: Vec<usize> = vh_indices.iter().map(|idx| idx.dim).collect();
    let vh_storage = T::dense_storage_with_shape(vh_vec, &vh_dims);
    let vh = TensorDynLen::from_indices(vh_indices, vh_storage);

    Ok(SvdFactorizeResult {
        u,
        vh,
        bond_index,
        singular_values,
        rank: r,
    })
}

/// Result of SVD for factorization (returns V^H instead of V).
pub(crate) struct SvdFactorizeResult {
    pub u: TensorDynLen,
    pub vh: TensorDynLen,
    pub bond_index: DynIndex,
    pub singular_values: Vec<f64>,
    pub rank: usize,
}
