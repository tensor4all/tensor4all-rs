//! QR decomposition for tensors.
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.

use crate::backend::qr_backend;
use crate::defaults::DynIndex;
use crate::global_default::GlobalDefault;
use crate::truncation::TruncationParams;
use crate::{unfold_split, StorageScalar, TensorDynLen};
use num_complex::{Complex64, ComplexFloat};
use std::any::TypeId;
use tensor4all_tensorbackend::mdarray::{DSlice, DTensor};
use tensor4all_tensorbackend::{
    dyn_ad_tensor_primal_to_storage, qr_dyn_ad_tensor_native, reshape_row_major_dyn_ad_tensor,
    Storage,
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

/// Compute the retained rank based on rtol truncation for QR.
///
/// For non-pivoting QR, the diagonal elements of R are NOT necessarily in
/// decreasing order, and a zero diagonal element does NOT mean the row is
/// negligible (off-diagonal elements in that row may be significant).
///
/// We use row norms of R: a row is negligible when its norm is below
/// `rtol * max_row_norm`. This is more robust than checking only diagonals.
///
/// # Arguments
/// * `r_full` - Full R matrix (k×n, upper triangular)
/// * `k` - Number of rows in R (min(m, n))
/// * `n` - Number of columns in R
/// * `rtol` - Relative tolerance for row norms
///
/// # Returns
/// The retained rank `r` (at least 1, at most k)
fn compute_retained_rank_qr<T>(r_full: &DTensor<T, 2>, k: usize, n: usize, rtol: f64) -> usize
where
    T: ComplexFloat,
    <T as ComplexFloat>::Real: Into<f64>,
{
    if k == 0 || n == 0 {
        return 1;
    }

    let max_diag = k.min(n);

    // Compute the norm of each row of R (upper triangular: row i has entries from column i..n)
    let mut row_norms = Vec::with_capacity(max_diag);
    for i in 0..max_diag {
        let mut norm_sq: f64 = 0.0;
        for j in i..n {
            let val: f64 = r_full[[i, j]].abs().into();
            norm_sq += val * val;
        }
        row_norms.push(norm_sq.sqrt());
    }

    // Find max row norm
    let max_row_norm = row_norms.iter().cloned().fold(0.0_f64, f64::max);

    if max_row_norm == 0.0 {
        return 1;
    }

    let threshold = rtol * max_row_norm;

    // Count rows with norm above threshold
    let mut r = 0;
    for &norm in &row_norms {
        if norm >= threshold {
            r += 1;
        }
    }

    // Ensure at least rank 1 is kept
    r.max(1)
}

fn compute_retained_rank_qr_from_storage(
    r_full: &Storage,
    k: usize,
    n: usize,
    rtol: f64,
) -> Result<usize, QrError> {
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
    let row_norms: Vec<f64> = match r_full {
        Storage::DenseF64(data) => {
            let s = data.as_slice();
            (0..max_diag)
                .map(|i| {
                    let mut norm_sq: f64 = 0.0;
                    for j in i..n {
                        let val = s[i * n + j].abs();
                        norm_sq += val * val;
                    }
                    norm_sq.sqrt()
                })
                .collect()
        }
        Storage::DenseC64(data) => {
            let s = data.as_slice();
            (0..max_diag)
                .map(|i| {
                    let mut norm_sq: f64 = 0.0;
                    for j in i..n {
                        let val = s[i * n + j].abs();
                        norm_sq += val * val;
                    }
                    norm_sq.sqrt()
                })
                .collect()
        }
        other => {
            return Err(QrError::ComputationError(anyhow::anyhow!(
                "native QR expected dense R storage, got {:?}",
                std::mem::discriminant(other)
            )));
        }
    };

    let max_row_norm = row_norms.iter().cloned().fold(0.0_f64, f64::max);
    if max_row_norm == 0.0 {
        return Ok(1);
    }

    let threshold = rtol * max_row_norm;
    let r = row_norms.iter().filter(|&&norm| norm >= threshold).count();
    Ok(r.max(1))
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
#[allow(private_bounds)]
pub fn qr<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
) -> Result<(TensorDynLen, TensorDynLen), QrError>
where
    T: StorageScalar
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + tensor4all_tensorbackend::backend::BackendLinalgScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
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
#[allow(private_bounds)]
pub fn qr_with<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &QrOptions,
) -> Result<(TensorDynLen, TensorDynLen), QrError>
where
    T: StorageScalar
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + tensor4all_tensorbackend::backend::BackendLinalgScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Determine rtol to use
    let rtol = options.truncation.effective_rtol(default_qr_rtol());
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(QrError::InvalidRtol(rtol));
    }

    // Unfold tensor into matrix (returns DTensor<T, 2>)
    let (mut a_tensor, _, m, n, left_indices, right_indices) = unfold_split::<T>(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
        .map_err(QrError::ComputationError)?;

    let native = t.as_native();
    let supports_native = native.is_dense()
        && TypeId::of::<T>() == TypeId::of::<f64>()
        && native.scalar_type() == tensor4all_tensorbackend::tenferro_dyadtensor::ScalarType::F64;
    if supports_native {
        let mut permuted_indices = left_indices.clone();
        permuted_indices.extend(right_indices.iter().cloned());
        let permuted = t.permute_indices(&permuted_indices);
        // Use AD-aware reshape to flatten to 2D matrix.
        // Direct .as_native().contiguous(RowMajor).reshape() can produce
        // wrong element ordering due to column-major/row-major ambiguity
        // in the tenferro reshape implementation.
        let matrix_native = reshape_row_major_dyn_ad_tensor(permuted.as_native(), &[m, n])
            .map_err(|e| {
                QrError::ComputationError(anyhow::anyhow!(
                    "native QR matrix conversion failed: {e}"
                ))
            })?;
        let (mut q_native, mut r_native) =
            qr_dyn_ad_tensor_native(&matrix_native).map_err(QrError::ComputationError)?;
        let k = m.min(n);
        let r_storage =
            dyn_ad_tensor_primal_to_storage(&r_native).map_err(QrError::ComputationError)?;
        let r = compute_retained_rank_qr_from_storage(&r_storage, k, n, rtol)?;
        if r < k {
            q_native = q_native.take_prefix(1, r).map_err(|e| {
                QrError::ComputationError(anyhow::anyhow!("native QR truncation on Q failed: {e}"))
            })?;
            r_native = r_native.take_prefix(0, r).map_err(|e| {
                QrError::ComputationError(anyhow::anyhow!("native QR truncation on R failed: {e}"))
            })?;
        }

        let bond_index = DynIndex::new_bond(r)
            .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))
            .map_err(QrError::ComputationError)?;

        let mut q_indices = left_indices.clone();
        q_indices.push(bond_index.clone());
        let q_dims: Vec<usize> = q_indices.iter().map(|idx| idx.dim).collect();
        let q_reshaped = reshape_row_major_dyn_ad_tensor(&q_native, &q_dims).map_err(|e| {
            QrError::ComputationError(anyhow::anyhow!("native QR Q reshape failed: {e}"))
        })?;
        let q =
            TensorDynLen::from_native(q_indices, q_reshaped).map_err(QrError::ComputationError)?;

        let mut r_indices = vec![bond_index.clone()];
        r_indices.extend_from_slice(&right_indices);
        let r_dims: Vec<usize> = r_indices.iter().map(|idx| idx.dim).collect();
        let r_reshaped = reshape_row_major_dyn_ad_tensor(&r_native, &r_dims).map_err(|e| {
            QrError::ComputationError(anyhow::anyhow!("native QR R reshape failed: {e}"))
        })?;
        let r_tensor =
            TensorDynLen::from_native(r_indices, r_reshaped).map_err(QrError::ComputationError)?;

        return Ok((q, r_tensor));
    }
    let k = m.min(n);

    // Call QR using selected backend
    // DTensor can be converted to DSlice via as_mut()
    let a_slice: &mut DSlice<T, 2> = a_tensor.as_mut();
    let (q_full, r_full) = qr_backend(a_slice);

    // Compute retained rank based on rtol truncation
    let r = compute_retained_rank_qr(&r_full, k, n, rtol);

    // Extract truncated QR: keep first r columns of Q and first r rows of R
    // Q_thin: m×r (first r columns of Q)
    let mut q_vec = Vec::with_capacity(m * r);
    for i in 0..m {
        for j in 0..r {
            q_vec.push(q_full[[i, j]]);
        }
    }

    // R_thin: r×n (first r rows of R)
    let mut r_vec = Vec::with_capacity(r * n);
    for i in 0..r {
        for j in 0..n {
            r_vec.push(r_full[[i, j]]);
        }
    }

    // Create bond index with "Link" tag (dimension r, not k)
    let bond_index = DynIndex::new_bond(r)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))
        .map_err(QrError::ComputationError)?;

    // Create Q tensor: [left_inds..., bond_index]
    let mut q_indices = left_indices.clone();
    q_indices.push(bond_index.clone());
    let q_dims: Vec<usize> = q_indices.iter().map(|idx| idx.dim).collect();
    let q_storage = T::dense_storage_with_shape(q_vec, &q_dims);
    let q = TensorDynLen::from_indices(q_indices, q_storage);

    // Create R tensor: [bond_index, right_inds...]
    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(&right_indices);
    let r_dims: Vec<usize> = r_indices.iter().map(|idx| idx.dim).collect();
    let r_storage = T::dense_storage_with_shape(r_vec, &r_dims);
    let r = TensorDynLen::from_indices(r_indices, r_storage);

    Ok((q, r))
}

/// Compute QR decomposition of a complex tensor with arbitrary rank, returning (Q, R).
///
/// This is a convenience wrapper around the generic `qr` function for `Complex64` tensors.
///
/// For the mathematical convention:
/// \[ A = Q * R \]
/// where Q is unitary and R is upper triangular.
///
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
#[inline]
pub fn qr_c64(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
) -> Result<(TensorDynLen, TensorDynLen), QrError> {
    qr::<Complex64>(t, left_inds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::DefaultIndex as Index;
    use std::sync::Arc;
    use tensor4all_tensorbackend::mdarray::DTensor;

    #[test]
    fn compute_retained_rank_qr_from_storage_truncates_and_keeps_one() {
        let r_full = Storage::DenseF64(
            tensor4all_tensorbackend::DenseStorageF64::from_vec_with_shape(
                vec![3.0, 1.0, 0.0, 1.0e-14],
                &[2, 2],
            ),
        );

        let retained = compute_retained_rank_qr_from_storage(&r_full, 2, 2, 1.0e-10).unwrap();
        assert_eq!(retained, 1);

        let zero = Storage::DenseF64(
            tensor4all_tensorbackend::DenseStorageF64::from_vec_with_shape(
                vec![0.0, 0.0, 0.0, 0.0],
                &[2, 2],
            ),
        );
        let retained_zero = compute_retained_rank_qr_from_storage(&zero, 2, 2, 1.0).unwrap();
        assert_eq!(retained_zero, 1);
    }

    #[test]
    fn compute_retained_rank_qr_from_storage_rejects_non_dense_r() {
        let diag = Storage::new_diag_f64(vec![1.0, 2.0]);
        let err = compute_retained_rank_qr_from_storage(&diag, 2, 2, 1.0e-12).unwrap_err();
        assert!(err
            .to_string()
            .contains("native QR expected dense R storage"));
    }

    #[test]
    fn compute_retained_rank_qr_from_storage_handles_empty_and_complex_dense() {
        let empty = Storage::DenseC64(
            tensor4all_tensorbackend::DenseStorageC64::from_vec_with_shape(vec![], &[0]),
        );
        assert_eq!(
            compute_retained_rank_qr_from_storage(&empty, 0, 2, 1.0e-12).unwrap(),
            1
        );

        let dense_c64 = Storage::DenseC64(
            tensor4all_tensorbackend::DenseStorageC64::from_vec_with_shape(
                vec![
                    Complex64::new(2.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0e-14, 0.0),
                ],
                &[2, 2],
            ),
        );
        assert_eq!(
            compute_retained_rank_qr_from_storage(&dense_c64, 2, 2, 1.0e-10).unwrap(),
            1
        );
    }

    #[test]
    fn set_default_qr_rtol_rejects_invalid_values() {
        let original = default_qr_rtol();
        assert!(set_default_qr_rtol(f64::NAN).is_err());
        assert!(set_default_qr_rtol(-1.0).is_err());
        set_default_qr_rtol(original).unwrap();
    }

    #[test]
    fn qr_options_report_rtol_and_default_roundtrips() {
        let original = default_qr_rtol();
        let options = QrOptions::with_rtol(1.0e-7);
        assert_eq!(options.rtol(), Some(1.0e-7));
        assert_eq!(QrOptions::default().rtol(), None);

        set_default_qr_rtol(1.0e-9).unwrap();
        assert_eq!(default_qr_rtol(), 1.0e-9);
        set_default_qr_rtol(original).unwrap();
    }

    #[test]
    fn qr_with_invalid_rtol_is_rejected_before_linalg() {
        let i = Index::new_dyn(2);
        let j = Index::new_dyn(2);
        let tensor = TensorDynLen::new(
            vec![i.clone(), j.clone()],
            Arc::new(Storage::new_dense_f64(4)),
        );

        let nan = qr_with::<f64>(
            &tensor,
            std::slice::from_ref(&i),
            &QrOptions::with_rtol(f64::NAN),
        );
        assert!(matches!(nan, Err(QrError::InvalidRtol(v)) if v.is_nan()));

        let negative = qr_with::<f64>(
            &tensor,
            std::slice::from_ref(&i),
            &QrOptions::with_rtol(-1.0),
        );
        assert!(matches!(negative, Err(QrError::InvalidRtol(v)) if v == -1.0));
    }

    #[test]
    fn qr_with_native_truncation_reduces_bond_dimension() {
        let i = Index::new_dyn(2);
        let j = Index::new_dyn(2);
        let mut data = vec![0.0; 4];
        data[0] = 1.0;
        data[3] = 1.0e-14;
        let tensor = TensorDynLen::new(
            vec![i.clone(), j.clone()],
            Arc::new(Storage::DenseF64(
                tensor4all_tensorbackend::DenseStorageF64::from_vec_with_shape(data, &[2, 2]),
            )),
        );

        let (q, r) = qr_with::<f64>(
            &tensor,
            std::slice::from_ref(&i),
            &QrOptions::with_rtol(1.0e-10),
        )
        .unwrap();
        assert_eq!(q.dims(), vec![2, 1]);
        assert_eq!(r.dims(), vec![1, 2]);
    }

    #[test]
    fn qr_with_complex_fallback_truncation_reduces_bond_dimension() {
        let i = Index::new_dyn(2);
        let j = Index::new_dyn(2);
        let mut data = vec![Complex64::new(0.0, 0.0); 4];
        data[0] = Complex64::new(1.0, 0.0);
        data[3] = Complex64::new(1.0e-14, 0.0);
        let tensor = TensorDynLen::new(
            vec![i.clone(), j.clone()],
            Arc::new(Storage::DenseC64(
                tensor4all_tensorbackend::DenseStorageC64::from_vec_with_shape(data, &[2, 2]),
            )),
        );

        let (q, r) = qr_with::<Complex64>(
            &tensor,
            std::slice::from_ref(&i),
            &QrOptions::with_rtol(1.0e-10),
        )
        .unwrap();
        assert_eq!(q.dims(), vec![2, 1]);
        assert_eq!(r.dims(), vec![1, 2]);
    }

    /// Helper: build an upper-triangular R matrix from diagonal and off-diagonal entries.
    fn make_upper_triangular(n: usize, entries: &[(usize, usize, f64)]) -> DTensor<f64, 2> {
        DTensor::<f64, 2>::from_fn([n, n], |idx| {
            entries
                .iter()
                .find(|(i, j, _)| *i == idx[0] && *j == idx[1])
                .map(|(_, _, v)| *v)
                .unwrap_or(0.0)
        })
    }

    #[test]
    fn test_retained_rank_zero_diagonal_nonzero_offdiag() {
        // 3×4 R with zero diagonal at row 1 but nonzero off-diag
        // R = [[10, 1, 1, 1],
        //      [ 0, 0, 5, 5],   ← diagonal=0, but row norm = sqrt(50) ≈ 7.07
        //      [ 0, 0, 0, 1]]
        let r = DTensor::<f64, 2>::from_fn([3, 4], |idx| match (idx[0], idx[1]) {
            (0, 0) => 10.0,
            (0, 1) => 1.0,
            (0, 2) => 1.0,
            (0, 3) => 1.0,
            (1, 2) => 5.0,
            (1, 3) => 5.0,
            (2, 3) => 1.0,
            _ => 0.0,
        });
        // rtol=1e-15: all rows should be retained
        assert_eq!(compute_retained_rank_qr(&r, 3, 4, 1e-15), 3);
    }

    #[test]
    fn test_retained_rank_all_zero_rows() {
        // R with only row 0 non-zero
        let r = make_upper_triangular(3, &[(0, 0, 5.0), (0, 1, 3.0), (0, 2, 1.0)]);
        assert_eq!(compute_retained_rank_qr(&r, 3, 3, 1e-15), 1);
    }

    #[test]
    fn test_retained_rank_full_rank() {
        // Fully non-degenerate upper triangular
        let r = make_upper_triangular(
            3,
            &[
                (0, 0, 10.0),
                (0, 1, 1.0),
                (0, 2, 1.0),
                (1, 1, 8.0),
                (1, 2, 1.0),
                (2, 2, 6.0),
            ],
        );
        assert_eq!(compute_retained_rank_qr(&r, 3, 3, 1e-15), 3);
    }

    #[test]
    fn test_retained_rank_rtol_truncation() {
        let r = make_upper_triangular(
            3,
            &[
                (0, 0, 10.0),
                (0, 1, 0.5),
                (0, 2, 0.1),
                (1, 1, 0.01),
                (2, 2, 0.001),
            ],
        );
        assert_eq!(compute_retained_rank_qr(&r, 3, 3, 0.01), 1);
        assert_eq!(compute_retained_rank_qr(&r, 3, 3, 1e-4), 2);
    }

    #[test]
    fn test_retained_rank_zero_matrix() {
        let r = DTensor::<f64, 2>::from_fn([3, 3], |_| 0.0);
        assert_eq!(compute_retained_rank_qr(&r, 3, 3, 1e-15), 1);
    }

    #[test]
    fn test_retained_rank_complex() {
        use num_complex::Complex64;
        let r = DTensor::<Complex64, 2>::from_fn([2, 3], |idx| match (idx[0], idx[1]) {
            (0, 0) => Complex64::new(5.0, 3.0),
            (0, 1) => Complex64::new(1.0, 0.0),
            (0, 2) => Complex64::new(0.0, 1.0),
            (1, 1) => Complex64::new(0.0, 0.0), // zero diagonal
            (1, 2) => Complex64::new(3.0, 4.0), // norm = 5.0
            _ => Complex64::new(0.0, 0.0),
        });
        // Row 1 has zero diagonal but norm = 5.0, should NOT be truncated
        assert_eq!(compute_retained_rank_qr(&r, 2, 3, 1e-15), 2);
    }
}
