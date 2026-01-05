//! Unified tensor factorization module.
//!
//! This module provides a unified `factorize()` function that dispatches to
//! SVD, QR, LU, or CI (Cross Interpolation) algorithms based on options.
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_core_linalg::{factorize, FactorizeOptions, FactorizeAlg, Canonical};
//!
//! let result = factorize(&tensor, &left_inds, &FactorizeOptions::default())?;
//! // result.left * result.right ≈ tensor
//! ```

use num_complex::{Complex64, ComplexFloat};
use tensor4all_core_common::index::{DynId, Index, NoSymmSpace, Symmetry, TagSet};
use tensor4all_core_tensor::{unfold_split, Storage, StorageScalar, TensorDynLen};
use tensor4all_matrixci::{rrlu, AbstractMatrixCI, MatrixLUCI, RrLUOptions, Scalar as MatrixScalar};
use thiserror::Error;

use crate::qr::{qr_with, QrOptions};
use crate::svd::{svd_with, SvdOptions};
use faer_traits::ComplexField;

/// Error type for factorize operations.
#[derive(Debug, Error)]
pub enum FactorizeError {
    #[error("Factorization failed: {0}")]
    ComputationError(#[from] anyhow::Error),
    #[error("Invalid rtol value: {0}. rtol must be finite and non-negative.")]
    InvalidRtol(f64),
    #[error("Unsupported storage type: {0}")]
    UnsupportedStorage(&'static str),
    #[error("Unsupported canonical direction for this algorithm: {0}")]
    UnsupportedCanonical(&'static str),
    #[error("SVD error: {0}")]
    SvdError(#[from] crate::svd::SvdError),
    #[error("QR error: {0}")]
    QrError(#[from] crate::qr::QrError),
}

/// Factorization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FactorizeAlg {
    /// Singular Value Decomposition
    #[default]
    SVD,
    /// QR decomposition
    QR,
    /// Rank-revealing LU decomposition
    LU,
    /// Cross Interpolation (LU-based)
    CI,
}

/// Canonical direction for factorization.
///
/// This determines which factor is "canonical" (orthogonal for SVD/QR,
/// or unit-diagonal for LU/CI).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Canonical {
    /// Left factor is canonical.
    /// - SVD: L=U (orthogonal), R=S*V
    /// - QR: L=Q (orthogonal), R=R
    /// - LU/CI: L has unit diagonal
    #[default]
    Left,
    /// Right factor is canonical.
    /// - SVD: L=U*S, R=V (orthogonal)
    /// - QR: Not supported (would need LQ)
    /// - LU/CI: U has unit diagonal
    Right,
}

/// Options for tensor factorization.
#[derive(Debug, Clone)]
pub struct FactorizeOptions {
    /// Factorization algorithm to use.
    pub alg: FactorizeAlg,
    /// Canonical direction.
    pub canonical: Canonical,
    /// Relative tolerance for truncation.
    /// If `None`, uses the algorithm's default.
    pub rtol: Option<f64>,
    /// Maximum rank for truncation.
    /// If `None`, no rank limit is applied.
    pub max_rank: Option<usize>,
}

impl Default for FactorizeOptions {
    fn default() -> Self {
        Self {
            alg: FactorizeAlg::SVD,
            canonical: Canonical::Left,
            rtol: None,
            max_rank: None,
        }
    }
}

impl FactorizeOptions {
    /// Create options for SVD factorization.
    pub fn svd() -> Self {
        Self {
            alg: FactorizeAlg::SVD,
            ..Default::default()
        }
    }

    /// Create options for QR factorization.
    pub fn qr() -> Self {
        Self {
            alg: FactorizeAlg::QR,
            ..Default::default()
        }
    }

    /// Create options for LU factorization.
    pub fn lu() -> Self {
        Self {
            alg: FactorizeAlg::LU,
            ..Default::default()
        }
    }

    /// Create options for CI factorization.
    pub fn ci() -> Self {
        Self {
            alg: FactorizeAlg::CI,
            ..Default::default()
        }
    }

    /// Set canonical direction.
    pub fn with_canonical(mut self, canonical: Canonical) -> Self {
        self.canonical = canonical;
        self
    }

    /// Set relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = Some(rtol);
        self
    }

    /// Set maximum rank.
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = Some(max_rank);
        self
    }
}

/// Result of tensor factorization.
#[derive(Debug, Clone)]
pub struct FactorizeResult<Id, Symm>
where
    Id: Clone,
    Symm: Clone,
{
    /// Left factor tensor.
    pub left: TensorDynLen<Id, Symm>,
    /// Right factor tensor.
    pub right: TensorDynLen<Id, Symm>,
    /// Bond index connecting left and right factors.
    pub bond_index: Index<Id, Symm, TagSet>,
    /// Singular values (only for SVD).
    pub singular_values: Option<Vec<f64>>,
    /// Rank of the factorization.
    pub rank: usize,
}

/// Factorize a tensor into left and right factors.
///
/// This function dispatches to the appropriate algorithm based on `options.alg`:
/// - `SVD`: Singular Value Decomposition
/// - `QR`: QR decomposition
/// - `LU`: Rank-revealing LU decomposition
/// - `CI`: Cross Interpolation
///
/// The `canonical` option controls which factor is "canonical":
/// - `Canonical::Left`: Left factor is orthogonal (SVD/QR) or unit-diagonal (LU/CI)
/// - `Canonical::Right`: Right factor is orthogonal (SVD) or unit-diagonal (LU/CI)
///
/// # Arguments
/// * `t` - Input tensor
/// * `left_inds` - Indices to place on the left side
/// * `options` - Factorization options
///
/// # Returns
/// A `FactorizeResult` containing the left and right factors, bond index,
/// singular values (for SVD), and rank.
///
/// # Errors
/// Returns `FactorizeError` if:
/// - The storage type is not supported (only DenseF64 and DenseC64)
/// - QR is used with `Canonical::Right`
/// - The underlying algorithm fails
pub fn factorize<Id, Symm>(
    t: &TensorDynLen<Id, Symm>,
    left_inds: &[Index<Id, Symm>],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<Id, Symm>, FactorizeError>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
{
    // Dispatch based on storage type
    match t.storage.as_ref() {
        Storage::DenseF64(_) => factorize_impl::<Id, Symm, f64>(t, left_inds, options),
        Storage::DenseC64(_) => factorize_impl::<Id, Symm, Complex64>(t, left_inds, options),
        Storage::DiagF64(_) | Storage::DiagC64(_) => Err(FactorizeError::UnsupportedStorage(
            "Diagonal storage not supported for factorize",
        )),
    }
}

/// Internal implementation with scalar type.
fn factorize_impl<Id, Symm, T>(
    t: &TensorDynLen<Id, Symm>,
    left_inds: &[Index<Id, Symm>],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<Id, Symm>, FactorizeError>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    T: StorageScalar
        + ComplexFloat
        + ComplexField
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    match options.alg {
        FactorizeAlg::SVD => factorize_svd::<Id, Symm, T>(t, left_inds, options),
        FactorizeAlg::QR => factorize_qr::<Id, Symm, T>(t, left_inds, options),
        FactorizeAlg::LU => factorize_lu::<Id, Symm, T>(t, left_inds, options),
        FactorizeAlg::CI => factorize_ci::<Id, Symm, T>(t, left_inds, options),
    }
}

/// SVD factorization implementation.
fn factorize_svd<Id, Symm, T>(
    t: &TensorDynLen<Id, Symm>,
    left_inds: &[Index<Id, Symm>],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<Id, Symm>, FactorizeError>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    T: StorageScalar
        + ComplexFloat
        + ComplexField
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    let svd_options = SvdOptions { rtol: options.rtol };

    let (u, s, v) = svd_with::<Id, Symm, T>(t, left_inds, &svd_options)?;

    // Extract singular values from diagonal tensor
    let singular_values = extract_singular_values(&s);
    let rank = singular_values.len();

    // Get bond indices from S tensor:
    // S has indices [bond_index, sim(bond_index)]
    // - U shares bond_index (S.indices[0])
    // - V shares bond_index (S.indices[0])
    // After contraction, we need to ensure left and right share the same bond index
    let bond_index = s.indices[0].clone();
    let sim_bond_index = s.indices[1].clone();

    // Convert tagged indices to untagged for replaceind
    let bond_idx_untagged = Index::new(bond_index.id.clone(), bond_index.symm.clone());
    let sim_bond_idx_untagged = Index::new(sim_bond_index.id.clone(), sim_bond_index.symm.clone());

    match options.canonical {
        Canonical::Left => {
            // L = U, R = S * V
            // After S * V contraction:
            // - S has [bond_index, sim_bond_index]
            // - V has [right_inds..., bond_index]
            // - Result has [sim_bond_index, right_inds...]
            // But U has [left_inds..., bond_index], so we need to replace
            // sim_bond_index with bond_index in the result for reconstruction to work.
            let right_contracted = s.contract_einsum(&v);
            let right = right_contracted.replaceind(&sim_bond_idx_untagged, &bond_idx_untagged);
            Ok(FactorizeResult {
                left: u,
                right,
                bond_index,
                singular_values: Some(singular_values),
                rank,
            })
        }
        Canonical::Right => {
            // L = U * S, R = V
            // After U * S contraction:
            // - U has [left_inds..., bond_index]
            // - S has [bond_index, sim_bond_index]
            // - Result has [left_inds..., sim_bond_index]
            // But V has [right_inds..., bond_index], so we need to replace
            // sim_bond_index with bond_index in the result for reconstruction to work.
            let left_contracted = u.contract_einsum(&s);
            let left = left_contracted.replaceind(&sim_bond_idx_untagged, &bond_idx_untagged);
            Ok(FactorizeResult {
                left,
                right: v,
                bond_index,
                singular_values: Some(singular_values),
                rank,
            })
        }
    }
}

/// QR factorization implementation.
fn factorize_qr<Id, Symm, T>(
    t: &TensorDynLen<Id, Symm>,
    left_inds: &[Index<Id, Symm>],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<Id, Symm>, FactorizeError>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    T: StorageScalar
        + ComplexFloat
        + ComplexField
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    if options.canonical == Canonical::Right {
        return Err(FactorizeError::UnsupportedCanonical(
            "QR only supports Canonical::Left (would need LQ for right)",
        ));
    }

    let qr_options = QrOptions { rtol: options.rtol };

    let (q, r) = qr_with::<Id, Symm, T>(t, left_inds, &qr_options)?;

    // Get bond index from Q tensor (last index)
    let bond_index = q.indices.last().unwrap().clone();
    // Rank is the last dimension of Q
    let rank = *q.dims.last().unwrap();

    Ok(FactorizeResult {
        left: q,
        right: r,
        bond_index,
        singular_values: None,
        rank,
    })
}

/// LU factorization implementation.
fn factorize_lu<Id, Symm, T>(
    t: &TensorDynLen<Id, Symm>,
    left_inds: &[Index<Id, Symm>],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<Id, Symm>, FactorizeError>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    T: StorageScalar
        + ComplexFloat
        + ComplexField
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Unfold tensor into matrix
    let (a_tensor, _, m, n, left_indices, right_indices) =
        unfold_split::<Id, T, Symm>(t, left_inds)
            .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))?;

    // Convert to Matrix type for rrlu
    let a_matrix = dtensor_to_matrix(&a_tensor, m, n);

    // Set up LU options
    let left_orthogonal = options.canonical == Canonical::Left;
    let lu_options = RrLUOptions {
        max_rank: options.max_rank.unwrap_or(usize::MAX),
        rel_tol: options.rtol.unwrap_or(1e-14),
        abs_tol: 0.0,
        left_orthogonal,
    };

    // Perform LU decomposition
    let lu = rrlu(&a_matrix, Some(lu_options));
    let rank = lu.npivots();

    // Extract L and U matrices (permuted)
    let l_matrix = lu.left(true);
    let u_matrix = lu.right(true);

    // Create bond index
    let dyn_bond_index = Index::new_link(rank)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))?;
    // Convert from Index<DynId, NoSymmSpace, TagSet> to Index<Id, Symm, TagSet>
    let bond_index: Index<Id, Symm, TagSet> = Index {
        id: dyn_bond_index.id.into(),
        symm: dyn_bond_index.symm.into(),
        tags: dyn_bond_index.tags,
    };

    // Convert L matrix back to tensor
    let l_vec = matrix_to_vec(&l_matrix);
    let mut l_indices = left_indices.clone();
    l_indices.push(bond_index.clone());
    let l_storage = T::dense_storage(l_vec);
    let left = TensorDynLen::from_indices(l_indices, l_storage);

    // Convert U matrix back to tensor
    let u_vec = matrix_to_vec(&u_matrix);
    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(&right_indices);
    let r_storage = T::dense_storage(u_vec);
    let right = TensorDynLen::from_indices(r_indices, r_storage);

    Ok(FactorizeResult {
        left,
        right,
        bond_index,
        singular_values: None,
        rank,
    })
}

/// CI (Cross Interpolation) factorization implementation.
fn factorize_ci<Id, Symm, T>(
    t: &TensorDynLen<Id, Symm>,
    left_inds: &[Index<Id, Symm>],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<Id, Symm>, FactorizeError>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    T: StorageScalar
        + ComplexFloat
        + ComplexField
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Unfold tensor into matrix
    let (a_tensor, _, m, n, left_indices, right_indices) =
        unfold_split::<Id, T, Symm>(t, left_inds)
            .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))?;

    // Convert to Matrix type for MatrixLUCI
    let a_matrix = dtensor_to_matrix(&a_tensor, m, n);

    // Set up LU options for CI
    let left_orthogonal = options.canonical == Canonical::Left;
    let lu_options = RrLUOptions {
        max_rank: options.max_rank.unwrap_or(usize::MAX),
        rel_tol: options.rtol.unwrap_or(1e-14),
        abs_tol: 0.0,
        left_orthogonal,
    };

    // Perform CI decomposition
    let ci = MatrixLUCI::from_matrix(&a_matrix, Some(lu_options));
    let rank = ci.rank();

    // Get left and right matrices from CI
    let l_matrix = ci.left();
    let r_matrix = ci.right();

    // Create bond index
    let dyn_bond_index = Index::new_link(rank)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))?;
    // Convert from Index<DynId, NoSymmSpace, TagSet> to Index<Id, Symm, TagSet>
    let bond_index: Index<Id, Symm, TagSet> = Index {
        id: dyn_bond_index.id.into(),
        symm: dyn_bond_index.symm.into(),
        tags: dyn_bond_index.tags,
    };

    // Convert L matrix back to tensor
    let l_vec = matrix_to_vec(&l_matrix);
    let mut l_indices = left_indices.clone();
    l_indices.push(bond_index.clone());
    let l_storage = T::dense_storage(l_vec);
    let left = TensorDynLen::from_indices(l_indices, l_storage);

    // Convert R matrix back to tensor
    let r_vec = matrix_to_vec(&r_matrix);
    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(&right_indices);
    let r_storage = T::dense_storage(r_vec);
    let right = TensorDynLen::from_indices(r_indices, r_storage);

    Ok(FactorizeResult {
        left,
        right,
        bond_index,
        singular_values: None,
        rank,
    })
}

/// Extract singular values from a diagonal tensor.
fn extract_singular_values<Id, Symm>(s: &TensorDynLen<Id, Symm>) -> Vec<f64>
where
    Id: Clone,
    Symm: Clone,
{
    match s.storage.as_ref() {
        Storage::DiagF64(diag) => diag.as_slice().to_vec(),
        Storage::DiagC64(diag) => {
            // Singular values should be real
            diag.as_slice().iter().map(|c| c.re).collect()
        }
        Storage::DenseF64(dense) => {
            // Extract diagonal from dense matrix
            let n = s.dims[0];
            (0..n).map(|i| dense.get(i * n + i)).collect()
        }
        Storage::DenseC64(dense) => {
            let n = s.dims[0];
            (0..n).map(|i| dense.get(i * n + i).re).collect()
        }
    }
}

/// Convert DTensor to Matrix (tensor4all-matrixci format).
fn dtensor_to_matrix<T>(
    tensor: &mdarray::DTensor<T, 2>,
    m: usize,
    n: usize,
) -> tensor4all_matrixci::Matrix<T>
where
    T: MatrixScalar + Clone,
{
    let mut matrix = tensor4all_matrixci::util::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            matrix[[i, j]] = tensor[[i, j]].clone();
        }
    }
    matrix
}

/// Convert Matrix to Vec for storage.
fn matrix_to_vec<T>(matrix: &tensor4all_matrixci::Matrix<T>) -> Vec<T>
where
    T: Clone,
{
    let m = tensor4all_matrixci::util::nrows(matrix);
    let n = tensor4all_matrixci::util::ncols(matrix);
    let mut vec = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            vec.push(matrix[[i, j]].clone());
        }
    }
    vec
}

#[cfg(test)]
mod tests {
    // Tests are in the tests/factorize.rs file
}
