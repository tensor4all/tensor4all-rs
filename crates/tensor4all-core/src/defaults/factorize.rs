//! Unified tensor factorization module.
//!
//! This module provides a unified `factorize()` function that dispatches to
//! SVD, QR, LU, or CI (Cross Interpolation) algorithms based on options.
//!
//! # Note
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.
//! Generic tensor types are not supported.
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_core::{factorize, FactorizeOptions, FactorizeAlg, Canonical};
//!
//! let result = factorize(&tensor, &left_inds, &FactorizeOptions::default())?;
//! // result.left * result.right ≈ tensor
//! ```

use crate::defaults::DynIndex;
use crate::{unfold_split, TensorDynLen};
use matrixci::{rrlu, AbstractMatrixCI, MatrixLUCI, RrLUOptions, Scalar as MatrixScalar};
use num_complex::{Complex64, ComplexFloat};
use tensor4all_tensorbackend::{native_tensor_primal_to_diag_f64, TensorElement};

use crate::qr::{qr_with, QrOptions};
use crate::svd::{svd_with, SvdOptions};

// Re-export types from tensor_like for backwards compatibility
pub use crate::tensor_like::{
    Canonical, FactorizeAlg, FactorizeError, FactorizeOptions, FactorizeResult,
};

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
pub fn factorize(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    if t.is_diag() {
        return Err(FactorizeError::UnsupportedStorage(
            "Diagonal storage not supported for factorize",
        ));
    }

    if t.is_f64() {
        factorize_impl_f64(t, left_inds, options)
    } else if t.is_complex() {
        factorize_impl_c64(t, left_inds, options)
    } else {
        Err(FactorizeError::UnsupportedStorage(
            "factorize currently supports only f64 and Complex64 tensors",
        ))
    }
}

fn factorize_impl_f64(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    match options.alg {
        FactorizeAlg::SVD => factorize_svd(t, left_inds, options),
        FactorizeAlg::QR => factorize_qr(t, left_inds, options),
        FactorizeAlg::LU => factorize_lu::<f64>(t, left_inds, options),
        FactorizeAlg::CI => factorize_ci::<f64>(t, left_inds, options),
    }
}

fn factorize_impl_c64(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    match options.alg {
        FactorizeAlg::SVD => factorize_svd(t, left_inds, options),
        FactorizeAlg::QR => factorize_qr(t, left_inds, options),
        FactorizeAlg::LU => factorize_lu::<Complex64>(t, left_inds, options),
        FactorizeAlg::CI => factorize_ci::<Complex64>(t, left_inds, options),
    }
}

/// SVD factorization implementation.
fn factorize_svd(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    let mut svd_options = SvdOptions::default();
    if let Some(rtol) = options.rtol {
        svd_options.truncation.rtol = Some(rtol);
    }
    if let Some(max_rank) = options.max_rank {
        svd_options.truncation.max_rank = Some(max_rank);
    }

    let (u, s, v) = svd_with::<f64>(t, left_inds, &svd_options)?;
    let bond_index = u.indices.last().unwrap().clone();
    let sim_bond_index = s.indices[1].clone();
    let singular_values = native_tensor_primal_to_diag_f64(s.as_native())
        .map_err(FactorizeError::ComputationError)?;
    let rank = singular_values.len();
    let perm_vh: Vec<usize> = std::iter::once(v.indices.len() - 1)
        .chain(0..v.indices.len() - 1)
        .collect();
    let vh = v.conj().permute(&perm_vh);

    match options.canonical {
        Canonical::Left => {
            // L = U (orthogonal), R = S * V^H
            let right_contracted = s.contract(&vh);
            let right = right_contracted.replaceind(&sim_bond_index, &bond_index);
            Ok(FactorizeResult {
                left: u,
                right,
                bond_index,
                singular_values: Some(singular_values),
                rank,
            })
        }
        Canonical::Right => {
            // L = U * S, R = V^H
            let left_contracted = u.contract(&s);
            let left = left_contracted.replaceind(&sim_bond_index, &bond_index);
            Ok(FactorizeResult {
                left,
                right: vh,
                bond_index,
                singular_values: Some(singular_values),
                rank,
            })
        }
    }
}

/// QR factorization implementation.
fn factorize_qr(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    if options.canonical == Canonical::Right {
        return Err(FactorizeError::UnsupportedCanonical(
            "QR only supports Canonical::Left (would need LQ for right)",
        ));
    }

    let mut qr_options = QrOptions::default();
    if let Some(rtol) = options.rtol {
        qr_options.truncation.rtol = Some(rtol);
    }

    let (q, r) = qr_with::<f64>(t, left_inds, &qr_options)?;

    // Get bond index from Q tensor (last index)
    let bond_index = q.indices.last().unwrap().clone();
    // Rank is the last dimension of Q
    let q_dims = q.dims();
    let rank = *q_dims.last().unwrap();

    Ok(FactorizeResult {
        left: q,
        right: r,
        bond_index,
        singular_values: None,
        rank,
    })
}

/// LU factorization implementation.
fn factorize_lu<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Unfold tensor into matrix
    let (a_tensor, _, m, n, left_indices, right_indices) = unfold_split(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))?;

    // Convert to Matrix type for rrlu
    let a_matrix = native_tensor_to_matrix::<T>(&a_tensor, m, n)?;

    // Set up LU options
    let left_orthogonal = options.canonical == Canonical::Left;
    let lu_options = RrLUOptions {
        max_rank: options.max_rank.unwrap_or(usize::MAX),
        rel_tol: options.rtol.unwrap_or(1e-14),
        abs_tol: 0.0,
        left_orthogonal,
    };

    // Perform LU decomposition
    let lu = rrlu(&a_matrix, Some(lu_options))?;
    let rank = lu.npivots();

    // Extract L and U matrices (permuted)
    let l_matrix = lu.left(true);
    let u_matrix = lu.right(true);

    // Create bond index
    let bond_index = DynIndex::new_bond(rank)
        .map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))?;

    // Convert L matrix back to tensor
    let l_vec = matrix_to_vec(&l_matrix);
    let mut l_indices = left_indices.clone();
    l_indices.push(bond_index.clone());
    let left =
        TensorDynLen::from_dense(l_indices, l_vec).map_err(FactorizeError::ComputationError)?;

    // Convert U matrix back to tensor
    let u_vec = matrix_to_vec(&u_matrix);
    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(&right_indices);
    let right =
        TensorDynLen::from_dense(r_indices, u_vec).map_err(FactorizeError::ComputationError)?;

    Ok(FactorizeResult {
        left,
        right,
        bond_index,
        singular_values: None,
        rank,
    })
}

/// CI (Cross Interpolation) factorization implementation.
fn factorize_ci<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Unfold tensor into matrix
    let (a_tensor, _, m, n, left_indices, right_indices) = unfold_split(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))?;

    // Convert to Matrix type for MatrixLUCI
    let a_matrix = native_tensor_to_matrix::<T>(&a_tensor, m, n)?;

    // Set up LU options for CI
    let left_orthogonal = options.canonical == Canonical::Left;
    let lu_options = RrLUOptions {
        max_rank: options.max_rank.unwrap_or(usize::MAX),
        rel_tol: options.rtol.unwrap_or(1e-14),
        abs_tol: 0.0,
        left_orthogonal,
    };

    // Perform CI decomposition
    let ci = MatrixLUCI::from_matrix(&a_matrix, Some(lu_options))?;
    let rank = ci.rank();

    // Get left and right matrices from CI
    let l_matrix = ci.left();
    let r_matrix = ci.right();

    // Create bond index
    let bond_index = DynIndex::new_bond(rank)
        .map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))?;

    // Convert L matrix back to tensor
    let l_vec = matrix_to_vec(&l_matrix);
    let mut l_indices = left_indices.clone();
    l_indices.push(bond_index.clone());
    let left =
        TensorDynLen::from_dense(l_indices, l_vec).map_err(FactorizeError::ComputationError)?;

    // Convert R matrix back to tensor
    let r_vec = matrix_to_vec(&r_matrix);
    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(&right_indices);
    let right =
        TensorDynLen::from_dense(r_indices, r_vec).map_err(FactorizeError::ComputationError)?;

    Ok(FactorizeResult {
        left,
        right,
        bond_index,
        singular_values: None,
        rank,
    })
}

/// Convert a native rank-2 tensor into a `matrixci::Matrix`.
fn native_tensor_to_matrix<T>(
    tensor: &tenferro::Tensor,
    m: usize,
    n: usize,
) -> Result<matrixci::Matrix<T>, FactorizeError>
where
    T: TensorElement + MatrixScalar + Copy,
{
    let data = T::dense_values_from_native(tensor).map_err(|e| {
        FactorizeError::ComputationError(anyhow::anyhow!(
            "failed to extract dense matrix entries from native tensor: {e}"
        ))
    })?;
    if data.len() != m * n {
        return Err(FactorizeError::ComputationError(anyhow::anyhow!(
            "native matrix materialization produced {} entries for shape ({m}, {n})",
            data.len()
        )));
    }

    let mut matrix = matrixci::util::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            matrix[[i, j]] = data[i + m * j];
        }
    }
    Ok(matrix)
}

/// Convert Matrix to Vec for storage.
fn matrix_to_vec<T>(matrix: &matrixci::Matrix<T>) -> Vec<T>
where
    T: Clone,
{
    let m = matrixci::util::nrows(matrix);
    let n = matrixci::util::ncols(matrix);
    let mut vec = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            vec.push(matrix[[i, j]].clone());
        }
    }
    vec
}

#[cfg(test)]
mod tests {
    // Tests are in the tests/factorize.rs file
}
