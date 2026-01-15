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
use crate::{unfold_split, Storage, StorageScalar, TensorDynLen};
use matrixci::{rrlu, AbstractMatrixCI, MatrixLUCI, RrLUOptions, Scalar as MatrixScalar};
use num_complex::{Complex64, ComplexFloat};

use crate::qr::{qr_with, QrOptions};
use crate::svd::{svd_with, SvdOptions};
use tensor4all_tensorbackend::faer_traits::ComplexField;

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
    // Dispatch based on storage type
    match t.storage().as_ref() {
        Storage::DenseF64(_) => factorize_impl::<f64>(t, left_inds, options),
        Storage::DenseC64(_) => factorize_impl::<Complex64>(t, left_inds, options),
        Storage::DiagF64(_) | Storage::DiagC64(_) => Err(FactorizeError::UnsupportedStorage(
            "Diagonal storage not supported for factorize",
        )),
    }
}

/// Internal implementation with scalar type.
fn factorize_impl<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: StorageScalar
        + ComplexFloat
        + ComplexField
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    match options.alg {
        FactorizeAlg::SVD => factorize_svd::<T>(t, left_inds, options),
        FactorizeAlg::QR => factorize_qr::<T>(t, left_inds, options),
        FactorizeAlg::LU => factorize_lu::<T>(t, left_inds, options),
        FactorizeAlg::CI => factorize_ci::<T>(t, left_inds, options),
    }
}

/// SVD factorization implementation.
fn factorize_svd<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: StorageScalar
        + ComplexFloat
        + ComplexField
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    let svd_options = SvdOptions { rtol: options.rtol };

    let (u, s, v) = svd_with::<T>(t, left_inds, &svd_options)?;

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

    match options.canonical {
        Canonical::Left => {
            // L = U, R = S * V
            let right_contracted = s.contract(&v);
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
            // L = U * S, R = V
            let left_contracted = u.contract(&s);
            let left = left_contracted.replaceind(&sim_bond_index, &bond_index);
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
fn factorize_qr<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
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

    let (q, r) = qr_with::<T>(t, left_inds, &qr_options)?;

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
fn factorize_lu<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: StorageScalar
        + ComplexFloat
        + ComplexField
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Unfold tensor into matrix
    let (a_tensor, _, m, n, left_indices, right_indices) = unfold_split::<T>(t, left_inds)
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
    let bond_index = DynIndex::new_bond(rank)
        .map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))?;

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
fn factorize_ci<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: StorageScalar
        + ComplexFloat
        + ComplexField
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Unfold tensor into matrix
    let (a_tensor, _, m, n, left_indices, right_indices) = unfold_split::<T>(t, left_inds)
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
    let bond_index = DynIndex::new_bond(rank)
        .map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))?;

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
fn extract_singular_values(s: &TensorDynLen) -> Vec<f64> {
    match s.storage().as_ref() {
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
    tensor: &tensor4all_tensorbackend::mdarray::DTensor<T, 2>,
    m: usize,
    n: usize,
) -> matrixci::Matrix<T>
where
    T: MatrixScalar + Clone,
{
    let mut matrix = matrixci::util::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            matrix[[i, j]] = tensor[[i, j]];
        }
    }
    matrix
}

/// Convert Matrix to Vec for storage.
fn matrix_to_vec<T>(matrix: &matrixci::Matrix<T>) -> Vec<T>
where
    T: Clone,
{
    let m = matrixci::util::nrows(matrix);
    let n = matrixci::util::ncols(matrix);
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
