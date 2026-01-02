use std::sync::Arc;
use tensor4all_index::index::{Index, DynId, NoSymmSpace, Symmetry};
use tensor4all_index::tagset::DefaultTagSet;
use tensor4all_tensor::{Storage, TensorDynLen};
use tensor4all_tensor::storage::DenseStorageF64;
use mdarray::{Dense, Slice, tensor};
use mdarray_linalg::svd::{SVD, SVDDecomp, SVDError as MdarraySvdError};
use mdarray_linalg_faer::Faer;
use num_complex::Complex64;
use thiserror::Error;

/// Error type for SVD operations in tensor4all-linalg.
#[derive(Debug, Error)]
pub enum SvdError {
    #[error("Tensor must have rank 2, got rank {0}")]
    InvalidRank(usize),

    #[error("Tensor storage must be DenseF64, got {0:?}")]
    UnsupportedStorage(String),

    #[error("mdarray-linalg SVD error: {0}")]
    BackendError(#[from] MdarraySvdError),
}

/// Compute SVD decomposition of a rank-2 tensor, returning (U, S, V).
///
/// This function mimics ITensor's SVD API, returning U, S, and V (not Vt).
/// The input tensor must be rank-2 with DenseF64 storage.
///
/// # Arguments
/// * `t` - Input tensor of rank 2 with DenseF64 storage
///
/// # Returns
/// A tuple `(U, S, V)` where:
/// - `U` is an m×k tensor with indices `[left_index, bond_index]`
/// - `S` is a k×k diagonal tensor with indices `[bond_index, bond_index]`
/// - `V` is an n×k tensor with indices `[right_index, bond_index]`
/// where `k = min(m, n)` is the bond dimension.
///
/// # Errors
/// Returns `SvdError` if the tensor is not rank-2, storage is not DenseF64,
/// or if the SVD computation fails.
pub fn svd<Id, Symm>(
    t: &TensorDynLen<Id, f64, Symm>,
) -> Result<
    (
        TensorDynLen<Id, f64, Symm>,
        TensorDynLen<Id, f64, Symm>,
        TensorDynLen<Id, f64, Symm>,
    ),
    SvdError,
>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
{
    // Validate rank
    if t.dims.len() != 2 {
        return Err(SvdError::InvalidRank(t.dims.len()));
    }

    // Validate storage type
    let dense_storage = match t.storage.as_ref() {
        Storage::DenseF64(ds) => ds,
        _ => {
            return Err(SvdError::UnsupportedStorage(format!(
                "{:?}",
                t.storage
            )));
        }
    };

    let m = t.dims[0];
    let n = t.dims[1];
    let k = m.min(n);

    // Get original indices
    let left_index = t.indices[0].clone();
    let right_index = t.indices[1].clone();

    // Create mdarray tensor from dense storage data
    // SVD destroys the input, so we need to clone the data
    let a_data = dense_storage.as_slice().to_vec();
    let mut a_tensor = tensor![[0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            a_tensor[[i, j]] = a_data[i * n + j];
        }
    }

    // Call SVD using faer backend
    let bd = Faer;
    let a_slice: &mut Slice<f64, (usize, usize), Dense> = a_tensor.as_mut();
    let SVDDecomp { s, u, vt } = bd.svd(a_slice)?;

    // Extract singular values.
    //
    // NOTE:
    // `mdarray-linalg-faer` writes singular values into a diagonal view created by
    // `into_faer_diag_mut`, which (by design) treats the **first row** as the
    // singular-value buffer (LAPACK-style convention). Therefore, the values live at
    // `s[0, i]`, not necessarily at `s[i, i]`.
    let mut s_vec = Vec::with_capacity(k);
    for i in 0..k {
        s_vec.push(s[[0, i]]);
    }

    // Convert U from m×m to m×k (take first k columns)
    let mut u_vec = Vec::with_capacity(m * k);
    for i in 0..m {
        for j in 0..k {
            u_vec.push(u[[i, j]]);
        }
    }

    // Convert backend `vt` (V^T) to V (n×k).
    //
    // `mdarray-linalg` returns `vt` as (conceptually) V^T. We want V (not Vt), so we take the
    // first k rows of V^T (which correspond to the first k columns of V) and transpose.
    let mut vt_vec = Vec::with_capacity(k * n);
    for i in 0..k {
        for j in 0..n {
            vt_vec.push(vt[[i, j]]);
        }
    }

    let mut v_vec = Vec::with_capacity(n * k);
    for j in 0..n {
        for i in 0..k {
            v_vec.push(vt_vec[i * n + j]);
        }
    }

    // Create bond index with "Link" tag
    // Convert from DynId to Id type
    let bond_dyn_id = DynId(tensor4all_index::index::generate_id());
    let bond_id: Id = bond_dyn_id.into();
    let bond_symm: Symm = NoSymmSpace::new(k).into();
    let mut bond_index: Index<Id, Symm, DefaultTagSet> = Index::new(bond_id, bond_symm);
    bond_index.tags_mut().add_tag("Link").map_err(|_| {
        SvdError::UnsupportedStorage("Failed to add Link tag".to_string())
    })?;

    // Create U tensor: [left_index, bond_index]
    let u_indices = vec![left_index.clone(), bond_index.clone()];
    let u_dims = vec![m, k];
    let u_storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(u_vec)));
    let u = TensorDynLen::new(u_indices, u_dims, u_storage);

    // Create S tensor: [bond_index, bond_index] (diagonal)
    let s_indices = vec![bond_index.clone(), bond_index.clone()];
    let s_storage = Arc::new(Storage::new_diag_f64(s_vec));
    let s = TensorDynLen::new(s_indices, vec![k, k], s_storage);

    // Create V tensor: [right_index, bond_index]
    let v_indices = vec![right_index.clone(), bond_index.clone()];
    let v_dims = vec![n, k];
    let v_storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(v_vec)));
    let v = TensorDynLen::new(v_indices, v_dims, v_storage);

    Ok((u, s, v))
}

/// Compute SVD decomposition of a rank-2 complex tensor, returning (U, S, V).
///
/// For complex-valued matrices, the mathematical convention is:
/// \[ A = U * Σ * V^H \]
/// where \(V^H\) is the conjugate-transpose of \(V\).
///
/// `mdarray-linalg` returns `vt` (conceptually \(V^T\) / \(V^H\) depending on scalar type),
/// and we return **V** (not Vt), so we build V by conjugate-transposing the leading k rows.
pub fn svd_c64<Id, Symm>(
    t: &TensorDynLen<Id, Complex64, Symm>,
) -> Result<
    (
        TensorDynLen<Id, Complex64, Symm>,
        TensorDynLen<Id, Complex64, Symm>,
        TensorDynLen<Id, Complex64, Symm>,
    ),
    SvdError,
>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
{
    if t.dims.len() != 2 {
        return Err(SvdError::InvalidRank(t.dims.len()));
    }

    let dense_storage = match t.storage.as_ref() {
        Storage::DenseC64(ds) => ds,
        _ => {
            return Err(SvdError::UnsupportedStorage(format!(
                "{:?}",
                t.storage
            )));
        }
    };

    let m = t.dims[0];
    let n = t.dims[1];
    let k = m.min(n);

    let left_index = t.indices[0].clone();
    let right_index = t.indices[1].clone();

    // Build mdarray tensor (SVD destroys input)
    let a_data = dense_storage.as_slice().to_vec();
    let mut a_tensor = tensor![[Complex64::new(0.0, 0.0); n]; m];
    for i in 0..m {
        for j in 0..n {
            a_tensor[[i, j]] = a_data[i * n + j];
        }
    }

    let bd = Faer;
    let a_slice: &mut Slice<Complex64, (usize, usize), Dense> = a_tensor.as_mut();
    let SVDDecomp { s, u, vt } = bd.svd(a_slice)?;

    // Singular values live in the first row (see `into_faer_diag_mut`).
    let mut s_vec = Vec::with_capacity(k);
    for i in 0..k {
        s_vec.push(s[[0, i]]);
    }

    // U is m×m; take first k columns -> m×k (row-major)
    let mut u_vec = Vec::with_capacity(m * k);
    for i in 0..m {
        for j in 0..k {
            u_vec.push(u[[i, j]]);
        }
    }

    // vt is n×n; take first k rows (k×n) and conjugate-transpose to get V (n×k)
    let mut vt_vec = Vec::with_capacity(k * n);
    for i in 0..k {
        for j in 0..n {
            vt_vec.push(vt[[i, j]]);
        }
    }

    let mut v_vec = Vec::with_capacity(n * k);
    for j in 0..n {
        for i in 0..k {
            v_vec.push(vt_vec[i * n + j].conj());
        }
    }

    // Bond index
    let bond_dyn_id = DynId(tensor4all_index::index::generate_id());
    let bond_id: Id = bond_dyn_id.into();
    let bond_symm: Symm = NoSymmSpace::new(k).into();
    let mut bond_index: Index<Id, Symm, DefaultTagSet> = Index::new(bond_id, bond_symm);
    bond_index.tags_mut().add_tag("Link").map_err(|_| {
        SvdError::UnsupportedStorage("Failed to add Link tag".to_string())
    })?;

    let u_indices = vec![left_index.clone(), bond_index.clone()];
    let u_storage = Arc::new(Storage::DenseC64(tensor4all_tensor::storage::DenseStorageC64::from_vec(u_vec)));
    let u_t = TensorDynLen::new(u_indices, vec![m, k], u_storage);

    let s_indices = vec![bond_index.clone(), bond_index.clone()];
    let s_storage = Arc::new(Storage::new_diag_c64(s_vec));
    let s_t = TensorDynLen::new(s_indices, vec![k, k], s_storage);

    let v_indices = vec![right_index.clone(), bond_index.clone()];
    let v_storage = Arc::new(Storage::DenseC64(tensor4all_tensor::storage::DenseStorageC64::from_vec(v_vec)));
    let v_t = TensorDynLen::new(v_indices, vec![n, k], v_storage);

    Ok((u_t, s_t, v_t))
}

