//! Compression algorithms for tensor trains

use crate::error::Result;
use crate::tensortrain::TensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{tensor3_zeros, Tensor3, Tensor3Ops};
use tenferro_algebra::Scalar as TfScalar;
use tenferro_linalg::LinalgScalar;
use tenferro_tensor::{KeepCountScalar, MemoryOrder, Tensor as TypedTensor};
use tensor4all_tcicore::matrix::{mat_mul, ncols, nrows, zeros, Matrix};
use tensor4all_tcicore::Scalar;
use tensor4all_tcicore::{rrlu, AbstractMatrixCI, MatrixLUCI, RrLUOptions};
use tensor4all_tensorbackend::BackendLinalgScalar;

/// Compression method for tensor trains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionMethod {
    /// LU decomposition (rank-revealing)
    #[default]
    LU,
    /// Cross interpolation based
    CI,
    /// SVD compression (currently unimplemented and returns an error)
    SVD,
}

/// Options for compression
#[derive(Debug, Clone)]
pub struct CompressionOptions {
    /// Compression method
    pub method: CompressionMethod,
    /// Tolerance for truncation (relative)
    pub tolerance: f64,
    /// Maximum bond dimension
    pub max_bond_dim: usize,
    /// Whether to normalize the error
    pub normalize_error: bool,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            method: CompressionMethod::LU,
            tolerance: 1e-12,
            max_bond_dim: usize::MAX,
            normalize_error: true,
        }
    }
}

/// Convert Tensor3 to Matrix for factorization (left matrix view)
fn tensor3_to_left_matrix<T: TTScalar + Scalar + Default>(tensor: &Tensor3<T>) -> Matrix<T> {
    let left_dim = tensor.left_dim();
    let site_dim = tensor.site_dim();
    let right_dim = tensor.right_dim();
    let rows = left_dim * site_dim;
    let cols = right_dim;

    let mut mat = zeros(rows, cols);
    for l in 0..left_dim {
        for s in 0..site_dim {
            for r in 0..right_dim {
                mat[[l * site_dim + s, r]] = *tensor.get3(l, s, r);
            }
        }
    }
    mat
}

/// Convert Tensor3 to Matrix for factorization (right matrix view)
fn tensor3_to_right_matrix<T: TTScalar + Scalar + Default>(tensor: &Tensor3<T>) -> Matrix<T> {
    let left_dim = tensor.left_dim();
    let site_dim = tensor.site_dim();
    let right_dim = tensor.right_dim();
    let rows = left_dim;
    let cols = site_dim * right_dim;

    let mut mat = zeros(rows, cols);
    for l in 0..left_dim {
        for s in 0..site_dim {
            for r in 0..right_dim {
                mat[[l, s * right_dim + r]] = *tensor.get3(l, s, r);
            }
        }
    }
    mat
}

/// Factorize a matrix into left and right factors
fn factorize<T>(
    matrix: &Matrix<T>,
    method: CompressionMethod,
    tolerance: f64,
    max_bond_dim: usize,
    left_orthogonal: bool,
) -> crate::error::Result<(Matrix<T>, Matrix<T>, usize)>
where
    T: TTScalar + Scalar + tensor4all_tcicore::MatrixLuciScalar,
    tensor4all_tcicore::DenseFaerLuKernel: tensor4all_tcicore::PivotKernel<T>,
{
    let reltol = if tolerance > 0.0 { tolerance } else { 1e-14 };
    let abstol = 0.0;

    let options = RrLUOptions {
        max_rank: max_bond_dim,
        rel_tol: reltol,
        abs_tol: abstol,
        left_orthogonal,
    };

    match method {
        CompressionMethod::LU => {
            let lu = rrlu(matrix, Some(options))?;
            let left = lu.left(true); // permuted
            let right = lu.right(true); // permuted
            let npivots = lu.npivots();
            Ok((left, right, npivots))
        }
        CompressionMethod::CI => {
            let luci = MatrixLUCI::from_matrix(matrix, Some(options))?;
            let left = luci.left();
            let right = luci.right();
            let npivots = luci.rank();
            Ok((left, right, npivots))
        }
        CompressionMethod::SVD => {
            // SVD compression requires additional trait bounds. We use type-erased dispatch
            // for the supported scalar types (f64, Complex64).
            svd_dispatch(matrix, tolerance, max_bond_dim, left_orthogonal)
        }
    }
}

/// Trait bounds for SVD-compatible scalars in TT compression
trait SVDCompressScalar:
    TTScalar
    + Scalar
    + Default
    + Copy
    + BackendLinalgScalar
    + TfScalar
    + num_complex::ComplexFloat
    + 'static
{
    fn sv_to_f64(real: <Self as LinalgScalar>::Real) -> f64;
    fn from_sv(real: <Self as LinalgScalar>::Real) -> Self;
}

impl SVDCompressScalar for f64 {
    fn sv_to_f64(real: <Self as LinalgScalar>::Real) -> f64 {
        real
    }
    fn from_sv(real: <Self as LinalgScalar>::Real) -> Self {
        real
    }
}

impl SVDCompressScalar for num_complex::Complex64 {
    fn sv_to_f64(real: <Self as LinalgScalar>::Real) -> f64 {
        real
    }
    fn from_sv(real: <Self as LinalgScalar>::Real) -> Self {
        num_complex::Complex64::new(real, 0.0)
    }
}

/// SVD dispatch: convert generic Matrix<T> to concrete type and call factorize_svd.
///
/// This is necessary because SVD requires additional trait bounds (LinalgScalar, etc.)
/// that aren't available on the generic `T: TTScalar + Scalar`.
fn svd_dispatch<T: TTScalar + Scalar>(
    matrix: &Matrix<T>,
    tolerance: f64,
    max_bond_dim: usize,
    left_orthogonal: bool,
) -> crate::error::Result<(Matrix<T>, Matrix<T>, usize)> {
    use std::any::Any;

    let m = nrows(matrix);
    let n = ncols(matrix);

    // Try f64
    if let Some(mat_f64) = (matrix as &dyn Any).downcast_ref::<Matrix<f64>>() {
        let (l, r, rank) = factorize_svd(mat_f64, tolerance, max_bond_dim, left_orthogonal)?;
        // Safety: T is f64 in this branch
        let left = unsafe { std::mem::transmute::<Matrix<f64>, Matrix<T>>(l) };
        let right = unsafe { std::mem::transmute::<Matrix<f64>, Matrix<T>>(r) };
        return Ok((left, right, rank));
    }

    // Try Complex64
    if let Some(mat_c64) = (matrix as &dyn Any).downcast_ref::<Matrix<num_complex::Complex64>>() {
        let (l, r, rank) = factorize_svd(mat_c64, tolerance, max_bond_dim, left_orthogonal)?;
        let left = unsafe { std::mem::transmute::<Matrix<num_complex::Complex64>, Matrix<T>>(l) };
        let right = unsafe { std::mem::transmute::<Matrix<num_complex::Complex64>, Matrix<T>>(r) };
        return Ok((left, right, rank));
    }

    Err(crate::error::TensorTrainError::InvalidOperation {
        message: format!(
            "SVD compression not supported for this scalar type (matrix {}x{})",
            m, n
        ),
    })
}

/// Helper: extract row-major data from a TypedTensor
fn typed_tensor_row_major<T: TfScalar + Copy>(
    tensor: &TypedTensor<T>,
    name: &str,
) -> crate::error::Result<Vec<T>> {
    let row_major = tensor.contiguous(MemoryOrder::RowMajor);
    let data = row_major.buffer().as_slice().ok_or_else(|| {
        crate::error::TensorTrainError::InvalidOperation {
            message: format!("SVD {name}: non-CPU tensor"),
        }
    })?;
    Ok(data.to_vec())
}

/// SVD-based factorization for TT compression
fn factorize_svd<T: SVDCompressScalar>(
    matrix: &Matrix<T>,
    tolerance: f64,
    max_bond_dim: usize,
    left_orthogonal: bool,
) -> crate::error::Result<(Matrix<T>, Matrix<T>, usize)>
where
    <T as LinalgScalar>::Real: KeepCountScalar,
{
    let m = nrows(matrix);
    let n = ncols(matrix);

    if m == 0 || n == 0 {
        return Err(crate::error::TensorTrainError::InvalidOperation {
            message: "Cannot factorize empty matrix".to_string(),
        });
    }

    // Convert matrixci::Matrix to TypedTensor (row-major) for SVD
    let mut data = vec![T::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            data[i * n + j] = matrix[[i, j]];
        }
    }
    let a_tensor =
        TypedTensor::<T>::from_slice(&data, &[m, n], MemoryOrder::RowMajor).map_err(|e| {
            crate::error::TensorTrainError::InvalidOperation {
                message: format!("SVD tensor creation failed: {e}"),
            }
        })?;

    let svd_result = tensor4all_tensorbackend::svd_backend(&a_tensor).map_err(|e| {
        crate::error::TensorTrainError::InvalidOperation {
            message: format!("SVD computation failed: {e:?}"),
        }
    })?;

    // Extract U, S, Vt as row-major vectors
    let u_data = typed_tensor_row_major(&svd_result.u, "U")?;
    let u_cols = svd_result.u.dims()[1];
    let s_data: Vec<<T as LinalgScalar>::Real> = typed_tensor_row_major(&svd_result.s, "S")?;
    let vt_data = typed_tensor_row_major(&svd_result.vt, "Vt")?;
    let vt_cols = svd_result.vt.dims()[1];

    // Determine rank based on tolerance and max_bond_dim
    let min_dim = m.min(n);
    let s_max = if !s_data.is_empty() {
        T::sv_to_f64(s_data[0])
    } else {
        0.0
    };

    let mut rank = 0;
    for sv_raw in &s_data[..min_dim] {
        if rank >= max_bond_dim {
            break;
        }
        let sv = T::sv_to_f64(*sv_raw);
        if sv < tolerance * s_max {
            break;
        }
        rank += 1;
    }
    rank = rank.max(1);

    // Build result matrices
    let mut left = zeros(m, rank);
    let mut right = zeros(rank, n);

    if left_orthogonal {
        // Left = U[:, :rank], Right = diag(S[:rank]) * Vt[:rank, :]
        for i in 0..m {
            for j in 0..rank {
                left[[i, j]] = u_data[i * u_cols + j];
            }
        }
        for i in 0..rank {
            let sv = T::from_sv(s_data[i]);
            for j in 0..n {
                right[[i, j]] = sv * vt_data[i * vt_cols + j];
            }
        }
    } else {
        // Left = U[:, :rank] * diag(S[:rank]), Right = Vt[:rank, :]
        for i in 0..m {
            for j in 0..rank {
                let sv = T::from_sv(s_data[j]);
                left[[i, j]] = u_data[i * u_cols + j] * sv;
            }
        }
        for i in 0..rank {
            for j in 0..n {
                right[[i, j]] = vt_data[i * vt_cols + j];
            }
        }
    }

    Ok((left, right, rank))
}

impl<T: TTScalar + Scalar + Default> TensorTrain<T> {
    /// Compress the tensor train in-place using the specified method
    ///
    /// This performs a two-sweep compression:
    /// 1. Left-to-right sweep with left-orthogonal factorization (no truncation)
    /// 2. Right-to-left sweep with truncation
    pub fn compress(&mut self, options: &CompressionOptions) -> Result<()>
    where
        T: tensor4all_tcicore::MatrixLuciScalar,
        tensor4all_tcicore::DenseFaerLuKernel: tensor4all_tcicore::PivotKernel<T>,
    {
        let n = self.len();
        if n <= 1 {
            return Ok(());
        }

        let tensors = self.site_tensors_mut();

        // Left-to-right sweep: make left-orthogonal without truncation
        for ell in 0..n - 1 {
            let left_dim = tensors[ell].left_dim();
            let site_dim = tensors[ell].site_dim();

            // Reshape to matrix: (left_dim * site_dim, right_dim)
            let mat = tensor3_to_left_matrix(&tensors[ell]);

            // Factorize without truncation
            let (left_factor, right_factor, new_bond_dim) = factorize(
                &mat,
                options.method,
                0.0,        // No truncation in left sweep
                usize::MAX, // No max bond dim in left sweep
                true,       // left orthogonal
            )?;

            // Update current tensor
            let mut new_tensor = tensor3_zeros(left_dim, site_dim, new_bond_dim);
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..new_bond_dim {
                        let row = l * site_dim + s;
                        if row < nrows(&left_factor) && r < ncols(&left_factor) {
                            new_tensor.set3(l, s, r, left_factor[[row, r]]);
                        }
                    }
                }
            }
            tensors[ell] = new_tensor;

            // Contract right_factor with next tensor
            let next_site_dim = tensors[ell + 1].site_dim();
            let next_right_dim = tensors[ell + 1].right_dim();

            // Build next tensor as matrix (old_left_dim, site_dim * right_dim)
            let next_mat = tensor3_to_right_matrix(&tensors[ell + 1]);

            // Multiply: right_factor * next_mat
            let contracted = mat_mul(&right_factor, &next_mat);

            // Update next tensor
            let mut new_next_tensor = tensor3_zeros(new_bond_dim, next_site_dim, next_right_dim);
            for l in 0..new_bond_dim {
                for s in 0..next_site_dim {
                    for r in 0..next_right_dim {
                        new_next_tensor.set3(l, s, r, contracted[[l, s * next_right_dim + r]]);
                    }
                }
            }
            tensors[ell + 1] = new_next_tensor;
        }

        // Right-to-left sweep: truncate
        for ell in (1..n).rev() {
            let site_dim = tensors[ell].site_dim();
            let right_dim = tensors[ell].right_dim();

            // Reshape to matrix: (left_dim, site_dim * right_dim)
            let mat = tensor3_to_right_matrix(&tensors[ell]);

            // Factorize with truncation
            let (left_factor, right_factor, new_bond_dim) = factorize(
                &mat,
                options.method,
                options.tolerance,
                options.max_bond_dim,
                false, // right orthogonal
            )?;

            // Update current tensor from right_factor
            let mut new_tensor = tensor3_zeros(new_bond_dim, site_dim, right_dim);
            for l in 0..new_bond_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        new_tensor.set3(l, s, r, right_factor[[l, s * right_dim + r]]);
                    }
                }
            }
            tensors[ell] = new_tensor;

            // Contract previous tensor with left_factor
            let prev_left_dim = tensors[ell - 1].left_dim();
            let prev_site_dim = tensors[ell - 1].site_dim();

            // Build prev tensor as matrix (left_dim * site_dim, old_right_dim)
            let prev_mat = tensor3_to_left_matrix(&tensors[ell - 1]);

            // Multiply: prev_mat * left_factor
            let contracted = mat_mul(&prev_mat, &left_factor);

            // Update prev tensor
            let mut new_prev_tensor = tensor3_zeros(prev_left_dim, prev_site_dim, new_bond_dim);
            for l in 0..prev_left_dim {
                for s in 0..prev_site_dim {
                    for r in 0..new_bond_dim {
                        new_prev_tensor.set3(l, s, r, contracted[[l * prev_site_dim + s, r]]);
                    }
                }
            }
            tensors[ell - 1] = new_prev_tensor;
        }

        Ok(())
    }

    /// Create a compressed copy of the tensor train
    pub fn compressed(&self, options: &CompressionOptions) -> Result<Self>
    where
        T: tensor4all_tcicore::MatrixLuciScalar,
        tensor4all_tcicore::DenseFaerLuKernel: tensor4all_tcicore::PivotKernel<T>,
    {
        let mut result = self.clone();
        result.compress(options)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
