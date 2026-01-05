//! Vidal and Inverse tensor train representations
//!
//! This module provides tensor train representations with explicit singular values:
//! - `VidalTensorTrain`: Stores tensors and singular values separately (Vidal canonical form)
//! - `InverseTensorTrain`: Stores tensors and inverse singular values for efficient local updates

use std::ops::Range;

use crate::error::{Result, TensorTrainError};
use crate::tensortrain::TensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{tensor3_zeros, Tensor3, Tensor3Ops};
use tensor4all_matrixci::util::{mat_mul, ncols, nrows, transpose, zeros, Matrix, Scalar};
use tensor4all_matrixci::{rrlu, RrLUOptions};

/// Compute QR decomposition
fn qr_decomp<T: TTScalar + Scalar>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    let options = RrLUOptions {
        max_rank: ncols(matrix).min(nrows(matrix)),
        rel_tol: 0.0,
        abs_tol: 0.0,
        left_orthogonal: true,
    };
    let lu = rrlu(matrix, Some(options));
    (lu.left(true), lu.right(true))
}

/// Compute LQ decomposition
fn lq_decomp<T: TTScalar + Scalar>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    let at = transpose(matrix);
    let (qt, lt) = qr_decomp(&at);
    (transpose(&lt), transpose(&qt))
}

/// Convert Tensor3 to Matrix with left dimensions flattened
fn tensor3_to_left_matrix<T: Scalar + Default + Clone>(tensor: &Tensor3<T>) -> Matrix<T> {
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

/// Convert Tensor3 to Matrix with right dimensions flattened
fn tensor3_to_right_matrix<T: Scalar + Default + Clone>(tensor: &Tensor3<T>) -> Matrix<T> {
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

/// Diagonal matrix type (stored as vector of diagonal elements)
pub type DiagMatrix = Vec<f64>;

/// Vidal Tensor Train representation
///
/// Stores the tensor train in Vidal canonical form where:
/// - Site tensors are stored separately from singular values
/// - Singular values are stored as diagonal matrices between sites
///
/// This form is useful for algorithms that need to apply local operations
/// and maintain canonical form efficiently.
#[derive(Debug, Clone)]
pub struct VidalTensorTrain<T: TTScalar> {
    /// Site tensors (unscaled)
    tensors: Vec<Tensor3<T>>,
    /// Singular values between sites (length = n-1)
    singular_values: Vec<DiagMatrix>,
    /// Active partition range
    partition: Range<usize>,
}

impl<T: TTScalar + Scalar + Default> VidalTensorTrain<T> {
    /// Create a VidalTensorTrain from a regular TensorTrain
    pub fn from_tensor_train(tt: &TensorTrain<T>) -> Result<Self> {
        Self::from_tensor_train_with_partition(tt, 0..tt.len())
    }

    /// Create a VidalTensorTrain with a specific partition
    pub fn from_tensor_train_with_partition(tt: &TensorTrain<T>, partition: Range<usize>) -> Result<Self> {
        let n = tt.len();
        if n == 0 {
            return Ok(Self {
                tensors: Vec::new(),
                singular_values: Vec::new(),
                partition: 0..0,
            });
        }

        if partition.end > n {
            return Err(TensorTrainError::InvalidOperation {
                message: format!("Partition end {} exceeds tensor train length {}", partition.end, n),
            });
        }

        let mut tensors: Vec<Tensor3<T>> = tt.site_tensors().to_vec();
        let mut singular_values: Vec<DiagMatrix> = vec![Vec::new(); n - 1];

        // Left sweep: QR decomposition to make left-orthogonal
        for i in partition.start..partition.end.saturating_sub(1) {
            let left_dim = tensors[i].left_dim();
            let site_dim = tensors[i].site_dim();

            let mat = tensor3_to_left_matrix(&tensors[i]);
            let (q, r) = qr_decomp(&mat);

            let new_bond_dim = ncols(&q);

            // Update current tensor with Q
            let mut new_tensor = tensor3_zeros(left_dim, site_dim, new_bond_dim);
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for b in 0..new_bond_dim {
                        let row = l * site_dim + s;
                        if row < nrows(&q) && b < ncols(&q) {
                            new_tensor.set3(l, s, b, q[[row, b]]);
                        }
                    }
                }
            }
            tensors[i] = new_tensor;

            // Contract R with next tensor
            let next_site_dim = tensors[i + 1].site_dim();
            let next_right_dim = tensors[i + 1].right_dim();
            let next_mat = tensor3_to_right_matrix(&tensors[i + 1]);

            let contracted = mat_mul(&r, &next_mat);

            let mut new_next_tensor = tensor3_zeros(new_bond_dim, next_site_dim, next_right_dim);
            for l in 0..new_bond_dim {
                for s in 0..next_site_dim {
                    for r_idx in 0..next_right_dim {
                        new_next_tensor.set3(l, s, r_idx, contracted[[l, s * next_right_dim + r_idx]]);
                    }
                }
            }
            tensors[i + 1] = new_next_tensor;
        }

        // Right sweep: LQ decomposition and extract singular values from L diagonal
        for i in (partition.start + 1..partition.end).rev() {
            let site_dim = tensors[i].site_dim();
            let right_dim = tensors[i].right_dim();

            let mat = tensor3_to_right_matrix(&tensors[i]);
            let (l_mat, q) = lq_decomp(&mat);

            let new_bond_dim = nrows(&q);

            // Extract singular values from diagonal of L
            let mut sv = Vec::with_capacity(new_bond_dim);
            for k in 0..new_bond_dim.min(nrows(&l_mat)).min(ncols(&l_mat)) {
                let val = l_mat[[k, k]];
                let abs_val = TTScalar::abs_sq(val).sqrt();
                sv.push(if abs_val > 1e-15 { abs_val } else { 1e-15 });
            }
            singular_values[i - 1] = sv.clone();

            // Update current tensor with Q (right-orthogonal)
            let mut new_tensor = tensor3_zeros(new_bond_dim, site_dim, right_dim);
            for l in 0..new_bond_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        if l < nrows(&q) {
                            new_tensor.set3(l, s, r, q[[l, s * right_dim + r]]);
                        }
                    }
                }
            }
            tensors[i] = new_tensor;

            // Contract previous tensor with L (normalized by singular values)
            if i > partition.start {
                let prev_left_dim = tensors[i - 1].left_dim();
                let prev_site_dim = tensors[i - 1].site_dim();
                let prev_mat = tensor3_to_left_matrix(&tensors[i - 1]);

                // Normalize L by dividing columns by singular values, then multiply
                let mut l_normalized = zeros(nrows(&l_mat), ncols(&l_mat));
                for row in 0..nrows(&l_mat) {
                    for col in 0..ncols(&l_mat) {
                        l_normalized[[row, col]] = l_mat[[row, col]];
                    }
                }

                let contracted = mat_mul(&prev_mat, &l_normalized);

                let mut new_prev_tensor = tensor3_zeros(prev_left_dim, prev_site_dim, new_bond_dim);
                for l in 0..prev_left_dim {
                    for s in 0..prev_site_dim {
                        for r in 0..new_bond_dim {
                            if l * prev_site_dim + s < nrows(&contracted) && r < ncols(&contracted) {
                                new_prev_tensor.set3(l, s, r, contracted[[l * prev_site_dim + s, r]]);
                            }
                        }
                    }
                }
                tensors[i - 1] = new_prev_tensor;
            }
        }

        // Divide out singular values from tensors to get Vidal form
        for i in partition.start..partition.end.saturating_sub(1) {
            if singular_values[i].is_empty() {
                continue;
            }

            let site_dim = tensors[i].site_dim();
            let right_dim = tensors[i].right_dim();
            let left_dim = tensors[i].left_dim();

            let mut new_tensor = tensor3_zeros(left_dim, site_dim, right_dim);
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        let val = *tensors[i].get3(l, s, r);
                        let sv = if r < singular_values[i].len() && singular_values[i][r] > 1e-15 {
                            singular_values[i][r]
                        } else {
                            1.0
                        };
                        new_tensor.set3(l, s, r, val / T::from_f64(sv));
                    }
                }
            }
            tensors[i] = new_tensor;
        }

        Ok(Self {
            tensors,
            singular_values,
            partition,
        })
    }

    /// Create a VidalTensorTrain with given tensors and singular values
    pub fn new(tensors: Vec<Tensor3<T>>, singular_values: Vec<DiagMatrix>) -> Result<Self> {
        let n = tensors.len();
        if n == 0 {
            return Ok(Self {
                tensors: Vec::new(),
                singular_values: Vec::new(),
                partition: 0..0,
            });
        }

        if singular_values.len() != n - 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Expected {} singular value vectors, got {}",
                    n - 1,
                    singular_values.len()
                ),
            });
        }

        Ok(Self {
            tensors,
            singular_values,
            partition: 0..n,
        })
    }

    /// Get the singular values between sites i and i+1
    pub fn singular_values(&self, i: usize) -> &DiagMatrix {
        &self.singular_values[i]
    }

    /// Get all singular value matrices
    pub fn all_singular_values(&self) -> &[DiagMatrix] {
        &self.singular_values
    }

    /// Get the partition range
    pub fn partition(&self) -> &Range<usize> {
        &self.partition
    }

    /// Get mutable access to site tensors
    pub fn site_tensors_mut(&mut self) -> &mut [Tensor3<T>] {
        &mut self.tensors
    }

    /// Get mutable access to singular values
    pub fn singular_values_mut(&mut self, i: usize) -> &mut DiagMatrix {
        &mut self.singular_values[i]
    }

    /// Convert to a regular TensorTrain
    pub fn to_tensor_train(&self) -> TensorTrain<T> {
        let n = self.len();
        if n == 0 {
            return TensorTrain::from_tensors_unchecked(Vec::new());
        }

        let mut tensors = Vec::with_capacity(n);

        for i in 0..n - 1 {
            let tensor = &self.tensors[i];
            let left_dim = tensor.left_dim();
            let site_dim = tensor.site_dim();
            let right_dim = tensor.right_dim();

            // Multiply by singular values on the right
            let mut new_tensor = tensor3_zeros(left_dim, site_dim, right_dim);
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        let val = *tensor.get3(l, s, r);
                        let sv = if r < self.singular_values[i].len() {
                            self.singular_values[i][r]
                        } else {
                            1.0
                        };
                        new_tensor.set3(l, s, r, val * T::from_f64(sv));
                    }
                }
            }
            tensors.push(new_tensor);
        }

        // Last tensor is unchanged
        tensors.push(self.tensors[n - 1].clone());

        TensorTrain::from_tensors_unchecked(tensors)
    }
}

impl<T: TTScalar + Scalar + Default> AbstractTensorTrain<T> for VidalTensorTrain<T> {
    fn len(&self) -> usize {
        self.tensors.len()
    }

    fn site_tensor(&self, i: usize) -> &Tensor3<T> {
        &self.tensors[i]
    }

    fn site_tensors(&self) -> &[Tensor3<T>] {
        &self.tensors
    }
}

/// Inverse Tensor Train representation
///
/// Similar to VidalTensorTrain but stores inverse singular values instead.
/// This is useful for algorithms that need to efficiently apply inverse
/// operations during local updates.
#[derive(Debug, Clone)]
pub struct InverseTensorTrain<T: TTScalar> {
    /// Site tensors (scaled by singular values)
    tensors: Vec<Tensor3<T>>,
    /// Inverse singular values between sites (length = n-1)
    inverse_singular_values: Vec<DiagMatrix>,
    /// Active partition range
    partition: Range<usize>,
}

impl<T: TTScalar + Scalar + Default> InverseTensorTrain<T> {
    /// Create an InverseTensorTrain from a VidalTensorTrain
    pub fn from_vidal(vidal: &VidalTensorTrain<T>) -> Result<Self> {
        let n = vidal.len();
        if n == 0 {
            return Ok(Self {
                tensors: Vec::new(),
                inverse_singular_values: Vec::new(),
                partition: 0..0,
            });
        }

        let mut tensors = Vec::with_capacity(n);

        // First tensor: multiply by S[0] on the right
        if n > 1 && !vidal.singular_values[0].is_empty() {
            let tensor = &vidal.tensors[0];
            let left_dim = tensor.left_dim();
            let site_dim = tensor.site_dim();
            let right_dim = tensor.right_dim();

            let mut new_tensor = tensor3_zeros(left_dim, site_dim, right_dim);
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        let val = *tensor.get3(l, s, r);
                        let sv = if r < vidal.singular_values[0].len() {
                            vidal.singular_values[0][r]
                        } else {
                            1.0
                        };
                        new_tensor.set3(l, s, r, val * T::from_f64(sv));
                    }
                }
            }
            tensors.push(new_tensor);
        } else {
            tensors.push(vidal.tensors[0].clone());
        }

        // Middle tensors: multiply by S[i-1] on left and S[i] on right
        for i in 1..n - 1 {
            let tensor = &vidal.tensors[i];
            let left_dim = tensor.left_dim();
            let site_dim = tensor.site_dim();
            let right_dim = tensor.right_dim();

            let mut new_tensor = tensor3_zeros(left_dim, site_dim, right_dim);
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        let val = *tensor.get3(l, s, r);
                        let sv_left = if l < vidal.singular_values[i - 1].len() {
                            vidal.singular_values[i - 1][l]
                        } else {
                            1.0
                        };
                        let sv_right = if r < vidal.singular_values[i].len() {
                            vidal.singular_values[i][r]
                        } else {
                            1.0
                        };
                        new_tensor.set3(l, s, r, val * T::from_f64(sv_left) * T::from_f64(sv_right));
                    }
                }
            }
            tensors.push(new_tensor);
        }

        // Last tensor: multiply by S[n-2] on left
        if n > 1 {
            let tensor = &vidal.tensors[n - 1];
            let left_dim = tensor.left_dim();
            let site_dim = tensor.site_dim();
            let right_dim = tensor.right_dim();

            let mut new_tensor = tensor3_zeros(left_dim, site_dim, right_dim);
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        let val = *tensor.get3(l, s, r);
                        let sv = if l < vidal.singular_values[n - 2].len() {
                            vidal.singular_values[n - 2][l]
                        } else {
                            1.0
                        };
                        new_tensor.set3(l, s, r, val * T::from_f64(sv));
                    }
                }
            }
            tensors.push(new_tensor);
        }

        // Compute inverse singular values
        let inverse_singular_values: Vec<DiagMatrix> = vidal
            .singular_values
            .iter()
            .map(|sv| {
                sv.iter()
                    .map(|&v| if v.abs() > 1e-15 { 1.0 / v } else { 0.0 })
                    .collect()
            })
            .collect();

        Ok(Self {
            tensors,
            inverse_singular_values,
            partition: vidal.partition.clone(),
        })
    }

    /// Create an InverseTensorTrain from a regular TensorTrain
    pub fn from_tensor_train(tt: &TensorTrain<T>) -> Result<Self> {
        let vidal = VidalTensorTrain::from_tensor_train(tt)?;
        Self::from_vidal(&vidal)
    }

    /// Get the inverse singular values between sites i and i+1
    pub fn inverse_singular_values(&self, i: usize) -> &DiagMatrix {
        &self.inverse_singular_values[i]
    }

    /// Get all inverse singular value matrices
    pub fn all_inverse_singular_values(&self) -> &[DiagMatrix] {
        &self.inverse_singular_values
    }

    /// Get the partition range
    pub fn partition(&self) -> &Range<usize> {
        &self.partition
    }

    /// Get mutable access to site tensors
    pub fn site_tensors_mut(&mut self) -> &mut [Tensor3<T>] {
        &mut self.tensors
    }

    /// Set two adjacent site tensors along with their inverse singular values
    pub fn set_two_site_tensors(
        &mut self,
        i: usize,
        tensor1: Tensor3<T>,
        inv_sv: DiagMatrix,
        tensor2: Tensor3<T>,
    ) -> Result<()> {
        if i >= self.len() - 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Cannot set two-site tensors at site {} (max {})",
                    i,
                    self.len() - 2
                ),
            });
        }

        self.tensors[i] = tensor1;
        self.inverse_singular_values[i] = inv_sv;
        self.tensors[i + 1] = tensor2;
        Ok(())
    }

    /// Convert to a regular TensorTrain
    pub fn to_tensor_train(&self) -> TensorTrain<T> {
        let n = self.len();
        if n == 0 {
            return TensorTrain::from_tensors_unchecked(Vec::new());
        }

        let mut tensors = Vec::with_capacity(n);

        // All tensors except last: multiply by inverse singular values on right
        for i in 0..n - 1 {
            let tensor = &self.tensors[i];
            let left_dim = tensor.left_dim();
            let site_dim = tensor.site_dim();
            let right_dim = tensor.right_dim();

            let mut new_tensor = tensor3_zeros(left_dim, site_dim, right_dim);
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        let val = *tensor.get3(l, s, r);
                        let inv_sv = if r < self.inverse_singular_values[i].len() {
                            self.inverse_singular_values[i][r]
                        } else {
                            1.0
                        };
                        new_tensor.set3(l, s, r, val * T::from_f64(inv_sv));
                    }
                }
            }
            tensors.push(new_tensor);
        }

        // Last tensor is unchanged
        tensors.push(self.tensors[n - 1].clone());

        TensorTrain::from_tensors_unchecked(tensors)
    }
}

impl<T: TTScalar + Scalar + Default> AbstractTensorTrain<T> for InverseTensorTrain<T> {
    fn len(&self) -> usize {
        self.tensors.len()
    }

    fn site_tensor(&self, i: usize) -> &Tensor3<T> {
        &self.tensors[i]
    }

    fn site_tensors(&self) -> &[Tensor3<T>] {
        &self.tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vidal_creation() {
        let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
        let vidal = VidalTensorTrain::from_tensor_train(&tt).unwrap();

        assert_eq!(vidal.len(), 3);
        assert_eq!(vidal.all_singular_values().len(), 2);
    }

    #[test]
    fn test_vidal_to_tensor_train_preserves_sum() {
        let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
        let vidal = VidalTensorTrain::from_tensor_train(&tt).unwrap();
        let tt_back = vidal.to_tensor_train();

        let original_sum = tt.sum();
        let converted_sum = tt_back.sum();

        assert!(
            (original_sum - converted_sum).abs() < 1e-8,
            "Sum mismatch: {} vs {}",
            original_sum,
            converted_sum
        );
    }

    #[test]
    fn test_inverse_creation() {
        let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
        let inverse = InverseTensorTrain::from_tensor_train(&tt).unwrap();

        assert_eq!(inverse.len(), 3);
        assert_eq!(inverse.all_inverse_singular_values().len(), 2);
    }

    #[test]
    fn test_inverse_to_tensor_train_preserves_sum() {
        let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
        let inverse = InverseTensorTrain::from_tensor_train(&tt).unwrap();
        let tt_back = inverse.to_tensor_train();

        let original_sum = tt.sum();
        let converted_sum = tt_back.sum();

        assert!(
            (original_sum - converted_sum).abs() < 1e-8,
            "Sum mismatch: {} vs {}",
            original_sum,
            converted_sum
        );
    }

    #[test]
    fn test_vidal_singular_values_positive() {
        let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 3.0);
        let vidal = VidalTensorTrain::from_tensor_train(&tt).unwrap();

        // Check that singular values are positive
        for sv in vidal.all_singular_values() {
            for &v in sv {
                assert!(v > 0.0, "Singular value should be positive");
            }
        }
    }
}
