//! Canonical forms for tensor trains
//!
//! This module provides tensor train representations with canonical forms:
//! - `SiteTensorTrain`: Center-canonical form where tensors left of center are
//!   left-orthogonal and tensors right of center are right-orthogonal.

use std::ops::Range;

use crate::error::{Result, TensorTrainError};
use crate::tensortrain::TensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{tensor3_zeros, Tensor3, Tensor3Ops};
use matrixci::util::{mat_mul, ncols, nrows, transpose, zeros, Matrix, Scalar};
use matrixci::{rrlu, RrLUOptions};

/// Compute QR decomposition using rank-revealing LU with left-orthogonal output
fn qr_decomp<T: TTScalar + Scalar>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    let options = RrLUOptions {
        max_rank: ncols(matrix).min(nrows(matrix)),
        rel_tol: 0.0, // No truncation
        abs_tol: 0.0,
        left_orthogonal: true,
    };
    let lu = rrlu(matrix, Some(options));
    (lu.left(true), lu.right(true))
}

/// Compute LQ decomposition (transpose, QR, transpose)
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

/// Site Tensor Train with center canonical form
///
/// A tensor train where:
/// - Tensors at indices < center are left-orthogonal
/// - Tensors at indices > center are right-orthogonal
/// - The tensor at the center index is the "center" tensor
#[derive(Debug, Clone)]
pub struct SiteTensorTrain<T: TTScalar> {
    /// Site tensors
    tensors: Vec<Tensor3<T>>,
    /// Center index (0-based)
    center: usize,
    /// Active partition range
    partition: Range<usize>,
}

impl<T: TTScalar + Scalar + Default> SiteTensorTrain<T> {
    /// Create a new SiteTensorTrain from tensors with specified center
    pub fn new(tensors: Vec<Tensor3<T>>, center: usize) -> Result<Self> {
        let n = tensors.len();
        if n == 0 {
            return Err(TensorTrainError::Empty);
        }
        if center >= n {
            return Err(TensorTrainError::InvalidOperation {
                message: format!("Center {} is out of range for {} tensors", center, n),
            });
        }

        // Validate dimensions
        for i in 0..n.saturating_sub(1) {
            if tensors[i].right_dim() != tensors[i + 1].left_dim() {
                return Err(TensorTrainError::DimensionMismatch { site: i });
            }
        }

        let mut result = Self {
            tensors,
            center,
            partition: 0..n,
        };
        result.canonicalize();
        Ok(result)
    }

    /// Create from TensorTrain with specified center
    pub fn from_tensor_train(tt: &TensorTrain<T>, center: usize) -> Result<Self> {
        let tensors = tt.site_tensors().to_vec();
        Self::new(tensors, center)
    }

    /// Get the center index
    pub fn center(&self) -> usize {
        self.center
    }

    /// Get the partition range
    pub fn partition(&self) -> &Range<usize> {
        &self.partition
    }

    /// Get mutable access to site tensors
    pub fn site_tensors_mut(&mut self) -> &mut [Tensor3<T>] {
        &mut self.tensors
    }

    /// Canonicalize the tensor train around the center
    fn canonicalize(&mut self) {
        let n = self.len();
        if n <= 1 {
            return;
        }

        // Left sweep: make tensors [0..center) left-orthogonal
        for i in 0..self.center {
            self.make_left_orthogonal(i);
        }

        // Right sweep: make tensors (center..n] right-orthogonal
        for i in (self.center + 1..n).rev() {
            self.make_right_orthogonal(i);
        }
    }

    /// Make tensor at site i left-orthogonal, pushing R to site i+1
    fn make_left_orthogonal(&mut self, i: usize) {
        if i >= self.len() - 1 {
            return;
        }

        let left_dim = self.tensors[i].left_dim();
        let site_dim = self.tensors[i].site_dim();

        // Reshape to (left_dim * site_dim, right_dim)
        let mat = tensor3_to_left_matrix(&self.tensors[i]);
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
        self.tensors[i] = new_tensor;

        // Contract R with next tensor
        let next_site_dim = self.tensors[i + 1].site_dim();
        let next_right_dim = self.tensors[i + 1].right_dim();
        let next_mat = tensor3_to_right_matrix(&self.tensors[i + 1]);

        // R * next_mat
        let contracted = mat_mul(&r, &next_mat);

        // Update next tensor
        let mut new_next_tensor = tensor3_zeros(new_bond_dim, next_site_dim, next_right_dim);
        for l in 0..new_bond_dim {
            for s in 0..next_site_dim {
                for r_idx in 0..next_right_dim {
                    new_next_tensor.set3(l, s, r_idx, contracted[[l, s * next_right_dim + r_idx]]);
                }
            }
        }
        self.tensors[i + 1] = new_next_tensor;
    }

    /// Make tensor at site i right-orthogonal, pushing L to site i-1
    fn make_right_orthogonal(&mut self, i: usize) {
        if i == 0 {
            return;
        }

        let site_dim = self.tensors[i].site_dim();
        let right_dim = self.tensors[i].right_dim();

        // Reshape to (left_dim, site_dim * right_dim)
        let mat = tensor3_to_right_matrix(&self.tensors[i]);
        let (l_mat, q) = lq_decomp(&mat);

        let new_bond_dim = nrows(&q);

        // Update current tensor with Q
        let mut new_tensor = tensor3_zeros(new_bond_dim, site_dim, right_dim);
        for l in 0..new_bond_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    new_tensor.set3(l, s, r, q[[l, s * right_dim + r]]);
                }
            }
        }
        self.tensors[i] = new_tensor;

        // Contract previous tensor with L
        let prev_left_dim = self.tensors[i - 1].left_dim();
        let prev_site_dim = self.tensors[i - 1].site_dim();
        let prev_mat = tensor3_to_left_matrix(&self.tensors[i - 1]);

        // prev_mat * L
        let contracted = mat_mul(&prev_mat, &l_mat);

        // Update previous tensor
        let mut new_prev_tensor = tensor3_zeros(prev_left_dim, prev_site_dim, new_bond_dim);
        for l in 0..prev_left_dim {
            for s in 0..prev_site_dim {
                for r in 0..new_bond_dim {
                    new_prev_tensor.set3(l, s, r, contracted[[l * prev_site_dim + s, r]]);
                }
            }
        }
        self.tensors[i - 1] = new_prev_tensor;
    }

    /// Move the center one position to the right
    pub fn move_center_right(&mut self) -> Result<()> {
        if self.center >= self.len() - 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: "Cannot move center right: already at rightmost position".to_string(),
            });
        }

        self.make_left_orthogonal(self.center);
        self.center += 1;
        Ok(())
    }

    /// Move the center one position to the left
    pub fn move_center_left(&mut self) -> Result<()> {
        if self.center == 0 {
            return Err(TensorTrainError::InvalidOperation {
                message: "Cannot move center left: already at leftmost position".to_string(),
            });
        }

        self.make_right_orthogonal(self.center);
        self.center -= 1;
        Ok(())
    }

    /// Move the center to a specific position
    pub fn set_center(&mut self, new_center: usize) -> Result<()> {
        if new_center >= self.len() {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "New center {} is out of range for {} tensors",
                    new_center,
                    self.len()
                ),
            });
        }

        while self.center < new_center {
            self.move_center_right()?;
        }
        while self.center > new_center {
            self.move_center_left()?;
        }
        Ok(())
    }

    /// Convert to a regular TensorTrain
    pub fn to_tensor_train(&self) -> TensorTrain<T> {
        TensorTrain::from_tensors_unchecked(self.tensors.clone())
    }

    /// Set the tensor at a specific site
    ///
    /// Note: This may invalidate the canonical form. Use with caution.
    pub fn set_site_tensor(&mut self, i: usize, tensor: Tensor3<T>) {
        self.tensors[i] = tensor;
    }

    /// Set two adjacent site tensors (useful for TEBD-like algorithms)
    pub fn set_two_site_tensors(
        &mut self,
        i: usize,
        tensor1: Tensor3<T>,
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
        self.tensors[i + 1] = tensor2;
        Ok(())
    }
}

impl<T: TTScalar + Scalar + Default> AbstractTensorTrain<T> for SiteTensorTrain<T> {
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

/// Center canonicalize a vector of tensors in place
pub fn center_canonicalize<T: TTScalar + Scalar + Default>(
    tensors: &mut [Tensor3<T>],
    center: usize,
) {
    let n = tensors.len();
    if n <= 1 || center >= n {
        return;
    }

    // Left sweep: make tensors [0..center) left-orthogonal
    for i in 0..center {
        let left_dim = tensors[i].left_dim();
        let site_dim = tensors[i].site_dim();

        let mat = tensor3_to_left_matrix(&tensors[i]);
        let (q, r) = qr_decomp(&mat);

        let new_bond_dim = ncols(&q);

        // Update current tensor
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
        if i + 1 < n {
            let next_site_dim = tensors[i + 1].site_dim();
            let next_right_dim = tensors[i + 1].right_dim();
            let next_mat = tensor3_to_right_matrix(&tensors[i + 1]);

            let contracted = mat_mul(&r, &next_mat);

            let mut new_next_tensor = tensor3_zeros(new_bond_dim, next_site_dim, next_right_dim);
            for l in 0..new_bond_dim {
                for s in 0..next_site_dim {
                    for r_idx in 0..next_right_dim {
                        new_next_tensor.set3(
                            l,
                            s,
                            r_idx,
                            contracted[[l, s * next_right_dim + r_idx]],
                        );
                    }
                }
            }
            tensors[i + 1] = new_next_tensor;
        }
    }

    // Right sweep: make tensors (center..n] right-orthogonal
    for i in (center + 1..n).rev() {
        let site_dim = tensors[i].site_dim();
        let right_dim = tensors[i].right_dim();

        let mat = tensor3_to_right_matrix(&tensors[i]);
        let (l_mat, q) = lq_decomp(&mat);

        let new_bond_dim = nrows(&q);

        // Update current tensor
        let mut new_tensor = tensor3_zeros(new_bond_dim, site_dim, right_dim);
        for l in 0..new_bond_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    new_tensor.set3(l, s, r, q[[l, s * right_dim + r]]);
                }
            }
        }
        tensors[i] = new_tensor;

        // Contract L with previous tensor
        if i > 0 {
            let prev_left_dim = tensors[i - 1].left_dim();
            let prev_site_dim = tensors[i - 1].site_dim();
            let prev_mat = tensor3_to_left_matrix(&tensors[i - 1]);

            let contracted = mat_mul(&prev_mat, &l_mat);

            let mut new_prev_tensor = tensor3_zeros(prev_left_dim, prev_site_dim, new_bond_dim);
            for l in 0..prev_left_dim {
                for s in 0..prev_site_dim {
                    for r in 0..new_bond_dim {
                        new_prev_tensor.set3(l, s, r, contracted[[l * prev_site_dim + s, r]]);
                    }
                }
            }
            tensors[i - 1] = new_prev_tensor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    // Generic test functions for f64 and Complex64

    fn test_site_tensor_train_creation_generic<T: TTScalar + Scalar + Default>() {
        let tt = TensorTrain::<T>::constant(&[2, 3, 2], T::from_f64(1.0));
        let stt = SiteTensorTrain::from_tensor_train(&tt, 1).unwrap();

        assert_eq!(stt.len(), 3);
        assert_eq!(stt.center(), 1);
    }

    fn test_site_tensor_train_preserves_values_generic<T: TTScalar + Scalar + Default>() {
        let tt = TensorTrain::<T>::constant(&[2, 3, 2], T::from_f64(2.0));
        let stt = SiteTensorTrain::from_tensor_train(&tt, 1).unwrap();

        // Check that evaluation is preserved
        let original_sum = tt.sum();
        let stt_sum = stt.sum();
        assert!(
            TTScalar::abs_sq(original_sum - stt_sum).sqrt() < 1e-10,
            "Sum mismatch"
        );
    }

    fn test_move_center_generic<T: TTScalar + Scalar + Default>() {
        let tt = TensorTrain::<T>::constant(&[2, 3, 4, 2], T::from_f64(1.0));
        let mut stt = SiteTensorTrain::from_tensor_train(&tt, 0).unwrap();

        assert_eq!(stt.center(), 0);

        stt.move_center_right().unwrap();
        assert_eq!(stt.center(), 1);

        stt.move_center_right().unwrap();
        assert_eq!(stt.center(), 2);

        stt.move_center_left().unwrap();
        assert_eq!(stt.center(), 1);
    }

    fn test_set_center_generic<T: TTScalar + Scalar + Default>() {
        let tt = TensorTrain::<T>::constant(&[2, 3, 4, 2], T::from_f64(1.0));
        let mut stt = SiteTensorTrain::from_tensor_train(&tt, 0).unwrap();

        stt.set_center(3).unwrap();
        assert_eq!(stt.center(), 3);

        stt.set_center(1).unwrap();
        assert_eq!(stt.center(), 1);

        // Sum should still be preserved
        let original_sum = tt.sum();
        let stt_sum = stt.sum();
        assert!(
            TTScalar::abs_sq(original_sum - stt_sum).sqrt() < 1e-10,
            "Sum mismatch after moving center"
        );
    }

    fn test_to_tensor_train_generic<T: TTScalar + Scalar + Default>() {
        let tt = TensorTrain::<T>::constant(&[2, 3, 2], T::from_f64(3.0));
        let stt = SiteTensorTrain::from_tensor_train(&tt, 1).unwrap();
        let tt_back = stt.to_tensor_train();

        let original_sum = tt.sum();
        let converted_sum = tt_back.sum();
        assert!(
            TTScalar::abs_sq(original_sum - converted_sum).sqrt() < 1e-10,
            "Sum mismatch after round-trip"
        );
    }

    fn test_center_canonicalize_function_generic<T: TTScalar + Scalar + Default>() {
        let tt = TensorTrain::<T>::constant(&[2, 3, 2], T::from_f64(1.0));
        let mut tensors: Vec<Tensor3<T>> = tt.site_tensors().to_vec();

        let original_sum = tt.sum();

        center_canonicalize(&mut tensors, 1);

        // Reconstruct and verify sum
        let tt_new = TensorTrain::from_tensors_unchecked(tensors);
        let new_sum = tt_new.sum();
        assert!(
            TTScalar::abs_sq(original_sum - new_sum).sqrt() < 1e-10,
            "Sum mismatch"
        );
    }

    fn test_evaluate_matches_original_generic<T: TTScalar + Scalar + Default>() {
        let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 2);
        t0.set3(0, 0, 0, T::from_f64(1.0));
        t0.set3(0, 0, 1, T::from_f64(0.5));
        t0.set3(0, 1, 0, T::from_f64(2.0));
        t0.set3(0, 1, 1, T::from_f64(1.0));

        let mut t1: Tensor3<T> = tensor3_zeros(2, 3, 1);
        for l in 0..2 {
            for s in 0..3 {
                t1.set3(l, s, 0, T::from_f64((l + s + 1) as f64));
            }
        }

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();
        let stt = SiteTensorTrain::from_tensor_train(&tt, 0).unwrap();

        // Check multiple evaluations
        for i in 0..2 {
            for j in 0..3 {
                let original = tt.evaluate(&[i, j]).unwrap();
                let canonical = stt.evaluate(&[i, j]).unwrap();
                assert!(
                    TTScalar::abs_sq(original - canonical).sqrt() < 1e-10,
                    "Evaluation mismatch at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    // f64 tests
    #[test]
    fn test_site_tensor_train_creation_f64() {
        test_site_tensor_train_creation_generic::<f64>();
    }

    #[test]
    fn test_site_tensor_train_preserves_values_f64() {
        test_site_tensor_train_preserves_values_generic::<f64>();
    }

    #[test]
    fn test_move_center_f64() {
        test_move_center_generic::<f64>();
    }

    #[test]
    fn test_set_center_f64() {
        test_set_center_generic::<f64>();
    }

    #[test]
    fn test_to_tensor_train_f64() {
        test_to_tensor_train_generic::<f64>();
    }

    #[test]
    fn test_center_canonicalize_function_f64() {
        test_center_canonicalize_function_generic::<f64>();
    }

    #[test]
    fn test_evaluate_matches_original_f64() {
        test_evaluate_matches_original_generic::<f64>();
    }

    // Complex64 tests
    #[test]
    fn test_site_tensor_train_creation_c64() {
        test_site_tensor_train_creation_generic::<Complex64>();
    }

    #[test]
    fn test_site_tensor_train_preserves_values_c64() {
        test_site_tensor_train_preserves_values_generic::<Complex64>();
    }

    #[test]
    fn test_move_center_c64() {
        test_move_center_generic::<Complex64>();
    }

    #[test]
    fn test_set_center_c64() {
        test_set_center_generic::<Complex64>();
    }

    #[test]
    fn test_to_tensor_train_c64() {
        test_to_tensor_train_generic::<Complex64>();
    }

    #[test]
    fn test_center_canonicalize_function_c64() {
        test_center_canonicalize_function_generic::<Complex64>();
    }

    #[test]
    fn test_evaluate_matches_original_c64() {
        test_evaluate_matches_original_generic::<Complex64>();
    }
}
