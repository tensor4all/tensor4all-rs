//! Contraction operations for tensor trains
//!
//! This module provides various ways to combine tensor trains:
//! - `dot`: Inner product (returns scalar)

use crate::compression::CompressionMethod;
use crate::einsum_helper::{einsum_tensors, tensor_to_row_major_vec};
use crate::error::{Result, TensorTrainError};
use crate::tensortrain::TensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::Tensor3Ops;
use tensor4all_tcicore::matrix::Matrix;
use tensor4all_tcicore::Scalar;
use tenferro_tensor::{MemoryOrder, Tensor as TfTensor};

/// Options for contraction with compression
#[derive(Debug, Clone)]
pub struct ContractionOptions {
    /// Tolerance for truncation
    pub tolerance: f64,
    /// Maximum bond dimension
    pub max_bond_dim: usize,
    /// Compression method (LU or CI)
    pub method: CompressionMethod,
}

impl Default for ContractionOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_bond_dim: usize::MAX,
            method: CompressionMethod::LU,
        }
    }
}

impl<T: TTScalar + Scalar + Default> TensorTrain<T> {
    /// Compute the inner product (dot product) of two tensor trains
    ///
    /// Returns: sum over all indices i of self\[i\] * other\[i\]
    pub fn dot(&self, other: &Self) -> Result<T> {
        if self.len() != other.len() {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Cannot compute dot product of tensor trains with different lengths: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        if self.is_empty() {
            return Ok(T::zero());
        }

        let n = self.len();

        // Start with contraction of first site
        // result[ra, rb] = sum_s a[0, s, ra] * b[0, s, rb]
        let a0 = self.site_tensor(0);
        let b0 = other.site_tensor(0);

        if a0.site_dim() != b0.site_dim() {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Site dimensions mismatch at site 0: {} vs {}",
                    a0.site_dim(),
                    b0.site_dim()
                ),
            });
        }

        let mut result = Matrix::from_raw_vec(
            a0.right_dim(),
            b0.right_dim(),
            tensor_to_row_major_vec(&einsum_tensors(
                "asr,ast->rt",
                &[a0.as_inner(), b0.as_inner()],
            )),
        );

        // Contract through remaining sites
        for i in 1..n {
            let a = self.site_tensor(i);
            let b = other.site_tensor(i);

            if a.site_dim() != b.site_dim() {
                return Err(TensorTrainError::InvalidOperation {
                    message: format!(
                        "Site dimensions mismatch at site {}: {} vs {}",
                        i,
                        a.site_dim(),
                        b.site_dim()
                    ),
                });
            }

            let result_tf = TfTensor::from_slice(
                result.as_slice(),
                &[result.nrows(), result.ncols()],
                MemoryOrder::RowMajor,
            )
            .expect("dot intermediate matrix dimensions should match tenferro");

            result = Matrix::from_raw_vec(
                a.right_dim(),
                b.right_dim(),
                tensor_to_row_major_vec(&einsum_tensors(
                    "ij,isk,jsl->kl",
                    &[&result_tf, a.as_inner(), b.as_inner()],
                )),
            );
        }

        // Final result should be 1x1
        Ok(result[[0, 0]])
    }
}

/// Convenience function to compute dot product
pub fn dot<T: TTScalar + Scalar + Default>(a: &TensorTrain<T>, b: &TensorTrain<T>) -> Result<T> {
    a.dot(b)
}

#[cfg(test)]
mod tests;
