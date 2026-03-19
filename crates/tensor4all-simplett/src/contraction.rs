//! Contraction operations for tensor trains
//!
//! This module provides various ways to combine tensor trains:
//! - `dot`: Inner product (returns scalar)

use crate::compression::CompressionMethod;
use crate::error::{Result, TensorTrainError};
use crate::tensortrain::TensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::Tensor3Ops;
use matrixci::util::{zeros, Matrix};
use matrixci::Scalar;

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

        let mut result: Matrix<T> = zeros(a0.right_dim(), b0.right_dim());
        for s in 0..a0.site_dim() {
            for ra in 0..a0.right_dim() {
                for rb in 0..b0.right_dim() {
                    result[[ra, rb]] = result[[ra, rb]] + *a0.get3(0, s, ra) * *b0.get3(0, s, rb);
                }
            }
        }

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

            // new_result[ra', rb'] = sum_{la, lb, s} result[la, lb] * a[la, s, ra'] * b[lb, s, rb']
            let mut new_result: Matrix<T> = zeros(a.right_dim(), b.right_dim());

            for la in 0..a.left_dim() {
                for lb in 0..b.left_dim() {
                    let r_val = result[[la, lb]];
                    for s in 0..a.site_dim() {
                        for ra in 0..a.right_dim() {
                            for rb in 0..b.right_dim() {
                                new_result[[ra, rb]] = new_result[[ra, rb]]
                                    + r_val * *a.get3(la, s, ra) * *b.get3(lb, s, rb);
                            }
                        }
                    }
                }
            }

            result = new_result;
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
