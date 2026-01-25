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
mod tests {
    use super::*;
    use crate::types::{tensor3_zeros, Tensor3};

    #[test]
    fn test_dot_constant() {
        let tt1 = TensorTrain::<f64>::constant(&[2, 3], 2.0);
        let tt2 = TensorTrain::<f64>::constant(&[2, 3], 3.0);

        let result = tt1.dot(&tt2).unwrap();

        // Each element product is 2.0 * 3.0 = 6.0
        // Sum over 2*3=6 elements: 6.0 * 6 = 36.0
        assert!((result - 36.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_different_tensors() {
        let mut t0_a: Tensor3<f64> = tensor3_zeros(1, 3, 2);
        for s in 0..3 {
            for r in 0..2 {
                t0_a.set3(0, s, r, (s + r + 1) as f64);
            }
        }

        let mut t1_a: Tensor3<f64> = tensor3_zeros(2, 2, 1);
        for l in 0..2 {
            for s in 0..2 {
                t1_a.set3(l, s, 0, (l + s + 1) as f64);
            }
        }

        let tt_a = TensorTrain::new(vec![t0_a.clone(), t1_a.clone()]).unwrap();

        let mut t0_b: Tensor3<f64> = tensor3_zeros(1, 3, 1);
        for s in 0..3 {
            t0_b.set3(0, s, 0, (s + 1) as f64 * 0.5);
        }

        let mut t1_b: Tensor3<f64> = tensor3_zeros(1, 2, 1);
        for s in 0..2 {
            t1_b.set3(0, s, 0, (s + 2) as f64 * 0.3);
        }

        let tt_b = TensorTrain::new(vec![t0_b, t1_b]).unwrap();

        let dot_result = tt_a.dot(&tt_b).unwrap();

        // Compute expected value by brute force
        let mut expected = 0.0;
        for i0 in 0..3 {
            for i1 in 0..2 {
                let val_a = tt_a.evaluate(&[i0, i1]).unwrap();
                let val_b = tt_b.evaluate(&[i0, i1]).unwrap();
                expected += val_a * val_b;
            }
        }

        assert!(
            (dot_result - expected).abs() < 1e-10,
            "dot = {}, expected = {}",
            dot_result,
            expected
        );
    }
}
