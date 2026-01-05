//! Abstract traits for tensor train objects

use crate::error::Result;
use crate::types::{LocalIndex, Tensor3, Tensor3Ops};
use num_traits::{One, Zero};

/// Scalar trait for tensor train elements
pub trait TTScalar:
    Clone
    + Copy
    + Zero
    + One
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + Default
    + Send
    + Sync
    + 'static
{
    /// Conjugate
    fn conj(self) -> Self;

    /// Absolute value squared
    fn abs_sq(self) -> f64;

    /// Create from f64
    fn from_f64(val: f64) -> Self;
}

impl TTScalar for f64 {
    fn conj(self) -> Self {
        self
    }

    fn abs_sq(self) -> f64 {
        self * self
    }

    fn from_f64(val: f64) -> Self {
        val
    }
}

impl TTScalar for f32 {
    fn conj(self) -> Self {
        self
    }

    fn abs_sq(self) -> f64 {
        (self * self) as f64
    }

    fn from_f64(val: f64) -> Self {
        val as f32
    }
}

impl TTScalar for num_complex::Complex64 {
    fn conj(self) -> Self {
        num_complex::Complex64::conj(&self)
    }

    fn abs_sq(self) -> f64 {
        self.norm_sqr()
    }

    fn from_f64(val: f64) -> Self {
        num_complex::Complex64::new(val, 0.0)
    }
}

impl TTScalar for num_complex::Complex32 {
    fn conj(self) -> Self {
        num_complex::Complex32::conj(&self)
    }

    fn abs_sq(self) -> f64 {
        self.norm_sqr() as f64
    }

    fn from_f64(val: f64) -> Self {
        num_complex::Complex32::new(val as f32, 0.0)
    }
}

/// Abstract trait for tensor train objects
///
/// A tensor train (also known as MPS) represents a high-dimensional tensor
/// as a product of low-rank tensors.
pub trait AbstractTensorTrain<T: TTScalar>: Sized {
    /// Number of sites (tensors) in the tensor train
    fn len(&self) -> usize;

    /// Check if the tensor train is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the site tensor at position i
    fn site_tensor(&self, i: usize) -> &Tensor3<T>;

    /// Get all site tensors
    fn site_tensors(&self) -> &[Tensor3<T>];

    /// Bond dimensions along the links between tensors
    /// Returns a vector of length L-1 where L is the number of sites
    fn link_dims(&self) -> Vec<usize> {
        if self.len() <= 1 {
            return Vec::new();
        }
        (1..self.len())
            .map(|i| self.site_tensor(i).left_dim())
            .collect()
    }

    /// Bond dimension at the link between tensor i and i+1
    fn link_dim(&self, i: usize) -> usize {
        self.site_tensor(i + 1).left_dim()
    }

    /// Site dimensions (physical dimensions) for each tensor
    fn site_dims(&self) -> Vec<usize> {
        (0..self.len())
            .map(|i| self.site_tensor(i).site_dim())
            .collect()
    }

    /// Site dimension at position i
    fn site_dim(&self, i: usize) -> usize {
        self.site_tensor(i).site_dim()
    }

    /// Maximum bond dimension (rank) of the tensor train
    fn rank(&self) -> usize {
        let lds = self.link_dims();
        if lds.is_empty() {
            1
        } else {
            *lds.iter().max().unwrap_or(&1)
        }
    }

    /// Evaluate the tensor train at a given index set
    fn evaluate(&self, indices: &[LocalIndex]) -> Result<T> {
        use crate::error::TensorTrainError;

        if indices.len() != self.len() {
            return Err(TensorTrainError::IndexLengthMismatch {
                expected: self.len(),
                got: indices.len(),
            });
        }

        if self.is_empty() {
            return Err(TensorTrainError::Empty);
        }

        // Start with the first tensor slice
        let first = self.site_tensor(0);
        let idx0 = indices[0];
        if idx0 >= first.site_dim() {
            return Err(TensorTrainError::IndexOutOfBounds {
                site: 0,
                index: idx0,
                max: first.site_dim(),
            });
        }

        // Vector of size right_dim
        let mut current: Vec<T> = first.slice_site(idx0);

        // Contract with remaining tensors
        for (site, &idx) in indices.iter().enumerate().skip(1) {
            let tensor = self.site_tensor(site);
            if idx >= tensor.site_dim() {
                return Err(TensorTrainError::IndexOutOfBounds {
                    site,
                    index: idx,
                    max: tensor.site_dim(),
                });
            }

            let slice = tensor.slice_site(idx);
            let left_dim = tensor.left_dim();
            let right_dim = tensor.right_dim();

            // Contract: current (of size left_dim) with slice (left_dim x right_dim)
            let mut next = vec![T::zero(); right_dim];
            for r in 0..right_dim {
                let mut sum = T::zero();
                for l in 0..left_dim {
                    sum = sum + current[l] * slice[l * right_dim + r];
                }
                next[r] = sum;
            }
            current = next;
        }

        // Should have a single element
        if current.len() != 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Final contraction resulted in {} elements, expected 1",
                    current.len()
                ),
            });
        }

        Ok(current[0])
    }

    /// Sum over all indices of the tensor train
    #[allow(clippy::needless_range_loop)]
    fn sum(&self) -> T {
        if self.is_empty() {
            return T::zero();
        }

        // Start with sum over first tensor
        let first = self.site_tensor(0);
        let mut current = vec![T::zero(); first.right_dim()];
        for s in 0..first.site_dim() {
            for r in 0..first.right_dim() {
                current[r] = current[r] + *first.get3(0, s, r);
            }
        }

        // Contract with sums of remaining tensors
        for site in 1..self.len() {
            let tensor = self.site_tensor(site);
            let left_dim = tensor.left_dim();
            let right_dim = tensor.right_dim();

            // Sum over site index
            let mut site_sum = vec![T::zero(); left_dim * right_dim];
            for l in 0..left_dim {
                for s in 0..tensor.site_dim() {
                    for r in 0..right_dim {
                        site_sum[l * right_dim + r] =
                            site_sum[l * right_dim + r] + *tensor.get3(l, s, r);
                    }
                }
            }

            // Contract with current
            let mut next = vec![T::zero(); right_dim];
            for r in 0..right_dim {
                let mut sum = T::zero();
                for l in 0..left_dim {
                    sum = sum + current[l] * site_sum[l * right_dim + r];
                }
                next[r] = sum;
            }
            current = next;
        }

        current[0]
    }

    /// Compute the squared Frobenius norm of the tensor train
    fn norm2(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }

        // Contract tt with its conjugate
        // result[la, la_conj, ra, ra_conj] at each step
        let first = self.site_tensor(0);
        let right_dim = first.right_dim();

        // current[ra, ra_conj] = sum_s first[0, s, ra] * conj(first[0, s, ra_conj])
        let mut current = vec![T::zero(); right_dim * right_dim];
        for s in 0..first.site_dim() {
            for ra in 0..right_dim {
                for ra_c in 0..right_dim {
                    let idx = ra * right_dim + ra_c;
                    current[idx] = current[idx] + *first.get3(0, s, ra) * first.get3(0, s, ra_c).conj();
                }
            }
        }

        // Contract through remaining tensors
        for site in 1..self.len() {
            let tensor = self.site_tensor(site);
            let left_dim = tensor.left_dim();
            let right_dim = tensor.right_dim();

            // new_current[ra, ra_conj] = sum_{la, la_conj, s}
            //     current[la, la_conj] * tensor[la, s, ra] * conj(tensor[la_conj, s, ra_conj])
            let mut new_current = vec![T::zero(); right_dim * right_dim];

            for la in 0..left_dim {
                for la_c in 0..left_dim {
                    let c_val = current[la * left_dim + la_c];
                    for s in 0..tensor.site_dim() {
                        for ra in 0..right_dim {
                            for ra_c in 0..right_dim {
                                let idx = ra * right_dim + ra_c;
                                new_current[idx] = new_current[idx]
                                    + c_val * *tensor.get3(la, s, ra) * tensor.get3(la_c, s, ra_c).conj();
                            }
                        }
                    }
                }
            }

            current = new_current;
        }

        // Final result should be a single element (1x1)
        current[0].abs_sq().sqrt()
    }

    /// Compute the Frobenius norm of the tensor train
    fn norm(&self) -> f64 {
        self.norm2().sqrt()
    }

    /// Compute the logarithm of the Frobenius norm of the tensor train
    ///
    /// This is more numerically stable than `norm().ln()` for tensor trains
    /// with very large or very small norms, as it avoids overflow/underflow
    /// by normalizing at each step.
    ///
    /// Returns `f64::NEG_INFINITY` for zero tensor trains.
    fn log_norm(&self) -> f64 {
        if self.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Contract tt with its conjugate, normalizing at each step
        // to avoid overflow/underflow
        let first = self.site_tensor(0);
        let right_dim = first.right_dim();

        // current[ra, ra_conj] = sum_s first[0, s, ra] * conj(first[0, s, ra_conj])
        let mut current = vec![T::zero(); right_dim * right_dim];
        for s in 0..first.site_dim() {
            for ra in 0..right_dim {
                for ra_c in 0..right_dim {
                    let idx = ra * right_dim + ra_c;
                    current[idx] = current[idx] + *first.get3(0, s, ra) * first.get3(0, s, ra_c).conj();
                }
            }
        }

        // Normalize and accumulate log scale
        let mut log_scale = 0.0;
        let scale = current.iter().map(|x| x.abs_sq()).fold(0.0, f64::max).sqrt();
        if scale > 0.0 {
            log_scale += scale.ln();
            let inv_scale = T::one() / T::from_f64(scale);
            for val in &mut current {
                *val = *val * inv_scale;
            }
        } else if self.len() == 1 {
            // Single site with zero norm
            return f64::NEG_INFINITY;
        }

        // Contract through remaining tensors
        for site in 1..self.len() {
            let tensor = self.site_tensor(site);
            let left_dim = tensor.left_dim();
            let right_dim = tensor.right_dim();

            // new_current[ra, ra_conj] = sum_{la, la_conj, s}
            //     current[la, la_conj] * tensor[la, s, ra] * conj(tensor[la_conj, s, ra_conj])
            let mut new_current = vec![T::zero(); right_dim * right_dim];

            for la in 0..left_dim {
                for la_c in 0..left_dim {
                    let c_val = current[la * left_dim + la_c];
                    for s in 0..tensor.site_dim() {
                        for ra in 0..right_dim {
                            for ra_c in 0..right_dim {
                                let idx = ra * right_dim + ra_c;
                                new_current[idx] = new_current[idx]
                                    + c_val * *tensor.get3(la, s, ra) * tensor.get3(la_c, s, ra_c).conj();
                            }
                        }
                    }
                }
            }

            // Normalize and accumulate log scale
            let scale = new_current.iter().map(|x| x.abs_sq()).fold(0.0, f64::max).sqrt();
            if scale > 0.0 {
                log_scale += scale.ln();
                let inv_scale = T::one() / T::from_f64(scale);
                for val in &mut new_current {
                    *val = *val * inv_scale;
                }
            }

            current = new_current;
        }

        // Final result: norm = sqrt(norm2), where norm2 = final_val * cumulative_scale
        // log(norm) = 0.5 * log(norm2) = 0.5 * (log(final_val) + log_scale)
        let final_val = current[0].abs_sq().sqrt();
        if final_val > 0.0 {
            0.5 * (final_val.ln() + log_scale)
        } else {
            f64::NEG_INFINITY
        }
    }
}
