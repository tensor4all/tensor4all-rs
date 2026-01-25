//! TensorTrain implementation

use crate::error::{Result, TensorTrainError};
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{tensor3_zeros, Tensor3, Tensor3Ops};

/// Tensor Train (Matrix Product State) representation
///
/// A tensor train represents a high-dimensional tensor as a product of
/// lower-dimensional tensors:
///
/// T\[i1, i2, ..., iL\] = A1\[i1\] * A2\[i2\] * ... * AL\[iL\]
///
/// where each Ak\[ik\] is a matrix of shape (rk-1, rk).
#[derive(Debug, Clone)]
pub struct TensorTrain<T: TTScalar> {
    /// The tensors that make up the tensor train
    /// Each tensor has shape (left_bond, site_dim, right_bond)
    tensors: Vec<Tensor3<T>>,
}

impl<T: TTScalar> TensorTrain<T> {
    /// Create a new tensor train from a list of 3D tensors
    ///
    /// Each tensor should have shape (left_bond, site_dim, right_bond)
    /// where the right_bond of tensor i equals the left_bond of tensor i+1.
    pub fn new(tensors: Vec<Tensor3<T>>) -> Result<Self> {
        // Validate dimensions
        for i in 0..tensors.len().saturating_sub(1) {
            if tensors[i].right_dim() != tensors[i + 1].left_dim() {
                return Err(TensorTrainError::DimensionMismatch { site: i });
            }
        }

        // First tensor should have left_dim = 1
        if !tensors.is_empty() && tensors[0].left_dim() != 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: "First tensor must have left dimension 1".to_string(),
            });
        }

        // Last tensor should have right_dim = 1
        if !tensors.is_empty() && tensors.last().unwrap().right_dim() != 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: "Last tensor must have right dimension 1".to_string(),
            });
        }

        Ok(Self { tensors })
    }

    /// Create a tensor train from tensors without dimension validation
    /// (for internal use when dimensions are known to be correct)
    pub(crate) fn from_tensors_unchecked(tensors: Vec<Tensor3<T>>) -> Self {
        Self { tensors }
    }

    /// Create a tensor train representing the zero function
    pub fn zeros(site_dims: &[usize]) -> Self {
        let tensors: Vec<Tensor3<T>> = site_dims.iter().map(|&d| tensor3_zeros(1, d, 1)).collect();
        Self { tensors }
    }

    /// Create a tensor train representing a constant function
    pub fn constant(site_dims: &[usize], value: T) -> Self {
        if site_dims.is_empty() {
            return Self {
                tensors: Vec::new(),
            };
        }

        let n = site_dims.len();
        let mut tensors = Vec::with_capacity(n);

        // First tensor: all ones
        let mut first = tensor3_zeros(1, site_dims[0], 1);
        for s in 0..site_dims[0] {
            first.set3(0, s, 0, T::one());
        }
        tensors.push(first);

        // Middle tensors: all ones
        for &d in &site_dims[1..n - 1] {
            let mut tensor = tensor3_zeros(1, d, 1);
            for s in 0..d {
                tensor.set3(0, s, 0, T::one());
            }
            tensors.push(tensor);
        }

        // Last tensor: multiply by value
        if n > 1 {
            let mut last = tensor3_zeros(1, site_dims[n - 1], 1);
            for s in 0..site_dims[n - 1] {
                last.set3(0, s, 0, value);
            }
            tensors.push(last);
        } else {
            // Single site: multiply the first (and only) tensor by value
            for s in 0..site_dims[0] {
                let current = *tensors[0].get3(0, s, 0);
                tensors[0].set3(0, s, 0, current * value);
            }
        }

        Self { tensors }
    }

    /// Get mutable access to the site tensors
    pub fn site_tensors_mut(&mut self) -> &mut [Tensor3<T>] {
        &mut self.tensors
    }

    /// Multiply the tensor train by a scalar
    pub fn scale(&mut self, factor: T) {
        if !self.tensors.is_empty() {
            let last = self.tensors.len() - 1;
            let tensor = &mut self.tensors[last];
            for l in 0..tensor.left_dim() {
                for s in 0..tensor.site_dim() {
                    for r in 0..tensor.right_dim() {
                        let val = *tensor.get3(l, s, r);
                        tensor.set3(l, s, r, val * factor);
                    }
                }
            }
        }
    }

    /// Create a scaled copy of the tensor train
    pub fn scaled(&self, factor: T) -> Self {
        let mut result = self.clone();
        result.scale(factor);
        result
    }

    /// Reverse the tensor train (swap left and right)
    pub fn reverse(&self) -> Self {
        let mut new_tensors = Vec::with_capacity(self.tensors.len());
        for tensor in self.tensors.iter().rev() {
            // Swap left and right dimensions
            let mut new_tensor =
                tensor3_zeros(tensor.right_dim(), tensor.site_dim(), tensor.left_dim());
            for l in 0..tensor.left_dim() {
                for s in 0..tensor.site_dim() {
                    for r in 0..tensor.right_dim() {
                        new_tensor.set3(r, s, l, *tensor.get3(l, s, r));
                    }
                }
            }
            new_tensors.push(new_tensor);
        }
        Self {
            tensors: new_tensors,
        }
    }
}

impl<T: TTScalar> TensorTrain<T> {
    /// Convert the tensor train to a full tensor
    ///
    /// Returns a flat vector containing all tensor elements in row-major order,
    /// along with the shape (site dimensions).
    ///
    /// Warning: This can be very large for high-dimensional tensors!
    pub fn fulltensor(&self) -> (Vec<T>, Vec<usize>) {
        if self.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let site_dims: Vec<usize> = self.site_dims();
        let total_size: usize = site_dims.iter().product();

        if total_size == 0 {
            return (Vec::new(), site_dims);
        }

        // Build full tensor by iterating over all indices
        let mut result = Vec::with_capacity(total_size);
        let mut indices = vec![0usize; site_dims.len()];

        loop {
            // Evaluate at current indices
            if let Ok(val) = self.evaluate(&indices) {
                result.push(val);
            } else {
                result.push(T::zero());
            }

            // Increment indices (row-major order, last index fastest)
            let mut carry = true;
            for i in (0..site_dims.len()).rev() {
                if carry {
                    indices[i] += 1;
                    if indices[i] >= site_dims[i] {
                        indices[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }

            if carry {
                break; // All indices wrapped around
            }
        }

        (result, site_dims)
    }
}

impl<T: TTScalar> AbstractTensorTrain<T> for TensorTrain<T> {
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
    use num_complex::Complex64;

    // Generic test functions for f64 and Complex64

    fn test_tensortrain_zeros_generic<T: TTScalar>() {
        let tt = TensorTrain::<T>::zeros(&[2, 3, 2]);
        assert_eq!(tt.len(), 3);
        assert_eq!(tt.site_dims(), vec![2, 3, 2]);
        assert_eq!(tt.rank(), 1);
    }

    fn test_tensortrain_constant_generic<T: TTScalar>() {
        let tt = TensorTrain::<T>::constant(&[2, 2], T::from_f64(5.0));
        assert_eq!(tt.len(), 2);

        // Sum should be 5.0 * 2 * 2 = 20.0
        let sum = tt.sum();
        assert!(sum.abs_sq().sqrt() - 20.0 < 1e-10);
    }

    fn test_tensortrain_evaluate_generic<T: TTScalar>() {
        // Create a simple tensor train that returns the product of indices + 1
        let _site_dims = [2, 3];

        // First tensor: values are 1 for index 0, 2 for index 1
        let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 1);
        t0.set3(0, 0, 0, T::from_f64(1.0));
        t0.set3(0, 1, 0, T::from_f64(2.0));

        // Second tensor: values are 1, 2, 3 for indices 0, 1, 2
        let mut t1: Tensor3<T> = tensor3_zeros(1, 3, 1);
        t1.set3(0, 0, 0, T::from_f64(1.0));
        t1.set3(0, 1, 0, T::from_f64(2.0));
        t1.set3(0, 2, 0, T::from_f64(3.0));

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();

        // tt([0, 0]) = 1 * 1 = 1
        let val00 = tt.evaluate(&[0, 0]).unwrap();
        assert!((val00 - T::from_f64(1.0)).abs_sq().sqrt() < 1e-10);
        // tt([1, 2]) = 2 * 3 = 6
        let val12 = tt.evaluate(&[1, 2]).unwrap();
        assert!((val12 - T::from_f64(6.0)).abs_sq().sqrt() < 1e-10);
    }

    fn test_tensortrain_scale_generic<T: TTScalar>() {
        let mut tt = TensorTrain::<T>::constant(&[2, 2], T::from_f64(1.0));
        tt.scale(T::from_f64(3.0));

        // Sum should be 3.0 * 2 * 2 = 12.0
        let sum = tt.sum();
        assert!((sum.abs_sq().sqrt() - 12.0).abs() < 1e-10);
    }

    fn test_tensortrain_reverse_generic<T: TTScalar>() {
        let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 1);
        t0.set3(0, 0, 0, T::from_f64(1.0));
        t0.set3(0, 1, 0, T::from_f64(2.0));

        let mut t1: Tensor3<T> = tensor3_zeros(1, 3, 1);
        t1.set3(0, 0, 0, T::from_f64(1.0));
        t1.set3(0, 1, 0, T::from_f64(2.0));
        t1.set3(0, 2, 0, T::from_f64(3.0));

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();
        let tt_rev = tt.reverse();

        assert_eq!(tt_rev.len(), 2);
        assert_eq!(tt_rev.site_dims(), vec![3, 2]);

        // Reversed evaluation: tt_rev([2, 1]) should equal tt([1, 2])
        let diff = tt_rev.evaluate(&[2, 1]).unwrap() - tt.evaluate(&[1, 2]).unwrap();
        assert!(diff.abs_sq().sqrt() < 1e-10);
    }

    fn test_fulltensor_generic<T: TTScalar>() {
        let tt = TensorTrain::<T>::constant(&[2, 3], T::from_f64(5.0));
        let (data, shape) = tt.fulltensor();

        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data.len(), 6);

        // All elements should be 5.0
        for val in &data {
            assert!((*val - T::from_f64(5.0)).abs_sq().sqrt() < 1e-10);
        }
    }

    fn test_fulltensor_matches_evaluate_generic<T: TTScalar>() {
        let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 1);
        t0.set3(0, 0, 0, T::from_f64(1.0));
        t0.set3(0, 1, 0, T::from_f64(2.0));

        let mut t1: Tensor3<T> = tensor3_zeros(1, 3, 1);
        t1.set3(0, 0, 0, T::from_f64(1.0));
        t1.set3(0, 1, 0, T::from_f64(2.0));
        t1.set3(0, 2, 0, T::from_f64(3.0));

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();
        let (data, shape) = tt.fulltensor();

        assert_eq!(shape, vec![2, 3]);

        // Check each element matches evaluate
        for i in 0..2 {
            for j in 0..3 {
                let idx = i * 3 + j;
                let expected = tt.evaluate(&[i, j]).unwrap();
                let diff = data[idx] - expected;
                assert!(diff.abs_sq().sqrt() < 1e-10, "Mismatch at [{}, {}]", i, j);
            }
        }
    }

    fn test_log_norm_matches_norm_generic<T: TTScalar>() {
        let tt = TensorTrain::<T>::constant(&[2, 3], T::from_f64(2.0));

        let norm = tt.norm();
        let log_norm = tt.log_norm();

        // log_norm should equal ln(norm)
        assert!(
            (log_norm - norm.ln()).abs() < 1e-10,
            "log_norm={}, ln(norm)={}",
            log_norm,
            norm.ln()
        );
    }

    fn test_log_norm_with_varied_values_generic<T: TTScalar>() {
        let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 2);
        t0.set3(0, 0, 0, T::from_f64(1.0));
        t0.set3(0, 0, 1, T::from_f64(0.5));
        t0.set3(0, 1, 0, T::from_f64(2.0));
        t0.set3(0, 1, 1, T::from_f64(1.0));

        let mut t1: Tensor3<T> = tensor3_zeros(2, 2, 1);
        t1.set3(0, 0, 0, T::from_f64(1.0));
        t1.set3(0, 1, 0, T::from_f64(2.0));
        t1.set3(1, 0, 0, T::from_f64(0.5));
        t1.set3(1, 1, 0, T::from_f64(1.5));

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();

        let norm = tt.norm();
        let log_norm = tt.log_norm();

        assert!(
            (log_norm - norm.ln()).abs() < 1e-10,
            "log_norm={}, ln(norm)={}",
            log_norm,
            norm.ln()
        );
    }

    fn test_log_norm_zero_tensor_generic<T: TTScalar>() {
        let tt = TensorTrain::<T>::zeros(&[2, 3]);
        let log_norm = tt.log_norm();

        assert!(log_norm.is_infinite() && log_norm < 0.0);
    }

    // f64 tests
    #[test]
    fn test_tensortrain_zeros_f64() {
        test_tensortrain_zeros_generic::<f64>();
    }

    #[test]
    fn test_tensortrain_constant_f64() {
        test_tensortrain_constant_generic::<f64>();
    }

    #[test]
    fn test_tensortrain_evaluate_f64() {
        test_tensortrain_evaluate_generic::<f64>();
    }

    #[test]
    fn test_tensortrain_scale_f64() {
        test_tensortrain_scale_generic::<f64>();
    }

    #[test]
    fn test_tensortrain_reverse_f64() {
        test_tensortrain_reverse_generic::<f64>();
    }

    #[test]
    fn test_fulltensor_f64() {
        test_fulltensor_generic::<f64>();
    }

    #[test]
    fn test_fulltensor_matches_evaluate_f64() {
        test_fulltensor_matches_evaluate_generic::<f64>();
    }

    #[test]
    fn test_log_norm_matches_norm_f64() {
        test_log_norm_matches_norm_generic::<f64>();
    }

    #[test]
    fn test_log_norm_with_varied_values_f64() {
        test_log_norm_with_varied_values_generic::<f64>();
    }

    #[test]
    fn test_log_norm_zero_tensor_f64() {
        test_log_norm_zero_tensor_generic::<f64>();
    }

    // Complex64 tests
    #[test]
    fn test_tensortrain_zeros_c64() {
        test_tensortrain_zeros_generic::<Complex64>();
    }

    #[test]
    fn test_tensortrain_constant_c64() {
        test_tensortrain_constant_generic::<Complex64>();
    }

    #[test]
    fn test_tensortrain_evaluate_c64() {
        test_tensortrain_evaluate_generic::<Complex64>();
    }

    #[test]
    fn test_tensortrain_scale_c64() {
        test_tensortrain_scale_generic::<Complex64>();
    }

    #[test]
    fn test_tensortrain_reverse_c64() {
        test_tensortrain_reverse_generic::<Complex64>();
    }

    #[test]
    fn test_fulltensor_c64() {
        test_fulltensor_generic::<Complex64>();
    }

    #[test]
    fn test_fulltensor_matches_evaluate_c64() {
        test_fulltensor_matches_evaluate_generic::<Complex64>();
    }

    #[test]
    fn test_log_norm_matches_norm_c64() {
        test_log_norm_matches_norm_generic::<Complex64>();
    }

    #[test]
    fn test_log_norm_with_varied_values_c64() {
        test_log_norm_with_varied_values_generic::<Complex64>();
    }

    #[test]
    fn test_log_norm_zero_tensor_c64() {
        test_log_norm_zero_tensor_generic::<Complex64>();
    }
}
