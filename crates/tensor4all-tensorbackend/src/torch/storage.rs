//! Torch tensor storage implementation.
//!
//! Provides `TorchStorage` which wraps a `tch::Tensor` and implements
//! the storage interface for tensor operations with autograd support.

use anyhow::Result;
use num_complex::Complex64;
use std::fmt::Debug;
use tch::{Kind, Tensor};

/// Storage backed by a PyTorch tensor.
///
/// This struct wraps a `tch::Tensor` and provides the storage interface
/// compatible with tensor4all's storage system. It enables:
///
/// - Autograd (automatic differentiation)
/// - Native PyTorch operations including einsum
/// - Future GPU support (v2)
///
/// # Type Safety
///
/// `TorchStorage` is parameterized by a scalar type `T` for API compatibility,
/// but the actual data type is determined by the underlying `tch::Tensor`.
/// Use `kind()` to query the actual dtype.
#[derive(Debug)]
pub struct TorchStorage<T> {
    tensor: Tensor,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Clone for TorchStorage<T> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.shallow_clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> TorchStorage<T> {
    /// Create a new TorchStorage from a tch::Tensor.
    ///
    /// Note: The scalar type `T` should match the tensor's dtype.
    pub fn from_tensor(tensor: Tensor) -> Self {
        Self {
            tensor,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get a reference to the underlying tch::Tensor.
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Get a mutable reference to the underlying tch::Tensor.
    pub fn tensor_mut(&mut self) -> &mut Tensor {
        &mut self.tensor
    }

    /// Consume and return the underlying tch::Tensor.
    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }

    /// Get the shape (dimensions) of the storage.
    pub fn dims(&self) -> Vec<usize> {
        self.tensor.size().iter().map(|&d| d as usize).collect()
    }

    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.tensor.dim()
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.tensor.numel()
    }

    /// Check if the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the tch::Kind (dtype) of the tensor.
    pub fn kind(&self) -> Kind {
        self.tensor.kind()
    }

    /// Check if this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.tensor.requires_grad()
    }

    /// Set whether this tensor requires gradient computation.
    pub fn set_requires_grad(&mut self, requires_grad: bool) -> &mut Self {
        self.tensor = self.tensor.set_requires_grad(requires_grad);
        self
    }

    /// Get the gradient of this tensor, if it exists.
    pub fn grad(&self) -> Option<TorchStorage<T>> {
        let grad = self.tensor.grad();
        if grad.defined() {
            Some(TorchStorage::from_tensor(grad))
        } else {
            None
        }
    }

    /// Compute gradients by backpropagating from this tensor.
    ///
    /// This tensor must be a scalar (0-dimensional or single element).
    pub fn backward(&self) -> Result<()> {
        if self.tensor.numel() != 1 {
            anyhow::bail!(
                "backward() requires a scalar tensor, but tensor has {} elements",
                self.tensor.numel()
            );
        }
        self.tensor.backward();
        Ok(())
    }

    /// Detach the tensor from the computation graph.
    ///
    /// Returns a new storage that shares data but doesn't track gradients.
    pub fn detach(&self) -> Self {
        Self::from_tensor(self.tensor.detach())
    }
}

impl TorchStorage<f64> {
    /// Create from a Vec<f64> with explicit shape.
    pub fn from_vec_with_shape(data: Vec<f64>, dims: &[usize]) -> Self {
        let tensor_dims: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let tensor = Tensor::from_slice(&data)
            .to_kind(Kind::Double)
            .view(&tensor_dims[..]);
        Self::from_tensor(tensor)
    }

    /// Create a scalar (0-dimensional) storage from a single value.
    pub fn from_scalar(val: f64) -> Self {
        let tensor = Tensor::from(val);
        Self::from_tensor(tensor)
    }

    /// Get underlying data as a Vec<f64>.
    ///
    /// Note: This copies data from the tensor.
    pub fn to_vec(&self) -> Vec<f64> {
        Vec::<f64>::try_from(&self.tensor.flatten(0, -1)).unwrap_or_default()
    }

    /// Get a single element at linear index.
    pub fn get(&self, i: usize) -> f64 {
        self.tensor.flatten(0, -1).double_value(&[i as i64])
    }

    /// Permute axes according to the given permutation.
    pub fn permute(&self, perm: &[usize]) -> Self {
        let perm_i64: Vec<i64> = perm.iter().map(|&p| p as i64).collect();
        let permuted = self.tensor.permute(&perm_i64);
        Self::from_tensor(permuted.contiguous())
    }

    /// Convert to Complex64 storage.
    pub fn to_complex(&self) -> TorchStorage<Complex64> {
        // Create complex tensor with zero imaginary part
        let real = self.tensor.to_kind(Kind::Double);
        let imag = Tensor::zeros_like(&real);
        let complex = Tensor::complex(&real, &imag);
        TorchStorage::from_tensor(complex)
    }
}

impl TorchStorage<Complex64> {
    /// Create from a Vec<Complex64> with explicit shape.
    pub fn from_vec_with_shape(data: Vec<Complex64>, dims: &[usize]) -> Self {
        let tensor_dims: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        // Split complex into real and imaginary parts
        let real: Vec<f64> = data.iter().map(|c| c.re).collect();
        let imag: Vec<f64> = data.iter().map(|c| c.im).collect();

        let real_tensor = Tensor::from_slice(&real).to_kind(Kind::Double);
        let imag_tensor = Tensor::from_slice(&imag).to_kind(Kind::Double);
        let complex_tensor = Tensor::complex(&real_tensor, &imag_tensor).view(&tensor_dims[..]);

        Self::from_tensor(complex_tensor)
    }

    /// Create a scalar (0-dimensional) storage from a single value.
    pub fn from_scalar(val: Complex64) -> Self {
        let real = Tensor::from(val.re);
        let imag = Tensor::from(val.im);
        let tensor = Tensor::complex(&real, &imag);
        Self::from_tensor(tensor)
    }

    /// Get underlying data as a Vec<Complex64>.
    ///
    /// Note: This copies data from the tensor.
    pub fn to_vec(&self) -> Vec<Complex64> {
        let flat = self.tensor.flatten(0, -1);
        let real = Vec::<f64>::try_from(&flat.real()).unwrap_or_default();
        let imag = Vec::<f64>::try_from(&flat.imag()).unwrap_or_default();
        real.into_iter()
            .zip(imag)
            .map(|(r, i)| Complex64::new(r, i))
            .collect()
    }

    /// Get a single element at linear index.
    pub fn get(&self, i: usize) -> Complex64 {
        let flat = self.tensor.flatten(0, -1);
        let r = flat.real().double_value(&[i as i64]);
        let im = flat.imag().double_value(&[i as i64]);
        Complex64::new(r, im)
    }

    /// Permute axes according to the given permutation.
    pub fn permute(&self, perm: &[usize]) -> Self {
        let perm_i64: Vec<i64> = perm.iter().map(|&p| p as i64).collect();
        let permuted = self.tensor.permute(&perm_i64);
        Self::from_tensor(permuted.contiguous())
    }
}

/// Convert axis IDs to einsum equation string.
///
/// Maps u32 IDs to letters a-zA-Z (max 52 unique labels).
/// Returns error if more than 52 unique IDs are needed.
#[allow(dead_code)]
pub fn ids_to_equation(input_ids: &[&[u32]], output_ids: &[u32]) -> Result<String> {
    use std::collections::HashMap;

    let mut id_to_char: HashMap<u32, char> = HashMap::new();
    let mut next_char = 0u8;

    let mut get_char = |id: u32| -> Result<char> {
        if let Some(&c) = id_to_char.get(&id) {
            return Ok(c);
        }
        if next_char >= 52 {
            anyhow::bail!(
                "Too many unique axis IDs ({}). Torch einsum supports max 52 (a-zA-Z).",
                id_to_char.len() + 1
            );
        }
        let c = if next_char < 26 {
            (b'a' + next_char) as char
        } else {
            (b'A' + next_char - 26) as char
        };
        id_to_char.insert(id, c);
        next_char += 1;
        Ok(c)
    };

    // Build input part of equation
    let mut parts: Vec<String> = Vec::new();
    for ids in input_ids {
        let chars: String = ids.iter().map(|&id| get_char(id)).collect::<Result<_>>()?;
        parts.push(chars);
    }

    // Build output part
    let output_chars: String = output_ids
        .iter()
        .map(|&id| get_char(id))
        .collect::<Result<_>>()?;

    Ok(format!("{}->{}", parts.join(","), output_chars))
}

/// Perform einsum on torch tensors.
///
/// This is a thin wrapper around `tch::Tensor::einsum`.
#[allow(dead_code)]
pub fn torch_einsum(equation: &str, tensors: &[&Tensor]) -> Result<Tensor> {
    let tensor_vec: Vec<Tensor> = tensors.iter().map(|t| (*t).shallow_clone()).collect();
    Ok(Tensor::einsum(equation, &tensor_vec, None::<&[i64]>))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_torch_storage_f64_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = TorchStorage::<f64>::from_vec_with_shape(data.clone(), &[2, 3]);

        assert_eq!(storage.dims(), vec![2, 3]);
        assert_eq!(storage.rank(), 2);
        assert_eq!(storage.len(), 6);
        assert!(!storage.is_empty());

        let retrieved = storage.to_vec();
        for (a, b) in data.iter().zip(retrieved.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_torch_storage_c64_basic() {
        let data = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
        ];
        let storage = TorchStorage::<Complex64>::from_vec_with_shape(data.clone(), &[3]);

        assert_eq!(storage.dims(), vec![3]);
        assert_eq!(storage.rank(), 1);
        assert_eq!(storage.len(), 3);

        let retrieved = storage.to_vec();
        for (a, b) in data.iter().zip(retrieved.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ids_to_equation() {
        // Simple matrix multiplication: ij,jk->ik
        let eq = ids_to_equation(&[&[0, 1], &[1, 2]], &[0, 2]).unwrap();
        assert_eq!(eq, "ab,bc->ac");

        // Batch matmul: bij,bjk->bik
        let eq = ids_to_equation(&[&[0, 1, 2], &[0, 2, 3]], &[0, 1, 3]).unwrap();
        assert_eq!(eq, "abc,acd->abd");

        // Inner product: i,i->
        let eq = ids_to_equation(&[&[0], &[0]], &[]).unwrap();
        assert_eq!(eq, "a,a->");
    }

    #[test]
    fn test_torch_einsum_matmul() {
        let a = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0])
            .view([2, 2])
            .to_kind(Kind::Double);
        let b = Tensor::from_slice(&[5.0f64, 6.0, 7.0, 8.0])
            .view([2, 2])
            .to_kind(Kind::Double);

        let result = torch_einsum("ij,jk->ik", &[&a, &b]).unwrap();

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        let result_vec = Vec::<f64>::try_from(&result.flatten(0, -1)).unwrap();

        for (a, b) in expected.iter().zip(result_vec.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_torch_storage_permute() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = TorchStorage::<f64>::from_vec_with_shape(data, &[2, 3]);

        let permuted = storage.permute(&[1, 0]); // Transpose

        assert_eq!(permuted.dims(), vec![3, 2]);
        // Original [2,3]: [[1,2,3],[4,5,6]]
        // Transposed [3,2]: [[1,4],[2,5],[3,6]]
        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let result = permuted.to_vec();
        for (a, b) in expected.iter().zip(result.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_torch_storage_autograd() {
        let data = vec![2.0];
        let mut storage = TorchStorage::<f64>::from_vec_with_shape(data, &[1]);
        storage.set_requires_grad(true);

        assert!(storage.requires_grad());

        // y = x^2
        let y_tensor = storage.tensor().pow_tensor_scalar(2.0);

        // backward
        y_tensor.backward();

        // grad should be 2*x = 4
        let grad = storage.grad().expect("Should have gradient");
        let grad_val = grad.get(0);
        assert_relative_eq!(grad_val, 4.0, epsilon = 1e-10);
    }
}
