//! Typed tensor enum for handling F64/C64 mixed inputs in einsum.
//!
//! This module provides `TypedTensor` enum that wraps either f64 or Complex64 tensors,
//! allowing einsum to accept mixed-type inputs and automatically promote to Complex64
//! when necessary.

use mdarray::{DynRank, Tensor};
use num_complex::Complex64;

/// Enum wrapping either f64 or Complex64 tensor.
#[derive(Debug, Clone)]
pub enum TypedTensor {
    F64(Tensor<f64, DynRank>),
    C64(Tensor<Complex64, DynRank>),
}

impl TypedTensor {
    /// Create from f64 tensor.
    pub fn from_f64(t: Tensor<f64, DynRank>) -> Self {
        Self::F64(t)
    }

    /// Create from Complex64 tensor.
    pub fn from_c64(t: Tensor<Complex64, DynRank>) -> Self {
        Self::C64(t)
    }

    /// Check if this is an f64 tensor.
    pub fn is_f64(&self) -> bool {
        matches!(self, Self::F64(_))
    }

    /// Check if this is a Complex64 tensor.
    pub fn is_c64(&self) -> bool {
        matches!(self, Self::C64(_))
    }

    /// Get shape dimensions.
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::F64(t) => t.shape().dims(),
            Self::C64(t) => t.shape().dims(),
        }
    }

    /// Get rank.
    pub fn rank(&self) -> usize {
        match self {
            Self::F64(t) => t.rank(),
            Self::C64(t) => t.rank(),
        }
    }

    /// Promote to Complex64 if not already.
    pub fn to_c64(&self) -> Tensor<Complex64, DynRank> {
        match self {
            Self::F64(t) => {
                let data: Vec<Complex64> =
                    t.iter().copied().map(|x| Complex64::new(x, 0.0)).collect();
                Tensor::from(data).into_shape(t.shape().dims()).into_dyn()
            }
            Self::C64(t) => t.clone(),
        }
    }

    /// Get as f64 tensor if possible (returns None for C64).
    pub fn as_f64(&self) -> Option<&Tensor<f64, DynRank>> {
        match self {
            Self::F64(t) => Some(t),
            Self::C64(_) => None,
        }
    }

    /// Get as Complex64 tensor if already C64.
    pub fn as_c64(&self) -> Option<&Tensor<Complex64, DynRank>> {
        match self {
            Self::F64(_) => None,
            Self::C64(t) => Some(t),
        }
    }

    /// Convert f64 result back to f64 if all inputs were f64 and result is real.
    /// Otherwise keep as C64.
    pub fn try_demote_to_f64(t: Tensor<Complex64, DynRank>) -> Self {
        // Check if all imaginary parts are zero
        let is_real = t.iter().all(|c| c.im.abs() < 1e-15);
        if is_real {
            let data: Vec<f64> = t.iter().map(|c| c.re).collect();
            Self::F64(Tensor::from(data).into_shape(t.shape().dims()).into_dyn())
        } else {
            Self::C64(t)
        }
    }
}

/// Determine if any input requires Complex64 promotion.
pub fn needs_c64_promotion(inputs: &[TypedTensor]) -> bool {
    inputs.iter().any(|t| t.is_c64())
}

/// Perform Einstein summation on TypedTensor inputs.
///
/// This function automatically handles mixed F64/C64 inputs:
/// - If all inputs are F64, performs F64 einsum and returns TypedTensor::F64
/// - If any input is C64, promotes all to C64 and performs C64 einsum
///
/// # Arguments
/// * `backend` - Matrix multiplication backend
/// * `inputs` - Slice of (axis_ids, TypedTensor) pairs
/// * `output_ids` - Axis IDs for the output tensor
/// * `sizes` - Dimension sizes for each axis ID
///
/// # Returns
/// Result tensor as TypedTensor (F64 if all inputs were F64, else C64)
pub fn einsum_typed<ID>(
    backend: &impl crate::MatMul<f64>,
    backend_c64: &impl crate::MatMul<Complex64>,
    inputs: &[(&[ID], TypedTensor)],
    output_ids: &[ID],
    sizes: &std::collections::HashMap<ID, usize>,
) -> TypedTensor
where
    ID: crate::AxisId + omeco::Label,
{
    let tensors: Vec<&TypedTensor> = inputs.iter().map(|(_, t)| t).collect();
    let all_f64 = !needs_c64_promotion(&tensors.iter().cloned().cloned().collect::<Vec<_>>());

    if all_f64 {
        // All inputs are F64 - perform F64 einsum
        let f64_inputs: Vec<(&[ID], &mdarray::Slice<f64, DynRank, mdarray::Dense>)> = inputs
            .iter()
            .map(|(ids, t)| {
                let tensor_ref = t.as_f64().expect("Expected F64 tensor");
                (*ids, tensor_ref.as_ref())
            })
            .collect();
        let result = crate::einsum_optimized(backend, &f64_inputs, output_ids, sizes);
        TypedTensor::F64(result)
    } else {
        // Mixed types - promote all to C64
        let c64_tensors: Vec<Tensor<Complex64, DynRank>> =
            inputs.iter().map(|(_, t)| t.to_c64()).collect();
        let c64_inputs: Vec<(&[ID], &mdarray::Slice<Complex64, DynRank, mdarray::Dense>)> = inputs
            .iter()
            .zip(c64_tensors.iter())
            .map(|((ids, _), tensor)| (*ids, tensor.as_ref()))
            .collect();
        let result = crate::einsum_optimized(backend_c64, &c64_inputs, output_ids, sizes);
        TypedTensor::C64(result)
    }
}

/// Perform Einstein summation on TypedTensor inputs with hyperedge support.
///
/// Similar to `einsum_typed` but handles hyperedge cases where an index
/// appears in 3+ tensors.
///
/// # Arguments
/// * `backend` - Matrix multiplication backend for f64
/// * `backend_c64` - Matrix multiplication backend for Complex64
/// * `inputs` - Slice of (axis_ids, TypedTensor) pairs
/// * `output_ids` - Axis IDs for the output tensor
/// * `sizes` - Dimension sizes for each axis ID
///
/// # Returns
/// Result tensor as TypedTensor (F64 if all inputs were F64, else C64)
pub fn einsum_typed_simple<ID>(
    backend: &impl crate::MatMul<f64>,
    backend_c64: &impl crate::MatMul<Complex64>,
    inputs: &[(&[ID], TypedTensor)],
    output_ids: &[ID],
    _sizes: &std::collections::HashMap<ID, usize>,
) -> TypedTensor
where
    ID: crate::AxisId,
{
    let tensors: Vec<&TypedTensor> = inputs.iter().map(|(_, t)| t).collect();
    let all_f64 = !needs_c64_promotion(&tensors.iter().cloned().cloned().collect::<Vec<_>>());

    if all_f64 {
        // All inputs are F64 - perform F64 einsum
        let f64_inputs: Vec<(&[ID], &mdarray::Slice<f64, DynRank, mdarray::Dense>)> = inputs
            .iter()
            .map(|(ids, t)| {
                let tensor_ref = t.as_f64().expect("Expected F64 tensor");
                (*ids, tensor_ref.as_ref())
            })
            .collect();
        let result = crate::einsum(backend, &f64_inputs, output_ids);
        TypedTensor::F64(result)
    } else {
        // Mixed types - promote all to C64
        let c64_tensors: Vec<Tensor<Complex64, DynRank>> =
            inputs.iter().map(|(_, t)| t.to_c64()).collect();
        let c64_inputs: Vec<(&[ID], &mdarray::Slice<Complex64, DynRank, mdarray::Dense>)> = inputs
            .iter()
            .zip(c64_tensors.iter())
            .map(|((ids, _), tensor)| (*ids, tensor.as_ref()))
            .collect();
        let result = crate::einsum(backend_c64, &c64_inputs, output_ids);
        TypedTensor::C64(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mdarray::tensor;

    #[test]
    fn test_typed_tensor_f64() {
        let t = tensor![1.0, 2.0, 3.0].into_dyn();
        let typed = TypedTensor::from_f64(t);
        assert!(typed.is_f64());
        assert!(!typed.is_c64());
        assert_eq!(typed.dims(), &[3]);
    }

    #[test]
    fn test_typed_tensor_c64() {
        let t = tensor![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)].into_dyn();
        let typed = TypedTensor::from_c64(t);
        assert!(typed.is_c64());
        assert!(!typed.is_f64());
        assert_eq!(typed.dims(), &[2]);
    }

    #[test]
    fn test_promote_to_c64() {
        let t = tensor![1.0, 2.0].into_dyn();
        let typed = TypedTensor::from_f64(t);
        let promoted = typed.to_c64();
        assert_eq!(promoted[[0]], Complex64::new(1.0, 0.0));
        assert_eq!(promoted[[1]], Complex64::new(2.0, 0.0));
    }

    #[test]
    fn test_demote_to_f64() {
        // Real result should demote
        let c64_real = tensor![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)].into_dyn();
        let demoted = TypedTensor::try_demote_to_f64(c64_real);
        assert!(demoted.is_f64());

        // Complex result should stay C64
        let c64_complex = tensor![Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0)].into_dyn();
        let not_demoted = TypedTensor::try_demote_to_f64(c64_complex);
        assert!(not_demoted.is_c64());
    }

    #[test]
    fn test_needs_c64_promotion() {
        let f64_tensor = TypedTensor::from_f64(tensor![1.0].into_dyn());
        let c64_tensor = TypedTensor::from_c64(tensor![Complex64::new(1.0, 0.0)].into_dyn());

        assert!(!needs_c64_promotion(&[f64_tensor.clone()]));
        assert!(needs_c64_promotion(&[c64_tensor.clone()]));
        assert!(needs_c64_promotion(&[f64_tensor, c64_tensor]));
    }
}
