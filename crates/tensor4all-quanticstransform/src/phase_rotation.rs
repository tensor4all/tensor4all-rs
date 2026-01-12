//! Phase rotation operator: f(x) = exp(i*θ*x) * g(x)
//!
//! This transformation multiplies the function by a phase factor.

use anyhow::Result;
use num_complex::Complex64;
use num_traits::One;
use std::f64::consts::PI;
use tensor4all_simplett::{types::tensor3_zeros, Tensor3Ops, TensorTrain};

use crate::common::{tensortrain_to_linear_operator, QuanticsOperator};

/// Create a phase rotation operator: f(x) = exp(i*θ*x) * g(x)
///
/// This MPO multiplies a function g(x) by the phase factor exp(i*θ*x).
///
/// In quantics representation, x = Σ_n x_n * 2^(R-n), so:
/// exp(i*θ*x) = Π_n exp(i*θ*2^(R-n)*x_n)
///
/// Each site contributes an independent phase factor, making this a diagonal
/// operator with bond dimension 1.
///
/// # Arguments
/// * `r` - Number of bits (sites)
/// * `theta` - Phase angle in radians
///
/// # Returns
/// LinearOperator representing the phase rotation
///
/// # Example
/// ```ignore
/// use tensor4all_quantics_transform::phase_rotation_operator;
/// use std::f64::consts::PI;
///
/// // Apply phase rotation by π/4
/// let op = phase_rotation_operator(8, PI / 4.0).unwrap();
/// ```
pub fn phase_rotation_operator(r: usize, theta: f64) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }

    let mpo = phase_rotation_mpo(r, theta)?;
    let site_dims = vec![2; r];
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create the phase rotation MPO as a TensorTrain.
///
/// Each site tensor is diagonal with entries:
/// - For x_n = 0: 1
/// - For x_n = 1: exp(i*θ*2^(R-1-n))
///
/// Uses big-endian convention: site n corresponds to bit 2^(R-1-n) (MSB at site 0).
/// This matches Julia Quantics.jl's convention.
fn phase_rotation_mpo(r: usize, theta: f64) -> Result<TensorTrain<Complex64>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }

    // Normalize theta to [0, 2π)
    let theta_mod = theta.rem_euclid(2.0 * PI);

    let mut tensors = Vec::with_capacity(r);

    for n in 0..r {
        // Phase for this site: θ * 2^(R-1-n) (big-endian: site 0 = MSB)
        // This matches Julia's convention: site n=1 (Julia) has power R-n
        let power = (r - 1 - n) as f64;
        let site_phase = (theta_mod * 2.0_f64.powf(power)).rem_euclid(2.0 * PI);

        // Diagonal MPO tensor: out_bit == in_bit
        // Shape: (1, 4, 1) where 4 = 2*2 for (out, in)
        let mut t = tensor3_zeros(1, 4, 1);

        // s = out_bit * 2 + in_bit
        // Diagonal entries: out_bit == in_bit
        // s=0: (0,0), s=3: (1,1)
        t.set3(0, 0, 0, Complex64::one()); // x_n = 0: phase = 1

        // x_n = 1: phase = exp(i * site_phase)
        let phase_factor = Complex64::new(site_phase.cos(), site_phase.sin());
        t.set3(0, 3, 0, phase_factor);

        tensors.push(t);
    }

    TensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create phase rotation MPO: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use tensor4all_simplett::AbstractTensorTrain;

    #[test]
    fn test_phase_rotation_mpo_structure() {
        let mpo = phase_rotation_mpo(4, PI / 4.0).unwrap();
        assert_eq!(mpo.len(), 4);

        // All tensors should have shape (1, 4, 1) - diagonal operator
        for i in 0..4 {
            assert_eq!(mpo.site_tensor(i).left_dim(), 1);
            assert_eq!(mpo.site_tensor(i).site_dim(), 4);
            assert_eq!(mpo.site_tensor(i).right_dim(), 1);
        }
    }

    #[test]
    fn test_phase_rotation_zero_theta() {
        // θ = 0 should give identity
        let mpo = phase_rotation_mpo(4, 0.0).unwrap();

        for i in 0..4 {
            let t = mpo.site_tensor(i);
            // Check diagonal entries are 1
            assert_relative_eq!(t.get3(0, 0, 0).re, 1.0, epsilon = 1e-10);
            assert_relative_eq!(t.get3(0, 0, 0).im, 0.0, epsilon = 1e-10);
            assert_relative_eq!(t.get3(0, 3, 0).re, 1.0, epsilon = 1e-10);
            assert_relative_eq!(t.get3(0, 3, 0).im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_phase_rotation_pi() {
        // θ = π should give (-1)^x
        // With big-endian: site 0 (MSB) has phase π * 2^(R-1), site R-1 (LSB) has phase π * 2^0
        let r = 3;
        let mpo = phase_rotation_mpo(r, PI).unwrap();

        // For site 0 (MSB), phase = π * 2^2 = 4π ≡ 0 (mod 2π), exp(0) = 1
        let t_first = mpo.site_tensor(0);
        assert_relative_eq!(t_first.get3(0, 0, 0).re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(t_first.get3(0, 3, 0).re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(t_first.get3(0, 3, 0).im, 0.0, epsilon = 1e-10);

        // For site 2 (LSB), phase = π * 2^0 = π, exp(i*π) = -1
        let t_last = mpo.site_tensor(r - 1);
        assert_relative_eq!(t_last.get3(0, 0, 0).re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(t_last.get3(0, 3, 0).re, -1.0, epsilon = 1e-10);
        assert_relative_eq!(t_last.get3(0, 3, 0).im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_phase_rotation_operator_creation() {
        let op = phase_rotation_operator(4, PI / 2.0);
        assert!(op.is_ok());
    }

    #[test]
    fn test_phase_rotation_error_zero_sites() {
        let result = phase_rotation_operator(0, PI);
        assert!(result.is_err());
    }

    #[test]
    fn test_phase_rotation_periodicity() {
        // θ and θ + 2π should give the same result
        let mpo1 = phase_rotation_mpo(4, PI / 3.0).unwrap();
        let mpo2 = phase_rotation_mpo(4, PI / 3.0 + 2.0 * PI).unwrap();

        for i in 0..4 {
            let t1 = mpo1.site_tensor(i);
            let t2 = mpo2.site_tensor(i);

            assert_relative_eq!(t1.get3(0, 0, 0).re, t2.get3(0, 0, 0).re, epsilon = 1e-10);
            assert_relative_eq!(t1.get3(0, 0, 0).im, t2.get3(0, 0, 0).im, epsilon = 1e-10);
            assert_relative_eq!(t1.get3(0, 3, 0).re, t2.get3(0, 3, 0).re, epsilon = 1e-10);
            assert_relative_eq!(t1.get3(0, 3, 0).im, t2.get3(0, 3, 0).im, epsilon = 1e-10);
        }
    }
}
