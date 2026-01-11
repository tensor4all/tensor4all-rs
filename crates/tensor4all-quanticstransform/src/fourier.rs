//! Quantics Fourier Transform (QFT) operator
//!
//! This implements the Chen & Lindsey method for efficient QFT in tensor train form.
//! Reference: J. Chen and M. Lindsey, "Direct Interpolative Construction of the
//! Discrete Fourier Transform as a Matrix Product Operator", arXiv:2404.03182.

use anyhow::Result;
use num_complex::Complex64;
use num_traits::Zero;
use std::f64::consts::PI;
use tensor4all_simplett::{
    compression::{CompressionMethod, CompressionOptions},
    types::tensor3_zeros,
    Tensor3Ops, TensorTrain,
};

use crate::common::{tensortrain_to_linear_operator, QuanticsOperator};

/// Options for Fourier transform construction.
#[derive(Clone, Debug)]
pub struct FourierOptions {
    /// Sign in the exponent: -1.0 (forward) or 1.0 (inverse)
    pub sign: f64,
    /// Maximum bond dimension after compression
    pub maxbonddim: usize,
    /// Tolerance for compression
    pub tolerance: f64,
    /// Number of Chebyshev basis functions (K+1 points)
    pub k: usize,
    /// Whether to normalize as an isometry
    pub normalize: bool,
}

impl Default for FourierOptions {
    fn default() -> Self {
        Self {
            sign: -1.0,
            maxbonddim: 12,
            tolerance: 1e-14,
            k: 25,
            normalize: true,
        }
    }
}

impl FourierOptions {
    /// Create options for forward Fourier transform.
    pub fn forward() -> Self {
        Self::default()
    }

    /// Create options for inverse Fourier transform.
    pub fn inverse() -> Self {
        Self {
            sign: 1.0,
            ..Self::default()
        }
    }
}

/// Convenience wrapper for forward/backward Fourier transform.
#[derive(Clone)]
pub struct FTCore {
    forward_mpo: TensorTrain<Complex64>,
    r: usize,
    options: FourierOptions,
}

impl FTCore {
    /// Create a new FTCore for r bits.
    pub fn new(r: usize, options: FourierOptions) -> Result<Self> {
        let forward_options = FourierOptions {
            sign: -1.0,
            ..options.clone()
        };
        let forward_mpo = quantics_fourier_mpo(r, &forward_options)?;
        Ok(Self {
            forward_mpo,
            r,
            options,
        })
    }

    /// Get the forward Fourier transform operator.
    pub fn forward(&self) -> Result<QuanticsOperator> {
        let site_dims = vec![2; self.r];
        tensortrain_to_linear_operator(&self.forward_mpo, &site_dims)
    }

    /// Get the backward (inverse) Fourier transform operator.
    pub fn backward(&self) -> Result<QuanticsOperator> {
        let inverse_options = FourierOptions {
            sign: 1.0,
            normalize: self.options.normalize,
            ..self.options.clone()
        };
        let inverse_mpo = quantics_fourier_mpo(self.r, &inverse_options)?;
        let site_dims = vec![2; self.r];
        tensortrain_to_linear_operator(&inverse_mpo, &site_dims)
    }

    /// Get the number of bits.
    pub fn r(&self) -> usize {
        self.r
    }
}

/// Create a Quantics Fourier Transform operator.
///
/// This implements the Chen & Lindsey construction of the DFT as a matrix product operator.
/// The resulting operator transforms a quantics tensor train representing a function
/// to its Fourier transform in quantics tensor train form.
///
/// # Index ordering
///
/// Before the Fourier transform, the leftmost index corresponds to the most significant
/// bit (largest length scale). After transformation, the leftmost index corresponds to
/// the least significant bit (smallest length scale) - this allows for a small bond
/// dimension construction.
///
/// # Arguments
/// * `r` - Number of bits
/// * `options` - Fourier transform options
///
/// # Returns
/// LinearOperator representing the QFT
pub fn quantics_fourier_operator(r: usize, options: FourierOptions) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }

    let mpo = quantics_fourier_mpo(r, &options)?;
    let site_dims = vec![2; r];
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create the QFT MPO as a TensorTrain using Chen & Lindsey construction.
fn quantics_fourier_mpo(r: usize, options: &FourierOptions) -> Result<TensorTrain<Complex64>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }

    let k = options.k;
    let sign = options.sign;

    // Get Chebyshev grid and barycentric weights
    let (grid, bary_weights) = chebyshev_grid(k);

    // Build core tensor A[alpha, tau, sigma, beta]
    // alpha, beta in 0..=K (K+1 values each)
    // tau, sigma in 0..1 (2 values each)
    let core_tensor = build_dft_core_tensor(&grid, &bary_weights, sign);

    // Construct tensor train
    let mut tensors = Vec::with_capacity(r);

    // First tensor: sum over alpha (contract with ones vector)
    // Shape: (1, 4, K+1) where 4 = 2*2 for (tau, sigma)
    {
        let mut t = tensor3_zeros(1, 4, k + 1);
        for tau in 0..2 {
            for sigma in 0..2 {
                for beta in 0..=k {
                    let mut sum = Complex64::zero();
                    for alpha in 0..=k {
                        sum += core_tensor[alpha][tau][sigma][beta];
                    }
                    let s = tau * 2 + sigma;
                    t.set3(0, s, beta, sum);
                }
            }
        }
        tensors.push(t);
    }

    // Middle tensors: full core tensor
    // Shape: (K+1, 4, K+1)
    for _ in 1..r - 1 {
        let mut t = tensor3_zeros(k + 1, 4, k + 1);
        for alpha in 0..=k {
            for tau in 0..2 {
                for sigma in 0..2 {
                    for beta in 0..=k {
                        let s = tau * 2 + sigma;
                        t.set3(alpha, s, beta, core_tensor[alpha][tau][sigma][beta]);
                    }
                }
            }
        }
        tensors.push(t);
    }

    // Last tensor: select beta = 0
    // Shape: (K+1, 4, 1)
    if r > 1 {
        let mut t = tensor3_zeros(k + 1, 4, 1);
        for alpha in 0..=k {
            for tau in 0..2 {
                for sigma in 0..2 {
                    let s = tau * 2 + sigma;
                    t.set3(alpha, s, 0, core_tensor[alpha][tau][sigma][0]);
                }
            }
        }
        tensors.push(t);
    }

    let mut tt = TensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create Fourier MPO: {}", e))?;

    // Compress the tensor train
    let compress_options = CompressionOptions {
        method: CompressionMethod::LU,
        tolerance: options.tolerance,
        max_bond_dim: options.maxbonddim,
        normalize_error: true,
    };
    let _ = tt.compress(&compress_options);

    // Normalize if requested
    if options.normalize {
        let norm_factor = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        for tensor in tt.site_tensors_mut() {
            let (left_dim, site_dim, right_dim) =
                (tensor.left_dim(), tensor.site_dim(), tensor.right_dim());
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        let val = *tensor.get3(l, s, r);
                        tensor.set3(l, s, r, val * norm_factor);
                    }
                }
            }
        }
    }

    Ok(tt)
}

/// Get Chebyshev grid points and barycentric weights.
///
/// Returns (grid, bary_weights) where:
/// - grid[j] = 0.5 * (1 - cos(π*j/K)) for j = 0, ..., K
/// - bary_weights are the barycentric interpolation weights
fn chebyshev_grid(k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut grid = Vec::with_capacity(k + 1);
    let mut bary_weights = Vec::with_capacity(k + 1);

    // Compute Chebyshev grid points
    for j in 0..=k {
        let x = 0.5 * (1.0 - (PI * j as f64 / k as f64).cos());
        grid.push(x);
    }

    // Compute barycentric weights
    for j in 0..=k {
        let mut weight = 1.0;
        for m in 0..=k {
            if j != m {
                weight /= grid[j] - grid[m];
            }
        }
        bary_weights.push(weight);
    }

    (grid, bary_weights)
}

/// Evaluate Lagrange polynomial P_alpha(x).
fn lagrange_polynomial(grid: &[f64], bary_weights: &[f64], alpha: usize, x: f64) -> f64 {
    // Check if x is very close to grid[alpha]
    if (x - grid[alpha]).abs() < 1e-14 {
        return 1.0;
    }

    // Compute product term
    let mut prod = 1.0;
    for &g in grid {
        prod *= x - g;
    }

    prod * bary_weights[alpha] / (x - grid[alpha])
}

/// Build the DFT core tensor A[alpha, tau, sigma, beta].
///
/// A[alpha, tau, sigma, beta] = P_alpha(x) * exp(2πi * sign * x * tau)
/// where x = (sigma + grid[beta]) / 2
fn build_dft_core_tensor(
    grid: &[f64],
    bary_weights: &[f64],
    sign: f64,
) -> Vec<Vec<Vec<Vec<Complex64>>>> {
    let k = grid.len() - 1;

    let mut tensor = vec![vec![vec![vec![Complex64::zero(); k + 1]; 2]; 2]; k + 1];

    for alpha in 0..=k {
        for tau in 0..2 {
            for sigma in 0..2 {
                for beta in 0..=k {
                    let x = (sigma as f64 + grid[beta]) / 2.0;
                    let p_alpha = lagrange_polynomial(grid, bary_weights, alpha, x);
                    let phase = 2.0 * PI * sign * x * tau as f64;
                    let exp_phase = Complex64::new(phase.cos(), phase.sin());
                    tensor[alpha][tau][sigma][beta] = Complex64::new(p_alpha, 0.0) * exp_phase;
                }
            }
        }
    }

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use tensor4all_simplett::AbstractTensorTrain;

    #[test]
    fn test_chebyshev_grid() {
        let (grid, weights) = chebyshev_grid(4);

        // Check endpoints
        assert_relative_eq!(grid[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(grid[4], 1.0, epsilon = 1e-10);

        // Check symmetry around 0.5
        assert_relative_eq!(grid[1] + grid[3], 1.0, epsilon = 1e-10);
        assert_relative_eq!(grid[2], 0.5, epsilon = 1e-10);

        // Weights should be non-zero
        for w in &weights {
            assert!(w.abs() > 1e-20);
        }
    }

    #[test]
    fn test_lagrange_polynomial_at_grid_points() {
        let (grid, weights) = chebyshev_grid(5);

        // P_alpha(grid[alpha]) = 1
        for alpha in 0..=5 {
            let val = lagrange_polynomial(&grid, &weights, alpha, grid[alpha]);
            assert_relative_eq!(val, 1.0, epsilon = 1e-10);
        }

        // P_alpha(grid[beta]) = 0 for alpha != beta
        for alpha in 0..=5 {
            for beta in 0..=5 {
                if alpha != beta {
                    let val = lagrange_polynomial(&grid, &weights, alpha, grid[beta]);
                    assert_relative_eq!(val, 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_fourier_mpo_structure() {
        let options = FourierOptions::default();
        let mpo = quantics_fourier_mpo(4, &options).unwrap();
        assert_eq!(mpo.len(), 4);

        // Bond dimensions should be compressed from K+1
        // After compression, they should be <= maxbonddim
        for i in 0..3 {
            assert!(mpo.site_tensor(i).right_dim() <= options.maxbonddim);
        }
    }

    #[test]
    fn test_fourier_operator_creation() {
        let options = FourierOptions::default();
        let op = quantics_fourier_operator(4, options);
        assert!(op.is_ok());
    }

    #[test]
    fn test_ftcore_creation() {
        let options = FourierOptions::default();
        let ft = FTCore::new(4, options);
        assert!(ft.is_ok());

        let ft = ft.unwrap();
        assert_eq!(ft.r(), 4);

        let forward = ft.forward();
        assert!(forward.is_ok());

        let backward = ft.backward();
        assert!(backward.is_ok());
    }

    #[test]
    fn test_fourier_inverse_sign() {
        let forward_options = FourierOptions::forward();
        let inverse_options = FourierOptions::inverse();

        assert_eq!(forward_options.sign, -1.0);
        assert_eq!(inverse_options.sign, 1.0);
    }

    #[test]
    fn test_fourier_error_zero_sites() {
        let options = FourierOptions::default();
        let result = quantics_fourier_operator(0, options);
        assert!(result.is_err());
    }
}
