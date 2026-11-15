//! Quantics Fourier Transform (QFT) operator
//!
//! This implements the Chen & Lindsey method for efficient QFT in tensor train form.
//! Reference: J. Chen and M. Lindsey, "Direct Interpolative Construction of the
//! Discrete Fourier Transform as a Matrix Product Operator", arXiv:2404.03182.

use crate::error::QuanticsTransformError;
use anyhow::Result;
use num_complex::Complex64;
use std::f64::consts::PI;
use tensor4all_core::{compress_chain_lu_mut, ColMajorArray};

use crate::common::{
    binary_cores_to_operator, checked_allocation_len, try_vec_with_capacity, QuanticsOperator,
};

fn checked_fourier_order(k: usize) -> Result<usize> {
    if k == 0 {
        return Err(anyhow::anyhow!(
            "Fourier Chebyshev order k must be positive"
        ));
    }
    k.checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("Fourier Chebyshev grid size overflows usize for k={k}"))
}

fn checked_tensor4_size(dims: [usize; 4], name: &str) -> Result<usize> {
    checked_allocation_len::<Complex64>(&dims, name)
}

fn checked_fourier_core_size(k: usize) -> Result<(usize, usize)> {
    let count = checked_fourier_order(k)?;
    let total = checked_tensor4_size([count, 2, 2, count], "Fourier core tensor")?;
    Ok((count, total))
}

fn checked_tensor3_size(
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
    name: &str,
) -> Result<usize> {
    checked_allocation_len::<Complex64>(&[left_dim, site_dim, right_dim], name)
}

pub(crate) fn zero_fourier_core(
    left_dim: usize,
    right_dim: usize,
    name: &str,
) -> Result<ColMajorArray<Complex64>> {
    let total_size = checked_tensor3_size(left_dim, 4, right_dim, name)?;
    let mut data = try_vec_with_capacity::<Complex64>(name, total_size)?;
    data.resize(total_size, Complex64::default());
    Ok(ColMajorArray::new(data, vec![left_dim, 4, right_dim])?)
}

/// Options for Fourier transform construction.
/// Controls the sign convention, compression parameters, and normalization
/// of the quantics Fourier transform MPO.
/// # Examples
/// ```
/// use tensor4all_quanticstransform::FourierOptions;
/// // Forward transform (default): sign = -1
/// let fwd = FourierOptions::forward();
/// assert_eq!(fwd.sign, -1.0);
/// assert!(fwd.normalize);
/// // Inverse transform: sign = +1
/// let inv = FourierOptions::inverse();
/// assert_eq!(inv.sign, 1.0);
/// // Custom options
/// let opts = FourierOptions {
///     max_bond_dim: Some(20),
///     tolerance: 1e-12,
///     ..FourierOptions::forward()
/// };
/// assert_eq!(opts.max_bond_dim, Some(20));
/// ```
#[derive(Clone, Debug)]
pub struct FourierOptions {
    /// Sign in the exponent: -1.0 (forward) or 1.0 (inverse)
    pub sign: f64,
    /// Maximum bond dimension after per-axis compression. `None` leaves it uncapped.
    pub max_bond_dim: Option<usize>,
    /// Relative LU pivot threshold. Zero selects the default floor of `1e-14`.
    pub tolerance: f64,
    /// Positive Chebyshev interpolation order, giving `k + 1` points.
    pub k: usize,
    /// Whether to scale each axis transform by `1/sqrt(2^r)`.
    pub normalize: bool,
}

impl Default for FourierOptions {
    fn default() -> Self {
        Self {
            sign: -1.0,
            max_bond_dim: Some(12),
            tolerance: 1e-14,
            k: 25,
            normalize: true,
        }
    }
}

impl FourierOptions {
    pub(crate) fn validate(&self) -> std::result::Result<(), QuanticsTransformError> {
        if self.sign != -1.0 && self.sign != 1.0 {
            return Err(QuanticsTransformError::InvalidConfiguration {
                message: "sign must be exactly -1.0 or 1.0".to_string(),
            });
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(QuanticsTransformError::InvalidConfiguration {
                message: "tolerance must be finite and nonnegative".to_string(),
            });
        }
        if self.max_bond_dim == Some(0) {
            return Err(QuanticsTransformError::InvalidConfiguration {
                message: "max_bond_dim must be positive when specified".to_string(),
            });
        }
        if self.k == 0 {
            return Err(QuanticsTransformError::InvalidConfiguration {
                message: "k must be positive".to_string(),
            });
        }
        Ok(())
    }

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
/// Caches compressed forward cores for [`FTCore::forward()`]. Each call to
/// [`FTCore::backward()`] constructs and compresses the inverse cores.
/// # Examples
/// ```
/// use tensor4all_quanticstransform::{FTCore, FourierOptions};
/// let ft = FTCore::new(4, FourierOptions::default()).unwrap();
/// assert_eq!(ft.r(), 4);
/// let fwd_op = ft.forward().unwrap();
/// assert_eq!(fwd_op.mpo().node_count(), 4);
/// let bwd_op = ft.backward().unwrap();
/// assert_eq!(bwd_op.mpo().node_count(), 4);
/// ```
#[derive(Clone)]
pub struct FTCore {
    forward_cores: Vec<ColMajorArray<Complex64>>,
    r: usize,
    options: FourierOptions,
}

impl FTCore {
    /// Create a new FTCore for r bits.
    ///
    /// # Errors
    ///
    /// Returns [`QuanticsTransformError::InvalidConfiguration`] for invalid
    /// Fourier options or site counts. It also returns an error when a
    /// dimension overflows.
    ///
    pub fn new(
        r: usize,
        options: FourierOptions,
    ) -> std::result::Result<Self, QuanticsTransformError> {
        options.validate()?;
        let forward_options = FourierOptions {
            sign: -1.0,
            ..options.clone()
        };
        let forward_cores = fourier_cores(r, &forward_options)?;
        Ok(Self {
            forward_cores,
            r,
            options,
        })
    }

    /// Get the forward Fourier transform operator.
    ///
    /// # Errors
    /// Returns [`QuanticsTransformError::Operation`] when the cached cores
    /// cannot be assembled into a linear operator.
    pub fn forward(&self) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
        binary_cores_to_operator(&self.forward_cores)
    }

    /// Get the backward (inverse) Fourier transform operator.
    ///
    /// # Errors
    ///
    /// Returns [`QuanticsTransformError::Operation`] when Fourier core
    /// construction or operator assembly fails.
    ///
    pub fn backward(&self) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
        let inverse_options = FourierOptions {
            sign: 1.0,
            normalize: self.options.normalize,
            ..self.options.clone()
        };
        binary_cores_to_operator(&fourier_cores(self.r, &inverse_options)?)
    }

    /// Get the number of bits.
    pub fn r(&self) -> usize {
        self.r
    }
}

/// Create a Quantics Fourier Transform operator.
/// This implements the Chen & Lindsey construction of the DFT as a matrix product operator.
/// The resulting operator transforms a quantics tensor train representing a function
/// to its Fourier transform in quantics tensor train form.
/// # Index ordering
/// Before the Fourier transform, the leftmost index corresponds to the most significant
/// bit (largest length scale). After transformation, the leftmost index corresponds to
/// the least significant bit (smallest length scale) - this allows for a small bond
/// dimension construction.
/// # Arguments
/// * `r` - Number of bits
/// * `options` - Fourier transform options
/// # Returns
/// LinearOperator representing the QFT
/// # Errors
///
/// Returns [`QuanticsTransformError::InvalidConfiguration`] for invalid
/// Fourier options or site counts. It also returns an error when operator
/// construction or allocation overflows.
///
/// # Examples
/// ```
/// use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
/// // Create a forward QFT operator for 4-bit quantics representation
/// let op = quantics_fourier_operator(4, FourierOptions::forward()).unwrap();
/// // The operator has one MPO tensor per bit
/// assert_eq!(op.mpo().node_count(), 4);
/// ```
pub fn quantics_fourier_operator(
    r: usize,
    options: FourierOptions,
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    binary_cores_to_operator(&fourier_cores(r, &options)?)
}

pub(crate) fn fourier_cores(
    r: usize,
    options: &FourierOptions,
) -> std::result::Result<Vec<ColMajorArray<Complex64>>, QuanticsTransformError> {
    options.validate()?;
    if r < 2 {
        return Err(QuanticsTransformError::InvalidConfiguration {
            message: format!("Number of sites must be at least 2, got {r}"),
        });
    }

    let (count, _) = checked_fourier_core_size(options.k)?;
    let mut cores = try_vec_with_capacity("Fourier MPO site list", r)?;
    let (grid, bary_weights) = chebyshev_grid(options.k)?;
    let bulk = build_dft_core_tensor(&grid, &bary_weights, options.sign)?;

    // Contract the leading alpha with ones and fix the trailing beta to zero.
    let mut first = zero_fourier_core(1, count, "Fourier first tensor")?;
    for (sum, values) in first
        .data_mut()
        .iter_mut()
        .zip(bulk.data().chunks_exact(count))
    {
        *sum = values.iter().copied().sum();
    }
    let mut last = zero_fourier_core(count, 1, "Fourier last tensor")?;
    last.data_mut().copy_from_slice(&bulk.data()[..count * 4]);
    cores.push(first);
    for _ in 2..r - 1 {
        let mut data = try_vec_with_capacity("Fourier middle tensor", bulk.len())?;
        data.extend_from_slice(bulk.data());
        cores.push(ColMajorArray::new(data, vec![count, 4, count]).map_err(anyhow::Error::from)?);
    }
    if r > 2 {
        cores.push(bulk);
    }
    cores.push(last);

    compress_chain_lu_mut(&mut cores, options.tolerance, options.max_bond_dim, true)
        .map_err(|err| anyhow::Error::new(err).context("Fourier MPO compression failed"))?;

    if options.normalize {
        let factor = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        for core in &mut cores {
            for value in core.data_mut() {
                *value *= factor;
            }
        }
    }

    Ok(cores)
}

/// Get Chebyshev grid points and barycentric weights.
/// Returns (grid, bary_weights) where:
/// - grid[j] = 0.5 * (1 - cos(π*j/K)) for j = 0, ..., K
/// - bary_weights are the barycentric interpolation weights
fn chebyshev_grid(k: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let count = checked_fourier_order(k)?;
    let mut grid = try_vec_with_capacity::<f64>("Fourier Chebyshev grid", count)?;
    let mut bary_weights = try_vec_with_capacity::<f64>("Fourier barycentric weights", count)?;

    // Compute Chebyshev grid points
    for j in 0..count {
        let x = 0.5 * (1.0 - (PI * j as f64 / k as f64).cos());
        grid.push(x);
    }

    // Compute barycentric weights
    for j in 0..count {
        let mut weight = 1.0;
        for m in 0..count {
            if j != m {
                weight /= grid[j] - grid[m];
            }
        }
        bary_weights.push(weight);
    }

    Ok((grid, bary_weights))
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

/// Build the Chen–Lindsey core A[alpha, tau, sigma, beta], with
/// the two physical bits fused as 2*tau + sigma (tau is output).
fn build_dft_core_tensor(
    grid: &[f64],
    bary_weights: &[f64],
    sign: f64,
) -> Result<ColMajorArray<Complex64>> {
    if grid.is_empty() || grid.len() != bary_weights.len() {
        anyhow::bail!("Fourier grid and barycentric weights have incompatible lengths");
    }
    let count = grid.len();
    let mut core = zero_fourier_core(count, count, "Fourier core tensor")?;
    for (beta, &point) in grid.iter().enumerate() {
        for tau in 0..2 {
            for sigma in 0..2 {
                let x = (sigma as f64 + point) / 2.0;
                let phase = 2.0 * PI * sign * x * tau as f64;
                let exp_phase = Complex64::new(phase.cos(), phase.sin());
                let start = count * (2 * tau + sigma + 4 * beta);
                for (alpha, value) in core.data_mut()[start..start + count].iter_mut().enumerate() {
                    let p_alpha = lagrange_polynomial(grid, bary_weights, alpha, x);
                    *value = Complex64::new(p_alpha, 0.0) * exp_phase;
                }
            }
        }
    }

    Ok(core)
}

#[cfg(test)]
mod tests;
