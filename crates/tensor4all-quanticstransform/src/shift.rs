//! Shift operator: f(x) = g(x + offset) mod 2^R
//!
//! This transformation shifts the argument by a constant offset.

use anyhow::Result;
use num_complex::Complex64;
use num_traits::{One, Zero};
use tensor4all_simplett::{types::tensor3_zeros, Tensor3Ops, TensorTrain};

use crate::common::{
    embed_single_var_mpo, tensortrain_to_linear_operator,
    tensortrain_to_linear_operator_asymmetric, BoundaryCondition, QuanticsOperator,
};

/// Create a shift operator: f(x) = g(x + offset) mod 2^R
///
/// This MPO transforms a function g(x) to f(x) = g(x + offset) for x = 0, 1, ..., 2^R - 1.
///
/// # Arguments
/// * `r` - Number of bits (sites)
/// * `offset` - Shift amount (can be negative)
/// * `bc` - Boundary condition
///
/// # Returns
/// LinearOperator representing the shift transformation
///
/// # Example
/// ```no_run
/// use tensor4all_quanticstransform::{shift_operator, BoundaryCondition};
///
/// // Shift by 10 positions with periodic boundary
/// let op = shift_operator(8, 10, BoundaryCondition::Periodic).unwrap();
/// ```
pub fn shift_operator(r: usize, offset: i64, bc: BoundaryCondition) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }

    let mpo = shift_mpo(r, offset, bc)?;
    let site_dims = vec![2; r];
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create a shift operator for one variable in a multi-variable system.
///
/// Acts as shift on `target_var` and identity on all other variables.
///
/// # Arguments
/// * `r` - Number of bits (sites)
/// * `offset` - Shift amount (can be negative)
/// * `bc` - Boundary condition
/// * `nvariables` - Total number of variables
/// * `target_var` - Which variable to shift (0-indexed)
pub fn shift_operator_multivar(
    r: usize,
    offset: i64,
    bc: BoundaryCondition,
    nvariables: usize,
    target_var: usize,
) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }

    let mpo = shift_mpo(r, offset, bc)?;
    let embedded = embed_single_var_mpo(&mpo, nvariables, target_var)?;
    let dim_multi = 1 << nvariables;
    let dims = vec![dim_multi; r];
    tensortrain_to_linear_operator_asymmetric(&embedded, &dims, &dims)
}

/// Create the shift MPO as a TensorTrain.
///
/// The shift operation computes x + offset using binary addition with carry propagation.
/// Uses big-endian convention: site n contains bit 2^(R-1-n) (MSB at site 0).
/// This matches Julia Quantics.jl's convention.
///
/// At each site n, we compute: out_n = x_n + offset_n + carry_in (mod 2)
/// with carry_out = (x_n + offset_n + carry_in) / 2
///
/// Carry propagates from LSB to MSB, so in big-endian:
/// - Site 0 (MSB): applies BC on left, receives carry from right
/// - Site R-1 (LSB): initial carry = 0, sends carry to left
fn shift_mpo(r: usize, offset: i64, bc: BoundaryCondition) -> Result<TensorTrain<Complex64>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }

    let n_max = 1i64 << r;

    // Normalize offset to [0, 2^R)
    let (nbc, offset_mod) = {
        let offset_mod = offset.rem_euclid(n_max);
        let nbc = (offset - offset_mod) / n_max;
        (nbc, offset_mod as usize)
    };

    // Convert offset to binary (big-endian: MSB first)
    // Site n contains bit 2^(R-1-n)
    // offset_bits[n] = bit at position (R-1-n)
    let offset_bits: Vec<usize> = (0..r).map(|n| (offset_mod >> (r - 1 - n)) & 1).collect();

    let mut tensors = Vec::with_capacity(r);

    // Carry states: index 0 = carry 0, index 1 = carry 1
    // For addition, carry can be 0 or 1.
    //
    // In big-endian with TensorTrain (left-to-right contraction):
    // - Carry flows right-to-left (LSB at R-1 to MSB at 0)
    // - t[left, s, right] where left = carry_out (going left), right = carry_in (from right)

    #[allow(clippy::needless_range_loop)]
    for n in 0..r {
        let y_bit = offset_bits[n]; // The constant bit at position (R-1-n)

        if r == 1 {
            // Single site case: no carry propagation needed
            let mut t = tensor3_zeros(1, 4, 1);
            for x_bit in 0..2 {
                let sum = x_bit + y_bit;
                let out_bit = sum % 2;
                let bc_factor = match bc {
                    BoundaryCondition::Periodic => Complex64::one(),
                    BoundaryCondition::Open => {
                        if sum >= 2 {
                            Complex64::zero()
                        } else {
                            Complex64::one()
                        }
                    }
                };
                let s = out_bit * 2 + x_bit;
                t.set3(0, s, 0, bc_factor);
            }
            tensors.push(t);
        } else if n == 0 {
            // First tensor (MSB): apply BC on left, receive carry from right
            // Shape (1, 4, 2): left=1 (BC applied), site=4, right=2 (carry_in from site 1)
            let bc_val = match bc {
                BoundaryCondition::Periodic => Complex64::one(),
                BoundaryCondition::Open => Complex64::zero(),
            };

            let mut t = tensor3_zeros(1, 4, 2);
            for carry_in in 0..2 {
                for x_bit in 0..2 {
                    let sum = x_bit + y_bit + carry_in;
                    let out_bit = sum % 2;
                    let carry_out = sum / 2;

                    // Weight by boundary condition based on carry_out
                    let weight = if carry_out == 0 {
                        Complex64::one()
                    } else {
                        bc_val
                    };

                    let s = out_bit * 2 + x_bit;
                    t.set3(0, s, carry_in, weight);
                }
            }
            tensors.push(t);
        } else if n == r - 1 {
            // Last tensor (LSB): initial carry = 0, send carry_out to left
            // Shape (2, 4, 1): left=2 (carry_out to site R-2), site=4, right=1 (no input)
            let mut t = tensor3_zeros(2, 4, 1);
            for x_bit in 0..2 {
                let sum = x_bit + y_bit; // carry_in = 0 at start
                let out_bit = sum % 2;
                let carry_out = sum / 2;
                let s = out_bit * 2 + x_bit;
                t.set3(carry_out, s, 0, Complex64::one());
            }
            tensors.push(t);
        } else {
            // Middle tensors: receive carry from right, send carry to left
            // Shape (2, 4, 2): left=2 (carry_out), site=4, right=2 (carry_in)
            let mut t = tensor3_zeros(2, 4, 2);
            for carry_in in 0..2 {
                for x_bit in 0..2 {
                    let sum = x_bit + y_bit + carry_in;
                    let out_bit = sum % 2;
                    let carry_out = sum / 2;
                    let s = out_bit * 2 + x_bit;
                    t.set3(carry_out, s, carry_in, Complex64::one());
                }
            }
            tensors.push(t);
        }
    }

    let mut mpo = TensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create shift MPO: {}", e))?;

    // Apply overall boundary condition factor for number of full cycles
    if nbc != 0 {
        let bc_factor = match bc {
            BoundaryCondition::Periodic => Complex64::one(),
            BoundaryCondition::Open => {
                // `nbc` is an Euclidean quotient, so negative offsets in (-n_max, 0)
                // still produce `nbc = -1`. Only true full-cycle offsets should zero.
                if offset >= n_max || offset <= -n_max {
                    Complex64::zero()
                } else {
                    Complex64::one()
                }
            }
        };
        mpo.scale(bc_factor);
    }

    Ok(mpo)
}

#[cfg(test)]
mod tests;
