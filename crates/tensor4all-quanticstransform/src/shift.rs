//! Shift operator: f(x) = g(x + offset) mod 2^R
//!
//! This transformation shifts the argument by a constant offset.

use anyhow::Result;
use num_complex::Complex64;
use num_traits::{One, Zero};
use tensor4all_simplett::{types::tensor3_zeros, Tensor3Ops, TensorTrain};

use crate::common::{tensortrain_to_linear_operator, BoundaryCondition, QuanticsOperator};

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
/// ```ignore
/// use tensor4all_quantics_transform::{shift_operator, BoundaryCondition};
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

/// Create the shift MPO as a TensorTrain.
///
/// The shift operation computes x + offset using binary addition with carry propagation.
/// Uses little-endian convention: site n contains bit 2^n (LSB at site 0).
///
/// At each site n, we compute: out_n = x_n + offset_n + carry_in (mod 2)
/// with carry_out = (x_n + offset_n + carry_in) / 2
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

    // Convert offset to binary (little-endian: LSB first)
    // offset = Σ_n offset_bits[n] * 2^n
    let offset_bits: Vec<usize> = (0..r).map(|n| (offset_mod >> n) & 1).collect();

    let mut tensors = Vec::with_capacity(r);

    // Carry states: index 0 = carry 0, index 1 = carry 1
    // For addition, carry can be 0 or 1 (since x + offset with offset >= 0).
    //
    // With little-endian convention:
    // - Site 0 is LSB, site R-1 is MSB
    // - Carry propagates from LSB to MSB (site 0 -> site R-1)
    // - First tensor (n=0, LSB): no carry input, has carry output
    // - Last tensor (n=R-1, MSB): has carry input, apply BC

    for n in 0..r {
        let y_bit = offset_bits[n]; // The constant bit at position n

        if r == 1 {
            // Single site case: no carry propagation needed
            let mut t = tensor3_zeros(1, 4, 1);
            for x_bit in 0..2 {
                let sum = x_bit + y_bit;
                let out_bit = sum % 2;
                let _carry = sum / 2;
                // For single site, boundary condition handles carry overflow
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
            // First tensor (LSB): no carry input, has carry output
            // Initial carry is 0 (implicit)
            let mut t = tensor3_zeros(1, 4, 2);
            for x_bit in 0..2 {
                let sum = x_bit + y_bit; // carry_in = 0 at start
                let out_bit = sum % 2;
                let carry_out = sum / 2;
                let s = out_bit * 2 + x_bit;
                t.set3(0, s, carry_out, Complex64::one());
            }
            tensors.push(t);
        } else if n == r - 1 {
            // Last tensor (MSB): has carry input, apply boundary condition
            let bc_val = match bc {
                BoundaryCondition::Periodic => Complex64::one(),
                BoundaryCondition::Open => Complex64::zero(),
            };

            let mut t = tensor3_zeros(2, 4, 1);
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
                    t.set3(carry_in, s, 0, weight);
                }
            }
            tensors.push(t);
        } else {
            // Middle tensors: both carry input and output
            let mut t = tensor3_zeros(2, 4, 2);
            for carry_in in 0..2 {
                for x_bit in 0..2 {
                    let sum = x_bit + y_bit + carry_in;
                    let out_bit = sum % 2;
                    let carry_out = sum / 2;
                    let s = out_bit * 2 + x_bit;
                    t.set3(carry_in, s, carry_out, Complex64::one());
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
                if nbc > 0 {
                    Complex64::zero() // Shifted out of bounds
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
mod tests {
    use super::*;
    use tensor4all_simplett::AbstractTensorTrain;

    #[test]
    fn test_shift_mpo_structure() {
        let mpo = shift_mpo(4, 5, BoundaryCondition::Periodic).unwrap();
        assert_eq!(mpo.len(), 4);

        // First tensor: shape (1, 4, 2)
        assert_eq!(mpo.site_tensor(0).left_dim(), 1);
        assert_eq!(mpo.site_tensor(0).site_dim(), 4);
        assert_eq!(mpo.site_tensor(0).right_dim(), 2);

        // Middle tensors: shape (2, 4, 2)
        assert_eq!(mpo.site_tensor(1).left_dim(), 2);
        assert_eq!(mpo.site_tensor(1).site_dim(), 4);
        assert_eq!(mpo.site_tensor(1).right_dim(), 2);

        // Last tensor: shape (2, 4, 1)
        assert_eq!(mpo.site_tensor(3).left_dim(), 2);
        assert_eq!(mpo.site_tensor(3).site_dim(), 4);
        assert_eq!(mpo.site_tensor(3).right_dim(), 1);
    }

    #[test]
    fn test_shift_zero() {
        // Shift by 0 should be identity-like
        let mpo = shift_mpo(4, 0, BoundaryCondition::Periodic).unwrap();
        assert_eq!(mpo.len(), 4);

        // Verify identity behavior: out_bit == x_bit when offset=0 and carry=0
        let t0 = mpo.site_tensor(0);
        // s = out_bit * 2 + x_bit
        // For x_bit=0, offset_bit=0: sum=0, out_bit=0, carry=0 -> s=0, cout=0
        assert_eq!(*t0.get3(0, 0, 0), Complex64::one());
        // For x_bit=1, offset_bit=0: sum=1, out_bit=1, carry=0 -> s=3, cout=0
        assert_eq!(*t0.get3(0, 3, 0), Complex64::one());
    }

    #[test]
    fn test_shift_operator_creation() {
        let op = shift_operator(4, 3, BoundaryCondition::Periodic);
        assert!(op.is_ok());
    }

    #[test]
    fn test_shift_negative() {
        // Negative shift should work with modular arithmetic
        let mpo = shift_mpo(4, -1, BoundaryCondition::Periodic).unwrap();
        assert_eq!(mpo.len(), 4);
        // -1 mod 16 = 15 = 1111 in binary
    }

    #[test]
    fn test_shift_single_site() {
        let mpo = shift_mpo(1, 1, BoundaryCondition::Periodic).unwrap();
        assert_eq!(mpo.len(), 1);
        assert_eq!(mpo.site_tensor(0).left_dim(), 1);
        assert_eq!(mpo.site_tensor(0).site_dim(), 4);
        assert_eq!(mpo.site_tensor(0).right_dim(), 1);
    }

    #[test]
    fn test_shift_error_zero_sites() {
        let result = shift_operator(0, 0, BoundaryCondition::Periodic);
        assert!(result.is_err());
    }
}
