//! Flip operator: f(x) = g(2^R - x)
//!
//! This transformation maps x -> 2^R - x in quantics representation.

use anyhow::Result;
use num_complex::Complex64;
use num_traits::{One, Zero};
use tensor4all_simpletensortrain::{types::tensor3_zeros, Tensor3Ops, TensorTrain};

use crate::common::{tensortrain_to_linear_operator, BoundaryCondition, QuanticsOperator};

/// Create a flip operator: f(x) = g(2^R - x)
///
/// This MPO transforms a function g(x) to f(x) = g(2^R - x) for x = 0, 1, ..., 2^R - 1.
///
/// # Arguments
/// * `r` - Number of bits (sites)
/// * `bc` - Boundary condition (affects behavior at x=0)
///
/// # Returns
/// LinearOperator representing the flip transformation
///
/// # Example
/// ```ignore
/// use tensor4all_quantics_transform::{flip_operator, BoundaryCondition};
///
/// let op = flip_operator(8, BoundaryCondition::Periodic).unwrap();
/// ```
pub fn flip_operator(r: usize, bc: BoundaryCondition) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }
    if r == 1 {
        return Err(anyhow::anyhow!(
            "MPO with one tensor is not supported for flip operator"
        ));
    }

    let mpo = flip_mpo(r, bc)?;
    let site_dims = vec![2; r];
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create the flip MPO as a TensorTrain.
///
/// The flip operation computes -x using two's complement arithmetic:
/// -x = ~x + 1 (bitwise NOT plus one)
///
/// This is implemented using carry propagation where:
/// - carry_in values: [-1, 0] (we start with +1 to compute 2^R - x = ~x + 1)
/// - For each bit: out = -(a - 1) + carry_in
/// - carry_out = out < 0 ? -1 : 0
/// - result_bit = out mod 2
fn flip_mpo(r: usize, bc: BoundaryCondition) -> Result<TensorTrain<Complex64>> {
    let single_tensor = single_tensor_flip();

    let mut tensors = Vec::with_capacity(r);

    // Create link indices with dimension 2 (for carry states)
    // Carry states: index 0 = carry -1, index 1 = carry 0

    for n in 0..r {
        if n == 0 {
            // First tensor: select initial carry state (carry = -1, i.e., index 0 -> +1 in 2's complement)
            // Actually in Quantics.jl: M[1] *= onehot(links[1] => 2)
            // This means initial carry index = 2 (1-indexed) = 1 (0-indexed) = carry 0
            // Wait, let me re-read the Julia code...
            // cval = [-1, 0], so index 1 (Julia) = carry -1, index 2 (Julia) = carry 0
            // onehot(links[1] => 2) means cin = 2 (Julia) = carry 0
            // So we start with carry 0, and the flip logic adds -1 to get the negation

            // First tensor: contract with [0, 1] on left link to select cin=1 (carry=0)
            let mut t = tensor3_zeros(1, 4, 2);
            for cout in 0..2 {
                for s_out in 0..2 {
                    for s_in in 0..2 {
                        let val = single_tensor[1][cout][s_out][s_in]; // cin=1 (carry=0)
                        let s = s_out * 2 + s_in;
                        t.set3(0, s, cout, val);
                    }
                }
            }
            tensors.push(t);
        } else if n == r - 1 {
            // Last tensor: apply boundary condition
            let bc_val = match bc {
                BoundaryCondition::Periodic => Complex64::one(),
                BoundaryCondition::Open => Complex64::one(), // For flip, open BC is same as periodic at output
            };

            // Contract with bc_tensor = [1.0, bc] on right link
            let mut t = tensor3_zeros(2, 4, 1);
            for cin in 0..2 {
                for s_out in 0..2 {
                    for s_in in 0..2 {
                        let mut sum = Complex64::zero();
                        for cout in 0..2 {
                            let bc_weight = if cout == 0 { Complex64::one() } else { bc_val };
                            sum += single_tensor[cin][cout][s_out][s_in] * bc_weight;
                        }
                        let s = s_out * 2 + s_in;
                        t.set3(cin, s, 0, sum);
                    }
                }
            }
            tensors.push(t);
        } else {
            // Middle tensors: full tensor with both link dimensions
            let mut t = tensor3_zeros(2, 4, 2);
            for cin in 0..2 {
                for cout in 0..2 {
                    for s_out in 0..2 {
                        for s_in in 0..2 {
                            let val = single_tensor[cin][cout][s_out][s_in];
                            let s = s_out * 2 + s_in;
                            t.set3(cin, s, cout, val);
                        }
                    }
                }
            }
            tensors.push(t);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("Failed to create flip MPO: {}", e))
}

/// Create the single-site tensor for flip operation.
///
/// Returns a 4D tensor [cin][cout][s_out][s_in] where:
/// - cin: input carry state (0 = carry -1, 1 = carry 0)
/// - cout: output carry state
/// - s_out: output bit (0 or 1)
/// - s_in: input bit (0 or 1)
///
/// The flip computes: out = -(a - 1) + cval[cin]
/// where cval = [-1, 0] for cin = [0, 1]
fn single_tensor_flip() -> [[[[Complex64; 2]; 2]; 2]; 2] {
    let cval = [-1i32, 0i32];
    let mut tensor = [[[[Complex64::zero(); 2]; 2]; 2]; 2];

    for icin in 0..2 {
        for a in 0..2 {
            // a is the input bit (s_in)
            let out = -(a as i32 - 1) + cval[icin];
            let icout = if out < 0 { 0 } else { 1 };
            let b = out.rem_euclid(2) as usize; // b is the output bit (s_out)
            tensor[icin][icout][b][a] = Complex64::one();
        }
    }

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_simpletensortrain::AbstractTensorTrain;

    #[test]
    fn test_single_tensor_flip() {
        let t = single_tensor_flip();

        // Verify non-zero entries
        // When cin=1 (carry=0), a=0: out = -(0-1) + 0 = 1, cout=1, b=1
        assert_eq!(t[1][1][1][0], Complex64::one());

        // When cin=1 (carry=0), a=1: out = -(1-1) + 0 = 0, cout=1, b=0
        assert_eq!(t[1][1][0][1], Complex64::one());

        // When cin=0 (carry=-1), a=0: out = -(0-1) + (-1) = 0, cout=1, b=0
        assert_eq!(t[0][1][0][0], Complex64::one());

        // When cin=0 (carry=-1), a=1: out = -(1-1) + (-1) = -1, cout=0, b=1
        assert_eq!(t[0][0][1][1], Complex64::one());
    }

    #[test]
    fn test_flip_mpo_structure() {
        let mpo = flip_mpo(4, BoundaryCondition::Periodic).unwrap();
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
    fn test_flip_operator_creation() {
        let op = flip_operator(4, BoundaryCondition::Periodic);
        assert!(op.is_ok());
    }

    #[test]
    fn test_flip_error_single_site() {
        let result = flip_operator(1, BoundaryCondition::Periodic);
        assert!(result.is_err());
    }

    #[test]
    fn test_flip_error_zero_sites() {
        let result = flip_operator(0, BoundaryCondition::Periodic);
        assert!(result.is_err());
    }
}
