//! Binary operation operator: computes a*x + b*y for quantics variables.
//!
//! This implements the binary operation transformation for two-variable functions.
//! The transformation computes linear combinations of the form a*x + b*y where
//! a, b ∈ {-1, 0, 1}.

use anyhow::Result;
use mdarray::DTensor;
use num_complex::Complex64;
use num_traits::{One, Zero};
use tensor4all_simplett::{types::tensor3_zeros, Tensor3Ops, TensorTrain};

use crate::common::{tensortrain_to_linear_operator, BoundaryCondition, QuanticsOperator};

/// Coefficients for binary operation.
/// Each coefficient must be -1, 0, or 1.
/// The pair (a, b) = (-1, -1) is not directly supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryCoeffs {
    /// Coefficient for the first variable x. Must be -1, 0, or 1.
    pub a: i8,
    /// Coefficient for the second variable y. Must be -1, 0, or 1.
    pub b: i8,
}

impl BinaryCoeffs {
    /// Create new binary coefficients.
    /// Returns error if |a| > 1, |b| > 1, or (a, b) == (-1, -1).
    pub fn new(a: i8, b: i8) -> Result<Self> {
        if a.abs() > 1 {
            return Err(anyhow::anyhow!("a must be -1, 0, or 1"));
        }
        if b.abs() > 1 {
            return Err(anyhow::anyhow!("b must be -1, 0, or 1"));
        }
        if a == -1 && b == -1 {
            return Err(anyhow::anyhow!("(a, b) = (-1, -1) is not supported"));
        }
        Ok(Self { a, b })
    }

    /// Create identity transformation for first variable: (a, b) = (1, 0)
    pub fn select_x() -> Self {
        Self { a: 1, b: 0 }
    }

    /// Create identity transformation for second variable: (a, b) = (0, 1)
    pub fn select_y() -> Self {
        Self { a: 0, b: 1 }
    }

    /// Create sum transformation: (a, b) = (1, 1)
    pub fn sum() -> Self {
        Self { a: 1, b: 1 }
    }

    /// Create difference transformation: (a, b) = (1, -1)
    pub fn difference() -> Self {
        Self { a: 1, b: -1 }
    }
}

/// Create a binary operation operator for two-variable quantics representation.
///
/// This operator transforms a function g(x, y) to f(x, y) = g(a1*x + b1*y, a2*x + b2*y)
/// where the variables x and y are in interleaved quantics representation:
/// sites = [x_1, y_1, x_2, y_2, ..., x_R, y_R]
///
/// # Arguments
/// * `r` - Number of bits per variable (total sites = 2*r)
/// * `coeffs1` - Coefficients (a1, b1) for first output variable
/// * `coeffs2` - Coefficients (a2, b2) for second output variable
/// * `bc` - Boundary conditions for [first_output, second_output]
///
/// # Returns
/// LinearOperator representing the binary transformation
///
/// # Example
/// ```ignore
/// use tensor4all_quantics_transform::{binaryop_operator, BinaryCoeffs, BoundaryCondition};
///
/// // Transform g(x, y) -> g(x+y, x-y)
/// let coeffs1 = BinaryCoeffs::sum();       // x + y
/// let coeffs2 = BinaryCoeffs::difference(); // x - y
/// let bc = [BoundaryCondition::Periodic, BoundaryCondition::Periodic];
/// let op = binaryop_operator(8, coeffs1, coeffs2, bc).unwrap();
/// ```
pub fn binaryop_operator(
    r: usize,
    coeffs1: BinaryCoeffs,
    coeffs2: BinaryCoeffs,
    bc: [BoundaryCondition; 2],
) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of bits must be positive"));
    }

    let mpo = binaryop_mpo(r, coeffs1, coeffs2, bc)?;
    // Interleaved sites: 2*r sites, each with dimension 2
    let site_dims = vec![2; 2 * r];
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create the binary operation MPO as a TensorTrain.
///
/// The MPO operates on interleaved sites [x_1, y_1, x_2, y_2, ..., x_R, y_R]
/// and computes two output variables:
/// - out1 = a1*x + b1*y
/// - out2 = a2*x + b2*y
fn binaryop_mpo(
    r: usize,
    coeffs1: BinaryCoeffs,
    coeffs2: BinaryCoeffs,
    bc: [BoundaryCondition; 2],
) -> Result<TensorTrain<Complex64>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of bits must be positive"));
    }

    let bc_val1: i8 = match bc[0] {
        BoundaryCondition::Periodic => 1,
        BoundaryCondition::Open => 0,
    };
    let bc_val2: i8 = match bc[1] {
        BoundaryCondition::Periodic => 1,
        BoundaryCondition::Open => 0,
    };

    let a1 = coeffs1.a;
    let b1 = coeffs1.b;
    let a2 = coeffs2.a;
    let b2 = coeffs2.b;

    let mut tensors = Vec::with_capacity(2 * r);

    for n in 0..r {
        // Carry flows LSB→MSB (right-to-left in TT), matching shift_mpo convention.
        // cin enters from the RIGHT (from position n+1, less significant bits).
        // cout exits to the LEFT (to position n-1, more significant bits).
        // At MSB (n=0): cout absorbed by BC (cout_on=false), cin from right (cin_on=true).
        // At LSB (n=r-1): cin=0 (cin_on=false), cout goes left (cout_on=true).
        let cin_on = n < r - 1; // cin from right, not at LSB
        let cout_on = n > 0; // cout to left, not at MSB

        let cin_dim = if cin_on { 3usize } else { 1 };
        let cout_dim = if cout_on { 3usize } else { 1 };

        // Left bond carries cout (toward MSB), right bond carries cin (from LSB)
        let left_bond = cout_dim * cout_dim; // (cout1, cout2)
        let right_bond = cin_dim * cin_dim; // (cin1, cin2)

        // Mid-bond encodes (cout_combined, z1, x) = left_bond * 4
        let mid_bond = left_bond * 4;

        // Get individual carry tensors for each output
        // Shape: (cin_dim, cout_dim, 2, 2, 2) = [cin, cout, x, y, out]
        let t1 = binaryop_tensor_single(a1, b1, cin_on, cout_on, bc_val1);
        let t2 = binaryop_tensor_single(a2, b2, cin_on, cout_on, bc_val2);

        // X-site tensor: T_x[left=cout_combined, s_x=z1*2+x, mid]
        // Pass through cout_combined, z1, x into mid-bond.
        let mut t_x = tensor3_zeros(left_bond, 4, mid_bond);
        for cc in 0..left_bond {
            for z1 in 0..2usize {
                for x in 0..2usize {
                    let s_x = z1 * 2 + x;
                    let mid = cc * 4 + z1 * 2 + x;
                    t_x.set3(cc, s_x, mid, Complex64::one());
                }
            }
        }

        // Y-site tensor: T_y[mid, s_y=z2*2+y, right=cin_combined]
        // Given cout (from mid), z1, x, y, cin (from right), verify carry equations
        // and compute z2.
        let mut t_y = tensor3_zeros(mid_bond, 4, right_bond);
        for cc_out in 0..left_bond {
            let cout1_idx = cc_out / cout_dim;
            let cout2_idx = cc_out % cout_dim;

            for z1 in 0..2usize {
                for x in 0..2usize {
                    let mid = cc_out * 4 + z1 * 2 + x;

                    for y in 0..2usize {
                        for cin1_idx in 0..cin_dim {
                            let v1 = t1[[cin1_idx, cout1_idx, x, y, z1]];
                            if v1 == Complex64::zero() {
                                continue;
                            }

                            for z2 in 0..2usize {
                                for cin2_idx in 0..cin_dim {
                                    let v2 = t2[[cin2_idx, cout2_idx, x, y, z2]];
                                    if v2 == Complex64::zero() {
                                        continue;
                                    }

                                    let cin_combined = cin1_idx * cin_dim + cin2_idx;
                                    let s_y = z2 * 2 + y;
                                    let weight = v1 * v2;
                                    t_y.set3(mid, s_y, cin_combined, weight);
                                }
                            }
                        }
                    }
                }
            }
        }

        tensors.push(t_x);
        tensors.push(t_y);
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("Failed to create binaryop MPO: {}", e))
}

/// Create a single binaryop tensor for one site.
///
/// This is a direct port of _binaryop_tensor from Quantics.jl.
/// Computes: a*x + b*y + carry_in = 2*carry_out + out
///
/// # Arguments
/// * `a`, `b` - Coefficients (-1, 0, or 1)
/// * `cin_on` - Whether carry input is enabled
/// * `cout_on` - Whether carry output is enabled
/// * `bc` - Boundary condition value (1 for periodic, 0 for open)
///
/// # Returns
/// Tensor of shape (cin_size, cout_size, 2, 2, 2) where:
/// - cin_size = 3 if cin_on else 1
/// - cout_size = 3 if cout_on else 1
/// - Last three dimensions are (x, y, out)
fn binaryop_tensor_single(
    a: i8,
    b: i8,
    cin_on: bool,
    cout_on: bool,
    bc: i8,
) -> DTensor<Complex64, 5> {
    let cin_states: Vec<i8> = if cin_on { vec![-1, 0, 1] } else { vec![0] };
    let cin_size = cin_states.len();
    let cout_size = if cout_on { 3 } else { 1 };

    // tensor[cin, cout, x, y, out] - shape: (cin_size, cout_size, 2, 2, 2)
    let mut tensor =
        DTensor::<Complex64, 5>::from_elem([cin_size, cout_size, 2, 2, 2], Complex64::zero());

    for (idx_cin, &cin) in cin_states.iter().enumerate() {
        for x in 0..2i8 {
            for y in 0..2i8 {
                let res = a * x + b * y + cin;
                let cout = if res >= 0 { res.abs() >> 1 } else { -1i8 };
                let out_bit = (res.abs() & 1) as usize;

                if cout_on {
                    let cout_idx = (cout + 1) as usize;
                    tensor[[idx_cin, cout_idx, x as usize, y as usize, out_bit]] = Complex64::one();
                } else {
                    let weight = if cout == 0 {
                        Complex64::one()
                    } else {
                        Complex64::new(bc as f64, 0.0)
                    };
                    tensor[[idx_cin, 0, x as usize, y as usize, out_bit]] = weight;
                }
            }
        }
    }

    tensor
}

/// Create a binaryop MPO for a simpler case: single output variable.
///
/// This computes f(x, y) = g(a*x + b*y, y) where x and y are interleaved.
/// Delegates to `binaryop_mpo` with coeffs2 = (0, 1) (identity on y).
pub fn binaryop_single_mpo(
    r: usize,
    a: i8,
    b: i8,
    bc: BoundaryCondition,
) -> Result<TensorTrain<Complex64>> {
    let coeffs1 = BinaryCoeffs::new(a, b)?;
    let coeffs2 = BinaryCoeffs::select_y(); // z2 = y (identity)
    binaryop_mpo(r, coeffs1, coeffs2, [bc, BoundaryCondition::Periodic])
}

/// Create a binary operation operator for a single transformation.
///
/// This transforms f(x, y) where the first variable is transformed by a*x + b*y
/// and the second variable y remains unchanged.
///
/// # Arguments
/// * `r` - Number of bits per variable
/// * `a`, `b` - Coefficients (-1, 0, or 1) with (a, b) ≠ (-1, -1)
/// * `bc` - Boundary condition
pub fn binaryop_single_operator(
    r: usize,
    a: i8,
    b: i8,
    bc: BoundaryCondition,
) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of bits must be positive"));
    }

    let mpo = binaryop_single_mpo(r, a, b, bc)?;
    let site_dims = vec![2; 2 * r];
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_simplett::AbstractTensorTrain;

    #[test]
    fn test_binary_coeffs_valid() {
        assert!(BinaryCoeffs::new(1, 1).is_ok());
        assert!(BinaryCoeffs::new(1, -1).is_ok());
        assert!(BinaryCoeffs::new(-1, 1).is_ok());
        assert!(BinaryCoeffs::new(0, 1).is_ok());
        assert!(BinaryCoeffs::new(1, 0).is_ok());
        assert!(BinaryCoeffs::new(0, 0).is_ok());
    }

    #[test]
    fn test_binary_coeffs_invalid() {
        assert!(BinaryCoeffs::new(-1, -1).is_err());
        assert!(BinaryCoeffs::new(2, 0).is_err());
        assert!(BinaryCoeffs::new(0, 2).is_err());
    }

    #[test]
    fn test_binaryop_tensor_single() {
        // Test (1, 1) coefficients (sum)
        let tensor = binaryop_tensor_single(1, 1, false, false, 1);
        assert_eq!(*tensor.shape(), (1, 1, 2, 2, 2)); // (cin_size, cout_size, x, y, out)

        // x=0, y=0: result=0, out=0
        assert_eq!(tensor[[0, 0, 0, 0, 0]], Complex64::one());
        // x=0, y=1: result=1, out=1
        assert_eq!(tensor[[0, 0, 0, 1, 1]], Complex64::one());
        // x=1, y=0: result=1, out=1
        assert_eq!(tensor[[0, 0, 1, 0, 1]], Complex64::one());
        // x=1, y=1: result=2, out=0 (with carry)
        assert_eq!(tensor[[0, 0, 1, 1, 0]], Complex64::one());
    }

    #[test]
    fn test_binaryop_tensor_difference() {
        // Test (1, -1) coefficients (difference)
        let tensor = binaryop_tensor_single(1, -1, false, false, 1);

        // x=0, y=0: result=0, out=0
        assert_eq!(tensor[[0, 0, 0, 0, 0]], Complex64::one());
        // x=0, y=1: result=-1, out=1 (|−1| mod 2 = 1)
        assert_eq!(tensor[[0, 0, 0, 1, 1]], Complex64::one());
        // x=1, y=0: result=1, out=1
        assert_eq!(tensor[[0, 0, 1, 0, 1]], Complex64::one());
        // x=1, y=1: result=0, out=0
        assert_eq!(tensor[[0, 0, 1, 1, 0]], Complex64::one());
    }

    #[test]
    fn test_binaryop_single_mpo_structure() {
        let mpo = binaryop_single_mpo(4, 1, 1, BoundaryCondition::Periodic).unwrap();
        // 4 bits per variable × 2 variables = 8 sites
        assert_eq!(mpo.len(), 8);
    }

    #[test]
    fn test_binaryop_single_operator_creation() {
        let op = binaryop_single_operator(4, 1, 1, BoundaryCondition::Periodic);
        assert!(op.is_ok());
    }

    #[test]
    fn test_binaryop_single_identity() {
        // a=1, b=0 should be similar to identity on x
        let mpo = binaryop_single_mpo(2, 1, 0, BoundaryCondition::Periodic).unwrap();
        assert_eq!(mpo.len(), 4);
    }

    #[test]
    fn test_binaryop_error_cases() {
        assert!(binaryop_single_mpo(0, 1, 1, BoundaryCondition::Periodic).is_err());
        assert!(binaryop_single_mpo(4, 2, 0, BoundaryCondition::Periodic).is_err());
        assert!(binaryop_single_mpo(4, -1, -1, BoundaryCondition::Periodic).is_err());
    }
}
