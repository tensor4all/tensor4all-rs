//! Affine transformation operator: y = A*x + b
//!
//! This implements general affine transformations with rational coefficients.
//! The transformation computes y = A*x + b where A is an M×N rational matrix
//! and b is an M-dimensional rational vector.
//!
//! Based on the algorithm from Quantics.jl/src/affine.jl

use std::collections::HashMap;

use anyhow::Result;
use num_complex::Complex64;
use num_integer::Integer;
use num_rational::Rational64;
use num_traits::One;
use tensor4all_simplett::{types::tensor3_zeros, Tensor3Ops, TensorTrain};

use crate::common::{
    tensortrain_to_linear_operator_asymmetric, BoundaryCondition, QuanticsOperator,
};

/// Affine transformation parameters.
///
/// Represents the transformation y = A*x + b where:
/// - A is an M×N matrix (stored row-major)
/// - b is an M-dimensional vector
/// - x is an N-dimensional input
/// - y is an M-dimensional output
#[derive(Clone, Debug)]
pub struct AffineParams {
    /// Transformation matrix A (M×N), stored row-major
    pub a: Vec<Rational64>,
    /// Translation vector b (M elements)
    pub b: Vec<Rational64>,
    /// Number of output dimensions (M)
    pub m: usize,
    /// Number of input dimensions (N)
    pub n: usize,
}

impl AffineParams {
    /// Create new affine parameters.
    ///
    /// # Arguments
    /// * `a` - M×N matrix in row-major order
    /// * `b` - M-dimensional translation vector
    /// * `m` - Number of output dimensions
    /// * `n` - Number of input dimensions
    pub fn new(a: Vec<Rational64>, b: Vec<Rational64>, m: usize, n: usize) -> Result<Self> {
        if a.len() != m * n {
            return Err(anyhow::anyhow!(
                "Matrix A has {} elements but expected {}×{}={}",
                a.len(),
                m,
                n,
                m * n
            ));
        }
        if b.len() != m {
            return Err(anyhow::anyhow!(
                "Vector b has {} elements but expected {}",
                b.len(),
                m
            ));
        }
        Ok(Self { a, b, m, n })
    }

    /// Create affine parameters from integer matrix and vector.
    pub fn from_integers(a: Vec<i64>, b: Vec<i64>, m: usize, n: usize) -> Result<Self> {
        let a_rat: Vec<Rational64> = a.into_iter().map(Rational64::from_integer).collect();
        let b_rat: Vec<Rational64> = b.into_iter().map(Rational64::from_integer).collect();
        Self::new(a_rat, b_rat, m, n)
    }

    /// Get element A[i, j] (0-indexed)
    #[allow(dead_code)]
    fn get_a(&self, i: usize, j: usize) -> Rational64 {
        self.a[i * self.n + j]
    }

    /// Convert to integer representation by scaling with LCM of denominators.
    /// Returns (A_int, b_int, scale) where A_int = scale * A and b_int = scale * b.
    fn to_integer_scaled(&self) -> (Vec<i64>, Vec<i64>, i64) {
        // Find LCM of all denominators
        let mut denom_lcm = 1i64;
        for r in &self.a {
            denom_lcm = denom_lcm.lcm(r.denom());
        }
        for r in &self.b {
            denom_lcm = denom_lcm.lcm(r.denom());
        }

        // Scale to integers
        let a_int: Vec<i64> = self
            .a
            .iter()
            .map(|r| (r * denom_lcm).to_integer())
            .collect();
        let b_int: Vec<i64> = self
            .b
            .iter()
            .map(|r| (r * denom_lcm).to_integer())
            .collect();

        (a_int, b_int, denom_lcm)
    }
}

/// Create an affine transformation operator.
///
/// This operator transforms a quantics tensor train representing a function
/// f(x_1, ..., x_N) to g(y_1, ..., y_M) where y = A*x + b.
///
/// # Arguments
/// * `r` - Number of bits per variable
/// * `params` - Affine transformation parameters
/// * `bc` - Boundary conditions for each output variable
///
/// # Returns
/// LinearOperator representing the affine transformation
///
/// # Example
/// ```ignore
/// use tensor4all_quantics_transform::{affine_operator, AffineParams, BoundaryCondition};
/// use num_rational::Rational64;
///
/// // Transform g(x, y) -> g(x + y, x - y) (rotation by 45°, scaled)
/// let a = vec![
///     Rational64::from_integer(1), Rational64::from_integer(1),  // row 0: x + y
///     Rational64::from_integer(1), Rational64::from_integer(-1), // row 1: x - y
/// ];
/// let b = vec![Rational64::from_integer(0), Rational64::from_integer(0)];
/// let params = AffineParams::new(a, b, 2, 2).unwrap();
/// let bc = vec![BoundaryCondition::Periodic; 2];
/// let op = affine_operator(8, &params, &bc).unwrap();
/// ```
pub fn affine_operator(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of bits must be positive"));
    }
    if bc.len() != params.m {
        return Err(anyhow::anyhow!(
            "Boundary conditions length {} doesn't match output dimensions {}",
            bc.len(),
            params.m
        ));
    }

    let mpo = affine_transform_mpo(r, params, bc)?;
    // Site dimensions: M output variables, N input variables
    // Input dimension per site: 2^N (N input bits)
    // Output dimension per site: 2^M (M output bits)
    let input_dim = 1 << params.n;
    let output_dim = 1 << params.m;
    let input_dims = vec![input_dim; r];
    let output_dims = vec![output_dim; r];
    tensortrain_to_linear_operator_asymmetric(&mpo, &input_dims, &output_dims)
}

/// Create the affine transformation MPO as a TensorTrain.
fn affine_transform_mpo(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<TensorTrain<Complex64>> {
    let (a_int, b_int, scale) = params.to_integer_scaled();
    let m = params.m;
    let n = params.n;

    // Convert boundary conditions to weights
    let bc_periodic: Vec<bool> = bc
        .iter()
        .map(|b| matches!(b, BoundaryCondition::Periodic))
        .collect();

    // Compute core tensors
    let tensors = affine_transform_tensors(r, &a_int, &b_int, scale, m, n, &bc_periodic)?;

    TensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create affine transform MPO: {}", e))
}

/// Compute the core tensors for the affine transformation.
///
/// This implements the algorithm from Quantics.jl that handles:
/// - Carry propagation for multi-bit arithmetic
/// - Scaling factor s from rational to integer conversion
fn affine_transform_tensors(
    r: usize,
    a_int: &[i64],
    b_int: &[i64],
    scale: i64,
    m: usize,
    n: usize,
    bc_periodic: &[bool],
) -> Result<Vec<tensor4all_simplett::Tensor3<Complex64>>> {
    let site_dim = 1 << (m + n); // 2^(M+N) for fused representation

    // Initial carry is zero vector
    let mut carries: Vec<Vec<i64>> = vec![vec![0i64; m]];

    // Working copy of b for bit extraction
    let mut b_work = b_int.to_vec();

    let mut tensors_data: Vec<Vec<Vec<Vec<bool>>>> = Vec::with_capacity(r);

    // Process from LSB (r-1) to MSB (0)
    for bit_pos in (0..r).rev() {
        // Extract current bit from b
        let b_curr: Vec<i64> = b_work
            .iter()
            .map(|&b| {
                let sign = if b >= 0 { 1 } else { -1 };
                sign * (b.abs() & 1)
            })
            .collect();

        // Compute core tensor for this bit position
        let (new_carries, data) =
            affine_transform_core(a_int, &b_curr, scale, m, n, &carries, bit_pos == r - 1)?;

        tensors_data.push(data);
        carries = new_carries;

        // Shift b right
        for b in &mut b_work {
            *b >>= 1;
        }
    }

    // Reverse to get MSB first order
    tensors_data.reverse();

    // Apply boundary conditions at the first tensor (MSB)
    // Filter carries based on boundary condition weights
    if !tensors_data.is_empty() {
        let first_data = &mut tensors_data[0];
        let mut weighted_first: Vec<Vec<Vec<bool>>> = vec![vec![vec![false; site_dim]; 1]; 1];

        for (carry_idx, carry) in carries.iter().enumerate() {
            // Check if this carry satisfies boundary conditions
            let weight = if bc_periodic.iter().all(|&p| p) {
                // All periodic: any carry is fine
                true
            } else {
                // Open boundaries: carry must be zero
                carry.iter().all(|&c| c == 0)
            };

            if weight && carry_idx < first_data.len() {
                for (site_idx, &val) in first_data[carry_idx][0].iter().enumerate() {
                    if val {
                        weighted_first[0][0][site_idx] = true;
                    }
                }
            }
        }

        tensors_data[0] = weighted_first;
    }

    // Convert to Tensor3 format
    let mut tensors = Vec::with_capacity(r);

    for (pos, data) in tensors_data.iter().enumerate() {
        let left_dim = if pos == 0 { 1 } else { data.len() };
        let right_dim = if pos == r - 1 {
            1
        } else {
            tensors_data.get(pos + 1).map_or(1, |d| d.len())
        };

        let mut t: tensor4all_simplett::Tensor3<Complex64> =
            tensor3_zeros(left_dim, site_dim, right_dim);

        for (l, left_data) in data.iter().enumerate() {
            for (ri, right_data) in left_data.iter().enumerate() {
                for (s, &val) in right_data.iter().enumerate() {
                    if val {
                        let l_idx = if pos == 0 { 0 } else { l };
                        let r_idx = if pos == r - 1 { 0 } else { ri };
                        if l_idx < left_dim && r_idx < right_dim {
                            t.set3(l_idx, s, r_idx, Complex64::one());
                        }
                    }
                }
            }
        }

        tensors.push(t);
    }

    Ok(tensors)
}

/// Compute a single core tensor for the affine transformation.
///
/// The core tensor encodes: 2 * carry_out = A * x + b_curr - scale * y + carry_in
///
/// Returns (new_carries, data) where:
/// - new_carries: list of possible outgoing carry vectors
/// - data[carry_out_idx][carry_in_idx][site_idx]: bool tensor
fn affine_transform_core(
    a_int: &[i64],
    b_curr: &[i64],
    scale: i64,
    m: usize,
    n: usize,
    carries_in: &[Vec<i64>],
    _is_lsb: bool,
) -> Result<(Vec<Vec<i64>>, Vec<Vec<Vec<bool>>>)> {
    let mut carry_out_map: HashMap<Vec<i64>, Vec<Vec<bool>>> = HashMap::new();
    let site_dim = 1 << (m + n);

    // Iterate over all input carries
    for (c_idx, carry_in) in carries_in.iter().enumerate() {
        // Iterate over all possible x values (N bits)
        for x_bits in 0..(1 << n) {
            let x: Vec<i64> = (0..n).map(|j| ((x_bits >> j) & 1) as i64).collect();

            // Compute z = A*x + b + carry_in
            let mut z: Vec<i64> = vec![0; m];
            for i in 0..m {
                z[i] = carry_in[i] + b_curr[i];
                for j in 0..n {
                    z[i] += a_int[i * n + j] * x[j];
                }
            }

            if scale % 2 == 1 {
                // Scale is odd: unique y that satisfies condition
                let y: Vec<i64> = z.iter().map(|&zi| zi & 1).collect();
                let y_bits: usize = y.iter().enumerate().map(|(i, &yi)| (yi as usize) << i).sum();

                // Compute carry_out = (z - scale * y) / 2
                let carry_out: Vec<i64> = z
                    .iter()
                    .zip(y.iter())
                    .map(|(&zi, &yi)| (zi - scale * yi) >> 1)
                    .collect();

                // Site index: y bits in lower positions, x bits in upper positions
                let site_idx = y_bits | (x_bits << m);

                let entry = carry_out_map.entry(carry_out).or_insert_with(|| {
                    vec![vec![false; site_dim]; carries_in.len()]
                });
                entry[c_idx][site_idx] = true;
            } else {
                // Scale is even: z must be even for valid y
                if z.iter().any(|&zi| zi % 2 != 0) {
                    continue;
                }

                // y can be any value
                for y_bits in 0..(1 << m) {
                    let y: Vec<i64> = (0..m).map(|i| ((y_bits >> i) & 1) as i64).collect();

                    // Compute carry_out = (z - scale * y) / 2
                    let carry_out: Vec<i64> = z
                        .iter()
                        .zip(y.iter())
                        .map(|(&zi, &yi)| (zi - scale * yi) >> 1)
                        .collect();

                    let site_idx = y_bits | (x_bits << m);

                    let entry = carry_out_map.entry(carry_out).or_insert_with(|| {
                        vec![vec![false; site_dim]; carries_in.len()]
                    });
                    entry[c_idx][site_idx] = true;
                }
            }
        }
    }

    // Convert to vectors
    let mut carries_out: Vec<Vec<i64>> = carry_out_map.keys().cloned().collect();
    carries_out.sort(); // For deterministic ordering

    let mut data: Vec<Vec<Vec<bool>>> = Vec::with_capacity(carries_out.len());
    for carry in &carries_out {
        data.push(carry_out_map[carry].clone());
    }

    Ok((carries_out, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_params_new() {
        let a = vec![
            Rational64::from_integer(1),
            Rational64::from_integer(0),
            Rational64::from_integer(0),
            Rational64::from_integer(1),
        ];
        let b = vec![Rational64::from_integer(0), Rational64::from_integer(0)];
        let params = AffineParams::new(a, b, 2, 2);
        assert!(params.is_ok());
    }

    #[test]
    fn test_affine_params_from_integers() {
        let a = vec![1, 0, 0, 1];
        let b = vec![0, 0];
        let params = AffineParams::from_integers(a, b, 2, 2);
        assert!(params.is_ok());
    }

    #[test]
    fn test_affine_params_to_integer_scaled() {
        // Test with rational coefficients
        let a = vec![
            Rational64::new(1, 2), // 1/2
            Rational64::new(1, 3), // 1/3
        ];
        let b = vec![Rational64::new(1, 6)]; // 1/6
        let params = AffineParams::new(a, b, 1, 2).unwrap();

        let (a_int, b_int, scale) = params.to_integer_scaled();

        // LCM of denominators (2, 3, 6) = 6
        assert_eq!(scale, 6);
        assert_eq!(a_int, vec![3, 2]); // [1/2 * 6, 1/3 * 6]
        assert_eq!(b_int, vec![1]); // [1/6 * 6]
    }

    #[test]
    fn test_affine_transform_identity() {
        // Identity transformation: y = x
        let a = vec![1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_operator_creation() {
        // Simple 2D transformation
        let a = vec![1i64, 1, 1, -1]; // [[1, 1], [1, -1]]
        let b = vec![0i64, 0];
        let params = AffineParams::from_integers(a, b, 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_error_zero_bits() {
        let a = vec![1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(0, &params, &bc);
        assert!(result.is_err());
    }

    #[test]
    fn test_affine_error_bc_mismatch() {
        let a = vec![1i64, 0, 0, 1];
        let b = vec![0i64, 0];
        let params = AffineParams::from_integers(a, b, 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic]; // Only 1 BC but M=2

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_err());
    }

    #[test]
    fn test_affine_params_dimension_error() {
        // a.len() != m * n
        let a = vec![Rational64::from_integer(1), Rational64::from_integer(0)]; // 2 elements
        let b = vec![Rational64::from_integer(0)];
        let params = AffineParams::new(a, b, 2, 2); // expects 4 elements
        assert!(params.is_err());

        // b.len() != m
        let a = vec![
            Rational64::from_integer(1),
            Rational64::from_integer(0),
            Rational64::from_integer(0),
            Rational64::from_integer(1),
        ];
        let b = vec![Rational64::from_integer(0)]; // 1 element but m=2
        let params = AffineParams::new(a, b, 2, 2);
        assert!(params.is_err());
    }

    #[test]
    fn test_affine_with_rational_coefficients() {
        // y = (1/2)*x
        let a = vec![Rational64::new(1, 2)];
        let b = vec![Rational64::from_integer(0)];
        let params = AffineParams::new(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_shift_only() {
        // y = x + 3
        let a = vec![1i64];
        let b = vec![3i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_scale_by_two() {
        // y = 2*x
        let a = vec![2i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_asymmetric_dimensions() {
        // M=1, N=2: y = x1 + x2 (sum of two inputs to one output)
        let a = vec![1i64, 1]; // 1×2 matrix
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_open_boundary() {
        // Identity with open boundary
        let a = vec![1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Open];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_negation() {
        // y = -x
        let a = vec![-1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_mpo_structure() {
        // Verify MPO tensor structure for identity transform
        let a = vec![1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let op = affine_operator(4, &params, &bc).unwrap();
        // Check that the operator was created successfully
        // (More detailed structure tests would require accessing internal TreeTN)
        let _ = op;
    }

    #[test]
    fn test_affine_larger_bits() {
        // Test with more bits
        let a = vec![1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(8, &params, &bc);
        assert!(result.is_ok());

        let result = affine_operator(16, &params, &bc);
        assert!(result.is_ok());
    }
}
