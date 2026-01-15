//! Cumulative sum operator
//!
//! This transformation computes cumulative sums: y_i = Σ_{j < i} x_j

use anyhow::Result;
use num_complex::Complex64;
use num_traits::{One, Zero};
use tensor4all_simplett::{types::tensor3_zeros, Tensor3Ops, TensorTrain};

use crate::common::{tensortrain_to_linear_operator, QuanticsOperator};

/// Create a cumulative sum operator: y_i = Σ_{j < i} x_j
///
/// This MPO implements a strict upper triangular matrix filled with ones.
/// For a function g defined on {0, 1, ..., 2^R - 1}, it computes:
/// f(i) = Σ_{j < i} g(j)
///
/// # Arguments
/// * `r` - Number of bits (sites)
///
/// # Returns
/// LinearOperator representing the cumulative sum
///
/// # Example
/// ```ignore
/// use tensor4all_quantics_transform::cumsum_operator;
///
/// let op = cumsum_operator(8).unwrap();
/// ```
pub fn cumsum_operator(r: usize) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }

    let mpo = cumsum_mpo(r)?;
    let site_dims = vec![2; r];
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create the cumulative sum MPO as a TensorTrain.
///
/// The cumulative sum is implemented as an upper triangular matrix.
/// The MPO tracks whether a comparison has been made:
/// - State 0: No comparison yet (y and x equal so far)
/// - State 1: Comparison made (y > x, so this entry is 1)
///
/// Tensor entries t[left, right, y, x]:
/// - t[0, 0, y, x] = 1 if y == x (both 0 or both 1)
/// - t[0, 1, 1, 0] = 1 (y > x at this position)
/// - t[1, 1, *, *] = 1 (comparison already made)
#[allow(clippy::needless_range_loop)]
fn cumsum_mpo(r: usize) -> Result<TensorTrain<Complex64>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }

    let single_tensor = upper_triangle_tensor();
    let mut tensors = Vec::with_capacity(r);

    for n in 0..r {
        if n == 0 {
            // First tensor: start in state 0 (no comparison yet)
            // Contract with [1, 0] on left link to select state 0
            let mut t = tensor3_zeros(1, 4, 2);
            for cout in 0..2 {
                for y_bit in 0..2 {
                    for x_bit in 0..2 {
                        let val = single_tensor[0][cout][y_bit][x_bit];
                        let s = y_bit * 2 + x_bit;
                        t.set3(0, s, cout, val);
                    }
                }
            }
            tensors.push(t);
        } else if n == r - 1 {
            // Last tensor: select entries where state is 1 (y > x)
            // The output is 1 only if comparison was made (state 1)
            // For upper triangle (strict), diagonal is excluded
            let mut t = tensor3_zeros(2, 4, 1);
            for cin in 0..2 {
                for y_bit in 0..2 {
                    for x_bit in 0..2 {
                        // Sum over cout states, weighted by whether comparison was made
                        let mut sum = Complex64::zero();
                        for cout in 0..2 {
                            let val = single_tensor[cin][cout][y_bit][x_bit];
                            // Only count if we end in state 1 (comparison made)
                            if cout == 1 {
                                sum += val;
                            }
                        }
                        let s = y_bit * 2 + x_bit;
                        t.set3(cin, s, 0, sum);
                    }
                }
            }
            tensors.push(t);
        } else {
            // Middle tensors: full tensor
            let mut t = tensor3_zeros(2, 4, 2);
            for cin in 0..2 {
                for cout in 0..2 {
                    for y_bit in 0..2 {
                        for x_bit in 0..2 {
                            let val = single_tensor[cin][cout][y_bit][x_bit];
                            let s = y_bit * 2 + x_bit;
                            t.set3(cin, s, cout, val);
                        }
                    }
                }
            }
            tensors.push(t);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("Failed to create cumsum MPO: {}", e))
}

/// Create the single-site tensor for upper triangular matrix.
///
/// Returns a 4D tensor [cin][cout][y_bit][x_bit] where:
/// - cin: input state (0 = no comparison yet, 1 = comparison made)
/// - cout: output state
/// - y_bit: output (row) bit
/// - x_bit: input (column) bit
///
/// The tensor implements strict upper triangle comparison:
/// - State 0: Comparing bits. If y > x, transition to state 1.
/// - State 1: Comparison made. All remaining entries are 1.
fn upper_triangle_tensor() -> [[[[Complex64; 2]; 2]; 2]; 2] {
    let mut tensor = [[[[Complex64::zero(); 2]; 2]; 2]; 2];

    // State 0 -> State 0: y == x (continue comparing)
    tensor[0][0][0][0] = Complex64::one(); // y=0, x=0
    tensor[0][0][1][1] = Complex64::one(); // y=1, x=1

    // State 0 -> State 1: y > x (comparison made, y is greater)
    tensor[0][1][1][0] = Complex64::one(); // y=1, x=0 (y > x at this bit)

    // State 0 -> nowhere: y < x (this entry is 0, not in upper triangle)
    // tensor[0][*][0][1] = 0 (implicit)

    // State 1 -> State 1: Comparison already made, all entries are 1
    tensor[1][1][0][0] = Complex64::one();
    tensor[1][1][0][1] = Complex64::one();
    tensor[1][1][1][0] = Complex64::one();
    tensor[1][1][1][1] = Complex64::one();

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_simplett::AbstractTensorTrain;

    #[test]
    fn test_upper_triangle_tensor() {
        let t = upper_triangle_tensor();

        // State 0 -> State 0: diagonal
        assert_eq!(t[0][0][0][0], Complex64::one());
        assert_eq!(t[0][0][1][1], Complex64::one());

        // State 0 -> State 1: y > x
        assert_eq!(t[0][1][1][0], Complex64::one());

        // State 0 -> nowhere: y < x
        assert_eq!(t[0][0][0][1], Complex64::zero());
        assert_eq!(t[0][1][0][1], Complex64::zero());

        // State 1 -> State 1: all ones
        assert_eq!(t[1][1][0][0], Complex64::one());
        assert_eq!(t[1][1][0][1], Complex64::one());
        assert_eq!(t[1][1][1][0], Complex64::one());
        assert_eq!(t[1][1][1][1], Complex64::one());
    }

    #[test]
    fn test_cumsum_mpo_structure() {
        let mpo = cumsum_mpo(4).unwrap();
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
    fn test_cumsum_operator_creation() {
        let op = cumsum_operator(4);
        assert!(op.is_ok());
    }

    #[test]
    fn test_cumsum_error_zero_sites() {
        let result = cumsum_operator(0);
        assert!(result.is_err());
    }
}
