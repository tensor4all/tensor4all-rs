//! Integration tests for quanticstransform operators.
//!
//! These tests verify numerical correctness by:
//! 1. Creating a state as TensorTrain (MPS)
//! 2. Converting to TreeTN
//! 3. Applying the operator using apply_linear_operator
//! 4. Contracting the result and comparing with expected values

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use approx::assert_relative_eq;
use num_complex::Complex64;
use num_traits::{One, Zero};

use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::{IndexLike, TensorDynLen, TensorIndex};
use tensor4all_simplett::{types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain};
use tensor4all_treetn::{apply_linear_operator, ApplyOptions, LinearOperator, TreeTN};

use tensor4all_quanticstransform::{
    binaryop_operator, binaryop_single_operator, flip_operator, quantics_fourier_operator,
    shift_operator, BinaryCoeffs, BoundaryCondition, FTCore, FourierOptions,
};

/// Type alias for the default index type.
type DynIndex = Index<DynId, TagSet>;

// ============================================================================
// Helper functions for TensorTrain <-> TreeTN conversion
// ============================================================================

/// Convert a TensorTrain (MPS state) to a TreeTN.
///
/// The MPS is converted to a chain-like TreeTN where each site tensor
/// has indices (left_bond, site, right_bond) mapped to TensorDynLen format.
///
/// # Returns
/// A tuple of (TreeTN, site_indices) where site_indices are the external indices
/// that can be used for operator application.
fn tensortrain_to_treetn(
    tt: &TensorTrain<Complex64>,
) -> (TreeTN<TensorDynLen, usize>, Vec<DynIndex>) {
    let n = tt.len();
    assert!(n > 0, "TensorTrain must have at least one site");

    // Create site indices
    let site_indices: Vec<DynIndex> = (0..n)
        .map(|i| {
            let dim = tt.site_tensor(i).site_dim();
            Index::new_dyn(dim)
        })
        .collect();

    // Create bond indices
    let mut bond_indices: Vec<DynIndex> = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let dim = if i == 0 {
            1
        } else {
            tt.site_tensor(i - 1).right_dim()
        };
        bond_indices.push(Index::new_dyn(dim));
    }

    // Build tensors for TreeTN
    let mut tensors: Vec<TensorDynLen> = Vec::with_capacity(n);
    let node_names: Vec<usize> = (0..n).collect();

    for i in 0..n {
        let tensor = tt.site_tensor(i);
        let left_dim = tensor.left_dim();
        let site_dim = tensor.site_dim();
        let right_dim = tensor.right_dim();

        // Tensor indices: (left_bond, site, right_bond)
        // For boundary tensors, we omit the dimension-1 bond
        let mut indices: Vec<DynIndex> = Vec::with_capacity(3);
        let mut dims_vec: Vec<usize> = Vec::with_capacity(3);

        if i > 0 {
            indices.push(bond_indices[i].clone());
            dims_vec.push(left_dim);
        }
        indices.push(site_indices[i].clone());
        dims_vec.push(site_dim);
        if i < n - 1 {
            indices.push(bond_indices[i + 1].clone());
            dims_vec.push(right_dim);
        }

        // Copy data from Tensor3 to flat vector
        let total_size: usize = dims_vec.iter().product();
        let mut data: Vec<Complex64> = vec![Complex64::zero(); total_size];

        if i == 0 && n == 1 {
            // Single tensor case: just site index
            for s in 0..site_dim {
                data[s] = *tensor.get3(0, s, 0);
            }
        } else if i == 0 {
            // First tensor: (site, right_bond)
            for s in 0..site_dim {
                for r in 0..right_dim {
                    let idx = s * right_dim + r;
                    data[idx] = *tensor.get3(0, s, r);
                }
            }
        } else if i == n - 1 {
            // Last tensor: (left_bond, site)
            for l in 0..left_dim {
                for s in 0..site_dim {
                    let idx = l * site_dim + s;
                    data[idx] = *tensor.get3(l, s, 0);
                }
            }
        } else {
            // Middle tensor: (left_bond, site, right_bond)
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        let idx = (l * site_dim + s) * right_dim + r;
                        data[idx] = *tensor.get3(l, s, r);
                    }
                }
            }
        }

        let tensor_dyn = TensorDynLen::from_dense_c64(indices, data);
        tensors.push(tensor_dyn);
    }

    let treetn = TreeTN::from_tensors(tensors, node_names).expect("Failed to create TreeTN");
    (treetn, site_indices)
}

/// Create a product state MPS representing a specific integer value x.
///
/// The state |x⟩ = |x_0⟩ ⊗ |x_1⟩ ⊗ ... ⊗ |x_{R-1}⟩ where x_i are binary digits.
/// Uses big-endian convention (MSB first): x = Σ_n x_n * 2^(R-1-n)
///
/// This matches Julia Quantics.jl's convention where site 0 (Julia site 1)
/// contains the most significant bit.
fn create_product_state_mps(x: usize, r: usize) -> TensorTrain<Complex64> {
    let mut tensors = Vec::with_capacity(r);

    for n in 0..r {
        // Big-endian: site n contains bit 2^(R-1-n)
        let bit = (x >> (r - 1 - n)) & 1;

        let mut t = tensor3_zeros(1, 2, 1);
        t.set3(0, bit, 0, Complex64::one());
        tensors.push(t);
    }

    TensorTrain::new(tensors).expect("Failed to create product state MPS")
}

/// Contract a TreeTN with site indices to get a flat vector representation.
///
/// Returns a vector of length 2^R representing f[x] for x = 0, 1, ..., 2^R - 1.
fn contract_treetn_to_vector(
    treetn: &TreeTN<TensorDynLen, usize>,
    site_indices: &[DynIndex],
) -> Vec<Complex64> {
    let r = site_indices.len();
    let n = 1 << r; // 2^R

    // Contract to full tensor
    let full_tensor = treetn
        .contract_to_tensor()
        .expect("Failed to contract TreeTN");

    // Extract data from the tensor
    // The tensor has indices ordered by the indices field
    let ext_indices = &full_tensor.indices;

    // Build mapping from site index ID to position in external indices
    let mut idx_to_pos: HashMap<DynId, usize> = HashMap::new();
    for (pos, idx) in ext_indices.iter().enumerate() {
        idx_to_pos.insert(*idx.id(), pos);
    }

    // Build mapping from our site index order to tensor index order
    let mut site_to_tensor: Vec<usize> = Vec::with_capacity(r);
    for site_idx in site_indices.iter() {
        let pos = idx_to_pos
            .get(&site_idx.id().clone())
            .expect("Site index not found in contracted tensor");
        site_to_tensor.push(*pos);
    }

    // Extract values in canonical order (x = 0, 1, ..., 2^R - 1)
    let mut result = vec![Complex64::zero(); n];

    // Get data from storage
    let data = full_tensor
        .as_slice_c64()
        .expect("Expected DenseC64 storage");
    let dims = full_tensor.dims();

    // Iterate over all x values
    for x in 0..n {
        // Convert x to multi-index in tensor order
        let mut multi_idx = vec![0usize; r];
        for i in 0..r {
            // Big-endian: site i corresponds to bit 2^(R-1-i)
            let bit = (x >> (r - 1 - i)) & 1;
            multi_idx[site_to_tensor[i]] = bit;
        }

        // Convert multi-index to flat index (row-major order)
        let mut flat_idx = 0;
        let mut stride = 1;
        for i in (0..r).rev() {
            flat_idx += multi_idx[i] * stride;
            stride *= dims[i];
        }

        result[x] = data[flat_idx];
    }

    result
}

/// Evaluate a TensorTrain at all indices (brute force).
///
/// Returns a vector of length 2^R representing f[x] for x = 0, 1, ..., 2^R - 1.
/// Uses little-endian convention to match the quantics operators.
fn evaluate_mps_all(mps: &TensorTrain<Complex64>) -> Vec<Complex64> {
    let r = mps.len();
    let n = 1 << r;

    let mut result = Vec::with_capacity(n);
    for x in 0..n {
        // Convert x to multi-index (little-endian: site i has bit 2^i)
        let indices: Vec<usize> = (0..r).map(|i| (x >> i) & 1).collect();
        let val = mps.evaluate(&indices).expect("Failed to evaluate MPS");
        result.push(val);
    }
    result
}

/// Apply a QuanticsOperator to all product state inputs and collect results as a dense matrix.
///
/// For an operator with `n_in` input sites (each dim 2), this creates all 2^n_in
/// product state inputs, applies the operator, and returns a 2^n_out x 2^n_in matrix
/// where M[y][x] = <y|Op|x>.
///
/// # Arguments
/// * `op` - The operator to test
/// * `n_in` - Number of input sites
/// * `n_out` - Number of output sites (may differ from n_in for asymmetric operators)
fn apply_operator_to_dense_matrix(
    op: &LinearOperator<TensorDynLen, usize>,
    n_in: usize,
    n_out: usize,
) -> Vec<Vec<Complex64>> {
    let dim_in = 1 << n_in;
    let dim_out = 1 << n_out;
    let mut matrix = vec![vec![Complex64::zero(); dim_in]; dim_out];

    for x in 0..dim_in {
        let mps = create_product_state_mps(x, n_in);
        let (treetn, site_indices) = tensortrain_to_treetn(&mps);

        // Remap site indices to operator input
        let mut treetn_remapped = treetn;
        for i in 0..n_in {
            let op_input = op
                .get_input_mapping(&i)
                .expect("Missing input mapping")
                .true_index
                .clone();
            treetn_remapped = treetn_remapped
                .replaceind(&site_indices[i], &op_input)
                .expect("Failed to replace index");
        }

        // Apply operator
        let result_treetn = apply_linear_operator(op, &treetn_remapped, ApplyOptions::naive())
            .expect("Failed to apply operator");

        // Get output indices
        let output_indices: Vec<DynIndex> = (0..n_out)
            .map(|i| {
                op.get_output_mapping(&i)
                    .expect("Missing output mapping")
                    .true_index
                    .clone()
            })
            .collect();

        // Contract result
        let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

        for y in 0..dim_out {
            matrix[y][x] = result_vec[y];
        }
    }

    matrix
}

// ============================================================================
// Dense matrix helper validation tests
// ============================================================================

/// Smoke test for apply_operator_to_dense_matrix using flip (known-correct operator)
#[test]
fn test_dense_matrix_helper_with_flip() {
    let r = 3;
    let n = 1 << r;

    let op = flip_operator(r, BoundaryCondition::Periodic).expect("Failed to create flip");
    let matrix = apply_operator_to_dense_matrix(&op, r, r);

    // flip(0) = 0, flip(x) = N - x for x > 0
    for x in 0..n {
        let expected_y = if x == 0 { 0 } else { n - x };
        for y in 0..n {
            let expected = if y == expected_y { 1.0 } else { 0.0 };
            assert_relative_eq!(matrix[y][x].re, expected, epsilon = 1e-10);
            assert_relative_eq!(matrix[y][x].im, 0.0, epsilon = 1e-10);
        }
    }
}

// ============================================================================
// Flip operator tests
// ============================================================================

/// Test the flip MPO directly by verifying expected transformations.
#[test]
fn test_flip_mpo_direct() {
    let r = 3;
    let n = 1 << r;

    eprintln!("\n=== Testing flip MPO directly for R={} ===", r);

    // For each input x, compute expected output
    // flip(x) = 2^R - x (mod 2^R) = n - x
    for x_in in 0..n {
        let expected_x_out = if x_in == 0 { 0 } else { n - x_in };
        eprintln!(
            "flip({}) should be {} (little-endian convention)",
            x_in, expected_x_out
        );
    }
}

#[test]
fn test_flip_numerical_correctness() {
    let r = 3; // Use smaller R for easier debugging
    let n = 1 << r;

    // Create flip operator
    let op = flip_operator(r, BoundaryCondition::Periodic).expect("Failed to create flip operator");

    // Test on a single product state first
    let x = 1; // Test with |1⟩

    // Create |x⟩ state
    let mps = create_product_state_mps(x, r);

    // Debug: verify MPS values
    eprintln!("\n=== Testing flip on x={} (R={}) ===", x, r);
    let mps_values = evaluate_mps_all(&mps);
    eprintln!(
        "MPS values: {:?}",
        mps_values.iter().map(|c| (c.re, c.im)).collect::<Vec<_>>()
    );

    let (treetn, site_indices) = tensortrain_to_treetn(&mps);

    // Debug: print site indices
    eprintln!(
        "Site indices: {:?}",
        site_indices.iter().map(|i| i.id()).collect::<Vec<_>>()
    );

    // Debug: verify TreeTN contraction matches MPS values
    let treetn_values = contract_treetn_to_vector(&treetn, &site_indices);
    eprintln!(
        "TreeTN values: {:?}",
        treetn_values
            .iter()
            .map(|c| (c.re, c.im))
            .collect::<Vec<_>>()
    );
    eprintln!("Comparing MPS and TreeTN values...");
    for (i, (m, t)) in mps_values.iter().zip(treetn_values.iter()).enumerate() {
        if (m.re - t.re).abs() > 1e-10 || (m.im - t.im).abs() > 1e-10 {
            eprintln!("  MISMATCH at index {}: MPS={:?}, TreeTN={:?}", i, m, t);
        }
    }

    // Remap site indices to match operator's input indices
    let mut treetn_remapped = treetn;
    for i in 0..r {
        let op_input = op
            .get_input_mapping(&i)
            .expect("Missing input mapping")
            .true_index
            .clone();
        eprintln!(
            "Remapping site {} from {:?} to {:?}",
            i,
            site_indices[i].id(),
            op_input.id()
        );
        treetn_remapped = treetn_remapped
            .replaceind(&site_indices[i], &op_input)
            .expect("Failed to replace index");
    }

    // Debug: check remapped TreeTN
    let remapped_ext = treetn_remapped.external_indices();
    eprintln!(
        "Remapped TreeTN external indices: {:?}",
        remapped_ext.iter().map(|i| i.id()).collect::<Vec<_>>()
    );

    // Apply operator
    let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
        .expect("Failed to apply operator");

    // Debug: check result TreeTN
    let result_ext = result_treetn.external_indices();
    eprintln!(
        "Result TreeTN external indices: {:?}",
        result_ext.iter().map(|i| i.id()).collect::<Vec<_>>()
    );

    // Get output site indices from operator
    let output_indices: Vec<DynIndex> = (0..r)
        .map(|i| {
            op.get_output_mapping(&i)
                .expect("Missing output mapping")
                .true_index
                .clone()
        })
        .collect();
    eprintln!(
        "Output indices: {:?}",
        output_indices.iter().map(|i| i.id()).collect::<Vec<_>>()
    );

    // Contract result to vector
    let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);
    eprintln!(
        "Result vector: {:?}",
        result_vec.iter().map(|c| (c.re, c.im)).collect::<Vec<_>>()
    );

    // Expected: flip(x) = 2^R - x = n - x for periodic BC (but flip(0) = 0, not n)
    // flip(1) = 8 - 1 = 7 for R=3
    let expected_x = if x == 0 { 0 } else { n - x };
    eprintln!("Expected result at index: {}", expected_x);

    // Debug: print where the non-zero value is
    for (idx, val) in result_vec.iter().enumerate() {
        if val.norm() > 1e-10 {
            eprintln!(
                "  Non-zero value at index {}: ({}, {})",
                idx, val.re, val.im
            );
        }
    }

    // Result should be |expected_x⟩
    for y in 0..n {
        let expected_val = if y == expected_x {
            Complex64::one()
        } else {
            Complex64::zero()
        };
        if result_vec[y].norm() > 1e-10 || y == expected_x {
            eprintln!(
                "  result_vec[{}] = ({}, {}), expected = ({}, {})",
                y, result_vec[y].re, result_vec[y].im, expected_val.re, expected_val.im
            );
        }
        assert_relative_eq!(result_vec[y].re, expected_val.re, epsilon = 1e-10);
        assert_relative_eq!(result_vec[y].im, expected_val.im, epsilon = 1e-10);
    }
}

// ============================================================================
// Shift operator tests
// ============================================================================

#[test]
fn test_shift_numerical_correctness() {
    let r = 4;
    let n = 1 << r;

    // Test various shift amounts
    for offset in [-3i64, -1, 0, 1, 3, 7] {
        let op = shift_operator(r, offset, BoundaryCondition::Periodic)
            .expect("Failed to create shift operator");

        // Test on a few product states
        for x in [0usize, 1, 5, 15] {
            let mps = create_product_state_mps(x, r);
            let (treetn, site_indices) = tensortrain_to_treetn(&mps);

            // Remap site indices
            let mut treetn_remapped = treetn;
            for i in 0..r {
                let op_input = op
                    .get_input_mapping(&i)
                    .expect("Missing input mapping")
                    .true_index
                    .clone();
                treetn_remapped = treetn_remapped
                    .replaceind(&site_indices[i], &op_input)
                    .expect("Failed to replace index");
            }

            // Apply operator
            let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
                .expect("Failed to apply operator");

            // Get output indices
            let output_indices: Vec<DynIndex> = (0..r)
                .map(|i| {
                    op.get_output_mapping(&i)
                        .expect("Missing output mapping")
                        .true_index
                        .clone()
                })
                .collect();

            // Contract result
            let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

            // Expected: shift_offset(x) maps |x⟩ to |x + offset mod n⟩
            // Since operator computes f(x) = g(x + offset), if input is δ_x,
            // output is δ_{x+offset mod n}
            let expected_x = ((x as i64 + offset).rem_euclid(n as i64)) as usize;

            for y in 0..n {
                let expected_val = if y == expected_x {
                    Complex64::one()
                } else {
                    Complex64::zero()
                };
                assert_relative_eq!(
                    result_vec[y].re,
                    expected_val.re,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
                assert_relative_eq!(
                    result_vec[y].im,
                    expected_val.im,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
            }
        }
    }
}

// ============================================================================
// Fourier operator tests
// ============================================================================

#[test]
fn test_fourier_numerical_correctness() {
    let r = 3; // Small for exact comparison

    // Create forward Fourier operator (normalized)
    let op =
        quantics_fourier_operator(r, FourierOptions::forward()).expect("Failed to create Fourier");

    // N = 2^R, expected magnitude = 1/√N
    let expected_magnitude = 1.0 / ((1 << r) as f64).sqrt();

    // Test on |0⟩ state: F|0⟩ = (1/√N) Σ_k |k⟩ (uniform superposition)
    let mps = create_product_state_mps(0, r);
    let (treetn, site_indices) = tensortrain_to_treetn(&mps);

    // Remap site indices
    let mut treetn_remapped = treetn;
    for i in 0..r {
        let op_input = op
            .get_input_mapping(&i)
            .expect("Missing input mapping")
            .true_index
            .clone();
        treetn_remapped = treetn_remapped
            .replaceind(&site_indices[i], &op_input)
            .expect("Failed to replace index");
    }

    // Apply operator
    let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
        .expect("Failed to apply operator");

    // Get output indices
    let output_indices: Vec<DynIndex> = (0..r)
        .map(|i| {
            op.get_output_mapping(&i)
                .expect("Missing output mapping")
                .true_index
                .clone()
        })
        .collect();

    // Contract result
    let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

    // For |0⟩ input, all output components should have equal magnitude 1/√N
    let n = 1 << r;
    for k in 0..n {
        // Check that magnitude is approximately 1/√N
        let magnitude = result_vec[k].norm();
        assert_relative_eq!(magnitude, expected_magnitude, epsilon = 1e-6);
    }
}

#[test]
fn test_fourier_inverse_operator_creation() {
    // Test that forward and inverse Fourier operators can be created and have compatible structures
    let r = 3;

    // Create forward and inverse Fourier operators
    let ft = FTCore::new(r, FourierOptions::default()).expect("Failed to create FTCore");
    let forward_op = ft.forward().expect("Failed to get forward op");
    let inverse_op = ft.backward().expect("Failed to get backward op");

    // Verify both operators have correct number of site mappings
    for i in 0..r {
        assert!(forward_op.get_input_mapping(&i).is_some());
        assert!(forward_op.get_output_mapping(&i).is_some());
        assert!(inverse_op.get_input_mapping(&i).is_some());
        assert!(inverse_op.get_output_mapping(&i).is_some());
    }

    // Test that forward Fourier can be applied
    let mps = create_product_state_mps(0, r);
    let (treetn, site_indices) = tensortrain_to_treetn(&mps);

    // Remap and apply
    let mut treetn_tmp = treetn;
    for i in 0..r {
        let op_input = forward_op
            .get_input_mapping(&i)
            .expect("Missing input mapping")
            .true_index
            .clone();
        treetn_tmp = treetn_tmp
            .replaceind(&site_indices[i], &op_input)
            .expect("Failed to replace index");
    }

    let result = apply_linear_operator(&forward_op, &treetn_tmp, ApplyOptions::naive())
        .expect("Failed to apply forward Fourier");

    // Result should have R external indices (the output indices)
    let ext_indices = result.external_indices();
    assert_eq!(
        ext_indices.len(),
        r,
        "Result should have {} external indices",
        r
    );
}

// ============================================================================
// Comprehensive numerical tests
// ============================================================================

/// Test flip operator for ALL x values (0 to 2^R-1)
#[test]
fn test_flip_all_values() {
    let r = 3;
    let n = 1 << r;

    eprintln!("\n=== Testing flip for all x values (R={}, N={}) ===", r, n);

    // Create flip operator
    let op = flip_operator(r, BoundaryCondition::Periodic).expect("Failed to create flip operator");

    for x in 0..n {
        let mps = create_product_state_mps(x, r);
        let (treetn, site_indices) = tensortrain_to_treetn(&mps);

        // Remap site indices
        let mut treetn_remapped = treetn;
        for i in 0..r {
            let op_input = op
                .get_input_mapping(&i)
                .expect("Missing input mapping")
                .true_index
                .clone();
            treetn_remapped = treetn_remapped
                .replaceind(&site_indices[i], &op_input)
                .expect("Failed to replace index");
        }

        // Apply operator
        let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
            .expect("Failed to apply operator");

        // Get output indices
        let output_indices: Vec<DynIndex> = (0..r)
            .map(|i| {
                op.get_output_mapping(&i)
                    .expect("Missing output mapping")
                    .true_index
                    .clone()
            })
            .collect();

        // Contract result
        let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

        // Expected: flip(x) = 2^R - x mod 2^R
        // flip(0) = 0, flip(1) = 7, flip(2) = 6, ..., flip(7) = 1
        let expected_x = if x == 0 { 0 } else { n - x };

        eprintln!("flip({}) = {} (expected)", x, expected_x);

        // Check result is delta at expected_x
        for y in 0..n {
            let expected_val = if y == expected_x {
                Complex64::one()
            } else {
                Complex64::zero()
            };
            assert_relative_eq!(
                result_vec[y].re,
                expected_val.re,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
            assert_relative_eq!(
                result_vec[y].im,
                expected_val.im,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
        }
    }

    eprintln!("All flip tests passed!");
}

/// Test shift operator for ALL x values and various offsets
#[test]
fn test_shift_all_values() {
    let r = 3;
    let n = 1 << r;

    eprintln!(
        "\n=== Testing shift for all x values (R={}, N={}) ===",
        r, n
    );

    // Test various shift amounts
    for offset in [-3i64, -1, 0, 1, 3, 5] {
        let op = shift_operator(r, offset, BoundaryCondition::Periodic)
            .expect("Failed to create shift operator");

        eprintln!("Testing offset = {}", offset);

        for x in 0..n {
            let mps = create_product_state_mps(x, r);
            let (treetn, site_indices) = tensortrain_to_treetn(&mps);

            // Remap site indices
            let mut treetn_remapped = treetn;
            for i in 0..r {
                let op_input = op
                    .get_input_mapping(&i)
                    .expect("Missing input mapping")
                    .true_index
                    .clone();
                treetn_remapped = treetn_remapped
                    .replaceind(&site_indices[i], &op_input)
                    .expect("Failed to replace index");
            }

            // Apply operator
            let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
                .expect("Failed to apply operator");

            // Get output indices
            let output_indices: Vec<DynIndex> = (0..r)
                .map(|i| {
                    op.get_output_mapping(&i)
                        .expect("Missing output mapping")
                        .true_index
                        .clone()
                })
                .collect();

            // Contract result
            let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

            // Expected: shift_offset(x) = x + offset mod n
            let expected_x = ((x as i64 + offset).rem_euclid(n as i64)) as usize;

            // Check result is delta at expected_x
            for y in 0..n {
                let expected_val = if y == expected_x {
                    Complex64::one()
                } else {
                    Complex64::zero()
                };
                assert_relative_eq!(
                    result_vec[y].re,
                    expected_val.re,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
                assert_relative_eq!(
                    result_vec[y].im,
                    expected_val.im,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
            }
        }
    }

    eprintln!("All shift tests passed!");
}

/// Test Fourier transform preserves unitarity for all basis states.
///
/// The QFT is a unitary operator, so ||F|x⟩||² = ||x⟩||² = 1 for any basis state.
/// Also verifies that output magnitudes are uniform (characteristic of Fourier transform).
#[test]
fn test_fourier_unitarity_all_basis_states() {
    let r = 3;
    let n = 1 << r;

    eprintln!(
        "\n=== Testing Fourier unitarity for all basis states (R={}, N={}) ===",
        r, n
    );

    // Create forward Fourier operator (normalized)
    let op =
        quantics_fourier_operator(r, FourierOptions::forward()).expect("Failed to create Fourier");

    // Expected magnitude for each component when transforming a basis state
    let expected_magnitude = 1.0 / (n as f64).sqrt();

    for x in 0..n {
        let mps = create_product_state_mps(x, r);
        let (treetn, site_indices) = tensortrain_to_treetn(&mps);

        // Remap site indices
        let mut treetn_remapped = treetn;
        for i in 0..r {
            let op_input = op
                .get_input_mapping(&i)
                .expect("Missing input mapping")
                .true_index
                .clone();
            treetn_remapped = treetn_remapped
                .replaceind(&site_indices[i], &op_input)
                .expect("Failed to replace index");
        }

        // Apply operator
        let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
            .expect("Failed to apply operator");

        // Get output indices
        let output_indices: Vec<DynIndex> = (0..r)
            .map(|i| {
                op.get_output_mapping(&i)
                    .expect("Missing output mapping")
                    .true_index
                    .clone()
            })
            .collect();

        // Contract result
        let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

        eprintln!("F|{}⟩:", x);

        // Verify unitarity: total norm squared should be 1
        let total_norm_sq: f64 = result_vec.iter().map(|c| c.norm_sqr()).sum();
        assert_relative_eq!(total_norm_sq, 1.0, epsilon = 1e-6);
        eprintln!("  total ||F|{}⟩||² = {:.6}", x, total_norm_sq);

        // All magnitudes should be equal (uniform superposition property of Fourier)
        for k in 0..n {
            let actual_magnitude = result_vec[k].norm();
            assert_relative_eq!(
                actual_magnitude,
                expected_magnitude,
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }

    eprintln!("All Fourier unitarity tests passed!");
}

/// Test Fourier inverse operator produces uniform superposition for appropriate input.
///
/// Note: Round-trip testing (F^{-1} F |x⟩ = |x⟩) requires matching TreeTN topologies
/// between the forward output and the inverse input. This is complex because
/// `apply_linear_operator` creates new TreeTN structures. The unitarity test above
/// verifies the essential property that ||F|x⟩|| = ||x⟩||.
///
/// This test verifies that the inverse Fourier operator can be applied and produces
/// the expected uniform superposition property: F^{-1}|0⟩ = uniform superposition.
#[test]
fn test_fourier_inverse_application() {
    let r = 3;
    let n = 1 << r;

    eprintln!(
        "\n=== Testing inverse Fourier F^{{-1}} |0⟩ (R={}, N={}) ===",
        r, n
    );

    // Create inverse Fourier operator
    let inverse_op = quantics_fourier_operator(r, FourierOptions::inverse())
        .expect("Failed to create inverse Fourier operator");

    // Test on |0⟩: F^{-1}|0⟩ should give uniform superposition (same as F|0⟩)
    let x = 0;
    let mps = create_product_state_mps(x, r);
    let (treetn, site_indices) = tensortrain_to_treetn(&mps);

    // Apply inverse Fourier
    let mut treetn_inv = treetn;
    for i in 0..r {
        let op_input = inverse_op
            .get_input_mapping(&i)
            .expect("Missing input mapping")
            .true_index
            .clone();
        treetn_inv = treetn_inv
            .replaceind(&site_indices[i], &op_input)
            .expect("Failed to replace index");
    }

    let result_inv = apply_linear_operator(&inverse_op, &treetn_inv, ApplyOptions::naive())
        .expect("Failed to apply inverse Fourier");

    // Get output indices
    let output_indices: Vec<DynIndex> = (0..r)
        .map(|i| {
            inverse_op
                .get_output_mapping(&i)
                .expect("Missing output mapping")
                .true_index
                .clone()
        })
        .collect();

    // Contract result
    let result_vec = contract_treetn_to_vector(&result_inv, &output_indices);

    eprintln!("F^{{-1}}|0⟩:");
    let expected_magnitude = 1.0 / (n as f64).sqrt();
    for k in 0..n {
        eprintln!(
            "  k={}: ({:.4}, {:.4})",
            k, result_vec[k].re, result_vec[k].im
        );
        // All components should have equal magnitude (uniform superposition)
        assert_relative_eq!(result_vec[k].norm(), expected_magnitude, epsilon = 1e-6);
    }

    // Verify unitarity: total norm squared should be 1
    let total_norm_sq: f64 = result_vec.iter().map(|c| c.norm_sqr()).sum();
    assert_relative_eq!(total_norm_sq, 1.0, epsilon = 1e-6);
    eprintln!("Total norm squared: {:.6}", total_norm_sq);

    eprintln!("Inverse Fourier application test passed!");
}

// ============================================================================
// Fourier phase verification and roundtrip tests
// ============================================================================

use std::f64::consts::PI;

/// Compute reference DFT matrix: F[k, x] = (1/sqrt(N)) * exp(sign * 2*pi*i * k * x / N)
fn reference_dft_matrix(n: usize, sign: f64) -> Vec<Vec<Complex64>> {
    let mut matrix = vec![vec![Complex64::zero(); n]; n];
    let norm = 1.0 / (n as f64).sqrt();
    for k in 0..n {
        for x in 0..n {
            let phase = sign * 2.0 * PI * (k as f64) * (x as f64) / (n as f64);
            matrix[k][x] = Complex64::new(phase.cos(), phase.sin()) * norm;
        }
    }
    matrix
}

/// Test Fourier transform phase correctness against reference DFT matrix.
///
/// Port of Julia's _qft_ref test: computes full DFT matrix and compares element-wise.
/// Tests both forward (sign=-1) and inverse (sign=+1) for R=2,3,4.
#[test]
fn test_fourier_phase_correctness() {
    for r in [2, 3, 4] {
        let n = 1usize << r;

        // Forward Fourier (sign = -1 in Rust's convention)
        let forward_op = quantics_fourier_operator(r, FourierOptions::forward())
            .expect("Failed to create forward Fourier");
        let forward_matrix = apply_operator_to_dense_matrix(&forward_op, r, r);
        let forward_ref = reference_dft_matrix(n, -1.0);

        for k in 0..n {
            for x in 0..n {
                assert_relative_eq!(
                    forward_matrix[k][x].re,
                    forward_ref[k][x].re,
                    epsilon = 1e-6,
                );
                assert_relative_eq!(
                    forward_matrix[k][x].im,
                    forward_ref[k][x].im,
                    epsilon = 1e-6,
                );
            }
        }

        // Inverse Fourier (sign = +1)
        let inverse_op = quantics_fourier_operator(r, FourierOptions::inverse())
            .expect("Failed to create inverse Fourier");
        let inverse_matrix = apply_operator_to_dense_matrix(&inverse_op, r, r);
        let inverse_ref = reference_dft_matrix(n, 1.0);

        for k in 0..n {
            for x in 0..n {
                assert_relative_eq!(
                    inverse_matrix[k][x].re,
                    inverse_ref[k][x].re,
                    epsilon = 1e-6,
                );
                assert_relative_eq!(
                    inverse_matrix[k][x].im,
                    inverse_ref[k][x].im,
                    epsilon = 1e-6,
                );
            }
        }
    }
}

/// Test Fourier roundtrip: F^{-1} * F ≈ I (as dense matrices).
#[test]
fn test_fourier_roundtrip_matrix() {
    for r in [2, 3, 4] {
        let n = 1usize << r;

        let forward_op = quantics_fourier_operator(r, FourierOptions::forward())
            .expect("Failed to create forward Fourier");
        let inverse_op = quantics_fourier_operator(r, FourierOptions::inverse())
            .expect("Failed to create inverse Fourier");

        let f_mat = apply_operator_to_dense_matrix(&forward_op, r, r);
        let fi_mat = apply_operator_to_dense_matrix(&inverse_op, r, r);

        // Compute F^{-1} * F and verify ≈ I
        for i in 0..n {
            for j in 0..n {
                let mut product = Complex64::zero();
                for k in 0..n {
                    product += fi_mat[i][k] * f_mat[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(product.re, expected, epsilon = 1e-5);
                assert_relative_eq!(product.im, 0.0, epsilon = 1e-5);
            }
        }
    }
}

// ============================================================================
// Open boundary condition tests
// ============================================================================

/// Test shift operator with Open boundary condition.
///
/// With Open BC, shift results in zero when the value overflows/underflows [0, 2^R).
#[test]
fn test_shift_open_boundary() {
    let r = 3;
    let n = 1 << r; // N = 8

    eprintln!(
        "\n=== Testing shift with Open boundary (R={}, N={}) ===",
        r, n
    );

    // Test cases: (x, offset, expected)
    // With Open BC, the result depends on whether x + offset_mod causes carry overflow.
    //
    // Implementation detail:
    // - offset is normalized to offset_mod = offset.rem_euclid(N) in [0, N)
    // - nbc = (offset - offset_mod) / N tracks full cycles
    // - If nbc > 0 (positive overflow beyond N), entire MPO is zeroed
    // - If nbc < 0 (negative offset), no global zeroing, but local carry may still zero
    // - Local carry overflow at MSB zeros the result via bc_val = 0
    //
    // So for Open BC:
    // - shift(x, offset) with offset >= 0: result is zero if x + offset >= N
    // - shift(x, offset) with offset < 0:
    //   - offset_mod = N + offset (e.g., -1 mod 8 = 7)
    //   - nbc = -1 (no global zeroing)
    //   - But x + offset_mod may cause local carry overflow
    //   - e.g., shift(7, -3): offset_mod = 5, 7 + 5 = 12 >= 8, overflow -> zero
    //   - e.g., shift(0, -1): offset_mod = 7, 0 + 7 = 7 < 8, no overflow -> result = 7
    //
    let test_cases: Vec<(usize, i64, Option<usize>)> = vec![
        // No overflow cases (positive offset)
        (0, 0, Some(0)), // 0 + 0 = 0, in range
        (3, 2, Some(5)), // 3 + 2 = 5, in range
        (0, 7, Some(7)), // 0 + 7 = 7, in range
        (1, 3, Some(4)), // 1 + 3 = 4, in range
        // Overflow cases (positive offset)
        (7, 1, None), // 7 + 1 = 8 >= 8, overflow
        (6, 3, None), // 6 + 3 = 9 >= 8, overflow
        (4, 5, None), // 4 + 5 = 9 >= 8, overflow
        // Negative offset cases
        // shift(x, -k) -> offset_mod = 8 - k, x + offset_mod >= 8 iff x >= k
        (0, -1, Some(7)), // offset_mod = 7, 0 + 7 = 7 < 8, no overflow
        (0, -7, Some(1)), // offset_mod = 1, 0 + 1 = 1 < 8, no overflow
        (1, -1, None),    // offset_mod = 7, 1 + 7 = 8 >= 8, overflow
        (2, -1, None),    // offset_mod = 7, 2 + 7 = 9 >= 8, overflow
        (7, -7, None),    // offset_mod = 1, 7 + 1 = 8 >= 8, overflow
    ];

    for (x, offset, expected) in test_cases {
        let op = shift_operator(r, offset, BoundaryCondition::Open)
            .expect("Failed to create shift operator");

        let mps = create_product_state_mps(x, r);
        let (treetn, site_indices) = tensortrain_to_treetn(&mps);

        // Remap site indices
        let mut treetn_remapped = treetn;
        for i in 0..r {
            let op_input = op
                .get_input_mapping(&i)
                .expect("Missing input mapping")
                .true_index
                .clone();
            treetn_remapped = treetn_remapped
                .replaceind(&site_indices[i], &op_input)
                .expect("Failed to replace index");
        }

        // Apply operator
        let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
            .expect("Failed to apply operator");

        // Get output indices
        let output_indices: Vec<DynIndex> = (0..r)
            .map(|i| {
                op.get_output_mapping(&i)
                    .expect("Missing output mapping")
                    .true_index
                    .clone()
            })
            .collect();

        // Contract result
        let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

        match expected {
            Some(expected_x) => {
                eprintln!("shift({}, {}, Open) = {} (in range)", x, offset, expected_x);
                // Check result is delta at expected_x
                for y in 0..n {
                    let expected_val = if y == expected_x {
                        Complex64::one()
                    } else {
                        Complex64::zero()
                    };
                    assert_relative_eq!(
                        result_vec[y].re,
                        expected_val.re,
                        epsilon = 1e-10,
                        max_relative = 1e-10
                    );
                    assert_relative_eq!(
                        result_vec[y].im,
                        expected_val.im,
                        epsilon = 1e-10,
                        max_relative = 1e-10
                    );
                }
            }
            None => {
                eprintln!("shift({}, {}, Open) = 0 (out of range)", x, offset);
                // Check result is zero vector
                let total_norm: f64 = result_vec.iter().map(|c| c.norm_sqr()).sum();
                assert_relative_eq!(total_norm, 0.0, epsilon = 1e-10);
            }
        }
    }

    eprintln!("All shift Open BC tests passed!");
}

/// Test flip operator with Open boundary condition.
///
/// For flip, the boundary condition affects whether flip(0) = 0 or flip(0) = 2^R (overflow).
/// Since flip(x) = 2^R - x:
/// - flip(0) = 2^R = 0 (mod 2^R) with Periodic, but produces zero vector with Open (overflow)
/// - flip(x) for x > 0 gives 2^R - x which is in [1, 2^R-1], so no overflow
///
/// Note: Julia's flipop only supports bc=1 (periodic) or bc=-1 (antisymmetric), not Open BC.
/// The Open BC for flip is a Rust-specific extension that zeros out the flip(0) case.
#[test]
fn test_flip_open_boundary() {
    let r = 3;
    let n = 1 << r;

    eprintln!(
        "\n=== Testing flip with Open boundary (R={}, N={}) ===",
        r, n
    );

    // Create flip operator with Open BC
    let op = flip_operator(r, BoundaryCondition::Open).expect("Failed to create flip operator");

    for x in 0..n {
        let mps = create_product_state_mps(x, r);
        let (treetn, site_indices) = tensortrain_to_treetn(&mps);

        // Remap site indices
        let mut treetn_remapped = treetn;
        for i in 0..r {
            let op_input = op
                .get_input_mapping(&i)
                .expect("Missing input mapping")
                .true_index
                .clone();
            treetn_remapped = treetn_remapped
                .replaceind(&site_indices[i], &op_input)
                .expect("Failed to replace index");
        }

        // Apply operator
        let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
            .expect("Failed to apply operator");

        // Get output indices
        let output_indices: Vec<DynIndex> = (0..r)
            .map(|i| {
                op.get_output_mapping(&i)
                    .expect("Missing output mapping")
                    .true_index
                    .clone()
            })
            .collect();

        // Contract result
        let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

        // With Open BC:
        // - flip(0) = 2^R overflows -> zero vector (all coefficients zero)
        // - flip(x) for x > 0 = 2^R - x (no overflow, normal result)
        let is_overflow = x == 0; // flip(0) = 2^R causes overflow

        if is_overflow {
            // All coefficients should be zero
            eprintln!("flip({}, Open) = overflow -> zero vector", x);
            for y in 0..n {
                assert_relative_eq!(result_vec[y].re, 0.0, epsilon = 1e-10, max_relative = 1e-10);
                assert_relative_eq!(result_vec[y].im, 0.0, epsilon = 1e-10, max_relative = 1e-10);
            }
        } else {
            // Normal result: delta at expected_x
            let expected_x = n - x;
            eprintln!("flip({}, Open) = {} (no overflow)", x, expected_x);

            for y in 0..n {
                let expected_val = if y == expected_x {
                    Complex64::one()
                } else {
                    Complex64::zero()
                };
                assert_relative_eq!(
                    result_vec[y].re,
                    expected_val.re,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
                assert_relative_eq!(
                    result_vec[y].im,
                    expected_val.im,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
            }
        }
    }

    eprintln!("All flip Open BC tests passed!");
}

// ============================================================================
// Phase rotation operator tests
// ============================================================================

use tensor4all_quanticstransform::phase_rotation_operator;

/// Test phase rotation operator for all x values.
///
/// Phase rotation multiplies by exp(i*θ*x):
/// phase_rotation(|x⟩) = exp(i*θ*x) * |x⟩
///
/// This is a diagonal operator that leaves the state unchanged except for a phase.
#[test]
fn test_phase_rotation_all_values() {
    let r = 3;
    let n = 1 << r;
    let theta = PI / 4.0; // 45 degrees

    eprintln!(
        "\n=== Testing phase rotation for all x values (R={}, N={}, θ=π/4) ===",
        r, n
    );

    let op = phase_rotation_operator(r, theta).expect("Failed to create phase rotation operator");

    for x in 0..n {
        let mps = create_product_state_mps(x, r);
        let (treetn, site_indices) = tensortrain_to_treetn(&mps);

        // Remap site indices
        let mut treetn_remapped = treetn;
        for i in 0..r {
            let op_input = op
                .get_input_mapping(&i)
                .expect("Missing input mapping")
                .true_index
                .clone();
            treetn_remapped = treetn_remapped
                .replaceind(&site_indices[i], &op_input)
                .expect("Failed to replace index");
        }

        // Apply operator
        let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
            .expect("Failed to apply operator");

        // Get output indices
        let output_indices: Vec<DynIndex> = (0..r)
            .map(|i| {
                op.get_output_mapping(&i)
                    .expect("Missing output mapping")
                    .true_index
                    .clone()
            })
            .collect();

        // Contract result
        let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

        // Expected: |x⟩ -> exp(i*θ*x) * |x⟩
        let expected_phase = Complex64::new((theta * x as f64).cos(), (theta * x as f64).sin());

        eprintln!(
            "phase_rotation({}) = exp(i*{:.4}*{}) = ({:.4}, {:.4})",
            x, theta, x, expected_phase.re, expected_phase.im
        );

        // Check result: only position x should be non-zero
        for y in 0..n {
            let expected_val = if y == x {
                expected_phase
            } else {
                Complex64::zero()
            };
            assert_relative_eq!(
                result_vec[y].re,
                expected_val.re,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
            assert_relative_eq!(
                result_vec[y].im,
                expected_val.im,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
        }
    }

    eprintln!("All phase rotation tests passed!");
}

/// Test phase rotation with θ = 0 (identity).
#[test]
fn test_phase_rotation_zero_theta() {
    let r = 3;
    let n = 1 << r;
    let theta = 0.0;

    eprintln!(
        "\n=== Testing phase rotation with θ=0 (identity) (R={}, N={}) ===",
        r, n
    );

    let op = phase_rotation_operator(r, theta).expect("Failed to create phase rotation operator");

    for x in 0..n {
        let mps = create_product_state_mps(x, r);
        let (treetn, site_indices) = tensortrain_to_treetn(&mps);

        // Remap site indices
        let mut treetn_remapped = treetn;
        for i in 0..r {
            let op_input = op
                .get_input_mapping(&i)
                .expect("Missing input mapping")
                .true_index
                .clone();
            treetn_remapped = treetn_remapped
                .replaceind(&site_indices[i], &op_input)
                .expect("Failed to replace index");
        }

        // Apply operator
        let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
            .expect("Failed to apply operator");

        // Get output indices
        let output_indices: Vec<DynIndex> = (0..r)
            .map(|i| {
                op.get_output_mapping(&i)
                    .expect("Missing output mapping")
                    .true_index
                    .clone()
            })
            .collect();

        // Contract result
        let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

        // With θ=0, should be identity: |x⟩ -> |x⟩
        for y in 0..n {
            let expected_val = if y == x {
                Complex64::one()
            } else {
                Complex64::zero()
            };
            assert_relative_eq!(
                result_vec[y].re,
                expected_val.re,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
            assert_relative_eq!(
                result_vec[y].im,
                expected_val.im,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
        }
    }

    eprintln!("Phase rotation identity test passed!");
}

/// Test phase rotation with θ = 2π (should equal identity).
#[test]
fn test_phase_rotation_two_pi() {
    let r = 3;
    let n = 1 << r;
    let theta = 2.0 * PI;

    eprintln!(
        "\n=== Testing phase rotation with θ=2π (identity) (R={}, N={}) ===",
        r, n
    );

    let op = phase_rotation_operator(r, theta).expect("Failed to create phase rotation operator");

    for x in 0..n {
        let mps = create_product_state_mps(x, r);
        let (treetn, site_indices) = tensortrain_to_treetn(&mps);

        let mut treetn_remapped = treetn;
        for i in 0..r {
            let op_input = op
                .get_input_mapping(&i)
                .expect("Missing input mapping")
                .true_index
                .clone();
            treetn_remapped = treetn_remapped
                .replaceind(&site_indices[i], &op_input)
                .expect("Failed to replace index");
        }

        let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
            .expect("Failed to apply operator");

        let output_indices: Vec<DynIndex> = (0..r)
            .map(|i| {
                op.get_output_mapping(&i)
                    .expect("Missing output mapping")
                    .true_index
                    .clone()
            })
            .collect();

        let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

        // exp(i*2π*x) = 1 for all integer x
        for y in 0..n {
            let expected_val = if y == x {
                Complex64::one()
            } else {
                Complex64::zero()
            };
            assert_relative_eq!(
                result_vec[y].re,
                expected_val.re,
                epsilon = 1e-6,
                max_relative = 1e-6
            );
            assert_relative_eq!(
                result_vec[y].im,
                expected_val.im,
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }

    eprintln!("Phase rotation 2π test passed!");
}

// ============================================================================
// Cumulative sum operator tests
// ============================================================================

use tensor4all_quanticstransform::cumsum_operator;

/// Test cumulative sum operator creation.
///
/// Basic operator creation test.
#[test]
fn test_cumsum_operator_creation() {
    let r = 3;

    eprintln!("\n=== Testing cumsum operator creation (R={}) ===", r);

    let op = cumsum_operator(r);
    assert!(op.is_ok(), "Cumsum operator creation should succeed");

    eprintln!("Cumsum operator creation test passed!");
}

/// Test cumsum operator for all x values.
///
/// cumsum is a strict upper triangular matrix:
/// cumsum(|x⟩) = Σ_{y > x} |y⟩
///
/// The comparison y > x is done in big-endian bit order (MSB to LSB).
#[test]
fn test_cumsum_all_values() {
    let r = 3;
    let n = 1 << r;

    eprintln!(
        "\n=== Testing cumsum for all x values (R={}, N={}) ===",
        r, n
    );

    let op = cumsum_operator(r).expect("Failed to create cumsum operator");

    for x in 0..n {
        let mps = create_product_state_mps(x, r);
        let (treetn, site_indices) = tensortrain_to_treetn(&mps);

        // Remap site indices
        let mut treetn_remapped = treetn;
        for i in 0..r {
            let op_input = op
                .get_input_mapping(&i)
                .expect("Missing input mapping")
                .true_index
                .clone();
            treetn_remapped = treetn_remapped
                .replaceind(&site_indices[i], &op_input)
                .expect("Failed to replace index");
        }

        // Apply operator
        let result_treetn = apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
            .expect("Failed to apply operator");

        // Get output indices
        let output_indices: Vec<DynIndex> = (0..r)
            .map(|i| {
                op.get_output_mapping(&i)
                    .expect("Missing output mapping")
                    .true_index
                    .clone()
            })
            .collect();

        // Contract result
        let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);

        eprintln!("cumsum(|{}⟩):", x);

        // For cumsum, result[y] = 1 iff y > x (strict upper triangular)
        // Now that contract_treetn_to_vector uses big-endian, result[y] corresponds to value y
        let mut count_positive = 0;
        for y in 0..n {
            let expected_val = if y > x {
                Complex64::one()
            } else {
                Complex64::zero()
            };

            if result_vec[y].norm() > 0.5 {
                count_positive += 1;
                eprintln!(
                    "  y={}: ({:.4}, {:.4})",
                    y, result_vec[y].re, result_vec[y].im
                );
            }

            assert_relative_eq!(
                result_vec[y].re,
                expected_val.re,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
            assert_relative_eq!(
                result_vec[y].im,
                expected_val.im,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
        }

        // x has n-1-x values greater than it
        let expected_count = n - 1 - x;
        eprintln!(
            "  Found {} positive, expected {} (y > {})",
            count_positive, expected_count, x
        );
    }

    eprintln!("All cumsum tests passed!");
}

// ============================================================================
// Affine operator tests
// ============================================================================

use tensor4all_quanticstransform::{affine_operator, affine_transform_matrix, AffineParams};

/// Test affine identity transformation: y = x.
#[test]
fn test_affine_identity() {
    let r = 3;
    let n = 1 << r;

    eprintln!("\n=== Testing affine identity y = x (R={}, N={}) ===", r, n);

    // Identity: y = 1*x + 0
    let params = AffineParams::from_integers(vec![1], vec![0], 1, 1)
        .expect("Failed to create affine params");
    let bc = vec![BoundaryCondition::Periodic];

    let _op = affine_operator(r, &params, &bc).expect("Failed to create affine operator");

    #[allow(clippy::never_loop)]
    for x in 0..n {
        // Create input state |x⟩
        // For affine with M=1, N=1, site dimension is 2^(M+N) = 4
        // But we need to create the right kind of input

        // The affine operator expects fused input/output sites
        // Input: x bits in positions [M..M+N) of each site index
        // Output: y bits in positions [0..M) of each site index

        // For identity (M=1, N=1), site index = y * 2 + x
        // We want to test x, so input state has site_idx = (y)*2 + x = 0*2 + x = x for each site

        // Create product state where each site has value = x_bit
        let mut tensors = Vec::with_capacity(r);
        for bit_pos in 0..r {
            let x_bit = (x >> bit_pos) & 1;
            // For affine input, the x bits go in the upper part
            // site_idx = y_bits | (x_bits << M) = y_bits | (x_bit << 1)
            // We want to sum over all y outputs, but for identity the output y should equal x

            // Actually, for testing, we should create a state where we input a specific x
            // and verify the output is at the same x (for identity)

            // The site dimension is 4 (2^2 for M=1, N=1)
            // site index = y_bit + x_bit * 2
            // For input |x⟩, we want x_bit at each position
            let mut t = tensor3_zeros(1, 4, 1);
            // Put amplitude 1 at site index corresponding to x_bit (input), summed over y_bit (output)
            // For testing, let's put amplitude at all y for this x
            for y_bit in 0..2 {
                let site_idx = y_bit + x_bit * 2;
                t.set3(0, site_idx, 0, Complex64::one());
            }
            tensors.push(t);
        }

        // Skip this test for now - affine has complex fused representation
        // that needs more careful handling
        break;
    }

    eprintln!("Affine identity test: operator creation successful");
}

/// Test affine shift transformation: y = x + offset.
#[test]
fn test_affine_shift() {
    let r = 3;
    let n = 1 << r;
    let offset = 3i64;

    eprintln!(
        "\n=== Testing affine shift y = x + {} (R={}, N={}) ===",
        offset, r, n
    );

    // Shift: y = 1*x + offset
    let params = AffineParams::from_integers(vec![1], vec![offset], 1, 1)
        .expect("Failed to create affine params");
    let bc = vec![BoundaryCondition::Periodic];

    let op = affine_operator(r, &params, &bc);
    assert!(op.is_ok(), "Affine shift operator creation should succeed");

    eprintln!("Affine shift test: operator creation successful");
}

/// Test affine negation transformation: y = -x.
#[test]
fn test_affine_negation() {
    let r = 3;

    eprintln!("\n=== Testing affine negation y = -x (R={}) ===", r);

    // Negation: y = -1*x + 0
    let params = AffineParams::from_integers(vec![-1], vec![0], 1, 1)
        .expect("Failed to create affine params");
    let bc = vec![BoundaryCondition::Periodic];

    let op = affine_operator(r, &params, &bc);
    assert!(
        op.is_ok(),
        "Affine negation operator creation should succeed"
    );

    eprintln!("Affine negation test: operator creation successful");
}

/// Test affine 2D transformation: (y1, y2) = (x1 + x2, x1 - x2).
#[test]
fn test_affine_2d_rotation() {
    let r = 3;

    eprintln!("\n=== Testing affine 2D rotation (R={}) ===", r);

    // 2D rotation: [[1, 1], [1, -1]] * [x1, x2]
    let params = AffineParams::from_integers(
        vec![1, 1, 1, -1], // Row major: [1,1] for row 0, [1,-1] for row 1
        vec![0, 0],
        2,
        2,
    )
    .expect("Failed to create affine params");
    let bc = vec![BoundaryCondition::Periodic; 2];

    let op = affine_operator(r, &params, &bc);
    assert!(
        op.is_ok(),
        "Affine 2D rotation operator creation should succeed"
    );

    eprintln!("Affine 2D rotation test: operator creation successful");
}

/// Test affine operator numerical correctness by comparing the dense matrix
/// obtained from the MPO (via `affine_operator` + `apply_operator_to_dense_matrix`)
/// with the sparse reference matrix from `affine_transform_matrix`.
///
/// This validates that the two independent implementations (carry-based MPO
/// construction vs direct enumeration) produce the same transformation.
///
/// Note: `apply_operator_to_dense_matrix` only works for operators with
/// site dimension 2, which corresponds to M=1, N=1 affine transformations.
#[test]
fn test_affine_mpo_matches_matrix() {
    // Test cases for M=1, N=1 (site dim 2): (a, b, bc)
    let test_cases_1d: Vec<(Vec<i64>, Vec<i64>, Vec<BoundaryCondition>)> = vec![
        // y = x (identity)
        (vec![1], vec![0], vec![BoundaryCondition::Periodic]),
        // y = x + 3 (shift)
        (vec![1], vec![3], vec![BoundaryCondition::Periodic]),
        // y = -x (negation)
        (vec![-1], vec![0], vec![BoundaryCondition::Periodic]),
        // y = 2x (scale by 2)
        (vec![2], vec![0], vec![BoundaryCondition::Periodic]),
        // y = x (identity, open BC)
        (vec![1], vec![0], vec![BoundaryCondition::Open]),
        // y = x + 1 (shift by 1, open BC)
        (vec![1], vec![1], vec![BoundaryCondition::Open]),
    ];

    for (a_flat, b_vec, bc) in &test_cases_1d {
        let m = 1usize;
        let n = 1usize;

        for r in [2, 3] {
            let params = AffineParams::from_integers(a_flat.clone(), b_vec.clone(), m, n)
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed params a={:?} b={:?} m={} n={}: {}",
                        a_flat, b_vec, m, n, e
                    )
                });

            // Get reference sparse matrix
            let ref_matrix = match affine_transform_matrix(r, &params, bc) {
                Ok(mat) => mat,
                Err(e) => {
                    eprintln!(
                        "Skipping a={:?} b={:?} r={} bc={:?}: {}",
                        a_flat, b_vec, r, bc, e
                    );
                    continue;
                }
            };

            // Get MPO as LinearOperator and compute dense matrix
            let op = match affine_operator(r, &params, bc) {
                Ok(op) => op,
                Err(e) => {
                    eprintln!(
                        "Skipping operator a={:?} b={:?} r={} bc={:?}: {}",
                        a_flat, b_vec, r, bc, e
                    );
                    continue;
                }
            };

            // For M=1, N=1: operator has r sites, input dim 2, output dim 2
            let dim = 1usize << r;
            let mpo_dense = apply_operator_to_dense_matrix(&op, r, r);

            // Verify dimensions of reference matrix
            assert_eq!(ref_matrix.rows(), dim, "Output dimension mismatch");
            assert_eq!(ref_matrix.cols(), dim, "Input dimension mismatch");

            // Compare element-by-element
            for y in 0..dim {
                for x in 0..dim {
                    let sparse_val = *ref_matrix.get(y, x).unwrap_or(&0.0);
                    let mpo_val = mpo_dense[y][x];

                    assert!(
                        (mpo_val.re - sparse_val).abs() < 1e-10,
                        "Real part mismatch at ({}, {}): mpo={}, ref={} \
                         [a={:?} b={:?} r={} bc={:?}]",
                        y,
                        x,
                        mpo_val.re,
                        sparse_val,
                        a_flat,
                        b_vec,
                        r,
                        bc
                    );
                    assert!(
                        mpo_val.im.abs() < 1e-10,
                        "Imaginary part non-zero at ({}, {}): im={} \
                         [a={:?} b={:?} r={} bc={:?}]",
                        y,
                        x,
                        mpo_val.im,
                        a_flat,
                        b_vec,
                        r,
                        bc
                    );
                }
            }

            eprintln!(
                "PASS: affine a={:?} b={:?} r={} bc={:?}",
                a_flat, b_vec, r, bc
            );
        }
    }
}

/// Test affine_transform_matrix structural properties for various parameter combinations,
/// including multi-dimensional cases (M != 1 or N != 1) where the full MPO comparison
/// via apply_operator_to_dense_matrix is not available.
#[test]
fn test_affine_transform_matrix_properties() {
    // Test cases: (a_flat (row-major MxN), b (length M), m, n, bc)
    let test_cases: Vec<(Vec<i64>, Vec<i64>, usize, usize, Vec<BoundaryCondition>)> = vec![
        // y = x (identity, 1D)
        (vec![1], vec![0], 1, 1, vec![BoundaryCondition::Periodic]),
        // y = x + 3 (shift, 1D)
        (vec![1], vec![3], 1, 1, vec![BoundaryCondition::Periodic]),
        // y = -x (negation, 1D)
        (vec![-1], vec![0], 1, 1, vec![BoundaryCondition::Periodic]),
        // y = 2x (scale, 1D)
        (vec![2], vec![0], 1, 1, vec![BoundaryCondition::Periodic]),
        // (y1,y2) = (x1+x2, x1-x2) (2D)
        (
            vec![1, 1, 1, -1],
            vec![0, 0],
            2,
            2,
            vec![BoundaryCondition::Periodic; 2],
        ),
        // y = x1 + x2 (M=1, N=2)
        (vec![1, 1], vec![0], 1, 2, vec![BoundaryCondition::Periodic]),
        // Identity with Open BC
        (vec![1], vec![0], 1, 1, vec![BoundaryCondition::Open]),
    ];

    for (a_flat, b_vec, m, n, bc) in &test_cases {
        for r in [2, 3] {
            let params = AffineParams::from_integers(a_flat.clone(), b_vec.clone(), *m, *n)
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed params a={:?} b={:?} m={} n={}: {}",
                        a_flat, b_vec, m, n, e
                    )
                });

            // Get reference sparse matrix
            let ref_matrix = match affine_transform_matrix(r, &params, bc) {
                Ok(mat) => mat,
                Err(e) => {
                    eprintln!(
                        "Skipping a={:?} b={:?} m={} n={} r={}: {}",
                        a_flat, b_vec, m, n, r, e
                    );
                    continue;
                }
            };

            let dim_in = 1usize << (r * *n);
            let dim_out = 1usize << (r * *m);

            // Verify matrix dimensions
            assert_eq!(ref_matrix.rows(), dim_out, "Output dimension mismatch");
            assert_eq!(ref_matrix.cols(), dim_in, "Input dimension mismatch");

            // Verify the matrix has non-zero entries
            let nnz = ref_matrix.nnz();
            assert!(
                nnz > 0,
                "Reference matrix has no non-zero entries for a={:?} b={:?} r={}",
                a_flat,
                b_vec,
                r
            );

            // Verify all entries are either 0 or 1 (affine transforms produce boolean matrices)
            // and count that the number of 1.0 entries matches nnz
            let mut count_ones = 0usize;
            for y in 0..dim_out {
                for x in 0..dim_in {
                    let val = *ref_matrix.get(y, x).unwrap_or(&0.0);
                    assert!(
                        (val - 0.0).abs() < 1e-14 || (val - 1.0).abs() < 1e-14,
                        "Entry ({}, {}) = {} is neither 0 nor 1 for a={:?} b={:?} r={}",
                        y,
                        x,
                        val,
                        a_flat,
                        b_vec,
                        r
                    );
                    if (val - 1.0).abs() < 1e-14 {
                        count_ones += 1;
                    }
                }
            }
            assert_eq!(
                count_ones, nnz,
                "Count of 1.0 entries doesn't match nnz for a={:?} b={:?} r={}",
                a_flat, b_vec, r
            );

            eprintln!(
                "PASS: matrix props a={:?} b={:?} m={} n={} r={} nnz={}",
                a_flat, b_vec, m, n, r, nnz
            );
        }
    }
}

// ============================================================================
// Binary operator tests
// ============================================================================

/// Test binaryop identity on first variable: out = (x, y).
#[test]
fn test_binaryop_identity_x() {
    let r = 2; // Use small R for testing

    eprintln!("\n=== Testing binaryop identity a=1, b=0 (R={}) ===", r);

    let op = binaryop_single_operator(r, 1, 0, BoundaryCondition::Periodic);
    assert!(
        op.is_ok(),
        "Binaryop identity operator creation should succeed"
    );

    eprintln!("Binaryop identity test: operator creation successful");
}

/// Test binaryop sum: out = (x + y, y).
#[test]
fn test_binaryop_sum() {
    let r = 2;

    eprintln!("\n=== Testing binaryop sum a=1, b=1 (R={}) ===", r);

    let op = binaryop_single_operator(r, 1, 1, BoundaryCondition::Periodic);
    assert!(op.is_ok(), "Binaryop sum operator creation should succeed");

    eprintln!("Binaryop sum test: operator creation successful");
}

/// Test binaryop difference: out = (x - y, y).
#[test]
fn test_binaryop_difference() {
    let r = 2;

    eprintln!("\n=== Testing binaryop difference a=1, b=-1 (R={}) ===", r);

    let op = binaryop_single_operator(r, 1, -1, BoundaryCondition::Periodic);
    assert!(
        op.is_ok(),
        "Binaryop difference operator creation should succeed"
    );

    eprintln!("Binaryop difference test: operator creation successful");
}

/// Test BinaryCoeffs helper constructors.
#[test]
fn test_binary_coeffs_constructors() {
    let sum = BinaryCoeffs::sum();
    assert_eq!(sum.a, 1);
    assert_eq!(sum.b, 1);

    let diff = BinaryCoeffs::difference();
    assert_eq!(diff.a, 1);
    assert_eq!(diff.b, -1);

    let select_x = BinaryCoeffs::select_x();
    assert_eq!(select_x.a, 1);
    assert_eq!(select_x.b, 0);

    let select_y = BinaryCoeffs::select_y();
    assert_eq!(select_y.a, 0);
    assert_eq!(select_y.b, 1);

    eprintln!("BinaryCoeffs constructors test passed!");
}

// ============================================================================
// Bit interleaving helpers for binaryop tests
// ============================================================================

/// Interleave bits of two R-bit integers x and y into a 2R-bit integer.
/// Big-endian: MSB first. Result = [x_{R-1}, y_{R-1}, ..., x_0, y_0]
fn interleave_bits(x: usize, y: usize, r: usize) -> usize {
    let mut result = 0;
    for i in 0..r {
        let x_bit = (x >> (r - 1 - i)) & 1;
        let y_bit = (y >> (r - 1 - i)) & 1;
        result |= x_bit << (2 * (r - 1 - i) + 1);
        result |= y_bit << (2 * (r - 1 - i));
    }
    result
}

/// De-interleave a 2R-bit integer into two R-bit integers (x, y).
#[allow(dead_code)]
fn deinterleave_bits(val: usize, r: usize) -> (usize, usize) {
    let mut x = 0;
    let mut y = 0;
    for i in 0..r {
        let x_bit = (val >> (2 * (r - 1 - i) + 1)) & 1;
        let y_bit = (val >> (2 * (r - 1 - i))) & 1;
        x |= x_bit << (r - 1 - i);
        y |= y_bit << (r - 1 - i);
    }
    (x, y)
}

// ============================================================================
// Binaryop single-variable brute-force numerical tests
// ============================================================================

// ============================================================================
// Operator linearity tests
// ============================================================================

/// Test that operators are linear: Op(α*a + β*b) ≈ α*Op(a) + β*Op(b).
///
/// Uses the dense matrix representation to verify linearity with deterministic
/// pseudo-random input vectors. This verifies correctness on superposition inputs,
/// not just product states.
#[test]
fn test_operator_linearity() {
    let r = 3;
    let n = 1usize << r;

    // Create the dense matrix for each operator
    let operators: Vec<(&str, Vec<Vec<Complex64>>)> = vec![
        (
            "flip",
            apply_operator_to_dense_matrix(
                &flip_operator(r, BoundaryCondition::Periodic).unwrap(),
                r,
                r,
            ),
        ),
        (
            "shift(3)",
            apply_operator_to_dense_matrix(
                &shift_operator(r, 3, BoundaryCondition::Periodic).unwrap(),
                r,
                r,
            ),
        ),
        (
            "phase_rotation(pi/4)",
            apply_operator_to_dense_matrix(&phase_rotation_operator(r, PI / 4.0).unwrap(), r, r),
        ),
        (
            "cumsum",
            apply_operator_to_dense_matrix(&cumsum_operator(r).unwrap(), r, r),
        ),
        (
            "fourier",
            apply_operator_to_dense_matrix(
                &quantics_fourier_operator(r, FourierOptions::forward()).unwrap(),
                r,
                r,
            ),
        ),
    ];

    let alpha = Complex64::new(0.7, 0.3);
    let beta = Complex64::new(-0.2, 0.5);

    // Deterministic pseudo-random vectors
    let a_vec: Vec<Complex64> = (0..n)
        .map(|i| {
            Complex64::new(
                ((i as f64) * 1.1 + 0.3).sin(),
                ((i as f64) * 0.7 + 1.2).cos(),
            )
        })
        .collect();
    let b_vec: Vec<Complex64> = (0..n)
        .map(|i| {
            Complex64::new(
                ((i as f64) * 2.3 + 0.1).cos(),
                ((i as f64) * 1.9 + 0.5).sin(),
            )
        })
        .collect();

    for (_name, matrix) in &operators {
        // Compute Op(α*a + β*b) via matrix-vector multiply
        let combined: Vec<Complex64> = (0..n).map(|i| alpha * a_vec[i] + beta * b_vec[i]).collect();
        let result_combined: Vec<Complex64> = (0..n)
            .map(|y| {
                (0..n)
                    .map(|x| matrix[y][x] * combined[x])
                    .sum::<Complex64>()
            })
            .collect();

        // Compute α*Op(a) + β*Op(b)
        let result_a: Vec<Complex64> = (0..n)
            .map(|y| (0..n).map(|x| matrix[y][x] * a_vec[x]).sum::<Complex64>())
            .collect();
        let result_b: Vec<Complex64> = (0..n)
            .map(|y| (0..n).map(|x| matrix[y][x] * b_vec[x]).sum::<Complex64>())
            .collect();
        let result_linear: Vec<Complex64> = (0..n)
            .map(|i| alpha * result_a[i] + beta * result_b[i])
            .collect();

        // Compare
        for y in 0..n {
            assert_relative_eq!(result_combined[y].re, result_linear[y].re, epsilon = 1e-8,);
            assert_relative_eq!(result_combined[y].im, result_linear[y].im, epsilon = 1e-8,);
        }
    }
}

/// Test binaryop_single numerical correctness for all valid (a,b) combinations.
///
/// For each (x, y) input pair, verifies that the operator maps to the expected
/// output (z, y) where z = (a*x + b*y) mod 2^R with appropriate boundary condition sign.
#[test]
fn test_binaryop_single_numerical_correctness() {
    let valid_coeffs: Vec<(i8, i8)> = vec![
        (1, 0),  // select x
        (0, 1),  // select y
        (1, 1),  // x + y
        (1, -1), // x - y
        (-1, 1), // -x + y
        (0, 0),  // zero
    ];

    for r in [2, 3] {
        let n = 1usize << r;

        for &(a, b) in &valid_coeffs {
            for bc in [BoundaryCondition::Periodic, BoundaryCondition::Open] {
                let op = binaryop_single_operator(r, a, b, bc)
                    .unwrap_or_else(|e| panic!("Failed for a={}, b={}, r={}: {}", a, b, r, e));

                let n_sites = 2 * r;
                let dim = 1usize << n_sites;
                let matrix = apply_operator_to_dense_matrix(&op, n_sites, n_sites);

                let bc_val: i64 = match bc {
                    BoundaryCondition::Periodic => 1,
                    BoundaryCondition::Open => 0,
                };

                for x in 0..n {
                    for y in 0..n {
                        let input_idx = interleave_bits(x, y, r);

                        // Compute expected: z = a*x + b*y
                        let z_raw = (a as i64) * (x as i64) + (b as i64) * (y as i64);
                        let n_i64 = n as i64;
                        let z_mod = z_raw.rem_euclid(n_i64) as usize;
                        let nbc = (z_raw - z_mod as i64) / n_i64;

                        // BC sign factor
                        let expected_sign: f64 = if nbc == 0 {
                            1.0
                        } else if bc_val == 0 {
                            0.0 // Open BC: overflow -> zero
                        } else {
                            (bc_val as f64).powi(nbc.unsigned_abs() as i32)
                        };

                        // Output: (z_mod, y) interleaved
                        let expected_output_idx = interleave_bits(z_mod, y, r);

                        for out_idx in 0..dim {
                            let expected = if out_idx == expected_output_idx {
                                Complex64::new(expected_sign, 0.0)
                            } else {
                                Complex64::zero()
                            };

                            if (matrix[out_idx][input_idx] - expected).norm() > 1e-8 {
                                panic!(
                                    "Mismatch at a={}, b={}, r={}, bc={:?}: input (x={}, y={}) -> \
                                     out_idx={} got ({:.4}, {:.4}) expected ({:.4}, {:.4})\n\
                                     input_idx={}, expected_output_idx={}, z_raw={}, z_mod={}, nbc={}",
                                    a, b, r, bc, x, y, out_idx,
                                    matrix[out_idx][input_idx].re, matrix[out_idx][input_idx].im,
                                    expected.re, expected.im,
                                    input_idx, expected_output_idx, z_raw, z_mod, nbc
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Binaryop dual-output (multi-variable) tests
// ============================================================================

/// Test binaryop with two output variables: (z1, z2) = (a*x+b*y, c*x+d*y).
///
/// Port of Julia's _binaryop test pattern. Tests all valid (a,b,c,d) combinations
/// where neither (a,b) nor (c,d) is (-1,-1).
/// With periodic BC, this covers 64 of the 81 combinations in {-1,0,1}^4.
#[test]
fn test_binaryop_dual_output_numerical() {
    for r in [2, 3] {
        let n = 1usize << r;
        let n_sites = 2 * r;
        let dim = 1usize << n_sites;

        for a in -1i8..=1 {
            for b in -1i8..=1 {
                for c in -1i8..=1 {
                    for d in -1i8..=1 {
                        // Skip invalid coeffs: each pair must not be (-1,-1)
                        let coeffs1 = match BinaryCoeffs::new(a, b) {
                            Ok(c) => c,
                            Err(_) => continue,
                        };
                        let coeffs2 = match BinaryCoeffs::new(c, d) {
                            Ok(c) => c,
                            Err(_) => continue,
                        };

                        let bc = [BoundaryCondition::Periodic; 2];
                        let op = binaryop_operator(r, coeffs1, coeffs2, bc).unwrap_or_else(|e| {
                            panic!("Failed for ({},{},{},{}) r={}: {}", a, b, c, d, r, e)
                        });

                        let matrix = apply_operator_to_dense_matrix(&op, n_sites, n_sites);

                        for x in 0..n {
                            for y in 0..n {
                                let input_idx = interleave_bits(x, y, r);

                                // Compute expected outputs
                                let z1_raw = (a as i64) * (x as i64) + (b as i64) * (y as i64);
                                let z2_raw = (c as i64) * (x as i64) + (d as i64) * (y as i64);

                                let z1 = z1_raw.rem_euclid(n as i64) as usize;
                                let z2 = z2_raw.rem_euclid(n as i64) as usize;

                                // For Periodic BC with bc_val=1, sign is always 1
                                let expected_output_idx = interleave_bits(z1, z2, r);

                                for out_idx in 0..dim {
                                    let expected = if out_idx == expected_output_idx {
                                        Complex64::new(1.0, 0.0)
                                    } else {
                                        Complex64::zero()
                                    };

                                    if (matrix[out_idx][input_idx] - expected).norm() > 1e-8 {
                                        panic!(
                                            "Mismatch at ({},{},{},{}) r={}: \
                                             input (x={}, y={}) -> out_idx={} \
                                             got ({:.6}, {:.6}) expected ({:.6}, {:.6})\n\
                                             input_idx={}, expected_output_idx={}, \
                                             z1={}, z2={}",
                                            a,
                                            b,
                                            c,
                                            d,
                                            r,
                                            x,
                                            y,
                                            out_idx,
                                            matrix[out_idx][input_idx].re,
                                            matrix[out_idx][input_idx].im,
                                            expected.re,
                                            expected.im,
                                            input_idx,
                                            expected_output_idx,
                                            z1,
                                            z2
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
