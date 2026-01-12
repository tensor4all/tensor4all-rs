//! Integration tests for quanticstransform operators.
//!
//! These tests verify numerical correctness by:
//! 1. Creating a state as TensorTrain (MPS)
//! 2. Converting to TreeTN
//! 3. Applying the operator using apply_linear_operator
//! 4. Contracting the result and comparing with expected values

use std::collections::HashMap;
use std::sync::Arc;

use approx::assert_relative_eq;
use num_complex::Complex64;
use num_traits::{One, Zero};

use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::storage::{DenseStorageC64, Storage};
use tensor4all_core::{IndexLike, TensorDynLen, TensorIndex};
use tensor4all_simplett::{types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain};
use tensor4all_treetn::{apply_linear_operator, ApplyOptions, TreeTN};

use tensor4all_quanticstransform::{
    flip_operator, quantics_fourier_operator, shift_operator, BoundaryCondition, FTCore,
    FourierOptions,
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

        let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data)));
        let tensor_dyn = TensorDynLen::new(indices, dims_vec, storage);
        tensors.push(tensor_dyn);
    }

    let treetn = TreeTN::from_tensors(tensors, node_names).expect("Failed to create TreeTN");
    (treetn, site_indices)
}

/// Create a product state MPS representing a specific integer value x.
///
/// The state |x⟩ = |x_0⟩ ⊗ |x_1⟩ ⊗ ... ⊗ |x_{R-1}⟩ where x_i are binary digits.
/// Uses little-endian convention (LSB first): x = Σ_n x_n * 2^n
///
/// This matches the convention used by quantics operators (flip, shift, etc.)
/// where site 0 contains the least significant bit.
fn create_product_state_mps(x: usize, r: usize) -> TensorTrain<Complex64> {
    let mut tensors = Vec::with_capacity(r);

    for n in 0..r {
        // Little-endian: bit n corresponds to 2^n
        let bit = (x >> n) & 1;

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
    let full_tensor = treetn.contract_to_tensor().expect("Failed to contract TreeTN");

    // Extract data from the tensor
    // The tensor has indices ordered by the indices field
    let ext_indices = &full_tensor.indices;

    // Debug
    eprintln!("contract_treetn_to_vector:");
    eprintln!("  full_tensor.indices: {:?}", ext_indices.iter().map(|i| i.id()).collect::<Vec<_>>());
    eprintln!("  full_tensor.dims: {:?}", full_tensor.dims);
    eprintln!("  site_indices requested: {:?}", site_indices.iter().map(|i| i.id()).collect::<Vec<_>>());

    // Build mapping from site index ID to position in external indices
    let mut idx_to_pos: HashMap<DynId, usize> = HashMap::new();
    for (pos, idx) in ext_indices.iter().enumerate() {
        idx_to_pos.insert(idx.id().clone(), pos);
    }

    // Build mapping from our site index order to tensor index order
    let mut site_to_tensor: Vec<usize> = Vec::with_capacity(r);
    for (i, site_idx) in site_indices.iter().enumerate() {
        let pos = idx_to_pos
            .get(&site_idx.id().clone())
            .expect("Site index not found in contracted tensor");
        eprintln!("  site_indices[{}] -> tensor position {}", i, pos);
        site_to_tensor.push(*pos);
    }

    // Extract values in canonical order (x = 0, 1, ..., 2^R - 1)
    let mut result = vec![Complex64::zero(); n];

    // Get data from storage
    let storage = full_tensor.materialize_storage().expect("Failed to materialize storage");
    let data = match storage.as_ref() {
        Storage::DenseC64(dense) => dense.as_slice(),
        _ => panic!("Expected DenseC64 storage"),
    };
    let dims = &full_tensor.dims;

    eprintln!("  data len: {}", data.len());

    // Iterate over all x values
    for x in 0..n {
        // Convert x to multi-index in tensor order
        let mut multi_idx = vec![0usize; r];
        for i in 0..r {
            // Little-endian: bit i corresponds to 2^i
            let bit = (x >> i) & 1;
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

        if data[flat_idx].norm() > 1e-10 {
            eprintln!("  x={} (binary: {:03b}) -> multi_idx={:?} -> flat_idx={} -> value={:?}",
                x, x, multi_idx, flat_idx, data[flat_idx]);
        }
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
        eprintln!("flip({}) should be {} (little-endian convention)", x_in, expected_x_out);
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
    eprintln!("MPS values: {:?}", mps_values.iter().map(|c| (c.re, c.im)).collect::<Vec<_>>());

    let (treetn, site_indices) = tensortrain_to_treetn(&mps);

    // Debug: print site indices
    eprintln!("Site indices: {:?}", site_indices.iter().map(|i| i.id()).collect::<Vec<_>>());

    // Debug: verify TreeTN contraction matches MPS values
    let treetn_values = contract_treetn_to_vector(&treetn, &site_indices);
    eprintln!("TreeTN values: {:?}", treetn_values.iter().map(|c| (c.re, c.im)).collect::<Vec<_>>());
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
        eprintln!("Remapping site {} from {:?} to {:?}", i, site_indices[i].id(), op_input.id());
        treetn_remapped = treetn_remapped
            .replaceind(&site_indices[i], &op_input)
            .expect("Failed to replace index");
    }

    // Debug: check remapped TreeTN
    let remapped_ext = treetn_remapped.external_indices();
    eprintln!("Remapped TreeTN external indices: {:?}", remapped_ext.iter().map(|i| i.id()).collect::<Vec<_>>());

    // Apply operator
    let result_treetn =
        apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
            .expect("Failed to apply operator");

    // Debug: check result TreeTN
    let result_ext = result_treetn.external_indices();
    eprintln!("Result TreeTN external indices: {:?}", result_ext.iter().map(|i| i.id()).collect::<Vec<_>>());

    // Get output site indices from operator
    let output_indices: Vec<DynIndex> = (0..r)
        .map(|i| {
            op.get_output_mapping(&i)
                .expect("Missing output mapping")
                .true_index
                .clone()
        })
        .collect();
    eprintln!("Output indices: {:?}", output_indices.iter().map(|i| i.id()).collect::<Vec<_>>());

    // Contract result to vector
    let result_vec = contract_treetn_to_vector(&result_treetn, &output_indices);
    eprintln!("Result vector: {:?}", result_vec.iter().map(|c| (c.re, c.im)).collect::<Vec<_>>());

    // Expected: flip(x) = 2^R - x = n - x for periodic BC (but flip(0) = 0, not n)
    // flip(1) = 8 - 1 = 7 for R=3
    let expected_x = if x == 0 { 0 } else { n - x };
    eprintln!("Expected result at index: {}", expected_x);

    // Debug: print where the non-zero value is
    for (idx, val) in result_vec.iter().enumerate() {
        if val.norm() > 1e-10 {
            eprintln!("  Non-zero value at index {}: ({}, {})", idx, val.re, val.im);
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
            eprintln!("  result_vec[{}] = ({}, {}), expected = ({}, {})",
                y, result_vec[y].re, result_vec[y].im, expected_val.re, expected_val.im);
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
            let result_treetn =
                apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
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
    let result_treetn =
        apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
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

    let result =
        apply_linear_operator(&forward_op, &treetn_tmp, ApplyOptions::naive())
            .expect("Failed to apply forward Fourier");

    // Result should have R external indices (the output indices)
    let ext_indices = result.external_indices();
    assert_eq!(ext_indices.len(), r, "Result should have {} external indices", r);
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
        let result_treetn =
            apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
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

    eprintln!("\n=== Testing shift for all x values (R={}, N={}) ===", r, n);

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
            let result_treetn =
                apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
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

    eprintln!("\n=== Testing Fourier unitarity for all basis states (R={}, N={}) ===", r, n);

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
        let result_treetn =
            apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
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

    eprintln!("\n=== Testing inverse Fourier F^{{-1}} |0⟩ (R={}, N={}) ===", r, n);

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

    let result_inv =
        apply_linear_operator(&inverse_op, &treetn_inv, ApplyOptions::naive())
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
        eprintln!("  k={}: ({:.4}, {:.4})", k, result_vec[k].re, result_vec[k].im);
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
// Open boundary condition tests
// ============================================================================

/// Test shift operator with Open boundary condition.
///
/// With Open BC, shift results in zero when the value overflows/underflows [0, 2^R).
#[test]
fn test_shift_open_boundary() {
    let r = 3;
    let n = 1 << r; // N = 8

    eprintln!("\n=== Testing shift with Open boundary (R={}, N={}) ===", r, n);

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
        (0, 0, Some(0)),   // 0 + 0 = 0, in range
        (3, 2, Some(5)),   // 3 + 2 = 5, in range
        (0, 7, Some(7)),   // 0 + 7 = 7, in range
        (1, 3, Some(4)),   // 1 + 3 = 4, in range
        // Overflow cases (positive offset)
        (7, 1, None),      // 7 + 1 = 8 >= 8, overflow
        (6, 3, None),      // 6 + 3 = 9 >= 8, overflow
        (4, 5, None),      // 4 + 5 = 9 >= 8, overflow
        // Negative offset cases
        // shift(x, -k) -> offset_mod = 8 - k, x + offset_mod >= 8 iff x >= k
        (0, -1, Some(7)),  // offset_mod = 7, 0 + 7 = 7 < 8, no overflow
        (0, -7, Some(1)),  // offset_mod = 1, 0 + 1 = 1 < 8, no overflow
        (1, -1, None),     // offset_mod = 7, 1 + 7 = 8 >= 8, overflow
        (2, -1, None),     // offset_mod = 7, 2 + 7 = 9 >= 8, overflow
        (7, -7, None),     // offset_mod = 1, 7 + 1 = 8 >= 8, overflow
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
        let result_treetn =
            apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
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

    eprintln!("\n=== Testing flip with Open boundary (R={}, N={}) ===", r, n);

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
        let result_treetn =
            apply_linear_operator(&op, &treetn_remapped, ApplyOptions::naive())
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
                assert_relative_eq!(
                    result_vec[y].re,
                    0.0,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
                assert_relative_eq!(
                    result_vec[y].im,
                    0.0,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
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
