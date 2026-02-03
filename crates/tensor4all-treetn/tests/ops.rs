//! Tests for TreeTN operations: norm, norm_squared, inner, to_dense, evaluate.

use std::collections::HashMap;

use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_treetn::TreeTN;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a DynIndex with specific size.
fn idx(size: usize) -> DynIndex {
    DynIndex::new_dyn(size)
}

/// Create a TensorDynLen from indices and f64 data.
fn make_tensor(indices: Vec<DynIndex>, data: Vec<f64>) -> TensorDynLen {
    TensorDynLen::from_dense_f64(indices, data)
}

/// Create a simple 2-node linear TreeTN with named nodes (usize).
/// Structure: node 0 (phys s0) -- bond -- node 1 (phys s1)
fn create_two_node_named() -> (
    TreeTN<TensorDynLen, usize>,
    DynIndex, // s0 (physical index at node 0)
    DynIndex, // bond
    DynIndex, // s1 (physical index at node 1)
) {
    let s0 = idx(2);
    let bond = idx(3);
    let s1 = idx(2);

    // t0 shape: [2, 3], t1 shape: [3, 2]
    let t0 = make_tensor(
        vec![s0.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    );
    let t1 = make_tensor(
        vec![bond.clone(), s1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );

    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();

    (tn, s0, bond, s1)
}

/// Create a 3-node chain TreeTN with named nodes.
/// Structure: node 0 -- bond01 -- node 1 -- bond12 -- node 2
fn create_three_node_named() -> (
    TreeTN<TensorDynLen, usize>,
    DynIndex, // s0
    DynIndex, // s1
    DynIndex, // s2
) {
    let s0 = idx(2);
    let bond01 = idx(3);
    let s1 = idx(2);
    let bond12 = idx(3);
    let s2 = idx(2);

    // Simple tensors with sequential data
    let dims0: usize = 2 * 3;
    let dims1: usize = 3 * 2 * 3;
    let dims2: usize = 3 * 2;

    let t0 = make_tensor(
        vec![s0.clone(), bond01.clone()],
        (0..dims0).map(|i| (i + 1) as f64).collect(),
    );
    let t1 = make_tensor(
        vec![bond01.clone(), s1.clone(), bond12.clone()],
        (0..dims1).map(|i| (i + 1) as f64 * 0.1).collect(),
    );
    let t2 = make_tensor(
        vec![bond12.clone(), s2.clone()],
        (0..dims2).map(|i| (i + 1) as f64).collect(),
    );

    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();

    (tn, s0, s1, s2)
}

// ============================================================================
// Tests for norm and norm_squared
// ============================================================================

#[test]
fn test_norm_single_node() {
    let s0 = idx(3);
    let t0 = make_tensor(vec![s0.clone()], vec![1.0, 2.0, 3.0]);
    let mut tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0], vec![0]).unwrap();

    let norm = tn.norm().unwrap();
    let expected = (1.0f64 + 4.0 + 9.0).sqrt();
    assert!(
        (norm - expected).abs() < 1e-10,
        "norm: got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_norm_squared_two_nodes() {
    let (mut tn, _, _, _) = create_two_node_named();

    let norm_sq = tn.norm_squared().unwrap();
    assert!(norm_sq > 0.0, "norm_squared must be positive");

    // Verify consistency: norm^2 == norm_squared
    let mut tn2 = tn.clone();
    let norm = tn2.norm().unwrap();
    assert!(
        (norm * norm - norm_sq).abs() / norm_sq < 1e-10,
        "norm^2 ({}) != norm_squared ({})",
        norm * norm,
        norm_sq
    );
}

#[test]
fn test_norm_against_dense() {
    let (mut tn, _, _, _) = create_three_node_named();

    let norm_tn = tn.norm().unwrap();

    // Compute norm from dense tensor
    let dense = tn.to_dense().unwrap();
    let norm_dense = dense.norm_squared().sqrt();

    assert!(
        (norm_tn - norm_dense).abs() / norm_dense < 1e-10,
        "TreeTN norm ({}) != dense norm ({})",
        norm_tn,
        norm_dense
    );
}

// ============================================================================
// Tests for inner product
// ============================================================================

#[test]
fn test_inner_self() {
    let (tn, _, _, _) = create_two_node_named();

    let inner = tn.inner(&tn).unwrap();
    let mut tn2 = tn.clone();
    let norm_sq = tn2.norm_squared().unwrap();

    assert!(
        (inner.real() - norm_sq).abs() / norm_sq.max(1e-15) < 1e-10,
        "inner(self, self) ({}) != norm_squared ({})",
        inner.real(),
        norm_sq
    );
}

#[test]
fn test_inner_different_networks() {
    let s0 = idx(2);
    let bond = idx(3);
    let s1 = idx(2);

    let t0a = make_tensor(
        vec![s0.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    );
    let t1a = make_tensor(
        vec![bond.clone(), s1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );
    let tn_a = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0a, t1a], vec![0, 1]).unwrap();

    let t0b = make_tensor(
        vec![s0.clone(), bond.clone()],
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    );
    let t1b = make_tensor(
        vec![bond.clone(), s1.clone()],
        vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    );
    let tn_b = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0b, t1b], vec![0, 1]).unwrap();

    // Compute inner product via TreeTN method
    let inner_val = tn_a.inner(&tn_b).unwrap();

    // Compute expected via dense tensors
    let dense_a = tn_a.to_dense().unwrap();
    let dense_b = tn_b.to_dense().unwrap();
    let expected = dense_a.inner_product(&dense_b).unwrap();

    assert!(
        (inner_val.real() - expected.real()).abs() < 1e-10,
        "inner({}, {}) != expected ({})",
        inner_val.real(),
        expected.real(),
        expected.real()
    );
}

#[test]
fn test_inner_three_nodes() {
    let (tn, _, _, _) = create_three_node_named();

    let inner = tn.inner(&tn).unwrap();
    let mut tn2 = tn.clone();
    let norm_sq = tn2.norm_squared().unwrap();

    assert!(
        (inner.real() - norm_sq).abs() / norm_sq.max(1e-15) < 1e-10,
        "inner(self, self) ({}) != norm_squared ({})",
        inner.real(),
        norm_sq
    );
}

// ============================================================================
// Tests for to_dense
// ============================================================================

#[test]
fn test_to_dense_single_node() {
    let s0 = idx(3);
    let t0 = make_tensor(vec![s0.clone()], vec![1.0, 2.0, 3.0]);
    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0.clone()], vec![0]).unwrap();

    let dense = tn.to_dense().unwrap();
    let dense_data = dense.as_slice_f64().unwrap();
    let expected_data = t0.as_slice_f64().unwrap();

    assert_eq!(dense_data.len(), expected_data.len());
    for (i, (&d, &e)) in dense_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (d - e).abs() < 1e-10,
            "to_dense mismatch at {}: {} vs {}",
            i,
            d,
            e
        );
    }
}

#[test]
fn test_to_dense_two_nodes() {
    let (tn, _, _, _) = create_two_node_named();
    let dense = tn.to_dense().unwrap();

    // Result should have only site indices (bond contracted)
    assert_eq!(dense.external_indices().len(), 2);
}

#[test]
fn test_to_dense_empty() {
    let tn = TreeTN::<TensorDynLen, usize>::new();
    let result = tn.to_dense();
    assert!(result.is_err());
}

// ============================================================================
// Tests for evaluate
// ============================================================================

#[test]
fn test_evaluate_single_node() {
    let s0 = idx(3);
    let t0 = make_tensor(vec![s0.clone()], vec![10.0, 20.0, 30.0]);
    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0], vec![0]).unwrap();

    // Evaluate at index 0
    let mut index_values = HashMap::new();
    index_values.insert(0usize, vec![0]);
    let val = tn.evaluate(&index_values).unwrap();
    assert!(
        (val.real() - 10.0).abs() < 1e-10,
        "evaluate at [0] = {}, expected 10.0",
        val.real()
    );

    // Evaluate at index 2
    index_values.insert(0usize, vec![2]);
    let val = tn.evaluate(&index_values).unwrap();
    assert!(
        (val.real() - 30.0).abs() < 1e-10,
        "evaluate at [2] = {}, expected 30.0",
        val.real()
    );
}

#[test]
fn test_evaluate_two_nodes() {
    let (tn, _, _, _) = create_two_node_named();

    // Get the dense representation for reference
    let dense = tn.to_dense().unwrap();
    let dense_data = dense.as_slice_f64().unwrap();

    // Get ordered site indices to know mapping
    let site_inds_0 = tn
        .site_space(&0)
        .unwrap()
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let site_inds_1 = tn
        .site_space(&1)
        .unwrap()
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let dim0 = site_inds_0[0].dim();
    let dim1 = site_inds_1[0].dim();

    // Evaluate at each combination and verify against dense
    for i in 0..dim0 {
        for j in 0..dim1 {
            let mut index_values = HashMap::new();
            index_values.insert(0usize, vec![i]);
            index_values.insert(1usize, vec![j]);
            let val = tn.evaluate(&index_values).unwrap();

            // Get the expected value from dense tensor
            // The dense tensor indices are ordered by node name (0, then 1)
            let flat_idx = i * dim1 + j;
            let expected = dense_data[flat_idx];

            assert!(
                (val.real() - expected).abs() < 1e-10,
                "evaluate at [{}, {}] = {}, expected {}",
                i,
                j,
                val.real(),
                expected
            );
        }
    }
}

#[test]
fn test_evaluate_three_nodes() {
    let (tn, _, _, _) = create_three_node_named();

    // Get dense for reference
    let dense = tn.to_dense().unwrap();
    let dense_data = dense.as_slice_f64().unwrap();

    let _dim0 = tn.site_space(&0).unwrap().iter().next().unwrap().dim();
    let dim1 = tn.site_space(&1).unwrap().iter().next().unwrap().dim();
    let dim2 = tn.site_space(&2).unwrap().iter().next().unwrap().dim();

    // Spot-check a few values
    for (i, j, k) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)] {
        let mut index_values = HashMap::new();
        index_values.insert(0usize, vec![i]);
        index_values.insert(1usize, vec![j]);
        index_values.insert(2usize, vec![k]);
        let val = tn.evaluate(&index_values).unwrap();

        let flat_idx = i * dim1 * dim2 + j * dim2 + k;
        let expected = dense_data[flat_idx];

        assert!(
            (val.real() - expected).abs() < 1e-8,
            "evaluate at [{}, {}, {}] = {}, expected {}",
            i,
            j,
            k,
            val.real(),
            expected
        );
    }
}

#[test]
fn test_evaluate_empty() {
    let tn = TreeTN::<TensorDynLen, usize>::new();
    let index_values = HashMap::new();
    let result = tn.evaluate(&index_values);
    assert!(result.is_err());
}

// ============================================================================
// Tests for add (which already exists but we test the public API)
// ============================================================================

#[test]
fn test_add_two_nodes() {
    let (tn_a, _, _, _) = create_two_node_named();
    let tn_b = tn_a.clone();

    let sum = tn_a.add(&tn_b).unwrap();

    // Verify numerically: sum.to_dense() == tn_a.to_dense() + tn_b.to_dense()
    let sum_dense = sum.to_dense().unwrap();
    let a_dense = tn_a.to_dense().unwrap();

    // sum should equal 2 * a_dense
    let sum_data = sum_dense.as_slice_f64().unwrap();
    let a_data = a_dense.as_slice_f64().unwrap();

    assert_eq!(sum_data.len(), a_data.len());
    for (i, (&s, &a)) in sum_data.iter().zip(a_data.iter()).enumerate() {
        let expected = 2.0 * a;
        assert!(
            (s - expected).abs() < 1e-10,
            "add mismatch at {}: {} vs {}",
            i,
            s,
            expected
        );
    }
}
