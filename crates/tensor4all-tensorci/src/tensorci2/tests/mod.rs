use super::*;
use std::cell::Cell;
use std::rc::Rc;
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn test_tensorci2_new() {
    let tci = TensorCI2::<f64>::new(vec![2, 3, 2]).unwrap();
    assert_eq!(tci.len(), 3);
    assert_eq!(tci.local_dims(), &[2, 3, 2]);
}

#[test]
fn test_tensorci2_requires_two_sites() {
    let result = TensorCI2::<f64>::new(vec![2]);
    assert!(result.is_err());
}

#[test]
fn test_crossinterpolate2_constant() {
    let f = |_: &MultiIndex| 1.0f64;
    let local_dims = vec![2, 2];
    let first_pivot = vec![vec![0, 0]];
    let options = TCI2Options::default();

    let (tci, _ranks, _errors) = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    assert_eq!(tci.len(), 2);
    assert!(tci.rank() >= 1);
}

#[test]
fn test_crossinterpolate2_with_batch_function() {
    // Use batch evaluation
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    let batched_f = |indices: &[MultiIndex]| -> Vec<f64> {
        indices
            .iter()
            .map(|idx| (idx[0] + idx[1] + 1) as f64)
            .collect()
    };

    let local_dims = vec![3, 3];
    let first_pivot = vec![vec![1, 1]];
    let options = TCI2Options::default();

    let (tci, _ranks, _errors) =
        crossinterpolate2(f, Some(batched_f), local_dims, first_pivot, options).unwrap();

    assert_eq!(tci.len(), 2);
}

#[test]
fn test_crossinterpolate2_rank2_function() {
    // f(i, j) = i + j
    let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
    let local_dims = vec![4, 4];
    let first_pivot = vec![vec![1, 1]];
    let options = TCI2Options {
        tolerance: 1e-12,
        max_iter: 10,
        ..Default::default()
    };

    let (tci, _ranks, errors) = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    // Should converge to rank 2
    assert!(tci.rank() <= 3, "Expected rank <= 3, got {}", tci.rank());

    // Check error is small
    let final_error = errors.last().copied().unwrap_or(f64::INFINITY);
    assert!(
        final_error < 0.1,
        "Expected small error, got {}",
        final_error
    );
}

#[test]
fn test_crossinterpolate2_rank2_function_rook_search() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
    let local_dims = vec![4, 4];
    let first_pivot = vec![vec![1, 1]];
    let options = TCI2Options {
        tolerance: 1e-12,
        max_iter: 10,
        pivot_search: PivotSearchStrategy::Rook,
        ..Default::default()
    };

    let (tci, _ranks, errors) = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    assert!(tci.rank() <= 3, "Expected rank <= 3, got {}", tci.rank());
    let final_error = errors.last().copied().unwrap_or(f64::INFINITY);
    assert!(
        final_error < 0.1,
        "Expected small error, got {}",
        final_error
    );
}

#[test]
fn test_crossinterpolate2_rook_search_uses_partial_batch_requests() {
    let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
    let max_batch = Rc::new(Cell::new(0usize));
    let total_requested = Rc::new(Cell::new(0usize));
    let batched_f = {
        let max_batch = Rc::clone(&max_batch);
        let total_requested = Rc::clone(&total_requested);
        move |indices: &[MultiIndex]| -> Vec<f64> {
            max_batch.set(max_batch.get().max(indices.len()));
            total_requested.set(total_requested.get() + indices.len());
            indices
                .iter()
                .map(|idx| ((idx[0] + 1) * (idx[1] + 1)) as f64)
                .collect()
        }
    };

    let local_dims = vec![8, 8];
    let first_pivot = vec![vec![0, 0]];
    let options = TCI2Options {
        max_iter: 1,
        max_bond_dim: 2,
        pivot_search: PivotSearchStrategy::Rook,
        ..Default::default()
    };

    let (_tci, _ranks, _errors) =
        crossinterpolate2(f, Some(batched_f), local_dims, first_pivot, options).unwrap();

    assert!(
        max_batch.get() < 64,
        "rook search should request partial batches, got full batch of {} entries",
        max_batch.get()
    );
    assert!(
        total_requested.get() < 64,
        "rook search should avoid evaluating the full Pi matrix, requested {} entries",
        total_requested.get()
    );
}

#[test]
fn test_crossinterpolate2_3sites_constant() {
    // 3-site constant function: f(i, j, k) = 1.0
    let f = |_: &MultiIndex| 1.0f64;
    let local_dims = vec![2, 2, 2];
    let first_pivot = vec![vec![0, 0, 0]];
    let options = TCI2Options::default();

    let result = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    );

    assert!(
        result.is_ok(),
        "crossinterpolate2 failed: {:?}",
        result.err()
    );
    let (tci, _ranks, _errors) = result.unwrap();

    assert_eq!(tci.len(), 3);
    assert!(tci.rank() >= 1);

    // Verify site tensor dimensions
    for p in 0..tci.len() {
        let t = tci.site_tensor(p);
        assert!(t.left_dim() > 0, "Site {} left_dim should be > 0", p);
        assert!(t.right_dim() > 0, "Site {} right_dim should be > 0", p);
    }

    // Test to_tensor_train conversion
    let tt_result = tci.to_tensor_train();
    assert!(
        tt_result.is_ok(),
        "to_tensor_train failed: {:?}",
        tt_result.err()
    );

    let tt = tt_result.unwrap();
    assert_eq!(tt.len(), 3);

    // Verify TT can be evaluated
    let val = tt.evaluate(&[0, 0, 0]).unwrap();
    assert!((val - 1.0).abs() < 1e-10, "Expected 1.0, got {}", val);
}

#[test]
fn test_crossinterpolate2_4sites_product() {
    // 4-site product function: f(i, j, k, l) = (1+i) * (1+j) * (1+k) * (1+l)
    let f = |idx: &MultiIndex| {
        (1 + idx[0]) as f64 * (1 + idx[1]) as f64 * (1 + idx[2]) as f64 * (1 + idx[3]) as f64
    };
    let local_dims = vec![2, 2, 2, 2];
    let first_pivot = vec![vec![0, 0, 0, 0]];
    let options = TCI2Options::default();

    let result = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    );

    assert!(
        result.is_ok(),
        "crossinterpolate2 failed: {:?}",
        result.err()
    );
    let (tci, _ranks, _errors) = result.unwrap();

    assert_eq!(tci.len(), 4);

    // Test to_tensor_train conversion
    let tt_result = tci.to_tensor_train();
    assert!(
        tt_result.is_ok(),
        "to_tensor_train failed: {:?}",
        tt_result.err()
    );

    let tt = tt_result.unwrap();
    assert_eq!(tt.len(), 4);

    // Verify evaluations
    let val = tt.evaluate(&[0, 0, 0, 0]).unwrap();
    assert!((val - 1.0).abs() < 1e-10, "f(0,0,0,0) = 1, got {}", val);

    let val = tt.evaluate(&[1, 1, 1, 1]).unwrap();
    assert!((val - 16.0).abs() < 1e-10, "f(1,1,1,1) = 16, got {}", val);
}

#[test]
fn test_crossinterpolate2_5sites_constant() {
    // 5-site constant function
    let f = |_: &MultiIndex| 2.5f64;
    let local_dims = vec![2, 2, 2, 2, 2];
    let first_pivot = vec![vec![0, 0, 0, 0, 0]];
    let options = TCI2Options::default();

    let result = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    );

    assert!(
        result.is_ok(),
        "crossinterpolate2 failed: {:?}",
        result.err()
    );
    let (tci, _ranks, _errors) = result.unwrap();

    assert_eq!(tci.len(), 5);

    let tt_result = tci.to_tensor_train();
    assert!(
        tt_result.is_ok(),
        "to_tensor_train failed: {:?}",
        tt_result.err()
    );

    let tt = tt_result.unwrap();

    // Sum should be 2.5 * 2^5 = 80
    let sum = tt.sum();
    assert!((sum - 80.0).abs() < 1e-8, "Expected sum=80, got {}", sum);
}

/// Regression test for issue #227:
/// crossinterpolate2 panics with NaN in LU for oscillatory functions (e.g. sin)
/// Reproduces the exact scenario from the issue: sin(10*x) on a quantics grid.
#[test]
fn test_crossinterpolate2_sin_quantics() {
    let r = 6;
    let local_dims = vec![2; r];

    // Quantics-to-coordinate mapping: indices [q0,..,q5] (each 0 or 1)
    // -> integer = sum q_i * 2^(R-1-i), coordinate x = integer / 2^R
    let f = |indices: &MultiIndex| -> f64 {
        let mut int_idx: usize = 0;
        for (i, &q) in indices.iter().enumerate() {
            int_idx += q * (1 << (r - 1 - i));
        }
        let x = int_idx as f64 / (1u64 << r) as f64;
        (10.0 * x).sin()
    };

    // Use [0,1,0,0,0,0] as initial pivot so f != 0
    // (x = 0.5, sin(10*0.5) = sin(5) ≈ -0.959)
    let first_pivot = vec![vec![0, 1, 0, 0, 0, 0]];
    let options = TCI2Options {
        tolerance: 1e-10,
        max_bond_dim: usize::MAX,
        max_iter: 20,
        ..Default::default()
    };

    // This previously panicked with "NaN in L matrix" inside rrlu_inplace
    let result = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    );

    assert!(
        result.is_ok(),
        "crossinterpolate2 failed for sin(10x): {:?}",
        result.err()
    );
}

#[test]
fn test_crossinterpolate2_rank2_full_grid_reconstruction() {
    // Reproduce issue #259: f(i,j) = i + j should be reconstructed exactly
    let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
    let local_dims = vec![4, 4];
    let first_pivot = vec![vec![1, 1]];
    let options = TCI2Options {
        tolerance: 1e-12,
        max_iter: 20,
        max_bond_dim: usize::MAX,
        ..Default::default()
    };

    let (tci, _ranks, _errors) = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    let tt = tci.to_tensor_train().unwrap();

    let mut max_error = 0.0f64;
    for i in 0..4 {
        for j in 0..4 {
            let expected = (i + j) as f64;
            let actual = tt.evaluate(&[i, j]).unwrap();
            max_error = max_error.max((actual - expected).abs());
        }
    }
    eprintln!("rank={} max_error={:.6e}", tci.rank(), max_error);
    assert!(
        max_error < 1e-10,
        "Full-grid reconstruction error too large: {max_error:.6e} (rank={})",
        tci.rank()
    );
}
