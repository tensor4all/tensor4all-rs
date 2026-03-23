use super::*;

#[test]
fn test_tensorci1_new() {
    let tci = TensorCI1::<f64>::new(vec![2, 3, 2]);
    assert_eq!(tci.len(), 3);
    assert_eq!(tci.local_dims(), &[2, 3, 2]);
}

#[test]
fn test_crossinterpolate1_constant() {
    let f = |_: &MultiIndex| 1.0f64;
    let local_dims = vec![2, 2];
    let first_pivot = vec![0, 0];
    let options = TCI1Options::default();

    let (tci, _ranks, _errors) = crossinterpolate1(f, local_dims, first_pivot, options).unwrap();
    assert_eq!(tci.len(), 2);
    assert!(tci.rank() >= 1);
}

#[test]
fn test_crossinterpolate1_simple() {
    // f(i, j) = i + j + 1
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    let local_dims = vec![3, 3];
    let first_pivot = vec![1, 1];
    let options = TCI1Options::default();

    let (tci, _ranks, _errors) = crossinterpolate1(f, local_dims, first_pivot, options).unwrap();
    assert_eq!(tci.len(), 2);
}

#[test]
fn test_crossinterpolate1_evaluate_at_pivot() {
    // f(i, j) = (i + 1) * (j + 1)
    let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
    let local_dims = vec![3, 3];
    let first_pivot = vec![1, 1];
    let options = TCI1Options::default();

    let (tci, _ranks, _errors) =
        crossinterpolate1(f, local_dims.clone(), first_pivot.clone(), options).unwrap();

    // The TCI should exactly reproduce the function at the pivot
    let val = tci.evaluate(&first_pivot).unwrap();
    let expected = f(&first_pivot);
    assert!(
        (val - expected).abs() < 1e-10,
        "TCI evaluate at pivot: got {}, expected {}",
        val,
        expected
    );
}

#[test]
fn test_crossinterpolate1_evaluate_on_cross() {
    // f(i, j) = (i + 1) * (j + 1)
    let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
    let local_dims = vec![3, 3];
    let first_pivot = vec![1, 1];
    let options = TCI1Options::default();

    let (tci, _ranks, _errors) =
        crossinterpolate1(f, local_dims.clone(), first_pivot.clone(), options).unwrap();

    // Test evaluation at points on the cross through the pivot
    // Points (i, 1) for all i
    for i in 0..3 {
        let idx = vec![i, 1];
        let val = tci.evaluate(&idx).unwrap();
        let expected = f(&idx);
        assert!(
            (val - expected).abs() < 1e-10,
            "TCI evaluate at {:?}: got {}, expected {}",
            idx,
            val,
            expected
        );
    }

    // Points (1, j) for all j
    for j in 0..3 {
        let idx = vec![1, j];
        let val = tci.evaluate(&idx).unwrap();
        let expected = f(&idx);
        assert!(
            (val - expected).abs() < 1e-10,
            "TCI evaluate at {:?}: got {}, expected {}",
            idx,
            val,
            expected
        );
    }
}

#[test]
fn test_add_pivot_row_inconsistent_index() {
    let mut tci = TensorCI1::<f64>::new(vec![2, 2]);
    let f = |_idx: &MultiIndex| 1.0;

    let err = tci.add_pivot_row(0, 0, &f).unwrap_err();
    assert!(matches!(err, TCIError::IndexInconsistency { .. }));
}

#[test]
fn test_add_pivot_col_inconsistent_index() {
    let mut tci = TensorCI1::<f64>::new(vec![2, 2]);
    let f = |_idx: &MultiIndex| 1.0;

    let err = tci.add_pivot_col(0, 0, &f).unwrap_err();
    assert!(matches!(err, TCIError::IndexInconsistency { .. }));
}

#[test]
fn test_crossinterpolate1_to_tensor_train() {
    let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
    let local_dims = vec![3, 3];
    let first_pivot = vec![1, 1];
    let options = TCI1Options::default();

    let (tci, _ranks, _errors) =
        crossinterpolate1(f, local_dims, first_pivot.clone(), options).unwrap();

    // Convert to TensorTrain
    let tt = tci.to_tensor_train().unwrap();
    assert_eq!(tt.len(), 2);

    // Evaluate at the pivot using the TensorTrain
    use tensor4all_simplett::AbstractTensorTrain;
    let val = tt.evaluate(&first_pivot).unwrap();
    let expected = f(&first_pivot);
    assert!(
        (val - expected).abs() < 1e-10,
        "TensorTrain evaluate at pivot: got {}, expected {}",
        val,
        expected
    );
}

#[test]
fn test_crossinterpolate1_3d() {
    // 3D function: f(i, j, k) = i + j + k + 1
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + idx[2] + 1) as f64;
    let local_dims = vec![2, 2, 2];
    let first_pivot = vec![0, 0, 0];
    let options = TCI1Options::default();

    let (tci, _ranks, _errors) =
        crossinterpolate1(f, local_dims, first_pivot.clone(), options).unwrap();

    assert_eq!(tci.len(), 3);

    // Test evaluation at the pivot
    let val = tci.evaluate(&first_pivot).unwrap();
    let expected = f(&first_pivot);
    assert!(
        (val - expected).abs() < 1e-10,
        "3D TCI evaluate at pivot: got {}, expected {}",
        val,
        expected
    );
}

#[test]
fn test_crossinterpolate1_rank2_function() {
    // A rank-2 function: f(i, j) = i + j (not separable)
    let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
    let local_dims = vec![4, 4];
    let first_pivot = vec![1, 1];
    let options = TCI1Options {
        tolerance: 1e-12,
        ..Default::default()
    };

    let (tci, _ranks, errors) =
        crossinterpolate1(f, local_dims.clone(), first_pivot, options).unwrap();

    // Should converge to rank 2
    assert!(tci.rank() <= 3, "Expected rank <= 3, got {}", tci.rank());

    // Check error is small
    let final_error = errors.last().copied().unwrap_or(f64::INFINITY);
    assert!(
        final_error < 1e-8,
        "Expected small error, got {}",
        final_error
    );

    // Test all values
    for i in 0..4 {
        for j in 0..4 {
            let idx = vec![i, j];
            let val = tci.evaluate(&idx).unwrap();
            let expected = f(&idx);
            assert!(
                (val - expected).abs() < 1e-8,
                "TCI evaluate at {:?}: got {}, expected {}",
                idx,
                val,
                expected
            );
        }
    }
}

#[test]
fn test_crossinterpolate1_converges() {
    // A smooth function that should converge well
    let f = |idx: &MultiIndex| {
        let x = idx[0] as f64 / 4.0;
        let y = idx[1] as f64 / 4.0;
        (x * y + 0.1).sin()
    };
    let local_dims = vec![5, 5];
    let first_pivot = vec![2, 2];
    let options = TCI1Options {
        tolerance: 1e-6,
        ..Default::default()
    };

    let (tci, _ranks, errors) =
        crossinterpolate1(f, local_dims.clone(), first_pivot, options).unwrap();

    // Should achieve reasonable rank
    assert!(tci.rank() <= 5, "Expected rank <= 5, got {}", tci.rank());

    // Check errors decrease
    if errors.len() > 1 {
        let first_error = errors[0];
        let last_error = errors.last().copied().unwrap();
        assert!(
            last_error <= first_error || last_error < 1e-6,
            "Errors should decrease or converge: first={}, last={}",
            first_error,
            last_error
        );
    }
}
