use super::*;
use crate::types::{tensor3_zeros, Tensor3};

#[test]
fn test_dot_constant() {
    let tt1 = TensorTrain::<f64>::constant(&[2, 3], 2.0);
    let tt2 = TensorTrain::<f64>::constant(&[2, 3], 3.0);

    let result = tt1.dot(&tt2).unwrap();

    // Each element product is 2.0 * 3.0 = 6.0
    // Sum over 2*3=6 elements: 6.0 * 6 = 36.0
    assert!((result - 36.0).abs() < 1e-10);
}

#[test]
fn test_dot_different_tensors() {
    let mut t0_a: Tensor3<f64> = tensor3_zeros(1, 3, 2);
    for s in 0..3 {
        for r in 0..2 {
            t0_a.set3(0, s, r, (s + r + 1) as f64);
        }
    }

    let mut t1_a: Tensor3<f64> = tensor3_zeros(2, 2, 1);
    for l in 0..2 {
        for s in 0..2 {
            t1_a.set3(l, s, 0, (l + s + 1) as f64);
        }
    }

    let tt_a = TensorTrain::new(vec![t0_a.clone(), t1_a.clone()]).unwrap();

    let mut t0_b: Tensor3<f64> = tensor3_zeros(1, 3, 1);
    for s in 0..3 {
        t0_b.set3(0, s, 0, (s + 1) as f64 * 0.5);
    }

    let mut t1_b: Tensor3<f64> = tensor3_zeros(1, 2, 1);
    for s in 0..2 {
        t1_b.set3(0, s, 0, (s + 2) as f64 * 0.3);
    }

    let tt_b = TensorTrain::new(vec![t0_b, t1_b]).unwrap();

    let dot_result = tt_a.dot(&tt_b).unwrap();

    // Compute expected value by brute force
    let mut expected = 0.0;
    for i0 in 0..3 {
        for i1 in 0..2 {
            let val_a = tt_a.evaluate(&[i0, i1]).unwrap();
            let val_b = tt_b.evaluate(&[i0, i1]).unwrap();
            expected += val_a * val_b;
        }
    }

    assert!(
        (dot_result - expected).abs() < 1e-10,
        "dot = {}, expected = {}",
        dot_result,
        expected
    );
}

#[test]
fn test_dot_length_mismatch() {
    let tt1 = TensorTrain::<f64>::constant(&[2, 3], 1.0);
    let tt2 = TensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    assert!(tt1.dot(&tt2).is_err());
}

#[test]
fn test_dot_site_dim_mismatch() {
    let tt1 = TensorTrain::<f64>::constant(&[2, 3], 1.0);
    let tt2 = TensorTrain::<f64>::constant(&[2, 4], 1.0);
    assert!(tt1.dot(&tt2).is_err());
}

#[test]
fn test_dot_site_dim_mismatch_middle() {
    // Mismatch at site 1 (not site 0) to cover the inner loop error path
    let mut t0: Tensor3<f64> = tensor3_zeros(1, 2, 1);
    t0.set3(0, 0, 0, 1.0);
    t0.set3(0, 1, 0, 1.0);

    let mut t1_a: Tensor3<f64> = tensor3_zeros(1, 3, 1);
    for s in 0..3 {
        t1_a.set3(0, s, 0, 1.0);
    }
    let mut t1_b: Tensor3<f64> = tensor3_zeros(1, 4, 1);
    for s in 0..4 {
        t1_b.set3(0, s, 0, 1.0);
    }

    let tt_a = TensorTrain::new(vec![t0.clone(), t1_a]).unwrap();
    let tt_b = TensorTrain::new(vec![t0, t1_b]).unwrap();
    assert!(tt_a.dot(&tt_b).is_err());
}

#[test]
fn test_dot_empty() {
    let tt1 = TensorTrain::<f64>::from_tensors_unchecked(Vec::new());
    let tt2 = TensorTrain::<f64>::from_tensors_unchecked(Vec::new());
    let result = tt1.dot(&tt2).unwrap();
    assert!((result - 0.0).abs() < 1e-15);
}

fn test_dot_generic<T: TTScalar + matrixci::Scalar + Default>() {
    let tt1 = TensorTrain::<T>::constant(&[2, 3], T::from_f64(2.0));
    let tt2 = TensorTrain::<T>::constant(&[2, 3], T::from_f64(3.0));

    let result = tt1.dot(&tt2).unwrap();
    // Each element: 2*3 = 6, sum over 2*3=6 elements = 36
    assert!(
        (result - T::from_f64(36.0)).abs_sq().sqrt() < 1e-10,
        "dot product mismatch"
    );
}

#[test]
fn test_dot_f64() {
    test_dot_generic::<f64>();
}

#[test]
fn test_dot_c64() {
    test_dot_generic::<num_complex::Complex64>();
}

#[test]
fn test_dot_convenience_function() {
    let tt1 = TensorTrain::<f64>::constant(&[2, 3], 2.0);
    let tt2 = TensorTrain::<f64>::constant(&[2, 3], 3.0);
    let result = dot(&tt1, &tt2).unwrap();
    assert!((result - 36.0).abs() < 1e-10);
}

#[test]
fn test_dot_three_sites() {
    // Test dot product with 3 sites to exercise the inner loop more fully
    let tt1 = TensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let tt2 = TensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let result = tt1.dot(&tt2).unwrap();
    // Each element: 1*2 = 2, total elements = 2*3*2 = 12, sum = 24
    assert!((result - 24.0).abs() < 1e-10);
}
