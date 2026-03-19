
use super::*;
use num_complex::Complex64;

// Generic test functions for f64 and Complex64

fn test_tensortrain_zeros_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::zeros(&[2, 3, 2]);
    assert_eq!(tt.len(), 3);
    assert_eq!(tt.site_dims(), vec![2, 3, 2]);
    assert_eq!(tt.rank(), 1);
}

fn test_tensortrain_constant_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::constant(&[2, 2], T::from_f64(5.0));
    assert_eq!(tt.len(), 2);

    // Sum should be 5.0 * 2 * 2 = 20.0
    let sum = tt.sum();
    assert!(sum.abs_sq().sqrt() - 20.0 < 1e-10);
}

fn test_tensortrain_constant_single_site_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::constant(&[3], T::from_f64(5.0));
    assert_eq!(tt.len(), 1);
    assert_eq!(tt.site_dims(), vec![3]);

    let sum = tt.sum();
    assert!((sum.abs_sq().sqrt() - 15.0).abs() < 1e-10);

    let (data, shape) = tt.fulltensor();
    assert_eq!(shape, vec![3]);
    assert_eq!(data.len(), 3);
    for value in data {
        assert!((value - T::from_f64(5.0)).abs_sq().sqrt() < 1e-10);
    }
}

fn test_tensortrain_evaluate_generic<T: TTScalar>() {
    // Create a simple tensor train that returns the product of indices + 1
    let _site_dims = [2, 3];

    // First tensor: values are 1 for index 0, 2 for index 1
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 1);
    t0.set3(0, 0, 0, T::from_f64(1.0));
    t0.set3(0, 1, 0, T::from_f64(2.0));

    // Second tensor: values are 1, 2, 3 for indices 0, 1, 2
    let mut t1: Tensor3<T> = tensor3_zeros(1, 3, 1);
    t1.set3(0, 0, 0, T::from_f64(1.0));
    t1.set3(0, 1, 0, T::from_f64(2.0));
    t1.set3(0, 2, 0, T::from_f64(3.0));

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // tt([0, 0]) = 1 * 1 = 1
    let val00 = tt.evaluate(&[0, 0]).unwrap();
    assert!((val00 - T::from_f64(1.0)).abs_sq().sqrt() < 1e-10);
    // tt([1, 2]) = 2 * 3 = 6
    let val12 = tt.evaluate(&[1, 2]).unwrap();
    assert!((val12 - T::from_f64(6.0)).abs_sq().sqrt() < 1e-10);
}

fn test_tensortrain_scale_generic<T: TTScalar>() {
    let mut tt = TensorTrain::<T>::constant(&[2, 2], T::from_f64(1.0));
    tt.scale(T::from_f64(3.0));

    // Sum should be 3.0 * 2 * 2 = 12.0
    let sum = tt.sum();
    assert!((sum.abs_sq().sqrt() - 12.0).abs() < 1e-10);
}

fn test_tensortrain_reverse_generic<T: TTScalar>() {
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 1);
    t0.set3(0, 0, 0, T::from_f64(1.0));
    t0.set3(0, 1, 0, T::from_f64(2.0));

    let mut t1: Tensor3<T> = tensor3_zeros(1, 3, 1);
    t1.set3(0, 0, 0, T::from_f64(1.0));
    t1.set3(0, 1, 0, T::from_f64(2.0));
    t1.set3(0, 2, 0, T::from_f64(3.0));

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    let tt_rev = tt.reverse();

    assert_eq!(tt_rev.len(), 2);
    assert_eq!(tt_rev.site_dims(), vec![3, 2]);

    // Reversed evaluation: tt_rev([2, 1]) should equal tt([1, 2])
    let diff = tt_rev.evaluate(&[2, 1]).unwrap() - tt.evaluate(&[1, 2]).unwrap();
    assert!(diff.abs_sq().sqrt() < 1e-10);
}

fn test_fulltensor_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::constant(&[2, 3], T::from_f64(5.0));
    let (data, shape) = tt.fulltensor();

    assert_eq!(shape, vec![2, 3]);
    assert_eq!(data.len(), 6);

    // All elements should be 5.0
    for val in &data {
        assert!((*val - T::from_f64(5.0)).abs_sq().sqrt() < 1e-10);
    }
}

fn test_fulltensor_matches_evaluate_generic<T: TTScalar>() {
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 1);
    t0.set3(0, 0, 0, T::from_f64(1.0));
    t0.set3(0, 1, 0, T::from_f64(2.0));

    let mut t1: Tensor3<T> = tensor3_zeros(1, 3, 1);
    t1.set3(0, 0, 0, T::from_f64(1.0));
    t1.set3(0, 1, 0, T::from_f64(2.0));
    t1.set3(0, 2, 0, T::from_f64(3.0));

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    let (data, shape) = tt.fulltensor();

    assert_eq!(shape, vec![2, 3]);

    // Check each element matches evaluate
    for i in 0..2 {
        for j in 0..3 {
            let idx = i + 2 * j;
            let expected = tt.evaluate(&[i, j]).unwrap();
            let diff = data[idx] - expected;
            assert!(diff.abs_sq().sqrt() < 1e-10, "Mismatch at [{}, {}]", i, j);
        }
    }
}

fn test_log_norm_matches_norm_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::constant(&[2, 3], T::from_f64(2.0));

    let norm = tt.norm();
    let log_norm = tt.log_norm();

    // log_norm should equal ln(norm)
    assert!(
        (log_norm - norm.ln()).abs() < 1e-10,
        "log_norm={}, ln(norm)={}",
        log_norm,
        norm.ln()
    );
}

fn test_log_norm_with_varied_values_generic<T: TTScalar>() {
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 2);
    t0.set3(0, 0, 0, T::from_f64(1.0));
    t0.set3(0, 0, 1, T::from_f64(0.5));
    t0.set3(0, 1, 0, T::from_f64(2.0));
    t0.set3(0, 1, 1, T::from_f64(1.0));

    let mut t1: Tensor3<T> = tensor3_zeros(2, 2, 1);
    t1.set3(0, 0, 0, T::from_f64(1.0));
    t1.set3(0, 1, 0, T::from_f64(2.0));
    t1.set3(1, 0, 0, T::from_f64(0.5));
    t1.set3(1, 1, 0, T::from_f64(1.5));

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    let norm = tt.norm();
    let log_norm = tt.log_norm();

    assert!(
        (log_norm - norm.ln()).abs() < 1e-10,
        "log_norm={}, ln(norm)={}",
        log_norm,
        norm.ln()
    );
}

fn test_log_norm_zero_tensor_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::zeros(&[2, 3]);
    let log_norm = tt.log_norm();

    assert!(log_norm.is_infinite() && log_norm < 0.0);
}

// f64 tests
#[test]
fn test_tensortrain_zeros_f64() {
    test_tensortrain_zeros_generic::<f64>();
}

#[test]
fn test_tensortrain_constant_f64() {
    test_tensortrain_constant_generic::<f64>();
}

#[test]
fn test_tensortrain_constant_single_site_f64() {
    test_tensortrain_constant_single_site_generic::<f64>();
}

#[test]
fn test_tensortrain_evaluate_f64() {
    test_tensortrain_evaluate_generic::<f64>();
}

#[test]
fn test_tensortrain_scale_f64() {
    test_tensortrain_scale_generic::<f64>();
}

#[test]
fn test_tensortrain_reverse_f64() {
    test_tensortrain_reverse_generic::<f64>();
}

#[test]
fn test_fulltensor_f64() {
    test_fulltensor_generic::<f64>();
}

#[test]
fn test_fulltensor_matches_evaluate_f64() {
    test_fulltensor_matches_evaluate_generic::<f64>();
}

#[test]
fn test_log_norm_matches_norm_f64() {
    test_log_norm_matches_norm_generic::<f64>();
}

#[test]
fn test_log_norm_with_varied_values_f64() {
    test_log_norm_with_varied_values_generic::<f64>();
}

#[test]
fn test_log_norm_zero_tensor_f64() {
    test_log_norm_zero_tensor_generic::<f64>();
}

// Complex64 tests
#[test]
fn test_tensortrain_zeros_c64() {
    test_tensortrain_zeros_generic::<Complex64>();
}

#[test]
fn test_tensortrain_constant_c64() {
    test_tensortrain_constant_generic::<Complex64>();
}

#[test]
fn test_tensortrain_constant_single_site_c64() {
    test_tensortrain_constant_single_site_generic::<Complex64>();
}

#[test]
fn test_tensortrain_evaluate_c64() {
    test_tensortrain_evaluate_generic::<Complex64>();
}

#[test]
fn test_tensortrain_scale_c64() {
    test_tensortrain_scale_generic::<Complex64>();
}

#[test]
fn test_tensortrain_reverse_c64() {
    test_tensortrain_reverse_generic::<Complex64>();
}

#[test]
fn test_fulltensor_c64() {
    test_fulltensor_generic::<Complex64>();
}

#[test]
fn test_fulltensor_matches_evaluate_c64() {
    test_fulltensor_matches_evaluate_generic::<Complex64>();
}

#[test]
fn test_log_norm_matches_norm_c64() {
    test_log_norm_matches_norm_generic::<Complex64>();
}

#[test]
fn test_log_norm_with_varied_values_c64() {
    test_log_norm_with_varied_values_generic::<Complex64>();
}

#[test]
fn test_log_norm_zero_tensor_c64() {
    test_log_norm_zero_tensor_generic::<Complex64>();
}
