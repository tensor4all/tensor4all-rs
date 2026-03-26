#![allow(dead_code)]

use num_complex::Complex64;
use std::fmt::Debug;

pub fn assert_real_samples_close<P, F>(samples: &[(P, f64)], expected: &F, tol: f64)
where
    P: AsRef<[usize]> + Debug,
    F: Fn(&[usize]) -> f64,
{
    let max_sample = samples
        .iter()
        .map(|(point, _)| expected(point.as_ref()).abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let (worst_point, max_diff) = samples.iter().fold((None, 0.0_f64), |acc, (point, got)| {
        let diff = (*got - expected(point.as_ref())).abs();
        if diff > acc.1 {
            (Some(format!("{:?}", point)), diff)
        } else {
            acc
        }
    });
    assert!(
        max_diff <= tol * max_sample,
        "maxabs diff {} at {:?} exceeds tol {} * max_sample {}",
        max_diff,
        worst_point,
        tol,
        max_sample
    );
}

pub fn assert_complex_samples_close<P, F>(samples: &[(P, Complex64)], expected: &F, tol: f64)
where
    P: AsRef<[usize]> + Debug,
    F: Fn(&[usize]) -> Complex64,
{
    let max_sample = samples
        .iter()
        .map(|(point, _)| expected(point.as_ref()).norm())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let (worst_point, max_diff) = samples.iter().fold((None, 0.0_f64), |acc, (point, got)| {
        let diff = (*got - expected(point.as_ref())).norm();
        if diff > acc.1 {
            (Some(format!("{:?}", point)), diff)
        } else {
            acc
        }
    });
    assert!(
        max_diff <= tol * max_sample,
        "maxabs diff {} at {:?} exceeds tol {} * max_sample {}",
        max_diff,
        worst_point,
        tol,
        max_sample
    );
}
