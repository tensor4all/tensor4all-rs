//! Reproduction: `phase_rotation_operator` loses precision for a negative `theta`.
//!
//! `phase_rotation_mpo` first maps `theta` into `[0, 2 pi)` and then multiplies it by
//! `2^(R-1-n)` (`src/phase_rotation.rs`, "Normalize theta to [0, 2 pi)"). For a negative `theta`
//! the first step computes `2 pi - |theta|`, whose rounding error (about 1e-15, independent of
//! `|theta|`) is then amplified by `2^(R-1)` at the most significant site. For
//! `theta = -2 pi / 2^R` the exact site phases are `-2 pi / 2^(n+1)`, i.e. `-pi, -pi/2, ...`,
//! which are representable to f64 roundoff without the first step.
//!
//! The tests compare the diagonal entries `<x| P_theta |x> = exp(i theta x)` of the operator
//! with the exact phase at a few `x`. They are `#[ignore]`d so CI stays green until a fix lands:
//!
//! ```text
//! cargo test --release -p tensor4all-quanticstransform --test repro_phase_rotation_negative_theta -- --ignored --nocapture
//! ```

use std::f64::consts::PI;

use num_complex::Complex64;
use tensor4all_quanticstransform::{phase_rotation_operator, QuanticsOperator};

/// `<x| op |x>`, by evaluating the MPO of `op` with both the output and the input index of every
/// site set to the bits of `x` (site 0 = most significant bit).
fn diagonal_entry(op: &QuanticsOperator, r: usize, x: u64) -> Complex64 {
    let mut indices = Vec::with_capacity(2 * r);
    let mut values = Vec::with_capacity(2 * r);
    for site in 0..r {
        let bit = ((x >> (r - 1 - site)) & 1) as usize;
        let output = &op.get_output_mapping(&site).unwrap().internal_index;
        let input = &op.get_input_mapping(&site).unwrap().internal_index;
        indices.extend([output.clone(), input.clone()]);
        values.extend([bit, bit]);
    }
    let value = op.mpo().evaluate_point(&indices, &values).unwrap();
    Complex64::new(value.real(), value.imag())
}

/// The largest `|<x| P_theta |x> - exp(2 pi i k x / 2^r)|` over a few `x`, for
/// `theta = 2 pi k / 2^r`. The exact phase is reduced in integers first, so the reference is
/// accurate to f64 roundoff for any `r`.
fn max_error(r: usize, k: i64) -> f64 {
    let n = 1i128 << r;
    let theta = 2.0 * PI * k as f64 / n as f64;
    let op = phase_rotation_operator(r, theta).unwrap();
    let xs = [1u64, 3, 1 << (r - 1), (1 << (r - 1)) + 1, (1 << r) - 1];
    xs.iter()
        .map(|&x| {
            let m = (i128::from(k) * i128::from(x)).rem_euclid(n);
            let angle = 2.0 * PI * m as f64 / n as f64;
            (diagonal_entry(&op, r, x) - Complex64::new(angle.cos(), angle.sin())).norm()
        })
        .fold(0.0, f64::max)
}

/// Control: a positive `theta = 2 pi / 2^r` is exact to roundoff at every `r`.
#[test]
fn phase_rotation_positive_theta_is_exact() {
    for r in [8, 20, 30] {
        let err = max_error(r, 1);
        println!("r = {r}, theta = +2 pi / 2^r: max |entry - exact| = {err:.1e}");
        assert!(err < 1e-13, "r = {r}: {err:e}");
    }
}

/// The same angle with the opposite sign: the error grows like `2^r` (about 1e-14 at r = 8,
/// 1e-10 at r = 20, 1e-7 at r = 30), although `exp(-i theta x) = conj(exp(i theta x))`.
#[test]
#[ignore = "reproduces the precision loss of phase_rotation_operator for a negative theta"]
fn phase_rotation_negative_theta_is_exact() {
    let errors: Vec<(usize, f64)> = [8, 20, 30].map(|r| (r, max_error(r, -1))).to_vec();
    for &(r, err) in &errors {
        println!("r = {r}, theta = -2 pi / 2^r: max |entry - exact| = {err:.1e}");
    }
    for (r, err) in errors {
        assert!(err < 1e-13, "r = {r}: {err:e}");
    }
}
