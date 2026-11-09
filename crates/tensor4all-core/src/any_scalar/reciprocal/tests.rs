use num_complex::Complex64;

use super::*;

fn pow2(k: i32) -> f64 {
    pow2_factors(k).product()
}

fn divide(value: AnyScalar, divisor: &AnyScalar) -> AnyScalar {
    divisor
        .reciprocal_factors()
        .unwrap()
        .into_iter()
        .fold(value, |acc, factor| acc * factor)
}

fn divide_real(value: f64, divisor: f64) -> f64 {
    divide(AnyScalar::new_real(value), &AnyScalar::new_real(divisor)).real()
}

fn divide_complex(value: Complex64, divisor: Complex64) -> Complex64 {
    let quotient = divide(
        AnyScalar::new_complex(value.re, value.im),
        &AnyScalar::new_complex(divisor.re, divisor.im),
    );
    Complex64::new(quotient.real(), quotient.imag())
}

#[test]
fn reciprocal_factors_reject_zero_and_non_finite_divisors() {
    for divisor in [
        AnyScalar::new_real(0.0),
        AnyScalar::new_real(f64::NAN),
        AnyScalar::new_real(f64::INFINITY),
        AnyScalar::new_complex(1.0, f64::NAN),
        AnyScalar::new_complex(0.0, 0.0),
    ] {
        let error = divisor.reciprocal_factors().unwrap_err();
        assert!(error.to_string().contains("cannot divide by"), "{error}");
    }
}

#[test]
fn reciprocal_factors_match_the_direct_reciprocal_for_a_real_divisor() {
    // Scaling value and divisor by the same 2^s leaves fl(x * fl(1/d))
    // unchanged while it stays normal, so every path must give the bits of
    // the direct reciprocal multiplication.
    for (x, d) in [
        (1.0, 7.0),
        (5.5, 7.0),
        (1.0, 0.1),
        (-2.75, 1.7),
        (1.0, 1.0 / 3.0),
    ] {
        let direct: f64 = x * (1.0 / d);
        assert_eq!(divide_real(x, d).to_bits(), direct.to_bits());
        for s in [-1000, -600, 600, 1000] {
            let actual = divide_real(x * pow2(s), d * pow2(s));
            assert_eq!(actual.to_bits(), direct.to_bits(), "{x} / {d} at 2^{s}");
        }
    }
}

#[test]
fn reciprocal_factors_round_a_subnormal_quotient_once() {
    // from_bits(2) / 3 is 2/3 of the smallest subnormal, which rounds to it.
    for k in [0, 600] {
        let actual = divide_real(f64::from_bits(2) * pow2(k), 3.0 * pow2(k));
        assert_eq!(actual.to_bits(), 1, "2^{k}: {actual:e}");
    }
    // x / d lies just above 2^-1075, half the smallest subnormal, and `d` is
    // about 2^600. Rounding the product before scaling it down would land
    // exactly on 2^-1075 and round to zero.
    let x = f64::from_bits(0x224d_82c0_7cc5_3ee0);
    let d = f64::from_bits(0x657d_82c0_7cc5_3edf);
    assert_eq!((x / d).to_bits(), 1);
    assert_eq!(divide_real(x, d).to_bits(), 1);
}

#[test]
fn reciprocal_factors_divide_by_a_tiny_or_huge_divisor() {
    // (value, divisor, expected quotient); 1 / divisor or |divisor|^2 leaves
    // the f64 range in each case.
    let cases = [
        (
            Complex64::new(pow2(-1000), 0.0),
            Complex64::new(3.0 * pow2(-1070), 0.0),
            Complex64::new(pow2(70) / 3.0, 0.0),
        ),
        (
            Complex64::new(pow2(-1000), 0.0),
            Complex64::new(3.0, 4.0) * pow2(-1070),
            Complex64::new(3.0, -4.0) * (pow2(70) / 25.0),
        ),
        (
            Complex64::new(pow2(1000), 0.0),
            Complex64::new(3.0, 4.0) * pow2(1018),
            Complex64::new(3.0, -4.0) * (pow2(-18) / 25.0),
        ),
        (
            Complex64::new(f64::MAX / 4.0, 0.0),
            Complex64::new(f64::MAX, 0.0),
            Complex64::new(0.25, 0.0),
        ),
    ];
    for (value, divisor, expected) in cases {
        let actual = divide_complex(value, divisor);
        for (a, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
            assert!(a.is_finite(), "{value} / {divisor}: {actual}");
            assert!(
                (a - e).abs() <= 4.0 * f64::EPSILON * e.abs(),
                "{value} / {divisor}: {actual}"
            );
        }
    }
}

#[test]
fn reciprocal_factors_keep_a_complex_component_the_reciprocal_would_lose() {
    // The imaginary part of 1 / divisor is about -2^-1200 or -2^-1400, below
    // the smallest subnormal, but that of value / divisor is normal. The
    // second divisor is inside the direct-reciprocal range of a real divisor.
    for (scale, small, expected_im) in [(600, -420, -1020), (400, -600, -1000)] {
        let actual = divide_complex(
            Complex64::new(pow2(scale), 0.0),
            Complex64::new(pow2(scale), pow2(small)),
        );
        assert_eq!(actual.re, 1.0);
        assert_eq!(actual.im, -pow2(expected_im), "2^{scale}: {actual}");
    }
}

#[test]
fn reciprocal_factors_keep_finite_components_of_a_rotated_quotient() {
    // |quotient| exceeds f64::MAX but both components are finite, so no
    // intermediate may grow the value past f64::MAX before the rotation.
    let value = Complex64::new(0.51 * f64::MAX, 0.0);
    let divisor = Complex64::new(0.35, 0.35);
    let expected = value * (Complex64::new(1.0, 0.0) / divisor);
    assert!(expected.re.is_finite() && expected.im.is_finite());
    assert!(expected.norm().is_infinite());
    for k in [0, 600] {
        let actual = divide_complex(value * pow2(-k), divisor * pow2(-k));
        for (a, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
            assert!(a.is_finite(), "2^-{k}: {actual}");
            assert!(
                (a - e).abs() <= 4.0 * f64::EPSILON * e.abs(),
                "2^-{k}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn reciprocal_factors_keep_finite_components_of_a_complex_value() {
    // A component of `x * w` can reach `|x| |w|`, about `1.4 * f64::MAX` for
    // `x = f64::MAX * (1 + i)` and `|w|` near one, although every component of
    // `x / d` is finite. `1.9 + 2^-1074 i` takes the unscaled-conjugate path.
    let value = Complex64::new(f64::MAX, f64::MAX);
    for divisor in [
        Complex64::new(1.9, 0.5),
        Complex64::new(1.9, f64::from_bits(1)),
        Complex64::new(1.9, 0.5) * pow2(150),
        Complex64::new(1.9, 0.5) * pow2(300),
    ] {
        let expected = value * (Complex64::new(1.0, 0.0) / divisor);
        assert!(expected.re.is_finite() && expected.im.is_finite());
        let actual = divide_complex(value, divisor);
        for (a, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
            assert!(
                (a - e).abs() <= 4.0 * f64::EPSILON * e.abs(),
                "{divisor}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn reciprocal_factors_track_the_divisor_gradient() {
    // d(x / d)/dd = -x / d^2 = -2^-3 * 2^-s for x = 2 * 2^s, d = 4 * 2^s,
    // on both the direct and the scaled path.
    for s in [0, 600] {
        let d = AnyScalar::new_real(4.0 * pow2(s)).enable_grad().unwrap();
        let quotient = divide(AnyScalar::new_real(2.0 * pow2(s)), &d);
        assert_eq!(quotient.real(), 0.5);
        assert!(quotient.tracks_grad());
        quotient.backward().unwrap();
        let grad = d.grad().unwrap().unwrap().real();
        let expected = -0.125 * pow2(-s);
        assert!((grad / expected - 1.0).abs() < 1e-14, "2^{s}: {grad:e}");
    }
}

#[test]
fn reciprocal_factors_divide_by_an_imaginary_divisor_exactly() {
    // Multiplying by -i is exact, so the real-divisor guarantees carry over.
    assert_eq!(
        divide_complex(
            Complex64::new(f64::from_bits(1), 0.0),
            Complex64::new(0.0, 1.0)
        ),
        Complex64::new(0.0, -f64::from_bits(1))
    );
    let x = f64::from_bits(0x224d_82c0_7cc5_3ee0);
    let d = f64::from_bits(0x657d_82c0_7cc5_3edf);
    let actual = divide_complex(Complex64::new(0.0, x), Complex64::new(0.0, d));
    assert_eq!((actual.re.to_bits(), actual.im), (1, 0.0));
}

#[test]
fn reciprocal_factors_bound_the_subnormal_error_of_a_complex_quotient() {
    // Each component is within `bound` units of max(2^-1074, ulp(q)) of the
    // quotient q. The reference q * 2^600 is computed with d scaled near one.
    let unit_of = |q: f64| f64::from_bits(1).max(q.abs() * f64::EPSILON);
    let divisors = [
        (Complex64::new(0.0, 1.5), 0.5),
        (Complex64::new(0.0, -0.3), 0.5),
        (Complex64::new(0.35, 0.35), 9.0),
        (Complex64::new(0.2, 0.1), 9.0),
        (Complex64::new(0.03, -0.01), 9.0),
        (Complex64::new(3.0, 4.0) * pow2(-600), 9.0),
        (Complex64::new(0.45, 0.25), 7.0),
        (Complex64::new(0.6, 0.8), 7.0),
        (Complex64::new(1.0, 1e-3), 7.0),
        (Complex64::new(1.9, -0.3), 7.0),
        (Complex64::new(3.0, 4.0), 7.0),
        (Complex64::new(3.0, 4.0) * pow2(150), 7.0),
        (Complex64::new(3.0, 4.0) * pow2(300), 7.0),
    ];
    for (d, bound) in divisors {
        let e = -(d.norm().log2().round() as i32);
        for bits in 1..1000_u64 {
            for x in [f64::from_bits(bits) * d.norm(), f64::from_bits(bits)] {
                let scaled = Complex64::new(x * pow2(300) * pow2(300 + e), 0.0) / (d * pow2(e));
                let q = scaled / pow2(600);
                let actual = divide_complex(Complex64::new(x, 0.0), d);
                for (a, e) in [(actual.re, q.re), (actual.im, q.im)] {
                    let error = (a - e).abs() / unit_of(e);
                    assert!(
                        error <= bound,
                        "{x:e} / {d}: {actual} vs {q} ({error} units)"
                    );
                }
            }
        }
    }
}

#[test]
fn reciprocal_factors_keep_a_component_that_scaling_the_divisor_would_flush() {
    // Scaling 1 + 2^-1074 i down flushes its imaginary part, but that of
    // 2^1023 / (1 + 2^-1074 i) is -2^-51. In the second case the large
    // target scales 2^-900 to 2^-1136, and the quotient's part is -2^-477.
    for (value, divisor, expected) in [
        (
            pow2(1023),
            Complex64::new(1.0, f64::from_bits(1)),
            Complex64::new(pow2(1023), -pow2(-51)),
        ),
        (
            pow2(1023),
            Complex64::new(pow2(300), pow2(-900)),
            Complex64::new(pow2(723), -pow2(-477)),
        ),
    ] {
        let actual = divide_complex(Complex64::new(value, 0.0), divisor);
        assert_eq!(actual, expected, "{value:e} / {divisor}");
    }
}

#[test]
fn reciprocal_factors_track_both_components_of_a_complex_divisor() {
    // The gradient of Re(1 / d) through the factors matches that of the
    // direct reciprocal on every path, including a purely imaginary divisor.
    for divisor in [
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, 2.0),
        Complex64::new(0.1, 0.2),
        Complex64::new(1.0, 2.0) * pow2(300),
    ] {
        let gradient = |via_factors: bool| {
            let d = AnyScalar::new_complex(divisor.re, divisor.im)
                .enable_grad()
                .unwrap();
            let quotient = if via_factors {
                divide(AnyScalar::new_real(1.0), &d)
            } else {
                AnyScalar::new_real(1.0) / d.clone()
            };
            quotient.real_part().backward().unwrap();
            let grad = d.grad().unwrap().unwrap();
            Complex64::new(grad.real(), grad.imag())
        };
        let (actual, expected) = (gradient(true), gradient(false));
        assert!(
            (actual - expected).norm() <= 1e-12 * expected.norm(),
            "{divisor}: {actual} vs {expected}"
        );
    }
}
