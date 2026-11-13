//! Complex-valued interpolants (tensor4all-rs#1276, step 0).

use num_complex::Complex64;
use tensor4all_tensorbackend::Matrix;

use super::*;

/// The retarded propagator `1 / (omega + i delta - epsilon)`, the function the
/// patched complex interpolation of #1276 is for.
fn retarded(omega: f64, delta: f64, epsilon: f64) -> Complex64 {
    Complex64::new(1.0, 0.0) / Complex64::new(omega - epsilon, delta)
}

/// Asserts that `complex` is `real` with every value lifted to `Complex64`:
/// the same core shapes, real parts bit for bit, imaginary parts zero.
fn assert_lifted_exactly(real: &SimpleTensorTrain<f64>, complex: &SimpleTensorTrain<Complex64>) {
    assert_eq!(real.len(), complex.len());
    for (site, (r, c)) in real
        .site_tensors()
        .iter()
        .zip(complex.site_tensors())
        .enumerate()
    {
        let shape = [r.left_dim(), r.site_dim(), r.right_dim()];
        assert_eq!(
            shape,
            [c.left_dim(), c.site_dim(), c.right_dim()],
            "site {site}"
        );
        for left in 0..shape[0] {
            for s in 0..shape[1] {
                for right in 0..shape[2] {
                    let (x, z) = (*r.get3(left, s, right), *c.get3(left, s, right));
                    assert_eq!(
                        z.re.to_bits(),
                        x.to_bits(),
                        "site {site} [{left},{s},{right}]"
                    );
                    assert_eq!(z.im, 0.0, "site {site} [{left},{s},{right}]");
                }
            }
        }
    }
}

/// The largest entrywise distance between two matrices of the same shape.
fn max_distance(a: &Matrix<Complex64>, b: &Matrix<Complex64>) -> f64 {
    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
    let mut max = 0.0_f64;
    for row in 0..a.nrows() {
        for col in 0..a.ncols() {
            max = max.max((a[[row, col]] - b[[row, col]]).norm());
        }
    }
    max
}

fn lift_matrix(m: &Matrix<f64>) -> Matrix<Complex64> {
    let mut out = Matrix::zeros(m.nrows(), m.ncols());
    for row in 0..m.nrows() {
        for col in 0..m.ncols() {
            out[[row, col]] = Complex64::new(m[[row, col]], 0.0);
        }
    }
    out
}

#[test]
fn complex_single_scale_reproduces_the_retarded_propagator_1d() {
    let (r, degree, a, b) = (6, 30, -1.0, 1.0);
    let f = |x: f64| retarded(0.3, 1.0, x * x);

    let tt =
        interpolate_single_scale(f, a, b, r, degree, &InterpolativeQttOptions::default()).unwrap();

    assert_tt_matches_grid_1d(&tt, &grid_1d(r, a, b), f, 1e-9);
}

#[test]
fn complex_single_scale_reproduces_the_retarded_propagator_2d() {
    let (r, degree) = (4, 32);
    let (lower, upper) = ([-1.0, -1.0], [1.0, 1.0]);
    let pi = std::f64::consts::PI;
    let f = |x: &[f64]| retarded(0.5, 1.0, 0.5 * ((pi * x[0]).cos() + (pi * x[1]).cos()));

    let tt = interpolate_single_scale_nd(
        f,
        &lower,
        &upper,
        r,
        degree,
        &InterpolativeQttOptions::default(),
    )
    .unwrap();

    assert_eq!(tt.site_dims(), vec![4; r]);
    assert_tt_matches_grid_nd(&tt, &grid_nd(r, &lower, &upper), f, 1e-10);
}

#[test]
fn complex_adaptive_refines_around_the_quasiparticle_peak() {
    // A broadening of 0.02 puts the poles of 1 / (omega + i delta - x^2) 0.018
    // from the real axis at x = +-0.55, which a degree-12 single-scale
    // interpolant does not resolve; the adaptive construction has to refine
    // there, through the complex error estimate.
    let (r, degree, a, b) = (8, 12, -1.0, 1.0);
    let f = |x: f64| retarded(0.3, 0.02, x * x);
    let options = InterpolativeQttOptions::default().with_tolerance(0.0);

    let adaptive = interpolate_adaptive(f, a, b, r, degree, 1e-8, &[], &options).unwrap();

    // Refinement shows in the bonds: a refined core carries `degree + 1` basis
    // functions plus one bond per refined interval.
    assert!(adaptive.rank() > degree + 1, "rank {}", adaptive.rank());
    assert_tt_matches_grid_1d(&adaptive, &grid_1d(r, a, b), f, 1e-7);

    let single = interpolate_single_scale(f, a, b, r, degree, &options).unwrap();
    let grid = grid_1d(r, a, b);
    let single_error = (0..1usize << r)
        .map(|i| {
            let q = grid.grididx_to_quantics(&[i]).unwrap();
            let x = grid.grididx_to_origcoord(&[i]).unwrap()[0];
            (single.evaluate(&q).unwrap() - f(x)).norm()
        })
        .fold(0.0, f64::max);
    assert!(single_error > 1e-4, "single-scale error {single_error}");
}

#[test]
fn complex_multi_scale_and_sparse_reproduce_the_retarded_propagator() {
    let (r, degree, a, b) = (6, 24, -1.0, 1.0);
    let f = |x: f64| retarded(0.3, 1.0, x * x);
    let options = InterpolativeQttOptions::default();
    let grid = grid_1d(r, a, b);

    let multi = interpolate_multi_scale(f, a, b, r, degree, &[0.0], &options).unwrap();
    assert_tt_matches_grid_1d(&multi, &grid, f, 1e-8);

    let sparse = interpolate_single_scale_sparse(f, a, b, r, degree, 6, &options).unwrap();
    assert_tt_matches_grid_1d(&sparse, &grid, f, 1e-6);
}

#[test]
fn zero_imaginary_part_reproduces_the_real_construction_bit_for_bit() {
    // Uncompressed, so the construction is the only arithmetic: lifting the
    // values to Complex64 must not change one bit of the real parts.
    let options = InterpolativeQttOptions::default().with_tolerance(0.0);
    let f = |x: f64| (3.0 * x).sin() / (1.0 + (x - 0.3).abs());
    let c = |x: f64| Complex64::new(f(x), 0.0);
    let g = |x: &[f64]| (x[0] - x[1]).cos() * (x[0] + 0.5).abs();
    let cg = |x: &[f64]| Complex64::new(g(x), 0.0);
    let (lower, upper) = ([-1.0, 0.0], [1.0, 1.0]);

    assert_lifted_exactly(
        &interpolate_single_scale(f, -1.0, 1.0, 5, 8, &options).unwrap(),
        &interpolate_single_scale(c, -1.0, 1.0, 5, 8, &options).unwrap(),
    );
    assert_lifted_exactly(
        &interpolate_single_scale_nd(g, &lower, &upper, 3, 5, &options).unwrap(),
        &interpolate_single_scale_nd(cg, &lower, &upper, 3, 5, &options).unwrap(),
    );
    assert_lifted_exactly(
        &interpolate_multi_scale(f, -1.0, 1.0, 5, 6, &[0.3], &options).unwrap(),
        &interpolate_multi_scale(c, -1.0, 1.0, 5, 6, &[0.3], &options).unwrap(),
    );
    let cusp = [vec![-0.5, 0.25]];
    assert_lifted_exactly(
        &interpolate_multi_scale_nd(g, &lower, &upper, 4, 4, &cusp, &options).unwrap(),
        &interpolate_multi_scale_nd(cg, &lower, &upper, 4, 4, &cusp, &options).unwrap(),
    );
    let real_adaptive = interpolate_adaptive(f, -1.0, 1.0, 6, 6, 1e-6, &[], &options).unwrap();
    assert!(real_adaptive.rank() > 7, "the cusp at 0.3 must be refined");
    assert_lifted_exactly(
        &real_adaptive,
        &interpolate_adaptive(c, -1.0, 1.0, 6, 6, 1e-6, &[], &options).unwrap(),
    );
    assert_lifted_exactly(
        &interpolate_adaptive_nd(g, &lower, &upper, 4, 4, 1e-6, &[], &options).unwrap(),
        &interpolate_adaptive_nd(cg, &lower, &upper, 4, 4, 1e-6, &[], &options).unwrap(),
    );
    assert_lifted_exactly(
        &interpolate_single_scale_sparse(f, -1.0, 1.0, 5, 8, 2, &options).unwrap(),
        &interpolate_single_scale_sparse(c, -1.0, 1.0, 5, 8, 2, &options).unwrap(),
    );
    assert_lifted_exactly(
        &interpolate_single_scale_sparse_nd(g, &lower, &upper, 3, 5, 2, &options).unwrap(),
        &interpolate_single_scale_sparse_nd(cg, &lower, &upper, 3, 5, 2, &options).unwrap(),
    );

    let basis = get_chebyshev_grid(6).unwrap();
    assert_eq!(
        estimate_interpolation_error(f, -1.0, 1.0, &basis).unwrap(),
        estimate_interpolation_error(c, -1.0, 1.0, &basis).unwrap(),
    );
    assert_eq!(
        estimate_interpolation_error_nd(g, &lower, &upper, &basis).unwrap(),
        estimate_interpolation_error_nd(cg, &lower, &upper, &basis).unwrap(),
    );
}

#[test]
fn zero_imaginary_part_agrees_with_the_real_path_after_compression() {
    // The SVD of a complex train is a different routine, so after compression
    // the two agree to roundoff, not bit for bit.
    let (r, degree, a, b) = (6, 16, -1.0, 1.0);
    let f = |x: f64| (-4.0 * x * x).exp() * (5.0 * x).cos();
    let options = InterpolativeQttOptions::default();

    let real = interpolate_single_scale(f, a, b, r, degree, &options).unwrap();
    let complex =
        interpolate_single_scale(|x| Complex64::new(f(x), 0.0), a, b, r, degree, &options).unwrap();

    assert_eq!(real.link_dims(), complex.link_dims());
    let grid = grid_1d(r, a, b);
    for i in 0..1usize << r {
        let q = grid.grididx_to_quantics(&[i]).unwrap();
        let (x, z) = (real.evaluate(&q).unwrap(), complex.evaluate(&q).unwrap());
        assert!(
            (z - Complex64::new(x, 0.0)).norm() < 1e-13,
            "grid index {i}: {x} vs {z}"
        );
    }
}

#[test]
fn complex_invert_qtt_is_the_sum_of_the_real_and_imaginary_inversions() {
    // Without truncation the construction and `invert_qtt` are both linear in
    // the function values, so the train of `g + i h` must invert to the sum of
    // the inversions of `g` and `h`. (A truncation is not linear, so the
    // compressed route is checked by its round trip below instead.)
    let (r, degree, a, b) = (7, 8, 0.0, 1.0);
    let g = |x: f64| (-x * x).exp();
    let h = |x: f64| (2.0 * x).sin();
    let gh = |x: f64| Complex64::new(g(x), h(x));
    let basis = get_chebyshev_grid(degree).unwrap();

    {
        let options = InterpolativeQttOptions::default().with_tolerance(0.0);
        let real = invert_qtt(
            &interpolate_single_scale(g, a, b, r, degree, &options).unwrap(),
            &basis,
            1,
        )
        .unwrap();
        let imag = invert_qtt(
            &interpolate_single_scale(h, a, b, r, degree, &options).unwrap(),
            &basis,
            1,
        )
        .unwrap();
        let complex_tt = interpolate_single_scale(gh, a, b, r, degree, &options).unwrap();
        let complex = invert_qtt(&complex_tt, &basis, 1).unwrap();

        // The exact route is decided on the complex train itself.
        assert_eq!(
            finest_level_scale(&complex_tt, &basis).unwrap(),
            Some(Complex64::new(1.0, 0.0))
        );
        assert_eq!(complex.len(), r - 1);
        for level in 0..r - 1 {
            let mut expected = lift_matrix(&real[level]);
            for row in 0..expected.nrows() {
                for col in 0..expected.ncols() {
                    expected[[row, col]] += Complex64::new(0.0, imag[level][[row, col]]);
                }
            }
            let distance = max_distance(&complex[level], &expected);
            assert!(distance < 1e-12, "level {level}: {distance}");
        }
    }
}

#[test]
fn complex_invert_qtt_matches_the_real_path_on_a_zero_imaginary_train() {
    let (r, degree, a, b) = (7, 8, 0.0, 1.0);
    let g = |x: f64| (-x * x).exp();
    let basis = get_chebyshev_grid(degree).unwrap();
    let options = InterpolativeQttOptions::default().with_tolerance(0.0);

    let real = invert_qtt(
        &interpolate_single_scale(g, a, b, r, degree, &options).unwrap(),
        &basis,
        1,
    )
    .unwrap();
    let complex = invert_qtt(
        &interpolate_single_scale(|x| Complex64::new(g(x), 0.0), a, b, r, degree, &options)
            .unwrap(),
        &basis,
        1,
    )
    .unwrap();

    for level in 0..r - 1 {
        let distance = max_distance(&complex[level], &lift_matrix(&real[level]));
        assert!(distance < 1e-14, "level {level}: {distance}");
    }
}

#[test]
fn complex_invert_qtt_round_trips_on_the_exact_and_blended_routes() {
    // The bounds of the real round-trip tests, on a complex function.
    let f = |x: f64| Complex64::new((-x * x).exp(), (2.0 * x).sin());
    let exact = invert_round_trip_error(
        f,
        8,
        10,
        &InterpolativeQttOptions::default().with_tolerance(0.0),
    );
    assert!(exact < 1.0e-11, "exact route: {exact}");
    let blended = invert_round_trip_error(
        f,
        10,
        10,
        &InterpolativeQttOptions::default().with_tolerance(1e-8),
    );
    assert!(blended < 1.0e-6, "blended route: {blended}");
}
