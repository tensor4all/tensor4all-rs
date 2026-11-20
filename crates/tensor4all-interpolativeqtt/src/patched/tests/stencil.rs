use super::*;

// Original dense Kronecker implementation, confined to small reference tests.
fn dense_stencil<T: InterpolativeScalar>(plan: &Plan<'_, T>) -> Matrix<T> {
    let mut weights = Vec::new();
    let tests = product_points(
        &plan.test_points_1d,
        plan.domain.ndims(),
        plan.num_test_points,
    );
    for alpha in &plan.nodes {
        for point in &tests {
            let mut value = 1.0;
            for (&index, &coordinate) in alpha.iter().zip(point) {
                value *= plan.basis.evaluate(index, coordinate).unwrap();
            }
            weights.push(T::from_f64(value));
        }
    }
    Matrix::from_col_major_vec(plan.num_test_points, plan.nodes.len(), weights)
}

fn compare_stencils<T: InterpolativeScalar>(f: impl Fn(&[f64]) -> T) {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut session = Session::new(backend.engine_mut());
    for dims in 1..=4 {
        for degree in [2, 3] {
            let options = PatchedInterpolativeOptions::default();
            let plan =
                Plan::<T>::new(&vec![0.0; dims], &vec![1.0; dims], 2, degree, &options).unwrap();
            assert_eq!(plan.weights.nrows(), 2 * (degree + 1));
            assert_eq!(plan.weights.ncols(), degree + 1);
            let values: Vec<_> = product_points(plan.basis.grid(), dims, plan.nodes.len())
                .iter()
                .map(|point| f(point))
                .collect();
            let targets: Vec<_> = product_points(&plan.test_points_1d, dims, plan.num_test_points)
                .iter()
                .map(|point| f(point))
                .collect();
            let expected = session
                .mat_mul(
                    &dense_stencil(&plan),
                    &Matrix::from_col_major_vec(values.len(), 1, values.clone()),
                )
                .unwrap()
                .into_col_major_vec();
            let actual = plan.apply_stencil(&values, &mut session).unwrap();
            let scale = values.iter().map(|x| x.abs_val()).fold(1.0_f64, f64::max);
            // Both contraction orders sum basis_dim weighted values. Bound
            // only their floating-point accumulation error, not interpolation error.
            let roundoff = 16.0 * f64::EPSILON * values.len() as f64 * scale;
            let mut dense_residual = 0.0_f64;
            let mut axis_residual = 0.0_f64;
            let mut prediction_difference = 0.0_f64;
            for ((&a, &b), &target) in actual.iter().zip(&expected).zip(&targets) {
                prediction_difference = prediction_difference.max((a - b).abs_val());
                dense_residual = dense_residual.max((b - target).abs_val());
                axis_residual = axis_residual.max((a - target).abs_val());
            }
            assert!(prediction_difference <= roundoff,
                "D={dims}, p={degree}: prediction difference {prediction_difference:e}, roundoff {roundoff:e}");
            assert!((dense_residual - axis_residual).abs() <= roundoff,
                "D={dims}, p={degree}: dense {dense_residual:e}, axes {axis_residual:e}, roundoff {roundoff:e}");
        }
    }
}

#[test]
fn axis_stencil_matches_dense_predictions_and_residuals() {
    let real = |x: &[f64]| {
        x.iter()
            .enumerate()
            .map(|(d, &x)| ((d + 1) as f64 * x).sin())
            .sum::<f64>()
    };
    compare_stencils(real);
    compare_stencils(|x| Complex64::new(real(x), (0.3 + x[0] - 2.0 * x[x.len() - 1]).exp()));
}

#[test]
fn validation_points_keep_axis_zero_fastest_order() {
    let options = PatchedInterpolativeOptions::default();
    let plan = Plan::<f64>::new(&[-2.0, 3.0, 0.5], &[2.0, 4.0, 2.0], 2, 2, &options).unwrap();
    let n = plan.test_points_1d.len();
    let mut expected = Vec::new();
    for &z in &plan.test_points_1d {
        for &y in &plan.test_points_1d {
            for &x in &plan.test_points_1d {
                expected.push(vec![-2.0 + 4.0 * x, 3.0 + y, 0.5 + 1.5 * z]);
            }
        }
    }
    let calls = std::cell::RefCell::new(Vec::new());
    serial(
        |p, _| {
            if p.shape()[1] == n.pow(3) {
                calls
                    .borrow_mut()
                    .push(p.data().chunks(3).map(<[f64]>::to_vec).collect::<Vec<_>>());
            }
            batch(p, |_| 1.0)
        },
        &[-2.0, 3.0, 0.5],
        &[6.0, 5.0, 3.5],
        2,
        2,
        &options,
    )
    .unwrap();
    // The first Chebyshev batch belongs to the lower half of each axis.
    assert_eq!(calls.borrow()[0], expected);
}

#[test]
fn higher_dimensional_degrees_run_with_default_options() {
    for (dims, degree) in [(3, 7), (4, 4)] {
        let f = |x: &[f64]| {
            1.0 + x
                .iter()
                .enumerate()
                .map(|(d, &v)| (d + 1) as f64 * v)
                .sum::<f64>()
        };
        let patches = serial(
            |p, _| batch(p, f),
            &vec![0.0; dims],
            &vec![1.0; dims],
            2,
            degree,
            &PatchedInterpolativeOptions::default(),
        )
        .unwrap();
        assert_eq!(patches.len(), 1);
        check_partition(&patches, dims, 2, f, 1e-8);
    }
}

// Original point construction is test-only, to verify coordinate ordering.
fn product_points(grid: &[f64], ndims: usize, count: usize) -> Vec<Vec<f64>> {
    product_indices(grid.len(), ndims, count)
        .into_iter()
        .map(|point| point.into_iter().map(|i| grid[i]).collect())
        .collect()
}
