use super::*;

#[test]
fn test_integrate_rejects_invalid_options_before_callback() {
    let calls = std::cell::Cell::new(0);
    let f = |_x: &[f64]| {
        calls.set(calls.get() + 1);
        1.0_f64
    };
    let error = integrate::<f64, _>(
        &f,
        &[0.0, 0.0],
        &[1.0, 1.0],
        15,
        TCI2Options {
            ncheck_history: 0,
            ..TCI2Options::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, TCIError::InvalidConfiguration { .. }));
    assert_eq!(calls.get(), 0);
}

#[test]
fn test_integrate_constant() {
    // Integral of 1.0 over [0, 1]^2 = 1.0
    let f = |_x: &[f64]| 1.0f64;
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 1.0];
    let options = TCI2Options {
        tolerance: 1e-10,
        ..TCI2Options::default()
    };

    let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
    assert!((result - 1.0).abs() < 1e-8, "Expected 1.0, got {}", result);
}

#[test]
fn test_integrate_polynomial() {
    // Integral of x*y over [0, 1]^2 = 1/4
    let f = |x: &[f64]| x[0] * x[1];
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 1.0];
    let options = TCI2Options {
        tolerance: 1e-10,
        ..TCI2Options::default()
    };

    let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
    assert!(
        (result - 0.25).abs() < 1e-8,
        "Expected 0.25, got {}",
        result
    );
}

#[test]
fn test_integrate_5d_polynomial() {
    // Integral of product of polynomials over [0, 1]^5
    // p(x) = 1 + x + x^2, integral = 1 + 1/2 + 1/3 = 11/6
    // 5D integral = (11/6)^5
    let polynomial = |x: f64| 1.0 + x + x * x;
    let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<f64>();

    let n = 5;
    let a = vec![0.0; n];
    let b = vec![1.0; n];
    let options = TCI2Options {
        tolerance: 1e-10,
        ..TCI2Options::default()
    };

    let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
    let expected = (11.0 / 6.0_f64).powi(5);
    assert!(
        (result - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        result
    );
}

#[test]
fn test_integrate_with_gk31() {
    // Same polynomial test with GK31 — should be more accurate
    let polynomial = |x: f64| 1.0 + x + x * x;
    let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<f64>();

    let n = 3;
    let a = vec![0.0; n];
    let b = vec![1.0; n];
    let options = TCI2Options {
        tolerance: 1e-12,
        ..TCI2Options::default()
    };

    let result: f64 = integrate(&f, &a, &b, 31, options).unwrap();
    let expected = (11.0 / 6.0_f64).powi(3);
    assert!(
        (result - expected).abs() < 1e-10,
        "GK31: Expected {}, got {}, diff={}",
        expected,
        result,
        (result - expected).abs()
    );
}

#[test]
fn test_gk_order_unsupported() {
    let f = |_x: &[f64]| 1.0f64;
    let a = vec![0.0];
    let b = vec![1.0];
    let options = TCI2Options::default();
    let result: std::result::Result<f64, _> = integrate(&f, &a, &b, 7, options);
    assert!(result.is_err());
}

#[test]
fn test_gk_supported_orders_are_reported() {
    assert!(gk_nodes_weights(15).is_ok());
    assert!(gk_nodes_weights(31).is_ok());
    assert!(gk_nodes_weights(41).is_ok());
    assert!(gk_nodes_weights(51).is_ok());
    assert!(gk_nodes_weights(61).is_ok());
}

#[test]
fn test_gk_order_error_lists_supported_orders() {
    let err = gk_nodes_weights(21).unwrap_err();
    let message = err.to_string();

    assert!(message.contains("21"));
    assert!(message.contains("15, 31, 41, 51, 61"));
}

#[test]
fn test_gk_weights_sum_to_two() {
    for order in [15, 31, 41, 51, 61] {
        let (_nodes, weights) = gk_nodes_weights(order).unwrap();
        let total: f64 = weights.iter().sum();
        assert!(
            (total - 2.0).abs() < 1e-14,
            "GK{order} weights sum to {total}"
        );
    }
}

#[test]
fn test_gk_nodes_are_symmetric_and_sorted() {
    for order in [15, 31, 41, 51, 61] {
        let (nodes, weights) = gk_nodes_weights(order).unwrap();
        assert_eq!(nodes.len(), order);
        assert_eq!(weights.len(), order);
        for pair in nodes.windows(2) {
            assert!(pair[0] < pair[1], "GK{order} nodes are not sorted");
        }
        for i in 0..nodes.len() {
            let j = nodes.len() - 1 - i;
            assert!((nodes[i] + nodes[j]).abs() < 1e-14);
            assert!((weights[i] - weights[j]).abs() < 1e-14);
        }
    }
}

#[test]
fn test_integrate_polynomial_with_gk61() {
    let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let options = TCI2Options {
        tolerance: 1e-10,
        max_iter: 20,
        seed: Some(42),
        ..TCI2Options::default()
    };

    let result: f64 = integrate(&f, &[0.0, 0.0], &[1.0, 1.0], 61, options).unwrap();

    assert!((result - 2.0 / 3.0).abs() < 1e-8);
}

#[test]
fn test_integrate_one_dimensional_reports_tci_requirement() {
    let f = |x: &[f64]| x[0] * x[0];
    let err = integrate::<f64, _>(&f, &[0.0], &[1.0], 15, TCI2Options::default()).unwrap_err();

    assert!(err
        .to_string()
        .contains("local_dims should have at least 2 elements"));
}

/// Port of Julia test_integration.jl: "Integrate real polynomials" with arbitrary bounds
#[test]
fn test_integrate_arbitrary_bounds() {
    // p(x) = 1 + 0.5*x + 0.3*x^2
    // integral of p from a to b = [x + 0.25*x^2 + 0.1*x^3]_a^b
    let polynomial = |x: f64| 1.0 + 0.5 * x + 0.3 * x * x;
    let poly_integral = |x: f64| x + 0.25 * x * x + 0.1 * x * x * x;
    let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<f64>();

    let n = 3;
    let a = vec![0.2, 0.5, 0.1];
    let b = vec![0.8, 0.9, 0.7];
    let expected: f64 = (0..n)
        .map(|i| poly_integral(b[i]) - poly_integral(a[i]))
        .product();

    let options = TCI2Options {
        tolerance: 1e-10,
        ..TCI2Options::default()
    };
    let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
    assert!(
        (result - expected).abs() < 1e-8,
        "Arbitrary bounds: expected {expected}, got {result}"
    );
}

/// Port of Julia test_integration.jl: "Integrate complex polynomials"
#[test]
fn test_integrate_complex_polynomial() {
    use num_complex::Complex64;

    let polynomial = |x: f64| Complex64::new(1.0 + x, 0.5 * x * x);
    let poly_integral = |x: f64| Complex64::new(x + 0.5 * x * x, 0.5 * x * x * x / 3.0);
    let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<Complex64>();

    let n = 3;
    let a = vec![0.0; n];
    let b = vec![1.0; n];
    let expected: Complex64 = (0..n)
        .map(|_| poly_integral(1.0) - poly_integral(0.0))
        .product();

    let options = TCI2Options {
        tolerance: 1e-10,
        ..TCI2Options::default()
    };
    let result: Complex64 = integrate(&f, &a, &b, 15, options).unwrap();
    assert!(
        (result - expected).norm() < 1e-8,
        "Complex: expected {expected}, got {result}"
    );
}
