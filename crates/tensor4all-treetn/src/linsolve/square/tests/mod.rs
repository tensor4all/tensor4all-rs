use super::*;
use tensor4all_core::{DynIndex, IdxTensor, TensorIndex};

fn create_simple_2site_mps() -> TreeTN<IdxTensor, String> {
    let mut mps = TreeTN::<IdxTensor, String>::new();
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 4]).unwrap();

    let n0 = mps.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = mps.add_tensor("site1".to_string(), t1).unwrap();
    mps.connect(n0, &b01, n1, &b01).unwrap();

    mps
}

fn create_simple_2site_mpo() -> TreeTN<IdxTensor, String> {
    let mut mpo = TreeTN::<IdxTensor, String>::new();
    let s0_in = DynIndex::new_dyn(2);
    let s0_out = DynIndex::new_dyn(2);
    let s1_in = DynIndex::new_dyn(2);
    let s1_out = DynIndex::new_dyn(2);
    let b_mpo = DynIndex::new_dyn(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0]; // Identity matrix
    let t0_mpo = IdxTensor::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        IdxTensor::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data).unwrap();

    let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
    let n1_mpo = mpo.add_tensor("site1".to_string(), t1_mpo).unwrap();
    mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

    mpo
}

#[test]
fn test_validate_linsolve_inputs_success() {
    let operator = create_simple_2site_mpo();
    let rhs = create_simple_2site_mps();
    let init = create_simple_2site_mps();

    // Should succeed for compatible inputs
    let result = validate_linsolve_inputs(&operator, &rhs, &init);
    assert!(result.is_ok());
}

#[test]
fn test_validate_linsolve_inputs_incompatible_dimensions() {
    let operator = create_simple_2site_mpo();
    let rhs = create_simple_2site_mps();

    // Create init with different dimensions
    let mut init = TreeTN::<IdxTensor, String>::new();
    let s0 = DynIndex::new_dyn(3); // Different dimension
    let s1 = DynIndex::new_dyn(3);
    let b01 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 6]).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 6]).unwrap();

    let n0 = init.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = init.add_tensor("site1".to_string(), t1).unwrap();
    init.connect(n0, &b01, n1, &b01).unwrap();

    // Should fail due to incompatible dimensions
    let result = validate_linsolve_inputs(&operator, &rhs, &init);
    assert!(result.is_err());
}

#[test]
fn test_square_linsolve_zero_sweeps_returns_solution_wrapper() {
    let operator = create_simple_2site_mpo();
    let rhs = create_simple_2site_mps();
    let init = create_simple_2site_mps();

    let result = square_linsolve(
        &operator,
        &rhs,
        init,
        &"site0".to_string(),
        LinsolveOptions::new(0),
        None,
        None,
    )
    .unwrap();

    assert_eq!(result.solution.node_count(), 2);
    assert_eq!(result.sweeps, 0);
    assert!(result.residual.is_some_and(|residual| residual < 1.0e-12));
    assert!(!result.converged);
}

#[test]
fn test_square_linsolve_can_skip_final_residual() {
    let operator = create_simple_2site_mpo();
    let rhs = create_simple_2site_mps();
    let init = create_simple_2site_mps();

    let result = square_linsolve(
        &operator,
        &rhs,
        init,
        &"site0".to_string(),
        LinsolveOptions::new(0).with_residual_check(false),
        None,
        None,
    )
    .unwrap();

    assert_eq!(result.solution.node_count(), 2);
    assert_eq!(result.sweeps, 0);
    assert!(result.residual.is_none());
    assert!(!result.converged);
}

#[test]
fn test_square_linsolve_validation_error_bubbles_up() {
    let operator = create_simple_2site_mpo();
    let rhs = create_simple_2site_mps();

    let mut init = TreeTN::<IdxTensor, String>::new();
    let s0 = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(3);
    let b01 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 6]).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 6]).unwrap();

    let n0 = init.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = init.add_tensor("site1".to_string(), t1).unwrap();
    init.connect(n0, &b01, n1, &b01).unwrap();

    let err = square_linsolve(
        &operator,
        &rhs,
        init,
        &"site0".to_string(),
        LinsolveOptions::new(0),
        None,
        None,
    )
    .unwrap_err()
    .to_string();

    assert!(
        err.contains("Operator cannot act on init")
            || err.contains("Result of operator action is not compatible with RHS"),
        "unexpected error: {err}"
    );
}

// Note: square_linsolve requires MPO and MPS to have compatible index structures.
// The MPO should have input indices matching the MPS site indices.
// These tests are simplified to only test validation, as full linsolve requires
// proper index mappings which are tested in integration tests.

fn pow2(k: i32) -> f64 {
    pow2_factors(k).product()
}

#[test]
fn divide_by_scalar_stays_finite_when_the_quotient_is_finite() {
    use num_complex::Complex64;
    let s = DynIndex::new_dyn(4);
    let values = [1.0, -2.5, 0.75, 3.0];
    // (divisor, rhs scale, expected quotient for a unit rhs entry)
    let cases = [
        // Subnormal: 1 / divisor overflows.
        (
            Complex64::new(3.0 * pow2(-1070), 0.0),
            pow2(-1000),
            Complex64::new(pow2(70) / 3.0, 0.0),
        ),
        // Smallest subnormal; the rhs entries are exact subnormals.
        (
            Complex64::new(pow2(-1074), 0.0),
            pow2(-1060),
            Complex64::new(pow2(14), 0.0),
        ),
        // Complex, |divisor|^2 underflows.
        (
            Complex64::new(3.0, 4.0) * pow2(-1070),
            pow2(-1000),
            Complex64::new(3.0, -4.0) * (pow2(70) / 25.0),
        ),
        // Complex, |divisor|^2 overflows.
        (
            Complex64::new(3.0, 4.0) * pow2(1018),
            pow2(1000),
            Complex64::new(3.0, -4.0) * (pow2(-18) / 25.0),
        ),
        // Real divisor near f64::MAX with an rhs near it.
        (
            Complex64::new(f64::MAX, 0.0),
            f64::MAX / 4.0,
            Complex64::new(0.25, 0.0),
        ),
        // Ordinary divisors on both sides of one.
        (Complex64::new(2.0, 0.0), 1.0, Complex64::new(0.5, 0.0)),
        (
            Complex64::new(0.75, 0.0),
            1.0,
            Complex64::new(4.0 / 3.0, 0.0),
        ),
    ];
    for (divisor, rhs_scale, unit_quotient) in cases {
        let rhs: Vec<Complex64> = values
            .iter()
            .map(|&v| Complex64::new(v * rhs_scale, 0.0))
            .collect();
        let tensor = IdxTensor::from_dense(vec![s.clone()], rhs).unwrap();
        let mut tn = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0]).unwrap();
        let divisor = AnyScalar::new_complex(divisor.re, divisor.im);
        divide_by_scalar(&mut tn, &divisor).unwrap();

        let expected: Vec<Complex64> = values.iter().map(|&v| unit_quotient * v).collect();
        let dense = tn.to_dense().unwrap();
        let expected = IdxTensor::from_dense(dense.external_indices(), expected).unwrap();
        let scale = expected.maxabs().unwrap();
        let error = dense.sub(&expected).unwrap().maxabs().unwrap();
        assert!(
            error.is_finite() && error <= 1e-15 * scale,
            "{divisor:?}: {error} vs {scale}"
        );
    }
}

#[test]
fn divide_by_scalar_rejects_zero_and_non_finite_divisors() {
    let s = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense(vec![s], vec![1.0, 2.0]).unwrap();
    let tn = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0]).unwrap();
    for divisor in [0.0, f64::NAN, f64::INFINITY] {
        let error = divide_by_scalar(&mut tn.clone(), &AnyScalar::new_real(divisor)).unwrap_err();
        assert!(error.to_string().contains("cannot divide by"), "{error}");
    }
}

fn divide_single_entry<E>(value: E, divisor: AnyScalar) -> E
where
    E: tensor4all_core::TensorElement + Copy,
{
    let s = DynIndex::new_dyn(1);
    let tensor = IdxTensor::from_dense(vec![s], vec![value]).unwrap();
    let mut tn = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0]).unwrap();
    divide_by_scalar(&mut tn, &divisor).unwrap();
    tn.to_dense().unwrap().to_vec::<E>().unwrap()[0]
}

#[test]
fn divide_by_scalar_keeps_finite_components_of_a_rotated_quotient() {
    use num_complex::Complex64;
    // |quotient| exceeds f64::MAX but both components are finite, so no
    // intermediate may grow the real entry past f64::MAX before the rotation.
    let entry = Complex64::new(0.51 * f64::MAX, 0.0);
    let divisor = Complex64::new(0.35, 0.35);
    let expected = entry * (Complex64::new(1.0, 0.0) / divisor);
    assert!(expected.re.is_finite() && expected.im.is_finite());
    // Componentwise: |expected| itself overflows, but its components do not.
    let scale = expected.re.abs().max(expected.im.abs());
    assert!(
        scale.is_finite() && scale > 0.0,
        "reference must be finite: {scale:e}"
    );
    let error_of = |actual: Complex64| {
        (actual.re - expected.re)
            .abs()
            .max((actual.im - expected.im).abs())
    };
    // The comparison rejects a wrong answer.
    assert!(error_of(Complex64::new(0.0, 0.0)) > 1e-15 * scale);
    // k = 0 takes the direct reciprocal, k = 600 the small-divisor path.
    for k in [0, 600] {
        let actual = divide_single_entry(
            entry * pow2(-k),
            AnyScalar::new_complex(divisor.re * pow2(-k), divisor.im * pow2(-k)),
        );
        let error = error_of(actual);
        assert!(error <= 1e-15 * scale, "2^{k}: {actual} vs {expected}");
    }
}

#[test]
fn divide_by_scalar_rounds_a_subnormal_quotient_once() {
    // from_bits(2) / 3 is 2/3 of the smallest subnormal, which rounds to it.
    // k = 0 takes the direct reciprocal, k = 600 the large-divisor path.
    for k in [0, 600] {
        let actual = divide_single_entry(
            f64::from_bits(2) * pow2(k),
            AnyScalar::new_real(3.0 * pow2(k)),
        );
        assert_eq!(actual.to_bits(), 1, "2^{k}: {actual:e}");
    }
}
