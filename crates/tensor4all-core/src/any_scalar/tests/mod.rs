use super::*;

fn with_forced_tensor_initialization_failure<T>(f: impl FnOnce() -> T) -> T {
    let previous =
        FORCE_ANY_SCALAR_TENSOR_INITIALIZATION_FAILURE.with(|failure| failure.replace(true));
    let result = f();
    FORCE_ANY_SCALAR_TENSOR_INITIALIZATION_FAILURE.with(|failure| failure.set(previous));
    result
}

#[test]
fn compact_sum_preserves_f32_and_c32_dtype_with_and_without_ad() {
    let indices = || vec![crate::DynIndex::new_dyn(2), crate::DynIndex::new_dyn(2)];
    for tensor in [
        IdxTensor::from_diag(indices(), vec![1.0_f32, 2.0_f32])
            .unwrap()
            .sum()
            .unwrap(),
        IdxTensor::from_diag(indices(), vec![1.0_f32, 2.0_f32])
            .unwrap()
            .enable_grad()
            .unwrap()
            .sum()
            .unwrap(),
    ] {
        assert!(matches!(tensor.value(), ScalarValue::F32(3.0)));
    }
    for tensor in [
        IdxTensor::from_diag(
            indices(),
            vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
        )
        .unwrap()
        .sum()
        .unwrap(),
        IdxTensor::from_diag(
            indices(),
            vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
        )
        .unwrap()
        .enable_grad()
        .unwrap()
        .sum()
        .unwrap(),
    ] {
        assert!(
            matches!(tensor.value(), ScalarValue::C32(value) if value == Complex32::new(4.0, 6.0))
        );
    }
}

#[test]
fn non_grad_scalar_arithmetic_uses_plain_values() {
    let a = AnyScalar::new_real(3.0);
    let b = AnyScalar::new_real(4.0);

    let value =
        ((a.clone() + b.clone()) * b.clone() - AnyScalar::new_real(8.0)) / AnyScalar::new_real(2.0);

    assert_eq!(value.as_f64(), Some(10.0));
    assert!(!value.tracks_grad());
    assert!(value.as_tensor().is_ok());
}

#[test]
fn tracked_scalar_arithmetic_preserves_autodiff() {
    let x = AnyScalar::new_real(2.0).enable_grad().unwrap();
    let y = &x * &x;

    assert!(y.tracks_grad());
    y.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.as_f64(), Some(4.0));
}

#[test]
fn scalar_tensor_initialization_failure_is_retained_for_tensor_operations() {
    let scalar = with_forced_tensor_initialization_failure(|| AnyScalar::new_real(2.0));
    assert_eq!(scalar.real(), 2.0);
    assert!(!scalar.tracks_grad());

    let error = scalar.as_tensor().unwrap_err();
    assert!(error.downcast_ref::<AnyScalarTensorError>().is_some());
    assert!(error
        .to_string()
        .contains("AnyScalar tensor initialization failed"));
    assert!(error
        .chain()
        .any(|cause| cause.to_string() == "forced AnyScalar eager initialization failure"));

    let error = scalar.clone().enable_grad().unwrap_err();
    assert!(error
        .source
        .downcast_ref::<AnyScalarTensorError>()
        .is_some());
    assert!(error
        .source
        .chain()
        .any(|cause| cause.to_string() == "forced AnyScalar eager initialization failure"));
}

#[test]
fn tracked_scalar_operation_failure_retains_error_and_graph_state() {
    let scalar = AnyScalar::new_real(2.0).enable_grad().unwrap();
    let result = with_forced_tensor_initialization_failure(|| scalar.powf(2.0));

    assert!(result.tracks_grad());
    assert_eq!(result.real(), 4.0);

    let error = result.as_tensor().unwrap_err();
    assert!(error.downcast_ref::<AnyScalarTensorError>().is_some());
    assert!(error
        .to_string()
        .contains("AnyScalar tensor initialization failed"));
    assert!(error
        .chain()
        .any(|cause| cause.to_string() == "forced AnyScalar eager initialization failure"));

    let error = result.clone().enable_grad().unwrap_err();
    assert!(error
        .source
        .downcast_ref::<AnyScalarTensorError>()
        .is_some());
    assert!(error
        .source
        .chain()
        .any(|cause| cause.to_string() == "forced AnyScalar eager initialization failure"));
}

#[test]
fn tracked_backend_failure_preserves_typed_diagnostic_through_fallback() {
    let lhs = AnyScalar::new_real(2.0).enable_grad().unwrap();
    let rhs = AnyScalar::new_real(3.0).enable_grad().unwrap();
    let operation = AnyScalar::from_eager_binary(&lhs, &rhs, "add", |_session, _lhs, _rhs| {
        Err(tenferro_ad::Error::from(
            tenferro_tensor::Error::backend_failure("forced_add", "forced tracked backend failure"),
        ))
    });
    let result = AnyScalar::fallback_result(operation, "add", || ScalarValue::F64(5.0), true);

    assert!(result.tracks_grad());
    assert_eq!(result.real(), 5.0);
    let error = result.as_tensor().unwrap_err();
    assert!(error.downcast_ref::<AnyScalarTensorError>().is_some());
    let stored = error.downcast_ref::<AnyScalarTensorError>().unwrap();
    match stored {
        AnyScalarTensorError::Operation { source, .. } => {
            assert!(source
                .downcast_ref::<tenferro_ad::Error>()
                .is_some_and(|error| error.to_string().contains("forced tracked backend failure")));
        }
        AnyScalarTensorError::Initialization { .. } => {
            panic!("operation failure was converted to initialization failure")
        }
    }
    let error = result.enable_grad().unwrap_err();
    assert!(error.to_string().contains("forced tracked backend failure"));
}

#[test]
fn every_infallible_scalar_fallback_retains_a_tracked_error() {
    let failed = AnyScalar {
        tensor: Err(AnyScalarTensorError::Operation {
            op: "seed",
            source: Arc::new(std::io::Error::other("forced tracked scalar failure")),
        }),
        value: ScalarValue::F64(2.0),
        tracks_grad: true,
    };
    let one = AnyScalar::new_real(1.0);

    let results = [
        &failed + &one,
        &failed * &one,
        &failed / &one,
        -&failed,
        failed.conj(),
        failed.real_part(),
        failed.imag_part(),
        failed.sqrt(),
        failed.powf(2.0),
        failed.powi(2),
    ];
    for result in results {
        assert!(result.tracks_grad());
        let error = result.as_tensor().unwrap_err();
        assert!(error.downcast_ref::<AnyScalarTensorError>().is_some());
        assert!(error
            .chain()
            .any(|cause| cause.to_string() == "forced tracked scalar failure"));
    }
}

#[test]
fn every_infallible_scalar_operation_retains_an_initialization_error() {
    let failed = with_forced_tensor_initialization_failure(|| AnyScalar::new_real(2.0));
    let one = AnyScalar::new_real(1.0);

    let results = [
        &failed + &one,
        &failed * &one,
        &failed / &one,
        -&failed,
        failed.conj(),
        failed.real_part(),
        failed.imag_part(),
        failed.sqrt(),
        failed.powf(2.0),
        failed.powi(0),
    ];
    for result in results {
        let error = result.as_tensor().unwrap_err();
        assert!(error.downcast_ref::<AnyScalarTensorError>().is_some());
        assert!(error
            .chain()
            .any(|cause| cause.to_string() == "forced AnyScalar eager initialization failure"));
    }
}
