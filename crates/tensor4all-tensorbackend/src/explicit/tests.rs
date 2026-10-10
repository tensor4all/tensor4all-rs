use super::*;
use tenferro::Tensor;
use tenferro_cpu::CpuBackend;

fn context(threads: usize) -> CpuExecutionContext {
    CpuExecutionContext::from_backend(CpuBackend::with_threads(threads).expect("CPU backend"))
}

fn matrix() -> Tensor {
    Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).expect("matrix")
}

/// Every explicit route agrees with the compatibility frontend on the same input.
#[test]
fn explicit_routes_match_the_compatibility_frontend() {
    let context = context(1);
    let a = matrix();
    let b = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).expect("rhs");

    let (explicit_qr, explicit_svd, explicit_contraction, explicit_reshape) = context
        .with_concrete_session(|session| {
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>((
                session.qr(&a)?.0,
                session.svd(&a)?.1,
                session.contraction(&a, &[1], &b, &[0])?,
                session.reshape(&a, &[4])?,
            ))
        })
        .expect("session entry")
        .expect("explicit routes");

    let compatibility_qr = crate::qr_native_tensor(&a).expect("compatibility QR").0;
    let compatibility_svd = crate::svd_native_tensor(&a).expect("compatibility SVD").1;
    let compatibility_contraction =
        crate::contract_native_tensor(&a, &[1], &b, &[0]).expect("compatibility contraction");
    let compatibility_reshape =
        crate::reshape_col_major_native_tensor(&a, &[4]).expect("compatibility reshape");

    for (explicit, compatibility, route) in [
        (explicit_qr, compatibility_qr, "qr"),
        (explicit_svd, compatibility_svd, "svd"),
        (
            explicit_contraction,
            compatibility_contraction,
            "contraction",
        ),
        (explicit_reshape, compatibility_reshape, "reshape"),
    ] {
        assert_eq!(explicit.shape(), compatibility.shape(), "{route} shape");
        let explicit_values = explicit.as_slice::<f64>().expect("f64 payload");
        let compatibility_values = compatibility.as_slice::<f64>().expect("f64 payload");
        for (left, right) in explicit_values.iter().zip(compatibility_values) {
            assert!(
                (left - right).abs() < 1e-12,
                "{route} disagrees: {left} vs {right}"
            );
        }
    }
}

/// The linalg solve route agrees with the linear system it solves. There is no
/// native-tensor counterpart in the compatibility frontend to compare against.
#[test]
fn explicit_solve_satisfies_the_linear_system() {
    let context = context(1);
    let a = matrix();
    let b = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).expect("rhs");

    let solved = context
        .with_concrete_session(|session| session.solve(&a, &b))
        .expect("session entry")
        .expect("explicit solve");
    let solved = solved.as_slice::<f64>().expect("f64 payload");
    // `a` is `[[1, 3], [2, 4]]` column-major.
    assert!((solved[0] + 3.0 * solved[1] - 5.0).abs() < 1e-12);
    assert!((2.0 * solved[0] + 4.0 * solved[1] - 6.0).abs() < 1e-12);
}

/// One session serves a whole batch.
///
/// Independence from the compatibility frontend is structural: this module never
/// names `with_default_session`, the default context or the eager runtime, and the
/// crate's session-entry audit covers the boundary. `default_context_hits` cannot
/// prove it here, because the rest of this parallel test binary initializes the
/// process-global context on its own.
#[test]
fn one_session_serves_a_batch() {
    let context = context(2);
    let a = matrix();

    let products = context
        .with_concrete_session(|session| {
            let mut products = Vec::new();
            for _ in 0..16 {
                products.push(session.contraction(&a, &[1], &a, &[0])?);
            }
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(products)
        })
        .expect("session entry")
        .expect("explicit contractions");

    assert_eq!(products.len(), 16);
    for product in products {
        assert_eq!(
            product.as_slice::<f64>().expect("f64 payload"),
            &[7.0, 10.0, 15.0, 22.0]
        );
    }
}

/// An invalid input is reported by the backend instead of being retried on another
/// route.
#[test]
fn invalid_input_reports_a_typed_error_instead_of_falling_back() {
    let context = context(1);
    let non_square = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).expect("matrix");
    let rhs = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).expect("rhs");

    let error = context
        .with_concrete_session(|session| session.solve(&non_square, &rhs))
        .expect("session entry")
        .expect_err("a non-square solve must fail");
    assert!(
        matches!(error, tenferro_tensor::Error::Validation { .. }),
        "expected a typed validation error, got {error}"
    );
}

/// A value produced by one context is usable in another context's session without a
/// conversion copy or a re-registration, because the route validates storage rather
/// than execution-context identity.
#[test]
fn values_cross_cpu_budget_contexts_without_a_conversion_copy() {
    let producer = context(1);
    let consumer = context(4);
    let a = matrix();
    let pointer = a.as_slice::<f64>().expect("f64 payload").as_ptr();

    let product = consumer
        .with_concrete_session(|session| session.contraction(&a, &[1], &a, &[0]))
        .expect("session entry")
        .expect("cross-context contraction");

    assert_eq!(
        product.as_slice::<f64>().expect("f64 payload"),
        &[7.0, 10.0, 15.0, 22.0]
    );
    assert_eq!(
        a.as_slice::<f64>().expect("f64 payload").as_ptr(),
        pointer,
        "the input storage must be borrowed, not moved or re-registered"
    );
    assert_eq!(producer.with_backend(|backend| backend.num_threads()), 1);
}

/// A nested canonical session entry is rejected. The legacy canonical-session guard
/// asserts rather than returning a typed error, which is the behaviour documented in
/// `docs/design/tensorbackend-session-entry.md`; turning it into a typed rejection is
/// a separate change to that guard.
#[test]
fn a_nested_explicit_session_entry_is_rejected() {
    let context = context(1);
    let nested = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        context.with_concrete_session(|_session| context.with_concrete_session(|_inner| ()))
    }));
    let panic = nested.expect_err("a nested canonical session entry must be rejected");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or_default();
    assert!(
        message.contains("recursive tensorbackend canonical session entry"),
        "unexpected rejection message: {message}"
    );
}
