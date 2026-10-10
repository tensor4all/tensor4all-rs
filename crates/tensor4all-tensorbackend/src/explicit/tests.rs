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
#[cfg(feature = "global-defaults")]
#[test]
fn explicit_routes_match_the_compatibility_frontend() {
    let context = context(1);
    let a = matrix();
    let b = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).expect("rhs");

    let (
        explicit_qr,
        explicit_svd,
        explicit_contraction,
        explicit_reshape,
        explicit_permute,
        explicit_einsum,
    ) = context
        .with_concrete_session(|session| {
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>((
                session.qr(&a)?.0,
                session.svd(&a)?.1,
                session.contraction(&a, &[1], &b, &[0])?,
                session.reshape(&a, &[4])?,
                session.permute(&a, &[1, 0])?,
                session.einsum(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2])?,
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
    let compatibility_permute =
        crate::permute_native_tensor(&a, &[1, 0]).expect("compatibility permute");
    let compatibility_einsum =
        crate::einsum_native_tensors(&[(&a, &[0usize, 1]), (&b, &[1usize, 2])], &[0, 2])
            .expect("compatibility einsum");

    for (explicit, compatibility, route) in [
        (explicit_qr, compatibility_qr, "qr"),
        (explicit_svd, compatibility_svd, "svd"),
        (
            explicit_contraction,
            compatibility_contraction,
            "contraction",
        ),
        (explicit_reshape, compatibility_reshape, "reshape"),
        (explicit_permute, compatibility_permute, "permute"),
        (explicit_einsum, compatibility_einsum, "einsum"),
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
    // The value is produced by the producer context's own session...
    let produced = producer
        .with_concrete_session(|session| {
            let a = matrix();
            session.contraction(&a, &[1], &a, &[0])
        })
        .expect("producer session entry")
        .expect("producer contraction");
    let pointer = produced.as_slice::<f64>().expect("f64 payload").as_ptr();

    // ...and consumed by a context with a different thread budget.
    let product = consumer
        .with_concrete_session(|session| session.contraction(&produced, &[1], &produced, &[0]))
        .expect("consumer session entry")
        .expect("cross-context contraction");

    // `A^2` for `A = [[1, 3], [2, 4]]` is `[[7, 15], [10, 22]]`, and squaring that
    // gives `[[199, 435], [290, 634]]`, i.e. `[199, 290, 435, 634]` column-major.
    assert_eq!(
        product.as_slice::<f64>().expect("f64 payload"),
        &[199.0, 290.0, 435.0, 634.0]
    );
    assert_eq!(
        produced.as_slice::<f64>().expect("f64 payload").as_ptr(),
        pointer,
        "the input storage must be borrowed, not moved or re-registered"
    );
    assert_eq!(producer.with_backend(|backend| backend.num_threads()), 1);
    assert_eq!(consumer.with_backend(|backend| backend.num_threads()), 4);
}

/// A nested canonical session entry is rejected typed, before any lock, and the
/// context stays usable afterwards.
#[test]
fn a_nested_explicit_session_entry_is_rejected_typed() {
    let context = context(1);
    let error = context
        .with_concrete_session(|_session| context.with_concrete_session(|_inner| ()))
        .expect("the outer session entry")
        .expect_err("a nested canonical session entry must be rejected");
    let crate::context::CpuExecutionContextError::SessionEntry { source } = &error else {
        panic!("the rejection must be a typed session entry error: {error}");
    };
    assert!(
        matches!(
            source.downcast_ref::<tenferro_tensor::SessionEntryError>(),
            Some(tenferro_tensor::SessionEntryError::Reentered { .. })
        ),
        "the rejection must preserve tenferro's typed reentry cause"
    );

    // The guard was restored: an independent session still works.
    let reused = context
        .with_concrete_session(|session| session.reshape(&matrix(), &[4]))
        .expect("session entry")
        .expect("explicit reshape");
    assert_eq!(reused.as_slice::<f64>().expect("f64 payload").len(), 4);
}

/// A label-list count that does not match the operand count is rejected typed, and
/// the session stays usable: the rejection replaces the former assertion.
#[test]
fn einsum_rejects_a_mismatched_label_count_typed() {
    let context = context(1);
    let a = matrix();
    let error = context
        .with_concrete_session(|session| session.einsum(&[&a], &[], &[]))
        .expect("session entry")
        .expect_err("a mismatched label count must be rejected");
    assert!(
        matches!(error, tenferro_einsum::Error::InvalidSubscripts { .. }),
        "expected a typed invalid-subscripts error, got {error}"
    );

    let recovered = context
        .with_concrete_session(|session| session.reshape(&a, &[4]))
        .expect("session entry")
        .expect("the session stays usable");
    assert_eq!(recovered.as_slice::<f64>().expect("f64 payload").len(), 4);
}

/// The explicit routes do not promote operands to a common dtype: a mixed-precision
/// contraction is rejected typed instead of being converted. The compatibility
/// frontend promotes, which is recorded as a difference in
/// `docs/design/859-dual-frontend-coexistence.md`.
#[test]
fn mixed_precision_operands_are_rejected_typed() {
    let context = context(1);
    let lhs = matrix();
    let rhs = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f32, 6.0]).expect("f32 rhs");
    let error = context
        .with_concrete_session(|session| session.contraction(&lhs, &[1], &rhs, &[0]))
        .expect("session entry")
        .expect_err("a mixed-precision contraction must be rejected");
    let message = error.to_string();
    assert!(
        !message.is_empty(),
        "the rejection must carry a diagnostic: {error}"
    );
}
