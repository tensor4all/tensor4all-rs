use std::sync::Arc;

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

/// Mixed-precision operands are promoted to a common dtype exactly as the
/// compatibility frontend promotes them, and the result carries the promoted dtype.
#[cfg(feature = "global-defaults")]
#[test]
fn mixed_precision_operands_promote_like_the_compatibility_frontend() {
    let context = context(1);
    let lhs = matrix();
    let rhs = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f32, 6.0]).expect("f32 rhs");

    let explicit = context
        .with_concrete_session(|session| session.contraction(&lhs, &[1], &rhs, &[0]))
        .expect("session entry")
        .expect("promoted contraction");
    let compatibility =
        crate::contract_native_tensor(&lhs, &[1], &rhs, &[0]).expect("compatibility contraction");

    assert_eq!(explicit.dtype(), compatibility.dtype());
    assert_eq!(
        explicit.as_slice::<f64>().expect("f64 payload"),
        compatibility.as_slice::<f64>().expect("f64 payload")
    );
    assert_eq!(
        explicit.as_slice::<f64>().expect("f64 payload"),
        &[23.0, 34.0]
    );
}

/// The session's `Debug` output is a summary, not a dump of the execution state.
#[test]
fn session_debug_is_a_summary() {
    let context = context(1);
    context
        .with_concrete_session(|session| {
            let rendered = format!("{session:?}");
            assert!(rendered.contains("explicit::Session"), "{rendered}");
        })
        .expect("session entry");
}

/// Axis lists of different lengths are rejected typed instead of contracting
/// something unexpected.
#[test]
fn contraction_rejects_mismatched_axes_typed() {
    let context = context(1);
    let a = matrix();
    let error = context
        .with_concrete_session(|session| session.contraction(&a, &[1], &a, &[]))
        .expect("session entry")
        .expect_err("mismatched axis lists must be rejected");
    assert!(
        matches!(error, tenferro_einsum::Error::InvalidSubscripts { .. }),
        "expected a typed invalid-subscripts error, got {error}"
    );
}

/// A label list whose length does not match the operand rank is rejected typed.
#[test]
fn einsum_rejects_a_mismatched_label_rank_typed() {
    let context = context(1);
    let a = matrix();
    let error = context
        .with_concrete_session(|session| session.einsum(&[&a], &[&[0, 1, 2]], &[0]))
        .expect("session entry")
        .expect_err("a mismatched label rank must be rejected");
    // The rank mismatch is reported by einsum planning, which is the typed rejection
    // for this route; the message names the offending operand.
    assert!(
        error.to_string().contains("subscript labels") || error.to_string().contains("rank"),
        "expected a diagnostic naming the label mismatch, got {error}"
    );
}

/// N-ary einsum promotes heterogeneous operands through the same path contraction
/// uses.
#[test]
fn einsum_promotes_heterogeneous_operands() {
    let context = context(1);
    let a = matrix();
    let b =
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 0.0, 0.0, 1.0]).expect("f32 identity");

    let product = context
        .with_concrete_session(|session| session.einsum(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]))
        .expect("session entry")
        .expect("promoted einsum");
    assert_eq!(product.dtype(), tenferro::DType::F64);
    assert_eq!(
        product.as_slice::<f64>().expect("f64 payload"),
        &[1.0, 2.0, 3.0, 4.0]
    );
}

/// `triangular_solve` solves a triangular system on the session, and reports a
/// shape mismatch typed.
#[test]
fn triangular_solve_solves_and_reports_a_shape_mismatch() {
    let context = context(1);
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 4.0]).expect("lower");
    let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 4.0]).expect("rhs");
    let x = context
        .with_concrete_session(|session| session.triangular_solve(&a, &b, true, true, false, false))
        .expect("session entry")
        .expect("triangular solve");
    assert_eq!(x.as_slice::<f64>().expect("f64 payload"), &[2.0, 0.5]);

    let wide = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).expect("wide rhs");
    let error = context
        .with_concrete_session(|session| {
            session.triangular_solve(&a, &wide, true, true, false, false)
        })
        .expect("session entry")
        .expect_err("a shape mismatch must be rejected");
    assert!(
        matches!(error, tenferro_tensor::Error::Validation { .. }),
        "expected a typed validation error, got {error}"
    );
}

/// `full_piv_lu` factors a matrix on the session, and reports a non-square input
/// typed.
#[test]
fn full_piv_lu_factors_and_reports_a_non_square_input() {
    let context = context(1);
    let a = matrix();
    let (p, l, u, q, _parity) = context
        .with_concrete_session(|session| session.full_piv_lu(&a))
        .expect("session entry")
        .expect("full-pivoting LU");
    for factor in [&p, &l, &u, &q] {
        assert_eq!(factor.shape(), &[2, 2]);
    }
    let l = l.as_slice::<f64>().expect("f64 payload");
    let u = u.as_slice::<f64>().expect("f64 payload");
    assert_eq!(l[2], 0.0, "L is lower triangular");
    assert_eq!(u[1], 0.0, "U is upper triangular");

    let tall = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).expect("tall");
    let error = context
        .with_concrete_session(|session| session.full_piv_lu(&tall))
        .expect("session entry")
        .expect_err("a non-square input must be rejected");
    assert!(
        !error.to_string().is_empty(),
        "the rejection must carry a diagnostic: {error}"
    );
}

/// A caller inside a Rayon worker is rejected typed: this frontend promises the
/// caller's own backend, and the compatibility frontend's inline worker fallback is
/// not a substitute for it.
#[test]
fn a_worker_caller_is_rejected_typed() {
    let context = Arc::new(context(1));
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-worker enclosing pool"),
    );
    let worker_context = Arc::clone(&context);
    let rejected = pool.install(move || worker_context.with_concrete_session(|_| ()).is_err());
    assert!(rejected, "a worker caller must be rejected typed");
    // The context stays usable from outside the pool.
    assert!(context.with_concrete_session(|_| ()).is_ok());
}
