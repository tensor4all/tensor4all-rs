//! F1 slice 3: `sum` on a caller-held explicit session.
//!
//! The reduction must agree with `IdxTensor::sum` for every dense host representation this slice serves -
//! including the narrow `f32`/`c32` payloads - and must refuse what it does not serve instead of falling back.

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{sum_on_session, IdxTensor};
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_tensorbackend::{CpuExecutionContext, Storage};

fn session_context() -> CpuExecutionContext {
    let backend = CpuBackend::with_threads(1).expect("1T CPU backend");
    CpuExecutionContext::from_backend(backend)
}

#[test]
fn sum_on_session_matches_the_dense_expansion_and_the_compatibility_route() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(3);
    let values = vec![1.0_f64, -2.5, 3.25, 4.0, -5.5, 6.75];

    let cases: Vec<(&str, IdxTensor)> = vec![
        (
            "dense",
            IdxTensor::from_dense(vec![i.clone(), k.clone()], values.clone()).unwrap(),
        ),
        (
            "from_storage",
            IdxTensor::from_storage(
                vec![i.clone(), k.clone()],
                Arc::new(Storage::from_dense_col_major(values.clone(), &[2, 3]).expect("storage")),
            )
            .unwrap(),
        ),
        (
            "rank zero",
            IdxTensor::from_dense(Vec::new(), vec![2.5_f64]).unwrap(),
        ),
    ];

    for (label, tensor) in cases {
        // Summing the input data is independent of both `IdxTensor::sum` and `Session::sum`.
        let expected = if label == "rank zero" {
            2.5
        } else {
            values.iter().sum::<f64>()
        };
        let compatibility = tensor.sum().expect("compatibility sum");
        assert!(
            (compatibility.real() - expected).abs() < 1e-12,
            "{label}: the compatibility route disagrees with the dense expansion"
        );
        let actual = context
            .with_concrete_session(|session| sum_on_session(session, &tensor))
            .expect("session entry")
            .expect("session sum");
        assert!(
            (actual.real() - expected).abs() < 1e-12,
            "{label}: {} != {expected}",
            actual.real()
        );
        assert!(actual.imag().abs() < 1e-12, "{label}: {}", actual.imag());
    }
}

#[test]
fn the_dtype_domain_is_the_shared_one_and_its_refusals_are_typed() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);

    // A complex value is served and keeps both components.
    let c64_tensor = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-0.5, 0.5),
            Complex64::new(0.0, -1.5),
        ],
    )
    .unwrap();
    let c64_sum = context
        .with_concrete_session(|session| sum_on_session(session, &c64_tensor))
        .expect("session entry")
        .expect("session sum");
    assert!((c64_sum.real() - 2.5).abs() < 1e-12, "{}", c64_sum.real());
    assert!(
        (c64_sum.imag() - (-1.0)).abs() < 1e-12,
        "{}",
        c64_sum.imag()
    );

    // f32 and c32 carry their own compact payloads, so this route serves them and keeps their dtype
    // instead of widening into the f64 accumulation.
    let f32_tensor =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.5_f32, 2.5, -1.0, 3.0]).unwrap();
    let c32_tensor = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![
            Complex32::new(1.0, 0.5),
            Complex32::new(-2.0, 1.0),
            Complex32::new(0.25, -0.75),
            Complex32::new(0.0, 0.0),
        ],
    )
    .unwrap();
    for (label, tensor) in [("f32", &f32_tensor), ("c32", &c32_tensor)] {
        let compatibility = tensor.sum().expect("compatibility sum");
        let served = context
            .with_concrete_session(|session| sum_on_session(session, tensor))
            .expect("session entry")
            .expect("a narrow dtype is served");
        assert_eq!(served.real(), compatibility.real(), "{label}");
        assert_eq!(served.imag(), compatibility.imag(), "{label}");
    }
}

#[test]
fn refusals_keep_the_held_session_usable() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let dense =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let diagonal = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0]).unwrap();
    // Built outside the session: tracking adopts the value into the compatibility eager runtime.
    let tracked = dense.clone().enable_grad().expect("tracked operand");

    context
        .with_concrete_session(|session| {
            let error = sum_on_session(session, &diagonal).expect_err("diagonal refused");
            match error {
                tensor4all_core::IdxTensorError::Operation { operation, .. } => {
                    assert_eq!(operation, "sum_on_session");
                }
                other => panic!("expected a typed operation refusal, got {other:?}"),
            }
            sum_on_session(session, &tracked).expect_err("tracked refused");

            // Several reductions inside one held region equal the separate calls.
            let first = sum_on_session(session, &dense).expect("first");
            let second = sum_on_session(session, &dense).expect("second");
            assert_eq!(first.real(), second.real());
            assert_eq!(first.real(), dense.sum().expect("compatibility").real());
            Ok::<(), tensor4all_core::IdxTensorError>(())
        })
        .expect("session entry")
        .expect("session work after refusals");
}

#[test]
fn complex_values_keep_compatibility_parity_through_every_accepted_representation() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let values = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(-0.5, 0.5),
        Complex64::new(0.0, -1.5),
    ];

    let dense = IdxTensor::from_dense(vec![i.clone(), k.clone()], values.clone()).unwrap();
    let stored = IdxTensor::from_storage(
        vec![i.clone(), k.clone()],
        Arc::new(Storage::from_dense_col_major(values.clone(), &[2, 2]).expect("storage")),
    )
    .unwrap();
    let rank_zero = IdxTensor::from_dense(Vec::new(), vec![Complex64::new(-2.0, 3.0)]).unwrap();

    for (label, tensor, expected) in [
        ("dense", dense, values.iter().copied().sum::<Complex64>()),
        (
            "from_storage",
            stored,
            values.iter().copied().sum::<Complex64>(),
        ),
        ("rank zero", rank_zero, Complex64::new(-2.0, 3.0)),
    ] {
        let compatibility = tensor.sum().expect("compatibility sum");
        assert!(
            (compatibility.real() - expected.re).abs() < 1e-12
                && (compatibility.imag() - expected.im).abs() < 1e-12,
            "{label}: the compatibility route disagrees with the input-data sum"
        );
        let actual = context
            .with_concrete_session(|session| sum_on_session(session, &tensor))
            .expect("session entry")
            .expect("session sum");
        assert!(
            (actual.real() - expected.re).abs() < 1e-12,
            "{label}: {} != {}",
            actual.real(),
            expected.re
        );
        assert!(
            (actual.imag() - expected.im).abs() < 1e-12,
            "{label}: {} != {}",
            actual.imag(),
            expected.im
        );
        // The scalar's carried tensor must agree with its cached value: a public operation that routes
        // through the tensor sees the same numbers.
        let conjugated = actual.conj();
        assert!((conjugated.real() - actual.real()).abs() < 1e-12, "{label}");
        assert!((conjugated.imag() + actual.imag()).abs() < 1e-12, "{label}");
    }
}

#[test]
fn a_zero_length_axis_keeps_the_backends_refusal_and_leaves_the_session_usable() {
    let context = session_context();
    let empty_index = Index::new_dyn(0);
    let k = Index::new_dyn(3);

    // The compatibility route accumulates from zero over the empty payload and answers 0.
    let empty = IdxTensor::from_storage(
        vec![empty_index.clone(), k.clone()],
        Arc::new(Storage::from_dense_col_major(Vec::<f64>::new(), &[0, 3]).expect("storage")),
    )
    .unwrap();
    let compatibility = empty.sum().expect("compatibility sum");
    assert_eq!(
        compatibility.real(),
        0.0,
        "compatibility accumulates from zero"
    );

    // The session route keeps its backend's refusal for a zero-length axis rather than special-casing it:
    // the documented non-parity. It is a session-side failure, not an operand refusal, so it also shows the
    // held session surviving one.
    let dense = IdxTensor::from_dense(vec![k.clone()], vec![1.0_f64, 2.0, 3.0]).unwrap();
    context
        .with_concrete_session(|session| {
            let error = sum_on_session(session, &empty).expect_err("a zero-length axis is refused");
            match error {
                tensor4all_core::IdxTensorError::Operation { operation, .. } => {
                    assert_eq!(operation, "sum_on_session");
                }
                other => panic!("expected a typed operation refusal, got {other:?}"),
            }
            let value = sum_on_session(session, &dense).expect("session sum after a refusal");
            assert_eq!(value.real(), 6.0);
            Ok::<(), tensor4all_core::IdxTensorError>(())
        })
        .expect("session entry")
        .expect("session work after a zero-length refusal");
}
