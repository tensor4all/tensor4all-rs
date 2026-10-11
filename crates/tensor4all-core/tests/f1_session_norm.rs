//! F1 slice 6: `norm_squared` and `norm` on a caller-held explicit session.
//!
//! The session route must reproduce the compatibility route's *scaled* accumulation numerically, not an
//! algebraic shortcut, and must refuse what it does not serve instead of falling back.

use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{norm_on_session, norm_squared_on_session, IdxTensor};
use tensor4all_tensorbackend::CpuExecutionContext;

fn session_context() -> CpuExecutionContext {
    let backend = CpuBackend::with_threads(1).expect("1T CPU backend");
    CpuExecutionContext::from_backend(backend)
}

#[test]
fn norm_squared_matches_the_compatibility_route_for_both_served_dtypes() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(3);

    let real = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![1.0_f64, -2.5, 3.25, 4.0, -5.5, 6.75],
    )
    .unwrap();
    let complex = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-0.5, 0.5),
            Complex64::new(0.0, -1.5),
            Complex64::new(3.0, 0.0),
            Complex64::new(-1.0, 2.0),
        ],
    )
    .unwrap();

    for (label, tensor) in [("f64", &real), ("c64", &complex)] {
        let expected = tensor.norm_squared().expect("compatibility norm_squared");
        let actual = context
            .with_concrete_session(|session| norm_squared_on_session(session, tensor))
            .expect("session entry")
            .expect("session norm_squared");
        assert_eq!(
            actual, expected,
            "{label}: the same accumulator, the same value"
        );
        assert_eq!(
            context
                .with_concrete_session(|session| norm_on_session(session, tensor))
                .expect("session entry")
                .expect("session norm"),
            tensor.norm().expect("compatibility norm"),
            "{label}: the square root of the same value"
        );
    }
}

#[test]
fn a_discriminating_fixture_shows_the_accumulator_is_shared() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let sixteen = Index::new_dyn(16);

    // `[1.0, 1.0]`: the compatibility accumulator squares `sqrt(2)`, which is not exactly 2, while a direct
    // sum of squares returns 2. Equality here therefore distinguishes the accumulator from a naive sum.
    let square = IdxTensor::from_dense(vec![i.clone()], vec![1.0_f64, 1.0]).unwrap();
    let naive = 1.0_f64 + 1.0;
    let compatibility = square.norm_squared().expect("compatibility");
    assert_eq!(naive, 2.0);
    assert_ne!(
        compatibility, naive,
        "the accumulator squares `sqrt(2)` rather than summing squares"
    );
    let actual = context
        .with_concrete_session(|session| norm_squared_on_session(session, &square))
        .expect("session entry")
        .expect("session norm_squared");
    assert_eq!(
        actual, compatibility,
        "the session route uses the same accumulator"
    );

    // Sixteen values whose individual squares underflow: a direct sum is zero, while the scaled accumulation
    // keeps a representable squared norm.
    let tiny = vec![1e-162_f64; 16];
    let accumulated = IdxTensor::from_dense(vec![sixteen.clone()], tiny.clone()).unwrap();
    let naive_tiny: f64 = tiny.iter().map(|value| value * value).sum();
    assert_eq!(naive_tiny, 0.0, "each direct square underflows");
    let compatibility_tiny = accumulated.norm_squared().expect("compatibility");
    assert!(
        compatibility_tiny > 0.0,
        "the scaled accumulation keeps it representable: {compatibility_tiny}"
    );
    let actual_tiny = context
        .with_concrete_session(|session| norm_squared_on_session(session, &accumulated))
        .expect("session entry")
        .expect("session norm_squared");
    assert_eq!(actual_tiny, compatibility_tiny);

    // Independently known results, not parity: real [3, 4] and complex [3 + 4i] both have squared norm 25.
    let known = IdxTensor::from_dense(vec![i.clone()], vec![3.0_f64, 4.0]).unwrap();
    let known_complex = IdxTensor::from_dense(
        vec![i.clone()],
        vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
    )
    .unwrap();
    for tensor in [&known, &known_complex] {
        let squared = context
            .with_concrete_session(|session| norm_squared_on_session(session, tensor))
            .expect("session entry")
            .expect("session norm_squared");
        assert_eq!(squared, 25.0);
        let norm = context
            .with_concrete_session(|session| norm_on_session(session, tensor))
            .expect("session entry")
            .expect("session norm");
        assert_eq!(norm, 5.0);
    }
}

#[test]
fn representations_and_nonfinite_values_are_served_or_refused_as_documented() {
    use std::sync::Arc;
    use tensor4all_tensorbackend::Storage;

    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);

    // A fresh materialized operand: the other admission/materialization branch.
    let stored = IdxTensor::from_storage(
        vec![i.clone(), k.clone()],
        Arc::new(Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap()),
    )
    .unwrap();
    assert_eq!(
        context
            .with_concrete_session(|session| norm_squared_on_session(session, &stored))
            .expect("session entry")
            .expect("session norm_squared"),
        stored.norm_squared().expect("compatibility")
    );

    // Rank zero, an empty payload and a zero-length axis are all accepted and agree with compatibility.
    for tensor in [
        IdxTensor::from_dense(Vec::new(), vec![3.0_f64]).unwrap(),
        IdxTensor::from_dense(Vec::new(), vec![Complex64::new(0.0, 4.0)]).unwrap(),
        IdxTensor::from_storage(
            vec![Index::new_dyn(0)],
            Arc::new(Storage::from_dense_col_major(Vec::<f64>::new(), &[0]).unwrap()),
        )
        .unwrap(),
        IdxTensor::from_storage(
            vec![Index::new_dyn(0), Index::new_dyn(2)],
            Arc::new(Storage::from_dense_col_major(Vec::<f64>::new(), &[0, 2]).unwrap()),
        )
        .unwrap(),
    ] {
        let expected = tensor.norm_squared().expect("compatibility");
        let actual = context
            .with_concrete_session(|session| norm_squared_on_session(session, &tensor))
            .expect("session entry")
            .expect("session norm_squared");
        assert_eq!(actual, expected, "{:?}", tensor.dims());
    }

    // A non-contiguous eager operand: a compatibility permutation, materialized on the session.
    let dense =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let view = dense.permute(&[1, 0]).expect("compatibility permute");
    assert_eq!(
        context
            .with_concrete_session(|session| norm_squared_on_session(session, &view))
            .expect("session entry")
            .expect("session norm_squared"),
        view.norm_squared().expect("compatibility")
    );

    // Infinities: several of them, and one in a complex component, without NaN.
    let infinities =
        IdxTensor::from_dense(vec![i.clone()], vec![f64::INFINITY, f64::NEG_INFINITY]).unwrap();
    let complex_infinity = IdxTensor::from_dense(
        vec![i.clone()],
        vec![Complex64::new(1.0, f64::INFINITY), Complex64::new(0.0, 0.0)],
    )
    .unwrap();
    let complex_nan = IdxTensor::from_dense(
        vec![i.clone()],
        vec![Complex64::new(1.0, f64::NAN), Complex64::new(0.0, 0.0)],
    )
    .unwrap();
    for tensor in [&infinities, &complex_infinity] {
        let expected = tensor.norm_squared().expect("compatibility");
        let actual = context
            .with_concrete_session(|session| norm_squared_on_session(session, tensor))
            .expect("session entry")
            .expect("session norm_squared");
        assert_eq!(actual, expected);
        assert!(actual.is_infinite());
    }
    // A NaN in the imaginary component is refused, through both entry points.
    for squared in [true, false] {
        let error = context
            .with_concrete_session(|session| {
                if squared {
                    norm_squared_on_session(session, &complex_nan)
                } else {
                    norm_on_session(session, &complex_nan).map(|_| 0.0)
                }
            })
            .expect("session entry")
            .expect_err("a NaN component is refused");
        assert!(
            matches!(error, tensor4all_core::IdxTensorError::NaNInput { .. }),
            "{error:?}"
        );
    }

    // `f32`/`c32` are **served** by the norms since slice 7: they are promoted into the same `f64`
    // accumulation the compatibility route uses.
    let f32_tensor = IdxTensor::from_dense(vec![i.clone()], vec![1.5_f32, 2.5]).unwrap();
    let compatibility = f32_tensor
        .norm_squared()
        .expect("compatibility norm_squared");
    for squared in [true, false] {
        let value = context
            .with_concrete_session(|session| {
                if squared {
                    norm_squared_on_session(session, &f32_tensor)
                } else {
                    norm_on_session(session, &f32_tensor)
                }
            })
            .expect("session entry")
            .expect("the norms serve f32");
        let expected = if squared {
            compatibility
        } else {
            compatibility.sqrt()
        };
        assert_eq!(value, expected);
    }
}

#[test]
fn nan_inputs_and_unserved_operands_are_refused_typed_and_the_session_survives() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let dense =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let with_nan = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![1.0_f64, f64::NAN, 3.0, 4.0],
    )
    .unwrap();
    let diagonal = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0]).unwrap();
    let tracked = dense.clone().enable_grad().expect("tracked operand");
    // Compatibility runs outside any held session.
    let compatibility_nan = with_nan.norm_squared().unwrap_err();

    context
        .with_concrete_session(|session| {
            // NaN input: the same typed failure the compatibility route reports.
            let error =
                norm_squared_on_session(session, &with_nan).expect_err("a NaN element is refused");
            assert!(
                matches!(error, tensor4all_core::IdxTensorError::NaNInput { .. }),
                "{error:?}"
            );
            assert!(matches!(
                compatibility_nan,
                tensor4all_core::IdxTensorError::NaNInput { .. }
            ));

            for (label, tensor) in [("diagonal", &diagonal), ("tracked", &tracked)] {
                let error = norm_squared_on_session(session, tensor)
                    .expect_err("an unserved operand is refused");
                match error {
                    tensor4all_core::IdxTensorError::Operation { operation, .. } => {
                        assert_eq!(operation, "norm_squared_on_session", "{label}");
                    }
                    other => panic!("{label}: expected a typed operation refusal, got {other:?}"),
                }
            }

            // The session keeps working afterwards, and repeated calls agree.
            let first = norm_squared_on_session(session, &dense).expect("first");
            let second = norm_squared_on_session(session, &dense).expect("second");
            assert_eq!(first, second);
            assert_eq!(first, dense.norm_squared().expect("compatibility"));
            Ok::<(), tensor4all_core::IdxTensorError>(())
        })
        .expect("session entry")
        .expect("session work after refusals");
}
