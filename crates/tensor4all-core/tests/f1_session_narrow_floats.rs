//! F1 slice 7: the norms serve `f32`/`c32`; every route that publishes into an `IdxTensor` refuses them.
//!
//! The implemented rule is conservative *per-operand* admission for compact-storage publication: only a route
//! whose whole result is a plain `f64` - the norms - admits the narrower floats, where they are promoted into
//! the same `f64` accumulation the compatibility route uses. A mixed narrow/wide pair could in principle have
//! produced a wide publishable result, and this slice deliberately does not admit it; the refusal says so.

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{
    contract_pair_on_session, inner_product_on_session, norm_on_session, norm_squared_on_session,
    permute_indices_on_session, permute_on_session, sum_on_session, IdxTensor,
    PairwiseContractionOptions,
};
use tensor4all_tensorbackend::CpuExecutionContext;

fn session_context() -> CpuExecutionContext {
    let backend = CpuBackend::with_threads(1).expect("1T CPU backend");
    CpuExecutionContext::from_backend(backend)
}

#[test]
fn sum_refuses_the_narrower_floats_because_its_carrier_cannot_hold_them() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);

    let f32_tensor =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.5_f32, -2.5, 3.0, 0.25]).unwrap();
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
        // Compatibility serves these, so the refusal is this seam's own carrier limit and not the input's.
        assert!(tensor.sum().is_ok(), "{label}");
        let error = context
            .with_concrete_session(|session| sum_on_session(session, tensor))
            .expect("session entry")
            .expect_err("the scalar carrier cannot hold a narrower float");
        match error {
            tensor4all_core::IdxTensorError::Operation { operation, .. } => {
                assert_eq!(operation, "sum_on_session", "{label}")
            }
            other => panic!("{label}: expected a typed operation refusal, got {other:?}"),
        }
    }
}

#[test]
fn norms_promote_the_narrower_floats_like_the_compatibility_route() {
    let context = session_context();
    let i = Index::new_dyn(2);

    let f32_tensor = IdxTensor::from_dense(vec![i.clone()], vec![3.0_f32, 4.0]).unwrap();
    let c32_tensor = IdxTensor::from_dense(
        vec![i.clone()],
        vec![Complex32::new(3.0, 4.0), Complex32::new(0.0, 0.0)],
    )
    .unwrap();

    // A fixture that discriminates the promotion from a narrow computation: `2^-80` squared is `2^-160`,
    // which is representable in `f64` but underflows to zero in `f32`.
    let tiny_f32 =
        IdxTensor::from_dense(vec![Index::new_dyn(1)], vec![f32::from_bits(0x2F80_0000)]).unwrap();
    let tiny_compatibility = tiny_f32
        .norm_squared()
        .expect("compatibility norm_squared of 2^-80");
    assert!(
        tiny_compatibility > 0.0,
        "the promoted accumulation keeps 2^-160 representable: {tiny_compatibility}"
    );
    assert_eq!(
        context
            .with_concrete_session(|session| norm_squared_on_session(session, &tiny_f32))
            .expect("session entry")
            .expect("session norm_squared"),
        tiny_compatibility,
        "a narrow computation would underflow to zero here"
    );

    for (label, tensor, expected_squared) in [
        ("f32", &f32_tensor, 25.0_f64),
        ("c32", &c32_tensor, 25.0_f64),
    ] {
        let compatibility = tensor.norm_squared().expect("compatibility norm_squared");
        assert!(
            (compatibility - expected_squared).abs() < 1e-5,
            "{label}: {compatibility}"
        );
        let actual = context
            .with_concrete_session(|session| norm_squared_on_session(session, tensor))
            .expect("session entry")
            .expect("session norm_squared");
        assert_eq!(actual, compatibility, "{label}");
        let actual_norm = context
            .with_concrete_session(|session| norm_on_session(session, tensor))
            .expect("session entry")
            .expect("session norm");
        assert_eq!(
            actual_norm,
            tensor.norm().expect("compatibility norm"),
            "{label}"
        );
    }
}

#[test]
fn the_tensor_routes_still_refuse_the_narrower_floats() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
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
    // Compatibility serves these, so the refusal is this seam's own dtype domain and not the input's.
    assert!(f32_tensor.permute(&[1, 0]).is_ok());

    context
        .with_concrete_session(|session| {
            for (label, tensor) in [("f32", &f32_tensor), ("c32", &c32_tensor)] {
                let error = permute_on_session(session, tensor, &[1, 0])
                    .expect_err("a tensor-publishing route refuses a narrower float");
                match error {
                    tensor4all_core::IdxTensorError::Operation { operation, source } => {
                        assert_eq!(operation, "permute_on_session", "{label}");
                        assert!(
                            source.to_string().contains("f64 and c64 operands"),
                            "{label}: {source}"
                        );
                    }
                    other => panic!("{label}: expected a typed operation refusal, got {other:?}"),
                }
                let error = contract_pair_on_session(
                    session,
                    tensor,
                    tensor,
                    PairwiseContractionOptions::new(),
                )
                .expect_err("contraction refuses it too");
                assert!(matches!(
                    error,
                    tensor4all_core::IdxTensorError::Operation { .. }
                ));
            }
            // The narrow operand is refused in either position of the inner product, and through a
            // nonidentity index alignment.
            let wide =
                IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
                    .unwrap();
            for (label, lhs, rhs) in [
                ("narrow left", &f32_tensor, &wide),
                ("narrow right", &wide, &f32_tensor),
            ] {
                let error = inner_product_on_session(session, lhs, rhs)
                    .expect_err("the inner product refuses a narrow operand");
                match error {
                    tensor4all_core::IdxTensorError::Operation { operation, source } => {
                        assert_eq!(operation, "contract_pair_on_session", "{label}");
                        assert!(
                            source.to_string().contains("f64 and c64 operands"),
                            "{label}"
                        );
                    }
                    other => panic!("{label}: expected a typed operation refusal, got {other:?}"),
                }
            }
            let error = permute_indices_on_session(session, &f32_tensor, &[k.clone(), i.clone()])
                .expect_err("a nonidentity alignment refuses a narrow operand");
            match error {
                tensor4all_core::IdxTensorError::Operation { operation, source } => {
                    assert_eq!(operation, "permute_on_session");
                    assert!(source.to_string().contains("f64 and c64 operands"));
                }
                other => panic!("expected a typed operation refusal, got {other:?}"),
            }

            // A route whose carrier holds `f64` still works after those refusals.
            let wide =
                IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
                    .unwrap();
            let value = sum_on_session(session, &wide).expect("session sum");
            assert_eq!(value.real(), 10.0);
            Ok::<(), tensor4all_core::IdxTensorError>(())
        })
        .expect("session entry")
        .expect("session work after refusals");
}

#[test]
fn complex64_still_takes_the_wide_path() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let tensor = IdxTensor::from_dense(
        vec![i.clone()],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
    )
    .unwrap();
    assert_eq!(
        context
            .with_concrete_session(|session| norm_squared_on_session(session, &tensor))
            .expect("session entry")
            .expect("session norm_squared"),
        tensor.norm_squared().expect("compatibility")
    );
}
