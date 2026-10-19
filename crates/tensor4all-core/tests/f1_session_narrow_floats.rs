//! F1 slice 7: every session route serves `f32`/`c32`, and tensor publication keeps their dtype.
//!
//! The original rule was conservative *per-operand* admission for compact-storage publication: only a route
//! whose whole result is a plain `f64` - the norms - admitted the narrower floats. Compact storage now holds
//! `f32`/`Complex32` payloads, so publication is dtype preserving instead of refusing, and the tensor-publishing
//! routes serve the narrow floats in their own dtype rather than widening them into `f64`.

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
fn sum_serves_the_narrower_floats_in_their_own_dtype() {
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
        let compatibility = tensor.sum().expect("compatibility sum");
        let served = context
            .with_concrete_session(|session| sum_on_session(session, tensor))
            .expect("session entry")
            .expect("the scalar carrier holds the narrower floats too");
        assert_eq!(served.real(), compatibility.real(), "{label}");
        assert_eq!(served.imag(), compatibility.imag(), "{label}");
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
fn the_tensor_routes_publish_the_narrower_floats_in_their_own_dtype() {
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

    // References are taken before the session scope: the compatibility route enters the eager runtime,
    // which a held session does not allow.
    let f32_permuted = f32_tensor.permute(&[1, 0]).expect("compatibility permute");
    let c32_permuted = c32_tensor.permute(&[1, 0]).expect("compatibility permute");
    let f32_aligned = f32_tensor
        .permute_indices(&[k.clone(), i.clone()])
        .expect("compatibility alignment");

    context
        .with_concrete_session(|session| {
            for (label, tensor, permuted) in [
                ("f32", &f32_tensor, &f32_permuted),
                ("c32", &c32_tensor, &c32_permuted),
            ] {
                // Permutation publishes a narrow result without widening it.
                let served = permute_on_session(session, tensor, &[1, 0]).expect("session permute");
                assert_eq!(served.indices(), permuted.indices(), "{label}");
                assert_eq!(served.dims(), permuted.dims(), "{label}");
                assert_eq!(served.is_f32(), tensor.is_f32(), "{label}");
                assert_eq!(served.is_c32(), tensor.is_c32(), "{label}");

                // A nonidentity index alignment publishes the same narrow dtype.
                let aligned = permute_indices_on_session(session, tensor, &[k.clone(), i.clone()])
                    .expect("session alignment");
                assert_eq!(aligned.is_f32(), tensor.is_f32(), "{label}");
                assert_eq!(aligned.is_c32(), tensor.is_c32(), "{label}");

                // The scalar contraction route serves a narrow pair.
                contract_pair_on_session(
                    session,
                    tensor,
                    tensor,
                    PairwiseContractionOptions::new(),
                )
                .expect("narrow contraction");

                // The inner product accepts the narrow operand in either position.
                let wide =
                    IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
                        .unwrap();
                for (position, lhs, rhs) in [
                    ("narrow left", tensor, &wide),
                    ("narrow right", &wide, tensor),
                ] {
                    inner_product_on_session(session, lhs, rhs)
                        .unwrap_or_else(|error| panic!("{label}/{position}: {error:?}"));
                }
            }

            let aligned = permute_indices_on_session(session, &f32_tensor, &[k.clone(), i.clone()])
                .expect("session alignment");
            assert_eq!(aligned.indices(), f32_aligned.indices());

            // A route whose carrier holds `f64` still works after the narrow ones.
            let wide =
                IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
                    .unwrap();
            let value = sum_on_session(session, &wide).expect("session sum");
            assert_eq!(value.real(), 10.0);
            Ok::<(), tensor4all_core::IdxTensorError>(())
        })
        .expect("session entry")
        .expect("session work after the narrow routes");
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
