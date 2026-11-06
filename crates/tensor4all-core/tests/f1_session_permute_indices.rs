//! F1 slice 5: aligning a tensor to a caller-supplied index order on a held explicit session.
//!
//! The session route must agree with `IdxTensor::permute_indices` on indices, values and layout for the
//! dense host representations it serves, and must refuse what it does not serve instead of falling back.

use num_complex::Complex64;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{permute_indices_on_session, IdxTensor};
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_tensorbackend::CpuExecutionContext;

fn session_context() -> CpuExecutionContext {
    let backend = CpuBackend::with_threads(1).expect("1T CPU backend");
    CpuExecutionContext::from_backend(backend)
}

#[test]
fn aligning_to_an_index_order_matches_the_compatibility_route() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let values: Vec<f64> = (1..=12).map(|value| value as f64 * 0.25 - 1.0).collect();
    let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone(), k.clone()], values).unwrap();

    // A genuine alignment (the order apps use before a contraction), the identity order, and a two-axis
    // swap that leaves the third in place.
    let cases: Vec<Vec<tensor4all_core::index::DynIndex>> = vec![
        vec![j.clone(), k.clone(), i.clone()],
        vec![i.clone(), j.clone(), k.clone()],
        vec![k.clone(), j.clone(), i.clone()],
    ];

    for new_indices in cases {
        let expected = tensor
            .permute_indices(&new_indices)
            .expect("compatibility permute_indices");
        let actual = context
            .with_concrete_session(|session| {
                permute_indices_on_session(session, &tensor, &new_indices)
            })
            .expect("session entry")
            .expect("session alignment");

        assert_eq!(actual.indices(), expected.indices(), "{new_indices:?}");
        assert_eq!(
            actual.to_vec::<f64>().unwrap(),
            expected.to_vec::<f64>().unwrap(),
            "{new_indices:?}"
        );
        // Alignment reorders the axes, so the per-axis dimensions are the permuted ones; the multiset is
        // what has to stay.
        assert_eq!(actual.dims(), expected.dims(), "{new_indices:?}");
        let mut actual_dims = actual.dims();
        let mut original_dims = tensor.dims();
        actual_dims.sort_unstable();
        original_dims.sort_unstable();
        assert_eq!(actual_dims, original_dims, "{new_indices:?}");
    }
}

#[test]
fn an_identity_order_is_metadata_only_for_any_representation() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let order = vec![i.clone(), k.clone()];

    let dense =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let diagonal = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0]).unwrap();

    for (label, tensor) in [("dense", &dense), ("diagonal", &diagonal)] {
        let expected = tensor
            .permute_indices(&order)
            .expect("compatibility permute_indices");
        let actual = context
            .with_concrete_session(|session| permute_indices_on_session(session, tensor, &order))
            .expect("session entry")
            .expect("session alignment");
        assert_eq!(actual.indices(), expected.indices(), "{label}");
        assert_eq!(
            actual.to_vec::<f64>().unwrap(),
            expected.to_vec::<f64>().unwrap(),
            "{label}"
        );
        // The identity case keeps the representation: a diagonal payload stays diagonal.
        assert_eq!(actual.is_diag(), tensor.is_diag(), "{label}");
    }
}

#[test]
fn complex_alignment_and_refusals_are_typed() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let values = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(2.0, 1.0),
        Complex64::new(-1.5, 0.25),
        Complex64::new(0.0, 2.0),
    ];
    let tensor = IdxTensor::from_dense(vec![i.clone(), k.clone()], values.clone()).unwrap();
    let diagonal = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0]).unwrap();
    let tracked = tensor.clone().enable_grad().expect("tracked operand");
    let foreign = Index::new_dyn(5);
    // Compatibility runs outside any held session.
    let compatibility = tensor
        .permute_indices(&[k.clone(), i.clone()])
        .expect("compatibility")
        .to_vec::<Complex64>()
        .unwrap();

    context
        .with_concrete_session(|session| {
            let aligned = permute_indices_on_session(session, &tensor, &[k.clone(), i.clone()])
                .expect("session alignment");
            assert_eq!(aligned.to_vec::<Complex64>().unwrap(), compatibility);

            // An index list that is not a permutation of the tensor's own indices.
            let error = permute_indices_on_session(session, &tensor, &[i.clone(), foreign.clone()])
                .expect_err("a foreign index is refused");
            match error {
                tensor4all_core::IdxTensorError::Operation { operation, .. } => {
                    assert_eq!(operation, "permute_indices_on_session");
                }
                other => panic!("expected a typed operation refusal, got {other:?}"),
            }

            // A nonidentity alignment of a representation this seam does not serve.
            let error = permute_indices_on_session(session, &diagonal, &[k.clone(), i.clone()])
                .expect_err("a diagonal payload is refused for a nonidentity alignment");
            match error {
                tensor4all_core::IdxTensorError::Operation { operation, .. } => {
                    assert_eq!(operation, "permute_on_session");
                }
                other => panic!("expected a typed operation refusal, got {other:?}"),
            }
            let error = permute_indices_on_session(session, &tracked, &[k.clone(), i.clone()])
                .expect_err("a tracked operand is refused for a nonidentity alignment");
            match error {
                tensor4all_core::IdxTensorError::Operation { operation, .. } => {
                    assert_eq!(operation, "permute_on_session");
                }
                other => panic!("expected a typed operation refusal, got {other:?}"),
            }
            Ok::<(), tensor4all_core::IdxTensorError>(())
        })
        .expect("session entry")
        .expect("session work after refusals");
}

#[test]
fn a_supplied_dimension_that_disagrees_with_the_tensor_is_refused() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(3);
    let tensor = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    // Index equality compares id, tags and prime level, not the dimension, so a supplied list can carry a
    // different shape. Refusing it is deliberately stricter than the compatibility identity order, which
    // accepts such metadata.
    let mut widened = k.clone();
    widened.dim = 4;
    let order = vec![widened.clone(), i.clone()];

    let error = context
        .with_concrete_session(|session| permute_indices_on_session(session, &tensor, &order))
        .expect("session entry")
        .expect_err("a mismatched dimension is refused");
    match error {
        tensor4all_core::IdxTensorError::Operation { operation, source } => {
            assert_eq!(operation, "permute_indices_on_session");
            assert!(
                source.to_string().contains("dimension 4"),
                "the refusal names the supplied dimension: {source}"
            );
        }
        other => panic!("expected a typed operation refusal, got {other:?}"),
    }

    // The identity order with the same mismatched metadata is refused here as well.
    let error = context
        .with_concrete_session(|session| permute_indices_on_session(session, &tensor, &order))
        .expect("session entry")
        .expect_err("identity order with mismatched metadata");
    assert!(matches!(
        error,
        tensor4all_core::IdxTensorError::Operation { .. }
    ));
}

#[test]
fn alignment_refusal_reasons_survive_and_the_session_keeps_working() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(3);
    let other = Index::new_dyn(2);
    let values: Vec<f64> = (1..=6).map(|value| value as f64).collect();
    let tensor = IdxTensor::from_dense(vec![i.clone(), k.clone()], values.clone()).unwrap();

    // Same id as `i` but a different prime level, so the sets differ by full index equality.
    let mut primed = i.clone();
    primed.plev = 1;

    context
        .with_concrete_session(|session| {
            // A list whose length differs, a foreign index, and a same-id index with a different prime:
            // each keeps the underlying reason of the compatibility helper.
            for (order, needle) in [
                (vec![i.clone()], "length"),
                (vec![other.clone(), k.clone()], "permutation"),
                (vec![primed.clone(), k.clone()], "permutation"),
            ] {
                let error = permute_indices_on_session(session, &tensor, &order)
                    .expect_err("an invalid index list is refused");
                match error {
                    tensor4all_core::IdxTensorError::Operation { operation, source } => {
                        assert_eq!(operation, "permute_indices_on_session");
                        assert!(source.to_string().contains(needle), "{order:?}: {source}");
                    }
                    other => panic!("expected a typed operation refusal, got {other:?}"),
                }
            }

            // The session is still usable for numerical work after those refusals: an explicit expected
            // alignment, independent of the compatibility route.
            let aligned = permute_indices_on_session(session, &tensor, &[k.clone(), i.clone()])
                .expect("session alignment after refusals");
            assert_eq!(aligned.indices(), [k.clone(), i.clone()]);
            assert_eq!(aligned.dims(), vec![3, 2]);
            // Column-major [i, k] read as [k, i]: the transposed layout of 1..=6.
            assert_eq!(
                aligned.to_vec::<f64>().unwrap(),
                vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
            );
            Ok::<(), tensor4all_core::IdxTensorError>(())
        })
        .expect("session entry")
        .expect("session work after refusals");
}
