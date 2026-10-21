//! F1 slice 4: a nonidentity permutation on a caller-held explicit session.
//!
//! The session route must agree with `IdxTensor::permute` on values, layout and indices for the dense host
//! representations this slice serves, and must refuse what it does not serve instead of falling back.

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tensor4all_core::index::{DefaultIndex as Index, DynIndex};
use tensor4all_core::{permute_on_session, IdxTensor};
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_tensorbackend::{CpuExecutionContext, Storage};

fn session_context() -> CpuExecutionContext {
    let backend = CpuBackend::with_threads(1).expect("1T CPU backend");
    CpuExecutionContext::from_backend(backend)
}

/// An independent oracle: element `[new]` of the permuted tensor equals element `[old]`, with
/// `new[axis] = old[perm[axis]]`.
fn permuted_index(indices: &[DynIndex], perm: &[usize]) -> Vec<DynIndex> {
    perm.iter().map(|&axis| indices[axis].clone()).collect()
}

#[test]
fn permutation_matches_the_compatibility_route_and_an_independent_oracle() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let values: Vec<f64> = (1..=12).map(|value| value as f64 * 0.5 - 1.0).collect();

    let dense =
        IdxTensor::from_dense(vec![i.clone(), j.clone(), k.clone()], values.clone()).unwrap();
    let stored = IdxTensor::from_storage(
        vec![i.clone(), j.clone(), k.clone()],
        Arc::new(Storage::from_dense_col_major(values.clone(), &[2, 3, 2]).expect("storage")),
    )
    .unwrap();

    for (label, tensor) in [("dense", dense), ("from_storage", stored)] {
        for perm in [vec![2, 1, 0], vec![1, 2, 0], vec![0, 2, 1]] {
            let expected = tensor.permute(&perm).expect("compatibility permute");
            let actual = context
                .with_concrete_session(|session| permute_on_session(session, &tensor, &perm))
                .expect("session entry")
                .expect("session permute");

            assert_eq!(
                actual.indices(),
                permuted_index(tensor.indices(), &perm),
                "{label} {perm:?}: the index list"
            );
            assert_eq!(actual.indices(), expected.indices(), "{label} {perm:?}");
            assert_eq!(
                actual.to_vec::<f64>().unwrap(),
                expected.to_vec::<f64>().unwrap(),
                "{label} {perm:?}: the values"
            );

            // The oracle: reading the result at `new` must equal the input at the old position that `new`
            // came from. The compatibility route is checked against the same statement.
            let actual_values = actual.to_vec::<f64>().unwrap();
            let old_values = tensor.to_vec::<f64>().unwrap();
            let dims: Vec<usize> = tensor.dims();
            for (linear, actual_value) in actual_values.iter().enumerate() {
                // New-axis positions, with the new dims in the requested order.
                let mut position = vec![0usize; perm.len()];
                let mut rest = linear;
                for (axis, &old_axis) in perm.iter().enumerate() {
                    position[axis] = rest % dims[old_axis];
                    rest /= dims[old_axis];
                }
                // The same element in the old layout: new axis `i` is old axis `perm[i]`, and the old
                // strides are column-major over the *old* dimension order.
                let mut old_linear = 0usize;
                for (axis, &old_axis) in perm.iter().enumerate() {
                    let stride: usize = dims[..old_axis].iter().product();
                    old_linear += position[axis] * stride;
                }
                assert_eq!(
                    *actual_value, old_values[old_linear],
                    "{label} {perm:?}: element {linear}"
                );
            }
        }
    }
}

#[test]
fn identity_is_metadata_only_rank_one_and_complex_are_served() {
    let context = session_context();
    let i = Index::new_dyn(3);
    let k = Index::new_dyn(2);

    // Identity: the same tensor comes back, and no session work is needed for it.
    let vector = IdxTensor::from_dense(vec![i.clone()], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let same = context
        .with_concrete_session(|session| permute_on_session(session, &vector, &[0]))
        .expect("session entry")
        .expect("session permute");
    assert_eq!(same.indices(), vector.indices());
    assert_eq!(same.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);

    let rank_zero = IdxTensor::from_dense(Vec::new(), vec![7.0_f64]).unwrap();
    let zero = context
        .with_concrete_session(|session| permute_on_session(session, &rank_zero, &[]))
        .expect("session entry")
        .expect("session permute");
    assert_eq!(zero.to_vec::<f64>().unwrap(), vec![7.0]);

    // Complex values keep both components through the session route.
    let complex = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(2.0, 1.0),
            Complex64::new(-1.5, 0.25),
            Complex64::new(0.0, 2.0),
            Complex64::new(0.5, -2.0),
            Complex64::new(3.0, 0.0),
        ],
    )
    .unwrap();
    let expected = complex.permute(&[1, 0]).expect("compatibility");
    let actual = context
        .with_concrete_session(|session| permute_on_session(session, &complex, &[1, 0]))
        .expect("session entry")
        .expect("session permute");
    assert_eq!(actual.indices(), expected.indices());
    assert_eq!(
        actual.to_vec::<Complex64>().unwrap(),
        expected.to_vec::<Complex64>().unwrap()
    );
}

#[test]
fn malformed_permutations_and_unserved_operands_are_refused_typed() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let dense =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let diagonal = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0]).unwrap();
    let tracked = dense.clone().enable_grad().expect("tracked operand");
    // Compatibility runs outside any held session.
    let compatibility = dense.permute(&[1, 0]).expect("compatibility");

    context
        .with_concrete_session(|session| {
            for (perm, needle) in [
                (vec![0usize], "permutation length must match tensor rank"),
                (vec![0, 2], "permutation axis 2 out of range"),
                (vec![0, 0], "duplicate axis 0 in permutation"),
            ] {
                let error = permute_on_session(session, &dense, &perm)
                    .expect_err("a malformed permutation is refused");
                match error {
                    tensor4all_core::IdxTensorError::Operation { operation, source } => {
                        assert_eq!(operation, "permute_on_session");
                        assert!(source.to_string().contains(needle), "{perm:?}: {}", source);
                    }
                    other => panic!("expected a typed operation refusal, got {other:?}"),
                }
            }

            for (label, tensor) in [("diagonal", &diagonal), ("tracked", &tracked)] {
                let error = permute_on_session(session, tensor, &[1, 0])
                    .expect_err("an unserved operand is refused");
                match error {
                    tensor4all_core::IdxTensorError::Operation { operation, .. } => {
                        assert_eq!(operation, "permute_on_session", "{label}");
                    }
                    other => panic!("{label}: expected a typed operation refusal, got {other:?}"),
                }
                // The identity shortcut is metadata only, so the same operand succeeds unchanged.
                let unchanged = permute_on_session(session, tensor, &[0, 1])
                    .expect("an identity permutation is metadata only");
                assert_eq!(unchanged.indices(), tensor.indices(), "{label}");
            }

            // Several permutations inside one held region agree with the compatibility route.
            let first = permute_on_session(session, &dense, &[1, 0]).expect("first");
            let second = permute_on_session(session, &first, &[1, 0]).expect("second");
            assert_eq!(first.indices(), compatibility.indices());
            assert_eq!(second.indices(), dense.indices());
            assert_eq!(
                second.to_vec::<f64>().unwrap(),
                dense.to_vec::<f64>().unwrap()
            );
            Ok::<(), tensor4all_core::IdxTensorError>(())
        })
        .expect("session entry")
        .expect("session work after refusals");
}

#[test]
fn the_identity_shortcut_and_the_dtype_domain_are_permutation_specific() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);

    // Narrow dtypes used to be refused for a nonidentity permutation because publication could not hold
    // them; compact storage holds `f32`/`c32` now, so both the identity shortcut and the real permutation
    // serve them and keep their dtype.
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

    // The compatibility reference is taken outside the session scope, which a held session forbids.
    let f32_compatibility = f32_tensor.permute(&[1, 0]).expect("compatibility permute");
    let c32_compatibility = c32_tensor.permute(&[1, 0]).expect("compatibility permute");

    context
        .with_concrete_session(|session| {
            for (label, tensor, compatibility) in [
                ("f32", &f32_tensor, &f32_compatibility),
                ("c32", &c32_tensor, &c32_compatibility),
            ] {
                let permuted = permute_on_session(session, tensor, &[1, 0])
                    .expect("a narrow dtype is served for a nonidentity permutation");
                assert_eq!(permuted.indices(), compatibility.indices(), "{label}");
                assert_eq!(permuted.dims(), compatibility.dims(), "{label}");
                assert_eq!(permuted.is_f32(), tensor.is_f32(), "{label}");
                assert_eq!(permuted.is_c32(), tensor.is_c32(), "{label}");

                let identity = permute_on_session(session, tensor, &[0, 1])
                    .expect("an identity permutation is served whatever the dtype");
                assert_eq!(identity.indices(), tensor.indices(), "{label}");
            }
        })
        .expect("session entry");
}

#[test]
fn a_non_contiguous_eager_operand_is_materialized_before_the_requested_permutation() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let values: Vec<f64> = (1..=12).map(|value| value as f64 * 0.25).collect();
    let dense = IdxTensor::from_dense(vec![i.clone(), j.clone(), k.clone()], values).unwrap();

    // A compatibility permutation outside the session: an eager transposed view over the original payload.
    let transposed_view = dense.permute(&[2, 0, 1]).expect("compatibility permute");
    assert_eq!(transposed_view.indices().len(), 3);

    // Permuting that operand again on the session must materialize the view first and then transpose, and
    // must agree with the compatibility route applied to the same operand.
    let expected = transposed_view
        .permute(&[1, 2, 0])
        .expect("compatibility permute");
    let actual = context
        .with_concrete_session(|session| permute_on_session(session, &transposed_view, &[1, 2, 0]))
        .expect("session entry")
        .expect("session permute");
    assert_eq!(actual.indices(), expected.indices());
    assert_eq!(
        actual.to_vec::<f64>().unwrap(),
        expected.to_vec::<f64>().unwrap()
    );
    // The permutation is a relabelling of the same elements, so the multiset of values is unchanged.
    let mut actual_sorted = actual.to_vec::<f64>().unwrap();
    actual_sorted.sort_by(f64::total_cmp);
    let mut original_sorted = dense.to_vec::<f64>().unwrap();
    original_sorted.sort_by(f64::total_cmp);
    assert_eq!(actual_sorted, original_sorted);
}

#[test]
fn complex_permutation_has_an_explicit_expected_vector() {
    let context = session_context();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(3);
    let values = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(2.0, 1.0),
        Complex64::new(-1.5, 0.25),
        Complex64::new(0.0, 2.0),
        Complex64::new(0.5, -2.0),
        Complex64::new(3.0, 0.0),
    ];
    let tensor = IdxTensor::from_dense(vec![i.clone(), k.clone()], values.clone()).unwrap();

    // Column-major [i, k] with dims [2, 3] read as [k, i]: the transposed layout of those values.
    let expected = vec![
        values[0], values[2], values[4], values[1], values[3], values[5],
    ];
    let actual = context
        .with_concrete_session(|session| permute_on_session(session, &tensor, &[1, 0]))
        .expect("session entry")
        .expect("session permute");
    assert_eq!(actual.to_vec::<Complex64>().unwrap(), expected);
}
