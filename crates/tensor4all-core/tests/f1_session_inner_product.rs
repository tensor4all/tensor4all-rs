//! F1 slice 2: the inner product on a caller-held explicit session.
//!
//! The session route must agree with `IdxTensor::inner_product` on values for the supported dense host
//! operands, including the index-alignment and reduction cases, and must refuse what this slice does not
//! serve instead of falling back.

use std::sync::Arc;

use num_complex::Complex64;
use tensor4all_core::index::{DefaultIndex as Index, DynIndex};
use tensor4all_core::{inner_product_on_session, ExecutionContext, IdxTensor};
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_tensorbackend::CpuExecutionContext;

fn cpu_context() -> CpuExecutionContext {
    let backend = CpuBackend::with_threads(1).expect("1T CPU backend");
    CpuExecutionContext::from_backend(backend)
}

fn explicit_setup() -> (Arc<CpuExecutionContext>, ExecutionContext) {
    let context = Arc::new(cpu_context());
    let handle = ExecutionContext::Cpu(Arc::clone(&context));
    (context, handle)
}

/// An independent oracle for `conj(lhs) · rhs`, computed from the operands' own index orders.
///
/// It walks the joint index space - shared indices take one value, an index only one operand carries is
/// free and therefore summed - and never calls either frontend, so it cannot inherit a shared planner or
/// alignment defect. The parity tests below check both routes against it.
fn conj_inner_product_oracle(lhs: &IdxTensor, rhs: &IdxTensor) -> Complex64 {
    let lhs_indices = lhs.indices().to_vec();
    let rhs_indices = rhs.indices().to_vec();
    let lhs_values = lhs.to_vec::<f64>().unwrap();
    let rhs_values = rhs.to_vec::<f64>().unwrap();

    let mut union: Vec<_> = lhs_indices.clone();
    for index in &rhs_indices {
        if !union.contains(index) {
            union.push(index.clone());
        }
    }
    let dims: Vec<usize> = union.iter().map(|index| index.dim).collect();
    let total: usize = dims.iter().product::<usize>().max(1);

    let offset = |indices: &[DynIndex], values: &[f64], assignment: &[usize]| -> f64 {
        let mut flat = 0usize;
        let mut stride = 1usize;
        for index in indices {
            let position = union
                .iter()
                .position(|candidate| candidate == index)
                .expect("every operand index is in the union");
            flat += assignment[position] * stride;
            stride *= dims[position];
        }
        values[flat]
    };

    let mut sum = Complex64::new(0.0, 0.0);
    for linear in 0..total {
        let mut assignment = vec![0usize; union.len()];
        let mut rest = linear;
        for axis in 0..dims.len() {
            assignment[axis] = rest % dims[axis];
            rest /= dims[axis];
        }
        let left = offset(&lhs_indices, &lhs_values, &assignment);
        let right = offset(&rhs_indices, &rhs_values, &assignment);
        sum += Complex64::new(left, 0.0).conj() * Complex64::new(right, 0.0);
    }
    sum
}

fn assert_close(label: &str, actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1e-12,
        "{label}: {actual} != {expected}"
    );
}

#[test]
fn inner_product_matches_the_independent_oracle() {
    let (context, _handle) = explicit_setup();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(3);
    let j = Index::new_dyn(2);

    let lhs = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    // Same index set in the same order, the same set permuted, a subset (the left operand keeps a free
    // index), a disjoint pair, and two rank-zero operands.
    let cases: Vec<(&str, IdxTensor)> = vec![
        (
            "same order",
            IdxTensor::from_dense(
                vec![i.clone(), k.clone()],
                vec![1.0_f64, 0.0, 0.0, 1.0, 2.0, -1.0],
            )
            .unwrap(),
        ),
        (
            "permuted",
            IdxTensor::from_dense(
                vec![k.clone(), i.clone()],
                vec![1.0_f64, 0.0, 0.0, 1.0, 0.5, -1.0],
            )
            .unwrap(),
        ),
        (
            "subset",
            IdxTensor::from_dense(vec![k.clone()], vec![1.0_f64, -1.0, 2.0]).unwrap(),
        ),
        (
            "disjoint",
            IdxTensor::from_dense(vec![j.clone()], vec![2.0_f64, -0.5]).unwrap(),
        ),
        (
            "rank zero",
            IdxTensor::from_dense(Vec::new(), vec![2.5_f64]).unwrap(),
        ),
    ];

    for (label, rhs) in cases {
        let expected = if label == "rank zero" {
            let left = IdxTensor::from_dense(Vec::new(), vec![1.5_f64]).unwrap();
            let oracle = conj_inner_product_oracle(&left, &rhs);
            let actual = context
                .with_concrete_session(|session| inner_product_on_session(session, &left, &rhs))
                .expect("session entry")
                .expect("session inner product");
            assert_close(label, Complex64::new(actual.real(), actual.imag()), oracle);
            continue;
        } else {
            conj_inner_product_oracle(&lhs, &rhs)
        };

        let actual = context
            .with_concrete_session(|session| inner_product_on_session(session, &lhs, &rhs))
            .expect("session entry")
            .expect("session inner product");
        assert_close(
            label,
            Complex64::new(actual.real(), actual.imag()),
            expected,
        );

        // The compatibility route agrees with the same independent oracle, which is what makes this a
        // parity check rather than two implementations sharing a bug.
        let compatibility = lhs
            .inner_product(&rhs)
            .expect("compatibility inner product");
        assert_close(
            label,
            Complex64::new(compatibility.real(), compatibility.imag()),
            expected,
        );
    }
}

#[test]
fn a_cyclic_permutation_aligns_the_right_operand() {
    let (context, _handle) = explicit_setup();
    let a = Index::new_dyn(2);
    let b = Index::new_dyn(3);
    let c = Index::new_dyn(2);

    let lhs_data: Vec<f64> = (0..12).map(|index| index as f64 * 0.5 - 1.0).collect();
    let rhs_data: Vec<f64> = (0..12).map(|index| 2.0 - index as f64 * 0.25).collect();
    let lhs = IdxTensor::from_dense(vec![a.clone(), b.clone(), c.clone()], lhs_data).unwrap();
    // A three-cycle, not a swap: a wrong permutation direction would pass a transposition but not this.
    let rhs = IdxTensor::from_dense(vec![c.clone(), a.clone(), b.clone()], rhs_data).unwrap();

    let expected = conj_inner_product_oracle(&lhs, &rhs);
    let actual = context
        .with_concrete_session(|session| inner_product_on_session(session, &lhs, &rhs))
        .expect("session entry")
        .expect("session inner product");
    assert_close(
        "cyclic",
        Complex64::new(actual.real(), actual.imag()),
        expected,
    );
}

#[test]
fn a_complex_oracle_makes_the_left_conjugation_observable() {
    let (context, _handle) = explicit_setup();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let lhs_real: Vec<Complex64> = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(0.5, 0.25),
        Complex64::new(-1.0, 2.0),
    ];
    let rhs_real: Vec<Complex64> = vec![
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.5),
        Complex64::new(2.0, -2.0),
    ];
    let lhs = IdxTensor::from_dense(vec![i.clone(), k.clone()], lhs_real.clone()).unwrap();
    let rhs = IdxTensor::from_dense(vec![k.clone(), i.clone()], rhs_real.clone()).unwrap();

    // Independent oracle: conj(lhs) paired with rhs by index, with rhs permuted into lhs order.
    let mut expected = Complex64::new(0.0, 0.0);
    for k_axis in 0..2usize {
        for i_axis in 0..2usize {
            // lhs carries [i, k] and rhs carries [k, i], both column-major with their first mode fastest.
            expected += lhs_real[i_axis + 2 * k_axis].conj() * rhs_real[k_axis + 2 * i_axis];
        }
    }
    let actual = context
        .with_concrete_session(|session| inner_product_on_session(session, &lhs, &rhs))
        .expect("session entry")
        .expect("session inner product");
    assert_close(
        "complex conjugation",
        Complex64::new(actual.real(), actual.imag()),
        expected,
    );

    let compatibility = lhs
        .inner_product(&rhs)
        .expect("compatibility inner product");
    assert_close(
        "complex conjugation (compatibility)",
        Complex64::new(compatibility.real(), compatibility.imag()),
        expected,
    );
}

#[test]
fn right_hand_operands_are_refused_before_any_session_work() {
    let (context, handle) = explicit_setup();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let dense =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let diagonal = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0]).unwrap();
    // A lazily initialized eager left operand is refused-by-proxy only if adaptation happens before it is
    // materialized: this one would do session work if the order were wrong.
    let eager = IdxTensor::from_dense_in(
        &handle,
        vec![i.clone(), k.clone()],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .expect("context-scoped construction");
    let tracked = dense.clone().enable_grad().expect("tracked operand");

    context
        .with_concrete_session(|session| {
            for (label, rhs) in [("diagonal", &diagonal), ("tracked", &tracked)] {
                let error = inner_product_on_session(session, &eager, rhs)
                    .expect_err("the right operand must be refused");
                match error {
                    tensor4all_core::IdxTensorError::Operation { operation, .. } => assert_eq!(
                        operation, "contract_pair_on_session",
                        "{label}: the refusal names the shared operation"
                    ),
                    other => panic!("{label}: expected a typed operation refusal, got {other:?}"),
                }
            }
            let value = inner_product_on_session(session, &eager, &dense).expect("inner product");
            assert!((value.real() - 30.0).abs() < 1e-12, "got {}", value.real());
            Ok::<(), tensor4all_core::IdxTensorError>(())
        })
        .expect("session entry")
        .expect("session work after refusals");
}
