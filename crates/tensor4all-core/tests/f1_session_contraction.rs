//! F1 slice 1: pairwise contraction on a caller-held explicit session.
//!
//! The entry point must agree with the compatibility route on values *and* indices, must serve both dense
//! host representations, and must refuse what this slice does not serve instead of falling back.

use std::sync::Arc;

use num_complex::Complex64;

use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{
    contract_pair, contract_pair_on_session, ExecutionContext, IdxTensor,
    PairwiseContractionOptions,
};
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_tensorbackend::{CpuExecutionContext, Storage};

fn cpu_context() -> CpuExecutionContext {
    let backend = CpuBackend::with_threads(1).expect("1T CPU backend");
    CpuExecutionContext::from_backend(backend)
}

/// One-thread explicit context plus the `ExecutionContext` handle the constructors take.
fn explicit_setup() -> (Arc<CpuExecutionContext>, ExecutionContext) {
    let context = Arc::new(cpu_context());
    let handle = ExecutionContext::Cpu(Arc::clone(&context));
    (context, handle)
}

/// The same contraction through the compatibility route, as the parity oracle.
fn compatibility(lhs: &IdxTensor, rhs: &IdxTensor) -> IdxTensor {
    contract_pair(lhs, rhs).expect("compatibility contraction")
}

#[test]
fn session_contraction_matches_the_compatibility_route() {
    let (context, _handle) = explicit_setup();
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);

    // Contracted, outer product (no shared index) and a contraction over every index.
    let cases: Vec<(IdxTensor, IdxTensor)> = vec![
        (
            IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
                .unwrap(),
            IdxTensor::from_dense(
                vec![k.clone(), j.clone()],
                vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .unwrap(),
        ),
        (
            IdxTensor::from_dense(vec![i.clone()], vec![3.0_f64, 4.0]).unwrap(),
            IdxTensor::from_dense(vec![j.clone()], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ),
        (
            IdxTensor::from_dense(vec![i.clone()], vec![3.0_f64, 4.0]).unwrap(),
            IdxTensor::from_dense(vec![i.clone()], vec![5.0_f64, 6.0]).unwrap(),
        ),
    ];

    for (lhs, rhs) in cases {
        let expected = compatibility(&lhs, &rhs);
        let actual = context
            .with_concrete_session(|session| {
                contract_pair_on_session(session, &lhs, &rhs, PairwiseContractionOptions::new())
            })
            .expect("session entry")
            .expect("session contraction");

        assert_eq!(
            actual.indices(),
            expected.indices(),
            "index lists (id, prime level and tags) must match the compatibility route"
        );
        let actual_values = actual.to_vec::<f64>().unwrap();
        let expected_values = expected.to_vec::<f64>().unwrap();
        for (a, e) in actual_values.iter().zip(expected_values.iter()) {
            assert!(
                (a - e).abs() < 1e-12,
                "session route {actual_values:?} must match compatibility {expected_values:?}"
            );
        }
    }
}

#[test]
fn session_contraction_result_order_is_the_planners() {
    // lhs [a, k, b] and rhs [c, k, d] with unequal dims: the result must be [a, b, c, d], which the
    // expected values below compute independently of both routes.
    let (context, _handle) = explicit_setup();
    let a = Index::new_dyn(2);
    let k = Index::new_dyn(3);
    let b = Index::new_dyn(2);
    let c = Index::new_dyn(2);
    let d = Index::new_dyn(2);

    let lhs_data: Vec<f64> = (0..12).map(|index| index as f64 + 1.0).collect();
    let rhs_data: Vec<f64> = (0..12).map(|index| (index as f64) * 0.5 - 1.0).collect();
    let lhs =
        IdxTensor::from_dense(vec![a.clone(), k.clone(), b.clone()], lhs_data.clone()).unwrap();
    let rhs =
        IdxTensor::from_dense(vec![c.clone(), k.clone(), d.clone()], rhs_data.clone()).unwrap();

    let actual = context
        .with_concrete_session(|session| {
            contract_pair_on_session(session, &lhs, &rhs, PairwiseContractionOptions::new())
        })
        .expect("session entry")
        .expect("session contraction");

    let flat = |dims: &[usize], indices: &[usize]| {
        let mut offset = 0usize;
        for (axis, &index) in indices.iter().enumerate() {
            offset += index * dims[..axis].iter().product::<usize>();
        }
        offset
    };
    let lhs_dims = [2usize, 3, 2]; // a, k, b
    let rhs_dims = [2usize, 3, 2]; // c, k, d
    let out_dims = [2usize, 2, 2, 2]; // a, b, c, d
    let mut expected = vec![0.0_f64; out_dims.iter().product::<usize>()];
    for a_axis in 0..2 {
        for b_axis in 0..2 {
            for c_axis in 0..2 {
                for d_axis in 0..2 {
                    let mut sum = 0.0;
                    for k_axis in 0..3 {
                        sum += lhs_data[flat(&lhs_dims, &[a_axis, k_axis, b_axis])]
                            * rhs_data[flat(&rhs_dims, &[c_axis, k_axis, d_axis])];
                    }
                    expected[flat(&out_dims, &[a_axis, b_axis, c_axis, d_axis])] = sum;
                }
            }
        }
    }

    let actual_indices = actual.indices().to_vec();
    assert_eq!(
        actual_indices,
        vec![a.clone(), b.clone(), c.clone(), d.clone()]
    );
    let actual_values = actual.to_vec::<f64>().unwrap();
    for (index, (a_value, e_value)) in actual_values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a_value - e_value).abs() < 1e-12,
            "result element {index}: {a_value} != {e_value} (full: {actual_values:?})"
        );
    }
}

#[test]
fn chained_contractions_share_one_held_session() {
    let (context, _handle) = explicit_setup();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    let lhs =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs =
        IdxTensor::from_dense(vec![k.clone(), j.clone()], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
    let expected_once = compatibility(&lhs, &rhs);
    let expected_twice = compatibility(&expected_once, &rhs);

    let twice = context
        .with_concrete_session(|session| {
            let once =
                contract_pair_on_session(session, &lhs, &rhs, PairwiseContractionOptions::new())?;
            contract_pair_on_session(session, &once, &rhs, PairwiseContractionOptions::new())
        })
        .expect("session entry")
        .expect("session contraction");

    assert_eq!(twice.indices(), expected_twice.indices());
    let actual = twice.to_vec::<f64>().unwrap();
    let expected = expected_twice.to_vec::<f64>().unwrap();
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-12, "{actual:?} != {expected:?}");
    }
}

#[test]
fn eager_and_permuted_host_operands_are_served() {
    let (context, handle) = explicit_setup();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let lhs_dense =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = IdxTensor::from_dense(
        vec![k.clone(), j.clone()],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    // Eager operand: an explicit context's own untracked value.
    let lhs_eager = IdxTensor::from_dense_in(
        &handle,
        vec![i.clone(), k.clone()],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .expect("context-scoped construction");
    // Non-contiguous operand: a nonidentity permutation materializes an eager transpose.
    let lhs_permuted = lhs_dense.permute(&[1, 0]).expect("permute");

    let expected = compatibility(&lhs_dense, &rhs).to_vec::<f64>().unwrap();
    for (label, lhs) in [("eager", &lhs_eager), ("permuted", &lhs_permuted)] {
        let actual = context
            .with_concrete_session(|session| {
                contract_pair_on_session(session, lhs, &rhs, PairwiseContractionOptions::new())
            })
            .expect("session entry")
            .expect("session contraction");
        let values = actual.to_vec::<f64>().unwrap();
        assert_eq!(values, expected, "operand representation: {label}");
    }
}

#[test]
fn structured_and_tracked_operands_are_refused_by_name() {
    let (context, _handle) = explicit_setup();
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let dense =
        IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let diagonal = IdxTensor::from_diag(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0]).unwrap();

    let refusal = context
        .with_concrete_session(|session| {
            contract_pair_on_session(
                session,
                &dense,
                &diagonal,
                PairwiseContractionOptions::new(),
            )
        })
        .expect("session entry")
        .expect_err("a diagonal payload is not served by this slice");
    let message = refusal.to_string();
    assert!(
        message.contains("contract_pair_on_session")
            && message.contains("not served by this slice"),
        "refusal must name the operation and the reason, got: {message}"
    );

    let tracked = dense.clone().enable_grad().expect("tracked operand");
    let refusal = context
        .with_concrete_session(|session| {
            contract_pair_on_session(session, &tracked, &dense, PairwiseContractionOptions::new())
        })
        .expect("session entry")
        .expect_err("a tracked value must not enter the plain route");
    let message = refusal.to_string();
    assert!(
        message.contains("contract_pair_on_session") && message.contains("tracked"),
        "refusal must name the operation and the reason, got: {message}"
    );

    // The session stays usable after a refusal.
    let value = context
        .with_concrete_session(|session| {
            contract_pair_on_session(session, &dense, &dense, PairwiseContractionOptions::new())
        })
        .expect("session entry")
        .expect("session contraction after a refusal");
    // Contracting the tensor with itself contracts every index, so the result is a scalar.
    assert_eq!(value.indices().len(), 0);
}

#[test]
fn the_entry_point_and_its_helpers_contain_no_default_context_entry() {
    // A direct-source guard, not a transitive zero-entry proof: it pins the text of the session module -
    // which carries both entry points - and of the operand helper in the crate's own module, against the
    // symbols that would mean reaching the default or compatibility path. It deliberately also bans the
    // eager-adopting scalar wrapper, because publication is as much a boundary as adaptation. Identity comparison cannot do this:
    // `is_global_default_cpu` itself initializes the process-global eager runtime, and it stays false
    // even when an adaptation secretly used the default runtime.
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let session_source = std::fs::read_to_string(manifest.join("src/defaults/contract_session.rs"))
        .expect("session module");
    let tensor_source = std::fs::read_to_string(manifest.join("src/defaults/idx_tensor.rs"))
        .expect("idx_tensor module");
    let operand_helper = tensor_source
        .split("pub(crate) fn session_dense_operand(")
        .nth(1)
        .expect("the operand helper is present")
        .split("/// Whether every index of this tensor is distinct")
        .next()
        .expect("the operand helper ends before the uniqueness witness");

    let banned = [
        "try_materialized_inner",
        "default_eager_ctx",
        "default_cpu_execution_context",
        "with_default_session",
        "explicit::detach",
        ".storage()",
        "ensure_storage_ready",
        // Scalar publication must take the value from the concrete result: `AnyScalar::from_tensor`
        // extracts it through the tensor's eager representation, which adopts a freshly materialized
        // value into the process-global eager runtime.
        "AnyScalar::from_tensor",
    ];
    let mut violations = Vec::new();
    for (name, source) in [
        ("defaults/contract_session.rs", session_source.as_str()),
        ("IdxTensor::session_dense_operand", operand_helper),
    ] {
        for needle in banned {
            if source.contains(needle) {
                violations.push(format!("{name} contains {needle}"));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "the session route must not reach the default or compatibility path: {violations:?}"
    );
}

#[test]
fn complex_and_mixed_dtype_operands_promote_like_the_compatibility_route() {
    let (context, _handle) = explicit_setup();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    let real =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let imag = IdxTensor::from_dense(
        vec![k.clone(), j.clone()],
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(0.0, -1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(-1.0, 1.0),
        ],
    )
    .unwrap();
    let real_rhs =
        IdxTensor::from_dense(vec![k.clone(), j.clone()], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();

    // A mixed f64/c64 pair: the session promotes on its own route, exactly as compatibility does.
    for (lhs, rhs, label) in [
        (&imag, &real_rhs, "c64 left"),
        (&real, &imag, "f64 left, c64 right"),
    ] {
        let expected = compatibility(lhs, rhs);
        let actual = context
            .with_concrete_session(|session| {
                contract_pair_on_session(session, lhs, rhs, PairwiseContractionOptions::new())
            })
            .expect("session entry")
            .expect("session contraction");
        assert_eq!(actual.indices(), expected.indices(), "{label}");
        let actual_values = actual.to_vec::<Complex64>().unwrap();
        let expected_values = expected.to_vec::<Complex64>().unwrap();
        for (a, e) in actual_values.iter().zip(expected_values.iter()) {
            assert!(
                (a - e).norm() < 1e-12,
                "{label}: {actual_values:?} != {expected_values:?}"
            );
        }
    }
}

#[test]
fn conjugation_options_match_the_compatibility_route() {
    let (context, _handle) = explicit_setup();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let lhs = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(-1.0, 2.0),
        ],
    )
    .unwrap();
    let rhs = IdxTensor::from_dense(
        vec![k.clone(), i.clone()],
        vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(2.0, -2.0),
        ],
    )
    .unwrap();

    for options in [
        PairwiseContractionOptions::new(),
        PairwiseContractionOptions::new().with_lhs_conj(true),
        PairwiseContractionOptions::new().with_rhs_conj(true),
        PairwiseContractionOptions::new()
            .with_lhs_conj(true)
            .with_rhs_conj(true),
    ] {
        let expected = tensor4all_core::contract_pair_with_operand_options(&lhs, &rhs, options)
            .expect("compatibility contraction");
        let actual = context
            .with_concrete_session(|session| contract_pair_on_session(session, &lhs, &rhs, options))
            .expect("session entry")
            .expect("session contraction");
        assert_eq!(actual.indices(), expected.indices(), "{options:?}");
        let actual_values = actual.to_vec::<Complex64>().unwrap();
        let expected_values = expected.to_vec::<Complex64>().unwrap();
        for (a, e) in actual_values.iter().zip(expected_values.iter()) {
            assert!(
                (a - e).norm() < 1e-12,
                "{options:?}: {actual_values:?} != {expected_values:?}"
            );
        }
    }
}

#[test]
fn a_repeated_index_is_refused_before_planning() {
    let (context, _handle) = explicit_setup();
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);

    // `replace_indices` can fold two indices into one through public API; the planner's result-length
    // arithmetic would underflow on it, so the route refuses it before planning.
    let doubled = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap()
        .replace_indices(std::slice::from_ref(&j), std::slice::from_ref(&i))
        .expect("index replacement");
    let duplicated = doubled.indices();
    assert_eq!(duplicated.len(), 2);
    assert_eq!(duplicated[0], duplicated[1], "the fixture repeats an index");

    let rhs =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
    let refusal = context
        .with_concrete_session(|session| {
            contract_pair_on_session(session, &doubled, &rhs, PairwiseContractionOptions::new())
        })
        .expect("session entry")
        .expect_err("a repeated index must be refused");
    let message = refusal.to_string();
    assert!(
        message.contains("contract_pair_on_session") && message.contains("distinct"),
        "refusal must name the operation and the reason, got: {message}"
    );
}

#[test]
fn fresh_storage_operands_and_same_session_recovery_are_served() {
    let (context, _handle) = explicit_setup();
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    // A fresh `from_storage` operand: materialized host storage that never went through an eager value.
    let lhs_storage = Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let lhs = IdxTensor::from_storage(vec![i.clone(), k.clone()], Arc::new(lhs_storage)).unwrap();
    let rhs = IdxTensor::from_dense(
        vec![k.clone(), j.clone()],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let expected = compatibility(&lhs, &rhs).to_vec::<f64>().unwrap();

    let diagonal = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0]).unwrap();

    context
        .with_concrete_session(|session| {
            // A refusal must leave the held session usable for the next call.
            contract_pair_on_session(session, &diagonal, &rhs, PairwiseContractionOptions::new())
                .expect_err("diagonal refused");
            let actual =
                contract_pair_on_session(session, &lhs, &rhs, PairwiseContractionOptions::new())?;
            assert_eq!(actual.to_vec::<f64>().unwrap(), expected);
            Ok::<(), tensor4all_core::IdxTensorError>(())
        })
        .expect("session entry")
        .expect("session contraction after a refusal");
}
