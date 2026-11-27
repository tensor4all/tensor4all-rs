//! Trees that mix real and complex nodes (#1501): a real node whose child
//! message is complex must be computed in the promoted kind, whatever centre
//! and hint the batch uses.

use super::*;

/// The #1501 reproducer: the chain `0 - 1 - 2` with real nodes 0 and 1 and a
/// complex leaf 2.
fn mixed_chain() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let (s0, s1, s2) = (
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
    );
    let (b01, b12) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
    let t0 =
        IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let t1 = IdxTensor::from_dense(
        vec![b01, s1.clone(), b12.clone()],
        (0..12).map(|k| k as f64 * 0.5 - 1.0).collect(),
    )
    .unwrap();
    let t2 = IdxTensor::from_dense(
        vec![b12, s2.clone()],
        (0..6)
            .map(|k| Complex64::new(k as f64 - 2.0, 0.25 * k as f64))
            .collect(),
    )
    .unwrap();
    let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();
    (tree, vec![s0, s1, s2])
}

/// Every centre with and without a hint, cold and warm, against the uncached
/// `TreeTN::evaluate`. Before #1501 the rows `center 0` (with and without
/// `around(0)`) and `center 1, around(1)` failed with `Tensor::as_slice: dtype
/// mismatch: expected F64, actual C64`: they need the message `1 -> 0`, where
/// the real node 1 reads the complex child message `2 -> 1`.
#[test]
fn a_real_node_with_a_complex_child_message_is_promoted() {
    let (tree, indices) = mixed_chain();
    // Points (0, 1, 0) and (1, 0, 1), and all eight points once.
    let mut values = vec![0usize, 1, 0, 1, 0, 1];
    for code in 0..8usize {
        values.extend((0..3).map(|k| (code >> k) & 1));
    }
    let shape = [3, values.len() / 3];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    assert!(expected.iter().any(|value| value.imag().abs() > 1e-6));
    for center in [0usize, 1, 2] {
        for hint in [EvaluationHint::default(), EvaluationHint::around(center)] {
            let mut evaluator = TreeTNCachedEvaluator::new(
                &tree,
                &indices,
                CachedEvaluatorOptions {
                    center: Some(center),
                    ..Default::default()
                },
            )
            .unwrap();
            for pass in ["cold", "warm"] {
                let actual = evaluator
                    .evaluate_batched_typed::<Complex64>(points, hint.clone())
                    .unwrap_or_else(|error| {
                        panic!("center {center}, hint {:?}, {pass}: {error}", hint.center)
                    })
                    .into_iter()
                    .map(Complex64::into_any_scalar)
                    .collect::<Vec<_>>();
                assert_typed_results::<Complex64>(&actual, &expected);
            }
        }
    }
}
