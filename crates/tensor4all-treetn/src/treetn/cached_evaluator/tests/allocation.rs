//! Allocation budgets of the typed batch and warm-assignment paths, measured with
//! the test-only counting allocator.

use super::*;

/// Builds a chain of `n_sites` rank-3 cores with the requested bond
/// dimension, matching the Guard's 16-site floating-zone fixture shape.
fn allocation_fixture_chain(
    n_sites: usize,
    bond_dim: usize,
) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use tensor4all_simplett::{tensor3_zeros, SimpleTensorTrain, Tensor3, Tensor3Ops};

    const LOCAL_DIM: usize = 2;
    let mut rng = ChaCha8Rng::seed_from_u64(709);
    let mut tensors: Vec<Tensor3<f64>> = Vec::with_capacity(n_sites);
    for site in 0..n_sites {
        let left_dim = if site == 0 { 1 } else { bond_dim };
        let right_dim = if site == n_sites - 1 { 1 } else { bond_dim };
        let mut tensor = tensor3_zeros(left_dim, LOCAL_DIM, right_dim);
        for left in 0..left_dim {
            for local in 0..LOCAL_DIM {
                for right in 0..right_dim {
                    tensor.set3(left, local, right, rng.random::<f64>());
                }
            }
        }
        tensors.push(tensor);
    }
    let train = SimpleTensorTrain::new(tensors).unwrap();
    crate::tensor_train_to_treetn(&train).unwrap()
}

/// [AI Supplied] #709 resource gate. Two claims are measured with the
/// counting allocator rather than asserted from the diff:
///
/// 1. the typed batch route allocates at least one fewer heap block per
///    result than the `AnyScalar` wrapper, because it constructs no
///    rank-zero tensor per result;
/// 2. a warm 16-site call stays far below the audited baseline of at
///    least 3,072 short-lived assignment vectors per 16-site/64-point
///    call recorded in
///    `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`.
///
/// Both evaluators are fully warm, so the measured difference is the
/// result boundary and the per-call assignment work, not cache misses.
#[test]
fn typed_batch_and_warm_assignment_allocations_stay_below_the_audited_baseline() {
    const N_SITES: usize = 16;
    const N_POINTS: usize = 64;
    const AUDITED_ASSIGNMENT_VECTORS: u64 = 3_072;

    let (tree, indices) = allocation_fixture_chain(N_SITES, 16);
    let varying = N_SITES / 2;
    let mut values = vec![0usize; N_SITES * N_POINTS];
    for point in 0..N_POINTS {
        values[varying + N_SITES * point] = point % 2;
    }
    let points = ColMajorArrayRef::new(&values, &[N_SITES, N_POINTS]).unwrap();
    let mut two_point_values = vec![0usize; N_SITES * 2];
    two_point_values[varying + N_SITES] = 1;
    let two_points = ColMajorArrayRef::new(&two_point_values, &[N_SITES, 2]).unwrap();

    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(varying),
            ..Default::default()
        },
    )
    .unwrap();
    let hint = EvaluationHint::around(varying);

    // Warm every message the fixture needs before measuring.
    evaluator
        .evaluate_batched_with_hint(points, hint.clone())
        .unwrap();
    evaluator
        .evaluate_batched_with_hint(two_points, hint.clone())
        .unwrap();

    let (wrapped, wrapper_allocations) = allocation_counter::measure(|| {
        evaluator
            .evaluate_batched_with_hint(points, hint.clone())
            .unwrap()
    });
    let (typed, typed_allocations) = allocation_counter::measure(|| {
        evaluator
            .evaluate_batched_typed::<f64>(points, hint.clone())
            .unwrap()
    });
    let (_, two_point_allocations) = allocation_counter::measure(|| {
        evaluator
            .evaluate_batched_typed::<f64>(two_points, hint.clone())
            .unwrap()
    });

    eprintln!(
        "#709 warm allocation counts: sites={N_SITES} points={N_POINTS} wrapper={wrapper_allocations} typed={typed_allocations} two_point_typed={two_point_allocations}"
    );

    assert_eq!(typed.len(), wrapped.len());
    for (typed, wrapped) in typed.iter().zip(&wrapped) {
        assert_eq!(*typed, wrapped.real());
        assert_eq!(wrapped.imag(), 0.0);
    }

    assert!(
        typed_allocations + N_POINTS as u64 <= wrapper_allocations,
        "the typed route must save at least one allocation per result: typed={typed_allocations} wrapper={wrapper_allocations}"
    );
    assert!(
        typed_allocations < AUDITED_ASSIGNMENT_VECTORS,
        "a warm {N_SITES}-site {N_POINTS}-point typed call must stay far below the audited {AUDITED_ASSIGNMENT_VECTORS} short-lived assignment vectors, got {typed_allocations}"
    );
    assert!(
        two_point_allocations < 256,
        "a warm two-point Guard-shaped call must stay small, got {two_point_allocations}"
    );
}
