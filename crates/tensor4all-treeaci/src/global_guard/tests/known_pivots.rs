//! Previously detected residuals cannot be resolved by a later random miss.
use super::*;

fn setup() -> (
    Vec<TreeTN<IdxTensor, usize>>,
    TreeAciOptions<usize>,
    DynIndex,
    DynIndex,
) {
    let (input, left, right) = delta_tree();
    let options = TreeAciOptions {
        nsearch_global_pivots: 1,
        max_nglobal_pivots: 1,
        nsweeps_global_search: 0,
        global_tolerance_margin: 1.0,
        ..Default::default()
    };
    (vec![input], options, left, right)
}

#[test]
fn known_failures_keep_their_slot_and_are_removed_after_actual_repair() {
    let (inputs, options, left, right) = setup();
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    state.output = zero_tree(left, right);
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let known = vec![vec![1, 1]];
    let mut batches = Vec::new();
    let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
        batches.push(batch.n_points());
        identity(batch, out)
    };
    let report = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut seeded_rng(),
        &mut operator,
        &known,
    )
    .unwrap();
    assert_eq!(report.pivots, known);
    assert_eq!(report.evaluated_points, 2);
    assert_eq!(batches, vec![2]);
    // The same known assignment is judged against the current output, so a
    // successful local repair releases its slot rather than leaving a latch.
    state.output = inputs[0].clone();
    let report = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut seeded_rng(),
        &mut identity,
        &known,
    )
    .unwrap();
    assert!(report.pivots.is_empty());
    assert_eq!(report.evaluated_points, 2);
}

#[test]
fn retained_points_are_preflighted_together_with_the_start_batch() {
    let (inputs, options, _, _) = setup();
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    evaluators.max_working_bytes = 1;
    let mut calls = 0;
    let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
        calls += 1;
        identity(batch, out)
    };
    let error = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut seeded_rng(),
        &mut operator,
        &[],
    )
    .unwrap_err();
    let crate::TreeAciError::ResourceLimit { requested, .. } = error else {
        panic!("expected working-byte refusal");
    };
    evaluators.max_working_bytes = requested;
    find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut seeded_rng(),
        &mut operator,
        &[],
    )
    .unwrap();
    assert_eq!(calls, 1);
    let mut called = false;
    let mut reject_operator = |_: crate::TreeElementwiseBatch<'_, f64>, _: &mut [f64]| {
        called = true;
        Ok(())
    };
    let error = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut seeded_rng(),
        &mut reject_operator,
        &[vec![1, 1]],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::TreeAciError::ResourceLimit {
            resource: "working bytes",
            ..
        }
    ));
    assert!(!called);
}

#[test]
fn retained_coordinates_and_count_are_checked_before_evaluation() {
    let (inputs, options, _, _) = setup();
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let mut called = false;
    let mut operator = |_: crate::TreeElementwiseBatch<'_, f64>, _: &mut [f64]| {
        called = true;
        Ok(())
    };
    for known in [
        vec![vec![1]],
        vec![vec![2, 1]],
        vec![vec![0, 0], vec![1, 1]],
    ] {
        assert!(find_global_pivots(
            &state,
            &mut evaluators,
            &options,
            &mut seeded_rng(),
            &mut operator,
            &known
        )
        .is_err());
    }
    let mut overflow = options;
    overflow.nsearch_global_pivots = usize::MAX;
    assert!(matches!(
        find_global_pivots(
            &state,
            &mut evaluators,
            &overflow,
            &mut seeded_rng(),
            &mut operator,
            &[vec![1, 1]]
        ),
        Err(crate::TreeAciError::SizeOverflow {
            context: "guard starting-point count"
        })
    ));
    assert!(!called);
}

#[test]
fn nonfinite_target_at_a_retained_point_is_not_hidden_by_random_starts() {
    let (inputs, options, _, _) = setup();
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let mut operator = |_: crate::TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
        assert_eq!(out.len(), 2);
        out[0] = 0.0;
        out[1] = f64::NAN;
        Ok(())
    };
    assert!(matches!(
        find_global_pivots(
            &state,
            &mut evaluators,
            &options,
            &mut seeded_rng(),
            &mut operator,
            &[vec![1, 1]]
        ),
        Err(crate::TreeAciError::NonFiniteValue {
            context: "guard target"
        })
    ));
}

#[test]
fn stronger_new_discoveries_fill_only_slots_not_held_by_known_failures() {
    let (mut inputs, mut options, left, right) = setup();
    let bond = DynIndex::new_dyn(2);
    inputs[0] = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![left.clone(), bond.clone()], vec![10.0, 0.0, 0.0, 1.0])
                .unwrap(),
            IdxTensor::from_dense(vec![bond, right.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    options.nsearch_global_pivots = 20;
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    state.output = zero_tree(left, right);
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let known = vec![vec![1, 1]];
    let report = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut seeded_rng(),
        &mut identity,
        &known,
    )
    .unwrap();
    assert_eq!(
        report.pivots, known,
        "the still-failing point cannot be forgotten for a stronger discovery"
    );
    assert_eq!(report.evaluated_points, 21);
    options.max_nglobal_pivots = 2;
    let report = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut seeded_rng(),
        &mut identity,
        &known,
    )
    .unwrap();
    assert_eq!(report.pivots, vec![vec![1, 1], vec![0, 0]]);
    let report = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut seeded_rng(),
        &mut identity,
        &[vec![1, 1], vec![0, 0]],
    )
    .unwrap();
    assert_eq!(
        report.pivots,
        vec![vec![0, 0], vec![1, 1]],
        "retained failures are deterministically sorted by their current residual"
    );
    assert_eq!(report.evaluated_points, 22);
}
