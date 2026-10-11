use super::*;
use rand::Rng;

fn delta_guard_inputs(
    nsearch: usize,
    sweeps: usize,
) -> (
    Vec<TreeTN<IdxTensor, usize>>,
    TreeAciOptions<usize>,
    DynIndex,
    DynIndex,
) {
    let (input, left, right) = delta_tree();
    let options = TreeAciOptions {
        nsearch_global_pivots: nsearch,
        max_nglobal_pivots: nsearch,
        nsweeps_global_search: sweeps,
        global_tolerance_margin: 1.0,
        ..TreeAciOptions::default()
    };
    (vec![input], options, left, right)
}

/// [AI Supplied] #686: the scale-estimation batch already contains every
/// starting target. A zero-sweep guard must use those residuals directly.
#[test]
fn zero_walk_sweeps_evaluate_starting_targets_in_one_batch() {
    const NSEARCH: usize = 8;
    let (inputs, options, left, right) = delta_guard_inputs(NSEARCH, 0);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    state.output = zero_tree(left, right);
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let mut batches = Vec::new();
    let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        batches.push(batch.n_points());
        identity(batch, output)
    };
    let report = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut seeded_rng(),
        &mut operator,
        &[],
    )
    .unwrap();
    let mut oracle_rng = seeded_rng();
    let expected = (0..NSEARCH)
        .map(|_| vec![oracle_rng.random_range(0..2), oracle_rng.random_range(0..2)])
        .filter(|point| point[0] == point[1])
        .collect::<Vec<_>>();
    // Candidate deduplication retains the first occurrence, including when
    // equal diagonal assignments are separated by a different assignment.
    let mut unique = Vec::new();
    for point in expected {
        if !unique.contains(&point) {
            unique.push(point);
        }
    }
    assert_eq!(report.pivots, unique);
    assert_eq!(batches, vec![NSEARCH]);
    assert_eq!(report.evaluated_points, NSEARCH as u64);
}

/// [AI Supplied] #686: each binary-site walk evaluates only the alternate
/// coordinate at its two sites, with its initial residual supplied by the
/// scale-estimation batch. Point accounting includes no duplicate seed.
#[test]
fn coordinate_walk_point_count_excludes_duplicate_seed_targets() {
    const NSEARCH: usize = 5;
    let (inputs, options, left, right) = delta_guard_inputs(NSEARCH, 1);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    state.output = zero_tree(left, right);
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let mut batches = Vec::new();
    let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        batches.push(batch.n_points());
        identity(batch, output)
    };
    let report = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut seeded_rng(),
        &mut operator,
        &[],
    )
    .unwrap();
    let mut expected_batches = vec![NSEARCH];
    expected_batches.extend(std::iter::repeat_n(1, 2 * NSEARCH));
    assert_eq!(batches, expected_batches);
    assert_eq!(report.evaluated_points, (3 * NSEARCH) as u64);
    assert!(!report.pivots.is_empty());
    assert!(report.pivots.iter().all(|point| point[0] == point[1]));
}
