//! `patched_interpolate` with `cache_candidates`: child patches also start
//! from the largest values of their inherited evaluation cache.
//!
//! The root of `chain("s", 4, 2)` is measured exhaustively and fails, so its
//! children inherit every value of their half; the children are then exact.

mod adaptive_common;

use adaptive_common::*;
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    PatchedInterpolationOptions, VerificationOptions,
};

/// `1 + x0 + 2 x1 + 4 x2 + 8 x3`: the values 1 to 16.
fn ramp(p: &[usize]) -> f64 {
    1.0 + (p[0] + 2 * p[1] + 4 * p[2] + 8 * p[3]) as f64
}

fn options(f: &dyn Fn(&[usize]) -> f64, problem: &Problem) -> PatchedInterpolationOptions {
    let (_, norm) = dense_reference(problem, f);
    // The derived split order fixes s0 first.
    l2_given(3, norm, 1e-12).with_verification(VerificationOptions::new().with_retries(0))
}

/// The initial pivots of the two children (calls 1 and 2), in child
/// coordinates `(s1, s2, s3)`.
fn child_pivots(
    f: &(dyn Fn(&[usize]) -> f64 + Sync),
    user: &[Vec<usize>],
    options: &PatchedInterpolationOptions,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, usize) {
    let problem = chain("s", 4, 2);
    let engine = ScriptedEngine::new(vec![
        Step::converged(Network::Constant(0.0)),
        Step::converged(Network::Exact),
    ]);
    let result = run(&engine, &problem, f, user, options).unwrap();
    let calls = engine.calls();
    assert_eq!(calls.len(), 3);
    (
        calls[1].initial_pivots.clone(),
        calls[2].initial_pivots.clone(),
        result.report.function_evaluations,
    )
}

#[test]
fn children_start_from_their_largest_cached_values_after_the_other_sources() {
    let problem = chain("s", 4, 2);
    let base = options(&ramp, &problem).with_n_initial_pivots(3);
    // User pivot (0, 0, 0, 0) for child 0; worst points of the root: 16 at
    // (1, 1, 1, 1) for child 1 and 15 at (0, 1, 1, 1) for child 0. The
    // largest cached values of either child are at (1, 1, 1), (0, 1, 1), and
    // (1, 0, 1); (1, 1, 1) is already a worst point.
    let user = [vec![0, 0, 0, 0]];
    let (first, second, evaluations) =
        child_pivots(&ramp, &user, &base.clone().with_cache_candidates(true));
    assert_eq!(
        first,
        [vec![0, 0, 0], vec![1, 1, 1], vec![0, 1, 1], vec![1, 0, 1]]
    );
    assert_eq!(second, [vec![1, 1, 1], vec![0, 1, 1], vec![1, 0, 1]]);
    // The candidates come from the cache: no evaluation is added.
    assert_eq!(evaluations, 16);

    // Without the option the same sources come first, then the random fill
    // tops up to n_initial_pivots = 3: no cached point is added.
    let (first, second, evaluations) = child_pivots(&ramp, &user, &base);
    assert_eq!(first[..2], [vec![0, 0, 0], vec![1, 1, 1]]);
    assert_eq!(first.len(), 3);
    assert_eq!(second[0], vec![1, 1, 1]);
    assert_eq!(second.len(), 3);
    assert_eq!(evaluations, 16);
}

#[test]
fn an_unbounded_candidate_target_takes_every_nonzero_cached_value() {
    // n_initial_pivots = usize::MAX neither overflows nor over-allocates:
    // child 0 starts from its worst point and then all eight cached values,
    // largest first (15, 13, ..., 1), and the patch size ends the list.
    let problem = chain("s", 4, 2);
    let options = options(&ramp, &problem)
        .with_n_initial_pivots(usize::MAX)
        .with_cache_candidates(true);
    let (first, _, _) = child_pivots(&ramp, &[], &options);
    assert_eq!(
        first,
        [
            vec![1, 1, 1],
            vec![0, 1, 1],
            vec![1, 0, 1],
            vec![0, 0, 1],
            vec![1, 1, 0],
            vec![0, 1, 0],
            vec![1, 0, 0],
            vec![0, 0, 0],
        ]
    );
}

#[test]
fn zero_cached_values_are_never_candidates() {
    // In child 0 (x0 = 0) only the four points with x1 = 1 are nonzero: 3,
    // 7, 11, and 15. The fifth candidate is the random fill.
    let f = |p: &[usize]| ramp(p) * p[1] as f64;
    let problem = chain("s", 4, 2);
    let options = options(&f, &problem)
        .with_n_initial_pivots(5)
        .with_cache_candidates(true);
    let (first, _, _) = child_pivots(&f, &[], &options);
    assert_eq!(
        first[..4],
        [vec![1, 1, 1], vec![1, 0, 1], vec![1, 1, 0], vec![1, 0, 0]]
    );
    // The fifth comes from the random fill of seed 0, not from the zero
    // cached values (a zero candidate would be (0, 0, 0), the smallest).
    assert_eq!(first[4], vec![0, 1, 0]);
}
