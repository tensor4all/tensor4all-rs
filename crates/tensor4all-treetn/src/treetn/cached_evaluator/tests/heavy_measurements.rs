//! Scheduled wall-time measurement using private cached-evaluator helpers.

use super::{measurements::random_measurement_chain, *};

/// Uncached-path duplicate of `evaluate_batched_with_hint`, calling
/// `compute_stacked_message` directly instead of
/// `get_or_compute_node_message`. Kept as measurement tooling (see
/// `message_cache_wall_time_on_realistic_floating_zone_walk`): times the
/// same floating-zone walk with and without the persistent message
/// cache, using one evaluator so construction overhead is identical
/// between the two conditions.
fn evaluate_batched_uncached(
    evaluator: &mut TreeTNCachedEvaluator<'_, usize>,
    values: ColMajorArrayRef<'_, usize>,
    center: &usize,
) -> Result<Vec<CachedScalar>> {
    let plan = RootedMessagePlan::new(evaluator.tree, center)?;
    let assignment_batches = evaluator.build_message_assignment_batches(&plan, values)?;
    let mut messages = HashMap::<usize, StackedMessage>::default();
    for node in &plan.postorder {
        let points = assignment_batches.get(node).unwrap().first_points.clone();
        let node_message = evaluator.compute_stacked_message(
            node,
            values,
            &points,
            &plan,
            &assignment_batches,
            &messages,
            &mut CachedEvaluationRoute::global(),
        )?;
        messages.insert(*node, node_message);
    }
    let mut component_batches = Vec::new();
    let mut environment_cache = HashMap::default();
    for neighbor in plan.children.get(center).cloned().unwrap_or_default() {
        let assignment_batch = assignment_batches.get(&neighbor).unwrap();
        let environment = messages.remove(&neighbor).unwrap();
        environment_cache.insert(neighbor, environment);
        component_batches.push(ComponentBatch {
            neighbor,
            point_to_assignment: assignment_batch.point_to_assignment.clone(),
        });
    }
    evaluator.contract_center_for_points(
        center,
        values,
        &component_batches,
        &environment_cache,
        &mut CachedEvaluationRoute::global(),
        Parallelism::Serial,
    )
}

/// Measurement, not a regression test: how much does the persistent
/// message cache save on the real `find_global_pivots` call pattern?
/// Drives `floating_zone_walk` with the same defaults
/// (`nsearch_global_pivots = 5`, `nsweeps_global_search = 100`) against a
/// 16-site chain at bond 128, once through the normal (now cached) path
/// and once through the direct, uncached path, on the same evaluator so
/// construction cost cancels out. Prints wall time and message counts for
/// both; asserts nothing about the ratio, since wall-clock numbers are
/// not a reproducible pass/fail condition.
///
/// The cached path is expected to be faster on this workload because the
/// raw chain path avoids constructing an `IdxTensor` per message and the
/// persistent cache avoids recomputing unchanged directed messages. The
/// test remains measurement tooling rather than a wall-clock regression
/// assertion; see `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`.
#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn message_cache_wall_time_on_realistic_floating_zone_walk() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;
    use tensor4all_core::floating_zone_walk;

    const N_SITES: usize = 16;
    const LOCAL_DIM: usize = 2;
    const BOND_DIM: usize = 128;
    const NSEARCH: usize = 5;
    const MAX_SWEEPS: usize = 100;

    fn build_tree(seed: u64) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        random_measurement_chain(N_SITES, LOCAL_DIM, BOND_DIM, seed)
    }

    fn starts() -> Vec<Vec<usize>> {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        (0..NSEARCH)
            .map(|_| {
                (0..N_SITES)
                    .map(|_| rng.random_range(0..LOCAL_DIM))
                    .collect()
            })
            .collect()
    }
    let site_dims = vec![LOCAL_DIM; N_SITES];

    // Cached scan-aware path used by TreeACI's global guard.
    let (tree, site_indices) = build_tree(7);
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &site_indices, CachedEvaluatorOptions::default())
            .unwrap();
    // Force a centre outside the timed region, matching the uncached arm.
    let warm_values0 = vec![0usize; N_SITES];
    let warm_shape0 = [N_SITES, 1];
    let warm_points0 = ColMajorArrayRef::new(&warm_values0, &warm_shape0).unwrap();
    evaluator.evaluate_batched(warm_points0).unwrap();
    let mut total_hits = 0usize;
    let mut total_misses = 0usize;
    let mut total_calls = 0usize;
    let cached_start = Instant::now();
    for start in &starts() {
        floating_zone_walk(
            &site_dims,
            start,
            MAX_SWEEPS,
            f64::INFINITY,
            |_scan_site: Option<usize>, points: &[Vec<usize>]| -> Result<Vec<f64>> {
                let mut values = vec![0usize; N_SITES * points.len()];
                for (p, point) in points.iter().enumerate() {
                    for (site, &v) in point.iter().enumerate() {
                        values[site + N_SITES * p] = v;
                    }
                }
                let shape = [N_SITES, points.len()];
                let arr = ColMajorArrayRef::new(&values, &shape).unwrap();
                let hint = (0..N_SITES)
                    .find(|&site| {
                        points[1..]
                            .iter()
                            .any(|point| point[site] != points[0][site])
                    })
                    .map(EvaluationHint::around)
                    .unwrap_or_default();
                let out = evaluator.evaluate_batched_with_hint(arr, hint)?;
                let stats = evaluator.stats_for_test();
                total_hits += stats.message_cache_hits;
                total_misses += stats.message_cache_misses;
                total_calls += 1;
                Ok(out.iter().map(|v| v.real().abs()).collect())
            },
        )
        .unwrap();
    }
    let cached_elapsed = cached_start.elapsed();

    // Uncached path: identical tree, walk, and varying-site center.
    let (tree2, site_indices2) = build_tree(7);
    let mut evaluator2 =
        TreeTNCachedEvaluator::new(&tree2, &site_indices2, CachedEvaluatorOptions::default())
            .unwrap();
    // Force a centre exactly as the cached run's first call would.
    let warm_values = vec![0usize; N_SITES];
    let warm_shape = [N_SITES, 1];
    let warm_points = ColMajorArrayRef::new(&warm_values, &warm_shape).unwrap();
    evaluator2.evaluate_batched(warm_points).unwrap();
    let center = *evaluator2.center().unwrap();
    let uncached_start = Instant::now();
    for start in &starts() {
        floating_zone_walk(
            &site_dims,
            start,
            MAX_SWEEPS,
            f64::INFINITY,
            |_scan_site: Option<usize>, points: &[Vec<usize>]| -> Result<Vec<f64>> {
                let mut values = vec![0usize; N_SITES * points.len()];
                for (p, point) in points.iter().enumerate() {
                    for (site, &v) in point.iter().enumerate() {
                        values[site + N_SITES * p] = v;
                    }
                }
                let shape = [N_SITES, points.len()];
                let arr = ColMajorArrayRef::new(&values, &shape).unwrap();
                let varying_center = (0..N_SITES).find(|&site| {
                    points[1..]
                        .iter()
                        .any(|point| point[site] != points[0][site])
                });
                let out = evaluate_batched_uncached(
                    &mut evaluator2,
                    arr,
                    varying_center.as_ref().unwrap_or(&center),
                )?;
                Ok(out.iter().map(|v| v.as_complex().re.abs()).collect())
            },
        )
        .unwrap();
    }
    let uncached_elapsed = uncached_start.elapsed();

    println!(
        "floating-zone walk at bond={BOND_DIM}: cached={cached_elapsed:?} uncached={uncached_elapsed:?} speedup={:.2}x total_calls={total_calls} total_hits={total_hits} total_misses={total_misses} node_hit_rate={:.3}",
        uncached_elapsed.as_secs_f64() / cached_elapsed.as_secs_f64(),
        total_hits as f64 / (total_hits + total_misses) as f64,
    );
    assert!(cached_elapsed.as_nanos() > 0);
}
