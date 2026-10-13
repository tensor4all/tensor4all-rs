//! Wall-time and work measurements on realistic floating-zone walks. The
//! `diagnostic_*` tests are ignored measurement tools, not regression checks.

use super::*;

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
    let mut messages = HashMap::<usize, StackedMessage>::new();
    for node in &plan.postorder {
        let points = assignment_batches.get(node).unwrap().first_points.clone();
        let node_message = evaluator.compute_stacked_message(
            node,
            values,
            &points,
            &plan,
            &assignment_batches,
            &messages,
        )?;
        messages.insert(*node, node_message);
    }
    let mut component_batches = Vec::new();
    let mut environment_cache = HashMap::new();
    for neighbor in plan.children.get(center).cloned().unwrap_or_default() {
        let assignment_batch = assignment_batches.get(&neighbor).unwrap();
        let environment = messages.remove(&neighbor).unwrap();
        environment_cache.insert(neighbor, environment);
        component_batches.push(ComponentBatch {
            neighbor,
            point_to_assignment: assignment_batch.point_to_assignment.clone(),
        });
    }
    evaluator.contract_center_for_points(center, values, &component_batches, &environment_cache)
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
fn message_cache_wall_time_on_realistic_floating_zone_walk() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;
    use tensor4all_core::floating_zone_walk;
    use tensor4all_simplett::{tensor3_zeros, SimpleTensorTrain, Tensor3, Tensor3Ops};

    const N_SITES: usize = 16;
    const LOCAL_DIM: usize = 2;
    const BOND_DIM: usize = 128;
    const NSEARCH: usize = 5;
    const MAX_SWEEPS: usize = 100;

    fn build_tree(seed: u64) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut tensors: Vec<Tensor3<f64>> = Vec::with_capacity(N_SITES);
        for site in 0..N_SITES {
            let left_dim = if site == 0 { 1 } else { BOND_DIM };
            let right_dim = if site == N_SITES - 1 { 1 } else { BOND_DIM };
            let mut tensor = tensor3_zeros(left_dim, LOCAL_DIM, right_dim);
            for l in 0..left_dim {
                for s in 0..LOCAL_DIM {
                    for r in 0..right_dim {
                        tensor.set3(l, s, r, rng.random::<f64>());
                    }
                }
            }
            tensors.push(tensor);
        }
        let tt = SimpleTensorTrain::new(tensors).unwrap();
        crate::tensor_train_to_treetn(&tt).unwrap()
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

/// Root-cause investigation for the slowdown found by
/// `message_cache_wall_time_on_realistic_floating_zone_walk`: which phase
/// inside `get_or_compute_node_message` actually accounts for the time,
/// on the real call pattern -- not inferred from the old bond=2 primitive
/// breakdown in `2026-08-17-treeaci-per-evaluation-cost.md`. Reuses the
/// same 16-site, bond=128 walk as the wall-time measurement.
#[test]
fn message_cache_phase_breakdown_on_realistic_floating_zone_walk() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use tensor4all_core::floating_zone_walk;
    use tensor4all_simplett::{tensor3_zeros, SimpleTensorTrain, Tensor3, Tensor3Ops};

    const N_SITES: usize = 16;
    const LOCAL_DIM: usize = 2;
    const BOND_DIM: usize = 128;
    const NSEARCH: usize = 5;
    const MAX_SWEEPS: usize = 100;

    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut tensors: Vec<Tensor3<f64>> = Vec::with_capacity(N_SITES);
    for site in 0..N_SITES {
        let left_dim = if site == 0 { 1 } else { BOND_DIM };
        let right_dim = if site == N_SITES - 1 { 1 } else { BOND_DIM };
        let mut tensor = tensor3_zeros(left_dim, LOCAL_DIM, right_dim);
        for l in 0..left_dim {
            for s in 0..LOCAL_DIM {
                for r in 0..right_dim {
                    tensor.set3(l, s, r, rng.random::<f64>());
                }
            }
        }
        tensors.push(tensor);
    }
    let tt = SimpleTensorTrain::new(tensors).unwrap();
    let (tree, site_indices) = crate::tensor_train_to_treetn(&tt).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &site_indices, CachedEvaluatorOptions::default())
            .unwrap();
    let warm_values = vec![0usize; N_SITES];
    let warm_shape = [N_SITES, 1];
    evaluator
        .evaluate_batched(ColMajorArrayRef::new(&warm_values, &warm_shape).unwrap())
        .unwrap();

    phase_timing::reset_all();

    let site_dims = vec![LOCAL_DIM; N_SITES];
    let mut start_rng = ChaCha8Rng::seed_from_u64(11);
    let starts: Vec<Vec<usize>> = (0..NSEARCH)
        .map(|_| {
            (0..N_SITES)
                .map(|_| start_rng.random_range(0..LOCAL_DIM))
                .collect()
        })
        .collect();
    for start in &starts {
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
                let out = evaluator.evaluate_batched(arr)?;
                Ok(out.iter().map(|v| v.real().abs()).collect())
            },
        )
        .unwrap();
    }

    use std::sync::atomic::Ordering;
    let key_and_lookup = phase_timing::KEY_AND_LOOKUP_NS.load(Ordering::Relaxed);
    let contract = phase_timing::CONTRACT_NS.load(Ordering::Relaxed);
    let tensor_values = phase_timing::TENSOR_VALUES_NS.load(Ordering::Relaxed);
    let insert = phase_timing::INSERT_NS.load(Ordering::Relaxed);
    let reconstruct = phase_timing::RECONSTRUCT_NS.load(Ordering::Relaxed);
    let total = key_and_lookup + contract + tensor_values + insert + reconstruct;
    println!(
        "phase breakdown at bond={BOND_DIM}: key_and_lookup={:.1}ms ({:.1}%) contract={:.1}ms ({:.1}%) tensor_values={:.1}ms ({:.1}%) insert={:.1}ms ({:.1}%) reconstruct={:.1}ms ({:.1}%) total={:.1}ms",
        key_and_lookup as f64 / 1e6, 100.0 * key_and_lookup as f64 / total as f64,
        contract as f64 / 1e6, 100.0 * contract as f64 / total as f64,
        tensor_values as f64 / 1e6, 100.0 * tensor_values as f64 / total as f64,
        insert as f64 / 1e6, 100.0 * insert as f64 / total as f64,
        reconstruct as f64 / 1e6, 100.0 * reconstruct as f64 / total as f64,
        total as f64 / 1e6,
    );
    assert!(total > 0);
}

/// [AI Supplied] Diagnostic-only split of a fully warm, same-batch call.
///
/// The existing floating-zone phase test intentionally changes points and
/// therefore measures cache misses.  This fixture repeats exactly one
/// 64-point batch after warming it, so any remaining environment or centre
/// work is work the persistent cache did not eliminate.
#[test]
#[ignore]
fn diagnostic_same_batch_warm_environment_vs_center_cost() {
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    const N_SITES: usize = 16;
    const LOCAL_DIM: usize = 2;
    const N_POINTS: usize = 64;
    const REPEATS: usize = 50;
    const CENTER: usize = N_SITES / 2;

    // [AI Supplied] Representation census for the packed-cache audit.
    eprintln!(
        "representation sizes: f64={} CachedScalar={} IndexKey={} AnyScalar={}",
        std::mem::size_of::<f64>(),
        std::mem::size_of::<CachedScalar>(),
        std::mem::size_of::<IndexKey>(),
        std::mem::size_of::<AnyScalar>(),
    );

    for bond_dim in [64usize, 128, 256] {
        let physical = (0..N_SITES)
            .map(|_| DynIndex::new_dyn(LOCAL_DIM))
            .collect::<Vec<_>>();
        let bonds = (0..N_SITES - 1)
            .map(|_| DynIndex::new_dyn(bond_dim))
            .collect::<Vec<_>>();
        let mut tensors = Vec::with_capacity(N_SITES);
        for site in 0..N_SITES {
            let mut indices = vec![physical[site].clone()];
            if site > 0 {
                indices.push(bonds[site - 1].clone());
            }
            if site + 1 < N_SITES {
                indices.push(bonds[site].clone());
            }
            let len = indices.iter().map(IndexLike::dim).product();
            tensors.push(IdxTensor::from_dense(indices, vec![1.0_f64; len]).unwrap());
        }
        let tree = TreeTN::from_tensors(tensors, (0..N_SITES).collect()).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &physical,
            CachedEvaluatorOptions::<usize> {
                center: Some(CENTER),
                ..CachedEvaluatorOptions::default()
            },
        )
        .unwrap();
        let mut values = vec![0usize; N_SITES * N_POINTS];
        for point in 0..N_POINTS {
            for site in 0..N_SITES {
                values[site + N_SITES * point] = (point >> (site % 6)) & 1;
            }
        }
        let shape = [N_SITES, N_POINTS];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        evaluator.evaluate_batched(points).unwrap();

        phase_timing::reset_all();
        let started = Instant::now();
        for _ in 0..REPEATS {
            std::hint::black_box(evaluator.evaluate_batched(points).unwrap());
        }
        let total_ns = started.elapsed().as_nanos() as u64;
        let environment_ns = phase_timing::BUILD_ENV_NS.load(Ordering::Relaxed);
        let center_ns = phase_timing::CENTER_NS.load(Ordering::Relaxed);
        let raw_capability_ns = phase_timing::RAW_CAPABILITY_NS.load(Ordering::Relaxed);
        let assignment_batch_ns = phase_timing::ASSIGNMENT_BATCH_NS.load(Ordering::Relaxed);
        let message_loop_ns = phase_timing::MESSAGE_LOOP_NS.load(Ordering::Relaxed);
        let component_assembly_ns = phase_timing::COMPONENT_ASSEMBLY_NS.load(Ordering::Relaxed);
        let raw_prep_ns = phase_timing::RAW_CENTER_PREP_NS.load(Ordering::Relaxed);
        let raw_contract_ns = phase_timing::RAW_CENTER_CONTRACT_NS.load(Ordering::Relaxed);
        let raw_dispatch_ns = phase_timing::RAW_CENTER_DISPATCH_NS.load(Ordering::Relaxed);
        let raw_prelude_ns = phase_timing::RAW_CENTER_PRELUDE_NS.load(Ordering::Relaxed);
        let raw_result_ns = phase_timing::RAW_CENTER_RESULT_NS.load(Ordering::Relaxed);
        let raw_values_copied = phase_timing::RAW_CENTER_VALUES_COPIED.load(Ordering::Relaxed);
        let raw_assignments_copied =
            phase_timing::RAW_CENTER_ASSIGNMENTS_COPIED.load(Ordering::Relaxed);
        let key_ns = phase_timing::KEY_AND_LOOKUP_NS.load(Ordering::Relaxed);
        let reconstruct_ns = phase_timing::RECONSTRUCT_NS.load(Ordering::Relaxed);
        let reconstructed_values = phase_timing::RECONSTRUCT_VALUES.load(Ordering::Relaxed);
        let final_env_values = phase_timing::FINAL_ENV_VALUES.load(Ordering::Relaxed);
        let contract_ns = phase_timing::CONTRACT_NS.load(Ordering::Relaxed);
        let insert_ns = phase_timing::INSERT_NS.load(Ordering::Relaxed);
        let stats = evaluator.stats_for_test();
        assert_eq!(stats.message_cache_misses, 0);
        assert!(stats.message_cache_hits > 0);
        assert_eq!(contract_ns, 0);
        assert_eq!(insert_ns, 0);
        eprintln!(
            "warm same-batch bond={bond_dim}: total={:.3}ms/call, environment={:.3}ms/call ({:.1}%) {{raw_capability={:.3}ms/call, assignments={:.3}ms/call, message_loop={:.3}ms/call, component_assembly={:.3}ms/call, reconstructed_values={}/call, final_env_values={}/call}}, center={:.3}ms/call ({:.1}%) {{raw_dispatch={:.3}ms/call: prelude={:.3}ms/call, prep={:.3}ms/call, contract={:.3}ms/call, AnyScalar_wrap={:.3}ms/call, copied_values={}/call, copied_assignments={}/call}}, key={:.3}ms/call, reconstruct={:.3}ms/call, hits/call={}",
            total_ns as f64 / REPEATS as f64 / 1e6,
            environment_ns as f64 / REPEATS as f64 / 1e6,
            100.0 * environment_ns as f64 / total_ns as f64,
            raw_capability_ns as f64 / REPEATS as f64 / 1e6,
            assignment_batch_ns as f64 / REPEATS as f64 / 1e6,
            message_loop_ns as f64 / REPEATS as f64 / 1e6,
            component_assembly_ns as f64 / REPEATS as f64 / 1e6,
            reconstructed_values / REPEATS as u64,
            final_env_values / REPEATS as u64,
            center_ns as f64 / REPEATS as f64 / 1e6,
            100.0 * center_ns as f64 / total_ns as f64,
            raw_dispatch_ns as f64 / REPEATS as f64 / 1e6,
            raw_prelude_ns as f64 / REPEATS as f64 / 1e6,
            raw_prep_ns as f64 / REPEATS as f64 / 1e6,
            raw_contract_ns as f64 / REPEATS as f64 / 1e6,
            raw_result_ns as f64 / REPEATS as f64 / 1e6,
            raw_values_copied / REPEATS as u64,
            raw_assignments_copied / REPEATS as u64,
            key_ns as f64 / REPEATS as f64 / 1e6,
            reconstruct_ns as f64 / REPEATS as f64 / 1e6,
            stats.message_cache_hits,
        );

        if bond_dim == 256 {
            // [AI Supplied] Reproduce Guard's moving-center call pattern
            // and count the rooted metadata retained for an immutable
            // chain after every site has served as a center.
            let scan_all_centers = |evaluator: &mut TreeTNCachedEvaluator<'_, usize>| {
                for scan_center in 0..N_SITES {
                    let mut scan_values = vec![0usize; N_SITES * 2];
                    scan_values[scan_center + N_SITES] = 1;
                    let scan_shape = [N_SITES, 2usize];
                    let scan_points = ColMajorArrayRef::new(&scan_values, &scan_shape).unwrap();
                    evaluator
                        .evaluate_batched_with_hint(
                            scan_points,
                            EvaluationHint::around(scan_center),
                        )
                        .unwrap();
                }
            };

            phase_timing::reset_all();
            let first_scan_started = Instant::now();
            scan_all_centers(&mut evaluator);
            let first_scan_ns = first_scan_started.elapsed().as_nanos() as u64;
            let plan_build_ns = phase_timing::PLAN_BUILD_NS.load(Ordering::Relaxed);
            let layout_build_ns = phase_timing::LAYOUT_BUILD_NS.load(Ordering::Relaxed);
            let new_plan_count = phase_timing::PLAN_COUNT.load(Ordering::Relaxed);

            let second_scan_started = Instant::now();
            scan_all_centers(&mut evaluator);
            let second_scan_ns = second_scan_started.elapsed().as_nanos() as u64;
            let retained_layout_refs = evaluator
                .directed_component_layouts()
                .values()
                .map(|layout| layout.layout.input_positions.len())
                .sum::<usize>();
            let unique_directed_components = 2 * (N_SITES - 1);
            let unique_directed_component_refs = N_SITES * (N_SITES - 1);
            eprintln!(
                "moving-center bond={bond_dim}: first_scan={:.3}ms second_scan={:.3}ms newly_built_plans={} plan_build={:.3}ms layout_build={:.3}ms retained_centers={} retained_directed_layouts={} retained_layout_refs={} unique_directed_components={} unique_directed_component_refs={}",
                first_scan_ns as f64 / 1e6,
                second_scan_ns as f64 / 1e6,
                new_plan_count,
                plan_build_ns as f64 / 1e6,
                layout_build_ns as f64 / 1e6,
                evaluator.rooted_plan_count_for_test(),
                evaluator.directed_component_layouts().len(),
                retained_layout_refs,
                unique_directed_components,
                unique_directed_component_refs,
            );
            assert_eq!(new_plan_count, (N_SITES - 1) as u64);
            assert_eq!(evaluator.rooted_plan_count_for_test(), N_SITES);
            assert_eq!(
                evaluator.directed_component_layouts().len(),
                unique_directed_components
            );
            assert_eq!(retained_layout_refs, unique_directed_component_refs);
            assert_eq!(unique_directed_component_refs, 240);
        }
    }
}

// Measurement tooling rather than a wall-clock regression assertion
// (mirroring `message_cache_wall_time_on_realistic_floating_zone_walk`'s
// own framing above), kept from the investigation for gw-rs issue
// tensor4all-rs#671's downstream follow-up. Compares this evaluator's
// realistic floating-zone-walk wall time on a plain 16-site chain
// against a same-size comb tree (one degree-3 hub, three 5-site arms),
// both at the same bond dimension, under the same cached evaluate_batched
// path used by `message_cache_wall_time_on_realistic_floating_zone_walk`
// above.
//
// Read this test's ratio as "does branching cost something at matched
// bond dimension," not as a proxy for real downstream wall time: this
// investigation measured 25.84x before the two-child raw path
// (`try_compute_branch_message_raw`) existed, a regression to 44.24x
// from an initial version of that path's wrong point-count-based BLAS
// gate, 29.26x after fixing the gate -- yet the real gw-rs downstream
// pipeline (which does not have equal bond dimensions between its chain
// and comb topologies) measured a genuine 2.96x-3.15x wall-time
// improvement per treeaci stage from the same fix. See
// `docs/worklogs/2026-08-22-treetn-branch-message-raw-path.md` for the
// full trace, including why the same-bond assumption here does not
// represent the real workload's shape.
#[test]
#[ignore]
fn diagnostic_chain_vs_comb_wall_time_on_realistic_floating_zone_walk() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;
    use tensor4all_core::floating_zone_walk;
    use tensor4all_simplett::{tensor3_zeros, SimpleTensorTrain, Tensor3, Tensor3Ops};

    const N_SITES: usize = 16;
    const LOCAL_DIM: usize = 2;
    const BOND_DIM: usize = 128;
    const NSEARCH: usize = 5;
    const MAX_SWEEPS: usize = 100;
    const ARM_LEN: usize = 5; // hub + 3*5 = 16 sites, matching N_SITES

    fn build_chain(seed: u64) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut tensors: Vec<Tensor3<f64>> = Vec::with_capacity(N_SITES);
        for site in 0..N_SITES {
            let left_dim = if site == 0 { 1 } else { BOND_DIM };
            let right_dim = if site == N_SITES - 1 { 1 } else { BOND_DIM };
            let mut tensor = tensor3_zeros(left_dim, LOCAL_DIM, right_dim);
            for l in 0..left_dim {
                for s in 0..LOCAL_DIM {
                    for r in 0..right_dim {
                        tensor.set3(l, s, r, rng.random::<f64>());
                    }
                }
            }
            tensors.push(tensor);
        }
        let tt = SimpleTensorTrain::new(tensors).unwrap();
        crate::tensor_train_to_treetn(&tt).unwrap()
    }

    // Hub (node 0) + three arms of ARM_LEN sites each: arm a's sites are
    // named `1 + a*ARM_LEN ..= a*ARM_LEN + ARM_LEN`. Center is fixed to
    // the tip of arm 0, so the hub (rooted toward that tip) has exactly
    // one parent (arm 0's first site) and two children (arms 1 and 2's
    // first sites) -- the one node in this tree that cannot use
    // `try_compute_chain_message_raw`'s single-child fast path.
    fn build_comb(seed: u64) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>, usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut gen = |n: usize| -> Vec<f64> { (0..n).map(|_| rng.random::<f64>()).collect() };

        let s_hub = DynIndex::new_dyn(LOCAL_DIM);
        let hub_bonds: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(BOND_DIM)).collect();
        let hub = IdxTensor::from_dense(
            std::iter::once(s_hub.clone())
                .chain(hub_bonds.iter().cloned())
                .collect(),
            gen(LOCAL_DIM * BOND_DIM * BOND_DIM * BOND_DIM),
        )
        .unwrap();

        let mut tensors = vec![hub];
        let mut node_labels = vec![0usize];
        let mut indices = vec![s_hub];
        let mut next_node = 1usize;
        let mut center = 0usize;

        for (arm, hub_bond) in hub_bonds.iter().enumerate() {
            let mut prev_bond = hub_bond.clone();
            for depth in 0..ARM_LEN {
                let s = DynIndex::new_dyn(LOCAL_DIM);
                let is_tip = depth == ARM_LEN - 1;
                let tensor = if is_tip {
                    IdxTensor::from_dense(
                        vec![prev_bond.clone(), s.clone()],
                        gen(BOND_DIM * LOCAL_DIM),
                    )
                    .unwrap()
                } else {
                    let next_bond = DynIndex::new_dyn(BOND_DIM);
                    let tensor = IdxTensor::from_dense(
                        vec![prev_bond.clone(), s.clone(), next_bond.clone()],
                        gen(BOND_DIM * LOCAL_DIM * BOND_DIM),
                    )
                    .unwrap();
                    prev_bond = next_bond;
                    tensor
                };
                tensors.push(tensor);
                node_labels.push(next_node);
                indices.push(s);
                if arm == 0 && is_tip {
                    center = next_node;
                }
                next_node += 1;
            }
        }

        let tree = TreeTN::<_, usize>::from_tensors(tensors, node_labels).unwrap();
        (tree, indices, center)
    }

    fn starts(n_sites: usize) -> Vec<Vec<usize>> {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        (0..NSEARCH)
            .map(|_| {
                (0..n_sites)
                    .map(|_| rng.random_range(0..LOCAL_DIM))
                    .collect()
            })
            .collect()
    }
    let site_dims = vec![LOCAL_DIM; N_SITES];

    fn timed_walk(
        mut evaluator: TreeTNCachedEvaluator<'_, usize>,
        n_sites: usize,
    ) -> std::time::Duration {
        let warm_values = vec![0usize; n_sites];
        let warm_shape = [n_sites, 1];
        let warm_points = ColMajorArrayRef::new(&warm_values, &warm_shape).unwrap();
        evaluator.evaluate_batched(warm_points).unwrap();
        let start_time = Instant::now();
        for start in &starts(n_sites) {
            floating_zone_walk(
                &vec![LOCAL_DIM; n_sites],
                start,
                MAX_SWEEPS,
                f64::INFINITY,
                |_scan_site: Option<usize>, points: &[Vec<usize>]| -> Result<Vec<f64>> {
                    let mut values = vec![0usize; n_sites * points.len()];
                    for (p, point) in points.iter().enumerate() {
                        for (site, &v) in point.iter().enumerate() {
                            values[site + n_sites * p] = v;
                        }
                    }
                    let shape = [n_sites, points.len()];
                    let arr = ColMajorArrayRef::new(&values, &shape).unwrap();
                    let out = evaluator.evaluate_batched(arr)?;
                    Ok(out.iter().map(|v| v.real().abs()).collect())
                },
            )
            .unwrap();
        }
        start_time.elapsed()
    }

    let (chain_tree, chain_indices) = build_chain(7);
    let chain_evaluator = TreeTNCachedEvaluator::new(
        &chain_tree,
        &chain_indices,
        CachedEvaluatorOptions::default(),
    )
    .unwrap();
    #[cfg(feature = "diagnostics")]
    contraction_diagnostics::reset_all();
    let chain_elapsed = timed_walk(chain_evaluator, N_SITES);
    #[cfg(feature = "diagnostics")]
    let chain_diagnostics = contraction_diagnostics::summary();
    #[cfg(not(feature = "diagnostics"))]
    let chain_diagnostics = "diagnostics feature disabled";
    let _ = &site_dims;

    let (comb_tree, comb_indices, comb_center) = build_comb(7);
    let comb_evaluator = TreeTNCachedEvaluator::new(
        &comb_tree,
        &comb_indices,
        CachedEvaluatorOptions {
            center: Some(comb_center),
            ..Default::default()
        },
    )
    .unwrap();
    #[cfg(feature = "diagnostics")]
    contraction_diagnostics::reset_all();
    let comb_elapsed = timed_walk(comb_evaluator, N_SITES);
    #[cfg(feature = "diagnostics")]
    let comb_diagnostics = contraction_diagnostics::summary();
    #[cfg(not(feature = "diagnostics"))]
    let comb_diagnostics = "diagnostics feature disabled";

    println!(
        "chain (N={N_SITES}, bond={BOND_DIM}): {chain_elapsed:?}\n\
         comb  (N={N_SITES}, bond={BOND_DIM}, 1 hub): {comb_elapsed:?}\n\
         ratio (comb/chain): {:.2}x\n\
         chain diagnostics: {chain_diagnostics}\n\
         comb diagnostics: {comb_diagnostics}",
        comb_elapsed.as_secs_f64() / chain_elapsed.as_secs_f64()
    );
    assert!(chain_elapsed.as_nanos() > 0);
    assert!(comb_elapsed.as_nanos() > 0);
}

/// Where a hinted two-point Guard evaluation spends its time on a chain
/// and on a branched tree of the same site count (tensor4all-rs #727).
///
/// The Guard walks one site at a time and hints the varying site as the
/// centre, so every call is two points that differ in one coordinate. On a
/// 13-site chain that call costs about 20 us; on a 13-site spider it costs
/// about 390 us, and the spider's cost does not move when its bonds shrink
/// from `8,8,8,8` to `2,2,2,2` -- so the gap is not arithmetic. This
/// attributes it. Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn diagnostic_hinted_walk_cost_on_a_chain_versus_a_branched_tree() {
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    const N_SITES: usize = 13;
    const LOCAL_DIM: usize = 2;
    const SWEEPS: usize = 5;
    const ARM_LENGTH: usize = 3;

    fn chain(bond_dim: usize) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        let physical = (0..N_SITES)
            .map(|_| DynIndex::new_dyn(LOCAL_DIM))
            .collect::<Vec<_>>();
        let bonds = (0..N_SITES - 1)
            .map(|_| DynIndex::new_dyn(bond_dim))
            .collect::<Vec<_>>();
        let mut tensors = Vec::with_capacity(N_SITES);
        for site in 0..N_SITES {
            let mut indices = vec![physical[site].clone()];
            if site > 0 {
                indices.push(bonds[site - 1].clone());
            }
            if site + 1 < N_SITES {
                indices.push(bonds[site].clone());
            }
            let len: usize = indices.iter().map(IndexLike::dim).product();
            tensors.push(
                IdxTensor::from_dense(
                    indices,
                    (0..len)
                        .map(|flat| 1.0 + (flat % 7) as f64)
                        .collect::<Vec<f64>>(),
                )
                .unwrap(),
            );
        }
        (
            TreeTN::from_tensors(tensors, (0..N_SITES).collect()).unwrap(),
            physical,
        )
    }

    fn spider(bond_dim: usize) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        let arms = 4;
        assert_eq!(arms * ARM_LENGTH + 1, N_SITES);
        let physical = (0..N_SITES)
            .map(|_| DynIndex::new_dyn(LOCAL_DIM))
            .collect::<Vec<_>>();
        let hub_bonds = (0..arms)
            .map(|_| DynIndex::new_dyn(bond_dim))
            .collect::<Vec<_>>();
        let arm_bonds = (0..arms)
            .map(|_| {
                (1..ARM_LENGTH)
                    .map(|_| DynIndex::new_dyn(bond_dim))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let dense = |indices: Vec<DynIndex>| {
            let len: usize = indices.iter().map(IndexLike::dim).product();
            IdxTensor::from_dense(
                indices,
                (0..len)
                    .map(|flat| 1.0 + (flat % 7) as f64)
                    .collect::<Vec<f64>>(),
            )
            .unwrap()
        };
        let mut hub_indices = vec![physical[0].clone()];
        hub_indices.extend(hub_bonds.iter().cloned());
        let mut tensors = vec![dense(hub_indices)];
        for arm in 0..arms {
            for position in 0..ARM_LENGTH {
                let node = 1 + arm * ARM_LENGTH + position;
                let mut indices = vec![physical[node].clone()];
                if position == 0 {
                    indices.push(hub_bonds[arm].clone());
                } else {
                    indices.push(arm_bonds[arm][position - 1].clone());
                }
                if position + 1 < ARM_LENGTH {
                    indices.push(arm_bonds[arm][position].clone());
                }
                tensors.push(dense(indices));
            }
        }
        (
            TreeTN::from_tensors(tensors, (0..N_SITES).collect()).unwrap(),
            physical,
        )
    }

    for (name, (tree, physical)) in [
        ("chain_bond8", chain(8)),
        ("spider_bond2", spider(2)),
        ("spider_bond8", spider(8)),
    ] {
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &physical, CachedEvaluatorOptions::default())
                .unwrap();
        let warm = vec![0usize; N_SITES];
        evaluator
            .evaluate_batched_typed::<f64>(
                ColMajorArrayRef::new(&warm, &[N_SITES, 1]).unwrap(),
                EvaluationHint::default(),
            )
            .unwrap();

        phase_timing::reset_all();
        let mut calls = 0usize;
        let started = Instant::now();
        for sweep in 0..SWEEPS {
            for site in 0..N_SITES {
                let mut values = vec![0usize; N_SITES * LOCAL_DIM];
                for point in 0..LOCAL_DIM {
                    for other in 0..N_SITES {
                        values[other + N_SITES * point] = if other == site {
                            point
                        } else {
                            (other + sweep) % LOCAL_DIM
                        };
                    }
                }
                let batch = ColMajorArrayRef::new(&values, &[N_SITES, LOCAL_DIM]).unwrap();
                std::hint::black_box(
                    evaluator
                        .evaluate_batched_typed::<f64>(batch, EvaluationHint::around(site))
                        .unwrap(),
                );
                calls += 1;
            }
        }
        let total_ns = started.elapsed().as_nanos() as u64;
        let per_call = |counter: u64| counter as f64 / calls as f64 / 1.0e3;
        let stats = evaluator.stats_for_test();
        eprintln!(
            "hinted walk {name}: total={:.1}us/call over {calls} calls | environment={:.1}us {{capability={:.1}, assignments={:.1}, message_loop={:.1}, component_assembly={:.1}}}, centre={:.1}us, key_lookup={:.1}us, reconstruct={:.1}us, contract={:.1}us, insert={:.1}us, plan_build={:.1}us, layout_build={:.1}us, plans={}, cache hits/misses={}/{}",
            per_call(total_ns),
            per_call(phase_timing::BUILD_ENV_NS.load(Ordering::Relaxed)),
            per_call(phase_timing::RAW_CAPABILITY_NS.load(Ordering::Relaxed)),
            per_call(phase_timing::ASSIGNMENT_BATCH_NS.load(Ordering::Relaxed)),
            per_call(phase_timing::MESSAGE_LOOP_NS.load(Ordering::Relaxed)),
            per_call(phase_timing::COMPONENT_ASSEMBLY_NS.load(Ordering::Relaxed)),
            per_call(phase_timing::CENTER_NS.load(Ordering::Relaxed)),
            per_call(phase_timing::KEY_AND_LOOKUP_NS.load(Ordering::Relaxed)),
            per_call(phase_timing::RECONSTRUCT_NS.load(Ordering::Relaxed)),
            per_call(phase_timing::CONTRACT_NS.load(Ordering::Relaxed)),
            per_call(phase_timing::INSERT_NS.load(Ordering::Relaxed)),
            per_call(phase_timing::PLAN_BUILD_NS.load(Ordering::Relaxed)),
            per_call(phase_timing::LAYOUT_BUILD_NS.load(Ordering::Relaxed)),
            phase_timing::PLAN_COUNT.load(Ordering::Relaxed),
            stats.message_cache_hits,
            stats.message_cache_misses,
        );
    }
}
