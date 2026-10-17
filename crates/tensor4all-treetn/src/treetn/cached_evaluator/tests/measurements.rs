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
            let mut tensor = tensor3_zeros(left_dim, LOCAL_DIM, right_dim).unwrap();
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
            let mut tensor = tensor3_zeros(left_dim, LOCAL_DIM, right_dim).unwrap();
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
