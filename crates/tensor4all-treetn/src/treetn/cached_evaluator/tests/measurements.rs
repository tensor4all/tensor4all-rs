//! Wall-time and work measurements on realistic floating-zone walks. The
//! `diagnostic_*` tests are ignored measurement tools, not regression checks.

use super::*;

// Preserve the old fixture's (left, site, right) RNG draw order, while building
// the same column-major chain directly without the optional SimpleTT bridge.
pub(super) fn random_measurement_chain(
    n_sites: usize,
    local_dim: usize,
    bond_dim: usize,
    seed: u64,
) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use tensor4all_core::ColMajorArray;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let cores = (0..n_sites)
        .map(|site| {
            let left = if site == 0 { 1 } else { bond_dim };
            let right = if site == n_sites - 1 { 1 } else { bond_dim };
            let mut data = vec![0.0; left * local_dim * right];
            for l in 0..left {
                for s in 0..local_dim {
                    for r in 0..right {
                        data[l + left * (s + local_dim * r)] = rng.random::<f64>();
                    }
                }
            }
            ColMajorArray::new(data, vec![left, local_dim, right]).unwrap()
        })
        .collect();
    TreeTN::from_chain_cores(cores).unwrap()
}

#[cfg(feature = "simplett-bridge")]
#[test]
fn native_measurement_fixture_preserves_legacy_values() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use tensor4all_simplett::{tensor3_zeros, SimpleTensorTrain, Tensor3Ops};

    // Reproduce the original SimpleTT fixture at small, non-binary dimensions
    // to catch changes in RNG draw order, flat layout, and boundary cores.
    for n_sites in [1, 3, 4] {
        let (local_dim, bond_dim, seed) = (3, 2, 7);
        let (native, sites) = random_measurement_chain(n_sites, local_dim, bond_dim, seed);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let tensors = (0..n_sites)
            .map(|site| {
                let left = if site == 0 { 1 } else { bond_dim };
                let right = if site == n_sites - 1 { 1 } else { bond_dim };
                let mut tensor = tensor3_zeros(left, local_dim, right).unwrap();
                for l in 0..left {
                    for s in 0..local_dim {
                        for r in 0..right {
                            tensor.set3(l, s, r, rng.random::<f64>());
                        }
                    }
                }
                tensor
            })
            .collect();
        let legacy = SimpleTensorTrain::new(tensors).unwrap();
        let (values, _) = legacy.full_tensor().unwrap();
        let expected = IdxTensor::from_dense(sites, values).unwrap();
        let actual = native.contract_to_tensor().unwrap();
        let residual = actual.sub(&expected).unwrap().maxabs().unwrap();
        assert!(
            residual < 1.0e-12,
            "fixture residual={residual}, sites={n_sites}"
        );
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

    const N_SITES: usize = 16;
    const LOCAL_DIM: usize = 2;
    const BOND_DIM: usize = 128;
    const NSEARCH: usize = 5;
    const MAX_SWEEPS: usize = 100;
    const ARM_LEN: usize = 5; // hub + 3*5 = 16 sites, matching N_SITES

    fn build_chain(seed: u64) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        random_measurement_chain(N_SITES, LOCAL_DIM, BOND_DIM, seed)
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
