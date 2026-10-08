use super::{optimize_default, TreeTciOptimizationResult, TreeTciOptions, TreeTciTermination};
use crate::test_support::assert_scalar_close;
use crate::{GlobalIndexBatch, TreeTCI2, TreeTciEdge, TreeTciGraph};
use anyhow::Result;
use tensor4all_core::IndexLike;

#[test]
fn accepted_global_pivots_prevent_sampled_convergence() {
    // Deliberately restrict local updates to the initial cross. Its local
    // pivot error is zero, but f = 1 + i + j has residual i*j against the
    // rank-1 approximation (1+i)(1+j). Global search must keep the run open.
    struct InitialCrossOnly;

    impl crate::PivotCandidateProposer for InitialCrossOnly {
        fn candidates_with_rng<T, R: rand::Rng + ?Sized>(
            &self,
            _state: &TreeTCI2<T>,
            _edge: TreeTciEdge,
            _rng: &mut R,
        ) -> crate::TreeTciResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
            Ok((vec![vec![0]], vec![vec![0]]))
        }
    }

    for enable_global_pivots in [false, true] {
        let mut state = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
        state.add_global_pivots(&[vec![0, 0]]).unwrap();
        state.max_sample_value = 1.0;
        let residual_points_seen = std::cell::Cell::new(0);
        let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
            Ok((0..batch.n_points())
                .map(|point| {
                    let i = batch.get(0, point).unwrap();
                    let j = batch.get(1, point).unwrap();
                    if i == 1 && j == 1 {
                        residual_points_seen.set(residual_points_seen.get() + 1);
                    }
                    (1 + i + j) as f64
                })
                .collect())
        };
        let options = TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 4,
            enable_global_pivots,
            nsearch: 20,
            seed: Some(7),
            ..Default::default()
        };
        let result =
            crate::optimize_with_proposer(&mut state, evaluate, &options, &InitialCrossOnly)
                .unwrap();
        let (reason, iterations) = if enable_global_pivots {
            assert!(residual_points_seen.get() > 0);
            (crate::TreeTciTermination::MaxIterations, 4)
        } else {
            assert_eq!(residual_points_seen.get(), 0);
            (crate::TreeTciTermination::Converged, 3)
        };
        assert_eq!(result.termination, reason);
        assert_eq!(result.ranks, vec![1; iterations]);
        assert_eq!(result.errors, vec![0.0; iterations]);
        assert_swept_pivot_sets(&state, usize::MAX);
    }
}

fn two_site_graph() -> TreeTciGraph {
    TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap()
}

#[test]
fn optimize_rejects_invalid_options_before_callback() {
    let invalid_tolerances = [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    for tolerance in invalid_tolerances {
        let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
        tci.add_global_pivots(&[vec![0, 0]]).unwrap();
        let calls = std::cell::Cell::new(0);
        let options = TreeTciOptions {
            tolerance,
            ..TreeTciOptions::default()
        };
        let error = optimize_default(
            &mut tci,
            |_| {
                calls.set(calls.get() + 1);
                Ok(vec![1.0])
            },
            &options,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            crate::TreeTciError::InvalidConfiguration { .. }
        ));
        assert_eq!(calls.get(), 0);
    }

    for options in [
        TreeTciOptions {
            max_iter: 0,
            ..TreeTciOptions::default()
        },
        TreeTciOptions {
            max_bond_dim: Some(0),
            ..TreeTciOptions::default()
        },
        TreeTciOptions {
            tol_margin_global_search: -1.0,
            ..TreeTciOptions::default()
        },
        TreeTciOptions {
            tol_margin_global_search: f64::NAN,
            ..TreeTciOptions::default()
        },
        TreeTciOptions {
            tol_margin_global_search: f64::INFINITY,
            ..TreeTciOptions::default()
        },
        TreeTciOptions {
            tol_margin_global_search: f64::NEG_INFINITY,
            ..TreeTciOptions::default()
        },
    ] {
        let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
        tci.add_global_pivots(&[vec![0, 0]]).unwrap();
        let error = optimize_default(&mut tci, |_| Ok(vec![1.0]), &options).unwrap_err();
        assert!(matches!(
            error,
            crate::TreeTciError::InvalidConfiguration { .. }
        ));
    }
}

#[test]
fn api_crossinterpolate2_rejects_invalid_options_before_callback() {
    let calls = std::cell::Cell::new(0);
    let error = crate::crossinterpolate2::<f64, _, _>(
        |_| {
            calls.set(calls.get() + 1);
            Ok(vec![1.0])
        },
        vec![2, 2],
        two_site_graph(),
        vec![vec![0, 0]],
        TreeTciOptions {
            tolerance: f64::NAN,
            ..TreeTciOptions::default()
        },
        None,
        &crate::DefaultProposer,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::TreeTciError::InvalidConfiguration { .. }
    ));
    assert_eq!(calls.get(), 0);
}

#[test]
fn optimize_default_converges_on_two_site_identity() {
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    let TreeTciOptimizationResult {
        ranks,
        errors,
        termination,
        ..
    } = optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 4,
            max_bond_dim: None,
            normalize_error: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(termination, TreeTciTermination::Converged);
    assert_eq!(ranks.last().copied(), Some(2));
    assert_scalar_close(
        errors.last().copied().unwrap_or(f64::NAN),
        0.0,
        tci.max_sample_value,
        1e-12,
    );
    assert_eq!(tci.max_bond_dim(), 2);
}

// Regression for the early-convergence stop documented in
// docs/design/treetci-termination.md: an already-converged run must stop
// after the confirmation window rather than exhausting max_iter.
#[test]
fn optimize_default_stops_early_once_converged() {
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    let TreeTciOptimizationResult {
        ranks,
        errors,
        termination,
        ..
    } = optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 4,
            max_bond_dim: None,
            normalize_error: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(termination, TreeTciTermination::Converged);
    // The 2x2 identity function is exactly rank 2 and converges on the first
    // sweep; the loop must not keep going through the remaining max_iter-1
    // sweeps once the error is already below tolerance.
    assert!(ranks.len() < 4);
    assert_eq!(ranks.len(), errors.len());
    assert_eq!(ranks.last().copied(), Some(2));
}

// Mirrors `TreeTCI.jl`'s `convergencecriterion` third disjunct
// (`all(lastranks .>= max_bond_dim)`, branch `local-fix-convergence`, commit
// 06563dd): once the rank has saturated at `max_bond_dim` for the trailing
// window, further sweeps cannot lower the error, so the loop should stop
// even though the error never crosses `tolerance`.
#[test]
fn optimize_default_stops_early_when_bond_dim_saturated() {
    let mut tci = TreeTCI2::<f64>::new(vec![3, 3], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    let TreeTciOptimizationResult {
        ranks,
        errors,
        termination,
        ..
    } = optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 10,
            max_bond_dim: Some(1),
            normalize_error: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(termination, TreeTciTermination::MaxBondDimension);

    // Rank-3 identity capped at max_bond_dim = 1 can never reach the 1e-12
    // tolerance; without the bond-dim-saturation criterion this would run
    // all 10 sweeps.
    assert!(ranks.len() < 10);
    assert!(ranks.iter().all(|&r| r <= 1));
    assert!(errors.last().copied().unwrap_or(0.0) > 1e-12);
}

// Convergence needs three sweeps of history, so a budget of two sweeps can
// only end at the iteration limit, even on an exactly representable function.
#[test]
fn optimize_default_reports_iteration_limit() {
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();
    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        Ok((0..batch.n_points())
            .map(|p| (batch.get(0, p).unwrap() + batch.get(1, p).unwrap() + 1) as f64)
            .collect())
    };

    let report = optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 2,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(report.termination, TreeTciTermination::MaxIterations);
    assert_eq!(report.ranks, vec![2, 2]);
    assert_eq!(report.errors.len(), 2);
}

// A cap equal to the exact rank: both stops hold after three sweeps, and the
// saturation stop is checked first.
#[test]
fn optimize_default_prefers_saturation_when_rank_equals_cap() {
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();
    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        Ok((0..batch.n_points())
            .map(|p| (batch.get(0, p).unwrap() + batch.get(1, p).unwrap() + 1) as f64)
            .collect())
    };

    let report = optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 10,
            max_bond_dim: Some(2),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(report.termination, TreeTciTermination::MaxBondDimension);
    assert_eq!(report.ranks, vec![2, 2, 2]);
    assert!(report.errors.iter().all(|&error| error < 1e-12));
}

/// Number of distinct multi-indices on the subtree `key`.
fn subtree_dim_product<T>(state: &TreeTCI2<T>, key: &crate::SubtreeKey) -> usize {
    key.as_slice()
        .iter()
        .map(|&site| state.local_dims[site])
        .product()
}

/// Check the invariants every finished optimization must leave behind: on
/// each edge, both sides hold the same number of pivots, at least one and at
/// most `min(max_bond_dim, maximal achievable rank of the edge)`.
fn assert_swept_pivot_sets<T>(state: &TreeTCI2<T>, max_bond_dim: usize) {
    for edge in state.graph.edges() {
        let (left_key, right_key) = state.graph.subregion_vertices(edge).unwrap();
        let left = crate::ncols_2d(&state.ijset[&left_key]).unwrap();
        let right = crate::ncols_2d(&state.ijset[&right_key]).unwrap();
        assert_eq!(left, right, "pivot counts differ across edge {edge:?}");
        let max_rank =
            subtree_dim_product(state, &left_key).min(subtree_dim_product(state, &right_key));
        assert!(
            left >= 1 && left <= max_bond_dim.min(max_rank),
            "edge {edge:?} holds {left} pivots; cap {max_bond_dim}, maximal rank {max_rank}"
        );
    }
}

/// Seed a state the way `crossinterpolate2` does, run `optimize_default`, and
/// check that the capped run stopped cleanly through the saturation stop on a
/// swept, materializable state within the cap.
fn run_capped_and_check<F>(
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: &[Vec<usize>],
    evaluate: F,
    options: &TreeTciOptions,
) where
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<f64>>,
{
    let cap = options.max_bond_dim.unwrap();
    let mut tci = TreeTCI2::<f64>::new(local_dims, graph).unwrap();
    tci.add_global_pivots(initial_pivots).unwrap();
    let n_sites = tci.local_dims.len();
    let flat: Vec<usize> = initial_pivots.iter().flatten().copied().collect();
    let batch = GlobalIndexBatch::new(&flat, n_sites, initial_pivots.len()).unwrap();
    tci.max_sample_value = evaluate(batch)
        .unwrap()
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));

    let TreeTciOptimizationResult {
        ranks,
        errors,
        termination,
        ..
    } = optimize_default(&mut tci, &evaluate, options).unwrap();
    assert_eq!(termination, TreeTciTermination::MaxBondDimension);

    // The cap is below the function's rank, so the loop must have stopped
    // through the bond-dimension saturation stop, not through convergence or
    // `max_iter`.
    assert_eq!(ranks, vec![cap; 3]);
    assert_eq!(errors.len(), ranks.len());
    assert_swept_pivot_sets(&tci, cap);

    let treetn = crate::to_treetn(&tci, &evaluate, None).unwrap();
    for edge in tci.graph.edges() {
        let (left_key, _) = tci.graph.subregion_vertices(edge).unwrap();
        let expected = crate::ncols_2d(&tci.ijset[&left_key]).unwrap();
        let (u, v) = tci.graph.separate_vertices(edge).unwrap();
        let bond = treetn
            .bond_index(treetn.edge_between(&u, &v).unwrap())
            .unwrap();
        assert_eq!(bond.dim(), expected, "bond dimension of {edge:?}");
    }
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

/// Pseudo-random values on a ~3% support of a local-dimension-4 index space
/// (the function from issue #692).
fn sparse_value(point: &[usize]) -> f64 {
    let key = point.iter().fold(0u64, |key, &v| key * 4 + v as u64);
    let h = mix64(key);
    if h % 32 < 1 {
        ((h >> 8) as f64 / u64::MAX as f64) - 0.5
    } else {
        0.0
    }
}

// Regression for #692: on a sparse function the global pivot search keeps
// finding missed support. Its pivots used to be injected after the sweep that
// completed the bond-dimension saturation window, after which the loop
// stopped, so the returned pivot sets were unswept: on edge (0, 1) the site-0
// side stayed at 4 (deduplicated) while the other side grew past 4, and
// materialization failed with "bond ranks disagree across edge".
#[test]
fn capped_sparse_chain_with_global_pivots_stops_on_swept_state() {
    const N_SITES: usize = 5;
    let mut pivots = Vec::new();
    let mut key = 1u64;
    while pivots.len() < 16 {
        key = mix64(key).max(1);
        let point: Vec<usize> = (0..N_SITES)
            .map(|site| ((key >> (2 * site)) % 4) as usize)
            .collect();
        if sparse_value(&point) != 0.0 {
            pivots.push(point);
        }
    }
    let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut point = vec![0usize; N_SITES];
        (0..batch.n_points())
            .map(|p| {
                for (site, slot) in point.iter_mut().enumerate() {
                    *slot = batch.get(site, p).unwrap();
                }
                Ok(sparse_value(&point))
            })
            .collect()
    };
    let options = TreeTciOptions {
        tolerance: 1e-12,
        max_iter: 10,
        max_bond_dim: Some(8),
        nsearch: 10,
        max_nglobal_pivot: 10,
        seed: Some(1),
        ..Default::default()
    };

    run_capped_and_check(
        vec![4; N_SITES],
        TreeTciGraph::linear_chain(N_SITES).unwrap(),
        &pivots,
        evaluate,
        &options,
    );

    // The public entry point from the issue report succeeds as well.
    let crate::TreeTciRunResult { treetn, ranks, .. } = crate::crossinterpolate2::<f64, _, _>(
        evaluate,
        vec![4; N_SITES],
        TreeTciGraph::linear_chain(N_SITES).unwrap(),
        pivots,
        options,
        None,
        &crate::DefaultProposer,
    )
    .unwrap();
    assert!(ranks.iter().all(|&rank| rank <= 8));
    assert!(treetn.link_dims().iter().all(|&dim| dim <= 8));
}

// Regression for #692 on a branching tree: vertex 1 has degree 3 and the
// function has rank 3 across every cut, so `max_bond_dim = 2` saturates. With
// global pivots enabled, every seed used to stop right after an injection
// and fail with "bond ranks disagree across edge (0, 1): left 2, right 5".
#[test]
fn capped_star_tree_with_global_pivots_stops_on_swept_state() {
    let graph = TreeTciGraph::new(
        4,
        &[
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(1, 3),
        ],
    )
    .unwrap();
    let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        (0..batch.n_points())
            .map(|p| {
                let mut phase = 0.0;
                for (site, weight) in [0.3, 0.5, 0.7, 0.9].into_iter().enumerate() {
                    phase += weight * batch.get(site, p).unwrap() as f64;
                }
                Ok(phase.cos() + 0.1)
            })
            .collect()
    };
    for seed in 0..8 {
        let options = TreeTciOptions {
            tolerance: 1e-10,
            max_bond_dim: Some(2),
            seed: Some(seed),
            ..Default::default()
        };
        run_capped_and_check(
            vec![2, 4, 4, 4],
            graph.clone(),
            &[vec![0, 0, 0, 0]],
            evaluate,
            &options,
        );
    }
}

#[test]
fn optimizer_threads_the_caller_stream_through_each_edge_pass() {
    use crate::{DefaultProposer, PivotCandidateProposer};
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::cell::RefCell;
    struct RecordDraws(RefCell<Vec<u64>>);
    impl PivotCandidateProposer for RecordDraws {
        fn seed(&self) -> u64 {
            panic!("caller-stream path must not read the seed")
        }
        fn candidates_with_rng<T, R: rand::Rng + ?Sized>(
            &self,
            state: &TreeTCI2<T>,
            edge: TreeTciEdge,
            rng: &mut R,
        ) -> crate::TreeTciResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
            self.0.borrow_mut().push(rng.next_u64());
            DefaultProposer.candidates_with_rng(state, edge, rng)
        }
    }
    let proposer = RecordDraws(RefCell::new(Vec::new()));
    let mut state = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    state.add_global_pivots(&[vec![0, 0]]).unwrap();
    state.max_sample_value = 1.0;
    let options = TreeTciOptions {
        enable_global_pivots: false,
        max_iter: 2,
        ..Default::default()
    };
    let evaluate = |batch: GlobalIndexBatch<'_>| Ok(vec![1.0; batch.n_points()]);
    let mut rng = ChaCha8Rng::seed_from_u64(73);
    let mut reference = rng.clone();
    super::optimize_with_proposer_with_rng(&mut state, evaluate, &options, &proposer, &mut rng)
        .unwrap();
    assert_eq!(
        *proposer.0.borrow(),
        (0..4).map(|_| reference.next_u64()).collect::<Vec<_>>()
    );
    assert_eq!(rng.next_u64(), reference.next_u64());
}
