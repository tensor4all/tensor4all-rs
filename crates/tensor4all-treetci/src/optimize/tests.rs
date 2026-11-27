use super::{
    optimize_default, StopRule, TreeTciOptimizationResult, TreeTciOptions, TreeTciTermination,
};
use crate::test_support::assert_scalar_close;
use crate::{
    AllEdges, DefaultProposer, EdgeVisitor, GlobalIndexBatch, PivotCandidateProposer, TreeTCI2,
    TreeTciEdge, TreeTciGraph,
};
use anyhow::Result;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};

/// A proposer that offers only the all-zero cross on each side, so local
/// updates can never grow a bond past rank 1.
struct InitialCrossOnly;

impl crate::PivotCandidateProposer for InitialCrossOnly {
    fn candidates_with_rng<T, R: rand::Rng + ?Sized>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
        _rng: &mut R,
    ) -> crate::TreeTciResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        let (left, right) = state.graph.subregion_vertices(edge)?;
        Ok((
            vec![vec![0; left.as_slice().len()]],
            vec![vec![0; right.as_slice().len()]],
        ))
    }
}

#[test]
fn accepted_global_pivots_prevent_sampled_convergence() {
    // Deliberately restrict local updates to the initial cross. Its local
    // pivot error is zero, but f = 1 + i + j has residual i*j against the
    // rank-1 approximation (1+i)(1+j). Global search must keep the run open.
    // Without it, the zero error comes from a 1 x 1 candidate matrix and is
    // not a measurement, so the run must not report `Converged` (#1153).
    // Under `Sampled` the recorded error includes the residual the search
    // measured (#1352); under `PerBond` it is the bond error alone, and the
    // accepted pivots keep the run open (#1405).
    #[derive(Default)]
    struct Observed(Vec<(f64, crate::TreeTciEvaluationStats)>);
    impl crate::TreeTciObserver for Observed {
        fn iteration(&mut self, it: &crate::TreeTciIteration<'_>) {
            self.0.push((it.error, *it.evaluation));
        }
    }

    for (stop_rule, enable_global_pivots) in [
        (StopRule::Sampled, false),
        (StopRule::Sampled, true),
        (StopRule::PerBond, false),
        (StopRule::PerBond, true),
    ] {
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
            stop_rule,
            max_iter: 4,
            rank_slack: 0.03,
            enable_global_pivots,
            nsearch: 20,
            seed: Some(7),
            ..Default::default()
        };
        let mut observed = Observed::default();
        let result = crate::optimize_with_proposer(
            &mut state,
            evaluate,
            &options,
            &InitialCrossOnly,
            &crate::AllEdges,
            &mut observed,
        )
        .unwrap();
        let (reason, iterations) = if enable_global_pivots {
            assert!(residual_points_seen.get() > 0);
            (crate::TreeTciTermination::MaxIterations, 4)
        } else {
            assert_eq!(residual_points_seen.get(), 0);
            (crate::TreeTciTermination::DegenerateCandidateSet, 3)
        };
        assert_eq!(result.termination, reason);
        assert_eq!(result.ranks, vec![1; iterations]);
        let expected_errors = if enable_global_pivots && stop_rule == StopRule::Sampled {
            // The global search actually measured residual one. The final
            // iteration skips search, so only its local zero is available.
            vec![1.0, 1.0, 1.0, 0.0]
        } else {
            vec![0.0; iterations]
        };
        assert_eq!(result.errors, expected_errors, "{stop_rule:?}");
        assert_eq!(result.nglobal_pivots.len(), iterations);
        if enable_global_pivots {
            assert!(result.nglobal_pivots[..iterations - 1]
                .iter()
                .all(|&n| n > 0));
            assert_eq!(result.nglobal_pivots[iterations - 1], 0);
        } else {
            assert_eq!(result.nglobal_pivots, vec![0; iterations]);
        }
        assert_eq!(
            observed
                .0
                .iter()
                .map(|(error, _)| *error)
                .collect::<Vec<_>>(),
            result.errors
        );
        assert_eq!(observed.0.last().unwrap().1, result.evaluation);
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
            ncheck_history: 0,
            ..TreeTciOptions::default()
        },
        TreeTciOptions {
            rank_slack: -0.01,
            ..TreeTciOptions::default()
        },
        TreeTciOptions {
            rank_slack: f64::NAN,
            ..TreeTciOptions::default()
        },
        TreeTciOptions {
            rank_slack: f64::INFINITY,
            ..TreeTciOptions::default()
        },
        TreeTciOptions {
            rank_slack: f64::NEG_INFINITY,
            ..TreeTciOptions::default()
        },
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
        let calls = std::cell::Cell::new(0);
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
}

#[test]
fn api_crossinterpolate2_rejects_invalid_options_before_callback() {
    let calls = std::cell::Cell::new(0);
    let error = crate::crossinterpolate2::<f64, _, _, _, _>(
        |_| {
            calls.set(calls.get() + 1);
            Ok::<_, anyhow::Error>(vec![1.0])
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
        &mut (),
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
            scale_tolerance: true,
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
            scale_tolerance: true,
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
            scale_tolerance: true,
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
    let crate::TreeTciRunResult { treetn, ranks, .. } =
        crate::crossinterpolate2::<f64, _, _, _, _>(
            evaluate,
            vec![4; N_SITES],
            TreeTciGraph::linear_chain(N_SITES).unwrap(),
            pivots,
            options,
            None,
            &crate::DefaultProposer,
            &mut (),
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
    super::optimize_with_proposer_with_rng(
        &mut state,
        evaluate,
        &options,
        &proposer,
        &mut (),
        &mut rng,
    )
    .unwrap();
    assert_eq!(
        *proposer.0.borrow(),
        (0..4).map(|_| reference.next_u64()).collect::<Vec<_>>()
    );
    assert_eq!(rng.next_u64(), reference.next_u64());
}

/// Branching tree used by the edge-order regressions: centre 1 with leaves
/// 0, 2, 3. Sorted edges are (0,1), (1,2), (1,3).
fn branching_graph() -> TreeTciGraph {
    TreeTciGraph::new(
        4,
        &[
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(1, 3),
        ],
    )
    .unwrap()
}

/// A sum of single-site terms, exactly rank 2 across every cut.
fn branching_value(point: &[usize]) -> f64 {
    1.0 + point[0] as f64 + 2.0 * point[1] as f64 + 3.0 * point[2] as f64 + 4.0 * point[3] as f64
}

fn branching_evaluate(batch: GlobalIndexBatch<'_>) -> Result<Vec<f64>> {
    let mut point = vec![0usize; batch.n_sites()];
    let mut values = Vec::with_capacity(batch.n_points());
    for p in 0..batch.n_points() {
        for (site, slot) in point.iter_mut().enumerate() {
            *slot = batch.get(site, p).unwrap();
        }
        values.push(branching_value(&point));
    }
    Ok(values)
}

/// Dense reference on the network's own site indices, in site order.
fn dense_branching_reference(tn: &tensor4all_treetn::TreeTN<IdxTensor, usize>) -> IdxTensor {
    let local_dims = [2usize; 4];
    let indices: Vec<DynIndex> = (0..4)
        .map(|site| tn.site_space(&site).unwrap().iter().next().unwrap().clone())
        .collect();
    let total: usize = local_dims.iter().product();
    let mut point = vec![0usize; 4];
    let mut values = Vec::with_capacity(total);
    for _ in 0..total {
        values.push(branching_value(&point));
        for (slot, &dim) in point.iter_mut().zip(&local_dims) {
            *slot += 1;
            if *slot < dim {
                break;
            }
            *slot = 0;
        }
    }
    IdxTensor::from_dense(indices, values).unwrap()
}

/// A visitor that always returns a fixed edge sequence.
struct FixedOrder(Vec<TreeTciEdge>);

impl EdgeVisitor for FixedOrder {
    fn visit_order<T>(&self, _state: &TreeTCI2<T>) -> Vec<TreeTciEdge> {
        self.0.clone()
    }
}

/// Records the edge of every proposer call, then delegates to the default.
struct RecordingProposer(std::cell::RefCell<Vec<TreeTciEdge>>);

impl PivotCandidateProposer for RecordingProposer {
    fn seed(&self) -> u64 {
        DefaultProposer.seed()
    }

    fn candidates_with_rng<T, R: rand::Rng + ?Sized>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
        rng: &mut R,
    ) -> crate::TreeTciResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        self.0.borrow_mut().push(edge);
        DefaultProposer.candidates_with_rng(state, edge, rng)
    }
}

fn branching_options() -> TreeTciOptions {
    TreeTciOptions {
        tolerance: 1e-12,
        max_iter: 5,
        enable_global_pivots: false,
        seed: Some(0),
        ..Default::default()
    }
}

/// Passing `AllEdges` explicitly is bit-identical to the default entry point.
#[test]
fn explicit_all_edges_matches_the_default_optimizer() {
    let evaluate = |batch: GlobalIndexBatch<'_>| Ok(vec![1.0; batch.n_points()]);

    let mut default_state = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    default_state.add_global_pivots(&[vec![0, 0]]).unwrap();
    default_state.max_sample_value = 1.0;
    let default =
        optimize_default(&mut default_state, evaluate, &TreeTciOptions::default()).unwrap();

    let mut visitor_state = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    visitor_state.add_global_pivots(&[vec![0, 0]]).unwrap();
    visitor_state.max_sample_value = 1.0;
    let explicit = crate::optimize_with_proposer(
        &mut visitor_state,
        evaluate,
        &TreeTciOptions::default(),
        &DefaultProposer,
        &AllEdges,
        &mut (),
    )
    .unwrap();

    assert_eq!(explicit.ranks, default.ranks);
    assert_eq!(explicit.errors, default.errors);
    assert_eq!(explicit.termination, default.termination);
    for edge in default_state.graph.edges() {
        let (left, right) = default_state.graph.subregion_vertices(edge).unwrap();
        assert_eq!(default_state.ijset[&left], visitor_state.ijset[&left]);
        assert_eq!(default_state.ijset[&right], visitor_state.ijset[&right]);
    }
}

/// A caller-chosen permutation is visited exactly and converges to the dense
/// function at `1e-12 * max|f|`.
#[test]
fn custom_edge_order_is_visited_and_converges() {
    let custom = vec![
        TreeTciEdge::new(1, 3),
        TreeTciEdge::new(1, 2),
        TreeTciEdge::new(0, 1),
    ];
    let proposer = RecordingProposer(std::cell::RefCell::new(Vec::new()));
    let mut state = TreeTCI2::<f64>::new(vec![2, 2, 2, 2], branching_graph()).unwrap();
    state.add_global_pivots(&[vec![0, 0, 0, 0]]).unwrap();
    state.max_sample_value = branching_value(&[0, 0, 0, 0]).abs();

    let result = crate::optimize_with_proposer(
        &mut state,
        branching_evaluate,
        &branching_options(),
        &proposer,
        &FixedOrder(custom.clone()),
        &mut (),
    )
    .unwrap();

    // Every inner pass records exactly the requested order.
    let recorded = proposer.0.borrow().clone();
    assert!(!recorded.is_empty());
    assert_eq!(recorded.len() % custom.len(), 0);
    for pass in recorded.chunks(custom.len()) {
        assert_eq!(pass, custom.as_slice());
    }
    assert_eq!(result.termination, TreeTciTermination::Converged);

    let tn = crate::to_treetn(&state, branching_evaluate, None).unwrap();
    let reference = dense_branching_reference(&tn);
    let residual = tn
        .to_dense()
        .unwrap()
        .sub(&reference)
        .unwrap()
        .maxabs()
        .unwrap();
    let scale = reference.maxabs().unwrap();
    assert!(
        residual <= 1e-12 * scale,
        "max abs residual {residual:e} exceeds 1e-12 * {scale:e}"
    );
}

/// A missing, duplicated, or unknown edge is rejected before any evaluation,
/// so no edge is left with a stale bond error.
#[test]
fn invalid_edge_order_is_rejected_before_mutation() {
    let cases: [(&str, Vec<TreeTciEdge>); 3] = [
        (
            "omitted",
            vec![TreeTciEdge::new(0, 1), TreeTciEdge::new(1, 2)],
        ),
        (
            "duplicated",
            vec![
                TreeTciEdge::new(0, 1),
                TreeTciEdge::new(0, 1),
                TreeTciEdge::new(1, 2),
                TreeTciEdge::new(1, 3),
            ],
        ),
        (
            "unknown",
            vec![
                TreeTciEdge::new(0, 1),
                TreeTciEdge::new(1, 2),
                TreeTciEdge::new(2, 3),
            ],
        ),
    ];

    for (name, order) in cases {
        let mut state = TreeTCI2::<f64>::new(vec![2, 2, 2, 2], branching_graph()).unwrap();
        state.add_global_pivots(&[vec![0, 0, 0, 0]]).unwrap();
        state.max_sample_value = 1.0;
        let before = state.max_bond_error();
        let calls = std::cell::Cell::new(0);

        let error = crate::optimize_with_proposer(
            &mut state,
            |batch: GlobalIndexBatch<'_>| {
                calls.set(calls.get() + 1);
                Ok(vec![1.0; batch.n_points()])
            },
            &branching_options(),
            &DefaultProposer,
            &FixedOrder(order),
            &mut (),
        )
        .unwrap_err();

        assert!(
            matches!(error, crate::TreeTciError::InvalidConfiguration { .. }),
            "{name}: unexpected error {error:?}"
        );
        assert_eq!(calls.get(), 0, "{name}: evaluator ran before validation");
        assert_eq!(state.max_bond_error(), before, "{name}: state mutated");
    }
}

#[test]
fn optimize_rejects_invalid_global_search_starts_before_callback() {
    let cases = [
        (
            vec![vec![0]],
            false,
            "global_search_starts[0] has 1 coordinates but the state has 2 sites",
        ),
        (
            vec![vec![0, 0, 0]],
            true,
            "global_search_starts[0] has 3 coordinates but the state has 2 sites",
        ),
        (
            vec![vec![0, 0], vec![0, 2]],
            true,
            "global_search_starts[1] has coordinate 2 at site 1, out of range for local dimension 2",
        ),
    ];
    for (starts, enable_global_pivots, needle) in cases {
        let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
        tci.add_global_pivots(&[vec![0, 0]]).unwrap();
        let calls = std::cell::Cell::new(0);
        let options = TreeTciOptions {
            enable_global_pivots,
            global_search_starts: starts,
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
        // The start set is checked against the state before any sweep, so the
        // message names the offending entry and the target was never called.
        assert!(error.to_string().contains(needle), "{error}");
        assert!(matches!(
            error,
            crate::TreeTciError::InvalidConfiguration { .. }
        ));
        assert_eq!(calls.get(), 0);
    }
}

/// Seeded capped run with default options otherwise; returns ranks, errors,
/// termination and evaluated points.
fn capped_run<F>(
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: &[Vec<usize>],
    evaluate: F,
    options: &TreeTciOptions,
) -> (Vec<usize>, Vec<f64>, TreeTciTermination, usize)
where
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<f64>>,
{
    let mut tci = TreeTCI2::<f64>::new(local_dims, graph).unwrap();
    tci.add_global_pivots(initial_pivots).unwrap();
    let n_sites = tci.local_dims.len();
    let flat: Vec<usize> = initial_pivots.iter().flatten().copied().collect();
    let batch = GlobalIndexBatch::new(&flat, n_sites, initial_pivots.len()).unwrap();
    tci.max_sample_value = evaluate(batch)
        .unwrap()
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let result = optimize_default(&mut tci, &evaluate, options).unwrap();
    (
        result.ranks,
        result.errors,
        result.termination,
        result.evaluation.evaluated_points,
    )
}

/// The sparse chain of #692 (seed 1, cap 8) and the star tree (seed 0, cap 2)
/// under default options.
fn capped_fixture_runs() -> [(Vec<usize>, Vec<f64>, TreeTciTermination, usize); 2] {
    const N_SITES: usize = 5;
    let pivots: Vec<Vec<usize>> = std::iter::successors(Some(1u64), |&key| Some(mix64(key).max(1)))
        .skip(1)
        .map(|key| {
            (0..N_SITES)
                .map(|site| ((key >> (2 * site)) % 4) as usize)
                .collect::<Vec<_>>()
        })
        .filter(|point| sparse_value(point) != 0.0)
        .take(16)
        .collect();
    let sparse = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        Ok(batch
            .data()
            .chunks_exact(batch.n_sites())
            .map(sparse_value)
            .collect())
    };
    let chain = capped_run(
        vec![4; N_SITES],
        TreeTciGraph::linear_chain(N_SITES).unwrap(),
        &pivots,
        sparse,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 10,
            max_bond_dim: Some(8),
            nsearch: 10,
            max_nglobal_pivot: 10,
            seed: Some(1),
            ..Default::default()
        },
    );
    let star_graph = TreeTciGraph::new(
        4,
        &[
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(1, 3),
        ],
    )
    .unwrap();
    let star_value = |point: &[usize]| {
        let phase: f64 = [0.3, 0.5, 0.7, 0.9]
            .iter()
            .zip(point)
            .map(|(weight, &coordinate)| weight * coordinate as f64)
            .sum();
        phase.cos() + 0.1
    };
    let star = capped_run(
        vec![2, 4, 4, 4],
        star_graph,
        &[vec![0, 0, 0, 0]],
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
            Ok(batch
                .data()
                .chunks_exact(batch.n_sites())
                .map(star_value)
                .collect())
        },
        &TreeTciOptions {
            tolerance: 1e-10,
            max_bond_dim: Some(2),
            seed: Some(0),
            ..Default::default()
        },
    );
    [chain, star]
}

// Capped runs under the default `PerBond` (#1405): the recorded error is the
// bond error, which stays above `tolerance` while the cap binds, so the
// saturation stop ends both runs. Recorded on the first implementation; the
// worklog compares them with 501ebb4a.
#[test]
fn capped_runs_per_bond_reproduce_recorded_values() {
    let recorded = [
        (
            vec![8, 8, 8],
            [0.9979861412184439, 0.9979861412184439, 0.9975496837919332],
            10299,
        ),
        (
            vec![2, 2, 2],
            [
                0.25932399324609984,
                0.25932399324609984,
                0.25932399324609984,
            ],
            1374,
        ),
    ];
    for (
        (ranks, errors, termination, points),
        (expected_ranks, expected_errors, expected_points),
    ) in capped_fixture_runs().into_iter().zip(recorded)
    {
        assert_eq!(termination, TreeTciTermination::MaxBondDimension);
        assert_eq!(ranks, expected_ranks);
        assert_eq!(points, expected_points);
        assert_eq!(errors.len(), expected_errors.len());
        for (error, expected) in errors.iter().zip(expected_errors) {
            assert!(
                (error - expected).abs() <= 1e-9 * expected,
                "errors {errors:?}, recorded {expected_errors:?}"
            );
        }
    }
}

#[test]
fn local_threshold_floor_ends_a_run_the_local_updates_cannot_fix() {
    // Four sites, so E = 3 and the floor is 1/3. f = 1 + i + j + k + l has
    // rank 2 across every bond, but the proposer keeps every bond at the
    // rank-1 initial cross, so the global search measures the same residual
    // in every iteration and no local threshold can remove it (#1371).
    for window in [1, 2, 3, 4] {
        let mut state =
            TreeTCI2::<f64>::new(vec![2; 4], TreeTciGraph::linear_chain(4).unwrap()).unwrap();
        state.add_global_pivots(&[vec![0; 4]]).unwrap();
        state.max_sample_value = 1.0;
        let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
            Ok(batch
                .data()
                .chunks_exact(batch.n_sites())
                .map(|point| (1 + point.iter().sum::<usize>()) as f64)
                .collect())
        };
        let options = TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 20,
            ncheck_history: window,
            nsearch: 20,
            seed: Some(7),
            stop_rule: StopRule::Sampled,
            ..Default::default()
        };
        let result = crate::optimize_with_proposer(
            &mut state,
            evaluate,
            &options,
            &InitialCrossOnly,
            &crate::AllEdges,
            &mut (),
        )
        .unwrap();
        assert_eq!(result.termination, TreeTciTermination::LocalThresholdFloor);
        // Two tightening sweeps followed by the configured window at the floor.
        assert_eq!(result.ranks, vec![1; 2 + window]);
        assert_eq!(result.nglobal_pivots.len(), 2 + window);
        assert!(result.nglobal_pivots.iter().all(|&count| count > 0));
        assert_eq!(result.local_threshold_factor, 1.0 / 3.0);
        assert!(result
            .errors
            .iter()
            .all(|&error| error >= options.tolerance));
        // The floor stop does not inject that iteration's pivots: the state is
        // swept and materializes.
        assert_swept_pivot_sets(&state, usize::MAX);
        let network = crate::to_treetn(&state, evaluate, None).unwrap();
        assert_eq!(network.node_count(), 4);
    }
}

#[test]
fn single_edge_sampled_run_keeps_injecting_until_max_iter() {
    // The run of `accepted_global_pivots_prevent_sampled_convergence` under
    // `Sampled`, with a longer `max_iter`. One edge puts the floor of the
    // local threshold at 1.0, so the threshold cannot descend and the floor
    // stop is not armed (#1371): three failing iterations must not end the
    // run. It runs to `max_iter`, and every iteration but the last injects
    // the residual pivot its search found, which the next sweep sees.
    struct RecordingCross(std::cell::RefCell<Vec<usize>>);
    impl PivotCandidateProposer for RecordingCross {
        fn candidates_with_rng<T, R: rand::Rng + ?Sized>(
            &self,
            state: &TreeTCI2<T>,
            edge: TreeTciEdge,
            rng: &mut R,
        ) -> crate::TreeTciResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
            let (left, _) = state.graph.subregion_vertices(edge)?;
            let pivots = crate::ncols_2d(&state.ijset[&left]).unwrap();
            self.0.borrow_mut().push(pivots);
            InitialCrossOnly.candidates_with_rng(state, edge, rng)
        }
    }

    const MAX_ITER: usize = 8;
    let mut state = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    state.add_global_pivots(&[vec![0, 0]]).unwrap();
    state.max_sample_value = 1.0;
    let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        Ok(batch
            .data()
            .chunks_exact(batch.n_sites())
            .map(|point| (1 + point[0] + point[1]) as f64)
            .collect())
    };
    let options = TreeTciOptions {
        tolerance: 1e-12,
        stop_rule: StopRule::Sampled,
        max_iter: MAX_ITER,
        nsearch: 20,
        seed: Some(7),
        ..Default::default()
    };
    let proposer = RecordingCross(Default::default());
    let result = crate::optimize_with_proposer(
        &mut state,
        evaluate,
        &options,
        &proposer,
        &crate::AllEdges,
        &mut (),
    )
    .unwrap();
    assert_eq!(result.termination, TreeTciTermination::MaxIterations);
    assert_eq!(result.ranks, vec![1; MAX_ITER]);
    // The search measures residual one in every iteration but the last,
    // which skips it.
    let mut expected_errors = vec![1.0; MAX_ITER - 1];
    expected_errors.push(0.0);
    assert_eq!(result.errors, expected_errors);
    assert_eq!(result.local_threshold_factor.to_bits(), 1.0_f64.to_bits());
    // Two passes over the one edge per iteration. The first pass of every
    // iteration after the first sees the initial pivot plus the injected one.
    let seen = proposer.0.into_inner();
    assert_eq!(seen.len(), 2 * MAX_ITER);
    let first_pass: Vec<usize> = seen.iter().step_by(2).copied().collect();
    let mut expected_first_pass = vec![2; MAX_ITER];
    expected_first_pass[0] = 1;
    assert_eq!(first_pass, expected_first_pass);
}

#[test]
fn bond_cap_takes_precedence_over_the_local_threshold_floor() {
    // A three-site chain (E = 2, floor 1/2) with a generic rank-3 target,
    // capped at 2. The first iteration offers only the initial cross, so the
    // ranks are [1, 2, 2, 2]: the cap window and the floor window (iteration
    // 0 at factor 1, iterations 1 to 3 at the floor) both complete at
    // iteration 3, and the cap is reported (#1371).
    use rand::{Rng, SeedableRng};

    struct CrossFirst(std::cell::Cell<usize>);
    impl PivotCandidateProposer for CrossFirst {
        fn candidates_with_rng<T, R: rand::Rng + ?Sized>(
            &self,
            state: &TreeTCI2<T>,
            edge: TreeTciEdge,
            rng: &mut R,
        ) -> crate::TreeTciResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
            // Two passes over two edges make the first iteration.
            let calls = self.0.get();
            self.0.set(calls + 1);
            if calls < 4 {
                InitialCrossOnly.candidates_with_rng(state, edge, rng)
            } else {
                DefaultProposer.candidates_with_rng(state, edge, rng)
            }
        }
    }

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
    let table: Vec<f64> = (0..27).map(|_| rng.random_range(1.0..2.0)).collect();
    let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        Ok(batch
            .data()
            .chunks_exact(batch.n_sites())
            .map(|point| table[point[0] + 3 * point[1] + 9 * point[2]])
            .collect())
    };
    let mut state =
        TreeTCI2::<f64>::new(vec![3; 3], TreeTciGraph::linear_chain(3).unwrap()).unwrap();
    state.add_global_pivots(&[vec![0; 3]]).unwrap();
    state.max_sample_value = 1.0;
    let options = TreeTciOptions {
        tolerance: 1e-8,
        stop_rule: StopRule::Sampled,
        max_iter: 30,
        max_bond_dim: Some(2),
        seed: Some(1),
        ..Default::default()
    };
    let result = crate::optimize_with_proposer(
        &mut state,
        evaluate,
        &options,
        &CrossFirst(Default::default()),
        &crate::AllEdges,
        &mut (),
    )
    .unwrap();
    assert_eq!(result.termination, TreeTciTermination::MaxBondDimension);
    assert_eq!(result.ranks, vec![1, 2, 2, 2]);
    assert!(result
        .errors
        .iter()
        .all(|&error| error >= options.tolerance));
    // The last three sweeps ran at the floor, so the floor window was
    // complete too.
    assert_eq!(result.local_threshold_factor, 0.5);
    assert_swept_pivot_sets(&state, 2);
}

#[test]
fn rank_slack_preserves_every_zero_slack_verdict() {
    for a in 0..8 {
        for b in 0..8 {
            for c in 0..8 {
                let history = [a, b, c];
                assert_eq!(super::ranks_stable(&history, 0.0), c == a.min(b).min(c));
            }
        }
    }
    assert!(super::ranks_stable(&[220, 218, 217], 0.0));
    assert!(!super::ranks_stable(&[217, 218, 220], 0.0));
    // Avoid changing exact zero-slack semantics through f64 rounding.
    assert!(!super::ranks_stable(&[usize::MAX - 1, usize::MAX], 0.0));
    assert!(!super::ranks_stable(&[], 0.03));
}

#[test]
fn rank_slack_accepts_noise_but_not_unbounded_growth() {
    assert!(super::ranks_stable(&[216, 213, 219], 0.03));
    assert!(!super::ranks_stable(&[216, 212, 219], 0.03));
    assert!(super::ranks_stable(&[3, 5], 0.4)); // equality is accepted
    assert!(!super::ranks_stable(&[3, 5], 0.39));
    assert!(super::ranks_stable(&[0, 0, 0], 0.03));
    assert!(super::ranks_stable(&[7], 0.0));
}

#[test]
fn history_window_controls_convergence_cap_and_short_calls() {
    for (window, max_iter, cap, expected_len, expected_reason) in [
        (1, 5, None, 1, TreeTciTermination::Converged),
        (1, 1, None, 1, TreeTciTermination::Converged),
        (usize::MAX, 5, None, 5, TreeTciTermination::MaxIterations),
        (2, 5, None, 2, TreeTciTermination::Converged),
        (4, 5, None, 4, TreeTciTermination::Converged),
        (6, 5, None, 5, TreeTciTermination::MaxIterations),
        (3, 1, None, 1, TreeTciTermination::MaxIterations),
        (1, 5, Some(1), 1, TreeTciTermination::MaxBondDimension),
        (2, 5, Some(1), 2, TreeTciTermination::MaxBondDimension),
        (6, 5, Some(1), 5, TreeTciTermination::MaxIterations),
    ] {
        for (enabled, limit) in [(false, 5), (true, 0), (true, 5)] {
            let mut state = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
            state.add_global_pivots(&[vec![0, 0]]).unwrap();
            let result = optimize_default(
                &mut state,
                |batch| Ok(vec![2.0; batch.n_points()]),
                &TreeTciOptions {
                    ncheck_history: window,
                    max_iter,
                    max_bond_dim: cap,
                    enable_global_pivots: enabled,
                    max_nglobal_pivot: limit,
                    seed: Some(0),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(result.termination, expected_reason);
            assert_eq!(result.ranks, vec![1; expected_len]);
            assert_eq!(result.errors, vec![0.0; expected_len]);
            assert_eq!(result.nglobal_pivots, vec![0; expected_len]);
        }
    }
}
