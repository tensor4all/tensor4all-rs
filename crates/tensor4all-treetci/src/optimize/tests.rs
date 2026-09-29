use super::{optimize_default, TreeTciOptions};
use crate::test_support::assert_scalar_close;
use crate::{GlobalIndexBatch, TreeTCI2, TreeTciEdge, TreeTciGraph};
use anyhow::Result;
use tensor4all_core::IndexLike;

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

    let (ranks, errors) = optimize_default(
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

    assert_eq!(ranks.last().copied(), Some(2));
    assert_scalar_close(
        errors.last().copied().unwrap_or(f64::NAN),
        0.0,
        tci.max_sample_value,
        1e-12,
    );
    assert_eq!(tci.max_bond_dim(), 2);
}

// Previously named `optimize_default_runs_all_iterations_like_upstream_tree_tci`
// and asserted `ranks.len() == 4` / `errors.len() == 4` (max_iter), pinning
// parity with upstream TreeTCI.jl's sweep loop, which has no early-convergence
// break at all. That upstream behavior is a known bug: see
// ~/gw/CombTCI/COMPATIBILITY.md's `treetci-fix-convergence-criterion.patch`
// (a *different*, scale-mismatch bug in the same convergence-check area,
// found and locally patched by the same user, not yet upstreamed) and
// ~/tensor4all-rust/treetci-optimize-no-early-stop-bug.md for this crate's
// specific issue (no break at all, not just a wrong-scale comparison).
// Renamed and flipped to assert the fixed (early-stopping) behavior instead.
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

    let (ranks, errors) = optimize_default(
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

    let (ranks, errors) = optimize_default(
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

    // Rank-3 identity capped at max_bond_dim = 1 can never reach the 1e-12
    // tolerance; without the bond-dim-saturation criterion this would run
    // all 10 sweeps.
    assert!(ranks.len() < 10);
    assert!(ranks.iter().all(|&r| r <= 1));
    assert!(errors.last().copied().unwrap_or(0.0) > 1e-12);
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

    let (ranks, errors) = optimize_default(&mut tci, &evaluate, options).unwrap();

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
    let (treetn, ranks, _) = crate::crossinterpolate2::<f64, _, _>(
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
