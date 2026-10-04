//! `patched_interpolate` with a driver-local dense test engine.
//!
//! The dense test engine evaluates the whole active domain, factorizes it
//! exactly, and reports `BondCapReached` when the exact rank reaches the cap,
//! so splitting is tested independently of any real engine.

mod adaptive_common;

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use adaptive_common::*;
use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, Index, IndexLike, TagSet};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_interpolate, PatchedInterpolationError, PatchedInterpolationResult,
};
use tensor4all_partitionedtreetn::{ErrorNorm, Projector};
use tensor4all_treetn::interpolation::{
    validate_layout, InterpolationError, InterpolationTermination,
};

#[test]
fn branched_topology_is_a_genuine_tree_with_a_degree_three_node() {
    let problem = branched();
    let degree = |node: &str| {
        let index = problem.topology.node_index(&node.to_string()).unwrap();
        problem.topology.graph().neighbors(index).count()
    };
    assert_eq!(degree("j"), 3);
    assert_eq!(problem.topology.edge_count(), 5);
    assert!(validate_layout(&problem.topology, &problem.node_sites).is_ok());
}

// ---------------------------------------------------------------------------
// Splits with the dense test engine
// ---------------------------------------------------------------------------

/// `f` switches its product function on `split`; with a cap of two the root
/// splits once at `split` and every child converges with rank one.
fn assert_single_split(problem: &Problem, split: &DynIndex) {
    let f = switch_on(problem.position(split));
    let options = sampled_max(2)
        .with_tolerance(tol(1e-12))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(10.0))
        .with_patch_order(vec![split.clone()]);
    let result = run(&DenseEngine::new(), problem, &f, &[], &options).unwrap();
    let report = &result.report;
    assert_eq!(report.splits, 1);
    assert!(zero_projectors(report).is_empty());
    assert_eq!(report.accepted.len(), split.dim());
    for (value, record) in report.accepted.iter().enumerate() {
        assert_eq!(
            record.projector,
            Projector::from_pairs([(split.clone(), value)]).unwrap()
        );
        assert_eq!(record.termination, InterpolationTermination::Converged);
        assert_eq!(record.max_bond_dim, 1);
    }
    assert_accurate(&result, problem, &f, 1e-12);
}

#[test]
fn splits_at_a_leaf() {
    let problem = branched();
    assert_single_split(&problem, &problem.site("a", 0));
}

#[test]
fn splits_at_an_internal_node() {
    let problem = branched();
    assert_single_split(&problem, &problem.site("c", 0));
}

#[test]
fn splits_at_the_junction() {
    let problem = branched();
    assert_single_split(&problem, &problem.site("j", 0));
}

#[test]
fn splits_at_a_multi_site_node() {
    let problem = branched();
    assert_single_split(&problem, &problem.site("d", 1));
}

#[test]
fn splits_on_a_chain() {
    let problem = chain3();
    assert_single_split(&problem, &problem.site("n1", 0));
}

#[test]
fn splits_until_every_site_of_a_node_is_fixed() {
    let problem = branched();
    let (d0, d1) = (problem.site("d", 0), problem.site("d", 1));
    let (p0, p1) = (problem.position(&d0), problem.position(&d1));
    // Fixing d0 alone leaves two product functions; fixing both leaves one.
    let f = move |point: &[usize]| product(point, 2 * point[p0] + point[p1]);
    let options = sampled_max(2)
        .with_tolerance(tol(1e-12))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(10.0))
        .with_patch_order(vec![d0.clone(), d1.clone()]);
    let engine = DenseEngine::new();
    let result = run(&engine, &problem, &f, &[], &options).unwrap();
    assert_eq!(result.report.splits, 3);
    assert_eq!(result.report.accepted.len(), 4);
    assert_accurate(&result, &problem, &f, 1e-12);
    // The last engine problems see node d without an active site.
    let last = engine.seen().last().cloned().unwrap();
    assert!(!last.site_order.contains(&d0) && !last.site_order.contains(&d1));
    assert_eq!(last.site_order.len(), problem.sites.len() - 2);
}

#[test]
fn single_node_topology_runs_the_engine_on_several_sites() {
    let problem = single_node(&[3, 4]);
    let f = |p: &[usize]| (1 + p[0] * p[1]) as f64;
    let options = sampled_max(2);
    let engine = DenseEngine::new();
    let result = run(&engine, &problem, &f, &[vec![2, 3]], &options).unwrap();
    assert_eq!(result.report.splits, 0);
    assert_eq!(result.report.accepted.len(), 1);
    assert_eq!(engine.seen().len(), 1);
    // The root scale is pinned from the candidates, which include (2, 3).
    assert_eq!(max_reference(&result.report), 7.0);
    let (residual, _) = dense_residual(&result, &problem, &f);
    assert!(residual < 1e-12);
}

#[test]
fn iteration_limit_splits_like_the_bond_cap() {
    let problem = chain3();
    let f = |p: &[usize]| product(p, 0);
    let engine = DenseEngine::with_fault(Fault::IterationLimit);
    let options = sampled_max(4).with_error_norm(ErrorNorm::sampled_max_with_reference(2.0));
    let result = run(&engine, &problem, &f, &[], &options).unwrap();
    // Root and both children split; the grandchildren are exact.
    assert_eq!(result.report.splits, 3);
    assert_eq!(result.report.accepted.len(), 4);
    assert_eq!(engine.seen().len(), 3);
    assert_accurate(&result, &problem, &f, options.tolerance.rtol);
}

#[test]
fn site_identity_uses_the_full_index() {
    let base = DynIndex::new_dyn(2);
    let primed = base.prime();
    let tagged = Index::new_with_tags(base.id, 2, TagSet::from_str("Site").unwrap());
    let problem = Problem::with_sites(
        BTreeMap::from([
            ("p".to_string(), vec![base.clone(), primed.clone()]),
            ("q".to_string(), vec![tagged.clone()]),
        ]),
        &[("p", "q")],
    );
    assert_eq!(
        problem.sites,
        [base.clone(), primed.clone(), tagged.clone()]
    );
    let f = switch_on(1);
    let options = sampled_max(2)
        .with_tolerance(tol(1e-12))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(10.0))
        .with_patch_order(vec![primed.clone()]);
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    assert_eq!(result.report.accepted.len(), 2);
    for (value, record) in result.report.accepted.iter().enumerate() {
        assert_eq!(record.projector.get(&primed), Some(value));
        assert_eq!(record.projector.get(&base), None);
        assert_eq!(record.projector.get(&tagged), None);
    }
    assert_accurate(&result, &problem, &f, 1e-12);

    // An index with the same ID but other tags is not a site of the problem.
    let other = Index::new_with_tags(base.id, 2, TagSet::from_str("Other").unwrap());
    expect_invalid(
        run(
            &DenseEngine::new(),
            &problem,
            &f,
            &[],
            &options.clone().with_patch_order(vec![other]),
        ),
        "is not a site index",
    );
}

#[test]
fn reports_are_in_canonical_path_order() {
    let problem = chain3();
    // Child s0 = 1 is a product and converges; child s0 = 0 splits again, so
    // the processing order is [1], [0, 0], [0, 1].
    let f = |p: &[usize]| {
        if p[0] == 1 {
            product(p, 0)
        } else {
            product(p, p[1])
        }
    };
    let options = sampled_max(2)
        .with_tolerance(tol(1e-12))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(10.0));
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    let (s0, s1) = (problem.site("n0", 0), problem.site("n1", 0));
    let projectors: Vec<Projector> = result
        .report
        .accepted
        .iter()
        .map(|record| record.projector.clone())
        .collect();
    assert_eq!(
        projectors,
        [
            Projector::from_pairs([(s0.clone(), 0), (s1.clone(), 0)]).unwrap(),
            Projector::from_pairs([(s0.clone(), 0), (s1.clone(), 1)]).unwrap(),
            Projector::from_pairs([(s0.clone(), 1)]).unwrap(),
        ]
    );
    assert_eq!(result.report.accepted[2].max_bond_dim, 1);
    assert_accurate(&result, &problem, &f, 1e-12);
}

#[test]
fn vanishing_regions_are_reported_as_zero_patches() {
    let problem = branched();
    let (j0, a0) = (problem.site("j", 0), problem.site("a", 0));
    let (pj, pa) = (problem.position(&j0), problem.position(&a0));
    let f = move |p: &[usize]| if p[pj] == 0 { 0.0 } else { product(p, p[pa]) };
    let options = sampled_max(2)
        .with_tolerance(tol(1e-12))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(10.0))
        .with_patch_order(vec![j0.clone(), a0.clone()]);
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    assert_eq!(
        zero_projectors(&result.report),
        [Projector::from_pairs([(j0.clone(), 0)]).unwrap()]
    );
    assert_eq!(result.report.accepted.len(), 3);
    assert!(result
        .report
        .accepted
        .iter()
        .all(|record| record.projector.get(&j0) == Some(1)));
    assert_accurate(&result, &problem, &f, 1e-12);
}

// ---------------------------------------------------------------------------
// Exact small patches and the reference scale
// ---------------------------------------------------------------------------

#[test]
fn one_site_patches_below_the_root_are_evaluated_exactly() {
    // s has dimension 3, t dimension 8 > n_initial_pivots. The rows s = 0, 1, 2
    // are zero, 1 + t, and nonzero only at t = 5: rank two at the root.
    let problem = Problem::new(&[("s", &[3]), ("t", &[8])], &[("s", "t")]);
    let s = problem.site("s", 0);
    let f = |p: &[usize]| match p[0] {
        0 => 0.0,
        1 => 1.0 + p[1] as f64,
        _ if p[1] == 5 => -3.0,
        _ => 0.0,
    };
    let options = sampled_max(2).with_n_initial_pivots(1);
    let engine = DenseEngine::new();
    let result = run(&engine, &problem, &f, &[vec![1, 0]], &options).unwrap();
    let report = &result.report;
    // The scale is pinned from the single root candidate, f(1, 0) = 1.
    assert_eq!(max_reference(report), 1.0);
    assert_eq!(report.splits, 1);
    assert_eq!(engine.seen().len(), 1);
    assert_eq!(
        zero_projectors(report),
        [Projector::from_pairs([(s.clone(), 0)]).unwrap()]
    );
    assert_eq!(report.accepted.len(), 2);
    let sparse = &report.accepted[1];
    assert_eq!(sparse.projector.get(&s), Some(2));
    assert_eq!(sparse.termination, InterpolationTermination::Converged);
    assert_eq!(sparse.engine_error_estimate, 0.0);
    assert_eq!(sparse.max_sample_magnitude, 3.0);
    assert_eq!(sparse.max_bond_dim, 1);
    // All 24 points were needed and each was evaluated once.
    assert_eq!(report.function_evaluations, 24);
    let (residual, _) = dense_residual(&result, &problem, &f);
    assert_eq!(residual, 0.0);
}

#[test]
fn a_one_site_root_pins_the_scale_from_its_exact_values() {
    let problem = single_node(&[8]);
    let f = |p: &[usize]| if p[0] == 5 { -4.0 } else { 0.0 };
    let options = sampled_max(2).with_n_initial_pivots(1);
    let engine = DenseEngine::new();
    let result = run(&engine, &problem, &f, &[], &options).unwrap();
    assert!(engine.seen().is_empty());
    assert_eq!(max_reference(&result.report), 4.0);
    assert_eq!(result.report.accepted.len(), 1);
    assert_eq!(result.report.accepted[0].max_sample_magnitude, 4.0);
    assert_eq!(result.report.function_evaluations, 8);
    let (residual, _) = dense_residual(&result, &problem, &f);
    assert_eq!(residual, 0.0);
}

#[test]
fn an_all_zero_exact_root_gives_an_empty_partition() {
    let problem = single_node(&[3]);
    let zero = |_: &[usize]| 0.0;
    let result = run(&DenseEngine::new(), &problem, &zero, &[], &sampled_max(2)).unwrap();
    assert!(result.partition.is_empty());
    assert!(result.report.accepted.is_empty());
    assert_eq!(zero_projectors(&result.report), [Projector::new()]);
    assert_eq!(max_reference(&result.report), 0.0);

    let given = sampled_max(2).with_error_norm(ErrorNorm::sampled_max_with_reference(2.0));
    let result = run(&DenseEngine::new(), &problem, &zero, &[], &given).unwrap();
    assert!(result.partition.is_empty());
    assert_eq!(max_reference(&result.report), 2.0);
}

#[test]
fn an_exact_root_keeps_nodes_without_sites() {
    // The site-free node "a" gets a dimension-one link to the node "b".
    let problem = Problem::new(&[("a", &[]), ("b", &[4])], &[("a", "b")]);
    let f = |p: &[usize]| 1.0 + p[0] as f64;
    let result = run(&DenseEngine::new(), &problem, &f, &[], &sampled_max(2)).unwrap();
    let data = result.partition.to_treetn().unwrap();
    assert_eq!(data.node_count(), 2);
    assert_eq!(data.link_dims(), [1]);
    assert!(data.site_space(&"a".to_string()).unwrap().is_empty());
    let (residual, _) = dense_residual(&result, &problem, &f);
    assert_eq!(residual, 0.0);
}

#[test]
fn a_zero_root_that_needs_the_engine_uses_the_given_scale_or_fails() {
    let problem = branched();
    let zero = |_: &[usize]| 0.0;
    let engine = DenseEngine::new();
    let options = sampled_max(2).with_error_norm(ErrorNorm::sampled_max_with_reference(1.0));
    let result = run(&engine, &problem, &zero, &[], &options).unwrap();
    assert!(result.partition.is_empty());
    assert_eq!(zero_projectors(&result.report), [Projector::new()]);
    assert_eq!(result.report.function_evaluations, 5);
    assert!(engine.seen().is_empty());

    let message = expect_invalid(
        run(&engine, &problem, &zero, &[], &sampled_max(2)),
        "cannot be pinned",
    );
    assert!(
        message.contains("give ErrorNorm::sampled_max_with_reference(max_abs) or initial pivots")
    );
}

// ---------------------------------------------------------------------------
// Scalar types, recycling, determinism, and the cache
// ---------------------------------------------------------------------------

/// Every node tensor of every patch is `Complex64`: re-embedded one-hot
/// factors and the ones of exact patches must not fall back to `f64`.
fn assert_all_complex(result: &PatchedInterpolationResult<Name>) {
    assert!(!result.partition.is_empty());
    for patch in result.partition.values() {
        let data = patch.data();
        for name in data.node_names() {
            let tensor = data.tensor(data.node_index(&name).unwrap()).unwrap();
            assert!(
                tensor.is_c64(),
                "node {name} of {:?} is not Complex64",
                patch.projector()
            );
        }
    }
}

#[test]
fn complex_patches_have_one_dtype() {
    let problem = branched();
    let j0 = problem.site("j", 0);
    let pj = problem.position(&j0);
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let phases: Vec<f64> = (0..2)
        .map(|_| rng.random_range(0.0..std::f64::consts::TAU))
        .collect();
    let f = move |p: &[usize]| Complex64::from_polar(product(p, p[pj]), phases[p[pj]]);
    let options = sampled_max(2)
        .with_tolerance(tol(1e-12))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(10.0))
        .with_patch_order(vec![j0]);
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    assert_eq!(result.report.accepted.len(), 2);
    assert_all_complex(&result);
    assert_accurate(&result, &problem, &f, 1e-12);
}

#[test]
fn complex_exact_patches_have_one_dtype() {
    // An exact root on two nodes: the site-free node holds a complex one.
    let problem = Problem::new(&[("a", &[]), ("b", &[4])], &[("a", "b")]);
    let g = |p: &[usize]| Complex64::new(p[0] as f64, 1.0);
    let result = run(&DenseEngine::new(), &problem, &g, &[], &sampled_max(2)).unwrap();
    assert_eq!(max_reference(&result.report), 10.0_f64.sqrt());
    assert_all_complex(&result);
    let (residual, _) = dense_residual(&result, &problem, &g);
    assert_eq!(residual, 0.0);

    // Splits down to one-site children: one-hot factors on two nodes.
    let problem = chain3();
    let mut rng = ChaCha8Rng::seed_from_u64(12);
    let phases: Vec<f64> = (0..4)
        .map(|_| rng.random_range(0.0..std::f64::consts::TAU))
        .collect();
    let f = move |p: &[usize]| {
        Complex64::from_polar(product(p, 2 * p[0] + p[1]), phases[2 * p[0] + p[1]])
    };
    let options = sampled_max(2).with_error_norm(ErrorNorm::sampled_max_with_reference(10.0));
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    assert_eq!(result.report.splits, 3);
    assert_eq!(result.report.accepted.len(), 4);
    assert!(result
        .report
        .accepted
        .iter()
        .all(|record| record.projector.len() == 2));
    assert_all_complex(&result);
    let (residual, _) = dense_residual(&result, &problem, &f);
    assert!(residual < 1e-12, "residual {residual}");
}

/// Initial pivots of each child problem, one list per child.
type ChildPivots = Vec<Vec<Vec<usize>>>;

/// The initial pivots each child problem received, in child coordinates,
/// and the ones recycling would give: the parent's returned pivots with the
/// split coordinate, the split site removed.
fn recycling_run(recycle: bool) -> (ChildPivots, ChildPivots) {
    let problem = chain3();
    let s0 = problem.site("n0", 0);
    // Rank two at the root and in both children; exact grandchildren.
    let f = |p: &[usize]| product(p, 2 * p[0] + p[1]);
    let options = sampled_max(2)
        .with_tolerance(tol(1e-12))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(10.0))
        .with_n_initial_pivots(1)
        .with_seed(3)
        .with_recycle_pivots(recycle);
    let engine = DenseEngine::new();
    let result = run(&engine, &problem, &f, &[], &options).unwrap();
    assert_accurate(&result, &problem, &f, 1e-12);
    let seen = engine.seen();
    assert_eq!(seen.len(), 3);
    assert!(!seen[1].site_order.contains(&s0));
    let received = seen[1..].iter().map(|s| s.initial_pivots.clone()).collect();
    let expected = (0..2)
        .map(|value| {
            let mut kept: Vec<Vec<usize>> = Vec::new();
            for point in &seen[0].returned_pivots {
                let local = point[1..].to_vec();
                if point[0] == value && !kept.contains(&local) {
                    kept.push(local);
                }
            }
            kept
        })
        .collect();
    (received, expected)
}

#[test]
fn recycled_pivots_seed_the_children() {
    let (received, expected) = recycling_run(true);
    assert!(expected
        .iter()
        .any(|pivots: &Vec<Vec<usize>>| !pivots.is_empty()));
    for (received, expected) in received.iter().zip(&expected) {
        if expected.is_empty() {
            assert_eq!(received.len(), 1);
        } else {
            assert_eq!(received, expected);
        }
    }
}

#[test]
fn without_recycling_children_start_from_random_candidates() {
    let (received, _) = recycling_run(false);
    assert!(received.iter().all(|pivots| pivots.len() == 1));
}

#[test]
fn runs_are_deterministic_and_seeds_depend_on_the_patch() {
    let problem = branched();
    let (j0, a0) = (problem.site("j", 0), problem.site("a", 0));
    let (pj, pa) = (problem.position(&j0), problem.position(&a0));
    let f = move |p: &[usize]| product(p, p[pj] + 2 * p[pa]);
    let options = sampled_max(2)
        .with_tolerance(tol(1e-12))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(20.0))
        .with_seed(99)
        .with_patch_order(vec![j0, a0]);
    let first_engine = DenseEngine::new();
    let first = run(&first_engine, &problem, &f, &[], &options).unwrap();
    let second_engine = DenseEngine::new();
    let second = run(&second_engine, &problem, &f, &[], &options).unwrap();

    let seeds = |engine: &DenseEngine| engine.seen().iter().map(|s| s.seed).collect::<Vec<_>>();
    let pivots = |engine: &DenseEngine| {
        engine
            .seen()
            .iter()
            .map(|s| s.initial_pivots.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(seeds(&first_engine), seeds(&second_engine));
    assert_eq!(pivots(&first_engine), pivots(&second_engine));
    let distinct: HashSet<u64> = seeds(&first_engine).into_iter().collect();
    assert_eq!(distinct.len(), first_engine.seen().len());
    assert_same_run(&problem, &first, &second);
}

#[test]
fn engine_requests_of_sampled_points_hit_the_cache() {
    let problem = branched();
    let f = switch_on(problem.position(&problem.site("j", 0)));
    let options = sampled_max(2)
        .with_error_norm(ErrorNorm::sampled_max_with_reference(10.0))
        .with_patch_order(vec![problem.site("j", 0)]);
    let engine = DenseEngine::new();
    let result = run(&engine, &problem, &f, &[], &options).unwrap();
    // The engine re-reads its initial pivots (all cached candidates) and the
    // children re-read the half of the root domain the root evaluated.
    let reread: usize = engine.seen().iter().map(|s| s.initial_pivots.len()).sum();
    assert!(result.report.cache_hits >= reread + 96);
    assert_eq!(result.report.function_evaluations, 96);
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[test]
fn a_partial_order_can_run_out_of_split_sites() {
    let problem = branched();
    let (j0, a0) = (problem.site("j", 0), problem.site("a", 0));
    // f depends on a0 but patch_order only lists j0.
    let f = switch_on(problem.position(&a0));
    let options = sampled_max(2)
        .with_error_norm(ErrorNorm::sampled_max_with_reference(10.0))
        .with_patch_order(vec![j0.clone()]);
    match run(&DenseEngine::new(), &problem, &f, &[], &options) {
        Err(PatchedInterpolationError::NoSplitIndexLeft { projector }) => {
            assert_eq!(projector, Projector::from_pairs([(j0, 0)]).unwrap());
        }
        other => panic!("expected NoSplitIndexLeft, got {other:?}"),
    }
}

#[test]
fn max_patches_limits_the_processed_patches() {
    let problem = branched();
    let a0 = problem.site("a", 0);
    let f = switch_on(problem.position(&a0));
    // One root and three children.
    let options = sampled_max(2)
        .with_error_norm(ErrorNorm::sampled_max_with_reference(10.0))
        .with_patch_order(vec![a0]);
    assert!(run(
        &DenseEngine::new(),
        &problem,
        &f,
        &[],
        &options.clone().with_max_patches(4)
    )
    .is_ok());
    match run(
        &DenseEngine::new(),
        &problem,
        &f,
        &[],
        &options.with_max_patches(3),
    ) {
        Err(PatchedInterpolationError::ResourceLimit { resource, limit }) => {
            assert_eq!((resource, limit), ("max_patches", 3));
        }
        other => panic!("expected ResourceLimit, got {other:?}"),
    }
}

/// A function that fails or misbehaves on the given call of the evaluator.
/// How a faulty evaluator answers a batch of the given number of points.
type Failure = fn(usize) -> anyhow::Result<Vec<f64>>;

fn faulty_evaluator(
    fail_on_call: usize,
    failure: Failure,
) -> impl Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>> + Send + Sync {
    let calls = AtomicUsize::new(0);
    move |batch| {
        let n_points = batch.shape()[1];
        if calls.fetch_add(1, Ordering::SeqCst) == fail_on_call {
            failure(n_points)
        } else {
            Ok(vec![1.0; n_points])
        }
    }
}

fn run_faulty(
    problem: &Problem,
    engine: &DenseEngine,
    evaluate: impl Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>> + Send + Sync,
) -> Result<PatchedInterpolationResult<Name>, PatchedInterpolationError> {
    let options = sampled_max(2).with_error_norm(ErrorNorm::sampled_max_with_reference(1.0));
    run_with(engine, problem, evaluate, &[], &options)
}

#[test]
fn evaluator_failures_are_reported_for_the_failing_patch() {
    let failures: [(Failure, &str); 4] = [
        (|_| Err(anyhow::anyhow!("user failure")), "user failure"),
        (|n| Ok(vec![f64::NAN; n]), "non-finite"),
        (|n| Ok(vec![f64::INFINITY; n]), "non-finite"),
        (|n| Ok(vec![1.0; n - 1]), "values for"),
    ];
    for (failure, needle) in failures {
        // The first call samples the root candidates (engine path) or the
        // exact values of a one-site root; the second call comes from inside
        // the engine.
        for (problem, call) in [(branched(), 0), (single_node(&[3]), 0), (branched(), 1)] {
            let (projector, source) = expect_interpolation(run_faulty(
                &problem,
                &DenseEngine::new(),
                faulty_evaluator(call, failure),
            ));
            assert!(projector.is_empty());
            match source {
                InterpolationError::Evaluator { source } => {
                    assert!(
                        format!("{source:#}").contains(needle),
                        "{source:#} lacks {needle}"
                    )
                }
                other => panic!("expected Evaluator, got {other:?}"),
            }
        }
    }
}

#[test]
fn engine_misbehavior_is_reported_for_the_failing_patch() {
    let problem = branched();
    let ok = || faulty_evaluator(usize::MAX, |_| unreachable!());
    let cases = [
        (Fault::WrongLayout, "outcome network"),
        (Fault::BadBatch, "batch shape"),
        (Fault::OutOfRange, "out of range"),
    ];
    for (fault, needle) in cases {
        let (projector, source) =
            expect_interpolation(run_faulty(&problem, &DenseEngine::with_fault(fault), ok()));
        assert!(projector.is_empty());
        assert!(
            format!("{source:#}").contains(needle),
            "{source:#} lacks {needle}"
        );
    }
    let (_, source) = expect_interpolation(run_faulty(
        &problem,
        &DenseEngine::with_fault(Fault::AllZero),
        ok(),
    ));
    assert!(matches!(source, InterpolationError::AllSamplesZero));
}

#[test]
fn malformed_pivots_fail_only_when_recycled() {
    let problem = chain3();
    let f = |p: &[usize]| product(p, 2 * p[0] + p[1]);
    let engine = DenseEngine::with_fault(Fault::BadPivots);
    let options = sampled_max(2).with_error_norm(ErrorNorm::sampled_max_with_reference(10.0));
    assert!(run(&engine, &problem, &f, &[], &options).is_ok());
    let (_, source) = expect_interpolation(run(
        &engine,
        &problem,
        &f,
        &[],
        &options.with_recycle_pivots(true),
    ));
    assert!(matches!(source, InterpolationError::Engine { .. }));
    assert!(source.to_string().contains("rows"));
}

#[test]
fn invalid_layouts_are_rejected_before_any_evaluation() {
    let site = DynIndex::new_dyn(2);
    let sites = |entries: &[(&str, Vec<DynIndex>)]| -> BTreeMap<Name, Vec<DynIndex>> {
        entries
            .iter()
            .map(|(n, s)| (n.to_string(), s.clone()))
            .collect()
    };
    // Every root here has at most one site, so an unvalidated exact path
    // would evaluate it.
    let cases = [
        (topology(&[], &[]), sites(&[]), "no nodes"),
        (
            topology(&["a", "b"], &[("a", "b")]),
            sites(&[("a", vec![site.clone()])]),
            "node_sites has 1 entries",
        ),
        (
            topology(&["a"], &[]),
            sites(&[("b", vec![site.clone()])]),
            "not in the topology",
        ),
        (
            topology(&["a", "b"], &[]),
            sites(&[("a", vec![site.clone()]), ("b", vec![])]),
            "edges",
        ),
        (
            topology(&["a", "b", "c"], &[("a", "a"), ("b", "c")]),
            sites(&[("a", vec![site.clone()]), ("b", vec![]), ("c", vec![])]),
            "not connected",
        ),
        (
            topology(&["a"], &[]),
            sites(&[("a", vec![])]),
            "no active site",
        ),
        (
            topology(&["a"], &[]),
            sites(&[("a", vec![DynIndex::new_dyn(0)])]),
            "dimension zero",
        ),
        (
            topology(&["a", "b"], &[("a", "b")]),
            sites(&[("a", vec![site.clone()]), ("b", vec![site.clone()])]),
            "more than once",
        ),
    ];
    for (topology, node_sites, needle) in cases {
        let expected = match validate_layout(&topology, &node_sites) {
            Err(InterpolationError::InvalidProblem { message }) => message,
            other => panic!("validate_layout must reject {needle:?}, got {other:?}"),
        };
        let n_sites = node_sites.values().map(Vec::len).sum::<usize>();
        let message = expect_invalid(
            patched_interpolate(
                &DenseEngine::new(),
                topology,
                node_sites,
                ColMajorArray::new(vec![], vec![n_sites, 0]).unwrap(),
                never,
                &sampled_max(2),
            ),
            needle,
        );
        assert_eq!(message, expected);
    }
}

#[test]
fn invalid_options_and_pivots_are_rejected_before_any_evaluation() {
    let problem = single_node(&[3]);
    let site = problem.site("only", 0);
    let mut resized = site.clone();
    resized.dim = 4;
    let base = sampled_max(2);
    let option_cases = [
        (
            base.clone().with_patch_order(vec![DynIndex::new_dyn(3)]),
            "is not a site index",
        ),
        (
            base.clone().with_patch_order(vec![resized]),
            "has dimension 4",
        ),
        (
            base.clone()
                .with_patch_order(vec![site.clone(), site.clone()]),
            "more than once",
        ),
        (base.clone().with_tolerance(tol(-1e-3)), "rtol"),
        (base.clone().with_tolerance(tol(f64::NAN)), "rtol"),
        (base.clone().with_tolerance(tol(f64::INFINITY)), "rtol"),
        (
            base.clone()
                .with_error_norm(ErrorNorm::sampled_max_with_reference(0.0)),
            "max_reference",
        ),
        (
            base.clone()
                .with_error_norm(ErrorNorm::sampled_max_with_reference(-1.0)),
            "max_reference",
        ),
        (
            base.clone()
                .with_error_norm(ErrorNorm::sampled_max_with_reference(f64::NAN)),
            "max_reference",
        ),
        (
            base.clone()
                .with_error_norm(ErrorNorm::sampled_max_with_reference(f64::INFINITY)),
            "max_reference",
        ),
        (sampled_max(0), "max_bond_dim"),
        (sampled_max(1), "max_bond_dim"),
        (base.clone().with_n_initial_pivots(0), "n_initial_pivots"),
        (base.clone().with_max_patches(0), "max_patches"),
    ];
    let valid_pivots = || ColMajorArray::new(vec![2], vec![1, 1]).unwrap();
    for (options, needle) in option_cases {
        expect_invalid(
            patched_interpolate(
                &DenseEngine::new(),
                problem.topology.clone(),
                problem.node_sites.clone(),
                valid_pivots(),
                never,
                &options,
            ),
            needle,
        );
    }
    let pivot_cases = [
        (ColMajorArray::new(vec![0], vec![1]).unwrap(), "2D array"),
        (
            ColMajorArray::new(vec![0, 0], vec![2, 1]).unwrap(),
            "has 2 rows",
        ),
        (
            ColMajorArray::new(vec![0, 3], vec![1, 2]).unwrap(),
            "coordinate 3",
        ),
    ];
    for (pivots, needle) in pivot_cases {
        expect_invalid(
            patched_interpolate(
                &DenseEngine::new(),
                problem.topology.clone(),
                problem.node_sites.clone(),
                pivots,
                never,
                &base,
            ),
            needle,
        );
    }
}

#[test]
fn errors_display_their_remedy() {
    let limit = PatchedInterpolationError::ResourceLimit {
        resource: "max_patches",
        limit: 3,
    };
    assert!(limit.to_string().contains("max_patches limit of 3"));
    let stuck = PatchedInterpolationError::NoSplitIndexLeft {
        projector: Projector::new(),
    };
    assert!(stuck.to_string().contains("list more sites in patch_order"));
    let invalid = PatchedInterpolationError::InvalidInput {
        message: "bad".to_string(),
    };
    assert_eq!(
        invalid.to_string(),
        "invalid patched interpolation input: bad"
    );
    let partition = PatchedInterpolationError::Partition {
        source: tensor4all_partitionedtreetn::PartitionedTreeTNError::Empty,
    };
    assert!(std::error::Error::source(&partition).is_some());
}

#[test]
fn an_overflowing_absolute_tolerance_is_reported_for_the_root() {
    let problem = branched();
    let f = switch_on(problem.position(&problem.site("a", 0)));
    // Both factors are finite, but their product is not.
    let options = sampled_max(2)
        .with_tolerance(tol(1e300))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(1e300));
    let (projector, source) =
        expect_interpolation(run(&DenseEngine::new(), &problem, &f, &[], &options));
    assert!(projector.is_empty());
    assert!(matches!(source, InterpolationError::InvalidProblem { .. }));
    assert!(source.to_string().contains("absolute_tolerance"));
}

#[test]
fn converged_outcomes_at_the_cap_are_engine_errors() {
    let problem = branched();
    let f = switch_on(problem.position(&problem.site("a", 0)));
    let options = sampled_max(2).with_error_norm(ErrorNorm::sampled_max_with_reference(10.0));
    let engine = DenseEngine::with_fault(Fault::ConvergedAtCap);
    let (projector, source) = expect_interpolation(run(&engine, &problem, &f, &[], &options));
    assert!(projector.is_empty());
    assert!(matches!(source, InterpolationError::Engine { .. }));
    assert!(
        source.to_string().contains("not strictly below the cap 2"),
        "{source}"
    );

    // A rank-one function converges strictly below the same cap.
    let product_only = |p: &[usize]| product(p, 0);
    let result = run(&engine, &problem, &product_only, &[], &options).unwrap();
    assert_eq!(result.report.accepted.len(), 1);
    assert_eq!(result.report.accepted[0].max_bond_dim, 1);
}

/// `f` is 1 everywhere except at `bad`, where both parts are near the largest
/// finite float: finite components whose magnitude overflows.
fn overflow_at(bad: Vec<usize>) -> impl Fn(&[usize]) -> Complex64 + Sync {
    move |p: &[usize]| {
        if p == bad.as_slice() {
            Complex64::new(1.5e308, 1.5e308)
        } else {
            Complex64::new(1.0, 0.0)
        }
    }
}

fn expect_magnitude_overflow(
    result: Result<PatchedInterpolationResult<Name>, PatchedInterpolationError>,
) -> Projector {
    let (projector, source) = expect_interpolation(result);
    match source {
        InterpolationError::Evaluator { source } => {
            let message = format!("{source:#}");
            assert!(message.contains("magnitude overflows"), "{message}");
            assert!(!message.contains("non-finite"), "{message}");
        }
        other => panic!("expected Evaluator, got {other:?}"),
    }
    projector
}

#[test]
fn overflowing_magnitudes_are_rejected_before_they_pin_a_scale() {
    // An exact root without max_reference would pin an infinite scale.
    let problem = single_node(&[3]);
    let projector = expect_magnitude_overflow(run(
        &DenseEngine::new(),
        &problem,
        &overflow_at(vec![1]),
        &[],
        &sampled_max(2),
    ));
    assert!(projector.is_empty());

    // A root that needs the engine: the user pivot is sampled first.
    let problem = branched();
    let bad = vec![2, 1, 0, 1, 0, 1];
    let projector = expect_magnitude_overflow(run(
        &DenseEngine::new(),
        &problem,
        &overflow_at(bad.clone()),
        &[bad],
        &sampled_max(2),
    ));
    assert!(projector.is_empty());

    // A child patch: the root samples only its pivot and splits at s; the
    // exact child s = 2 meets the value.
    let problem = Problem::new(&[("s", &[3]), ("t", &[8])], &[("s", "t")]);
    let s = problem.site("s", 0);
    let options = sampled_max(2)
        .with_n_initial_pivots(1)
        .with_patch_order(vec![s.clone()]);
    let projector = expect_magnitude_overflow(run(
        &DenseEngine::with_fault(Fault::CapAfterPivots),
        &problem,
        &overflow_at(vec![2, 5]),
        &[vec![0, 0]],
        &options,
    ));
    assert_eq!(projector, Projector::from_pairs([(s, 2)]).unwrap());
}
