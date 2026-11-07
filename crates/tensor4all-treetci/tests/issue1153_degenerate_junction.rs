//! Regression test for issue #1153: bonds into a degree-3 vertex of local
//! dimension 1 must not stay pinned at rank 1 while the run reports
//! `Converged`.
//!
//! The tree is a comb: three binary chains of `R` sites joined at a junction
//! site of local dimension 1 (site `3 R`). Site numbering is
//! `site = 3 * level + chain`, so the sorted edge order goes leaf level to
//! junction on every pass. Global pivots are disabled so that only the local
//! updates and the initial pivots can grow the junction bonds.
//!
//! Before the fix, every junction bond stayed at rank 1: the junction-side
//! candidate set was (pivots of the other two bonds) x (local dimension 1) =
//! 1 candidate, the bond error was computed on a 4 x 1 matrix and was zero,
//! and the run returned `Converged` with a maximum relative error of 1.68
//! (`delta = 4`) and 1.32 (`delta = 1`) on random points. A run that cannot
//! measure a bond error now reports `DegenerateCandidateSet`, and the
//! initial pivots are enriched around the junction as in xfac, so the bonds
//! grow and the run converges.

use anyhow::Result;
use rand::{Rng, SeedableRng};
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treetci::{
    optimize_with_proposer, to_treetn, AllEdges, DefaultProposer, GlobalIndexBatch, TreeTCI2,
    TreeTciEdge, TreeTciGraph, TreeTciOptimizationResult, TreeTciOptions, TreeTciTermination,
};
use tensor4all_treetn::TreeTNCachedEvaluator;

/// Bits per axis.
const R: usize = 10;
const N: usize = 1 << R;
const L: f64 = 64.0;
const JUNCTION: usize = 3 * R;
const N_SITES: usize = 3 * R + 1;
const TOLERANCE: f64 = 1e-6;

/// Site of bit `level` (0 = least significant, the leaf) of `axis`.
fn site(level: usize, axis: usize) -> usize {
    3 * level + axis
}

/// A smooth 3D function: separable windows times a radially oscillating term
/// whose frequency is set by `delta`.
fn f(n: [usize; 3], delta: f64) -> f64 {
    let k: Vec<f64> = n
        .iter()
        .map(|&i| -L / 2.0 + i as f64 * L / N as f64)
        .collect();
    let kk = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
    let nz2 = if kk > 0.0 {
        k[2] * k[2] / (kk * kk)
    } else {
        0.0
    };
    let sinc = |x: f64| if x == 0.0 { 1.0 } else { x.sin() / x };
    let w = sinc(k[0] / 16.0) * sinc(k[1] / 16.0) * sinc(k[2] / 4.0);
    -w * (nz2 + (-(kk * 0.25).powi(2) / 2.0).exp() * (kk * delta).cos() * (1.0 - nz2))
}

fn decode(point: &[usize]) -> [usize; 3] {
    let mut n = [0; 3];
    for (axis, value) in n.iter_mut().enumerate() {
        for level in 0..R {
            *value |= point[site(level, axis)] << level;
        }
    }
    n
}

fn comb_graph() -> Result<TreeTciGraph> {
    let mut edges = Vec::new();
    for axis in 0..3 {
        for level in 0..R - 1 {
            edges.push(TreeTciEdge::new(site(level, axis), site(level + 1, axis)));
        }
        edges.push(TreeTciEdge::new(site(R - 1, axis), JUNCTION));
    }
    Ok(TreeTciGraph::new(N_SITES, &edges)?)
}

struct Run {
    result: TreeTciOptimizationResult,
    junction_ranks: Vec<usize>,
    max_rel_error: f64,
}

/// Run the issue's reproducer (`sorted` order, `global off`) and measure the
/// maximum relative error of the materialized tree on 2000 random points.
fn run(delta: f64) -> Result<Run> {
    let graph = comb_graph()?;
    let mut local_dims = vec![2; 3 * R];
    local_dims.push(1);

    // Start at the grid centre.
    let mut start = vec![0; N_SITES];
    for axis in 0..3 {
        start[site(R - 1, axis)] = 1;
    }

    let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let (s, p) = (batch.n_sites(), batch.n_points());
        Ok((0..p)
            .map(|j| f(decode(&batch.data()[j * s..(j + 1) * s]), delta))
            .collect())
    };

    let options = TreeTciOptions {
        tolerance: TOLERANCE,
        max_iter: 30,
        enable_global_pivots: false,
        seed: Some(11),
        ..Default::default()
    };
    let mut state = TreeTCI2::<f64>::new(local_dims, graph)?;
    state.add_global_pivots(&[start])?;
    let result = optimize_with_proposer(
        &mut state,
        evaluate,
        &options,
        &DefaultProposer,
        &AllEdges,
        &mut (),
    )?;

    let junction_ranks = (0..3)
        .map(|axis| -> Result<usize> {
            let (key, _) = state
                .graph()
                .subregion_vertices(TreeTciEdge::new(site(R - 1, axis), JUNCTION))?;
            Ok(state.pivot_set(&key).map_or(0, |pivots| pivots.shape()[1]))
        })
        .collect::<Result<Vec<_>>>()?;

    let tree = to_treetn(&state, evaluate, None)?;
    let indices = (0..N_SITES)
        .map(|s| {
            tree.site_space(&s)
                .and_then(|set| set.iter().next().cloned())
                .ok_or_else(|| anyhow::anyhow!("no site index at {s}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut evaluator = TreeTNCachedEvaluator::<usize>::new(&tree, &indices, Default::default())?;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let n_test = 2000;
    let mut points = Vec::with_capacity(n_test * N_SITES);
    let mut exact = Vec::with_capacity(n_test);
    for _ in 0..n_test {
        let n = [
            rng.random_range(0..N),
            rng.random_range(0..N),
            rng.random_range(0..N),
        ];
        let mut point = vec![0; N_SITES];
        for (axis, &value) in n.iter().enumerate() {
            for level in 0..R {
                point[site(level, axis)] = (value >> level) & 1;
            }
        }
        points.extend_from_slice(&point);
        exact.push(f(n, delta));
    }
    let shape = [N_SITES, n_test];
    let approx = evaluator.evaluate_batched(ColMajorArrayRef::new(&points, &shape)?)?;
    let scale = exact.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let max_rel_error = approx
        .iter()
        .zip(&exact)
        .fold(0.0f64, |m, (a, e)| m.max((a.real() - e).abs()))
        / scale;

    Ok(Run {
        result,
        junction_ranks,
        max_rel_error,
    })
}

fn check(delta: f64) -> Result<()> {
    let run = run(delta)?;
    println!(
        "delta {delta}: termination {:?}, iterations {}, junction ranks {:?}, max rel error {:.3e}",
        run.result.termination,
        run.result.ranks.len(),
        run.junction_ranks,
        run.max_rel_error
    );
    // The junction bonds must leave rank 1.
    assert!(
        run.junction_ranks.iter().all(|&rank| rank > 1),
        "junction bonds stayed at rank 1: {:?}",
        run.junction_ranks
    );
    // With the enriched initial pivots the local updates grow the junction
    // bonds and the run converges near the tolerance (measured: 1.43e-6 for
    // `delta = 4`, 1.45e-6 for `delta = 1`; the issue reported 1.68 / 1.32).
    assert_eq!(run.result.termination, TreeTciTermination::Converged);
    assert!(
        run.max_rel_error < 10.0 * TOLERANCE,
        "Converged with max relative error {:.3e}",
        run.max_rel_error
    );
    Ok(())
}

#[test]
fn comb_junction_of_local_dimension_one_delta4() -> Result<()> {
    check(4.0)
}

#[test]
fn comb_junction_of_local_dimension_one_delta1() -> Result<()> {
    check(1.0)
}

/// A product function is exactly rank 1. After the enriched initial pivots
/// collapse to rank 1, the junction side of every junction bond has one
/// candidate again, so the zero bond errors are not measurements and the run
/// must report `DegenerateCandidateSet`, not `Converged`, with or without
/// the global search.
#[test]
fn product_function_reports_unmeasured_junction_bonds() -> Result<()> {
    let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let (s, p) = (batch.n_sites(), batch.n_points());
        Ok((0..p)
            .map(|j| {
                batch.data()[j * s..(j + 1) * s]
                    .iter()
                    .map(|&x| 1.0 + x as f64)
                    .product()
            })
            .collect())
    };
    let mut local_dims = vec![2; 3 * R];
    local_dims.push(1);
    for enable_global_pivots in [false, true] {
        let mut state = TreeTCI2::<f64>::new(local_dims.clone(), comb_graph()?)?;
        state.add_global_pivots(&[vec![0; N_SITES]])?;
        let options = TreeTciOptions {
            tolerance: TOLERANCE,
            enable_global_pivots,
            seed: Some(1),
            ..Default::default()
        };
        let result = optimize_with_proposer(
            &mut state,
            evaluate,
            &options,
            &DefaultProposer,
            &AllEdges,
            &mut (),
        )?;
        assert_eq!(
            result.termination,
            TreeTciTermination::DegenerateCandidateSet
        );
        assert_eq!(result.ranks, vec![1; 3]);
        assert_eq!(result.errors, vec![0.0; 3]);
    }
    Ok(())
}
