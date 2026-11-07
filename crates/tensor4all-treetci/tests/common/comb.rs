//! The comb reproducer shared by issues #1153 and #1159.
//!
//! The tree is a comb: three binary chains of `R` sites joined at a junction
//! site of local dimension 1 (site `3 R`). Site numbering is
//! `site = 3 * level + chain`, so the sorted edge order goes leaf level to
//! junction on every pass.

use anyhow::Result;
use rand::{Rng, SeedableRng};
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treetci::{
    optimize_with_proposer, to_treetn, AllEdges, GlobalIndexBatch, PivotCandidateProposer,
    TreeTCI2, TreeTciEdge, TreeTciGraph, TreeTciOptimizationResult, TreeTciOptions,
};
use tensor4all_treetn::TreeTNCachedEvaluator;

/// Bits per axis.
pub const R: usize = 10;
const N: usize = 1 << R;
const L: f64 = 64.0;
pub const JUNCTION: usize = 3 * R;
pub const N_SITES: usize = 3 * R + 1;
pub const TOLERANCE: f64 = 1e-6;

/// Target function on the comb.
#[derive(Clone, Copy, Debug)]
pub enum Target {
    /// The non-separable function of #1153: separable windows times a
    /// radially oscillating term whose frequency is `delta`.
    Oscillating { delta: f64 },
    /// The window factor alone, `sinc(kx/16) sinc(ky/16) sinc(kz/4)`: exactly
    /// rank 1 across every junction bond (#1159).
    Separable,
}

/// Site of bit `level` (0 = least significant, the leaf) of `axis`.
pub fn site(level: usize, axis: usize) -> usize {
    3 * level + axis
}

fn f(n: [usize; 3], target: Target) -> f64 {
    let k: Vec<f64> = n
        .iter()
        .map(|&i| -L / 2.0 + i as f64 * L / N as f64)
        .collect();
    let sinc = |x: f64| if x == 0.0 { 1.0 } else { x.sin() / x };
    let w = sinc(k[0] / 16.0) * sinc(k[1] / 16.0) * sinc(k[2] / 4.0);
    let delta = match target {
        Target::Separable => return w,
        Target::Oscillating { delta } => delta,
    };
    let kk = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
    let nz2 = if kk > 0.0 {
        k[2] * k[2] / (kk * kk)
    } else {
        0.0
    };
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

/// The comb graph of the reproducer: three chains of `R` binary sites meeting at
/// one junction of local dimension 1.
///
/// # Errors
///
/// [`TreeTciError::InvalidGraph`] if an edge names a site outside `0..N_SITES`
/// or repeats one, or if the edge list is disconnected, missing a vertex, or
/// otherwise not a tree.
pub fn comb_graph() -> Result<TreeTciGraph> {
    let mut edges = Vec::new();
    for axis in 0..3 {
        for level in 0..R - 1 {
            edges.push(TreeTciEdge::new(site(level, axis), site(level + 1, axis)));
        }
        edges.push(TreeTciEdge::new(site(R - 1, axis), JUNCTION));
    }
    Ok(TreeTciGraph::new(N_SITES, &edges)?)
}

/// Local dimensions of the comb: binary sites and a junction of dimension 1.
pub fn comb_local_dims() -> Vec<usize> {
    let mut local_dims = vec![2; 3 * R];
    local_dims.push(1);
    local_dims
}

pub struct Run {
    pub result: TreeTciOptimizationResult,
    pub junction_ranks: Vec<usize>,
    pub max_rel_error: f64,
}

/// Run the reproducer (`sorted` order, seed 11, `max_iter = 30`, memo on)
/// from the grid centre and measure the maximum relative error of the
/// materialized tree on 2000 random points.
///
/// # Errors
///
/// Returns an error if the graph cannot be built ([`comb_graph`]), if the
/// optimizer reports a failure, or if the sampled relative error is not finite.
pub fn run<P: PivotCandidateProposer>(
    target: Target,
    enable_global_pivots: bool,
    proposer: &P,
) -> Result<Run> {
    let graph = comb_graph()?;

    // Start at the grid centre.
    let mut start = vec![0; N_SITES];
    for axis in 0..3 {
        start[site(R - 1, axis)] = 1;
    }

    let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let (s, p) = (batch.n_sites(), batch.n_points());
        Ok((0..p)
            .map(|j| f(decode(&batch.data()[j * s..(j + 1) * s]), target))
            .collect())
    };

    let options = TreeTciOptions {
        tolerance: TOLERANCE,
        max_iter: 30,
        enable_global_pivots,
        seed: Some(11),
        evaluation_cache_bytes: Some(256 * 1024 * 1024),
        ..Default::default()
    };
    let mut state = TreeTCI2::<f64>::new(comb_local_dims(), graph)?;
    state.add_global_pivots(&[start])?;
    let result =
        optimize_with_proposer(&mut state, evaluate, &options, proposer, &AllEdges, &mut ())?;

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
        exact.push(f(n, target));
    }
    let shape = [N_SITES, n_test];
    let approx = evaluator.evaluate_batched(ColMajorArrayRef::new(&points, &shape)?)?;
    let scale = exact.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let max_rel_error = approx
        .iter()
        .zip(&exact)
        .fold(0.0f64, |m, (a, e)| m.max((a.real() - e).abs()))
        / scale;

    let run = Run {
        result,
        junction_ranks,
        max_rel_error,
    };
    println!(
        "{target:?} global {enable_global_pivots}: termination {:?}, iterations {}, \
         max rank {}, junction ranks {:?}, distinct points {}, max rel error {:.3e}, \
         reported error {:.3e}",
        run.result.termination,
        run.result.ranks.len(),
        run.result.ranks.last().copied().unwrap_or(0),
        run.junction_ranks,
        run.result.evaluation.evaluated_points,
        run.max_rel_error,
        run.result.errors.last().copied().unwrap_or(f64::NAN),
    );
    Ok(run)
}
