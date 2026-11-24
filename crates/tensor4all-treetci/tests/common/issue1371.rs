//! Targets and checks shared by the #1371 local-threshold regressions.

use anyhow::Result;
use tensor4all_core::{ColMajorArrayRef, IdxTensor};
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciEdge, TreeTciGraph,
    TreeTciOptions, TreeTciRunResult,
};
use tensor4all_treetn::TreeTNCachedEvaluator;

/// Run `crossinterpolate2` on `graph` from the all-zero pivot, materializing
/// at site 0.
pub fn run_tree(
    value: impl Fn(&[usize]) -> f64,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    options: TreeTciOptions,
) -> TreeTciRunResult {
    let n_sites = local_dims.len();
    crossinterpolate2(
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
            Ok(batch
                .data()
                .chunks_exact(batch.n_sites())
                .map(&value)
                .collect())
        },
        local_dims,
        graph,
        vec![vec![0; n_sites]],
        options,
        Some(0),
        &DefaultProposer,
        &mut (),
    )
    .unwrap()
}

/// Max-abs error of the network over every configuration, divided by the
/// largest target magnitude. Column-major: site 0 varies fastest.
pub fn dense_relative_error(
    result: &TreeTciRunResult,
    value: impl Fn(&[usize]) -> f64,
    n_sites: usize,
    dim: usize,
) -> f64 {
    let indices = (0..n_sites)
        .map(|site| {
            let space = result.treetn.site_space(&site).unwrap();
            space.iter().next().unwrap().clone()
        })
        .collect();
    let total = dim.pow(n_sites as u32);
    let values = (0..total)
        .map(|flat| {
            let point: Vec<usize> = (0..n_sites)
                .map(|site| (flat / dim.pow(site as u32)) % dim)
                .collect();
            value(&point)
        })
        .collect();
    let reference = IdxTensor::from_dense(indices, values).unwrap();
    let difference = result.treetn.to_dense().unwrap().sub(&reference).unwrap();
    difference.maxabs().unwrap() / reference.maxabs().unwrap()
}

/// Max-abs error over 20,000 fixed xorshift points of a binary network,
/// divided by the largest target magnitude among them.
pub fn sampled_relative_error(
    result: &TreeTciRunResult,
    value: impl Fn(&[usize]) -> f64,
    n_sites: usize,
) -> f64 {
    let points = 20_000;
    let mut stream = 0x9e37_79b9_7f4a_7c15_u64;
    let mut data = Vec::with_capacity(n_sites * points);
    let mut exact = Vec::with_capacity(points);
    let mut point = vec![0; n_sites];
    for _ in 0..points {
        for bit in &mut point {
            stream ^= stream << 13;
            stream ^= stream >> 7;
            stream ^= stream << 17;
            *bit = (stream >> 63) as usize;
        }
        data.extend_from_slice(&point);
        exact.push(value(&point));
    }
    let indices: Vec<_> = (0..n_sites)
        .map(|site| {
            let space = result.treetn.site_space(&site).unwrap();
            space.iter().next().unwrap().clone()
        })
        .collect();
    let mut evaluator =
        TreeTNCachedEvaluator::<usize>::new(&result.treetn, &indices, Default::default()).unwrap();
    let shape = [n_sites, points];
    let got = evaluator
        .evaluate_batched(ColMajorArrayRef::new(&data, &shape).unwrap())
        .unwrap();
    let scale = exact.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(&exact)
        .fold(0.0_f64, |m, (g, e)| m.max((g.real() - e).abs()))
        / scale
}

/// The #1371 target: a Gaussian ridge between two 8-bit coordinates on a
/// 16-site binary chain.
pub fn gaussian_ridge(point: &[usize]) -> f64 {
    let a = point[..8].iter().fold(0, |n, &bit| 2 * n + bit) as f64;
    let c = point[8..].iter().fold(0, |n, &bit| 2 * n + bit) as f64;
    (-((a - c) / 40.0).powi(2)).exp()
}

/// Slow-decay target on the 60-site comb: three Coulomb-like peaks in 3D at
/// 20 bits per axis, with site `3 * level + axis` holding bit `19 - level`.
pub fn comb60_target(point: &[usize]) -> f64 {
    let mut coordinates = [0usize; 3];
    for (site, &bit) in point.iter().enumerate() {
        coordinates[site % 3] |= bit << (19 - site / 3);
    }
    let [x, y, z] = coordinates.map(|n| -1.0 + 2.0 * n as f64 / (1usize << 20) as f64);
    [
        (0.13, -0.21, 0.07, 0.05, 1.0),
        (-0.37, 0.29, -0.11, 0.08, -0.6),
        (0.41, 0.33, 0.27, 0.03, 0.4),
    ]
    .iter()
    .map(|&(a, b, c, s, weight)| {
        weight / ((x - a).powi(2) + (y - b).powi(2) + (z - c).powi(2) + s * s).sqrt()
    })
    .sum()
}

/// The 60-site comb: three 20-site chains whose first sites are joined by a
/// path (59 edges).
pub fn comb60_graph() -> TreeTciGraph {
    let mut edges = vec![TreeTciEdge::new(0, 1), TreeTciEdge::new(1, 2)];
    for level in 0..19 {
        for axis in 0..3 {
            edges.push(TreeTciEdge::new(3 * level + axis, 3 * (level + 1) + axis));
        }
    }
    TreeTciGraph::new(60, &edges).unwrap()
}
