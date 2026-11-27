//! Reproduce the numerical acceptance families of issue #1264 without timing.
//!
//! Arguments: `chain|comb first_seed last_seed per-bond|sampled [bits_per_axis]`.
//! Defaults: `chain 1 20 per-bond` (24 sites); comb defaults to 60 sites.
//! Compare each seed at rank slack 0 and 0.03 on the same 20,000 held-out points.
use anyhow::{bail, ensure, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, StopRule, TreeTciEdge, TreeTciGraph,
    TreeTciOptions,
};

fn target(bits: &[usize], r: usize) -> f64 {
    let mut coordinates = [0u64; 3];
    for (site, &bit) in bits.iter().enumerate() {
        coordinates[site % 3] |= (bit as u64) << (r - 1 - site / 3);
    }
    let h = 2.0 / (1u64 << r) as f64;
    let [x, y, z] = coordinates.map(|q| -1.0 + h * q as f64);
    [
        (0.13_f64, -0.21_f64, 0.07_f64, 0.05_f64, 1.0_f64),
        (-0.37, 0.29, -0.11, 0.08, -0.6),
        (0.41, 0.33, 0.27, 0.03, 0.4),
    ]
    .iter()
    .map(|&(a, b, c, s, w)| {
        w / ((x - a).powi(2) + (y - b).powi(2) + (z - c).powi(2) + s * s).sqrt()
    })
    .sum()
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let kind = args.first().map(String::as_str).unwrap_or("chain");
    let first_seed: u64 = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(1);
    let last_seed: u64 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(20);
    ensure!(
        first_seed <= last_seed,
        "first_seed must not exceed last_seed"
    );
    let stop_rule = match args.get(3).map(String::as_str).unwrap_or("per-bond") {
        "per-bond" => StopRule::PerBond,
        "sampled" => StopRule::Sampled,
        _ => bail!("stop rule must be per-bond or sampled"),
    };
    let default_r = match kind {
        "chain" => 8,
        "comb" => 20,
        _ => bail!("topology must be chain or comb"),
    };
    let r: usize = args
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(default_r);
    ensure!((1..=20).contains(&r), "bits_per_axis must be in 1..=20");
    let n_sites = 3 * r;
    let tolerance = if kind == "chain" { 1e-6 } else { 1e-8 };
    let graph = if kind == "chain" {
        TreeTciGraph::linear_chain(n_sites)?
    } else {
        let mut edges = vec![TreeTciEdge::new(0, 1), TreeTciEdge::new(1, 2)];
        for level in 0..r - 1 {
            for axis in 0..3 {
                edges.push(TreeTciEdge::new(3 * level + axis, 3 * (level + 1) + axis));
            }
        }
        TreeTciGraph::new(n_sites, &edges)?
    };
    let n_samples = 20_000;
    let mut rng = ChaCha8Rng::seed_from_u64(1264);
    let samples: Vec<Vec<usize>> = (0..n_samples)
        .map(|_| (0..n_sites).map(|_| rng.random_range(0..2)).collect())
        .collect();
    let expected: Vec<_> = samples.iter().map(|bits| target(bits, r)).collect();
    for seed in first_seed..=last_seed {
        for rank_slack in [0.0, 0.03] {
            let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
                let mut bits = vec![0; n_sites];
                Ok((0..batch.n_points())
                    .map(|p| {
                        for (s, bit) in bits.iter_mut().enumerate() {
                            *bit = batch.get(s, p).unwrap();
                        }
                        target(&bits, r)
                    })
                    .collect())
            };
            let result = crossinterpolate2(
                evaluate,
                vec![2; n_sites],
                graph.clone(),
                vec![vec![0; n_sites]],
                TreeTciOptions {
                    tolerance,
                    rank_slack,
                    stop_rule,
                    max_iter: 30,
                    seed: Some(seed),
                    ..Default::default()
                },
                Some(0),
                &DefaultProposer,
                &mut (),
            )?;
            let (indices, _) = result.treetn.all_site_indices()?;
            let positions: Vec<_> = (0..n_sites)
                .map(|site| {
                    let space = result.treetn.site_space(&site).unwrap();
                    let index = space.iter().next().unwrap();
                    indices.iter().position(|i| i == index).unwrap()
                })
                .collect();
            let mut data = vec![0; n_sites * n_samples];
            for (p, bits) in samples.iter().enumerate() {
                for (site, &bit) in bits.iter().enumerate() {
                    data[p * n_sites + positions[site]] = bit;
                }
            }
            let actual = result.treetn.evaluate(
                &indices,
                ColMajorArrayRef::new(&data, &[n_sites, n_samples])?,
            )?;
            let mut difference_squared = 0.0;
            let mut reference_squared = 0.0;
            let mut max_difference = 0.0_f64;
            let mut max_reference = 0.0_f64;
            for (a, &e) in actual.iter().zip(&expected) {
                let d = a.real() - e;
                difference_squared += d * d;
                reference_squared += e * e;
                max_difference = max_difference.max(d.abs());
                max_reference = max_reference.max(e.abs());
            }
            println!(
                "topology={kind} bits={r} seed={seed} rule={stop_rule:?} slack={rank_slack} termination={:?} sweeps={} evaluations={} relative_l2={:.8e} relative_max={:.8e} ranks={:?} errors={:?} nglobal_pivots={:?}",
                result.termination, result.ranks.len(), result.evaluation.evaluated_points,
                (difference_squared / reference_squared).sqrt(), max_difference / max_reference,
                result.ranks, result.errors, result.nglobal_pivots,
            );
        }
    }
    Ok(())
}
