//! Reproducible opt-in confirmation evidence for #1352/#1371/#1372.
//!
//! Gaussian/comb definitions and fixed xorshift validation follow Simon's public
//! reproducers in issues #1371 and #1264. This is a diagnostic driver, not a new
//! interpolation policy. Timings are publishable only under `ai-run measure`;
//! the call includes replay and final materialization, excludes validation.
//! Dense validation is limited to the explicitly small 16-site Gaussian.

use anyhow::{ensure, Context, Result};
use std::cell::Cell;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_tensorbackend::{explicit::Session, CpuBackend, CpuExecutionContext};
use tensor4all_treetci::{
    crossinterpolate2_in, DefaultProposer, GlobalIndexBatch, TreeTciEdge, TreeTciGraph,
    TreeTciIteration, TreeTciObserver, TreeTciOptions,
};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

trait Problem {
    const NAME: &'static str;
    const SITES: usize;
    fn value(point: &[usize]) -> f64;
    fn graph() -> Result<TreeTciGraph> {
        Ok(TreeTciGraph::linear_chain(Self::SITES)?)
    }
}

struct Gaussian;
impl Problem for Gaussian {
    const NAME: &'static str = "gaussian16";
    const SITES: usize = 16;
    fn value(point: &[usize]) -> f64 {
        let a = point[..8].iter().fold(0, |n, &bit| 2 * n + bit) as f64;
        let c = point[8..].iter().fold(0, |n, &bit| 2 * n + bit) as f64;
        (-((a - c) / 40.0).powi(2)).exp()
    }
}

struct Comb;
impl Problem for Comb {
    const NAME: &'static str = "comb60";
    const SITES: usize = 60;
    fn value(point: &[usize]) -> f64 {
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
    fn graph() -> Result<TreeTciGraph> {
        let mut edges = vec![TreeTciEdge::new(0, 1), TreeTciEdge::new(1, 2)];
        for level in 0..19 {
            for axis in 0..3 {
                edges.push(TreeTciEdge::new(3 * level + axis, 3 * (level + 1) + axis));
            }
        }
        Ok(TreeTciGraph::new(60, &edges)?)
    }
}

struct Observed<'a> {
    oracle_max: &'a Cell<f64>,
    scale_at_stop: f64,
}
impl TreeTciObserver for Observed<'_> {
    fn iteration(&mut self, _: &TreeTciIteration<'_>) {
        self.scale_at_stop = self.oracle_max.get();
    }
}

fn indices(tree: &TreeTN<IdxTensor, usize>, sites: usize) -> Result<Vec<DynIndex>> {
    (0..sites)
        .map(|site| {
            tree.site_space(&site)
                .context("validation site missing")?
                .iter()
                .next()
                .cloned()
                .context("validation physical index missing")
        })
        .collect()
}

fn validate<P: Problem>(
    tree: &TreeTN<IdxTensor, usize>,
    context: &CpuExecutionContext,
) -> Result<(f64, f64, usize)> {
    let indices = indices(tree, P::SITES)?;
    if P::SITES <= 16 {
        let points = 1usize << P::SITES;
        let mut point = vec![0; P::SITES];
        let mut values = Vec::with_capacity(points);
        for _ in 0..points {
            values.push(P::value(&point));
            for bit in &mut point {
                *bit ^= 1;
                if *bit != 0 {
                    break;
                }
            }
        }
        let reference = IdxTensor::from_dense(indices, values)?;
        let difference = tree.to_dense()?.sub(&reference)?;
        return Ok((
            difference.maxabs()? / reference.maxabs()?,
            difference.norm()? / reference.norm()?,
            points,
        ));
    }
    // The same fixed independent xorshift stream is used by every policy/seed.
    let points = 20_000;
    let mut stream = 0x9e3779b97f4a7c15u64;
    let mut data = Vec::with_capacity(P::SITES * points);
    let mut exact = Vec::with_capacity(points);
    let mut point = vec![0; P::SITES];
    for _ in 0..points {
        for bit in &mut point {
            stream ^= stream << 13;
            stream ^= stream >> 7;
            stream ^= stream << 17;
            *bit = (stream >> 63) as usize;
        }
        data.extend_from_slice(&point);
        exact.push(P::value(&point));
    }
    let mut evaluator = TreeTNCachedEvaluator::new(
        tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )?;
    let shape = [P::SITES, points];
    let batch = ColMajorArrayRef::new(&data, &shape)?;
    let got: Vec<f64> = context.with_backend(|backend| {
        let mut session = Session::new(backend.engine_mut());
        evaluator.evaluate_batched_typed_on_session(&mut session, batch, EvaluationHint::default())
    })?;
    let (mut maximum, mut scale, mut sum_sq, mut norm_sq) = (0.0_f64, 0.0_f64, 0.0, 0.0);
    for (&value, &reference) in got.iter().zip(&exact) {
        let difference = (value - reference).abs();
        maximum = maximum.max(difference);
        scale = scale.max(reference.abs());
        sum_sq += difference * difference;
        norm_sq += reference * reference;
    }
    Ok((maximum / scale, (sum_sq / norm_sq).sqrt(), points))
}

fn run<P: Problem>(seed: u64, tolerance: f64, confirm: bool, timed: bool) -> Result<()> {
    ensure!(
        tolerance.is_finite() && tolerance > 0.0,
        "positive finite tolerance required"
    );
    let backend = CpuBackend::with_threads(1)?;
    ensure!(backend.num_threads() == 1, "provider thread budget changed");
    let context = CpuExecutionContext::from_backend(backend);
    let oracle_max = Cell::new(0.0_f64);
    let callback_points = Cell::new(0usize);
    let mut observed = Observed {
        oracle_max: &oracle_max,
        scale_at_stop: 0.0,
    };
    let options = TreeTciOptions {
        tolerance,
        max_iter: 30,
        seed: Some(seed),
        confirm_convergence: confirm,
        validation_cache_bytes: 32 * 1024 * 1024,
        evaluation_cache_bytes: None,
        ..Default::default()
    };
    let nsearch = options.nsearch;
    if timed {
        let prime = std::time::Instant::now();
        while prime.elapsed().as_secs_f64() < 2.0 {
            std::hint::spin_loop();
        }
    }
    let started = std::time::Instant::now();
    let result = crossinterpolate2_in(
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
            let values = batch
                .data()
                .chunks_exact(P::SITES)
                .map(P::value)
                .collect::<Vec<_>>();
            oracle_max.set(values.iter().fold(oracle_max.get(), |a, b| a.max(b.abs())));
            callback_points.set(callback_points.get() + values.len());
            Ok(values)
        },
        vec![2; P::SITES],
        P::graph()?,
        vec![vec![0; P::SITES]],
        options,
        Some(0),
        &DefaultProposer,
        &mut observed,
        &context,
    )?;
    let seconds = started.elapsed().as_secs_f64();
    ensure!(
        callback_points.get() == result.evaluation.evaluated_points,
        "oracle counter mismatch"
    );
    let (relmax, rel_l2, validation_points) = validate::<P>(&result.treetn, &context)?;
    let reported = *result.errors.last().context("missing recorded error")?;
    ensure!(
        relmax.is_finite() && rel_l2.is_finite() && reported.is_finite(),
        "non-finite diagnostic"
    );
    let error_bits = result
        .errors
        .iter()
        .map(|e| e.to_bits())
        .collect::<Vec<_>>();
    let confirmation_counts = [
        result.evaluation.confirmation_checks,
        result.evaluation.confirmation_failures,
        result.evaluation.confirmation_pivots,
    ];
    println!(
        "{{\"case\":\"{}\",\"seed\":{},\"tolerance\":{:.17e},\"confirm\":{},\"termination\":\"{:?}\",\"oracle_points\":{},\"requested_points\":{},\"reported_error\":{:.17e},\"reported_scale\":{:.17e},\"validation_relmax\":{:.17e},\"validation_rel_l2\":{:.17e},\"validation_points\":{},\"nsearch\":{},\"ranks\":{:?},\"error_bits\":{:?},\"confirmation_counts\":{:?},\"seconds\":{:.9}}}",
        P::NAME, seed, tolerance, confirm, result.termination,
        result.evaluation.evaluated_points, result.evaluation.requested_points,
        reported, observed.scale_at_stop, relmax, rel_l2, validation_points,
        nsearch, result.ranks, error_bits, confirmation_counts, seconds,
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 3,
        "usage: confirmation_comparison <all|gaussian16|comb60> <on|off> <checked|timed>"
    );
    ensure!(
        matches!(args[0].as_str(), "all" | "gaussian16" | "comb60"),
        "unknown case"
    );
    ensure!(
        matches!(args[1].as_str(), "on" | "off"),
        "unknown confirmation setting"
    );
    ensure!(
        matches!(args[2].as_str(), "checked" | "timed"),
        "unknown run kind"
    );
    let confirm = args[1] == "on";
    let timed = args[2] == "timed";
    tensor4all_tensorbackend::set_default_cpu_threads(std::num::NonZeroUsize::MIN)?;
    if args[0] != "comb60" {
        for tolerance in [1e-4, 1e-6, 1e-8] {
            for seed in 1..=6 {
                run::<Gaussian>(seed, tolerance, confirm, timed)?;
            }
        }
    }
    if args[0] != "gaussian16" {
        for seed in 1..=3 {
            run::<Comb>(seed, 1e-8, confirm, timed)?;
        }
    }
    Ok(())
}
