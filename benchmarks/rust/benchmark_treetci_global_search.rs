// End-to-end TreeTCI runs dominated by the global pivot search (issue #792).
//
// The body is included by
// `crates/tensor4all-treetci/examples/benchmark_global_search.rs`.
// Protocol notes are in `benchmarks/README.md`.
//
// Every case runs the default TreeTCI loop (`DefaultProposer`, global pivot
// search enabled, fixed seed) and reports the wall time together with the
// quantities that must match between a baseline and a candidate build for the
// timings to be comparable at matched accuracy: the number of function
// evaluations, the rank and error histories, a fingerprint of every pivot set
// the run produced, and the sampled error of the materialized result.
use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::hint::black_box;
use std::time::Instant;

use anyhow::{anyhow, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
use tensor4all_treetci::{
    optimize_with_proposer, to_treetn, DefaultProposer, GlobalIndexBatch, SubtreeKey,
    TreeTCI2, TreeTciEdge, TreeTciGraph, TreeTciOptions,
};
use tensor4all_treetn::{CachedEvaluatorOptions, TreeTNCachedEvaluator};

/// Timed runs per case; the table reports the minimum and the median.
const REPEATS: usize = 3;
/// Random points used for the sampled error of the final approximation.
const N_SAMPLES: usize = 2000;

type Target = Box<dyn Fn(&[usize]) -> f64>;

struct Case {
    name: &'static str,
    local_dims: Vec<usize>,
    edges: Vec<TreeTciEdge>,
    f: Target,
}

/// Quantics value of binary digits (most significant first) in `[0, 1)`.
fn quantics(bits: &[usize]) -> f64 {
    bits.iter()
        .enumerate()
        .map(|(k, &b)| b as f64 * 0.5f64.powi(k as i32 + 1))
        .sum()
}

fn chain_edges(n_sites: usize) -> Vec<TreeTciEdge> {
    (0..n_sites - 1)
        .map(|site| TreeTciEdge::new(site, site + 1))
        .collect()
}

/// `cos(3 (x + y + z) + 0.3)` on 129 binary sites, three 43-bit variables.
/// The exact function has rank 2 across every bond.
fn chain_cos() -> Case {
    let n_sites = 129;
    let chunk = n_sites / 3;
    Case {
        name: "chain_cos_129",
        local_dims: vec![2; n_sites],
        edges: chain_edges(n_sites),
        f: Box::new(move |point: &[usize]| {
            let sum: f64 = point.chunks(chunk).map(quantics).sum();
            (3.0 * sum + 0.3).cos()
        }),
    }
}

/// A one-dimensional quantics chain with `R = 20` bits.
fn quantics_chain() -> Case {
    let n_sites = 20;
    Case {
        name: "quantics_chain_r20",
        local_dims: vec![2; n_sites],
        edges: chain_edges(n_sites),
        f: Box::new(|point: &[usize]| {
            let x = quantics(point);
            (-3.0 * x).exp() * (40.0 * x).cos() + 1.0 / (1.0 + 25.0 * (x - 0.4).powi(2))
        }),
    }
}

/// A branched tree: centre site 0 (degree 3) and three arms of 10 binary
/// sites each, most significant bit next to the centre.
fn branched_tree() -> Case {
    let arm = 10;
    let n_sites = 1 + 3 * arm;
    let mut edges = Vec::with_capacity(n_sites - 1);
    for a in 0..3 {
        let base = 1 + a * arm;
        edges.push(TreeTciEdge::new(0, base));
        for k in 0..arm - 1 {
            edges.push(TreeTciEdge::new(base + k, base + k + 1));
        }
    }
    Case {
        name: "tree_3x10_plus_centre",
        local_dims: vec![2; n_sites],
        edges,
        f: Box::new(move |point: &[usize]| {
            let x = quantics(&point[1..1 + arm]);
            let y = quantics(&point[1 + arm..1 + 2 * arm]);
            let z = quantics(&point[1 + 2 * arm..1 + 3 * arm]);
            let flag = point[0] as f64;
            let gauss = |u: f64, c: f64, w: f64| (-((u - c) / w).powi(2)).exp();
            (1.0 + 0.5 * flag) * gauss(x, 0.3, 0.12) * gauss(y, 0.6, 0.12) * gauss(z, 0.5, 0.3)
                + 0.2 * (x * y + flag * z)
        }),
    }
}

fn hash_pivot_sets(
    hasher: &mut DefaultHasher,
    ijset: &std::collections::HashMap<SubtreeKey, ColMajorArray<usize>>,
) {
    let mut entries: Vec<_> = ijset.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (key, pivots) in entries {
        key.as_slice().hash(hasher);
        pivots.shape().hash(hasher);
        pivots.data().hash(hasher);
    }
}

struct RunResult {
    seconds: f64,
    evaluations: u64,
    ranks: Vec<usize>,
    errors: Vec<f64>,
    pivot_fingerprint: u64,
    sample_fingerprint: u64,
    sampled_rel_error: f64,
}

fn run_case(case: &Case, samples: &[usize], exact: &[f64]) -> Result<RunResult> {
    let n_sites = case.local_dims.len();
    let evaluations = Cell::new(0u64);
    let f = &case.f;
    let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        evaluations.set(evaluations.get() + batch.n_points() as u64);
        Ok(batch
            .data()
            .chunks(batch.n_sites())
            .map(f)
            .collect())
    };
    let options = TreeTciOptions {
        tolerance: 1e-8,
        normalize_error: true,
        seed: Some(1),
        ..Default::default()
    };

    let started = Instant::now();
    let graph = TreeTciGraph::new(n_sites, &case.edges)?;
    let mut state = TreeTCI2::<f64>::new(case.local_dims.clone(), graph)?;
    let initial = vec![0usize; n_sites];
    state.add_global_pivots(std::slice::from_ref(&initial))?;
    state.max_sample_value = f(&initial).abs();
    let tensor4all_treetci::TreeTciOptimizationResult { ranks, errors, .. } =
        optimize_with_proposer(&mut state, evaluate, &options, &DefaultProposer)?;
    let treetn = to_treetn(&state, evaluate, None)?;
    let seconds = started.elapsed().as_secs_f64();

    let mut hasher = DefaultHasher::new();
    for history in &state.ijset_history {
        hash_pivot_sets(&mut hasher, history);
    }
    hash_pivot_sets(&mut hasher, &state.ijset);
    let pivot_fingerprint = hasher.finish();

    let site_indices: Vec<DynIndex> = (0..n_sites)
        .map(|site| {
            let node = treetn
                .node_index(&site)
                .ok_or_else(|| anyhow!("missing site {site}"))?;
            let tensor = treetn
                .tensor(node)
                .ok_or_else(|| anyhow!("missing tensor for site {site}"))?;
            Ok(tensor.indices()[0].clone())
        })
        .collect::<Result<_>>()?;
    let shape = [n_sites, exact.len()];
    let mut evaluator = TreeTNCachedEvaluator::new(
        &treetn,
        &site_indices,
        CachedEvaluatorOptions::<usize>::default(),
    )?;
    let values = evaluator.evaluate_batched(ColMajorArrayRef::new(samples, &shape)?)?;
    let mut hasher = DefaultHasher::new();
    let mut max_abs_error = 0.0f64;
    for (value, reference) in values.iter().zip(exact) {
        value.real().to_bits().hash(&mut hasher);
        max_abs_error = max_abs_error.max((value.real() - reference).abs());
    }
    let max_abs_exact = exact.iter().fold(0.0f64, |m, v| m.max(v.abs()));

    Ok(RunResult {
        seconds,
        evaluations: evaluations.get(),
        ranks,
        errors,
        pivot_fingerprint,
        sample_fingerprint: hasher.finish(),
        sampled_rel_error: black_box(max_abs_error / max_abs_exact),
    })
}

fn main() -> Result<()> {
    let commit = option_env!("T4A_BENCH_GIT_COMMIT").unwrap_or("unknown");
    let threads = ["RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        .iter()
        .map(|name| format!("{name}={}", std::env::var(name).unwrap_or_default()))
        .collect::<Vec<_>>()
        .join(" ");
    println!("build commit={commit} {threads} repeats={REPEATS}");

    let filter = std::env::args().nth(1);
    for case in [chain_cos(), quantics_chain(), branched_tree()] {
        if filter.as_deref().is_some_and(|name| name != case.name) {
            continue;
        }
        let n_sites = case.local_dims.len();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut samples = Vec::with_capacity(n_sites * N_SAMPLES);
        for _ in 0..N_SAMPLES {
            for &dim in &case.local_dims {
                samples.push(rng.random_range(0..dim));
            }
        }
        let exact: Vec<f64> = samples.chunks(n_sites).map(|p| (case.f)(p)).collect();

        let mut times = Vec::with_capacity(REPEATS);
        let mut first: Option<RunResult> = None;
        for _ in 0..REPEATS {
            let result = run_case(&case, &samples, &exact)?;
            times.push(result.seconds);
            if let Some(reference) = &first {
                // Repeats of one build must preserve the pivot fingerprint
                // and function-evaluation count.
                if reference.pivot_fingerprint != result.pivot_fingerprint
                    || reference.evaluations != result.evaluations
                {
                    return Err(anyhow!("{}: repeated runs disagree", case.name));
                }
            } else {
                first = Some(result);
            }
        }
        times.sort_by(f64::total_cmp);
        let result = first.ok_or_else(|| anyhow!("no runs"))?;
        println!(
            "case={} sites={} min_s={:.4} median_s={:.4} evals={} iters={} final_rank={} \
             pivots={:016x} samples={:016x} sampled_rel_err={:.2e}",
            case.name,
            n_sites,
            times[0],
            times[REPEATS / 2],
            result.evaluations,
            result.ranks.len(),
            result.ranks.last().copied().unwrap_or(0),
            result.pivot_fingerprint,
            result.sample_fingerprint,
            result.sampled_rel_error,
        );
        println!("  ranks={:?}", result.ranks);
        println!(
            "  errors={:?}",
            result
                .errors
                .iter()
                .map(|e| format!("{:016x}", e.to_bits()))
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}
