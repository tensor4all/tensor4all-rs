//! Bounded paired RSI/TreeACI experiment. Run through benchmarks/tree-rsi/run.py.
//! Output is JSONL under target, never a committed benchmark result.
// Share deterministic input construction only; no ACI dependency in the RSI crate.
mod diagnostics;
#[path = "../../../crates/tensor4all-treersi/examples/support/mod.rs"]
mod support;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use std::{fs::OpenOptions, io::Write, path::Path, time::Instant};
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treeaci::TreeAciOptions;
use tensor4all_treersi::{hadamard_many, TreeRsiOptions};
use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};

type Tree = TreeTN<IdxTensor, usize>;
type Error = Box<dyn std::error::Error>;

/// Fixed experiment settings; `main` always uses [`PROTOCOL`].
struct Protocol<'a> {
    cases: &'a [(&'a str, usize, usize)],
    seeds: &'a [u64],
    blocks: usize,
    samples: usize,
    target: f64,
    algorithms: [&'a str; 2],
}

const PROTOCOL: Protocol<'static> = Protocol {
    cases: &[
        ("chain", 16, 2),
        ("chain", 32, 2),
        ("chain", 32, 4),
        ("chain", 64, 4),
        ("binary", 15, 2),
        ("star", 64, 1),
    ],
    seeds: &[1, 2, 3],
    blocks: 3,
    samples: 2048,
    target: 1e-8,
    algorithms: ["rsi", "treeaci"],
};

fn values(tree: &Tree, sites: &[DynIndex], points: &[usize]) -> Result<Vec<f64>, Error> {
    let mut evaluator = TreeTNCachedEvaluator::new(tree, sites, CachedEvaluatorOptions::default())?;
    let mut out = Vec::with_capacity(points.len() / sites.len());
    for chunk in points.chunks(sites.len() * 512) {
        for x in evaluator.evaluate_batched(ColMajorArrayRef::new(
            chunk,
            &[sites.len(), chunk.len() / sites.len()],
        )?)? {
            out.push(x.real());
        }
    }
    Ok(out)
}

fn relative_error(actual: &[f64], expected: &[f64]) -> Option<f64> {
    if actual.len() != expected.len()
        || actual.is_empty()
        || actual.iter().chain(expected).any(|x| !x.is_finite())
    {
        return None;
    }
    let scale = expected.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return if actual.iter().all(|&x| x == 0.0) {
            Some(0.0)
        } else {
            None
        };
    }
    let numerator = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| ((a - b) / scale).powi(2))
        .sum::<f64>();
    let denominator = expected.iter().map(|b| (b / scale).powi(2)).sum::<f64>();
    let value = (numerator / denominator).sqrt();
    value.is_finite().then_some(value)
}

fn run(
    algorithm: &str,
    inputs: &[Tree],
    rank: usize,
    seed: u64,
    root: usize,
) -> Result<(Tree, f64, serde_json::Value), Box<dyn std::error::Error>> {
    match algorithm {
        "rsi" => {
            let options = TreeRsiOptions {
                max_bond_dim: Some(rank),
                seed,
                root: Some(root),
                rel_tol: 1e-12,
                ..Default::default()
            };
            let start = Instant::now();
            let result = std::hint::black_box(hadamard_many::<f64, _>(
                std::hint::black_box(inputs),
                &options,
            ))?;
            let elapsed = start.elapsed().as_secs_f64();
            let diagnostics = json!({
                "sketch_dim":result.diagnostics.sketch_dim,
                "local_pivots":result.diagnostics.edges.iter().map(|e| e.relative_pivot).collect::<Vec<_>>()
            });
            Ok((result.tree, elapsed, diagnostics))
        }
        "treeaci" => {
            let options = TreeAciOptions {
                max_bond_dim: Some(rank),
                rng_seed: seed,
                root: Some(root),
                tolerance: 1e-12,
                scale_tolerance: true,
                ..Default::default()
            };
            let start = Instant::now();
            let result = std::hint::black_box(tensor4all_treeaci::hadamard_many::<f64, _>(
                std::hint::black_box(inputs),
                &options,
            ))?;
            let elapsed = start.elapsed().as_secs_f64();
            let diagnostics = json!({
                "termination":format!("{:?}",result.termination),
                "sweep_ranks":result.max_ranks,"local_errors":result.max_errors,
                "global_pivots":result.global_pivots_found
            });
            Ok((result.tree, elapsed, diagnostics))
        }
        _ => Err("unknown algorithm".into()),
    }
}

fn main() -> Result<(), Error> {
    let path = std::env::args()
        .nth(1)
        .ok_or("supply a new output path; use benchmarks/tree-rsi/run.py")?;
    experiment(
        &PROTOCOL,
        &path,
        &std::env::var("RSI_SOURCE_SHA256")?,
        &std::env::var("RSI_BINARY_SHA256")?,
    )
}

fn experiment(
    protocol: &Protocol<'_>,
    path: &str,
    source_sha256: &str,
    binary_sha256: &str,
) -> Result<(), Error> {
    let Protocol {
        cases,
        seeds,
        blocks: timing_blocks,
        samples,
        target,
        algorithms,
    } = *protocol;
    if let Some(parent) = Path::new(&path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    let mut write = |value: serde_json::Value| -> std::io::Result<()> {
        writeln!(file, "{value}")?;
        file.flush()
    };
    write(
        json!({"type":"protocol","cases":cases,"seeds":seeds,"blocks":timing_blocks,"samples":samples,
        "sample_seed":20260929u64,"target":target,"metric":"sampled relative L2 of amplitude differences",
        "timed_region":"public product call on prebuilt inputs; includes API RNG initialization",
        "selection":"one fixed setting, no grid search; all seeds must meet target before timing a case",
        "source_sha256":source_sha256,"binary_sha256":binary_sha256}),
    )?;
    let mut accepted = true;
    for &(topology, n, bond) in cases {
        let name = format!("{topology}-n{n}-chi{bond}");
        let sites = (0..n).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
        let edges = (1..n)
            .map(|i| {
                (
                    match topology {
                        "chain" => i - 1,
                        "binary" => (i - 1) / 2,
                        _ => 0,
                    },
                    i,
                )
            })
            .collect::<Vec<_>>();
        let inputs = [11, 29]
            .into_iter()
            .map(|seed| {
                support::random_tree(&sites, &edges, bond, seed, |x, _| {
                    (0.8 + 0.2 * x) / (bond as f64).sqrt()
                })
            })
            .collect::<Vec<_>>();
        let mut rng = ChaCha8Rng::seed_from_u64(20260929);
        let points = (0..n * samples)
            .map(|_| rng.random_range(0..2usize))
            .collect::<Vec<_>>();
        // Input truth is evaluated once, outside every warmup and timed call.
        let a = values(&inputs[0], &sites, &points)?;
        let b = values(&inputs[1], &sites, &points)?;
        let exact = a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<_>>();
        let mut case_ok = true;
        for phase in ["accuracy", "timing"] {
            if phase == "timing" && !case_ok {
                continue;
            }
            let blocks = if phase == "accuracy" {
                1
            } else {
                timing_blocks
            };
            for block in 0..blocks {
                for &seed in seeds {
                    let [first, second] = algorithms;
                    let order = if (block + seed as usize).is_multiple_of(2) {
                        [first, second]
                    } else {
                        [second, first]
                    };
                    for algorithm in order {
                        match run(algorithm, &inputs, bond * bond, seed, n - 1).and_then(
                            |(tree, seconds, mut diagnostics)| {
                                diagnostics["edge_ranks"] = json!(diagnostics::edge_ranks(&tree)?);
                                let error =
                                    relative_error(&values(&tree, &sites, &points)?, &exact);
                                Ok((seconds, error, diagnostics))
                            },
                        ) {
                            Ok((seconds, error, diagnostics)) => {
                                case_ok &= error.is_some_and(|e| e <= target);
                                write(
                                    json!({"type":"observation","case":name,"phase":phase,"block":block,
                                    "seed":seed,"algorithm":algorithm,"seconds":seconds,"error":error,"cap":bond*bond,"diagnostics":diagnostics}),
                                )?;
                            }
                            Err(error) => {
                                case_ok = false;
                                write(
                                    json!({"type":"failure","case":name,"phase":phase,"block":block,
                                    "seed":seed,"algorithm":algorithm,"failure":error.to_string()}),
                                )?;
                            }
                        }
                    }
                }
            }
        }
        accepted &= case_ok;
    }
    if !accepted {
        return Err(
            "fixed-budget accuracy target not met; this alone does not establish an implementation defect".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
