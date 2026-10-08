// Paired end-to-end target memo experiment. See the dated protocol in benchmarks/.
use anyhow::Result;
use std::hint::black_box;
use std::time::Instant;
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciEdge, TreeTciGraph, TreeTciOptions,
};

fn main() -> Result<()> {
    let memo = std::env::args().nth(1).as_deref() == Some("memo");
    for expensive in [false, true] {
        for arms in [false, true] {
            for size in [8, 16] {
                let n_sites = if arms { 1 + 3 * (size / 2) } else { size };
                let mut edges = Vec::with_capacity(n_sites - 1);
                if arms {
                    for arm in 0..3 {
                        let start = 1 + arm * (size / 2);
                        edges.push(TreeTciEdge::new(0, start));
                        for site in start..start + size / 2 - 1 {
                            edges.push(TreeTciEdge::new(site, site + 1));
                        }
                    }
                } else {
                    edges.extend((0..n_sites - 1).map(|site| TreeTciEdge::new(site, site + 1)));
                }
                let dims = if arms {
                    let mut dims = vec![2; n_sites];
                    dims[0] = 1;
                    dims
                } else {
                    vec![2; n_sites]
                };
                let started = Instant::now();
                let result = crossinterpolate2::<f64, _, _>(
                    |batch: GlobalIndexBatch<'_>| {
                        Ok(batch
                            .data()
                            .chunks_exact(batch.n_sites())
                            .map(|point| {
                                let x = point
                                    .iter()
                                    .enumerate()
                                    .map(|(i, &b)| b as f64 / (i + 1) as f64)
                                    .sum::<f64>();
                                if expensive {
                                    // Fixed synthetic oracle cost, independent of grouping and memo.
                                    // Black-box intermediates keep the compiler from eliding the work.
                                    let mut work = x;
                                    for _ in 0..1024 {
                                        work = black_box(work.sin() + 0.2);
                                    }
                                    black_box(work);
                                }
                                2.0 + x.cos()
                            })
                            .collect())
                    },
                    dims,
                    TreeTciGraph::new(n_sites, &edges)?,
                    vec![vec![0; n_sites]],
                    TreeTciOptions {
                        seed: Some(802),
                        evaluation_cache_bytes: memo.then_some(256 * 1024 * 1024),
                        ..Default::default()
                    },
                    None,
                    &DefaultProposer,
                )?;
                let seconds = started.elapsed().as_secs_f64();
                let stats = result.evaluation;
                let samples = [vec![0; n_sites], vec![1; n_sites]];
                // Evaluate the output outside the timed boundary using the shared tree evaluator.
                let indices = (0..n_sites)
                    .map(|site| {
                        let node = result
                            .treetn
                            .node_index(&site)
                            .ok_or_else(|| anyhow::anyhow!("missing site"))?;
                        let tensor = result
                            .treetn
                            .tensor(node)
                            .ok_or_else(|| anyhow::anyhow!("missing tensor"))?;
                        Ok(tensor.indices()[0].clone())
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut evaluator = tensor4all_treetn::TreeTNCachedEvaluator::<usize>::new(
                    &result.treetn,
                    &indices,
                    Default::default(),
                )?;
                let mut samples = samples.concat();
                if arms {
                    samples[n_sites] = 0;
                }
                let values = evaluator.evaluate_batched(tensor4all_core::ColMajorArrayRef::new(
                    &samples,
                    &[n_sites, 2],
                )?)?;
                let bits = values
                    .iter()
                    .map(|v| v.real().to_bits())
                    .collect::<Vec<_>>();
                for (point, value) in samples.chunks_exact(n_sites).zip(values) {
                    let x = point
                        .iter()
                        .enumerate()
                        .map(|(i, &b)| b as f64 / (i + 1) as f64)
                        .sum::<f64>();
                    anyhow::ensure!(
                        (value.real() - (2.0 + x.cos())).abs() < 1e-7,
                        "sample error"
                    );
                }
                println!("case={}_{}_{} seconds={seconds:.9} requested={} evaluated={} entries={} bytes={} drops={} ranks={:?} errors={:?} termination={:?} samples={:?}",
                    if expensive {"expensive"} else {"cheap"}, if arms {"branch"} else {"chain"}, size,
                    stats.requested_points, stats.evaluated_points, stats.cached_entries, stats.retained_bytes, stats.dropped_inserts,
                    result.ranks, result.errors.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), result.termination, bits);
            }
        }
    }
    Ok(())
}
