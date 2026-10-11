//! Diagnostic measurement for #670 / #859 F3: how the G0-shaped TreeTCI path scales with the
//! process-global worker count.
//!
//! The downstream report (#670) is a *symptom*: the G0 stage (`crossinterpolate2` with global pivots on a
//! small tree) took longer on a faster machine, and the report explicitly says the dominant cost is not
//! established - nested parallelism, over-parallelization of small contractions, or missing reuse. This
//! binary measures the first question the report leaves open: does giving the path more workers help it, or
//! does it hurt it? It is not a benchmark of a route; it is a controlled schedule, run once per worker count.
//!
//! Run it once per worker count, because the process-global context takes its count from
//! `RAYON_NUM_THREADS` when it first initializes:
//!
//! ```text
//! RAYON_NUM_THREADS=1  cargo run --release -p tensor4all-treetci --bench g0_stage_budget
//! RAYON_NUM_THREADS=4  ...
//! RAYON_NUM_THREADS=64 ...
//! ```
//!
//! Each run prints the observed worker count, the per-repetition stage time and the run's ranks and final
//! error, so the comparison rests on a measured schedule rather than on the environment variable alone.

use std::time::Instant;

use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciGraph, TreeTciOptions,
};

/// The G0-shaped workload: a small tree, a cheap local integrator, global pivots on.
fn main() {
    let sites: usize = std::env::var("T4A_G0_SITES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let repetitions: usize = std::env::var("T4A_G0_REPETITIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);
    let local_dim = 4usize;

    let workers_env = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<unset>".to_string());
    let workers_observed = tensor4all_tensorbackend::default_cpu_execution_context()
        .with_backend(|backend| backend.num_threads());

    // A per-site table, summed over the batch: the shape of a tensor-product integrand.
    let table: Vec<f64> = (0..local_dim)
        .map(|index| (index as f64) * 0.25 - 0.5)
        .collect();
    let oracle = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let mut total = 1.0;
            for site in 0..sites {
                total += table[batch.get(site, point).expect("a site within the batch")];
            }
            values.push(total);
        }
        Ok(values)
    };

    let graph = TreeTciGraph::linear_chain(sites).expect("a linear chain graph");
    let local_dims = vec![local_dim; sites];
    let pivots = vec![vec![0usize; sites]];
    let options = TreeTciOptions {
        tolerance: 1e-8,
        max_iter: 30,
        max_bond_dim: Some(16),
        normalize_error: true,
        enable_global_pivots: true,
        ..TreeTciOptions::default()
    };

    for repetition in 0..repetitions {
        let start = Instant::now();
        let result = crossinterpolate2::<f64, _, _>(
            oracle,
            local_dims.clone(),
            graph.clone(),
            pivots.clone(),
            options.clone(),
            None,
            &DefaultProposer,
        )
        .expect("a G0-shaped TreeTCI run");
        let elapsed = start.elapsed();
        println!(
            "g0_stage_budget: sites={sites} local_dim={local_dim} repetitions={repetitions} \
             workers_env={workers_env} workers_observed={workers_observed} repetition={repetition} \
             elapsed={elapsed:?} ranks={:?} final_error={:?}",
            result.ranks,
            result.errors.last()
        );
    }
}
