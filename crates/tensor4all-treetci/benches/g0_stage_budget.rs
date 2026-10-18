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
//! RAYON_NUM_THREADS=1  cargo bench -p tensor4all-treetci --bench g0_stage_budget
//! RAYON_NUM_THREADS=4  ...
//! RAYON_NUM_THREADS=64 ...
//! ```
//!
//! Each run prints the observed worker count, the per-repetition stage time, the peak number of process
//! threads while the stage ran, the process's CPU utilisation across the call, and the run's ranks and final
//! error, so the comparison rests on a measured schedule rather than on the environment variable alone.
//! `T4A_G0_GLOBAL_PIVOTS=0` runs the stage with the global pivot search disabled - an ablation that removes
//! the search and any pivots it would have accepted, not a measurement of the search alone - while
//! `T4A_G0_SITES` and `T4A_G0_REPETITIONS` size the run. The options pin the seed, and each repetition prints
//! the complete ranks and error histories and the termination, so an ablation can be checked for having done
//! the same sweep work.
//!
//! Linux-only and x86-64 `USER_HZ = 100` are assumed for the CPU figures; they are reported as raw tick deltas
//! next to the conversion, and as "below tick resolution" when the delta cannot support a ratio. The thread
//! count is a sampled maximum over an interval that approximately covers the call, not an exact stage peak.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tenferro_cpu::CpuBackend;
use tensor4all_tensorbackend::CpuExecutionContext;
use tensor4all_treetci::{
    crossinterpolate2, crossinterpolate2_in, DefaultProposer, GlobalIndexBatch, TreeTciGraph,
    TreeTciOptions,
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
    // Off by default is what the downstream report sees; running both isolates the global pivot search,
    // which the report names as the step after every non-final sweep.
    let global_pivots = std::env::var("T4A_G0_GLOBAL_PIVOTS")
        .map(|value| value != "0")
        .unwrap_or(true);
    // `0` runs the legacy entry (no caller context). A positive value gives the factorization its own
    // context with that many workers, which is the point of the two-budget protocol: the process-global
    // context stays at whatever `RAYON_NUM_THREADS` fixed at initialization, and only the caller's budget
    // moves. Built outside the timed region.
    let caller_threads: usize = std::env::var("T4A_G0_CALLER_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let caller_context = (caller_threads > 0).then(|| {
        let backend = CpuBackend::with_threads(caller_threads).expect("caller CPU backend");
        let context = CpuExecutionContext::from_backend(backend);
        assert_eq!(
            context.with_backend(|backend| backend.num_threads()),
            caller_threads,
            "the caller context's worker count"
        );
        context
    });

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
        // Pinned so the two configurations differ in the pivot search and not in the random stream.
        seed: Some(20_261_011),
        tolerance: 1e-8,
        max_iter: 30,
        max_bond_dim: Some(16),
        scale_tolerance: true,
        enable_global_pivots: global_pivots,
        ..TreeTciOptions::default()
    };

    for repetition in 0..repetitions {
        // Sample the process while the stage runs, so "more workers hurt" can be read next to how many
        // threads actually existed and how much CPU the stage used.
        let sampling = Arc::new(AtomicBool::new(true));
        let peak_threads = Arc::new(AtomicUsize::new(0));
        let sampler = {
            let sampling = Arc::clone(&sampling);
            let peak_threads = Arc::clone(&peak_threads);
            std::thread::spawn(move || {
                while sampling.load(Ordering::Relaxed) {
                    if let Some(threads) = read_threads() {
                        peak_threads.fetch_max(threads, Ordering::Relaxed);
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        };
        let cpu_before = read_cpu_ticks();
        let start = Instant::now();
        let result = match &caller_context {
            Some(context) => crossinterpolate2_in::<f64, _, _>(
                oracle,
                local_dims.clone(),
                graph.clone(),
                pivots.clone(),
                options.clone(),
                None,
                &DefaultProposer,
                context,
            ),
            None => crossinterpolate2::<f64, _, _>(
                oracle,
                local_dims.clone(),
                graph.clone(),
                pivots.clone(),
                options.clone(),
                None,
                &DefaultProposer,
            ),
        }
        .expect("a G0-shaped TreeTCI run");
        let elapsed = start.elapsed();
        // Read the process's own CPU time as close to the call as possible: the sampler's teardown must not
        // land inside the numerator while staying outside the denominator.
        let cpu_after = read_cpu_ticks();
        sampling.store(false, Ordering::Relaxed);
        let _ = sampler.join();
        let ticks_per_second = 100usize; // `USER_HZ` on Linux x86-64; see the module docs.
        let (ticks, utilisation) = match (cpu_before, cpu_after) {
            (Some(before), Some(after)) => {
                let delta = after.saturating_sub(before);
                // A ratio needs enough ticks to mean anything; a 10 ms quantum cannot support one for a
                // short run.
                let ratio = if delta >= 2 && elapsed.as_secs_f64() > 0.0 {
                    format!(
                        "{:.2}",
                        (delta as f64 / ticks_per_second as f64) / elapsed.as_secs_f64()
                    )
                } else {
                    "below_tick_resolution".to_string()
                };
                (delta.to_string(), ratio)
            }
            _ => ("unavailable".to_string(), "unavailable".to_string()),
        };
        println!(
            "g0_stage_budget: sites={sites} local_dim={local_dim} global_pivots={global_pivots} \
             repetitions={repetitions} workers_env={workers_env} workers_observed={workers_observed} \
             caller_threads={caller_threads} repetition={repetition} elapsed={elapsed:?} \
             peak_process_threads={} cpu_ticks={ticks} \
             cpu_ticks_per_second={ticks_per_second} cpu_utilisation={utilisation} termination={:?} \
             ranks={:?} errors={:?}",
            peak_threads.load(Ordering::Relaxed),
            result.termination,
            result.ranks,
            result.errors
        );
    }
}

/// The process's thread count, sampled while the stage runs.
fn read_threads() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("Threads:"))
        .and_then(|value| value.trim().parse().ok())
}

/// This process's consumed CPU time in clock ticks (user + system), from `/proc/self/stat`.
fn read_cpu_ticks() -> Option<usize> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let after_comm = stat.rsplit_once(") ")?.1;
    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    // Field 14 and 15 of the whole line are utime and stime, i.e. indices 11 and 12 after the comm field.
    let user: usize = fields.get(11)?.parse().ok()?;
    let system: usize = fields.get(12)?.parse().ok()?;
    Some(user + system)
}
