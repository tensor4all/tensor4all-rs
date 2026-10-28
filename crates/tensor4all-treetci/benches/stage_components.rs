//! Component breakdown for #670 / #859 F3: where the G0 stage's time actually goes.
//!
//! The other bench (`g0_stage_budget`) measures the *schedule*: the stage's time and the process's threads and
//! CPU ticks per worker count, plus an ablation of the global pivot search. This one measures *components* on a
//! representative state, using only the public API, so that "the search dominates" can be split into its own
//! materialization and its readout, and so that the caller's own oracle cost is separated from the machinery:
//!
//! ```text
//! oracle      the caller's evaluator, called directly with a full batch, reported as ns per point
//! materialize `to_treetn` on the state: its oracle points are counted, so machinery = total - oracle points
//! search      `find_global_pivots` on the same state: it materializes once per search, so
//!             readout = total - its oracle points - the measured materialization machinery
//! run         a full `crossinterpolate2` for the end-to-end reference, with and without the global search
//! ```
//!
//! The representative state is grown with the search's own accepted pivots (a few unmeasured rounds) rather
//! than the fixture's single initial pivot, so the components are measured at a state the stage really reaches.
//! Reported per repetition: elapsed, evaluated points, oracle share and machinery share. These are differences
//! of measured quantities, not internal instrumentation: the oracle share assumes the caller's per-point cost is
//! the same in every stage, and the readout figure absorbs anything else the search does.
//!
//! Run once per worker count, because the process-global context takes its count from `RAYON_NUM_THREADS` when
//! it first initializes. `T4A_G0_SITES` and `T4A_G0_REPETITIONS` size the measurement, and the options pin the
//! seed. Peak threads and CPU utilisation stay in `g0_stage_budget`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tensor4all_treetci::{
    crossinterpolate2, find_global_pivots, optimize_default, to_treetn, DefaultProposer,
    GlobalIndexBatch, TreeTCI2, TreeTciGraph, TreeTciOptions,
};

/// The number of points one oracle call is charged for in the direct measurement.
const ORACLE_BATCH_POINTS: usize = 65_536;
/// Unmeasured sweeps that bring the state to the shape a run reaches.
const STATE_WARMUP_ITERATIONS: usize = 4;

fn main() {
    let sites: usize = env_usize("T4A_G0_SITES", 8);
    let repetitions: usize = env_usize("T4A_G0_REPETITIONS", 3);
    let local_dim = 4usize;

    let workers_env = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<unset>".to_string());
    let workers_observed = tensor4all_tensorbackend::default_cpu_execution_context()
        .with_backend(|backend| backend.num_threads());

    let table: Vec<f64> = (0..local_dim)
        .map(|index| (index as f64) * 0.25 - 0.5)
        .collect();
    // Captures only a shared table and two integers, so the closure is `Copy`
    // and every measurement can hand its own copy to the stage.
    let table = &table;
    let oracle = move |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let mut total = 1.0;
            for site in 0..sites {
                let index = batch.get(site, point).expect("site in batch");
                total += table[index] * table[site % local_dim];
            }
            values.push(total);
        }
        Ok(values)
    };

    let graph = TreeTciGraph::linear_chain(sites).expect("graph");
    let mut state = TreeTCI2::<f64>::new(vec![local_dim; sites], graph.clone()).expect("state");
    state
        .add_global_pivots(&[vec![0usize; sites]])
        .expect("initial pivot");
    state.flush_pivot_errors();

    // Bring the state to the shape a run reaches, with the stage's own optimizer.
    // Not measured: this is the fixture, and the components below are measured on
    // the state it produces.
    let warmup_options = TreeTciOptions {
        seed: Some(20_261_011),
        tolerance: 1e-8,
        max_iter: STATE_WARMUP_ITERATIONS,
        max_bond_dim: Some(16),
        scale_tolerance: true,
        enable_global_pivots: true,
        ..TreeTciOptions::default()
    };
    let warmup = optimize_default(&mut state, oracle, &warmup_options).expect("state warm-up");
    // The search budget a run uses, for the per-search component below.
    let search_options = TreeTciOptions {
        seed: Some(20_261_011),
        tolerance: 1e-8,
        max_iter: 30,
        max_bond_dim: Some(16),
        scale_tolerance: true,
        enable_global_pivots: true,
        ..TreeTciOptions::default()
    };

    // The oracle's own cost, called directly with a full batch.
    let indices = (0..sites * ORACLE_BATCH_POINTS)
        .map(|slot| (slot % local_dim + slot / local_dim) % local_dim)
        .collect::<Vec<_>>();
    let batch = GlobalIndexBatch::new(&indices, sites, ORACLE_BATCH_POINTS).expect("batch");
    let oracle_started = Instant::now();
    let values = oracle(batch).expect("oracle");
    let oracle_elapsed = oracle_started.elapsed();
    assert_eq!(values.len(), ORACLE_BATCH_POINTS);
    let ns_per_point = oracle_elapsed.as_secs_f64() * 1e9 / ORACLE_BATCH_POINTS as f64;

    println!(
        "stage_components: sites={sites} local_dim={local_dim} workers_env={workers_env} \
         workers_observed={workers_observed} repetitions={repetitions} \
         warmup_iterations={} warmup_ranks={:?} state_max_bond_dim={} oracle_ns_per_point={ns_per_point:.1} \
         search_nsearch={} search_max_pivots={}",
        warmup.ranks.len(),
        warmup.ranks,
        state.max_bond_dim(),
        search_options.nsearch,
        search_options.max_nglobal_pivot
    );

    for repetition in 0..repetitions {
        // --- materialization on the representative state ---
        let evaluated = AtomicUsize::new(0);
        let calls = AtomicUsize::new(0);
        let counting = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
            evaluated.fetch_add(batch.n_points(), Ordering::Relaxed);
            calls.fetch_add(1, Ordering::Relaxed);
            oracle(batch)
        };
        let started = Instant::now();
        let tree = to_treetn(&state, counting, None).expect("materialization");
        let materialize_elapsed = started.elapsed();
        let materialize_points = evaluated.load(Ordering::Relaxed);
        let materialize_calls = calls.load(Ordering::Relaxed);
        let materialize_oracle_ms = materialize_points as f64 * ns_per_point / 1e6;
        let materialize_total_ms = materialize_elapsed.as_secs_f64() * 1e3;
        let materialize_machinery_ms = materialize_total_ms - materialize_oracle_ms;
        println!(
            "stage_components: repetition={repetition} component=materialize elapsed={materialize_total_ms:.3}ms \
             points={materialize_points} calls={materialize_calls} oracle_ms={materialize_oracle_ms:.3} \
             machinery_ms={materialize_machinery_ms:.3} maxabs={:.6}",
            tree.to_dense().expect("dense").maxabs().expect("maxabs")
        );

        // --- the global pivot search on the same state ---
        let evaluated = AtomicUsize::new(0);
        let calls = AtomicUsize::new(0);
        let counting = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
            evaluated.fetch_add(batch.n_points(), Ordering::Relaxed);
            calls.fetch_add(1, Ordering::Relaxed);
            oracle(batch)
        };
        let started = Instant::now();
        // The run's own search budget, so the per-search figure corresponds to
        // the searches the end-to-end stage performs.
        let pivots = find_global_pivots(
            &state,
            counting,
            search_options.nsearch,
            &search_options.global_search_starts,
            search_options.max_nglobal_pivot,
            search_options.tol_margin_global_search,
            search_options.tolerance,
            20_261_011,
        )
        .expect("search");
        let search_elapsed = started.elapsed();
        let search_points = evaluated.load(Ordering::Relaxed);
        let search_calls = calls.load(Ordering::Relaxed);
        let search_oracle_ms = search_points as f64 * ns_per_point / 1e6;
        let search_total_ms = search_elapsed.as_secs_f64() * 1e3;
        // The search materializes the state once before walking candidates.
        let search_readout_ms = search_total_ms - search_oracle_ms - materialize_machinery_ms;
        println!(
            "stage_components: repetition={repetition} component=search elapsed={search_total_ms:.3}ms \
             points={search_points} calls={search_calls} oracle_ms={search_oracle_ms:.3} \
             accepted={} readout_ms={search_readout_ms:.3}",
            pivots.len()
        );

        // --- the end-to-end stage, with and without the search ---
        for global_pivots in [true, false] {
            let evaluated = AtomicUsize::new(0);
            let counting = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
                evaluated.fetch_add(batch.n_points(), Ordering::Relaxed);
                oracle(batch)
            };
            let options = TreeTciOptions {
                seed: Some(20_261_011),
                tolerance: 1e-8,
                max_iter: 30,
                max_bond_dim: Some(16),
                scale_tolerance: true,
                enable_global_pivots: global_pivots,
                ..TreeTciOptions::default()
            };
            let started = Instant::now();
            let result = crossinterpolate2::<f64, _, _>(
                counting,
                vec![local_dim; sites],
                graph.clone(),
                vec![vec![0usize; sites]],
                options,
                None,
                &DefaultProposer,
            )
            .expect("run");
            let elapsed = started.elapsed();
            let run_points = evaluated.load(Ordering::Relaxed);
            let total_ms = elapsed.as_secs_f64() * 1e3;
            println!(
                "stage_components: repetition={repetition} component=run global_pivots={global_pivots} \
                 elapsed={total_ms:.3}ms points={run_points} oracle_ms={:.3} machinery_ms={:.3} \
                 ranks={:?} termination={:?}",
                run_points as f64 * ns_per_point / 1e6,
                total_ms - run_points as f64 * ns_per_point / 1e6,
                result.ranks,
                result.termination
            );
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
