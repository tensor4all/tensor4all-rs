//! The iteration observer: what it reports, and that it changes nothing.

use anyhow::Result;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
use tensor4all_treetci::{
    crossinterpolate2, crossinterpolate2_in, crossinterpolate2_with_rng,
    crossinterpolate2_with_rng_in, optimize_with_proposer, optimize_with_proposer_in,
    optimize_with_proposer_with_rng, to_treetn, AllEdges, DefaultProposer, GlobalIndexBatch,
    TreeTCI2, TreeTciEvaluationStats, TreeTciGraph, TreeTciIteration, TreeTciObserver,
    TreeTciOptimizationResult, TreeTciOptions,
};

const N_SITES: usize = 5;

/// `1 / (1 + sum_i (i + 1) x_i)` on binary sites: smooth, not separable, so the
/// bond dimension grows over several iterations.
fn target(batch: GlobalIndexBatch<'_>) -> Result<Vec<f64>> {
    Ok((0..batch.n_points())
        .map(|point| {
            let sum: usize = (0..N_SITES)
                .map(|site| (site + 1) * batch.get(site, point).unwrap())
                .sum();
            1.0 / (1.0 + sum as f64)
        })
        .collect())
}

fn options() -> TreeTciOptions {
    TreeTciOptions {
        tolerance: 1e-10,
        max_iter: 8,
        // `None` would seed the global pivot search from OS entropy.
        seed: Some(2026),
        ..Default::default()
    }
}

fn graph() -> TreeTciGraph {
    TreeTciGraph::linear_chain(N_SITES).unwrap()
}

/// One call of the observer, copied out of the borrowed [`TreeTciIteration`].
#[derive(Clone, Copy, Debug)]
struct Call {
    iteration: usize,
    max_bond_dim: usize,
    error: f64,
    evaluation: TreeTciEvaluationStats,
}

/// Records every iteration it is told about.
#[derive(Default)]
struct Recorder {
    calls: Vec<Call>,
}

impl TreeTciObserver for Recorder {
    fn iteration(&mut self, it: &TreeTciIteration<'_>) {
        self.calls.push(Call {
            iteration: it.iteration,
            max_bond_dim: it.max_bond_dim,
            error: it.error,
            evaluation: *it.evaluation,
        });
    }
}

impl Recorder {
    /// The calls are one per iteration and report the recorded ranks and errors bitwise.
    fn assert_matches(&self, ranks: &[usize], errors: &[f64]) {
        assert_eq!(self.calls.len(), ranks.len(), "one call per iteration");
        for (k, call) in self.calls.iter().enumerate() {
            assert_eq!(call.iteration, k);
            assert_eq!(call.max_bond_dim, ranks[k], "iteration {k}");
            assert_eq!(call.error.to_bits(), errors[k].to_bits(), "iteration {k}");
        }
    }
}

struct Run {
    result: TreeTciOptimizationResult,
    /// Points the target function was asked for, counted by the caller.
    evaluated_by_caller: usize,
    dense: Vec<f64>,
}

fn state() -> TreeTCI2<f64> {
    let mut state = TreeTCI2::<f64>::new(vec![2; N_SITES], graph()).unwrap();
    state.add_global_pivots(&[vec![0; N_SITES]]).unwrap();
    state.max_sample_value = 1.0;
    state
}

fn run_optimizer<O: TreeTciObserver>(observer: &mut O) -> Run {
    let mut state = state();
    let mut evaluated_by_caller = 0usize;
    let result = optimize_with_proposer(
        &mut state,
        |batch| {
            evaluated_by_caller += batch.n_points();
            target(batch)
        },
        &options(),
        &DefaultProposer,
        &AllEdges,
        observer,
    )
    .unwrap();
    let dense = to_treetn(&state, target, None)
        .unwrap()
        .to_dense()
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    Run {
        result,
        evaluated_by_caller,
        dense,
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn the_observer_reports_every_iteration_exactly_as_recorded() {
    let mut recorder = Recorder::default();
    let run = run_optimizer(&mut recorder);
    assert!(
        run.result.ranks.len() >= 2,
        "the target needs several iterations"
    );
    recorder.assert_matches(&run.result.ranks, &run.result.errors);
}

#[test]
fn evaluation_stats_never_decrease_and_stay_below_the_final_stats() {
    let mut recorder = Recorder::default();
    let run = run_optimizer(&mut recorder);
    let counters = |c: &Call| {
        let e = c.evaluation;
        [
            e.requested_points,
            e.evaluated_points,
            e.cache_hits,
            e.cache_misses,
        ]
    };
    for pair in recorder.calls.windows(2) {
        for (before, after) in counters(&pair[0]).iter().zip(counters(&pair[1])) {
            assert!(*before <= after, "{:?} then {:?}", pair[0], pair[1]);
        }
    }
    let last = recorder.calls.last().expect("at least one iteration");
    let end = run.result.evaluation;
    assert!(last.evaluation.requested_points <= end.requested_points);
    assert!(last.evaluation.evaluated_points <= end.evaluated_points);
    assert!(
        recorder.calls[0].evaluation.requested_points > 0,
        "the stats are live, not a default"
    );
}

#[test]
fn an_observer_changes_nothing() {
    let plain = run_optimizer(&mut ());
    let mut recorder = Recorder::default();
    let observed = run_optimizer(&mut recorder);
    assert_eq!(plain.result.ranks, observed.result.ranks);
    assert_eq!(bits(&plain.result.errors), bits(&observed.result.errors));
    assert_eq!(plain.result.termination, observed.result.termination);
    assert_eq!(plain.result.evaluation, observed.result.evaluation);
    // No extra evaluation of the target, counted on the caller's side.
    assert_eq!(plain.evaluated_by_caller, observed.evaluated_by_caller);
    assert_eq!(bits(&plain.dense), bits(&observed.dense));
}

#[test]
fn crossinterpolate2_with_an_observer_matches_one_without() {
    let interpolate = |observer: &mut Recorder| {
        crossinterpolate2::<f64, _, _, _>(
            target,
            vec![2; N_SITES],
            graph(),
            vec![vec![0; N_SITES]],
            options(),
            None,
            &DefaultProposer,
            observer,
        )
        .unwrap()
    };
    let dense = |r: &tensor4all_treetci::TreeTciRunResult| {
        r.treetn.to_dense().unwrap().to_vec::<f64>().unwrap()
    };
    let plain = crossinterpolate2::<f64, _, _, _>(
        target,
        vec![2; N_SITES],
        graph(),
        vec![vec![0; N_SITES]],
        options(),
        None,
        &DefaultProposer,
        &mut (),
    )
    .unwrap();
    let mut recorder = Recorder::default();
    let observed = interpolate(&mut recorder);
    assert_eq!(plain.ranks, observed.ranks);
    assert_eq!(bits(&plain.errors), bits(&observed.errors));
    assert_eq!(plain.termination, observed.termination);
    assert_eq!(plain.evaluation, observed.evaluation);
    assert_eq!(bits(&dense(&plain)), bits(&dense(&observed)));
    recorder.assert_matches(&observed.ranks, &observed.errors);
}

/// Every public entry that runs the loop takes an observer and reports to it.
#[test]
fn every_entry_point_reports_its_iterations() {
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap());
    let pivots = || vec![vec![0; N_SITES]];
    let dims = || vec![2; N_SITES];

    let mut recorder = Recorder::default();
    let run = crossinterpolate2_in::<f64, _, _, _>(
        target,
        dims(),
        graph(),
        pivots(),
        options(),
        None,
        &DefaultProposer,
        &mut recorder,
        &context,
    )
    .unwrap();
    recorder.assert_matches(&run.ranks, &run.errors);

    let mut recorder = Recorder::default();
    let run = crossinterpolate2_with_rng::<f64, _, _, _, _>(
        target,
        dims(),
        graph(),
        pivots(),
        options(),
        None,
        &DefaultProposer,
        &mut recorder,
        &mut ChaCha8Rng::seed_from_u64(3),
    )
    .unwrap();
    recorder.assert_matches(&run.ranks, &run.errors);

    let mut recorder = Recorder::default();
    let run = crossinterpolate2_with_rng_in::<f64, _, _, _, _>(
        target,
        dims(),
        graph(),
        pivots(),
        options(),
        None,
        &DefaultProposer,
        &mut recorder,
        &mut ChaCha8Rng::seed_from_u64(3),
        &context,
    )
    .unwrap();
    recorder.assert_matches(&run.ranks, &run.errors);

    let mut recorder = Recorder::default();
    let result = optimize_with_proposer_in(
        &mut state(),
        target,
        &options(),
        &DefaultProposer,
        &mut recorder,
        &context,
    )
    .unwrap();
    recorder.assert_matches(&result.ranks, &result.errors);

    let mut recorder = Recorder::default();
    let result = optimize_with_proposer_with_rng(
        &mut state(),
        target,
        &options(),
        &DefaultProposer,
        &mut recorder,
        &mut ChaCha8Rng::seed_from_u64(3),
    )
    .unwrap();
    recorder.assert_matches(&result.ranks, &result.errors);
}
