//! F3 step 3d: the caller's execution context reaches **both** materializations on the TCI path.
//!
//! The factorization and the materialization solve must run in the caller's context. Value parity does not
//! show that (it holds with the global route too), so these tests hold the process-global backend while a run
//! completes: a run that resolved either materialization through the global session would block instead.

use std::time::Duration;

use tensor4all_tensorbackend::CpuBackend;
use tensor4all_tensorbackend::{with_default_backend, CpuExecutionContext};
use tensor4all_treetci::{
    crossinterpolate2, crossinterpolate2_in, DefaultProposer, GlobalIndexBatch, TreeTciGraph,
    TreeTciOptions, TreeTciRunResult,
};

fn context_with_workers(workers: usize) -> CpuExecutionContext {
    CpuExecutionContext::from_backend(CpuBackend::with_threads(workers).expect("CPU backend"))
}

fn options() -> TreeTciOptions {
    TreeTciOptions {
        seed: Some(20_261_011),
        tolerance: 1e-8,
        max_iter: 8,
        max_bond_dim: Some(6),
        scale_tolerance: true,
        enable_global_pivots: true,
        nsearch: 2,
        max_nglobal_pivot: 2,
        ..TreeTciOptions::default()
    }
}

fn oracle(
    sites: usize,
) -> impl Fn(GlobalIndexBatch<'_>) -> anyhow::Result<Vec<f64>> + Send + 'static {
    move |batch: GlobalIndexBatch<'_>| {
        Ok((0..batch.n_points())
            .map(|point| {
                (0..sites).fold(1.0, |total, site| {
                    total + batch.get(site, point).expect("site in batch") as f64 * 0.5 + 1.0
                })
            })
            .collect())
    }
}

fn run_in_context(context: &CpuExecutionContext) -> TreeTciRunResult {
    let sites = 6;
    let graph = TreeTciGraph::linear_chain(sites).expect("graph");
    let result = crossinterpolate2_in::<f64, _, _>(
        oracle(sites),
        vec![3; sites],
        graph,
        vec![vec![0usize; sites]],
        options(),
        None,
        &DefaultProposer,
        context,
    )
    .expect("context run");
    // The global search must actually have been reached and accepted pivots:
    // that is what puts the search-time materialization on this run's path.
    assert!(
        result.ranks.iter().all(|rank| *rank > 0),
        "a nonzero run has nonzero ranks: {:?}",
        result.ranks
    );
    assert_eq!(
        result.ranks.len(),
        result.errors.len(),
        "one rank and one error per completed iteration"
    );
    result
}

/// The dense maximum of a run's tensor train. Readout uses the implicit route, so it runs after
/// the process-global backend is released.
fn dense_maxabs(result: &TreeTciRunResult) -> f64 {
    result
        .treetn
        .to_dense()
        .expect("dense")
        .maxabs()
        .expect("maxabs")
}

#[test]
fn a_run_completes_with_the_process_global_backend_held() {
    let context = context_with_workers(1);

    // Warm the process-global eager runtime: physical tensor publication adopts into it, and its cold
    // initialization takes the global context's backend lock, which this test holds.
    let warm = dense_maxabs(&run_in_context(&context));
    assert!(warm > 0.0);

    let reference = crossinterpolate2::<f64, _, _>(
        oracle(6),
        vec![3; 6],
        TreeTciGraph::linear_chain(6).expect("graph"),
        vec![vec![0usize; 6]],
        options(),
        None,
        &DefaultProposer,
    )
    .expect("plain run");

    // The worker thread lets a wrong route fail this test instead of hanging the suite; the panic releases
    // the lock. Both materializations (search-time and final) are inside the run.
    let scoped = with_default_backend(|_global| {
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let context = context_with_workers(1);
            let result = run_in_context(&context);
            let _ = done_tx.send(());
            result
        });
        match done_rx.recv_timeout(Duration::from_secs(180)) {
            Ok(()) => worker.join().expect("worker thread"),
            Err(_) => panic!(
                "the run did not finish while the process-global backend was held: a materialization on the \
                 path still resolved its solve through the global session"
            ),
        }
    });
    let scoped_max = dense_maxabs(&scoped);

    let plain_max = reference
        .treetn
        .to_dense()
        .expect("dense")
        .maxabs()
        .unwrap();
    assert!(
        (scoped_max - plain_max).abs() <= 1e-6 * plain_max.abs().max(1.0),
        "scoped {scoped_max} != plain {plain_max}"
    );
}
