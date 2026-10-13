use std::sync::mpsc;
use std::time::Duration;

use super::*;
use tenferro::program::{CoreSemanticOp, ProgramInputSpec};
use tenferro::{DType, TensorSessionOpsExt, TraceContext};
use tenferro_ad::EagerTensor;

fn context() -> CpuExecutionContext {
    CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap())
}

#[test]
fn explicit_session_runs_a_concrete_operation() {
    let context = context();
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap();
    let result = context
        .with_session(|session| lhs.matmul(&rhs, session))
        .unwrap()
        .unwrap();

    assert_eq!(result.as_slice::<f64>().unwrap(), &[23.0, 34.0]);
}

#[test]
fn recursive_session_entry_fails_before_lock_and_restores_guard() {
    let context = context();
    let error = context
        .with_session(|_| context.with_session(|_| ()))
        .expect("the outer session entry")
        .expect_err("a recursive canonical session entry must be rejected");
    let CpuExecutionContextError::SessionEntry { source } = &error else {
        panic!("the rejection must be a typed session entry error: {error}");
    };
    assert!(matches!(
        source.downcast_ref::<tenferro_tensor::SessionEntryError>(),
        Some(tenferro_tensor::SessionEntryError::Reentered { .. })
    ));

    // The guard is restored, so an independent entry still works.
    assert_eq!(context.with_session(|_| 7usize).unwrap(), 7);
}

#[test]
fn explicit_plain_graph_and_eager_paths_share_only_the_supplied_backend() {
    let context = context();
    assert!(format!("{context:?}").contains("graph_initialized: false"));
    assert_eq!(context.with_backend(|backend| backend.num_threads()), 1);

    let mut trace = TraceContext::new();
    let input = trace
        .input(ProgramInputSpec::new(DType::F64, [2_usize.into()]))
        .unwrap();
    let output = trace.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
    let graph = trace.finish(&[output]).unwrap();
    let compiled = context.compile_graph(&graph).unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();
    let output = context.run_graph(&compiled, &[&input]).unwrap();
    assert_eq!(output[0].as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
    context.run_graph(&compiled, &[&input]).unwrap();
    let cached = context.graph_cache_stats().unwrap().prepared_plans;
    assert!(cached.entries > 0);
    assert!(cached.hits > 0);
    context.reset_graph_runtime().unwrap();
    assert_eq!(
        context.graph_cache_stats().unwrap().prepared_plans.entries,
        0
    );

    let eager = context.eager_runtime().unwrap();
    assert!(Arc::ptr_eq(&eager, &context.eager_runtime().unwrap()));
}

#[test]
fn separate_eager_contexts_reject_cross_context_operations() {
    let first = context().eager_runtime().unwrap();
    let second = context().eager_runtime().unwrap();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        first.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
        second,
    )
    .unwrap();
    let error = first
        .with_eager_session(|session| session.add(&a, &b))
        .expect_err("cross-context addition must be rejected");
    assert!(matches!(error, tenferro_ad::Error::ContextMismatch { .. }));
}

#[test]
fn caller_supplied_context_stays_caller_owned_after_context_drop() {
    let executor = Arc::new(CpuContext::with_threads(1).unwrap());
    let context =
        CpuExecutionContext::from_backend(CpuBackend::from_context(Arc::clone(&executor)));
    assert_eq!(context.with_backend(|backend| backend.num_threads()), 1);
    drop(context);
    assert_eq!(executor.num_threads(), 1);
}

#[test]
fn independent_contexts_do_not_share_a_backend_mutex() {
    let first = Arc::new(context());
    let second = Arc::new(context());
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    let handles = [first, second].map(|context| {
        let entered_tx = entered_tx.clone();
        let release_rx = Arc::clone(&release_rx);
        std::thread::spawn(move || {
            context.with_backend(|_| {
                entered_tx.send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
            });
        })
    });
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    release_tx.send(()).unwrap();
    release_tx.send(()).unwrap();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn session_from_a_rayon_worker_completes_without_waiting_on_the_context_pool() {
    use rayon::prelude::*;

    // The context owns a two-worker Rayon pool, and the enclosing pool hands
    // the worker waiting for a session install more of its own items. A
    // session entry that installs into the context pool therefore re-enters a
    // session on a thread whose execution is already active; the one-worker
    // case makes that sequence certain.
    let context = Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(2).unwrap(),
    ));
    for enclosing_workers in [1usize, 2] {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(enclosing_workers)
                .build()
                .unwrap(),
        );
        let (sender, receiver) = mpsc::channel();
        let context = Arc::clone(&context);
        std::thread::spawn(move || {
            let results = pool.install(|| {
                (0..2usize)
                    .into_par_iter()
                    .map(|_| {
                        let lhs = Tensor::from_vec_col_major(
                            vec![2, 2],
                            vec![1.0_f64, 2.0, 3.0, 4.0],
                        )
                        .unwrap();
                        let rhs =
                            Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap();
                        // tenferro-rs #2004 rejects a worker entry it cannot
                        // wait for with a typed `SessionEntry` error instead of
                        // blocking on the context pool. Either outcome finishes,
                        // which is what this test observes. The worker path reaches
                        // tenferro through the process-global arbiter, which the
                        // rest of this parallel suite also uses, so a rejection is
                        // retried until the unrelated holder releases.
                        let deadline = std::time::Instant::now()
                            + std::time::Duration::from_secs(30);
                        let outcome = loop {
                            let outcome = context.with_session(|session| {
                                lhs.matmul(&rhs, session)
                            });
                            let contended = matches!(
                                &outcome,
                                Err(CpuExecutionContextError::SessionEntry { source })
                                    if matches!(
                                        source.downcast_ref::<tenferro_tensor::SessionEntryError>(),
                                        Some(tenferro_tensor::SessionEntryError::Contended { .. })
                                    )
                            );
                            if contended && std::time::Instant::now() < deadline {
                                std::thread::sleep(std::time::Duration::from_millis(1));
                                continue;
                            }
                            break outcome;
                        };
                        match outcome {
                            Ok(Ok(product)) => {
                                Some(product.as_slice::<f64>().unwrap().to_vec())
                            }
                            Ok(Err(error)) => panic!("session operation failed: {error}"),
                            Err(error) => {
                                let CpuExecutionContextError::SessionEntry { source } = error
                                else {
                                    panic!("worker entry must fail with the typed session error");
                                };
                                let rejection = source
                                    .downcast_ref::<tenferro_tensor::SessionEntryError>()
                                    .expect("the rejection must preserve tenferro's typed cause");
                                assert!(
                                    matches!(
                                        rejection,
                                        tenferro_tensor::SessionEntryError::Contended { .. }
                                    ),
                                    "a worker cannot wait, so the rejection is contention: {rejection}"
                                );
                                None
                            }
                        }
                    })
                    .collect::<Vec<_>>()
            });
            let _ = sender.send(results);
        });

        let results = receiver.recv_timeout(Duration::from_secs(60)).expect(
            "a session entered from a Rayon worker must finish instead of waiting on the context pool",
        );
        assert_eq!(results.len(), 2, "enclosing workers = {enclosing_workers}");
        assert!(
            results.iter().any(Option::is_some),
            "at least one concurrent worker entry must complete a session, got {results:?}"
        );
        for values in results.into_iter().flatten() {
            assert_eq!(values, vec![23.0, 34.0]);
        }
    }
}

#[test]
fn graph_pool_stats_and_reset_track_the_context_backend_pool() {
    use tenferro_cpu::with_cpu_exec_session;
    use tenferro_tensor::TensorBuffer;

    let context = context();
    let mut trace = TraceContext::new();
    let input = trace
        .input(ProgramInputSpec::new(DType::F64, [2_usize.into()]))
        .unwrap();
    let output = trace.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
    let graph = trace.finish(&[output]).unwrap();
    let compiled = context.compile_graph(&graph).unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();
    let output = context.run_graph(&compiled, &[&input]).unwrap();
    assert_eq!(output[0].as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
    drop(output);

    // Return one host tensor to the pool the context owns.
    context
        .with_session(|session| {
            with_cpu_exec_session(session, |cpu| {
                cpu.reclaim_buffer(Tensor::from_vec_col_major(vec![8], vec![1.0_f64; 8]).unwrap());
            })
        })
        .unwrap()
        .expect("the CPU context must expose its execution session");

    // The graph-side view reports the pool the context's backend holds.
    let retained = context.graph_buffer_pool_stats().unwrap();
    assert!(retained.buffers >= 1);
    assert!(retained.capacity_bytes >= 8 * std::mem::size_of::<f64>());
    assert_eq!(
        retained,
        context.with_backend(|backend| backend.buffer_pool_stats().unwrap())
    );

    context.reset_graph_buffer_pool().unwrap();
    assert_eq!(
        context.graph_buffer_pool_stats().unwrap(),
        BufferPoolStats::default()
    );
    assert_eq!(
        context.with_backend(|backend| backend.buffer_pool_stats().unwrap()),
        BufferPoolStats::default()
    );

    // Releasing the pool keeps the runtime and its prepared plan cache.
    let output = context.run_graph(&compiled, &[&input]).unwrap();
    assert_eq!(output[0].as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
    assert!(context.graph_cache_stats().unwrap().prepared_plans.entries > 0);
}

#[cfg(feature = "global-defaults")]
#[test]
fn explicit_paths_stay_independent_of_the_default_context() {
    let context = context();
    context.with_backend(|backend| assert_eq!(backend.num_threads(), 1));
    let eager = context.eager_runtime().unwrap();

    // The explicit context owns its eager runtime and is not the process-global one.
    // This replaces a former before/after comparison of how often the process-global
    // context was initialized: that counter is process-wide, and any other test that
    // legitimately reaches the compatibility entry initialized it concurrently, which
    // made the comparison flaky.
    let global = defaults::default_eager_ctx().expect("default eager context");
    assert!(!Arc::ptr_eq(&eager, &global));
    assert!(!ExecutionContext::Cpu(Arc::new(context)).is_global_default_cpu());
}
