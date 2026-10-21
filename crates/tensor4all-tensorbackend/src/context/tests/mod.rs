use std::sync::mpsc;
use std::time::Duration;

use super::*;
use tenferro::program::{CoreSemanticOp, ProgramInputSpec};
use tenferro::{DType, TensorSessionOpsExt, TraceContext};
use tenferro_ad::EagerTensor;

fn context() -> CpuExecutionContext {
    CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap())
}

/// One 2x2 matrix product on `engine`.
fn multiply_on(engine: &mut tenferro_cpu::CpuEngine) -> tenferro_tensor::Result<Tensor> {
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap();
    lhs.matmul(&rhs, engine)
}

#[test]
fn explicit_session_runs_a_concrete_operation() {
    let context = context();
    let result = context.with_session(multiply_on).unwrap().unwrap();
    assert_eq!(result.as_slice::<f64>().unwrap(), &[23.0, 34.0]);
}

#[test]
fn recursive_session_entry_fails_before_lock_and_restores_guard() {
    let context = context();
    let error = context
        .with_session(|_| context.with_session(|_| ()))
        .expect("the outer session entry")
        .expect_err("a recursive canonical session entry must be rejected");
    assert!(
        matches!(error, CpuExecutionContextError::Reentered),
        "the rejection must be the typed re-entry error: {error}"
    );

    // The guard is restored, so an independent entry still works.
    assert_eq!(context.with_session(|_| 7usize).unwrap(), 7);
}

#[test]
fn a_session_of_another_context_is_not_a_reentry() {
    let first = context();
    let second = context();

    // Re-entry is per context: the second context's engine is independent even
    // while this thread holds the first one.
    let value = first
        .with_session(|_| {
            second
                .with_session(|_| 7usize)
                .expect("the second context's session")
        })
        .expect("the first context's session");
    assert_eq!(value, 7);
}

#[test]
fn a_session_inside_a_backend_borrow_is_rejected_typed() {
    let context = context();
    let error = context
        .with_backend(|_| {
            context
                .with_session(|_| ())
                .map_err(|error| error.to_string())
        })
        .expect_err("a session entry inside a borrowed engine must be rejected");
    assert!(
        error.contains("recursive CPU execution-context session entry"),
        "the rejection is the typed entry error: {error}"
    );

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

/// The eager runtime is engine-independent: one runtime accepts any engine of the
/// registered type, and two runtimes stay separate domains.
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
    let mut engine = CpuEngine::serial();
    let error = first
        .with_eager_session(&mut engine, |session| session.add(&a, &b))
        .expect_err("cross-context addition must be rejected");
    assert!(matches!(error, tenferro_ad::Error::ContextMismatch { .. }));
}

/// A backend built from a caller-supplied engine runs on that engine, with the
/// caller's thread policy, and drops it with the context.
#[test]
fn caller_supplied_context_stays_caller_owned() {
    let engine = CpuEngine::new(
        tenferro_cpu::thread_pool_builder()
            .num_threads(2)
            .build()
            .unwrap(),
    );
    let context = CpuExecutionContext::from_backend(CpuBackend::from(engine));
    assert_eq!(context.with_backend(|backend| backend.num_threads()), 2);
    assert!(context.with_backend(|backend| backend.engine().pool().is_some()));
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

/// A session entered from a Rayon worker runs inline on a serial engine instead of
/// waiting for an engine it belongs to (tensor4all-rs#830): the worker that is handed
/// more of the enclosing pool's work never blocks on the context engine.
#[test]
fn session_from_a_rayon_worker_runs_inline() {
    use rayon::prelude::*;

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
                    .map(|_| context.with_session(multiply_on))
                    .collect::<Vec<_>>()
            });
            let _ = sender.send(results);
        });

        let results = receiver.recv_timeout(Duration::from_secs(60)).expect(
            "a session entered from a Rayon worker must finish instead of waiting on the \
             context engine",
        );
        assert_eq!(results.len(), 2, "enclosing workers = {enclosing_workers}");
        for outcome in results {
            let product = outcome
                .expect("a worker entry is never rejected")
                .expect("the inline engine computes the product");
            assert_eq!(product.as_slice::<f64>().unwrap(), vec![23.0, 34.0]);
        }
    }
}

/// The graph buffer-pool view is the context engine's own pool, and resetting it
/// keeps the runtime and its prepared plans.
#[test]
fn graph_pool_stats_and_reset_track_the_context_engine_pool() {
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
        .with_session(|engine| {
            engine.reclaim_buffer(Tensor::from_vec_col_major(vec![8], vec![1.0_f64; 8]).unwrap());
        })
        .expect("the context lends its engine");

    let retained = context.graph_buffer_pool_stats().unwrap();
    assert!(retained.buffers >= 1);
    assert!(retained.capacity_bytes >= 8 * std::mem::size_of::<f64>());
    assert_eq!(
        retained,
        context.with_backend(|backend| backend.engine().buffer_pool_stats())
    );

    context.reset_graph_buffer_pool().unwrap();
    assert_eq!(
        context.graph_buffer_pool_stats().unwrap(),
        BufferPoolStats::default()
    );
    assert_eq!(
        context.with_backend(|backend| backend.engine().buffer_pool_stats()),
        BufferPoolStats::default()
    );

    // Releasing the pool keeps the runtime and its prepared plan cache.
    let output = context.run_graph(&compiled, &[&input]).unwrap();
    assert_eq!(output[0].as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
    assert!(context.graph_cache_stats().unwrap().prepared_plans.entries > 0);
}

/// A panic that poisons the context lock hands the next caller an engine it can use:
/// a broken engine is replaced by a fresh one with the same thread count.
#[test]
fn a_poisoned_lock_recovers_a_broken_engine() {
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(2).unwrap());

    // Break the engine and poison the lock from inside it.
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        context.with_backend(|backend| {
            backend.engine_mut().mark_broken();
            panic!("poison the context lock");
        })
    }));
    assert!(poisoned.is_err(), "the closure panics");

    // The next entry recovers: the broken engine is replaced by a usable one of the
    // same thread count, and the context keeps working.
    assert!(!context.with_backend(|backend| backend.engine().is_broken()));
    assert_eq!(context.with_backend(|backend| backend.num_threads()), 2);
    let product = context.with_session(multiply_on).unwrap().unwrap();
    assert_eq!(product.as_slice::<f64>().unwrap(), &[23.0, 34.0]);
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
