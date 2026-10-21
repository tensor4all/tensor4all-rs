use std::error::Error;
use std::sync::mpsc;
use std::time::Duration;

use super::*;
use tenferro::program::{CoreSemanticOp, ProgramInputSpec};
use tenferro::{DType, TensorSessionOpsExt, TraceContext};
use tenferro_ad::EagerTensor;
use tenferro_einsum::{EagerSessionEinsumExt, EinsumSubscripts};

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
fn recursive_graph_entries_are_rejected_typed() {
    let context = context();
    let mut trace = TraceContext::new();
    let input = trace
        .input(ProgramInputSpec::new(DType::F64, [2_usize.into()]))
        .unwrap();
    let output = trace.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
    let graph = trace.finish(&[output]).unwrap();
    let compiled = context.compile_graph(&graph).unwrap();
    let values = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();

    // Graph execution and the graph cache/pool entries lock the engine, so a
    // session of the same context rejects them instead of waiting for its own
    // lock.
    let reentered = context
        .with_session(|_| context.run_graph(&compiled, &[&values]))
        .expect("the outer session entry")
        .expect_err("a graph run inside a session must be rejected");
    assert!(
        matches!(reentered, CpuExecutionContextError::Reentered),
        "the rejection must be the typed re-entry error: {reentered}"
    );

    for error in [
        context
            .with_session(|_| context.graph_cache_stats())
            .expect("the outer session entry")
            .expect_err("cache statistics inside a session must be rejected"),
        context
            .with_session(|_| context.graph_buffer_pool_stats())
            .expect("the outer session entry")
            .expect_err("pool statistics inside a session must be rejected"),
        context
            .with_session(|_| context.reset_graph_buffer_pool())
            .expect("the outer session entry")
            .expect_err("a pool reset inside a session must be rejected"),
        context
            .with_session(|_| context.reset_graph_runtime())
            .expect("the outer session entry")
            .expect_err("a runtime reset inside a session must be rejected"),
    ] {
        assert!(
            matches!(error, CpuExecutionContextError::Reentered),
            "the rejection must be the typed re-entry error: {error}"
        );
    }

    // The guard is restored, so the same entries work outside a session.
    let output = context.run_graph(&compiled, &[&values]).unwrap();
    assert_eq!(output[0].as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
    context.graph_cache_stats().unwrap();
    context.reset_graph_buffer_pool().unwrap();
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

/// A session entered from a Rayon worker of *another* pool completes while the
/// context engine is held by a thread waiting for that pool's result.
///
/// The inline worker path must not touch the engine lock: here the engine holder
/// itself waits for the worker, so an engine-locking worker entry would wait for
/// its own holder forever.
#[test]
fn a_worker_session_completes_while_another_thread_holds_the_engine() {
    let context = Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(2).unwrap(),
    ));
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap(),
    );
    let (entered_tx, entered_rx) = mpsc::channel();
    let (holding_tx, holding_rx) = mpsc::channel();
    let holder_context = Arc::clone(&context);
    let holder_pool = Arc::clone(&pool);
    let holder = std::thread::spawn(move || {
        // The thread holds the engine for the whole closure and then waits for a
        // worker of `pool` to finish its own session of the same context.
        holder_context.with_session(|_| {
            holding_tx.send(()).unwrap();
            let context = Arc::clone(&holder_context);
            let entered_tx = entered_tx.clone();
            holder_pool.install(move || {
                entered_tx.send(()).unwrap();
                context.with_session(multiply_on)
            })
        })
    });
    holding_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the holder thread enters the context engine");

    // The worker entry must be observable; if it waits for the held engine the
    // holder never returns and the join below times out.
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = done_tx.send(holder.join());
    });
    entered_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the worker enters the same context's session");
    done_rx
        .recv_timeout(Duration::from_secs(60))
        .expect("a worker session must not wait for an engine its holder already owns")
        .expect("the holder thread joins")
        .expect("the worker session completes")
        .expect("the pool install returns the worker result")
        .expect("the inline engine computes the product");
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

/// A panic that poisons the context lock keeps the caller's engine: the lock is
/// recovered, and the broken engine refuses operations with its typed error
/// instead of being silently replaced by a default engine.
#[test]
fn a_poisoned_lock_keeps_the_broken_engine_and_refuses_typed() {
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(2).unwrap());

    // Break the engine and poison the lock from inside it.
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        context.with_backend(|backend| {
            backend.engine_mut().mark_broken();
            panic!("poison the context lock");
        })
    }));
    assert!(poisoned.is_err(), "the closure panics");

    // The lock is usable again and the engine is the caller's own: same thread
    // count, still broken.
    assert!(context.with_backend(|backend| backend.engine().is_broken()));
    assert_eq!(context.with_backend(|backend| backend.num_threads()), 2);

    // Operations are refused with tenferro's typed broken-engine error rather
    // than panicking or running on a substitute engine.
    let error = context
        .with_session(multiply_on)
        .expect("the context still lends its engine")
        .expect_err("a broken engine refuses operations");
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::RuntimeState);
    assert!(
        error
            .source()
            .and_then(|source| source.downcast_ref::<tenferro_cpu::CpuEngineError>())
            == Some(&tenferro_cpu::CpuEngineError::Broken),
        "the refusal must be the typed broken-engine error: {error}"
    );
}

/// An engine built with a non-canonical runtime id runs graphs and eager
/// operations: the context registers the graph runtime and installs the eager
/// extension modules under the engine's own id.
#[test]
fn a_custom_runtime_engine_id_runs_graphs_and_eager_operations() {
    let custom = tenferro::EngineId::new("tensor4all-test.cpu.v1").unwrap();
    let engine = CpuEngine::serial().with_runtime_engine_id(custom.clone());
    let context = CpuExecutionContext::from_backend(CpuBackend::from(engine));

    let mut trace = TraceContext::new();
    let input = trace
        .input(ProgramInputSpec::new(DType::F64, [2_usize.into()]))
        .unwrap();
    let output = trace.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
    let graph = trace.finish(&[output]).unwrap();
    let compiled = context
        .compile_graph(&graph)
        .expect("a custom-id context compiles graphs");
    let values = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();
    let output = context
        .run_graph(&compiled, &[&values])
        .expect("graph preparation finds the engine under its own id");
    assert_eq!(output[0].as_slice::<f64>().unwrap(), &[-1.0, 2.0]);

    // The eager route installs its einsum/linalg extension modules under the
    // same id, so an extension operation also finds this engine.
    let runtime = context.eager_runtime().expect("custom-id eager runtime");
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        Arc::clone(&runtime),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap(),
        Arc::clone(&runtime),
    )
    .unwrap();
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let mut engine = CpuEngine::serial().with_runtime_engine_id(custom);
    let product = runtime
        .with_eager_session(&mut engine, |session| {
            session.einsum_subscripts(&[&lhs, &rhs], &subscripts)
        })
        .expect("the custom-id engine finds its einsum extension");
    assert_eq!(
        product.value().unwrap().as_slice::<f64>().unwrap(),
        &[23.0, 34.0]
    );
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

/// An implicit entry on a context-owned runtime runs on that context's engine, not
/// on the process-global one, and falls back to the global context once the owner
/// is gone.
#[cfg(feature = "global-defaults")]
#[test]
fn runtime_engine_follows_the_owning_context_and_falls_back_after_drop() {
    let default_threads = default_context().with_backend(|backend| backend.num_threads());
    let context_threads = if default_threads == 1 { 2 } else { 1 };
    let context = CpuExecutionContext::from_backend(
        CpuBackend::with_threads(context_threads).expect("CPU backend"),
    );
    let runtime = context.eager_runtime().unwrap();

    // Hold the process-global engine on another thread. A runtime entry that fell
    // back to the global context would block here instead of resolving the owning
    // context's engine.
    let (held_tx, held_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        with_default_backend(|_| {
            held_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
    });
    held_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the holder thread enters the process-global engine");

    let worker_runtime = Arc::clone(&runtime);
    let (result_tx, result_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result = with_runtime_engine(&worker_runtime, |engine| {
            engine
                .as_any_mut()
                .downcast_mut::<CpuEngine>()
                .map(|engine| engine.num_threads())
        });
        let _ = result_tx.send(result);
    });
    let resolved = result_rx.recv_timeout(Duration::from_secs(10));
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    let threads = resolved
        .expect("a context-owned runtime must not wait for the process-global engine")
        .expect("the owning context's session entry");
    worker.join().unwrap();
    assert_eq!(
        threads,
        Some(context_threads),
        "the owning context's engine is used"
    );

    // Keep the runtime but drop its owner: the dead entry falls back to the
    // process-global context's engine.
    drop(context);
    let fallback = with_runtime_engine(&runtime, |engine| {
        engine
            .as_any_mut()
            .downcast_mut::<CpuEngine>()
            .map(|engine| engine.num_threads())
    })
    .expect("the process-global fallback session entry");
    assert_eq!(fallback, Some(default_threads));
}

/// A default pool built from a thread narrowed to one CPU is still sized and
/// masked from the main thread's snapshot.
///
/// Its own test process: the first [`CpuBackend::new`] fixes the process-wide
/// default, so another test in the same binary could legitimately resolve it
/// first. The child re-execution keeps the first call in this test.
#[cfg(target_os = "linux")]
#[test]
fn default_pool_from_a_narrowed_thread_uses_the_main_mask() {
    const CHILD: &str = "T4A_TENSORBACKEND_AFFINITY_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let main_mask = affinity::main_thread_mask().unwrap();
        let main_cpus = main_mask.cpus();
        let mask_words = main_mask.words_len();
        assert!(
            !main_cpus.is_empty(),
            "a running process has at least one CPU"
        );
        let expected = std::thread::available_parallelism().unwrap().get();
        let first = main_cpus[0];
        let second = *main_cpus.get(1).unwrap_or(&first);

        let (threads, backend) = std::thread::spawn(move || {
            let first_mask = affinity::CpuMask::single(first, mask_words);
            affinity::set_current_thread_to(&first_mask).unwrap();
            let backend = CpuBackend::new();
            (backend.num_threads(), backend)
        })
        .join()
        .unwrap();
        assert_eq!(
            threads, expected,
            "the construction thread's one-CPU mask must not size the pool"
        );

        // A second construction from another mask keeps the fixed count.
        let second_threads = std::thread::spawn(move || {
            let second_mask = affinity::CpuMask::single(second, mask_words);
            affinity::set_current_thread_to(&second_mask).unwrap();
            CpuBackend::new().num_threads()
        })
        .join()
        .unwrap();
        assert_eq!(second_threads, expected);
        assert_eq!(
            set_default_cpu_threads(NonZeroUsize::new(2).expect("two workers")),
            Err(AlreadyInitialized)
        );

        if expected > 1 {
            let pool = backend
                .engine()
                .pool()
                .expect("a resolved count above one owns a pool");
            let workers = pool.broadcast(|ctx| {
                let own = affinity::current_thread_cpus().unwrap();
                let spawned = std::thread::spawn(affinity::current_thread_cpus)
                    .join()
                    .unwrap()
                    .unwrap();
                (ctx.index(), own, spawned)
            });
            assert_eq!(workers.len(), expected);
            for (index, own, spawned) in workers {
                assert_eq!(
                    own, main_cpus,
                    "worker {index} must inherit the full main-thread mask"
                );
                assert_eq!(
                    spawned, main_cpus,
                    "a thread spawned by worker {index} must inherit the full mask"
                );
            }
        }
        return;
    }

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "context::tests::default_pool_from_a_narrowed_thread_uses_the_main_mask",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env_remove("RAYON_NUM_THREADS")
        .env_remove("RAYON_RS_NUM_CPUS")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "the affinity child failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// An explicit-count pool built from a narrowed thread still covers the whole
/// main-thread mask.
#[cfg(target_os = "linux")]
#[test]
fn explicit_pool_from_a_narrowed_thread_covers_the_main_mask() {
    let main_mask = affinity::main_thread_mask().unwrap();
    let main_cpus = main_mask.cpus();
    let mask_words = main_mask.words_len();
    let first = main_cpus[0];
    let backend = std::thread::spawn(move || {
        let first_mask = affinity::CpuMask::single(first, mask_words);
        affinity::set_current_thread_to(&first_mask).unwrap();
        CpuBackend::with_threads(2).unwrap()
    })
    .join()
    .unwrap();
    let pool = backend.engine().pool().expect("two workers own a pool");
    let workers = pool.broadcast(|_| affinity::current_thread_cpus().unwrap());
    assert_eq!(workers.len(), 2);
    for mask in workers {
        assert_eq!(mask, main_cpus);
    }
}

/// Every pinned worker runs on exactly its one listed CPU, including the
/// one-element list that still builds a one-worker pool.
#[cfg(target_os = "linux")]
#[test]
fn pinned_workers_run_on_exactly_their_cpu() {
    let allowed = CpuBackend::allowed_cpus().unwrap();
    let cpus = &allowed[..allowed.len().min(2)];
    let backend = CpuBackend::pinned(cpus).unwrap();
    assert_eq!(backend.num_threads(), cpus.len());
    let pool = backend
        .engine()
        .pool()
        .expect("pinned never takes the serial path");
    let workers = pool.broadcast(|ctx| (ctx.index(), affinity::current_thread_cpus().unwrap()));
    assert_eq!(workers.len(), cpus.len());
    for (index, mask) in workers {
        assert_eq!(mask, vec![cpus[index]], "worker {index}");
    }
}

/// The pinned constructor reports each invalid CPU list as its typed error,
/// including an id far outside the mask without indexing or allocating for it.
#[cfg(target_os = "linux")]
#[test]
fn pinned_rejects_invalid_cpu_lists_typed() {
    assert!(matches!(
        CpuBackend::pinned(&[]),
        Err(CpuThreadError::EmptyCpuList)
    ));

    let allowed = CpuBackend::allowed_cpus().unwrap();
    let cpu = allowed[0];
    assert!(matches!(
        CpuBackend::pinned(&[cpu, cpu]),
        Err(CpuThreadError::DuplicateCpu { cpu: repeated }) if repeated == cpu
    ));

    let outside = allowed.last().copied().unwrap() + 1;
    assert!(matches!(
        CpuBackend::pinned(&[outside]),
        Err(CpuThreadError::CpuNotAllowed { cpu: rejected }) if rejected == outside
    ));
    assert!(matches!(
        CpuBackend::pinned(&[usize::MAX]),
        Err(CpuThreadError::CpuNotAllowed { cpu: rejected }) if rejected == usize::MAX
    ));
}

/// Off Linux the two affinity entry points report the typed platform error.
#[cfg(not(target_os = "linux"))]
#[test]
fn affinity_is_unsupported_off_linux() {
    assert!(matches!(
        CpuBackend::allowed_cpus(),
        Err(CpuThreadError::UnsupportedPlatform)
    ));
    assert!(matches!(
        CpuBackend::pinned(&[0]),
        Err(CpuThreadError::UnsupportedPlatform)
    ));
}
