use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tenferro::program::{CoreSemanticOp, ProgramInputSpec};
use tenferro::{DType, ExecutionEngine, Tensor, TraceContext};
use tenferro_ad::EagerTensor;
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EagerEinsumOps, EinsumSubscripts};
use tenferro_tensor::{AllocationDomainId, SharedTensorAllocationDomain};

use super::{Runtime, RuntimeError};
use crate::CpuBackend;

fn pooled(threads: usize) -> CpuEngine {
    CpuBackend::with_threads(threads)
        .expect("a pooled CPU backend")
        .into_engine()
}

fn matrix(rt: &Runtime, rows: usize, cols: usize, offset: f64) -> EagerTensor {
    let data = (0..rows * cols)
        .map(|k| ((k as f64 + offset) * 0.37).sin())
        .collect::<Vec<_>>();
    Arc::clone(rt.eager_runtime())
        .constant_from(Tensor::from_vec_col_major(vec![rows, cols], data).unwrap())
        .unwrap()
}

/// A value computed on `exec`: the input doubled, so it is an engine result,
/// not a host constant.
fn computed_on(rt: &Runtime, exec: &mut CpuEngine, input: &EagerTensor) -> EagerTensor {
    rt.with_eager(exec, |session| session.add(input, input))
        .expect("doubling on the engine")
}

fn contract_on(rt: &Runtime, exec: &mut CpuEngine, a: &EagerTensor, b: &EagerTensor) -> Vec<f64> {
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let product = rt
        .with_eager(exec, |session| session.einsum(&[a, b], &subscripts))
        .expect("contraction on the engine");
    let values = product.value().unwrap();
    values.as_slice::<f64>().unwrap().to_vec()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// One runtime serves a serial and a 4-thread engine: a value made on each
/// contracts on either engine, and the result equals, bit for bit, the same
/// contraction of values made on that engine alone.
#[test]
fn two_engines_share_one_runtime_and_results_are_bitwise_equal() {
    let rt = Runtime::new().unwrap();
    let mut serial = CpuEngine::serial();
    let mut pool = pooled(4);
    let a_input = matrix(&rt, 48, 40, 0.0);
    let b_input = matrix(&rt, 40, 36, 7.0);

    let a_serial = computed_on(&rt, &mut serial, &a_input);
    let b_pooled = computed_on(&rt, &mut pool, &b_input);

    for (name, exec) in [("serial", &mut serial), ("pooled", &mut pool)] {
        let mixed = contract_on(&rt, exec, &a_serial, &b_pooled);
        // Single-engine reference: both operands made on this engine.
        let a_here = computed_on(&rt, exec, &a_input);
        let b_here = computed_on(&rt, exec, &b_input);
        let single = contract_on(&rt, exec, &a_here, &b_here);
        assert_eq!(
            bits(&mixed),
            bits(&single),
            "the {name} engine must give the single-engine result bit for bit"
        );
    }
}

/// Values of one runtime may be produced concurrently by tasks that each own
/// an engine, and then combined on another engine.
#[test]
fn lane_tasks_with_their_own_engines_share_one_runtime() {
    let rt = Runtime::new().unwrap();
    let input = matrix(&rt, 8, 8, 1.0);
    let mut lanes: Vec<CpuEngine> = (0..3).map(|_| CpuEngine::serial()).collect();
    let parts = std::thread::scope(|scope| {
        let handles = lanes
            .iter_mut()
            .map(|exec| {
                let rt = rt.clone();
                let input = input.clone();
                scope.spawn(move || computed_on(&rt, exec, &input))
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut big = pooled(2);
    let sum = rt
        .with_eager(&mut big, |session| {
            let partial = session.add(&parts[0], &parts[1])?;
            session.add(&partial, &parts[2])
        })
        .unwrap();
    let expected = input
        .value()
        .unwrap()
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .map(|value| 6.0 * value)
        .collect::<Vec<_>>();
    let got = sum.value().unwrap().as_slice::<f64>().unwrap().to_vec();
    for (got, expected) in got.iter().zip(&expected) {
        assert!((got - expected).abs() <= 1e-15 * expected.abs().max(1.0));
    }
    assert_eq!(sum.ctx_id(), rt.id());
}

/// The graph runtime is shared too: a program prepared for one CPU engine is
/// reused by another, and both give the same output bit for bit.
#[test]
fn run_graph_serves_several_engines_from_one_prepared_cache() {
    let rt = Runtime::new().unwrap();
    let trace = TraceContext::new();
    let input = trace
        .input(ProgramInputSpec::new(DType::F64, [5_usize.into()]))
        .unwrap();
    let output = trace.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
    let compiled = rt.compile_graph(&trace.finish(&[output]).unwrap()).unwrap();
    let values = Tensor::from_vec_col_major(vec![5], vec![1.0_f64, -2.0, 3.5, 0.0, -0.25]).unwrap();

    let mut serial = CpuEngine::serial();
    let mut pool = pooled(4);
    let a = rt.run_graph(&compiled, &[&values], &mut serial).unwrap();
    let after_first = rt.graph_cache_stats().unwrap().prepared_plans;
    let b = rt.run_graph(&compiled, &[&values], &mut pool).unwrap();
    assert_eq!(
        bits(a[0].as_slice::<f64>().unwrap()),
        bits(b[0].as_slice::<f64>().unwrap())
    );
    assert_eq!(
        a[0].as_slice::<f64>().unwrap(),
        &[-1.0, 2.0, -3.5, -0.0, 0.25]
    );
    let stats = rt.graph_cache_stats().unwrap().prepared_plans;
    assert!(after_first.entries > 0);
    assert_eq!(
        stats.entries, after_first.entries,
        "the second engine prepares nothing new"
    );
    assert!(
        stats.hits > after_first.hits,
        "the second engine reuses the plans the first one prepared"
    );
}

#[derive(Debug)]
struct NeverAllocates {
    id: AllocationDomainId,
}

impl SharedTensorAllocationDomain for NeverAllocates {
    fn id(&self) -> AllocationDomainId {
        self.id
    }

    fn allocate(&self, _dtype: DType, _shape: &[usize]) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "NeverAllocates::allocate",
            "the test domain is refused before any allocation",
        ))
    }
}

/// An engine with a custom runtime id or its own allocation domain is
/// refused with the typed error before any work, never registered lazily.
#[test]
fn custom_id_and_foreign_allocation_domain_are_unsupported_engines() {
    let rt = Runtime::new().unwrap();
    let custom = tenferro::EngineId::new("tensor4all-test.cpu.v1").unwrap();
    let domain: Arc<dyn SharedTensorAllocationDomain> = Arc::new(NeverAllocates {
        id: AllocationDomainId::fresh(),
    });
    let refused = [
        (
            CpuEngine::serial().with_runtime_engine_id(custom),
            "runtime engine id",
        ),
        (
            CpuEngine::serial().with_allocation_domain(domain),
            "allocation domain",
        ),
    ];
    let trace = TraceContext::new();
    let input = trace
        .input(ProgramInputSpec::new(DType::F64, [1_usize.into()]))
        .unwrap();
    let output = trace.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
    let compiled = rt.compile_graph(&trace.finish(&[output]).unwrap()).unwrap();
    let values = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let x = matrix(&rt, 1, 1, 0.0);
    for (mut exec, what) in refused {
        match rt.check_engine(&exec) {
            Err(RuntimeError::UnsupportedEngine { reason, .. }) => {
                assert!(reason.contains(what), "{what}: {reason}")
            }
            other => panic!("{what}: expected UnsupportedEngine, got {other:?}"),
        }
        assert!(matches!(
            rt.run_graph(&compiled, &[&values], &mut exec),
            Err(RuntimeError::UnsupportedEngine { .. })
        ));
        let ran = AtomicBool::new(false);
        let error = rt
            .with_eager(&mut exec, |session| {
                ran.store(true, Ordering::SeqCst);
                session.neg(&x)
            })
            .expect_err("an unsupported engine is refused");
        assert!(
            !ran.load(Ordering::SeqCst),
            "{what}: refused before `f` runs"
        );
        assert!(error.to_string().contains("not supported by this runtime"));
    }
    // The refusals did not register anything: the runtime still has exactly
    // the one prepared-plan-free graph cache it started with.
    assert_eq!(rt.graph_cache_stats().unwrap().prepared_plans.entries, 0);
}

#[test]
fn runtime_is_a_shareable_handle() {
    fn assert_send_sync<T: Send + Sync + Clone>() {}
    assert_send_sync::<Runtime>();

    let rt = Runtime::new().unwrap();
    let clone = rt.clone();
    assert_eq!(clone.id(), rt.id());
    assert!(Arc::ptr_eq(clone.eager_runtime(), rt.eager_runtime()));
    assert_ne!(Runtime::new().unwrap().id(), rt.id());
    assert!(format!("{rt:?}").contains("Runtime"));
}

/// Resetting the graph side keeps the eager runtime and its values.
#[test]
fn reset_graph_keeps_the_eager_runtime() {
    let rt = Runtime::new().unwrap();
    let x = matrix(&rt, 2, 2, 0.0);
    let trace = TraceContext::new();
    let input = trace
        .input(ProgramInputSpec::new(DType::F64, [1_usize.into()]))
        .unwrap();
    let output = trace.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
    let compiled = rt.compile_graph(&trace.finish(&[output]).unwrap()).unwrap();
    let values = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let mut exec = CpuEngine::serial();
    rt.run_graph(&compiled, &[&values], &mut exec).unwrap();
    assert!(rt.graph_cache_stats().unwrap().prepared_plans.entries > 0);

    rt.reset_graph().unwrap();
    assert_eq!(rt.graph_cache_stats().unwrap().prepared_plans.entries, 0);
    let y = rt.with_eager(&mut exec, |session| session.neg(&x)).unwrap();
    assert_eq!(y.ctx_id(), rt.id());
}

/// A context's runtime is built for its engine's own target: an engine with
/// an allocation domain is supported by the runtime built for it, and a
/// domain-less engine is refused by that runtime.
#[test]
fn a_runtime_built_for_a_domain_engine_supports_that_engine() {
    let domain: Arc<dyn SharedTensorAllocationDomain> = Arc::new(NeverAllocates {
        id: AllocationDomainId::fresh(),
    });
    let engine = CpuEngine::serial().with_allocation_domain(Arc::clone(&domain));
    let rt = Runtime::for_cpu_engine(&super::CpuTarget::of(&engine)).unwrap();
    rt.check_engine(&engine)
        .expect("the runtime supports the engine it was built for");
    let same_domain = pooled(2).with_allocation_domain(domain);
    rt.check_engine(&same_domain)
        .expect("another engine with the same id and domain is supported");
    assert!(matches!(
        rt.check_engine(&CpuEngine::serial()),
        Err(RuntimeError::UnsupportedEngine { .. })
    ));
}

/// The work runs on the caller's `exec`, not on some other engine: a broken
/// engine with the supported target refuses, while a healthy one computes.
#[test]
fn work_runs_on_the_engine_the_caller_passes() {
    let rt = Runtime::new().unwrap();
    let x = matrix(&rt, 2, 3, 0.0);
    let trace = TraceContext::new();
    let input = trace
        .input(ProgramInputSpec::new(DType::F64, [2_usize.into()]))
        .unwrap();
    let output = trace.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
    let compiled = rt.compile_graph(&trace.finish(&[output]).unwrap()).unwrap();
    let values = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let mut broken = CpuEngine::serial();
    broken.mark_broken();
    rt.check_engine(&broken)
        .expect("a broken engine still has the supported target");
    rt.with_eager(&mut broken, |session| session.add(&x, &x))
        .expect_err("eager work runs on the broken engine and is refused");
    rt.run_graph(&compiled, &[&values], &mut broken)
        .expect_err("graph work runs on the broken engine and is refused");

    let mut healthy = CpuEngine::serial();
    rt.with_eager(&mut healthy, |session| session.add(&x, &x))
        .expect("the same work on a healthy engine computes");
    rt.run_graph(&compiled, &[&values], &mut healthy).unwrap();
}
