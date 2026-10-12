//! Caller-owned-context overhead for the TreeTCI interpolation engine (issue #859 F7).
//!
//! [`TreeTciInterpolatorIn`] routes the optimizer and final materialization of
//! one `TreeInterpolator::interpolate` run through a caller-owned CPU context
//! instead of the process-global one. This benchmark compares the two engines
//! on one explicit thread so the comparison is route-for-route, not a
//! thread-count benchmark.
//!
//! # Thread environment
//!
//! The caller-owned engine holds `CpuBackend::with_threads(1)` and every BLAS /
//! Rayon / OMP backend the two routes may touch is pinned to one thread via
//! `RAYON_NUM_THREADS`, `BLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
//! `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS`, so the effective thread count
//! of the measured comparison is 1.
//!
//! # Running
//!
//! ```text
//! taskset -c 0 cargo bench -p tensor4all-partitionedtreetn --bench f7_engine_context_overhead -- --noplot
//! ```

use std::collections::BTreeMap;
use std::sync::{Arc, Once};
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tenferro_cpu::CpuBackend;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
use tensor4all_tensorbackend::CpuExecutionContext;
use tensor4all_treetci::{TreeTciInterpolator, TreeTciInterpolatorIn, TreeTciOptions};
use tensor4all_treetn::interpolation::TreeInterpolator;
use tensor4all_treetn::{interpolation::InterpolationProblem, NodeNameNetwork};

static ONE_THREAD: Once = Once::new();
fn pin_one_thread() {
    ONE_THREAD.call_once(|| {
        // SAFETY: benches are one-shot processes; the env is set before any
        // backend in this process is created.
        unsafe {
            for var in [
                "RAYON_NUM_THREADS",
                "BLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            ] {
                std::env::set_var(var, "1");
            }
        }
    });
}

fn problem() -> InterpolationProblem<usize> {
    let mut topology = NodeNameNetwork::new();
    topology.add_node(0usize).unwrap();
    topology.add_node(1usize).unwrap();
    topology.add_edge(&0, &1).unwrap();
    let node_sites = BTreeMap::from([
        (0usize, vec![DynIndex::new_dyn(4)]),
        (1, vec![DynIndex::new_dyn(4)]),
    ]);
    let pivots = ColMajorArray::new(vec![0, 0], vec![2, 1]).unwrap();
    InterpolationProblem::new(topology, node_sites, pivots, 1e-10, None, 0).unwrap()
}

fn evaluate(batch: ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>> {
    // A smooth two-variable function; 16 points, cheap to sample. Column-major
    // data: for point `p`, row `r` sits at `data[p * rows + r]`.
    let rows = batch.shape()[0];
    let count = batch.shape()[1];
    let mut out = Vec::with_capacity(count);
    for point in 0..count {
        let x = batch.data()[point * rows] as f64 / 3.0;
        let y = batch.data()[point * rows + 1] as f64 / 3.0;
        out.push((1.0 + x).sin() + 0.25 * (2.0 * y + x).cos());
    }
    Ok(out)
}

fn bench_plain(criterion: &mut Criterion) {
    pin_one_thread();
    let problem = problem();
    let engine = TreeTciInterpolator::default();
    let mut group = criterion.benchmark_group("f7/plain");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_global_route", |bench| {
        bench.iter(|| black_box(engine.interpolate(&problem, evaluate)))
    });
}

fn bench_context(criterion: &mut Criterion) {
    pin_one_thread();
    let problem = problem();
    let context = Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).expect("1T backend"),
    ));
    let engine = TreeTciInterpolatorIn::new(TreeTciOptions::default(), context).expect("engine");
    let mut group = criterion.benchmark_group("f7/context");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_caller_owned", |bench| {
        bench.iter(|| black_box(engine.interpolate(&problem, evaluate)))
    });
}

criterion_group!(benches, bench_plain, bench_context);
criterion_main!(benches);
