//! Caller-owned-context overhead for SRC tensor-train contraction (issue #859 F7).
//!
//! `contract_in` sends the SRC method's session-bound work through a
//! caller-owned `ExecutionContext` instead of the process-global one (Zip-up,
//! Fit and Naive keep their existing route; the F7 inventory records the
//! treetn-side twins they need). This benchmark compares the two routes on one
//! explicit thread so the comparison is route-for-route, not a thread-count
//! benchmark.
//!
//! # Thread environment
//!
//! The caller-owned context is `CpuBackend::with_threads(1)` and every BLAS /
//! Rayon / OMP backend the two routes may touch is pinned to one thread via
//! `RAYON_NUM_THREADS`, `BLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
//! `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS`, so the effective thread count
//! of the measured comparison is 1.
//!
//! # Input provenance
//!
//! The caller-owned route requires its inputs to belong to the supplied
//! context (tensors built with `IdxTensor::from_dense_in`); the plain route
//! uses its own process-global-built trains with the same values.
//!
//! # Running
//!
//! ```text
//! taskset -c 0 cargo bench -p tensor4all-itensorlike --bench f7_contract_src_overhead -- --noplot
//! ```

use std::sync::{Arc, Once};
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_itensorlike::{contract, contract_in, ContractOptions, TensorTrain};
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_tensorbackend::{CpuExecutionContext, ExecutionContext};

/// Pin every BLAS/Rayon/OMP backend the two routes may touch to one thread
/// before either route creates it.
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

fn site_values(indices: &[DynIndex], node: usize) -> Vec<f64> {
    let dim = indices.iter().map(IndexLike::dim).product::<usize>();
    (0..dim)
        .map(|flat| ((3 * flat + 5 * node) % 7) as f64 / 5.0)
        .collect()
}

fn mpo(
    sites: &[DynIndex],
    _seed: usize,
    build: &impl Fn(Vec<DynIndex>, Vec<f64>) -> IdxTensor,
) -> TensorTrain {
    let nsites = sites.len();
    let bonds: Vec<DynIndex> = (0..nsites - 1).map(|_| DynIndex::new_dyn(2)).collect();
    let mut tensors = Vec::with_capacity(nsites);
    for (i, site) in sites.iter().enumerate() {
        let mut indices = Vec::new();
        if i > 0 {
            indices.push(bonds[i - 1].clone());
        }
        indices.push(site.clone());
        if i + 1 < nsites {
            indices.push(bonds[i].clone());
        }
        let values = site_values(&indices, i);
        tensors.push(build(indices, values));
    }
    TensorTrain::new(tensors).unwrap()
}

fn options() -> ContractOptions {
    ContractOptions::src()
        .with_src_options(tensor4all_itensorlike::SrcOptions::adaptive(1.0e-8, 8))
        .with_max_bond_dim(16)
}

fn bench_plain_zipup(criterion: &mut Criterion) {
    pin_one_thread();
    let sites: Vec<DynIndex> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let global = |seed: usize| {
        mpo(&sites, seed, &|indices, values| {
            IdxTensor::from_dense(indices, values).expect("global node")
        })
    };
    let a = global(1);
    let b = global(4);
    let options = ContractOptions::zipup().with_max_bond_dim(16);
    let mut group = criterion.benchmark_group("f7/plain-zipup");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_global_route", |bench| {
        bench.iter_batched(
            || (a.clone(), b.clone()),
            |(a, b)| contract(black_box(&a), black_box(&b), &options),
            BatchSize::SmallInput,
        )
    });
}

fn bench_context_zipup(criterion: &mut Criterion) {
    pin_one_thread();
    let sites: Vec<DynIndex> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).expect("1T backend"),
    )));
    let build = |indices: Vec<DynIndex>, values: Vec<f64>| {
        IdxTensor::from_dense_in(&context, indices, values).expect("in-context node")
    };
    let a = mpo(&sites, 1, &build);
    let b = mpo(&sites, 4, &build);
    let options = ContractOptions::zipup().with_max_bond_dim(16);
    let mut group = criterion.benchmark_group("f7/context-zipup");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_caller_owned", |bench| {
        bench.iter_batched(
            || (a.clone(), b.clone()),
            |(a, b)| contract_in(black_box(&a), black_box(&b), &options, &context),
            BatchSize::SmallInput,
        )
    });
}

fn bench_plain_fit(criterion: &mut Criterion) {
    pin_one_thread();
    let sites: Vec<DynIndex> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let global = |seed: usize| {
        mpo(&sites, seed, &|indices, values| {
            IdxTensor::from_dense(indices, values).expect("global node")
        })
    };
    let a = global(1);
    let b = global(4);
    let options = ContractOptions::fit().with_max_bond_dim(16).with_nsweeps(2);
    let mut group = criterion.benchmark_group("f7/plain-fit");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_global_route", |bench| {
        bench.iter_batched(
            || (a.clone(), b.clone()),
            |(a, b)| contract(black_box(&a), black_box(&b), &options),
            BatchSize::SmallInput,
        )
    });
}

fn bench_context_fit(criterion: &mut Criterion) {
    pin_one_thread();
    let sites: Vec<DynIndex> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).expect("1T backend"),
    )));
    let build = |indices: Vec<DynIndex>, values: Vec<f64>| {
        IdxTensor::from_dense_in(&context, indices, values).expect("in-context node")
    };
    let a = mpo(&sites, 1, &build);
    let b = mpo(&sites, 4, &build);
    let options = ContractOptions::fit().with_max_bond_dim(16).with_nsweeps(2);
    let mut group = criterion.benchmark_group("f7/context-fit");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_caller_owned", |bench| {
        bench.iter_batched(
            || (a.clone(), b.clone()),
            |(a, b)| contract_in(black_box(&a), black_box(&b), &options, &context),
            BatchSize::SmallInput,
        )
    });
}

fn bench_plain_naive(criterion: &mut Criterion) {
    pin_one_thread();
    let sites: Vec<DynIndex> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let global = |seed: usize| {
        mpo(&sites, seed, &|indices, values| {
            IdxTensor::from_dense(indices, values).expect("global node")
        })
    };
    let a = global(1);
    let b = global(4);
    let options = ContractOptions::naive()
        .with_max_bond_dim(16)
        .with_dense_reference_limit(1 << 20);
    let mut group = criterion.benchmark_group("f7/plain-naive");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_global_route", |bench| {
        bench.iter_batched(
            || (a.clone(), b.clone()),
            |(a, b)| contract(black_box(&a), black_box(&b), &options),
            BatchSize::SmallInput,
        )
    });
}

fn bench_context_naive(criterion: &mut Criterion) {
    pin_one_thread();
    let sites: Vec<DynIndex> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).expect("1T backend"),
    )));
    let build = |indices: Vec<DynIndex>, values: Vec<f64>| {
        IdxTensor::from_dense_in(&context, indices, values).expect("in-context node")
    };
    let a = mpo(&sites, 1, &build);
    let b = mpo(&sites, 4, &build);
    let options = ContractOptions::naive()
        .with_max_bond_dim(16)
        .with_dense_reference_limit(1 << 20);
    let mut group = criterion.benchmark_group("f7/context-naive");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_caller_owned", |bench| {
        bench.iter_batched(
            || (a.clone(), b.clone()),
            |(a, b)| contract_in(black_box(&a), black_box(&b), &options, &context),
            BatchSize::SmallInput,
        )
    });
}

fn bench_plain(criterion: &mut Criterion) {
    pin_one_thread();
    let sites: Vec<DynIndex> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let global = |seed: usize| {
        mpo(&sites, seed, &|indices, values| {
            IdxTensor::from_dense(indices, values).expect("global node")
        })
    };
    let a = global(1);
    let b = global(4);
    let options = options();
    let mut group = criterion.benchmark_group("f7/plain");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_global_route", |bench| {
        bench.iter_batched(
            || (a.clone(), b.clone()),
            |(a, b)| contract(black_box(&a), black_box(&b), &options),
            BatchSize::SmallInput,
        )
    });
}

fn bench_context(criterion: &mut Criterion) {
    pin_one_thread();
    let sites: Vec<DynIndex> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).expect("1T backend"),
    )));
    let build = |indices: Vec<DynIndex>, values: Vec<f64>| {
        IdxTensor::from_dense_in(&context, indices, values).expect("in-context node")
    };
    let a = mpo(&sites, 1, &build);
    let b = mpo(&sites, 4, &build);
    let options = options();
    let mut group = criterion.benchmark_group("f7/context");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_caller_owned", |bench| {
        bench.iter_batched(
            || (a.clone(), b.clone()),
            |(a, b)| contract_in(black_box(&a), black_box(&b), &options, &context),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    benches,
    bench_plain,
    bench_context,
    bench_plain_zipup,
    bench_context_zipup,
    bench_plain_fit,
    bench_context_fit,
    bench_plain_naive,
    bench_context_naive
);
criterion_main!(benches);
