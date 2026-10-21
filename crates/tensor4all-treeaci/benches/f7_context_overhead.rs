//! Caller-owned-context overhead for tree-valued elementwise ACI (issue #859 F7).
//!
//! `tree_elementwise_batched_with_rng_in` sends the local update's session-bound
//! numerics (the per-edge LUCI factorizations and the candidate-matrix products)
//! through a caller-owned `CpuExecutionContext` instead of the process-global
//! one. This benchmark compares the two routes on one explicit thread so the
//! comparison is a route-for-route check, not a thread-count benchmark.
//!
//! # Thread environment
//!
//! Both the caller-owned context (`CpuBackend::with_threads(1)`) and the pinned
//! environment (`RAYON_NUM_THREADS`, `BLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
//! `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1) keep every backend the two
//! routes may touch on a single thread; the effective thread count of the
//! measured comparison is therefore 1.
//!
//! # Running
//!
//! ```text
//! taskset -c 0 cargo bench -p tensor4all-treeaci --bench f7_context_overhead -- --noplot
//! ```

use std::sync::Once;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_tensorbackend::CpuExecutionContext;
use tensor4all_treeaci::{
    tree_elementwise_batched, tree_elementwise_batched_with_rng_in, TreeAciOptions,
    TreeElementwiseBatch,
};
use tensor4all_treetn::TreeTN;

/// Pin every BLAS/Rayon/OMP backend the two routes may touch to one thread
/// before either route creates it. The caller-owned route also uses
/// `CpuBackend::with_threads(1)`, so the measured comparison runs at an
/// effective thread count of 1.
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

const LOCAL_DIM: usize = 2;
const N_INPUTS: usize = 2;
const TOLERANCE: f64 = 1e-8;

fn dense_tensor(indices: Vec<DynIndex>, input: usize, node: usize) -> IdxTensor {
    let dims: Vec<usize> = indices
        .iter()
        .map(tensor4all_core::IndexLike::dim)
        .collect();
    let count: usize = dims.iter().product();
    let values = (0..count)
        .map(|flat| {
            let sum: usize = dims.iter().map(|dim| (flat / dim) % 2).sum();
            let radix = 1 + 7 * node + 11 * input;
            ((2 * flat + radix * node + input) % 5) as f64 / 3.0 + 0.01 * sum as f64
        })
        .collect();
    IdxTensor::from_dense(indices, values).expect("fixture tensor")
}

/// A short chain of `n_sites` sites with a fixed bond dimension, mirroring the
/// `aci_scaling` fixture shape so the two benches speak the same language.
fn chain_inputs(n_sites: usize, chi: usize) -> (Vec<DynIndex>, Vec<TreeTN<IdxTensor, usize>>) {
    let sites: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(LOCAL_DIM)).collect();
    let links: Vec<DynIndex> = (0..n_sites - 1).map(|_| DynIndex::new_dyn(chi)).collect();
    let inputs = (0..N_INPUTS)
        .map(|input| {
            let tensors = (0..n_sites)
                .map(|node| {
                    let mut indices = vec![sites[node].clone()];
                    if node > 0 {
                        indices.push(links[node - 1].clone());
                    }
                    if node + 1 < n_sites {
                        indices.push(links[node].clone());
                    }
                    dense_tensor(indices, input, node)
                })
                .collect::<Vec<_>>();
            TreeTN::from_tensors(tensors, (0..n_sites).collect()).expect("chain fixture")
        })
        .collect();
    (sites, inputs)
}

fn multiply(
    batch: TreeElementwiseBatch<'_, f64>,
    out: &mut [f64],
) -> tensor4all_treeaci::Result<()> {
    for (point, value) in out.iter_mut().enumerate() {
        let mut product = 1.0;
        for input in 0..N_INPUTS {
            product *= batch.get(input, point)?;
        }
        *value = product;
    }
    Ok(())
}

fn options() -> TreeAciOptions<usize> {
    TreeAciOptions {
        tolerance: TOLERANCE,
        ..TreeAciOptions::default()
    }
}

fn bench_plain(criterion: &mut Criterion) {
    pin_one_thread();
    let (sites, inputs) = chain_inputs(4, 3);
    let options = options();
    let mut group = criterion.benchmark_group("f7/plain");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_global_route", |bench| {
        bench.iter_batched(
            || inputs.clone(),
            |inputs| tree_elementwise_batched(black_box(multiply), &inputs, &options),
            BatchSize::SmallInput,
        )
    });
    drop((sites, group));
}

fn bench_context(criterion: &mut Criterion) {
    pin_one_thread();
    let (_sites, inputs) = chain_inputs(4, 3);
    let options = options();
    let context =
        CpuExecutionContext::from_backend(CpuBackend::with_threads(1).expect("1T backend"));
    let mut group = criterion.benchmark_group("f7/context");
    group.measurement_time(Duration::from_millis(800));
    group.bench_function("one_thread_caller_owned", |bench| {
        bench.iter_batched(
            || (inputs.clone(), ChaCha8Rng::seed_from_u64(0)),
            |(inputs, mut rng)| {
                tree_elementwise_batched_with_rng_in(
                    black_box(multiply),
                    &inputs,
                    &options,
                    &mut rng,
                    &context,
                )
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_plain, bench_context);
criterion_main!(benches);
