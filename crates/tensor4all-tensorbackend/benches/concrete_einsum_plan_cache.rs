//! Concrete einsum plan-cache churn benchmark for issue #891.
//!
//! Run:
//! `cargo bench -p tensor4all-tensorbackend --bench concrete_einsum_plan_cache`
//!
//! The timed unit is one full cycle over 400 distinct `A[i,j] B[j,k] C[k,l]`
//! complex shapes. A miss prepares a `ConcreteEinsumPlan`, which is the fixed
//! per-call cost the cache removes; before the sampled-LRU cache this cycle
//! re-prepared every plan because the clear-on-full cache thrashed. The
//! per-cycle preparation count is pinned by the unit regression test
//! `tenferro_bridge::tests::concrete_einsum_plan_cache_keeps_issue_891_working_set_resident`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::Complex64;
use tenferro::{Tensor as NativeTensor, TensorRead};
use tensor4all_tensorbackend::{einsum_native_tensor_reads, NativeTensorReadInput};

/// Distinct shapes in the issue #891 churn cycle.
const SIGNATURES: usize = 400;

fn complex_tensor(shape: &[usize], seed: usize) -> NativeTensor {
    let len: usize = shape.iter().product();
    let values = (0..len)
        .map(|index| Complex64::new(((index + seed) % 13) as f64 * 0.1, 0.0))
        .collect::<Vec<_>>();
    NativeTensor::from_vec_col_major(shape.to_vec(), values).expect("benchmark complex tensor")
}

fn workload() -> Vec<[NativeTensor; 3]> {
    (0..SIGNATURES)
        .map(|k| {
            let shape = [
                2 + k % 7,
                2 + (k / 7) % 7,
                2 + (k / 49) % 7,
                2 + (k / 343) % 7,
            ];
            [
                complex_tensor(&[shape[0], shape[1]], 0),
                complex_tensor(&[shape[1], shape[2]], 1),
                complex_tensor(&[shape[2], shape[3]], 2),
            ]
        })
        .collect()
}

fn run_cycle(tensors: &[[NativeTensor; 3]]) -> usize {
    let mut result_shape = 0;
    for t in tensors {
        let reads = [
            NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&t[0])),
            NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&t[1])),
            NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&t[2])),
        ];
        let output = einsum_native_tensor_reads(
            &[
                (&reads[0], &[0, 1]),
                (&reads[1], &[1, 2]),
                (&reads[2], &[2, 3]),
            ],
            &[0, 3],
        )
        .expect("benchmark contraction should succeed");
        result_shape = output.shape().len();
    }
    result_shape
}

fn bench_concrete_plan_cache_churn(c: &mut Criterion) {
    let tensors = workload();
    assert_eq!(run_cycle(&tensors), 2);

    let mut group = c.benchmark_group("concrete_einsum_plan_cache");
    group.sample_size(10);
    group.bench_function("churn_400_shapes", |b| {
        b.iter(|| black_box(run_cycle(black_box(&tensors))));
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3));
    targets = bench_concrete_plan_cache_churn
}
criterion_main!(benches);
