//! Benchmark matrix for structured `select_indices` (#737).
//!
//! Both structured selection paths scale with the output tensor size and used
//! to allocate per output element. `typed_select_class` exercises the path
//! where no selected class is kept (payload-only rewrite); `dense_select_kept`
//! exercises the path where a selected class is also a kept axis (the former
//! full-slice clone plus `Vec<Option<usize>>` conversion).

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_tensorbackend::Storage;

/// Build a copy-class structured tensor with axis classes `[0, 1, 0]` and payload
/// shape `[n, m]`, returning the tensor and its three indices.
fn structured_copy_tensor(n: usize, m: usize) -> (IdxTensor, DynIndex, DynIndex, DynIndex) {
    let i = DynIndex::new_dyn(n);
    let j = DynIndex::new_dyn(m);
    let k = DynIndex::new_dyn(n);
    let payload: Vec<f64> = (0..n * m).map(|x| x as f64).collect();
    let storage =
        Storage::new_structured(payload, vec![n, m], vec![1, n as isize], vec![0, 1, 0]).unwrap();
    let tensor =
        IdxTensor::from_storage(vec![i.clone(), j.clone(), k.clone()], Arc::new(storage)).unwrap();
    (tensor, i, j, k)
}

fn bench_structured_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("structured_select");
    for (label, n, m) in [("small", 32usize, 32usize), ("medium", 128, 128)] {
        let (tensor, _i, j, k) = structured_copy_tensor(n, m);
        let typed = std::slice::from_ref(&j);
        let dense = std::slice::from_ref(&k);
        group.bench_with_input(
            BenchmarkId::new("typed_select_class", label),
            &(n, m),
            |b, _| {
                b.iter(|| {
                    black_box(&tensor)
                        .select_indices(black_box(typed), black_box(&[m / 2]))
                        .unwrap()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("dense_select_kept_class", label),
            &(n, m),
            |b, _| {
                b.iter(|| {
                    black_box(&tensor)
                        .select_indices(black_box(dense), black_box(&[n / 2]))
                        .unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_structured_select);
criterion_main!(benches);
