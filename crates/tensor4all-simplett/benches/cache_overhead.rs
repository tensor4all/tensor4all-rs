//! Benchmark to measure overhead of TTCache::evaluate_many
//!
//! Tests robustness of auto_split heuristic with different data split positions.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tensor4all_simplett::cache::TTCache;
use tensor4all_simplett::tensortrain::SimpleTensorTrain;
use tensor4all_simplett::types::{tensor3_zeros, MultiIndex, Tensor3, Tensor3Ops};

/// Generate TCI-like indices with a specific split position
fn generate_tci_like_indices(
    n_left: usize,
    n_right: usize,
    n_sites: usize,
    local_dim: usize,
    split: usize,
    seed: u64,
) -> Vec<MultiIndex> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let left_parts: Vec<Vec<usize>> = (0..n_left)
        .map(|_| (0..split).map(|_| rng.random_range(0..local_dim)).collect())
        .collect();

    let right_parts: Vec<Vec<usize>> = (0..n_right)
        .map(|_| {
            (0..n_sites - split)
                .map(|_| rng.random_range(0..local_dim))
                .collect()
        })
        .collect();

    let mut indices = Vec::with_capacity(n_left * n_right);
    for left in &left_parts {
        for right in &right_parts {
            let mut idx = left.clone();
            idx.extend(right.iter().cloned());
            indices.push(idx);
        }
    }
    indices
}

/// Create a TT with specified bond dimension
fn create_tt_with_bond_dim(
    n_sites: usize,
    local_dim: usize,
    bond_dim: usize,
) -> SimpleTensorTrain<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut tensors: Vec<Tensor3<f64>> = Vec::with_capacity(n_sites);

    for i in 0..n_sites {
        let left_dim = if i == 0 { 1 } else { bond_dim };
        let right_dim = if i == n_sites - 1 { 1 } else { bond_dim };

        let mut t: Tensor3<f64> = tensor3_zeros(left_dim, local_dim, right_dim).unwrap();
        for l in 0..left_dim {
            for s in 0..local_dim {
                for r in 0..right_dim {
                    t.set3(l, s, r, rng.random::<f64>());
                }
            }
        }
        tensors.push(t);
    }

    SimpleTensorTrain::new(tensors).unwrap()
}

/// Benchmark split position robustness
///
/// Tests auto_split vs fixed correct/wrong positions for given data split.
fn bench_split_robustness(
    c: &mut Criterion,
    group_name: &str,
    data_split: usize, // Where data is actually split
    correct_split: usize,
    wrong_split: usize,
) {
    const N_SITES: usize = 100;
    const LOCAL_DIM: usize = 2;
    const N_LEFT: usize = 50;
    const N_RIGHT: usize = 50;
    const BOND_DIM: usize = 50;

    let tt = create_tt_with_bond_dim(N_SITES, LOCAL_DIM, BOND_DIM);
    let indices = generate_tci_like_indices(N_LEFT, N_RIGHT, N_SITES, LOCAL_DIM, data_split, 123);

    let mut group = c.benchmark_group(group_name);
    group.sample_size(20);

    // Auto split (should find the correct position)
    group.bench_function("auto", |b| {
        b.iter(|| {
            let mut cache = TTCache::new(&tt);
            cache.evaluate_many(black_box(&indices), None).unwrap()
        })
    });

    // Correct split position
    group.bench_function("fixed_correct", |b| {
        b.iter(|| {
            let mut cache = TTCache::new(&tt);
            cache
                .evaluate_many(black_box(&indices), Some(correct_split))
                .unwrap()
        })
    });

    // Wrong split position
    group.bench_function("fixed_wrong", |b| {
        b.iter(|| {
            let mut cache = TTCache::new(&tt);
            cache
                .evaluate_many(black_box(&indices), Some(wrong_split))
                .unwrap()
        })
    });

    group.finish();
}

/// Data split at 1/2 (middle): correct=1/2, wrong=1/4
fn bench_data_split_half(c: &mut Criterion) {
    bench_split_robustness(c, "data_split_1_2", 50, 50, 25);
}

/// Data split at 1/4: correct=1/4, wrong=1/2
fn bench_data_split_quarter(c: &mut Criterion) {
    bench_split_robustness(c, "data_split_1_4", 25, 25, 50);
}

/// Overhead measurement with rank-1 TT (pure overhead, minimal computation)
fn bench_overhead(c: &mut Criterion) {
    const N_SITES: usize = 100;
    const LOCAL_DIM: usize = 2;

    let tt = SimpleTensorTrain::<f64>::constant(&vec![LOCAL_DIM; N_SITES], 1.0);

    let mut group = c.benchmark_group("overhead");
    group.sample_size(20);

    for (n_left, n_right) in [(10, 10), (50, 50), (100, 100)] {
        let indices =
            generate_tci_like_indices(n_left, n_right, N_SITES, LOCAL_DIM, N_SITES / 2, 42);
        let n_total = indices.len();

        group.bench_with_input(
            BenchmarkId::new(format!("{}x{}", n_left, n_right), n_total),
            &indices,
            |b, indices| {
                b.iter(|| {
                    let mut cache = TTCache::new(&tt);
                    cache.evaluate_many(black_box(indices), None).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Isolate key overhead at several widths, ranks, batch sizes and cache states.
/// Cold timing excludes cache construction/drop, but includes lazy key metadata.
fn bench_cache_keys(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_keys");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_millis(500));

    for (n_sites, bond_dim, n_side) in [
        (32, 1, 8),
        (100, 1, 8),
        (100, 1, 32),
        (100, 8, 32),
        (300, 1, 8),
        (300, 8, 8),
    ] {
        // Dense cores with a known product value, independent of the bond rank.
        let tensors = (0..n_sites)
            .map(|site| {
                let left = if site == 0 { 1 } else { bond_dim };
                let right = if site + 1 == n_sites { 1 } else { bond_dim };
                let mut tensor = tensor3_zeros(left, 2, right).unwrap();
                for l in 0..left {
                    for s in 0..2 {
                        for r in 0..right {
                            tensor.set3(l, s, r, (1.0 + s as f64 / 1024.0) / right as f64);
                        }
                    }
                }
                tensor
            })
            .collect();
        let tt = SimpleTensorTrain::<f64>::new(tensors).unwrap();
        let split = n_sites / 2;
        let indices = generate_tci_like_indices(n_side, n_side, n_sites, 2, split, 627);
        let case = format!("n{n_sites}_chi{bond_dim}_batch{}", indices.len());
        let mut cache = TTCache::new(&tt);
        let values = cache.evaluate_many(&indices, Some(split)).unwrap();
        for (idx, value) in indices.iter().zip(&values) {
            let expected = (1025.0_f64 / 1024.0).powi(idx.iter().sum::<usize>() as i32);
            assert!((value - expected).abs() < 1e-12);
        }
        assert_eq!(cache.evaluate_many(&indices, None).unwrap(), values);
        let left = &indices[0][..split];
        let right = &indices[0][split..];
        let left_value =
            (1025.0_f64 / 1024.0).powi(left.iter().sum::<usize>() as i32) / bond_dim as f64;
        let right_value = (1025.0_f64 / 1024.0).powi(right.iter().sum::<usize>() as i32);
        assert!(cache
            .evaluate_left(left)
            .unwrap()
            .iter()
            .all(|v| (v - left_value).abs() < 1e-12));
        assert!(cache
            .evaluate_right(right)
            .unwrap()
            .iter()
            .all(|v| (v - right_value).abs() < 1e-12));

        group.bench_function(BenchmarkId::new("cold_fixed", &case), |b| {
            b.iter_batched_ref(
                || TTCache::new(&tt),
                |cache| {
                    black_box(
                        cache
                            .evaluate_many(black_box(&indices), Some(split))
                            .unwrap(),
                    )
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(BenchmarkId::new("warm_fixed", &case), |b| {
            b.iter(|| {
                black_box(
                    cache
                        .evaluate_many(black_box(&indices), Some(split))
                        .unwrap(),
                )
            });
        });
        group.bench_function(BenchmarkId::new("warm_auto", &case), |b| {
            b.iter(|| black_box(cache.evaluate_many(black_box(&indices), None).unwrap()));
        });
        group.bench_function(BenchmarkId::new("left_hit", &case), |b| {
            b.iter(|| black_box(cache.evaluate_left(black_box(left)).unwrap()));
        });
        group.bench_function(BenchmarkId::new("right_hit", &case), |b| {
            b.iter(|| black_box(cache.evaluate_right(black_box(right)).unwrap()));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cache_keys,
    bench_overhead,
    bench_data_split_half,
    bench_data_split_quarter
);
criterion_main!(benches);
