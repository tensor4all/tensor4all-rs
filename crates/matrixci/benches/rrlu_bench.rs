use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::prelude::*;
use matrixci::util::{from_vec2d, Matrix};
use matrixci::{rrlu_inplace, RrLUOptions};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate a random f64 matrix as matrixci::Matrix
fn random_matrix(n: usize, m: usize, seed: u64) -> Matrix<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let data: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..m).map(|_| rng.random::<f64>()).collect())
        .collect();
    from_vec2d(data)
}

/// Generate a random faer Mat<f64>
fn random_faer_matrix(n: usize, m: usize, seed: u64) -> Mat<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Mat::from_fn(n, m, |_, _| rng.random::<f64>())
}

fn bench_rrlu(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrlu_full_rank");

    for &size in &[10, 50, 100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("rrlu_inplace", size), &size, |b, &n| {
            b.iter_batched(
                || random_matrix(n, n, 42),
                |mut m| {
                    let opts = RrLUOptions {
                        max_rank: n,
                        rel_tol: 0.0,
                        abs_tol: 0.0,
                        ..Default::default()
                    };
                    rrlu_inplace(&mut m, Some(opts)).unwrap();
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("faer_lu_fullpiv", size), &size, |b, &n| {
            b.iter_batched(
                || random_faer_matrix(n, n, 42),
                |m| {
                    m.full_piv_lu();
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_rrlu);
criterion_main!(benches);
