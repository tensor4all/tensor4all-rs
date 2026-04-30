use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tenferro_tensor::{cpu::CpuBackend, Tensor};
use tensor4all_tcicore::matrixluci::{
    DenseLuKernel, DenseMatrixSource, PivotKernel, PivotKernelOptions,
};

fn random_column_major(nrows: usize, ncols: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = vec![0.0; nrows * ncols];
    for col in 0..ncols {
        for row in 0..nrows {
            data[row + nrows * col] = rng.random::<f64>();
        }
    }
    data
}

fn bench_dense_vs_tenferro(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_no_truncation_vs_tenferro");
    let kernel = DenseLuKernel;
    let options = PivotKernelOptions::no_truncation();

    for &size in &[32usize, 64, 100, 128] {
        let data = random_column_major(size, size, 42);

        group.bench_with_input(BenchmarkId::new("matrixluci", size), &size, |b, &n| {
            b.iter(|| {
                let src = DenseMatrixSource::from_column_major(&data, n, n);
                black_box(kernel.factorize(&src, &options).unwrap());
            });
        });

        let mut backend = CpuBackend::new();
        group.bench_with_input(
            BenchmarkId::new("tenferro_full_piv_lu", size),
            &size,
            |b, &n| {
                b.iter(|| {
                    let mat = Tensor::from_vec(vec![n, n], data.clone());
                    black_box(mat.full_piv_lu(&mut backend).unwrap());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dense_vs_tenferro);
criterion_main!(benches);
