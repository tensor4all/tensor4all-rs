//! Binary `direct_sum` of two chain cores (tensor4all/tensor4all-rs#1107).
//!
//! Each operand is a complex core `(left, site, right)` of bond dimension `D`
//! and site dimension 2; the two bonds are paired and the site is common, the
//! shape a `TreeTN::add` of two chains sums at every bulk node. In result order
//! `(site, left, right)` neither operand is contiguous along the first axis.
//!
//! Run on one core:
//!
//! ```text
//! taskset -c 11 env RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
//!     cargo bench -p tensor4all-core --bench direct_sum
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;
use tensor4all_core::{direct_sum, DynIndex, IdxTensor};

/// Bond dimensions, from per-call overhead (1) to copy-bound (160).
const BONDS: [usize; 5] = [1, 4, 10, 40, 160];

fn core(left: &DynIndex, site: &DynIndex, right: &DynIndex, bond: usize, seed: f64) -> IdxTensor {
    let data: Vec<Complex64> = (0..bond * 2 * bond)
        .map(|k| Complex64::new(seed + k as f64, 0.5))
        .collect();
    IdxTensor::from_dense(vec![left.clone(), site.clone(), right.clone()], data)
        .expect("valid core")
}

fn bench_direct_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_sum_chain_cores");
    let site = DynIndex::new_dyn(2);
    for bond in BONDS {
        let (l0, r0) = (DynIndex::new_dyn(bond), DynIndex::new_dyn(bond));
        let (l1, r1) = (DynIndex::new_dyn(bond), DynIndex::new_dyn(bond));
        let a = core(&l0, &site, &r0, bond, 1.0);
        let b = core(&l1, &site, &r1, bond, 2.0);
        let pairs = [(l0, l1), (r0, r1)];
        group.bench_with_input(BenchmarkId::from_parameter(bond), &bond, |bencher, _| {
            bencher.iter(|| black_box(direct_sum(&a, &b, &pairs).expect("direct sum")))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_direct_sum);
criterion_main!(benches);
