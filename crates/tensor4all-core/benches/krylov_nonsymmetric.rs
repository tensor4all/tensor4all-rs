//! BiCGSTAB versus restarted GMRES on a nonsymmetric dense system.
//!
//! The operator is the column-major tridiagonal matrix of a 1D
//! convection-diffusion stencil with a shifted, diagonally dominant diagonal,
//! `3 x_i - 1.5 x_{i-1} - 0.5 x_{i+1}`, applied by dense `contract`. Both solvers start from zero with
//! `rtol = 1e-10` relative to `||b||`.
//!
//! Before timing, each size prints the solver's reported `iterations`, the
//! operator applications it used, and its final relative true residual.
//! BiCGSTAB reports cycles and GMRES reports inner steps, so compare the two by
//! operator applications.
//!
//! ```text
//! cargo bench -p tensor4all-core --bench krylov_nonsymmetric
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor4all_core::krylov::{bicgstab, gmres, BiCgStabOptions, GmresOptions};
use tensor4all_core::{contract, DynIndex, IdxTensor};

const SIZES: [usize; 2] = [64, 512];
const RTOL: f64 = 1e-10;

struct System {
    i: DynIndex,
    input: DynIndex,
    matrix: IdxTensor,
    rhs: IdxTensor,
    zero: IdxTensor,
}

fn system(n: usize) -> System {
    let i = DynIndex::new_dyn(n);
    let input = i.prime();
    let mut data = vec![0.0_f64; n * n];
    for col in 0..n {
        data[col + col * n] = 3.0;
        if col > 0 {
            data[col + (col - 1) * n] = -1.5;
        }
        if col + 1 < n {
            data[col + (col + 1) * n] = -0.5;
        }
    }
    let matrix = IdxTensor::from_dense(vec![i.clone(), input.clone()], data).expect("valid matrix");
    let rhs_data: Vec<f64> = (0..n).map(|k| ((k % 7) as f64 - 3.0) / 3.0).collect();
    let rhs = IdxTensor::from_dense(vec![i.clone()], rhs_data).expect("valid rhs");
    let zero = IdxTensor::zeros::<f64>(vec![i.clone()]).expect("valid zeros");
    System {
        i,
        input,
        matrix,
        rhs,
        zero,
    }
}

fn apply(system: &System, x: &IdxTensor) -> anyhow::Result<IdxTensor> {
    Ok(contract(&[
        &system.matrix,
        &x.replaceind(&system.i, &system.input)?,
    ])?)
}

fn gmres_options() -> GmresOptions {
    GmresOptions {
        rtol: RTOL,
        ..GmresOptions::default()
    }
}

fn bicgstab_options() -> BiCgStabOptions {
    BiCgStabOptions {
        rtol: RTOL,
        ..BiCgStabOptions::default()
    }
}

fn report(system: &System, n: usize) {
    let relative = |x: &IdxTensor| {
        let residual = system
            .rhs
            .sub(&apply(system, x).expect("apply"))
            .expect("sub");
        residual.norm().expect("norm") / system.rhs.norm().expect("norm")
    };
    let calls = AtomicUsize::new(0);
    let counted = |x: &IdxTensor| {
        calls.fetch_add(1, Ordering::Relaxed);
        apply(system, x)
    };
    let result = bicgstab(counted, &system.rhs, &system.zero, &bicgstab_options())
        .expect("bicgstab converges");
    assert!(result.converged);
    eprintln!(
        "n = {n:4}  bicgstab: {:3} cycles, {:4} applications, relative residual {:.1e}",
        result.iterations,
        calls.swap(0, Ordering::Relaxed),
        relative(&result.solution)
    );
    let result =
        gmres(counted, &system.rhs, &system.zero, &gmres_options()).expect("gmres converges");
    assert!(result.converged);
    eprintln!(
        "n = {n:4}  gmres:    {:3} steps,  {:4} applications, relative residual {:.1e}",
        result.iterations,
        calls.load(Ordering::Relaxed),
        relative(&result.solution)
    );
}

fn bench_krylov(c: &mut Criterion) {
    let mut group = c.benchmark_group("krylov_nonsymmetric");
    for n in SIZES {
        let system = system(n);
        report(&system, n);
        group.bench_with_input(BenchmarkId::new("bicgstab", n), &n, |bencher, _| {
            bencher.iter(|| {
                black_box(
                    bicgstab(
                        |x: &IdxTensor| apply(&system, x),
                        &system.rhs,
                        &system.zero,
                        &bicgstab_options(),
                    )
                    .expect("bicgstab"),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("gmres", n), &n, |bencher, _| {
            bencher.iter(|| {
                black_box(
                    gmres(
                        |x: &IdxTensor| apply(&system, x),
                        &system.rhs,
                        &system.zero,
                        &gmres_options(),
                    )
                    .expect("gmres"),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_krylov);
criterion_main!(benches);
