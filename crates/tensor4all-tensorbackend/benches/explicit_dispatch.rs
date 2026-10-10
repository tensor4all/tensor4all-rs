//! Explicit frontend dispatch and bridge cost (#859 B1).
//!
//! Compares, on an explicit one-worker backend pinned to one CPU, the three ways a caller
//! reaches the same operation through this crate:
//!
//! * `compatibility` — the process-global convenience entry, one session entry per call;
//! * `explicit_scoped` — `CpuExecutionContext::with_concrete_session`, one session entry per
//!   call on a caller-supplied backend;
//! * `explicit_held` — `explicit::HeldSession`, one session entry for the whole stage;
//! * `bridge` — `explicit::lift` plus `explicit::detach`, the named tracking boundary, whose
//!   cost is reported separately from the kernel numbers rather than counted as zero.
//!
//! Every arm validates its result outside the timed region. This harness measures dispatch
//! and bridge cost, not kernel throughput.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro::Tensor;
use tenferro_cpu::CpuBackend;
use tensor4all_tensorbackend::{explicit, qr_native_tensor, CpuExecutionContext};

/// Matrices per arm.
const SIZES: &[usize] = &[2, 16];

fn context() -> CpuExecutionContext {
    CpuExecutionContext::from_backend(CpuBackend::with_threads(1).expect("one-worker backend"))
}

fn matrix(size: usize) -> Tensor {
    let len = size * size;
    Tensor::from_vec_col_major(vec![size, size], (1..=len).map(|v| v as f64).collect())
        .expect("benchmark matrix")
}

fn bench_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("explicit_dispatch_qr");

    for &size in SIZES {
        let a = matrix(size);

        group.bench_with_input(BenchmarkId::new("compatibility", size), &size, |b, _| {
            black_box(qr_native_tensor(&a).expect("compatibility QR"));
            b.iter(|| black_box(qr_native_tensor(&a).expect("compatibility QR")));
        });

        group.bench_with_input(BenchmarkId::new("explicit_scoped", size), &size, |b, _| {
            let context = context();
            black_box(
                context
                    .with_concrete_session(|session| session.qr(&a))
                    .expect("session entry")
                    .expect("scoped QR"),
            );
            b.iter(|| {
                black_box(
                    context
                        .with_concrete_session(|session| session.qr(&a))
                        .expect("session entry")
                        .expect("scoped QR"),
                )
            });
        });

        group.bench_with_input(BenchmarkId::new("explicit_held", size), &size, |b, _| {
            let context = context();
            let backend: CpuBackend = context.with_backend(|backend| backend.clone());
            let mut held = explicit::HeldSession::open(&backend).expect("held session");
            black_box(held.with_session(|view| view.qr(&a)).expect("held QR"));
            b.iter(|| black_box(held.with_session(|view| view.qr(&a)).expect("held QR")));
            held.close().expect("affinity restores");
        });
    }

    group.finish();
}

fn bench_bridge(c: &mut Criterion) {
    let mut group = c.benchmark_group("explicit_bridge");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("lift_detach", size), &size, |b, _| {
            let context = context();
            black_box(
                explicit::detach(&explicit::lift(&context, matrix(size)).expect("lift"))
                    .expect("detach"),
            );
            b.iter(|| {
                black_box(
                    explicit::detach(&explicit::lift(&context, matrix(size)).expect("lift"))
                        .expect("detach"),
                )
            });
        });
    }

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5))
        .sample_size(100)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_qr, bench_bridge
}
criterion_main!(benches);
