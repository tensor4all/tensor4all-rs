//! F2 (#859 / #510): the TT inner product of `tensor4all-simplett` on a held explicit session.
//!
//! Three arms, and the boundary each one times:
//!
//! | arm | what it runs | timed boundary |
//! | --- | --- | --- |
//! | `compatibility` | `SimpleTensorTrain::inner_product` | the whole call: the compatibility route, which traces and runs a compiled graph on the process-global context, plus the crate's per-site matrix/tensor conversions and per-operand duplication |
//! | `held_session` | `inner_product_with_session` with the session opened once by the caller | the whole call inside an already held session; session entry is not repeated |
//! | `direct_session` | the same site loop, inlined here, calling `Session::einsum_reads` directly | the loop itself, without the TT-level method |
//!
//! Both explicit arms run on a context built with one worker; the compatibility arm runs on the
//! process-global context, whose worker count this bench forces to one before anything else initializes it
//! and verifies before measuring. The new route has **no** input-adaptation boundary to time: both operands
//! are already concrete `TypedTensor` values inside the train.
//!
//! Two further groups answer the rest of F2's recorded identity. `simplett_inner_product_threads` repeats
//! the held arm on explicit contexts with 1, 2 and 4 workers: the compatibility arm cannot join it, because
//! the process-global context fixes its worker count when it first initializes. `simplett_inner_product_
//! first_call` reports the very first call on a freshly built context next to the steady-state number for
//! the same size, so "cold" is a measured first call rather than a criterion label.
//!
//! The difference between `compatibility` and `held_session` therefore covers compiled-graph trace and
//! runtime work, operand duplication, the removed `2N - 1` matrix/tensor conversions and the different
//! execution paths together, and it is a comparison of the two *routes*, not of session holding alone:
//! both explicit arms hold the session, and there is no scoped-explicit arm here. This bench does not claim
//! to isolate any single cost; `direct_session` bounds what the TT-level wrapper adds on top of the same
//! session route. Each arm's output is checked against the compatibility route before anything is timed.

use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use tenferro_cpu::CpuBackend;
use tenferro_tensor::TensorRead;
use tensor4all_simplett::types::{tensor3_zeros, Tensor3, Tensor3Ops};
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, TTScalar};
use tensor4all_tensorbackend::default_cpu_execution_context;
use tensor4all_tensorbackend::explicit::{HeldSession, Session};

/// One train: `sites` sites of physical dimension 4 with bond dimension `bond` (unit boundary bonds).
fn train(sites: usize, bond: usize, offset: usize) -> SimpleTensorTrain<f64> {
    let tensors: Vec<Tensor3<f64>> = (0..sites)
        .map(|site| {
            let left = if site == 0 { 1 } else { bond };
            let right = if site + 1 == sites { 1 } else { bond };
            let mut tensor = tensor3_zeros(left, 4, right).unwrap();
            for l in 0..left {
                for s in 0..4 {
                    for r in 0..right {
                        let raw = site * 17 + l * 5 + s * 3 + r * 7 + offset * 23;
                        tensor.set3(l, s, r, (raw % 13) as f64 - 6.0);
                    }
                }
            }
            tensor
        })
        .collect();
    SimpleTensorTrain::new(tensors).expect("a well-formed train")
}

/// The site loop of `inner_product_with_session`, without the TT-level method.
fn direct_session<T: TTScalar>(
    a: &SimpleTensorTrain<T>,
    b: &SimpleTensorTrain<T>,
    session: &mut Session<'_>,
) -> T {
    let sites = a.len();
    let a0 = a.site_tensor(0);
    let b0 = b.site_tensor(0);
    let mut environment = session
        .einsum_reads(
            &[T::tensor_read(a0.as_inner()), T::tensor_read(b0.as_inner())],
            &[&[0, 1, 2], &[0, 1, 3]],
            &[2, 3],
        )
        .expect("first site");
    for site in 1..sites {
        let a_site = a.site_tensor(site);
        let b_site = b.site_tensor(site);
        environment = session
            .einsum_reads(
                &[
                    TensorRead::from_tensor(&environment),
                    T::tensor_read(a_site.as_inner()),
                    T::tensor_read(b_site.as_inner()),
                ],
                &[&[0, 1], &[0, 2, 3], &[1, 2, 4]],
                &[3, 4],
            )
            .expect("later site");
    }
    *environment
        .as_slice::<T>()
        .expect("host result")
        .first()
        .expect("1x1 environment")
}

fn bench_inner_product(c: &mut Criterion) {
    // The worker count was forced in `main` before the process-global context initialized; this arm verifies
    // both counts rather than setting them, and measures one backend directly.
    let backend = CpuBackend::with_threads(1).expect("one-worker CPU backend");
    let explicit_threads = backend.num_threads();
    let compatibility_threads = default_cpu_execution_context()
        .with_backend(|context_backend| context_backend.num_threads());
    assert_eq!(
        (compatibility_threads, explicit_threads),
        (1, 1),
        "both arms must run on exactly one worker"
    );
    println!(
        "inner_product_bench: compatibility workers={compatibility_threads}, explicit workers={explicit_threads}"
    );

    let cases = [(4usize, 4usize), (8, 4), (16, 4)];

    // The compatibility arm runs with no explicit session held: the explicit routes own admission for the
    // duration of a held session, and entering the compatibility route from inside one is rejected by
    // design ("an execution is already active on this thread"). The two families are therefore measured in
    // separate groups rather than interleaved.
    {
        let mut group = c.benchmark_group("simplett_inner_product_compatibility");
        for (sites, bond) in cases {
            let a = train(sites, bond, 0);
            let b = train(sites, bond, 5);
            group.throughput(Throughput::Elements((sites * 4) as u64));
            group.bench_with_input(
                BenchmarkId::new("compatibility", format!("{sites}x4_bond{bond}")),
                &(&a, &b),
                |bencher, (a, b)| {
                    bencher.iter(|| black_box(a.inner_product(b).expect("compatibility")))
                },
            );
        }
        group.finish();
    }

    // Every arm is validated outside the timed region, and compatibility is validated before the held
    // session opens: entering the compatibility route from inside a held session is rejected by design.
    let expected: Vec<f64> = cases
        .iter()
        .map(|&(sites, bond)| {
            train(sites, bond, 0)
                .inner_product(&train(sites, bond, 5))
                .expect("compatibility")
        })
        .collect();

    // One session held across the explicit group's arms, so they measure the computation, not entry.
    let mut held = HeldSession::open(&backend).expect("held explicit session");
    {
        let mut group = c.benchmark_group("simplett_inner_product_session");
        for (case, (sites, bond)) in cases.iter().enumerate() {
            let a = train(*sites, *bond, 0);
            let b = train(*sites, *bond, 5);
            let expected = expected[case];
            group.throughput(Throughput::Elements((sites * 4) as u64));

            // Outside the timed region: the explicit arms must agree with the compatibility value computed
            // before the session opened, so a wrong direct loop cannot pass as an attractive timing.
            let held_value = held
                .with_session(|session| a.inner_product_with_session(&b, session))
                .expect("session inner product");
            let direct_value = held.with_session(|session| direct_session(&a, &b, session));
            assert!(
                (held_value - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{sites}x4_bond{bond}: held {held_value} != compatibility {expected}"
            );
            assert!(
                (direct_value - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{sites}x4_bond{bond}: direct {direct_value} != compatibility {expected}"
            );

            group.bench_with_input(
                BenchmarkId::new("held_session", format!("{sites}x4_bond{bond}")),
                &(&a, &b),
                |bencher, (a, b)| {
                    bencher.iter(|| {
                        black_box(
                            held.with_session(|session| a.inner_product_with_session(b, session))
                                .expect("session inner product"),
                        )
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("direct_session", format!("{sites}x4_bond{bond}")),
                &(&a, &b),
                |bencher, (a, b)| {
                    bencher.iter(|| {
                        black_box(held.with_session(|session| direct_session(a, b, session)))
                    })
                },
            );
        }
        group.finish();
    }
    held.close().expect("close the held session");
}

/// Repeat the held arm on explicit contexts with different worker counts.
///
/// The compatibility arm is absent by construction: the process-global context takes its worker count when
/// it first initializes, so a multi-thread comparison against it is not available in this bench.
fn bench_threads(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplett_inner_product_threads");
    let (sites, bond) = (16usize, 4usize);
    let a = train(sites, bond, 0);
    let b = train(sites, bond, 5);
    let expected = a.inner_product(&b).expect("compatibility");
    group.throughput(Throughput::Elements((sites * 4) as u64));

    for workers in [1usize, 2, 4] {
        let backend = CpuBackend::with_threads(workers).expect("CPU backend");
        assert_eq!(backend.num_threads(), workers, "the measured worker count");
        let mut held = HeldSession::open(&backend).expect("held explicit session");
        // Validate outside timing, then measure the same trains at this worker count.
        let value = held
            .with_session(|session| a.inner_product_with_session(&b, session))
            .expect("session inner product");
        assert!(
            (value - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "{workers} workers: {value} != compatibility {expected}"
        );
        group.bench_with_input(
            BenchmarkId::new("held_session", format!("{workers}w")),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    black_box(
                        held.with_session(|session| a.inner_product_with_session(b, session))
                            .expect("session inner product"),
                    )
                })
            },
        );
        held.close().expect("close the held session");
    }
    group.finish();
}

/// One observation of the first call on a freshly built context, next to a call made after three warm-ups.
///
/// Neither number is a criterion sample: the first call is taken once, with backend construction and session
/// opening outside the timer and process-wide allocator, code and cache state left as they are, so this is a
/// first call on a fresh backend rather than process-cold execution. The numbers are printed rather than
/// benchmarked, because a criterion group would repeat the call and average the first one away.
fn bench_first_call() {
    println!("inner_product_first_call: first call on a fresh context against the steady state");
    for (sites, bond) in [(4usize, 4usize), (16, 4)] {
        let a = train(sites, bond, 0);
        let b = train(sites, bond, 5);
        let expected = a.inner_product(&b).expect("compatibility");

        let backend = CpuBackend::with_threads(1).expect("one-worker CPU backend");
        let mut held = HeldSession::open(&backend).expect("held explicit session");
        let start = std::time::Instant::now();
        let first = held
            .with_session(|session| a.inner_product_with_session(&b, session))
            .expect("first session inner product");
        let first_elapsed = start.elapsed();
        black_box(first);
        assert!(
            (first - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "the first call must agree with compatibility"
        );
        // Warm the same context, then time the same call again.
        for _ in 0..3 {
            black_box(
                held.with_session(|session| a.inner_product_with_session(&b, session))
                    .expect("warm-up inner product"),
            );
        }
        let start = std::time::Instant::now();
        let warm = held
            .with_session(|session| a.inner_product_with_session(&b, session))
            .expect("warm session inner product");
        let warm_elapsed = start.elapsed();
        black_box(warm);
        assert!((warm - expected).abs() <= 1e-9 * expected.abs().max(1.0));
        held.close().expect("close the held session");

        println!(
            "inner_product_first_call: {sites}x4_bond{bond} first={first_elapsed:?} warm={warm_elapsed:?}"
        );
    }
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_inner_product, bench_threads
}

fn main() {
    // The one-worker comparison needs the process-global context to be initialized with one worker, so this
    // has to come before anything calls the compatibility route - including the first-call block below.
    // SAFETY: no worker pool exists yet at this point.
    unsafe { std::env::set_var("RAYON_NUM_THREADS", "1") };
    bench_first_call();
    benches();
    // `criterion_main!` would do this; the custom `main` has to keep the aggregate report.
    Criterion::default().configure_from_args().final_summary();
}
