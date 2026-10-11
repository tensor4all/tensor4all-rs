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
//! The difference between `compatibility` and `held_session` therefore covers compiled-graph trace and
//! runtime work, operand duplication, the removed `2N - 1` matrix/tensor conversions and the different
//! execution paths together, and it is a comparison of the two *routes*, not of session holding alone:
//! both explicit arms hold the session, and there is no scoped-explicit arm here. This bench does not claim
//! to isolate any single cost; `direct_session` bounds what the TT-level wrapper adds on top of the same
//! session route. Each arm's output is checked against the compatibility route before anything is timed.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
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
            let mut tensor = tensor3_zeros(left, 4, right);
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
    // Force and verify one worker on both routes before anything is measured.
    // SAFETY: the bench is single-threaded here and no other thread reads the environment yet.
    unsafe { std::env::set_var("RAYON_NUM_THREADS", "1") };
    // The measured backend, verified directly rather than through a second context.
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

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_inner_product
}
criterion_main!(benches);
