//! Isolated native einsum benchmarks.
//!
//! Run:
//! `cargo test --release -p tensor4all-tensorbackend --test bench_einsum_native -- --ignored --nocapture --test-threads=1`

use std::hint::black_box;
use std::time::{Duration, Instant};

use tensor4all_tensorbackend::{
    dense_native_tensor_from_col_major, einsum_native_tensors, print_and_reset_contract_profile,
};

const PHYS_DIM: usize = 2;
const BOND_DIM: usize = 16;

fn make_data(dims: &[usize], offset: usize) -> Vec<f64> {
    let len: usize = dims.iter().product();
    (0..len)
        .map(|i| (((i + offset) * 17 + 3) % 31) as f64 / 31.0 - 0.5)
        .collect()
}

fn dense_tensor(dims: &[usize], offset: usize) -> tenferro::Tensor {
    dense_native_tensor_from_col_major(&make_data(dims, offset), dims).unwrap()
}

fn time_best_of<R>(label: &str, repeats: usize, mut f: impl FnMut() -> R) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..3 {
        let started = Instant::now();
        for _ in 0..repeats {
            black_box(f());
        }
        best = best.min(started.elapsed());
    }

    let per_call_us = best.as_secs_f64() * 1e6 / repeats as f64;
    eprintln!(
        "  {label:<30} total={:.3}s  per_call={per_call_us:.3}us",
        best.as_secs_f64()
    );
    best
}

#[cfg(feature = "einsum-dispatch-profile")]
fn reset_dispatch_profile() {
    tenferro_einsum::print_and_reset_profile();
    print_and_reset_contract_profile();
}

#[cfg(not(feature = "einsum-dispatch-profile"))]
fn reset_dispatch_profile() {
    print_and_reset_contract_profile();
}

#[cfg(feature = "einsum-dispatch-profile")]
fn print_dispatch_profile(label: &str) {
    eprintln!("  dispatch profile for {label}:");
    tenferro_einsum::print_and_reset_profile();
    print_and_reset_contract_profile();
}

#[cfg(not(feature = "einsum-dispatch-profile"))]
fn print_dispatch_profile(label: &str) {
    eprintln!("  dispatch profile for {label}:");
    print_and_reset_contract_profile();
}

fn run_dispatch_profile<R>(label: &str, repeats: usize, mut f: impl FnMut() -> R) {
    reset_dispatch_profile();
    for _ in 0..repeats {
        black_box(f());
    }
    print_dispatch_profile(label);
}

#[test]
#[ignore = "benchmark"]
fn bench_native_einsum_arity() {
    eprintln!("\n=== native einsum benchmark (bd={BOND_DIM}, d={PHYS_DIM}) ===");

    let a = dense_tensor(&[PHYS_DIM, BOND_DIM], 0);
    let b = dense_tensor(&[PHYS_DIM, BOND_DIM], 1);
    let c = dense_tensor(&[PHYS_DIM, BOND_DIM], 2);
    let d = dense_tensor(&[PHYS_DIM, BOND_DIM], 3);
    let lhs = dense_tensor(&[BOND_DIM, PHYS_DIM], 4);
    let rhs = dense_tensor(&[PHYS_DIM, BOND_DIM], 5);

    assert_eq!(
        einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[1, 2])], &[0, 2])
            .unwrap()
            .dims(),
        &[BOND_DIM, BOND_DIM]
    );
    assert_eq!(
        einsum_native_tensors(&[(&a, &[0, 1]), (&b, &[0, 2]), (&c, &[0, 3])], &[1, 2, 3])
            .unwrap()
            .dims(),
        &[BOND_DIM, BOND_DIM, BOND_DIM]
    );
    assert_eq!(
        einsum_native_tensors(
            &[(&a, &[0, 1]), (&b, &[0, 2]), (&c, &[0, 3]), (&d, &[0, 4])],
            &[1, 2, 3, 4]
        )
        .unwrap()
        .dims(),
        &[BOND_DIM, BOND_DIM, BOND_DIM, BOND_DIM]
    );

    let binary = time_best_of("binary matmul", 5_000, || {
        einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[1, 2])], &[0, 2]).unwrap()
    });

    let ternary_generic = time_best_of("ternary star generic", 3_000, || {
        einsum_native_tensors(&[(&a, &[0, 1]), (&b, &[0, 2]), (&c, &[0, 3])], &[1, 2, 3]).unwrap()
    });
    let ternary_pairwise = time_best_of("ternary star pairwise", 3_000, || {
        let ab = einsum_native_tensors(&[(&a, &[0, 1]), (&b, &[0, 2])], &[0, 1, 2]).unwrap();
        einsum_native_tensors(&[(&ab, &[0, 1, 2]), (&c, &[0, 3])], &[1, 2, 3]).unwrap()
    });

    let quaternary_generic = time_best_of("quaternary star generic", 300, || {
        einsum_native_tensors(
            &[(&a, &[0, 1]), (&b, &[0, 2]), (&c, &[0, 3]), (&d, &[0, 4])],
            &[1, 2, 3, 4],
        )
        .unwrap()
    });
    let quaternary_pairwise = time_best_of("quaternary star pairwise", 300, || {
        let ab = einsum_native_tensors(&[(&a, &[0, 1]), (&b, &[0, 2])], &[0, 1, 2]).unwrap();
        let abc =
            einsum_native_tensors(&[(&ab, &[0, 1, 2]), (&c, &[0, 3])], &[0, 1, 2, 3]).unwrap();
        einsum_native_tensors(&[(&abc, &[0, 1, 2, 3]), (&d, &[0, 4])], &[1, 2, 3, 4]).unwrap()
    });

    eprintln!(
        "\n  ratio ternary generic/pairwise   = {:.2}x",
        ternary_generic.as_secs_f64() / ternary_pairwise.as_secs_f64()
    );
    eprintln!(
        "  ratio quaternary generic/pairwise = {:.2}x",
        quaternary_generic.as_secs_f64() / quaternary_pairwise.as_secs_f64()
    );
    eprintln!(
        "  ratio ternary generic/binary      = {:.2}x",
        ternary_generic.as_secs_f64() / binary.as_secs_f64()
    );
}

#[test]
#[ignore = "benchmark"]
fn bench_native_einsum_fit_patterns() {
    eprintln!("\n=== native einsum fit-pattern benchmark ===");

    let env3_a = dense_tensor(&[16, 8, 8], 10);
    let env3_b = dense_tensor(&[8, 2, 2, 8], 11);
    let env3_c = dense_tensor(&[8, 2, 2, 8], 12);

    let env4_a = dense_tensor(&[8, 2, 2, 8], 20);
    let env4_b = dense_tensor(&[8, 2, 2, 8], 21);
    let env4_c = dense_tensor(&[2, 2, 16, 16], 22);
    let env4_d = dense_tensor(&[8, 8, 16], 23);

    run_dispatch_profile("fit env 3-way generic", 200, || {
        einsum_native_tensors(
            &[
                (&env3_a, &[3, 0, 1]),
                (&env3_b, &[0, 4, 2, 5]),
                (&env3_c, &[1, 2, 6, 7]),
            ],
            &[3, 4, 5, 6, 7],
        )
        .unwrap()
    });
    run_dispatch_profile("fit env 3-way pairwise", 200, || {
        let xa = einsum_native_tensors(
            &[(&env3_a, &[3, 0, 1]), (&env3_b, &[0, 4, 2, 5])],
            &[3, 1, 4, 2, 5],
        )
        .unwrap();
        einsum_native_tensors(
            &[(&xa, &[3, 1, 4, 2, 5]), (&env3_c, &[1, 2, 6, 7])],
            &[3, 4, 5, 6, 7],
        )
        .unwrap()
    });
    run_dispatch_profile("fit env 4-way generic", 100, || {
        einsum_native_tensors(
            &[
                (&env4_a, &[6, 1, 0, 2]),
                (&env4_b, &[7, 0, 3, 4]),
                (&env4_c, &[1, 3, 5, 8]),
                (&env4_d, &[2, 4, 5]),
            ],
            &[6, 7, 8],
        )
        .unwrap()
    });
    run_dispatch_profile("fit env 4-way pairwise A-D-B", 100, || {
        let ad = einsum_native_tensors(
            &[(&env4_a, &[6, 1, 0, 2]), (&env4_d, &[2, 4, 5])],
            &[6, 1, 0, 4, 5],
        )
        .unwrap();
        let adb = einsum_native_tensors(
            &[(&ad, &[6, 1, 0, 4, 5]), (&env4_b, &[7, 0, 3, 4])],
            &[6, 1, 5, 7, 3],
        )
        .unwrap();
        einsum_native_tensors(
            &[(&adb, &[6, 1, 5, 7, 3]), (&env4_c, &[1, 3, 5, 8])],
            &[6, 7, 8],
        )
        .unwrap()
    });

    let env3_generic = time_best_of("fit env 3-way generic", 2_000, || {
        einsum_native_tensors(
            &[
                (&env3_a, &[3, 0, 1]),
                (&env3_b, &[0, 4, 2, 5]),
                (&env3_c, &[1, 2, 6, 7]),
            ],
            &[3, 4, 5, 6, 7],
        )
        .unwrap()
    });
    let env3_pairwise = time_best_of("fit env 3-way pairwise", 2_000, || {
        let xa = einsum_native_tensors(
            &[(&env3_a, &[3, 0, 1]), (&env3_b, &[0, 4, 2, 5])],
            &[3, 1, 4, 2, 5],
        )
        .unwrap();
        einsum_native_tensors(
            &[(&xa, &[3, 1, 4, 2, 5]), (&env3_c, &[1, 2, 6, 7])],
            &[3, 4, 5, 6, 7],
        )
        .unwrap()
    });

    let env4_generic = time_best_of("fit env 4-way generic", 600, || {
        einsum_native_tensors(
            &[
                (&env4_a, &[6, 1, 0, 2]),
                (&env4_b, &[7, 0, 3, 4]),
                (&env4_c, &[1, 3, 5, 8]),
                (&env4_d, &[2, 4, 5]),
            ],
            &[6, 7, 8],
        )
        .unwrap()
    });
    let env4_pairwise_adb = time_best_of("fit env 4-way pairwise A-D-B", 600, || {
        let ad = einsum_native_tensors(
            &[(&env4_a, &[6, 1, 0, 2]), (&env4_d, &[2, 4, 5])],
            &[6, 1, 0, 4, 5],
        )
        .unwrap();
        let adb = einsum_native_tensors(
            &[(&ad, &[6, 1, 0, 4, 5]), (&env4_b, &[7, 0, 3, 4])],
            &[6, 1, 5, 7, 3],
        )
        .unwrap();
        einsum_native_tensors(
            &[(&adb, &[6, 1, 5, 7, 3]), (&env4_c, &[1, 3, 5, 8])],
            &[6, 7, 8],
        )
        .unwrap()
    });
    let env4_pairwise_cda = time_best_of("fit env 4-way pairwise C-D-A", 600, || {
        let cd = einsum_native_tensors(
            &[(&env4_c, &[1, 3, 5, 8]), (&env4_d, &[2, 4, 5])],
            &[1, 3, 8, 2, 4],
        )
        .unwrap();
        let cda = einsum_native_tensors(
            &[(&cd, &[1, 3, 8, 2, 4]), (&env4_a, &[6, 1, 0, 2])],
            &[3, 8, 4, 6, 0],
        )
        .unwrap();
        einsum_native_tensors(
            &[(&cda, &[3, 8, 4, 6, 0]), (&env4_b, &[7, 0, 3, 4])],
            &[6, 7, 8],
        )
        .unwrap()
    });

    eprintln!(
        "\n  ratio fit env 3-way generic/pairwise     = {:.2}x",
        env3_generic.as_secs_f64() / env3_pairwise.as_secs_f64()
    );
    eprintln!(
        "  ratio fit env 4-way generic/(A-D-B)      = {:.2}x",
        env4_generic.as_secs_f64() / env4_pairwise_adb.as_secs_f64()
    );
    eprintln!(
        "  ratio fit env 4-way generic/(C-D-A)      = {:.2}x",
        env4_generic.as_secs_f64() / env4_pairwise_cda.as_secs_f64()
    );
}
