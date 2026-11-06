//! SVD and QR panic on more than one CPU worker when a column block of the
//! input is exactly zero ("chunk_size must not be zero" in t4a-faer 0.24.7's
//! blocked QR, tensor4all/faer#51, fixed in t4a-faer 0.24.9).
//!
//! `cargo test -p tensor4all-core --test integration bug_zero_block_threads::` fails on main;
//! `-- --ignored --nocapture` prints the table over thread counts and shapes.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use tensor4all_core::qr::{qr_with_in, QrOptions};
use tensor4all_core::svd::{svd_with_in, SvdOptions};
use tensor4all_core::{
    DynIndex, ExecutionContext, IdxTensor, SvdTruncationPolicy, TensorContractionLike,
};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};

/// m x n, column-major, the first `zero` columns exactly 0 and the others
/// uniform in [-0.5, 0.5) from a fixed LCG: the matrix of tensor4all/faer#51.
fn zero_block_matrix(m: usize, n: usize, zero: usize) -> Vec<f64> {
    let mut s = 12345u64;
    let mut a = vec![0.0_f64; m * n];
    for c in 0..n {
        for r in 0..m {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if c >= zero {
                a[r + m * c] = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            }
        }
    }
    a
}

fn context(threads: usize) -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(threads).expect("CPU backend"),
    )))
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Qr,
    Svd,
}

/// Factorizes the matrix with `threads` workers and returns whether the factors
/// reconstruct it (`isapprox`, rtol 1e-10), or `None` if the factorization
/// panicked.
fn factorize(op: Op, m: usize, n: usize, zero: usize, threads: usize) -> Option<bool> {
    let ctx = context(threads);
    let (i, j) = (DynIndex::new_dyn(m), DynIndex::new_dyn(n));
    let t = IdxTensor::from_dense_in(&ctx, vec![i.clone(), j], zero_block_matrix(m, n, zero))
        .expect("dense matrix");
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let back = match op {
            Op::Qr => {
                let (q, r) =
                    qr_with_in::<f64>(&t, std::slice::from_ref(&i), &QrOptions::new(), &ctx)
                        .unwrap();
                q.contract_pair(&r).unwrap()
            }
            Op::Svd => {
                let options = SvdOptions::new().with_policy(SvdTruncationPolicy::new(0.0));
                let (u, s, v) =
                    svd_with_in::<f64>(&t, std::slice::from_ref(&i), &options, &ctx).unwrap();
                u.contract_pair(&s)
                    .unwrap()
                    .contract_pair(&v.conj())
                    .unwrap()
            }
        };
        t.isapprox(&back, 1e-10, 0.0).unwrap()
    }))
    .ok()
}

#[test]
fn qr_with_a_zero_column_block_on_2_workers() {
    let ok = factorize(Op::Qr, 1024, 96, 32, 2).expect("QR of 1024 x 96 on 2 workers panicked");
    assert!(ok, "QR of 1024 x 96 on 2 workers: wrong factors");
}

#[test]
fn svd_with_a_zero_column_block_on_2_workers() {
    let ok = factorize(Op::Svd, 1024, 96, 32, 2).expect("SVD of 1024 x 96 on 2 workers panicked");
    assert!(ok, "SVD of 1024 x 96 on 2 workers: wrong factors");
}

#[test]
fn the_same_matrices_on_1_worker() {
    for op in [Op::Qr, Op::Svd] {
        let ok = factorize(op, 1024, 96, 32, 1).expect("1 worker panicked");
        assert!(ok, "{op:?} of 1024 x 96 on 1 worker: wrong factors");
    }
}

#[test]
#[ignore = "prints the table; run with --ignored --nocapture"]
fn table_over_threads_and_shapes() {
    let shapes = [
        (1024, 96, 32),
        (512, 128, 32),
        (256, 128, 32),
        (256, 128, 0),
        (200, 64, 32),
    ];
    let threads = [1, 2, 4, 8, 16];
    for op in [Op::Qr, Op::Svd] {
        println!("{op:?}: m x n (zero columns) | threads {threads:?}");
        for (m, n, zero) in shapes {
            let row: Vec<&str> = threads
                .iter()
                .map(|&t| match factorize(op, m, n, zero, t) {
                    Some(true) => "ok",
                    Some(false) => "wrong",
                    None => "PANIC",
                })
                .collect();
            println!("  {m} x {n} ({zero}) | {}", row.join(" | "));
        }
    }
}
