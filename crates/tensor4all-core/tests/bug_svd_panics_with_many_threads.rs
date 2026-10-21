//! Reproducer: with the default features, `svd_with_in` panics on a 256 x 128 matrix when the
//! CPU context has 8 or more threads.
//!
//! The panic is in faer: tenferro 74aa7fa reaches `t4a-faer =0.24.6` through `tlinalg =0.3.1`,
//! and that release panics in the Householder apply of rectangular matrices with 8 or more
//! threads (tensor4all/faer#25; the gate of tensor4all/faer#12 is in t4a-faer 0.24.7, which
//! `tlinalg` 0.4.1 and later pin).
//!
//! The matrix is the one of tensor4all/faer#25: a(i, j) = ((7 i + 13 j) mod 17) / 17 + delta_ij.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tensor4all_core::svd::{svd_with_in, SvdOptions};
use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor, SvdTruncationPolicy};
use tensor4all_tensorbackend::CpuExecutionContext;

/// Whether the SVD of the m x n matrix above returns (`true`) or panics (`false`) on a CPU
/// context with `threads` threads.
fn svd_returns(m: usize, n: usize, threads: usize) -> bool {
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(threads).unwrap(),
    )));
    let (i, j) = (DynIndex::new_dyn(m), DynIndex::new_dyn(n));
    let mut a = vec![0.0_f64; m * n]; // column-major
    for c in 0..n {
        for r in 0..m {
            a[r + m * c] = ((7 * r + 13 * c) % 17) as f64 / 17.0 + if r == c { 1.0 } else { 0.0 };
        }
    }
    let t = IdxTensor::from_dense_in(&context, vec![i.clone(), j], a).unwrap();
    let options = SvdOptions::new().with_policy(SvdTruncationPolicy::new(0.0));
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        svd_with_in::<f64>(&t, std::slice::from_ref(&i), &options, &context).unwrap();
    }))
    .is_ok()
}

#[test]
#[ignore = "t4a-faer 0.24.6 panics in the Householder apply with 8 or more threads"]
fn svd_of_a_rectangular_matrix_with_16_threads() {
    assert!(svd_returns(256, 128, 16), "256 x 128, 16 threads: panicked");
    assert!(svd_returns(128, 256, 16), "128 x 256, 16 threads: panicked");
}

/// Control: the same matrices with 4 threads.
#[test]
fn svd_of_a_rectangular_matrix_with_4_threads() {
    assert!(svd_returns(256, 128, 4), "256 x 128, 4 threads: panicked");
    assert!(svd_returns(128, 256, 4), "128 x 256, 4 threads: panicked");
}

