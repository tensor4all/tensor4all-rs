//! Regression tests for #1008 (tensor4all/faer#25): SVD and QR of rectangular
//! matrices panicked inside t4a-faer's threaded Householder apply on 8 or more
//! CPU workers. tenferro's tlinalg 0.3.1 pulled t4a-faer 0.24.6, which carries
//! the bug; the pinned t4a-faer 0.24.7 fixes it. These tests keep that
//! multi-worker path exercised, so a dependency downgrade fails here instead of
//! in a TreeTN contraction.

use std::sync::Arc;

use tensor4all_core::qr::{qr_with_in, QrOptions};
use tensor4all_core::svd::{svd_with_in, SvdOptions};
use tensor4all_core::{
    DynIndex, ExecutionContext, IdxTensor, SvdTruncationPolicy, TensorContractionLike,
};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};

/// `a(i, j) = ((7 i + 13 j) mod 17) / 17 + delta_ij`, column-major: the matrix
/// of tensor4all/faer#25.
fn rect_matrix(m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; m * n];
    for c in 0..n {
        for r in 0..m {
            a[r + m * c] = ((7 * r + 13 * c) % 17) as f64 / 17.0 + if r == c { 1.0 } else { 0.0 };
        }
    }
    a
}

fn multithreaded_context(threads: usize) -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(threads).expect("multi-worker CPU backend"),
    )))
}

/// The shapes that panicked in t4a-faer 0.24.6 at 16 workers.
const SHAPES: [(usize, usize); 3] = [(256, 128), (128, 256), (512, 256)];

fn dense_matrix(context: &ExecutionContext, m: usize, n: usize) -> (DynIndex, IdxTensor) {
    let (i, j) = (DynIndex::new_dyn(m), DynIndex::new_dyn(n));
    let t = IdxTensor::from_dense_in(context, vec![i.clone(), j], rect_matrix(m, n))
        .expect("dense matrix");
    (i, t)
}

#[test]
fn svd_of_rectangular_matrices_on_16_workers() {
    let context = multithreaded_context(16);
    let options = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-12));
    for (m, n) in SHAPES {
        let (i, t) = dense_matrix(&context, m, n);
        let (u, s, v) = svd_with_in::<f64>(&t, std::slice::from_ref(&i), &options, &context)
            .unwrap_or_else(|e| panic!("SVD of {m} x {n} on 16 workers failed: {e}"));
        // `contract_pair` matches index identities, so `v.conj()` (V^H) needs no
        // physical axis permutation.
        let reconstructed = u
            .contract_pair(&s)
            .unwrap()
            .contract_pair(&v.conj())
            .unwrap();
        assert!(
            t.isapprox(&reconstructed, 1e-10, 0.0).unwrap(),
            "SVD of {m} x {n} on 16 workers: maxabs diff = {}",
            t.sub(&reconstructed).unwrap().maxabs().unwrap()
        );
    }
}

#[test]
fn qr_of_rectangular_matrices_on_16_workers() {
    let context = multithreaded_context(16);
    let options = QrOptions::new();
    for (m, n) in SHAPES {
        let (i, t) = dense_matrix(&context, m, n);
        let (q, r) = qr_with_in::<f64>(&t, std::slice::from_ref(&i), &options, &context)
            .unwrap_or_else(|e| panic!("QR of {m} x {n} on 16 workers failed: {e}"));
        let reconstructed = q.contract_pair(&r).unwrap();
        assert!(
            t.isapprox(&reconstructed, 1e-10, 0.0).unwrap(),
            "QR of {m} x {n} on 16 workers: maxabs diff = {}",
            t.sub(&reconstructed).unwrap().maxabs().unwrap()
        );
    }
}
