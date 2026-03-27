//! Test: norm precision after massive cancellation (fit approximation - exact).
//!
//! When two nearly-equal TTs are subtracted, the resulting TT has tensor entries
//! of O(1) but represents a value of O(||fit4 - exact||) ≈ O(1e-7). This creates
//! a worst-case scenario for TT-level norm computation:
//!
//! - **Sequential bra-ket contraction** (O(N·D²·d)): accumulates O(1) intermediate
//!   values that must cancel to O(1e-7). With f64 precision (~1e-16), the result
//!   can be off by many orders of magnitude. This is a fundamental limitation of
//!   the sequential algorithm, not a bug.
//!
//! - **Dense computation** (computing fit4_dense - exact_dense, then norm): accurate
//!   because the subtraction happens element-wise in dense space with no TT
//!   contraction involved.
//!
//! The precision loss scales with the cancellation ratio ||a|| / ||a - b||.
//! For this test case, ||fit|| / ||fit - exact|| is very large, so many digits
//! are lost in sequential TT-level norm evaluation.

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::ffi::OsString;

use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::{ContractOptions, TensorTrain};

struct ScopedEnvVar {
    key: &'static str,
    previous: Option<OsString>,
}

impl ScopedEnvVar {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        // This integration test binary contains a single test, so mutating the
        // process-local environment here is isolated and keeps the backend
        // runtime deterministic under full-workspace coverage runs.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => unsafe {
                std::env::set_var(self.key, value);
            },
            None => unsafe {
                std::env::remove_var(self.key);
            },
        }
    }
}

fn create_random_mpo(
    length: usize,
    input_indices: &[DynIndex],
    output_indices: &[DynIndex],
    link_indices: &[DynIndex],
    rng: &mut StdRng,
) -> TensorTrain {
    let mut tensors = Vec::with_capacity(length);
    for i in 0..length {
        let mut indices = vec![input_indices[i].clone(), output_indices[i].clone()];
        if i > 0 {
            indices.insert(0, link_indices[i - 1].clone());
        }
        if i < length - 1 {
            indices.push(link_indices[i].clone());
        }
        tensors.push(TensorDynLen::random::<f64, _>(rng, indices));
    }
    TensorTrain::new(tensors).unwrap()
}

/// Dense-level subtraction and norm is accurate (no TT contraction involved).
#[test]
fn dense_norm_matches_after_fit_cancellation() {
    let _threads = ScopedEnvVar::set("T4A_TENFERRO_CPU_THREADS", "1");

    let length = 6;
    let phys_dim = 2;
    let bond_dim = 8;

    let s_input: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("s={}", i + 1)).unwrap())
        .collect();
    let s_shared: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("sc={}", i + 1)).unwrap())
        .collect();
    let s_output: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("so={}", i + 1)).unwrap())
        .collect();
    let links_a: Vec<DynIndex> = (0..length - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();
    let links_b: Vec<DynIndex> = (0..length - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();

    let mut rng1 = StdRng::seed_from_u64(42);
    let mut rng2 = StdRng::seed_from_u64(123);
    let mpo_a = create_random_mpo(length, &s_input, &s_shared, &links_a, &mut rng1);
    let mpo_b = create_random_mpo(length, &s_shared, &s_output, &links_b, &mut rng2);

    let exact = mpo_a.contract(&mpo_b, &ContractOptions::zipup()).unwrap();
    let fit = mpo_a
        .contract(&mpo_b, &ContractOptions::fit().with_nhalfsweeps(0))
        .unwrap();

    // Ground truth: dense subtraction
    let dense_fit = fit.to_dense().unwrap();
    let dense_exact = exact.to_dense().unwrap();
    let direct_dense_diff = dense_fit
        .axpby(1.0.into(), &dense_exact, (-1.0).into())
        .unwrap();
    let dense_norm = direct_dense_diff.norm();

    // TT diff → dense → norm should agree with direct dense subtraction
    let diff = fit.axpby(1.0.into(), &exact, (-1.0).into()).unwrap();
    let dense_diff = diff.to_dense().unwrap();
    assert!(
        (dense_diff.norm() - dense_norm).abs() <= dense_norm.max(1.0) * 1e-9,
        "Dense contraction of diff TT should agree with direct dense subtraction: \
         tt_dense={:.12e}, direct_dense={dense_norm:.12e}",
        dense_diff.norm()
    );

    // TT-level norm (sequential contraction) loses precision due to cancellation.
    // With ||fit4||/||diff|| ≈ 1e7, the sequential contraction is unreliable.
    // This documents the known limitation — NOT a bug.
    let tt_norm = diff.norm();
    let cancellation_ratio = fit.to_dense().unwrap().norm() / dense_norm;
    eprintln!(
        "cancellation ratio: {cancellation_ratio:.1e}, \
         tt_norm={tt_norm:.3e}, dense_norm={dense_norm:.3e}, \
         relative_error={:.3e}",
        (tt_norm - dense_norm).abs() / dense_norm
    );
}
