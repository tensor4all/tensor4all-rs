//! Regression test: TensorTrain::norm() is exponentially slow for large tensor trains.
//!
//! After the tenferro refactor (8798c98), `inner()` delegates to `TreeTN::inner()`
//! which uses `contract_naive`. This contracts all nodes without exploiting the
//! chain structure, causing cost to scale exponentially with the number of sites.
//!
//! For a TT with N sites of physical dimension d and bond dimension D,
//! the efficient algorithm costs O(N * D^2 * d), completing in microseconds.
//! The current implementation scales as O(d^N), making it unusable for N >= ~25.
//!
//! Measured on Apple M4 Max (--release), bond dim = 2, physical dim = 2:
//!
//!   | N sites | before 8798c98 | after 8798c98 (main) |
//!   |---------|----------------|----------------------|
//!   |       4 |       < 0.01s  |             < 0.01s  |
//!   |      20 |         0.005s |               0.008s |
//!   |      25 |         0.006s |               0.18s  |
//!   |      40 |        ~0.01s  |             > 60s    |
//!   |      90 |         0.015s |        OOM (SIGKILL) |
//!
//! Root cause: `TreeTN::inner()` calls `contract_naive` instead of
//! efficient sequential bra-ket contraction along the chain.

use tensor4all_core::defaults::tensordynlen::TensorDynLen;
use tensor4all_core::DynIndex;
use tensor4all_itensorlike::TensorTrain;

/// Create a TT with `n_sites` sites, each of physical dimension 2,
/// and bond dimension 2.
fn make_tt(n_sites: usize) -> TensorTrain {
    let mut tensors = Vec::with_capacity(n_sites);
    for k in 0..n_sites {
        let site_idx = DynIndex::new_dyn_with_tag(2, &format!("s={}", k + 1)).unwrap();
        if k == 0 {
            let bond_r = DynIndex::new_dyn(2);
            let t =
                TensorDynLen::from_dense(vec![site_idx, bond_r], vec![1.0, 0.5, 0.3, 1.0]).unwrap();
            tensors.push(t);
        } else if k == n_sites - 1 {
            let bond_l = tensors[k - 1].indices().last().unwrap().clone();
            let t =
                TensorDynLen::from_dense(vec![bond_l, site_idx], vec![1.0, 0.2, 0.7, 1.0]).unwrap();
            tensors.push(t);
        } else {
            let bond_l = tensors[k - 1].indices().last().unwrap().clone();
            let bond_r = DynIndex::new_dyn(2);
            let t = TensorDynLen::from_dense(
                vec![bond_l, site_idx, bond_r],
                vec![1.0, 0.0, 0.5, 0.3, 0.0, 1.0, 0.2, 0.8],
            )
            .unwrap();
            tensors.push(t);
        }
    }
    TensorTrain::new(tensors).expect("Failed to create TensorTrain")
}

/// Sanity check: norm() on a small TT returns a positive finite value.
#[test]
fn test_norm_small_tt_works() {
    let tt = make_tt(4);
    let n = tt.norm();
    assert!(n > 0.0, "norm should be positive, got {n}");
    assert!(n.is_finite(), "norm should be finite, got {n}");
}

/// Regression test: norm() on a 25-site TT must complete in under 0.1s.
/// The efficient O(N*D^2*d) algorithm achieves this easily.
#[test]
fn test_norm_25_site_tt_should_be_fast() {
    let tt = make_tt(25);
    let start = std::time::Instant::now();
    let n = tt.norm();
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("25 sites: norm={n:.6e}, elapsed={elapsed:.3}s");
    assert!(n > 0.0, "norm should be positive, got {n}");
    assert!(n.is_finite(), "norm should be finite, got {n}");
    assert!(
        elapsed < 0.1,
        "norm() on 25-site TT with bond dim 2 took {elapsed:.3}s; \
         efficient implementation should take < 0.01s"
    );
}

/// Regression test: norm() on a 90-site TT must not OOM.
/// The efficient algorithm needs O(90 * 4 * 2) = 720 operations.
#[test]
fn test_norm_large_tt_no_oom() {
    let tt = make_tt(90);
    assert_eq!(tt.len(), 90);
    assert_eq!(tt.maxbonddim(), 2);

    let start = std::time::Instant::now();
    let n = tt.norm();
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("90 sites: norm={n:.6e}, elapsed={elapsed:.3}s");
    assert!(n > 0.0, "norm should be positive, got {n}");
    assert!(n.is_finite(), "norm should be finite, got {n}");
    assert!(
        elapsed < 1.0,
        "norm() on 90-site TT with bond dim 2 took {elapsed:.1}s; \
         efficient implementation should take < 0.1s"
    );
}
