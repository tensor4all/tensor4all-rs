//! The per-add error bound of [`TreeTN::pairwise_sum`], checked on the random
//! family of tensor4all/tensor4all-rs#1107's reproducer at `N = 64`.
//!
//! One truncation with a relative squared cutoff `tau` discards at most a
//! relative error `2 (R - 1) sqrt(tau)` over its `2 (R - 1)` two-site steps, so
//! the result of a reduction with `L` adds on its longest path is within
//! `(1 + 2 (R - 1) sqrt(tau))^L - 1` of the exact sum at `kappa = 1`, the
//! largest partial-sum norm over the result norm. `L` is `N - 1` for the
//! sequential chain and `ceil(log2 N)` for the balanced tree, both stated in the
//! rustdoc of `TreeTN::pairwise_sum`.
//!
//! Heavy: run with `cargo test -p tensor4all-treetn --release --test integration
//! heavy_pairwise_sum -- --ignored --nocapture`.

use std::time::Instant;

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::{PairwiseSumOrder, PerAddCutoff, TreeTN};

type Net = TreeTN<IdxTensor, usize>;

/// #1107's reproducer: `R = 12` sites of dimension 2, bond `D = 10`.
const SITES: usize = 12;
const BOND: usize = 10;
const TERMS: usize = 64;
/// The cutoff of #1107's Table 1 and Table 2.
const PER_ADD_CUTOFF: f64 = 1.0e-8;
const CENTER: usize = 0;

/// splitmix64 step on `state`; uniform on `[-1, 1)`.
fn signed(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
}

/// The random family: `TERMS` chains with independent uniform cores.
fn random_terms(sites: &[DynIndex]) -> Vec<Net> {
    (0..TERMS)
        .map(|term| {
            let mut state = 5000 + term as u64;
            let bonds: Vec<DynIndex> = (1..sites.len()).map(|_| DynIndex::new_dyn(BOND)).collect();
            let tensors: Vec<IdxTensor> = (0..sites.len())
                .map(|node| {
                    let left = if node > 0 { BOND } else { 1 };
                    let right = if node + 1 < sites.len() { BOND } else { 1 };
                    let indices: Vec<DynIndex> = [
                        (node > 0).then(|| bonds[node - 1].clone()),
                        Some(sites[node].clone()),
                        (node + 1 < sites.len()).then(|| bonds[node].clone()),
                    ]
                    .into_iter()
                    .flatten()
                    .collect();
                    // Column-major over (left, site, right).
                    let data: Vec<f64> = (0..right)
                        .flat_map(|r| (0..2).flat_map(move |s| (0..left).map(move |l| (l, s, r))))
                        .map(|_| signed(&mut state))
                        .collect();
                    IdxTensor::from_dense(indices, data).unwrap()
                })
                .collect();
            TreeTN::from_tensors(tensors, (0..sites.len()).collect()).unwrap()
        })
        .collect()
}

/// `(1 + 2 (R - 1) sqrt(tau))^L - 1`, the documented bound at `kappa = 1`.
fn documented_bound(sites: usize, per_add_cutoff: f64, levels: usize) -> f64 {
    let one_add = 2.0 * (sites as f64 - 1.0) * per_add_cutoff.sqrt();
    (1.0 + one_add).powi(levels as i32) - 1.0
}

#[test]
#[ignore = "heavy: 64 chains of bond 10; run in release"]
fn pairwise_sum_errors_stay_inside_the_documented_per_add_bound() {
    let sites: Vec<DynIndex> = (0..SITES).map(|_| DynIndex::new_dyn(2)).collect();
    let terms = random_terms(&sites);
    let refs: Vec<&Net> = terms.iter().collect();

    // The exact sum densely: 2^12 entries.
    let mut exact = vec![0.0_f64; 1 << SITES];
    for term in &terms {
        let dense = term.to_dense().unwrap().to_vec::<f64>().unwrap();
        for (target, value) in exact.iter_mut().zip(dense) {
            *target += value;
        }
    }
    let exact_norm = exact.iter().map(|value| value * value).sum::<f64>().sqrt();

    let exact_bond = TreeTN::exact_sum(&refs, tensor4all_treetn::SumEngine::Binary)
        .unwrap()
        .network
        .link_dims()
        .into_iter()
        .max()
        .unwrap();

    for (order, levels) in [
        (PairwiseSumOrder::Sequential, TERMS - 1),
        (PairwiseSumOrder::Balanced, TERMS.trailing_zeros() as usize),
    ] {
        let started = Instant::now();
        let sum = TreeTN::pairwise_sum(
            &refs,
            order,
            Some(PerAddCutoff {
                center: &CENTER,
                cutoff: PER_ADD_CUTOFF,
                max_bond_dim: None,
            }),
        )
        .unwrap();
        let elapsed = started.elapsed().as_secs_f64();

        let dense = sum.to_dense().unwrap().to_vec::<f64>().unwrap();
        let error = dense
            .iter()
            .zip(&exact)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
            / exact_norm;
        let bound = documented_bound(SITES, PER_ADD_CUTOFF, levels);
        let bond = sum.link_dims().into_iter().max().unwrap();
        println!("{order:?}: error {error:e}, bound {bound:e}, bond {bond}, {elapsed:.2} s");

        // The run must actually truncate, or the bound says nothing.
        assert!(bond < exact_bond, "{order:?}: bond {bond} is not truncated");
        assert!(
            error <= bound,
            "{order:?}: relative error {error:e} exceeds the documented bound {bound:e}"
        );
    }
}

#[test]
#[ignore = "heavy: 64 chains of bond 10; run in release"]
fn pairwise_fit_reaches_exact_then_truncate_accuracy() {
    use tensor4all_core::SvdTruncationPolicy;
    use tensor4all_treetn::{FitOptions, SumEngine, TruncationOptions};

    let sites: Vec<_> = (0..SITES).map(|_| DynIndex::new_dyn(2)).collect();
    let terms = random_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();
    let mut exact = terms[0].to_dense().unwrap();
    for term in &terms[1..] {
        exact = exact.add(&term.to_dense().unwrap()).unwrap();
    }
    let policy = SvdTruncationPolicy::new(PER_ADD_CUTOFF)
        .with_relative()
        .with_squared_values()
        .with_discarded_tail_sum();
    let reference = TreeTN::exact_sum(&refs, SumEngine::Nary)
        .unwrap()
        .network
        .truncate(
            [CENTER],
            TruncationOptions::default().with_svd_policy(policy),
        )
        .unwrap();
    let result = TreeTN::pairwise_sum_fit(
        &terms,
        PairwiseSumOrder::Sequential,
        PerAddCutoff {
            center: &CENTER,
            cutoff: PER_ADD_CUTOFF,
            max_bond_dim: None,
        },
        FitOptions::new(1).with_svd_policy(policy),
    )
    .unwrap();
    let reference_error = reference
        .to_dense()
        .unwrap()
        .sub(&exact)
        .unwrap()
        .maxabs()
        .unwrap();
    let fitted_error = result
        .network
        .to_dense()
        .unwrap()
        .sub(&exact)
        .unwrap()
        .maxabs()
        .unwrap();
    let roundoff = f64::EPSILON * exact.maxabs().unwrap();
    println!("exact-then-truncate max error {reference_error:e}; fitted {fitted_error:e}; roundoff scale {roundoff:e}");
    assert!(
        fitted_error <= 10.0 * reference_error.max(roundoff),
        "fit error {fitted_error:e}, reference {reference_error:e}"
    );
    assert_eq!(result.truncations.len(), 2 * (SITES - 1));
}
