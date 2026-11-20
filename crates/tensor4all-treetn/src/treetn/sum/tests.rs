//! `TreeTN::exact_sum`: the engines build the same network, `NaryWithin`
//! switches at its limit, and the size estimate is the built sum's.

use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, IndexLike, TensorElement};

use super::*;
use crate::{fit_sum_with_report, FitOptions};

fn diagonal_term<T: TensorElement + From<f64> + Copy>(
    site: &[DynIndex],
    values: [T; 2],
) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(2);
    TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![site[0].clone(), bond.clone()],
                vec![T::from(1.0), T::from(0.0), T::from(0.0), T::from(1.0)],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![bond, site[1].clone()],
                vec![values[0], T::from(0.0), T::from(0.0), values[1]],
            )
            .unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap()
}

#[test]
fn pairwise_fit_recovers_all_summands_after_seed_truncation() {
    let sites: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    for phase in [AnyScalar::new_real(1.0), AnyScalar::new_complex(0.0, 1.0)] {
        let term = if phase.is_complex() {
            diagonal_term(
                &sites,
                [
                    num_complex::Complex64::new(0.0, 3.0),
                    num_complex::Complex64::new(0.0, 1.0),
                ],
            )
        } else {
            diagonal_term(&sites, [3.0, 1.0])
        };
        let terms = vec![term; 3];
        for order in [PairwiseSumOrder::Sequential, PairwiseSumOrder::Balanced] {
            let result = TreeTN::pairwise_sum_fit(
                &terms,
                order,
                PerAddCutoff {
                    center: &0,
                    cutoff: 0.0,
                    max_bond_dim: Some(1),
                },
                FitOptions::new(1).with_max_bond_dim(2),
            )
            .unwrap();
            let expected = IdxTensor::from_dense(sites.clone(), vec![9.0, 0.0, 0.0, 3.0])
                .unwrap()
                .scale(phase.clone())
                .unwrap();
            assert!(
                result
                    .network
                    .to_dense()
                    .unwrap()
                    .sub(&expected)
                    .unwrap()
                    .maxabs()
                    .unwrap()
                    < 1e-12
            );
            assert_eq!(result.network.link_dims(), vec![2]);
        }
    }
}

#[test]
fn sum_fit_reports_discarded_norm_for_each_region() {
    let sites: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    for phase in [AnyScalar::new_real(1.0), AnyScalar::new_complex(0.0, 1.0)] {
        let terms = vec![diagonal_term(&sites, [3.0, 1.0])
            .scale(phase.clone())
            .unwrap()];
        let result = fit_sum_with_report(
            &terms,
            &terms[0],
            &0,
            FitOptions::new(1).with_max_bond_dim(1),
        )
        .unwrap();
        assert_eq!(result.truncations.len(), 2);
        for report in &result.truncations {
            assert!(report.region == [0, 1] || report.region == [1, 0]);
            assert!((report.discarded_norm - 1.0).abs() < 1e-12, "{report:?}");
        }
        let expected = IdxTensor::from_dense(sites.clone(), vec![3.0, 0.0, 0.0, 0.0])
            .unwrap()
            .scale(phase.clone())
            .unwrap();
        assert!(
            result
                .network
                .to_dense()
                .unwrap()
                .sub(&expected)
                .unwrap()
                .maxabs()
                .unwrap()
                < 1e-12
        );
    }
}

#[test]
fn pairwise_fit_zero_sweeps_returns_the_seed_and_one_site_has_no_regions() {
    let sites: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let terms = vec![diagonal_term(&sites, [3.0, 1.0]); 3];
    let result = TreeTN::pairwise_sum_fit(
        &terms,
        PairwiseSumOrder::Sequential,
        PerAddCutoff {
            center: &0,
            cutoff: 0.0,
            max_bond_dim: Some(1),
        },
        FitOptions::new(0).with_max_bond_dim(2),
    )
    .unwrap();
    let expected = IdxTensor::from_dense(sites, vec![9.0, 0.0, 0.0, 0.0]).unwrap();
    assert!(
        result
            .network
            .to_dense()
            .unwrap()
            .sub(&expected)
            .unwrap()
            .maxabs()
            .unwrap()
            < 1e-12
    );
    assert!(result.truncations.is_empty());

    let site = DynIndex::new_dyn(2);
    let term = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![1.0, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    let result = TreeTN::pairwise_sum_fit(
        &[term.clone(), term],
        PairwiseSumOrder::Balanced,
        PerAddCutoff {
            center: &0,
            cutoff: 0.0,
            max_bond_dim: None,
        },
        FitOptions::new(1),
    )
    .unwrap();
    assert_eq!(
        result.network.to_dense().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0, 4.0]
    );
    assert!(result.truncations.is_empty());
}

#[test]
fn pairwise_fit_rejects_invalid_seed_and_fit_options_before_shortcuts() {
    let sites: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let terms = vec![diagonal_term(&sites, [3.0, 1.0])];
    let seed = || PerAddCutoff {
        center: &0,
        cutoff: 0.0,
        max_bond_dim: None,
    };
    for options in [
        FitOptions::new(0).with_max_bond_dim(0),
        FitOptions::new(0).with_convergence_tol(f64::NAN),
    ] {
        assert!(
            TreeTN::pairwise_sum_fit(&terms, PairwiseSumOrder::Sequential, seed(), options)
                .is_err()
        );
    }
    for invalid in [
        PerAddCutoff {
            cutoff: f64::NAN,
            ..seed()
        },
        PerAddCutoff {
            max_bond_dim: Some(0),
            ..seed()
        },
        PerAddCutoff {
            center: &99,
            ..seed()
        },
    ] {
        assert!(TreeTN::pairwise_sum_fit(
            &terms,
            PairwiseSumOrder::Sequential,
            invalid,
            FitOptions::new(0)
        )
        .is_err());
    }
    assert!(TreeTN::<IdxTensor, usize>::pairwise_sum_fit(
        &[],
        PairwiseSumOrder::Sequential,
        seed(),
        FitOptions::new(0)
    )
    .is_err());
}

#[test]
fn pairwise_fit_branching_result_and_ranks_repeat() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let run = || {
        TreeTN::pairwise_sum_fit(
            &terms,
            PairwiseSumOrder::Balanced,
            PerAddCutoff {
                center: &0,
                cutoff: 1e-3,
                max_bond_dim: Some(1),
            },
            FitOptions::new(2).with_max_bond_dim(2),
        )
        .unwrap()
    };
    let first = run();
    let second = run();
    assert_eq!(first.network.link_dims(), second.network.link_dims());
    assert_eq!(first.truncations.len(), 12);
    assert_eq!(first.truncations.len(), second.truncations.len());
    let expected = TreeTN::add_many(&terms.iter().collect::<Vec<_>>())
        .unwrap()
        .to_dense()
        .unwrap();
    let dense = first.network.to_dense().unwrap();
    assert!(dense.sub(&expected).unwrap().maxabs().unwrap() < 1e-10);
    assert!(
        dense
            .sub(&second.network.to_dense().unwrap())
            .unwrap()
            .maxabs()
            .unwrap()
            < 1e-12
    );
}

/// A four-node star `1 - 0 - 2, 0 - 3` over `sites`, with bond dimensions
/// `bonds` to the leaves and values derived from `seed`.
fn star_term(sites: &[DynIndex], bonds: [usize; 3], seed: usize) -> TreeTN<IdxTensor, usize> {
    let values = |len: usize, seed: usize| -> Vec<f64> {
        (0..len)
            .map(|k| ((seed * 31 + k) as f64 * 0.618).sin())
            .collect()
    };
    let links: Vec<DynIndex> = bonds.iter().map(|&dim| DynIndex::new_dyn(dim)).collect();
    let mut tensors = vec![IdxTensor::from_dense(
        vec![
            sites[0].clone(),
            links[0].clone(),
            links[1].clone(),
            links[2].clone(),
        ],
        values(2 * bonds.iter().product::<usize>(), seed),
    )
    .unwrap()];
    for (leaf, link) in links.iter().enumerate() {
        tensors.push(
            IdxTensor::from_dense(
                vec![link.clone(), sites[leaf + 1].clone()],
                values(2 * link.dim(), seed + leaf + 1),
            )
            .unwrap(),
        );
    }
    TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap()
}

fn star_sites() -> Vec<DynIndex> {
    (0..4).map(|_| DynIndex::new_dyn(2)).collect()
}

const BONDS: [[usize; 3]; 3] = [[1, 2, 3], [2, 1, 1], [3, 2, 2]];

/// Summed bonds 6, 5, 6: center 2 * 6 * 5 * 6 elements, leaves 6 * 2, 5 * 2
/// and 6 * 2, at 16 bytes each.
const STAR_ESTIMATE: usize = (2 * 6 * 5 * 6 + 6 * 2 + 5 * 2 + 6 * 2) * 16;

fn star_terms(sites: &[DynIndex]) -> Vec<TreeTN<IdxTensor, usize>> {
    BONDS
        .iter()
        .enumerate()
        .map(|(seed, &bonds)| star_term(sites, bonds, 10 * seed))
        .collect()
}

fn assert_same_tensors(left: &TreeTN<IdxTensor, usize>, right: &TreeTN<IdxTensor, usize>) {
    for node in left.node_names() {
        let a = left.tensor(left.node_index(&node).unwrap()).unwrap();
        let b = right.tensor(right.node_index(&node).unwrap()).unwrap();
        assert_eq!(a.dims(), b.dims(), "node {node}");
        assert_eq!(
            a.to_vec::<f64>().unwrap(),
            b.to_vec::<f64>().unwrap(),
            "node {node}"
        );
    }
}

#[test]
fn exact_sum_engines_build_the_same_network() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();

    let binary = TreeTN::exact_sum(&refs, SumEngine::Binary).unwrap();
    let nary = TreeTN::exact_sum(&refs, SumEngine::Nary).unwrap();

    assert!(!binary.used_nary);
    assert!(nary.used_nary);
    assert_eq!(binary.estimated_nary_bytes, STAR_ESTIMATE);
    assert_eq!(nary.estimated_nary_bytes, STAR_ESTIMATE);
    assert_same_tensors(&binary.network, &nary.network);
}

#[test]
fn exact_sum_default_engine_is_binary() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();

    let sum = TreeTN::exact_sum(&refs, SumEngine::default()).unwrap();

    assert!(!sum.used_nary);
}

#[test]
fn nary_within_switches_exactly_at_its_limit() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();

    let at_limit = TreeTN::exact_sum(
        &refs,
        SumEngine::NaryWithin {
            limit_bytes: STAR_ESTIMATE,
        },
    )
    .unwrap();
    let below_limit = TreeTN::exact_sum(
        &refs,
        SumEngine::NaryWithin {
            limit_bytes: STAR_ESTIMATE - 1,
        },
    )
    .unwrap();

    assert!(at_limit.used_nary);
    assert!(!below_limit.used_nary);
    assert_same_tensors(&at_limit.network, &below_limit.network);
}

#[test]
fn exact_sum_of_one_term_returns_it() {
    let sites = star_sites();
    let term = star_term(&sites, [1, 2, 3], 0);

    for engine in [SumEngine::Binary, SumEngine::Nary] {
        let sum = TreeTN::exact_sum(&[&term], engine).unwrap();
        assert_same_tensors(&sum.network, &term);
        assert_eq!(sum.network.link_dims(), term.link_dims());
    }
}

#[test]
fn exact_sum_checks_the_terms_before_estimating() {
    let sites = star_sites();
    let term = star_term(&sites, [1, 1, 1], 0);
    let single = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(vec![sites[0].clone()], vec![1.0, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();

    let no_terms = TreeTN::<IdxTensor, usize>::exact_sum(&[], SumEngine::Nary).unwrap_err();
    assert!(
        format!("{no_terms:#}").contains("at least one term"),
        "{no_terms:#}"
    );
    let mismatch = TreeTN::exact_sum(&[&term, &single], SumEngine::Nary).unwrap_err();
    assert!(
        format!("{mismatch:#}").contains("different topologies"),
        "{mismatch:#}"
    );
}

#[test]
fn the_sum_estimate_matches_exact_sum_and_checks_the_terms() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();
    assert_eq!(
        TreeTN::estimate_exact_sum_bytes(&refs).unwrap(),
        STAR_ESTIMATE
    );

    let single = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(vec![sites[0].clone()], vec![1.0, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    let no_terms = TreeTN::<IdxTensor, usize>::estimate_exact_sum_bytes(&[]).unwrap_err();
    assert!(
        format!("{no_terms:#}").contains("at least one term"),
        "{no_terms:#}"
    );
    let mismatch = TreeTN::estimate_exact_sum_bytes(&[&terms[0], &single]).unwrap_err();
    assert!(
        format!("{mismatch:#}").contains("different topologies"),
        "{mismatch:#}"
    );
}

#[test]
fn balanced_engine_builds_the_same_network() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();

    let binary = TreeTN::exact_sum(&refs, SumEngine::Binary).unwrap();
    let balanced = TreeTN::exact_sum(&refs, SumEngine::Balanced).unwrap();
    let nary = TreeTN::exact_sum(&refs, SumEngine::Nary).unwrap();

    assert!(!balanced.used_nary);
    assert_eq!(balanced.estimated_nary_bytes, STAR_ESTIMATE);
    assert_same_tensors(&balanced.network, &binary.network);
    assert_same_tensors(&balanced.network, &nary.network);
}

#[test]
fn pairwise_sum_without_cutoff_equals_the_exact_engines() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();

    let sequential = TreeTN::pairwise_sum(&refs, PairwiseSumOrder::Sequential, None).unwrap();
    let balanced = TreeTN::pairwise_sum(&refs, PairwiseSumOrder::Balanced, None).unwrap();
    let nary = TreeTN::exact_sum(&refs, SumEngine::Nary).unwrap();

    assert_same_tensors(&sequential, &balanced);
    assert_same_tensors(&sequential, &nary.network);
    assert_eq!(sequential.link_dims(), balanced.link_dims());
}

#[test]
fn pairwise_sum_balanced_odd_level_carries_a_term_up() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    for count in 1..=terms.len() {
        let refs: Vec<_> = terms[..count].iter().collect();
        let balanced = TreeTN::pairwise_sum(&refs, PairwiseSumOrder::Balanced, None).unwrap();
        let sequential = TreeTN::pairwise_sum(&refs, PairwiseSumOrder::Sequential, None).unwrap();
        let nary = TreeTN::exact_sum(&refs, SumEngine::Nary).unwrap();
        assert_same_tensors(&balanced, &sequential);
        assert_same_tensors(&balanced, &nary.network);
    }
}

#[test]
fn pairwise_sum_truncates_after_every_add() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();
    let exact = TreeTN::exact_sum(&refs, SumEngine::Nary).unwrap().network;

    for order in [PairwiseSumOrder::Sequential, PairwiseSumOrder::Balanced] {
        let truncated = TreeTN::pairwise_sum(
            &refs,
            order,
            Some(PerAddCutoff {
                center: &0,
                cutoff: 1.0e-20,
                max_bond_dim: Some(2),
            }),
        )
        .unwrap();
        // The per-add cap holds on every bond of the result.
        assert!(truncated.link_dims().iter().all(|&dim| dim <= 2));
        // An unattainably small per-add cutoff still reproduces the sum to
        // round-off on this fixture.
        let exact_dense = exact.to_dense().unwrap();
        assert!(
            truncated
                .to_dense()
                .unwrap()
                .distance(&exact_dense)
                .unwrap()
                < 1.0e-10
        );
    }
}

#[test]
fn pairwise_sum_rejects_invalid_per_add_cutoffs() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();

    for cutoff in [-1.0, f64::NAN, f64::INFINITY] {
        let error = TreeTN::pairwise_sum(
            &refs,
            PairwiseSumOrder::Balanced,
            Some(PerAddCutoff {
                center: &0,
                cutoff,
                max_bond_dim: None,
            }),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("threshold"), "{error:#}");
    }

    let zero_cap = TreeTN::pairwise_sum(
        &refs,
        PairwiseSumOrder::Sequential,
        Some(PerAddCutoff {
            center: &0,
            cutoff: 1.0e-12,
            max_bond_dim: Some(0),
        }),
    )
    .unwrap_err();
    assert!(
        format!("{zero_cap:#}").contains("max_bond_dim"),
        "{zero_cap:#}"
    );

    let no_terms = TreeTN::<IdxTensor, usize>::pairwise_sum(
        &[],
        PairwiseSumOrder::Balanced,
        Some(PerAddCutoff {
            center: &0,
            cutoff: 1.0e-12,
            max_bond_dim: None,
        }),
    )
    .unwrap_err();
    assert!(
        format!("{no_terms:#}").contains("at least one term"),
        "{no_terms:#}"
    );

    for inputs in [&refs[..1], refs.as_slice()] {
        let bad_center = TreeTN::pairwise_sum(
            inputs,
            PairwiseSumOrder::Balanced,
            Some(PerAddCutoff {
                center: &99,
                cutoff: 1.0e-12,
                max_bond_dim: None,
            }),
        )
        .unwrap_err();
        assert!(matches!(
            bad_center,
            TreeTNOperationError::InvalidGraph { .. }
        ));
    }
}
