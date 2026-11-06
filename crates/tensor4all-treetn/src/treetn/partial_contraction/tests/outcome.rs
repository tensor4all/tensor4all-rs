use super::*;
use std::{error::Error, num::NonZeroUsize};

use crate::contraction::{measure_tree_zipup, ContractionOutcome, ContractionReport, EdgeRank};
use num_complex::Complex32;
use tensor4all_core::TensorElement;

fn completed<V: Clone + Hash + Eq + Ord + Send + Sync + Debug>(
    outcome: ContractionOutcome<IdxTensor, V>,
) -> (TreeTN<IdxTensor, V>, ContractionReport<V>) {
    match outcome {
        ContractionOutcome::Completed { network, report } => (network, report),
        other => panic!("expected Completed, got {other:?}"),
    }
}

fn assert_dense<V: Clone + Hash + Eq + Ord + Send + Sync + Debug>(
    network: &TreeTN<IdxTensor, V>,
    expected: &IdxTensor,
    tolerance: f64,
) {
    let dense = network.to_dense().unwrap();
    let error = dense.sub(expected).unwrap().maxabs().unwrap();
    assert!(error < tolerance, "dense max-absolute residual {error:e}");
}

fn assert_report<V: Clone + Hash + Eq + Ord + Send + Sync + Debug>(
    network: &TreeTN<IdxTensor, V>,
    report: &ContractionReport<V>,
    center: &V,
) {
    let expected: Vec<_> = network
        .edges_to_canonicalize_by_names(center)
        .unwrap()
        .into_iter()
        .map(|(child, parent)| {
            let bond = network.edge_between(&child, &parent).unwrap();
            EdgeRank {
                rank: network.bond_index(bond).unwrap().dim(),
                child,
                parent,
            }
        })
        .collect();
    assert_eq!(report.edge_ranks, expected);
}

fn source_chain(error: &(dyn Error + 'static)) -> Vec<String> {
    let mut chain = vec![error.to_string()];
    let mut next = error.source();
    while let Some(error) = next {
        chain.push(error.to_string());
        next = error.source();
    }
    chain
}

struct HadamardFixture {
    a: TreeTN<IdxTensor, usize>,
    b: TreeTN<IdxTensor, usize>,
    spec: PartialContractionSpec<DynIndex>,
    expected: IdxTensor,
}

impl HadamardFixture {
    fn new<T: TensorElement>(
        branched: bool,
        real: impl Fn(f64) -> T,
        phased: impl Fn(f64) -> T,
    ) -> Self {
        let count = if branched { 4 } else { 3 };
        let left: Vec<_> = (0..count).map(|_| DynIndex::new_dyn(2)).collect();
        let right: Vec<_> = left.iter().map(DynIndex::prime).collect();
        let edges: Vec<_> = (1..count)
            .map(|n| (if branched { 0 } else { n - 1 }, n))
            .collect();
        let mut trees = Vec::new();
        for (sites, is_left) in [(&left, true), (&right, false)] {
            let bonds: Vec<_> = edges.iter().map(|_| DynIndex::new_dyn(2)).collect();
            let tensors = (0..count)
                .map(|node| {
                    let mut indices = vec![sites[node].clone()];
                    for (bond, &(u, v)) in bonds.iter().zip(&edges) {
                        if u == node || v == node {
                            indices.push(bond.clone());
                        }
                    }
                    let mut values = vec![real(0.); 1 << indices.len()];
                    let last = values.len() - 1;
                    values[0] = if node == 0 {
                        if is_left {
                            phased(1.)
                        } else {
                            real(3.)
                        }
                    } else {
                        real(1.)
                    };
                    values[last] = if node == 0 {
                        if is_left {
                            phased(2.)
                        } else {
                            real(4.)
                        }
                    } else {
                        real(1.)
                    };
                    IdxTensor::from_dense(indices, values).unwrap()
                })
                .collect();
            trees.push(TreeTN::from_tensors(tensors, (0..count).collect()).unwrap());
        }
        let mut values = vec![real(0.); 1 << count];
        values[0] = phased(3.);
        values[(1 << count) - 1] = phased(8.);
        let expected = IdxTensor::from_dense(left.clone(), values).unwrap();
        let a = trees.remove(0);
        let b = trees.remove(0);
        Self {
            a,
            b,
            expected,
            spec: PartialContractionSpec {
                contract_pairs: vec![],
                diagonal_pairs: left.into_iter().zip(right).collect(),
                output_order: None,
            },
        }
    }
}

fn methods() -> Vec<ContractionOptions> {
    let mut rsvd = ContractionOptions::zipup()
        .with_max_bond_dim(2)
        .with_factorize_alg(FactorizeAlg::RSVD);
    rsvd.rsvd_seed = 73;
    let mut rsvd_fit = rsvd.clone();
    rsvd_fit.method = ContractionMethod::Fit;
    rsvd_fit.nfullsweeps = 1;
    vec![
        ContractionOptions::zipup().with_max_bond_dim(2),
        ContractionOptions::fit()
            .with_max_bond_dim(2)
            .with_nfullsweeps(1),
        ContractionOptions::new(ContractionMethod::Naive)
            .with_max_bond_dim(2)
            .with_dense_reference_limit(4096),
        ContractionOptions::src()
            .with_max_bond_dim(2)
            .with_src_options(SrcOptions::fixed().with_seed(31)),
        rsvd,
        rsvd_fit,
    ]
}

fn check_hadamard<T: TensorElement>(
    real: impl Fn(f64) -> T + Copy,
    phased: impl Fn(f64) -> T + Copy,
    tolerance: f64,
    full_support: bool,
) {
    for branched in [false, true] {
        let fixture = HadamardFixture::new(branched, real, phased);
        for options in methods() {
            let ordinary = match partial_contract(
                &fixture.a,
                &fixture.b,
                &fixture.spec,
                &0,
                options.clone(),
            ) {
                Ok(network) => network,
                Err(error) => {
                    assert!(!full_support, "unexpected refusal: {error:?}");
                    let actual = partial_contract_with_outcome(
                        &fixture.a,
                        &fixture.b,
                        &fixture.spec,
                        &0,
                        options,
                        NonZeroUsize::new(1),
                    )
                    .unwrap_err();
                    assert_eq!(source_chain(&actual), source_chain(&error));
                    continue;
                }
            };
            assert_dense(&ordinary, &fixture.expected, tolerance);
            let (network, report) = completed(
                partial_contract_with_outcome(
                    &fixture.a,
                    &fixture.b,
                    &fixture.spec,
                    &0,
                    options.clone(),
                    None,
                )
                .unwrap(),
            );
            assert_dense(&network, &fixture.expected, tolerance);
            assert_eq!(network.node_names(), ordinary.node_names());
            assert_report(&network, &report, &0);
            for (left, right) in &fixture.spec.diagonal_pairs {
                assert!(network.external_indices().contains(left));
                assert!(!network.external_indices().contains(right));
            }
            for threshold in [1, 2, 3] {
                let outcome = partial_contract_with_outcome(
                    &fixture.a,
                    &fixture.b,
                    &fixture.spec,
                    &0,
                    options.clone(),
                    NonZeroUsize::new(threshold),
                )
                .unwrap();
                if threshold <= 2 {
                    let ContractionOutcome::Saturated { edge } = outcome else {
                        panic!("expected saturation")
                    };
                    assert_eq!(edge, report.edge_ranks[0]);
                } else {
                    let (network, actual) = completed(outcome);
                    assert_eq!(actual, report);
                    assert_dense(&network, &fixture.expected, tolerance);
                }
            }
        }
    }
}

#[test]
fn real_hadamard_outcomes_preserve_values_indices_and_method_dispatch() {
    check_hadamard::<f64>(|x| x, |x| x, 1e-10, true);
}
#[test]
fn complex_hadamard_outcomes_preserve_values_indices_and_method_dispatch() {
    check_hadamard::<Complex64>(
        |x| Complex64::new(x, 0.),
        |x| Complex64::new(x, x),
        1e-10,
        true,
    );
}
#[test]
fn single_precision_support_or_refusal_matches_partial_contract() {
    check_hadamard::<f32>(|x| x as f32, |x| x as f32, 2e-5, false);
    check_hadamard::<Complex32>(
        |x| Complex32::new(x as f32, 0.),
        |x| Complex32::new(x as f32, x as f32),
        2e-5,
        false,
    );
}

#[test]
fn public_partial_entry_skips_work_but_ordered_output_finishes_first() {
    let mut fixture = HadamardFixture::new::<f64>(true, |x| x, |x| x);
    let options = ContractionOptions::zipup().with_max_bond_dim(2);
    let (ordinary, full_count) = measure_tree_zipup(|| {
        partial_contract(&fixture.a, &fixture.b, &fixture.spec, &0, options.clone())
    });
    assert_dense(&ordinary.unwrap(), &fixture.expected, 1e-10);
    let (outcome, count) = measure_tree_zipup(|| {
        partial_contract_with_outcome(
            &fixture.a,
            &fixture.b,
            &fixture.spec,
            &0,
            options.clone(),
            NonZeroUsize::new(2),
        )
    });
    assert!(matches!(
        outcome.unwrap(),
        ContractionOutcome::Saturated { .. }
    ));
    assert_eq!((full_count, count), (3, 1));

    fixture.spec.output_order = Some(
        fixture
            .spec
            .diagonal_pairs
            .iter()
            .rev()
            .map(|(a, _)| a.clone())
            .collect(),
    );
    let (network, report) = completed(
        partial_contract_with_outcome(
            &fixture.a,
            &fixture.b,
            &fixture.spec,
            &0,
            options.clone(),
            None,
        )
        .unwrap(),
    );
    assert_dense(&network, &fixture.expected, 1e-10);
    assert_eq!(
        network.all_site_indices().unwrap().0,
        *fixture.spec.output_order.as_ref().unwrap()
    );
    assert_report(&network, &report, &0);
    let (outcome, count) = measure_tree_zipup(|| {
        partial_contract_with_outcome(
            &fixture.a,
            &fixture.b,
            &fixture.spec,
            &0,
            options,
            NonZeroUsize::new(2),
        )
    });
    let ContractionOutcome::Saturated { edge } = outcome.unwrap() else {
        panic!("expected saturation")
    };
    assert_eq!(edge, report.edge_ranks[0]);
    assert_eq!(count, full_count);
}

#[test]
fn contract_diagonal_mixed_and_outer_specs_keep_the_known_uniform_result() {
    let input = make_partial_contraction_inputs();
    for contract_pair in [false, true] {
        for diagonal_pair in [false, true] {
            let spec = PartialContractionSpec {
                contract_pairs: if contract_pair {
                    vec![(input.s_contract_a.clone(), input.s_contract_b.clone())]
                } else {
                    vec![]
                },
                diagonal_pairs: if diagonal_pair {
                    vec![(input.s_multiply_a.clone(), input.s_multiply_b.clone())]
                } else {
                    vec![]
                },
                output_order: None,
            };
            for options in methods() {
                let (network, report) = completed(
                    partial_contract_with_outcome(
                        &input.tn_a,
                        &input.tn_b,
                        &spec,
                        &"B".to_string(),
                        options,
                        None,
                    )
                    .unwrap(),
                );
                let indices = network.external_indices();
                let count = indices.iter().map(IndexLike::dim).product();
                let expected = IdxTensor::from_dense(
                    indices,
                    vec![if contract_pair { 32. } else { 16. }; count],
                )
                .unwrap();
                assert_dense(&network, &expected, 1e-10);
                assert_report(&network, &report, &"B".to_string());
            }
        }
    }
}

fn constant_star(count: usize) -> TreeTN<IdxTensor, usize> {
    let bonds: Vec<_> = (1..count).map(|_| DynIndex::new_dyn(1)).collect();
    let mut hub = bonds.clone();
    hub.push(DynIndex::new_dyn(2));
    let mut tensors = vec![IdxTensor::from_dense(hub, vec![1.; 2]).unwrap()];
    tensors.extend(
        bonds.into_iter().map(|bond| {
            IdxTensor::from_dense(vec![bond, DynIndex::new_dyn(2)], vec![1.; 2]).unwrap()
        }),
    );
    TreeTN::from_tensors(tensors, (0..count).collect()).unwrap()
}

#[test]
fn mismatched_topologies_finish_structural_alignment_before_reporting() {
    let a = constant_star(4);
    let b = constant_star(5);
    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![],
        output_order: None,
    };
    for options in [
        ContractionOptions::zipup(),
        ContractionOptions::zipup().with_mismatched_topology_dense_limit(1),
    ] {
        let (network, report) = completed(
            partial_contract_with_outcome(&a, &b, &spec, &0, options.clone(), None).unwrap(),
        );
        assert_report(&network, &report, &0);
        assert_eq!(report.edge_ranks.len(), 4);
        assert!(report.edge_ranks.iter().all(|edge| edge.rank == 1));
        let expected = IdxTensor::from_dense(network.external_indices(), vec![1.; 512]).unwrap();
        assert_dense(&network, &expected, 1e-10);
        let (outcome, count) = measure_tree_zipup(|| {
            partial_contract_with_outcome(&a, &b, &spec, &0, options, NonZeroUsize::new(1))
        });
        let ContractionOutcome::Saturated { edge } = outcome.unwrap() else {
            panic!("expected saturation")
        };
        assert_eq!(edge, report.edge_ranks[0]);
        assert_eq!(count, 4);
    }
}

#[test]
fn scalar_dense_fallback_and_its_opt_in_error_are_preserved() {
    // A center absent from the input union forces the existing structural
    // attempt to fail. The explicitly permitted scalar fallback wraps at it.
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);
    let a = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![i.clone()], vec![1., 2.]).unwrap()],
        vec![0],
    )
    .unwrap();
    let b = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![bond.clone()], vec![1.]).unwrap(),
            IdxTensor::from_dense(vec![bond, j.clone()], vec![3., 4.]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let spec = PartialContractionSpec {
        contract_pairs: vec![(i, j)],
        diagonal_pairs: vec![],
        output_order: None,
    };
    for limit in [None, Some(1), Some(4)] {
        let mut options = ContractionOptions::src().with_max_bond_dim(2);
        options.mismatched_topology_dense_limit = limit;
        let ordinary = partial_contract(&a, &b, &spec, &99, options.clone());
        let actual =
            partial_contract_with_outcome(&a, &b, &spec, &99, options, NonZeroUsize::new(1));
        match ordinary {
            Ok(network) => {
                assert_eq!(limit, Some(4));
                let (actual, report) = completed(actual.unwrap());
                assert!(report.edge_ranks.is_empty());
                let expected = IdxTensor::from_dense(vec![], vec![11.]).unwrap();
                assert_dense(&network, &expected, 1e-12);
                assert_dense(&actual, &expected, 1e-12);
            }
            Err(error) => assert_eq!(source_chain(&actual.unwrap_err()), source_chain(&error)),
        }
    }
}

#[test]
fn moving_a_contract_pair_keeps_the_output_indices_and_known_values() {
    let fixture = HadamardFixture::new::<f64>(false, |x| x, |x| x);
    let spec = PartialContractionSpec {
        contract_pairs: vec![(
            fixture.spec.diagonal_pairs[0].0.clone(),
            fixture.spec.diagonal_pairs[2].1.clone(),
        )],
        diagonal_pairs: vec![],
        output_order: None,
    };
    let indices = vec![
        fixture.spec.diagonal_pairs[1].0.clone(),
        fixture.spec.diagonal_pairs[2].0.clone(),
        fixture.spec.diagonal_pairs[0].1.clone(),
        fixture.spec.diagonal_pairs[1].1.clone(),
    ];
    let mut values = vec![0.; 16];
    values[0] = 3.;
    values[15] = 8.;
    let expected = IdxTensor::from_dense(indices, values).unwrap();
    for options in methods() {
        let ordinary =
            partial_contract(&fixture.a, &fixture.b, &spec, &0, options.clone()).unwrap();
        let (network, report) = completed(
            partial_contract_with_outcome(&fixture.a, &fixture.b, &spec, &0, options, None)
                .unwrap(),
        );
        assert_dense(&ordinary, &expected, 1e-10);
        assert_dense(&network, &expected, 1e-10);
        assert_eq!(report.edge_ranks.len(), network.edge_count());
    }
}

#[test]
fn compatibility_context_acceptance_and_errors_match_the_existing_adapter() {
    use std::sync::Arc;
    use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, ExecutionContext};
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )));
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let a = TreeTN::from_tensors(
        vec![IdxTensor::from_dense_in(&context, vec![i.clone()], vec![2., 3.]).unwrap()],
        vec![0],
    )
    .unwrap();
    let b = TreeTN::from_tensors(
        vec![IdxTensor::from_dense_in(&context, vec![j.clone()], vec![5., 7.]).unwrap()],
        vec![0],
    )
    .unwrap();
    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![(i, j)],
        output_order: None,
    };
    let ordinary = partial_contract(&a, &b, &spec, &0, ContractionOptions::zipup());
    let actual =
        partial_contract_with_outcome(&a, &b, &spec, &0, ContractionOptions::zipup(), None);
    match ordinary {
        Err(error) => assert_eq!(source_chain(&actual.unwrap_err()), source_chain(&error)),
        Ok(ordinary) => {
            let (network, report) = completed(actual.unwrap());
            assert!(report.edge_ranks.is_empty());
            assert_eq!(
                network.validate_context(&context).is_ok(),
                ordinary.validate_context(&context).is_ok()
            );
            assert_eq!(
                network.to_dense().unwrap().to_vec::<f64>().unwrap(),
                vec![10., 21.]
            );
            assert_eq!(
                ordinary.to_dense().unwrap().to_vec::<f64>().unwrap(),
                vec![10., 21.]
            );
        }
    }
}

#[test]
fn invalid_pairs_options_and_output_order_errors_are_not_hidden_by_saturation() {
    let fixture = HadamardFixture::new::<f64>(true, |x| x, |x| x);
    let mut cases = Vec::new();
    for order in [
        vec![],
        vec![DynIndex::new_dyn(2); 4],
        vec![fixture.spec.diagonal_pairs[0].0.clone(); 4],
    ] {
        let mut spec = fixture.spec.clone();
        spec.output_order = Some(order);
        cases.push((spec, ContractionOptions::zipup()));
    }
    let mut spec = fixture.spec.clone();
    spec.diagonal_pairs.push(spec.diagonal_pairs[0].clone());
    cases.push((spec, ContractionOptions::zipup()));
    let mut spec = fixture.spec.clone();
    spec.diagonal_pairs[0].0 = DynIndex::new_dyn(2);
    cases.push((spec, ContractionOptions::zipup()));
    let mut spec = fixture.spec.clone();
    spec.diagonal_pairs[0].1 = DynIndex::new_dyn(3);
    cases.push((spec, ContractionOptions::zipup()));
    cases.push((
        fixture.spec.clone(),
        ContractionOptions::zipup().with_max_bond_dim(0),
    ));
    cases.push((
        fixture.spec.clone(),
        ContractionOptions::new(ContractionMethod::Naive),
    ));
    for (spec, options) in cases {
        let expected =
            partial_contract(&fixture.a, &fixture.b, &spec, &0, options.clone()).unwrap_err();
        let actual = partial_contract_with_outcome(
            &fixture.a,
            &fixture.b,
            &spec,
            &0,
            options,
            NonZeroUsize::new(1),
        )
        .unwrap_err();
        assert_eq!(source_chain(&actual), source_chain(&expected));
    }
}

#[test]
fn diagonal_scalar_node_keeps_the_gradient_and_edgeless_threshold_rule() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense(vec![i.clone()], vec![2., 3.])
        .unwrap()
        .enable_grad()
        .unwrap();
    let a = TreeTN::from_tensors(vec![tensor.clone()], vec![0]).unwrap();
    let b = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![j.clone()], vec![5., 7.]).unwrap()],
        vec![0],
    )
    .unwrap();
    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![(i.clone(), j)],
        output_order: None,
    };
    let (network, report) = completed(
        partial_contract_with_outcome(
            &a,
            &b,
            &spec,
            &0,
            ContractionOptions::zipup(),
            NonZeroUsize::new(1),
        )
        .unwrap(),
    );
    assert!(report.edge_ranks.is_empty());
    let dense = network.to_dense().unwrap();
    assert_eq!(dense.external_indices(), vec![i]);
    assert_eq!(dense.to_vec::<f64>().unwrap(), vec![10., 21.]);
    dense.sum().unwrap().backward().unwrap();
    assert_eq!(
        tensor.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![5., 7.]
    );
}
