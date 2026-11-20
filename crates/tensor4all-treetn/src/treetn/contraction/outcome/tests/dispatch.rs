//! Post-hoc outcomes preserve each existing contraction dispatch.

mod early_abort;

use std::error::Error;
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::treetn::contraction::SessionLeaves;
use crate::{
    contraction::{
        contract, contract_in, contract_with_outcome, contract_with_outcome_in, ContractionOptions,
        ContractionOutcome, ContractionReport, EdgeRank,
    },
    TreeTN,
};
use num_complex::{Complex32, Complex64};
use tensor4all_core::{DynIndex, FactorizeAlg, IdxTensor, TensorElement};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, ExecutionContext};

struct Fixture {
    a: TreeTN<IdxTensor, usize>,
    b: TreeTN<IdxTensor, usize>,
    expected: IdxTensor,
}

impl Fixture {
    /// The exact product is |00...0> + 2 |11...1>, with an optional complex phase.
    /// Each shared physical index has a primed output partner with the same ID.
    fn new<T: TensorElement>(
        branched: bool,
        omit_center_output: bool,
        real: impl Fn(f64) -> T + Copy,
        phased: impl Fn(f64) -> T + Copy,
    ) -> Self {
        let count = if branched { 4 } else { 3 };
        let edges: Vec<_> = (1..count)
            .map(|node| (if branched { 0 } else { node - 1 }, node))
            .collect();
        let output_nodes: Vec<_> = (usize::from(omit_center_output)..count).collect();
        Self::with_edges(&edges, &output_nodes, real, phased)
    }

    fn with_edges<T: TensorElement>(
        edges: &[(usize, usize)],
        output_nodes: &[usize],
        real: impl Fn(f64) -> T + Copy,
        phased: impl Fn(f64) -> T + Copy,
    ) -> Self {
        let count = edges.len() + 1;
        let shared: Vec<_> = (0..count).map(|_| DynIndex::new_dyn(2)).collect();
        let outputs: Vec<_> = shared.iter().map(DynIndex::prime).collect();
        let a_bonds: Vec<_> = edges.iter().map(|_| DynIndex::new_dyn(2)).collect();
        let b_bonds: Vec<_> = edges.iter().map(|_| DynIndex::new_dyn(1)).collect();
        let mut a_tensors = Vec::new();
        let mut b_tensors = Vec::new();
        for node in 0..count {
            let mut a_indices = vec![shared[node].clone()];
            if output_nodes.contains(&node) {
                a_indices.push(outputs[node].clone());
            }
            let mut b_indices = vec![shared[node].clone()];
            for (edge, &(left, right)) in edges.iter().enumerate() {
                if left == node || right == node {
                    a_indices.push(a_bonds[edge].clone());
                    b_indices.push(b_bonds[edge].clone());
                }
            }
            let mut values = vec![real(0.); 1 << a_indices.len()];
            let last = values.len() - 1;
            values[0] = if node == 0 { phased(1.) } else { real(1.) };
            values[last] = if node == 0 { phased(2.) } else { real(1.) };
            a_tensors.push(IdxTensor::from_dense(a_indices, values).unwrap());
            b_tensors.push(IdxTensor::from_dense(b_indices, vec![real(1.); 2]).unwrap());
        }
        let output_sites: Vec<_> = outputs
            .into_iter()
            .enumerate()
            .filter_map(|(node, index)| output_nodes.contains(&node).then_some(index))
            .collect();
        let mut expected = vec![real(0.); 1 << output_sites.len()];
        let last = expected.len() - 1;
        expected[0] = phased(if last == 0 { 3. } else { 1. });
        if last != 0 {
            expected[last] = phased(2.);
        }
        Self {
            a: TreeTN::from_tensors(a_tensors, (0..count).collect()).unwrap(),
            b: TreeTN::from_tensors(b_tensors, (0..count).collect()).unwrap(),
            expected: IdxTensor::from_dense(output_sites, expected).unwrap(),
        }
    }

    fn adopt(&mut self, context: &ExecutionContext) {
        self.a.adopt_into_in(context).unwrap();
        self.b.adopt_into_in(context).unwrap();
    }
}

fn context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )))
}

fn completed(
    outcome: ContractionOutcome<IdxTensor, usize>,
) -> (TreeTN<IdxTensor, usize>, ContractionReport<usize>) {
    match outcome {
        ContractionOutcome::Completed { network, report } => (network, report),
        other => panic!("expected Completed, got {other:?}"),
    }
}

fn assert_dense(
    tree: &TreeTN<IdxTensor, usize>,
    expected: &IdxTensor,
    tol: f64,
    context: Option<&ExecutionContext>,
) {
    // The owner-level test uses the existing explicit leaf seam for a scoped
    // full contraction. Read back only that final tensor for the host oracle.
    let dense = match context {
        Some(context) => tree
            .contract_to_tensor_with_leaves(&SessionLeaves(context.clone()))
            .unwrap(),
        None => tree.to_dense().unwrap(),
    };
    let host =
        IdxTensor::from_storage(dense.indices().to_vec(), dense.to_storage().unwrap()).unwrap();
    let error = host.sub(expected).unwrap().maxabs().unwrap();
    assert!(error < tol, "known-GHZ max-absolute error {error:e}");
}

fn assert_report(
    tree: &TreeTN<IdxTensor, usize>,
    report: &ContractionReport<usize>,
    center: usize,
) {
    let root = if tree.node_index(&center).is_some() {
        center
    } else {
        assert_eq!(tree.canonical_region().len(), 1);
        *tree.canonical_region().iter().next().unwrap()
    };
    let expected: Vec<_> = tree
        .edges_to_canonicalize_by_names(&root)
        .unwrap()
        .into_iter()
        .map(|(child, parent)| EdgeRank {
            child,
            parent,
            rank: 2,
        })
        .collect();
    assert_eq!(report.edge_ranks, expected);
    assert_eq!(tree.link_dims(), vec![2; tree.edge_count()]);
}

fn methods() -> Vec<ContractionOptions> {
    vec![
        ContractionOptions::zipup().with_max_bond_dim(2),
        ContractionOptions::fit()
            .with_max_bond_dim(2)
            .with_nfullsweeps(1),
        ContractionOptions::new(crate::contraction::ContractionMethod::Naive)
            .with_max_bond_dim(2)
            .with_dense_reference_limit(4096),
        ContractionOptions::src().with_max_bond_dim(2),
    ]
}

fn check_methods<T: TensorElement>(
    real: impl Fn(f64) -> T + Copy,
    phased: impl Fn(f64) -> T + Copy,
    tol: f64,
    full_support: bool,
) {
    for branched in [false, true] {
        let fixture = Fixture::new(branched, false, real, phased);
        for options in methods() {
            let baseline = match contract(&fixture.a, &fixture.b, &0, options.clone()) {
                Ok(tree) => tree,
                Err(error) => {
                    assert!(!full_support, "unexpected baseline refusal: {error:?}");
                    let reported = contract_with_outcome(
                        &fixture.a,
                        &fixture.b,
                        &0,
                        options,
                        NonZeroUsize::new(1),
                    )
                    .unwrap_err();
                    assert_eq!(source_chain(&reported), source_chain(&error));
                    continue;
                }
            };
            let (tree, report) = completed(
                contract_with_outcome(&fixture.a, &fixture.b, &0, options.clone(), None).unwrap(),
            );
            assert_dense(&baseline, &fixture.expected, tol, None);
            assert_dense(&tree, &fixture.expected, tol, None);
            assert_eq!(tree.node_names(), baseline.node_names());
            assert_report(&tree, &report, 0);
            for threshold in [1, 2] {
                let outcome = contract_with_outcome(
                    &fixture.a,
                    &fixture.b,
                    &0,
                    options.clone(),
                    NonZeroUsize::new(threshold),
                )
                .unwrap();
                let ContractionOutcome::Saturated { edge } = outcome else {
                    panic!("expected saturation")
                };
                assert_eq!(edge, report.edge_ranks[0]);
            }
            let (below, below_report) = completed(
                contract_with_outcome(&fixture.a, &fixture.b, &0, options, NonZeroUsize::new(3))
                    .unwrap(),
            );
            assert_eq!(below_report, report);
            assert_dense(&below, &fixture.expected, tol, None);
        }
    }
}

#[test]
fn all_methods_on_real_chains_and_branched_trees() {
    check_methods::<f64>(|x| x, |x| x, 1e-10, true);
}
#[test]
fn all_methods_on_complex_chains_and_branched_trees() {
    check_methods::<Complex64>(
        |x| Complex64::new(x, 0.),
        |x| Complex64::new(x, x),
        1e-10,
        true,
    );
}
#[test]
fn f32_support_and_errors_match_the_existing_methods() {
    check_methods::<f32>(|x| x as f32, |x| x as f32, 2e-5, false);
}
#[test]
fn c32_support_and_errors_match_the_existing_methods() {
    check_methods::<Complex32>(
        |x| Complex32::new(x as f32, 0.),
        |x| Complex32::new(x as f32, x as f32),
        2e-5,
        false,
    );
}

#[test]
fn every_context_method_preserves_values_and_final_ranks() {
    let context = context();
    for branched in [false, true] {
        let mut fixture = Fixture::new::<Complex64>(
            branched,
            false,
            |x| Complex64::new(x, 0.),
            |x| Complex64::new(x, x),
        );
        fixture.adopt(&context);
        for options in methods() {
            let ordinary =
                contract_in(&fixture.a, &fixture.b, &0, options.clone(), &context).unwrap();
            let (tree, report) = completed(
                contract_with_outcome_in(
                    &fixture.a,
                    &fixture.b,
                    &0,
                    options.clone(),
                    None,
                    &context,
                )
                .unwrap(),
            );
            assert_dense(&ordinary, &fixture.expected, 1e-10, Some(&context));
            assert_dense(&tree, &fixture.expected, 1e-10, Some(&context));
            assert_report(&tree, &report, 0);
            let outcome = contract_with_outcome_in(
                &fixture.a,
                &fixture.b,
                &0,
                options,
                NonZeroUsize::new(2),
                &context,
            )
            .unwrap();
            let ContractionOutcome::Saturated { edge } = outcome else {
                panic!("expected saturation")
            };
            assert_eq!(edge, report.edge_ranks[0]);
        }
    }
}

#[test]
fn a_pruned_center_uses_the_outputs_canonical_root() {
    let fixture = Fixture::new::<f64>(false, true, |x| x, |x| x);
    let (tree, report) = completed(
        contract_with_outcome(
            &fixture.a,
            &fixture.b,
            &0,
            ContractionOptions::zipup().with_max_bond_dim(2),
            None,
        )
        .unwrap(),
    );
    assert!(tree.node_index(&0).is_none());
    assert_eq!(tree.node_count(), 2);
    assert_report(&tree, &report, 0);
    assert_dense(&tree, &fixture.expected, 1e-10, None);
}

#[test]
fn the_probe_threshold_is_independent_of_the_truncation_cap() {
    let fixture = Fixture::new::<f64>(true, false, |x| x, |x| x);
    let options = ContractionOptions::zipup().with_max_bond_dim(1);
    let (tree, report) = completed(
        contract_with_outcome(
            &fixture.a,
            &fixture.b,
            &0,
            options.clone(),
            NonZeroUsize::new(2),
        )
        .unwrap(),
    );
    assert_eq!(tree.link_dims(), vec![1; 3]);
    assert!(report.edge_ranks.iter().all(|edge| edge.rank == 1));
    let outcome =
        contract_with_outcome(&fixture.a, &fixture.b, &0, options, NonZeroUsize::new(1)).unwrap();
    let ContractionOutcome::Saturated { edge } = outcome else {
        panic!("rank one reaches threshold one")
    };
    assert_eq!(edge, report.edge_ranks[0]);
}

#[test]
fn seeded_rsvd_keeps_the_current_dispatch_in_both_execution_routes() {
    let context = context();
    for scoped in [false, true] {
        let mut fixture = Fixture::new::<Complex64>(
            true,
            false,
            |x| Complex64::new(x, 0.),
            |x| Complex64::new(x, x),
        );
        if scoped {
            fixture.adopt(&context);
        }
        for options in [ContractionOptions::zipup(), ContractionOptions::fit()] {
            let mut options = options
                .with_max_bond_dim(2)
                .with_factorize_alg(FactorizeAlg::RSVD);
            options.rsvd_seed = 73;
            let ordinary = if scoped {
                contract_in(&fixture.a, &fixture.b, &0, options.clone(), &context)
            } else {
                contract(&fixture.a, &fixture.b, &0, options.clone())
            }
            .unwrap();
            let outcome = if scoped {
                contract_with_outcome_in(&fixture.a, &fixture.b, &0, options, None, &context)
            } else {
                contract_with_outcome(&fixture.a, &fixture.b, &0, options, None)
            }
            .unwrap();
            let (tree, report) = completed(outcome);
            assert_dense(
                &ordinary,
                &fixture.expected,
                1e-10,
                scoped.then_some(&context),
            );
            assert_dense(&tree, &fixture.expected, 1e-10, scoped.then_some(&context));
            assert_report(&tree, &report, 0);
        }
    }
}

#[test]
fn scalar_outputs_complete_even_at_threshold_one_and_keep_gradients() {
    let site = DynIndex::new_dyn(2);
    let a_tensor = IdxTensor::from_dense(vec![site.clone()], vec![1., 2.])
        .unwrap()
        .enable_grad()
        .unwrap();
    let a = TreeTN::from_tensors(vec![a_tensor.clone()], vec![0usize]).unwrap();
    let b = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![3., 4.]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let (tree, report) = completed(
        contract_with_outcome(
            &a,
            &b,
            &0,
            ContractionOptions::zipup(),
            NonZeroUsize::new(1),
        )
        .unwrap(),
    );
    assert!(report.edge_ranks.is_empty());
    let scalar = tree.to_dense().unwrap();
    assert!((scalar.to_vec::<f64>().unwrap()[0] - 11.).abs() < 1e-12);
    scalar.backward().unwrap();
    assert_eq!(
        a_tensor.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![3., 4.]
    );
}

fn source_chain(error: &(dyn Error + 'static)) -> Vec<String> {
    let mut result = vec![error.to_string()];
    let mut next = error.source();
    while let Some(source) = next {
        result.push(source.to_string());
        next = source.source();
    }
    result
}

#[test]
fn failures_preserve_source_chains() {
    let fixture = Fixture::new::<f64>(true, false, |x| x, |x| x);
    for options in [
        ContractionOptions::zipup().with_max_bond_dim(0),
        ContractionOptions::src(),
        ContractionOptions::new(crate::contraction::ContractionMethod::Naive),
    ] {
        let ordinary = contract(&fixture.a, &fixture.b, &0, options.clone()).unwrap_err();
        let reported =
            contract_with_outcome(&fixture.a, &fixture.b, &0, options, NonZeroUsize::new(1))
                .unwrap_err();
        assert_eq!(source_chain(&ordinary), source_chain(&reported));
    }
}
