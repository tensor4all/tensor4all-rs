use super::*;
use std::cell::Cell;

use crate::treetn::contraction::{
    zipup_control::{AbortAtRank, Complete, ZipupControl},
    GlobalLeaves, LeafOps, ZipupTopologyMode,
};
use tensor4all_core::{FactorizeOptions, FactorizeResult, TensorIndex};

struct CountedLeaves<'a> {
    base: &'a dyn LeafOps<IdxTensor>,
    factorizations: Cell<usize>,
    contractions: Cell<usize>,
    fail_at: Option<usize>,
}

impl<'a> CountedLeaves<'a> {
    fn new(base: &'a dyn LeafOps<IdxTensor>) -> Self {
        Self {
            base,
            factorizations: Cell::new(0),
            contractions: Cell::new(0),
            fail_at: None,
        }
    }

    fn before_factorization(&self) -> anyhow::Result<()> {
        let count = self.factorizations.get() + 1;
        self.factorizations.set(count);
        anyhow::ensure!(
            self.fail_at != Some(count),
            "injected later factorization failure"
        );
        Ok(())
    }
}

impl LeafOps<IdxTensor> for CountedLeaves<'_> {
    fn contract_pair(&self, a: &IdxTensor, b: &IdxTensor) -> anyhow::Result<IdxTensor> {
        self.contractions.set(self.contractions.get() + 1);
        self.base.contract_pair(a, b)
    }

    fn contract_list(&self, tensors: &[IdxTensor]) -> anyhow::Result<IdxTensor> {
        self.contractions.set(self.contractions.get() + 1);
        self.base.contract_list(tensors)
    }

    fn contract_refs(&self, tensors: &[&IdxTensor]) -> anyhow::Result<IdxTensor> {
        self.contractions.set(self.contractions.get() + 1);
        self.base.contract_refs(tensors)
    }

    fn outer_product(&self, a: &IdxTensor, b: &IdxTensor) -> anyhow::Result<IdxTensor> {
        self.base.outer_product(a, b)
    }

    fn factorize(
        &self,
        tensor: &IdxTensor,
        left: &[DynIndex],
        options: &FactorizeOptions,
    ) -> anyhow::Result<FactorizeResult<IdxTensor>> {
        self.before_factorization()?;
        self.base.factorize(tensor, left, options)
    }

    fn factorize_auto(
        &self,
        tensor: &IdxTensor,
        left: &[DynIndex],
        options: &FactorizeOptions,
    ) -> anyhow::Result<FactorizeResult<IdxTensor>> {
        self.before_factorization()?;
        self.base.factorize_auto(tensor, left, options)
    }
}

fn run<C: ZipupControl<IdxTensor, usize>>(
    fixture: &Fixture,
    options: &ContractionOptions,
    leaves: &dyn LeafOps<IdxTensor>,
    control: C,
) -> anyhow::Result<C::Output> {
    fixture.a.contract_zipup_controlled(
        &fixture.b,
        &0,
        crate::CanonicalForm::Unitary,
        options.svd_policy,
        options.max_bond_dim,
        ZipupTopologyMode::PruneScalarSubtrees,
        leaves,
        control,
    )
}

fn check_tree<T: TensorElement>(real: impl Fn(f64) -> T + Copy, phased: impl Fn(f64) -> T + Copy) {
    let context = context();
    for edges in [
        vec![(0, 1), (0, 2), (0, 3)],
        vec![(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6)],
    ] {
        for scoped in [false, true] {
            let nodes: Vec<_> = (0..=edges.len()).collect();
            let mut fixture = Fixture::with_edges(&edges, &nodes, real, phased);
            if scoped {
                fixture.adopt(&context);
            }
            let session = SessionLeaves(context.clone());
            let base: &dyn LeafOps<IdxTensor> = if scoped { &session } else { &GlobalLeaves };
            let options = ContractionOptions::zipup().with_max_bond_dim(2);
            let full = CountedLeaves::new(base);
            let network = run(&fixture, &options, &full, Complete).unwrap();
            assert_dense(
                &network,
                &fixture.expected,
                1e-10,
                scoped.then_some(&context),
            );
            let (_, report) = completed(super::super::super::finish(network, &0, None).unwrap());
            assert_eq!(full.factorizations.get(), edges.len());
            let input_order = fixture.a.edges_to_canonicalize_by_names(&0).unwrap();
            assert_eq!(
                input_order,
                report
                    .edge_ranks
                    .iter()
                    .map(|e| (e.child, e.parent))
                    .collect::<Vec<_>>()
            );
            for threshold in [1, 2, 3] {
                let counted = CountedLeaves::new(base);
                let outcome = run(
                    &fixture,
                    &options,
                    &counted,
                    AbortAtRank {
                        center: &0,
                        threshold: NonZeroUsize::new(threshold).unwrap(),
                    },
                )
                .unwrap();
                if threshold <= 2 {
                    let ContractionOutcome::Saturated { edge } = outcome else {
                        panic!("expected abort")
                    };
                    assert_eq!(edge, report.edge_ranks[0]);
                    assert_eq!(counted.factorizations.get(), 1);
                    assert!(counted.contractions.get() < full.contractions.get());
                } else {
                    let (network, actual) = completed(outcome);
                    assert_eq!(actual, report);
                    assert_eq!(counted.factorizations.get(), full.factorizations.get());
                    assert_dense(
                        &network,
                        &fixture.expected,
                        1e-10,
                        scoped.then_some(&context),
                    );
                }
            }
        }
    }
}

#[test]
fn real_tree_aborts_before_later_factorizations_and_matches_the_final_report() {
    check_tree::<f64>(|x| x, |x| x);
}

#[test]
fn complex_tree_aborts_before_later_factorizations_and_matches_the_final_report() {
    check_tree::<Complex64>(|x| Complex64::new(x, 0.), |x| Complex64::new(x, x));
}

#[test]
fn a_chain_keeps_its_complete_schedule_before_classification() {
    let fixture = Fixture::new::<f64>(false, false, |x| x, |x| x);
    let options = ContractionOptions::zipup().with_max_bond_dim(2);
    let full = CountedLeaves::new(&GlobalLeaves);
    let network = run(&fixture, &options, &full, Complete).unwrap();
    assert_dense(&network, &fixture.expected, 1e-10, None);
    let (_, report) = completed(super::super::super::finish(network, &0, None).unwrap());
    let counted = CountedLeaves::new(&GlobalLeaves);
    let outcome = run(
        &fixture,
        &options,
        &counted,
        AbortAtRank {
            center: &0,
            threshold: NonZeroUsize::new(2).unwrap(),
        },
    )
    .unwrap();
    let ContractionOutcome::Saturated { edge } = outcome else {
        panic!("expected saturation")
    };
    assert_eq!(edge, report.edge_ranks[0]);
    assert_eq!(counted.factorizations.get(), full.factorizations.get());
    assert_eq!(counted.contractions.get(), full.contractions.get());
}

#[test]
fn pruned_scalar_branches_never_trigger_the_edge_hook() {
    let edges = [(0, 1), (0, 2), (0, 3)];
    for output_nodes in [vec![], vec![2, 3]] {
        let fixture = Fixture::with_edges::<f64>(&edges, &output_nodes, |x| x, |x| x);
        let options = ContractionOptions::zipup().with_max_bond_dim(2);
        let full = CountedLeaves::new(&GlobalLeaves);
        let network = run(&fixture, &options, &full, Complete).unwrap();
        assert_dense(&network, &fixture.expected, 1e-10, None);
        let (_, report) = completed(super::super::super::finish(network, &0, None).unwrap());
        let counted = CountedLeaves::new(&GlobalLeaves);
        let outcome = run(
            &fixture,
            &options,
            &counted,
            AbortAtRank {
                center: &0,
                threshold: NonZeroUsize::new(1).unwrap(),
            },
        )
        .unwrap();
        if output_nodes.is_empty() {
            let (network, actual) = completed(outcome);
            assert!(actual.edge_ranks.is_empty());
            assert_eq!(counted.factorizations.get(), 0);
            assert_dense(&network, &fixture.expected, 1e-10, None);
        } else {
            let ContractionOutcome::Saturated { edge } = outcome else {
                panic!("expected abort")
            };
            assert_eq!(edge, report.edge_ranks[0]);
            assert_ne!(edge.child, 1);
            assert_eq!(counted.factorizations.get(), 1);
        }
    }
}

#[test]
fn abort_skips_later_execution_errors_but_keeps_errors_before_saturation() {
    let fixture = Fixture::new::<f64>(true, false, |x| x, |x| x);
    let options = ContractionOptions::zipup().with_max_bond_dim(2);
    for fail_at in [1, 2] {
        let counted = CountedLeaves {
            fail_at: Some(fail_at),
            ..CountedLeaves::new(&GlobalLeaves)
        };
        let ordinary = run(&fixture, &options, &counted, Complete).unwrap_err();
        counted.factorizations.set(0);
        let outcome = run(
            &fixture,
            &options,
            &counted,
            AbortAtRank {
                center: &0,
                threshold: NonZeroUsize::new(2).unwrap(),
            },
        );
        if fail_at == 1 {
            let error = outcome.unwrap_err();
            assert_eq!(
                source_chain(error.as_ref()),
                source_chain(ordinary.as_ref())
            );
        } else {
            assert!(matches!(
                outcome.unwrap(),
                ContractionOutcome::Saturated { .. }
            ));
            assert_eq!(counted.factorizations.get(), 1);
        }
    }
}

#[test]
fn public_entries_stop_at_the_first_threshold_reaching_edge() {
    use crate::treetn::contraction::zipup_control::tests::measure;

    let context = context();
    for scoped in [false, true] {
        let mut fixture = Fixture::new::<f64>(true, false, |x| x, |x| x);
        if scoped {
            fixture.adopt(&context);
        }
        let call = |threshold| {
            let options = ContractionOptions::zipup().with_max_bond_dim(2);
            if scoped {
                contract_with_outcome_in(&fixture.a, &fixture.b, &0, options, threshold, &context)
            } else {
                contract_with_outcome(&fixture.a, &fixture.b, &0, options, threshold)
            }
        };
        let (ordinary, ordinary_count) = measure(|| {
            let options = ContractionOptions::zipup().with_max_bond_dim(2);
            if scoped {
                contract_in(&fixture.a, &fixture.b, &0, options, &context)
            } else {
                contract(&fixture.a, &fixture.b, &0, options)
            }
        });
        assert_dense(
            &ordinary.unwrap(),
            &fixture.expected,
            1e-10,
            scoped.then_some(&context),
        );
        let (full, full_count) = measure(|| call(None));
        let (network, report) = completed(full.unwrap());
        assert_dense(
            &network,
            &fixture.expected,
            1e-10,
            scoped.then_some(&context),
        );
        assert_eq!(ordinary_count, 3);
        assert_eq!(full_count, ordinary_count);
        let (outcome, aborted_count) = measure(|| call(NonZeroUsize::new(2)));
        let ContractionOutcome::Saturated { edge } = outcome.unwrap() else {
            panic!("expected abort")
        };
        assert_eq!(edge, report.edge_ranks[0]);
        assert_eq!(aborted_count, 1);
        let (below, below_count) = measure(|| call(NonZeroUsize::new(3)));
        let (network, actual) = completed(below.unwrap());
        assert_eq!(actual, report);
        assert_eq!(below_count, full_count);
        assert_dense(
            &network,
            &fixture.expected,
            1e-10,
            scoped.then_some(&context),
        );

        // Validation is still performed before the first edge can stop work.
        let foreign = super::context();
        assert!(contract_with_outcome_in(
            &fixture.a,
            &fixture.b,
            &0,
            ContractionOptions::zipup(),
            NonZeroUsize::new(1),
            &foreign
        )
        .is_err());
        if scoped {
            let mut other = Fixture::new::<f64>(true, false, |x| x, |x| x);
            other.adopt(&foreign);
            let (reported, count) = measure(|| {
                contract_with_outcome_in(
                    &fixture.a,
                    &other.b,
                    &0,
                    ContractionOptions::zipup(),
                    NonZeroUsize::new(1),
                    &context,
                )
            });
            let ordinary = contract_in(
                &fixture.a,
                &other.b,
                &0,
                ContractionOptions::zipup(),
                &context,
            )
            .unwrap_err();
            assert_eq!(
                source_chain(&reported.unwrap_err()),
                source_chain(&ordinary)
            );
            assert_eq!(count, 0);
        }
        assert!(contract_with_outcome(
            &fixture.a,
            &fixture.b,
            &99,
            ContractionOptions::zipup(),
            NonZeroUsize::new(1)
        )
        .is_err());
    }
}

#[test]
fn a_smaller_first_edge_is_kept_before_a_later_edge_saturates() {
    let a_bonds: Vec<_> = (1..=3).map(DynIndex::new_dyn).collect();
    let mut hub = vec![DynIndex::new_dyn(2)];
    hub.extend(a_bonds.iter().cloned());
    let mut a_tensors = vec![IdxTensor::from_dense(hub, vec![1.; 12]).unwrap()];
    for (bond, values) in a_bonds.into_iter().zip([
        vec![1., 1.],
        vec![1., 0., 0., 1.],
        vec![1., 0., 0., 1., 0., 0.],
    ]) {
        a_tensors.push(IdxTensor::from_dense(vec![DynIndex::new_dyn(2), bond], values).unwrap());
    }
    let a = TreeTN::from_tensors(a_tensors, vec![0, 1, 2, 3]).unwrap();
    let bonds: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(1)).collect();
    let mut tensors = vec![IdxTensor::from_dense(bonds.clone(), vec![1.]).unwrap()];
    tensors.extend(
        bonds
            .into_iter()
            .map(|bond| IdxTensor::from_dense(vec![bond], vec![1.]).unwrap()),
    );
    let b = TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap();
    let options = ContractionOptions::zipup().with_max_bond_dim(3);
    let (network, report) =
        completed(contract_with_outcome(&a, &b, &0, options.clone(), None).unwrap());
    assert_eq!(
        report.edge_ranks.iter().map(|e| e.rank).collect::<Vec<_>>(),
        vec![1, 2, 2]
    );
    // The full tensor is a product of ones despite stored dimensions [1, 2, 2].
    // A stored-rank report is deliberately not a minimal Schmidt-rank report.
    let expected = IdxTensor::from_dense(network.external_indices(), vec![1.; 16]).unwrap();
    assert_dense(&network, &expected, 1e-10, None);
    let (outcome, count) = crate::treetn::contraction::zipup_control::tests::measure(|| {
        contract_with_outcome(&a, &b, &0, options, NonZeroUsize::new(2))
    });
    let ContractionOutcome::Saturated { edge } = outcome.unwrap() else {
        panic!("expected abort")
    };
    assert_eq!(edge, report.edge_ranks[1]);
    assert_eq!(count, 2);
}
