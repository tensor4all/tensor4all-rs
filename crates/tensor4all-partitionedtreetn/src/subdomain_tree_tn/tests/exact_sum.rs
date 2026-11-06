//! `SubDomainTreeTN::exact_sum`: every engine gives the same patch, the first
//! term's metadata is kept, and the terms are checked as `add` checks them.

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::{SumEngine, TreeTN};

use crate::error::PartitionedTreeTNError;
use crate::projector::Projector;
use crate::subdomain_tree_tn::SubDomainTreeTN;

/// A two-node patch `0 - 1` over `sites`, projected to `sites[0] = coordinate`
/// (the masked entries are zero), with bond dimension 1.
fn patch(sites: &[DynIndex; 2], coordinate: usize, scale: f64) -> SubDomainTreeTN<usize> {
    let bond = DynIndex::new_dyn(1);
    let mut first = vec![0.0; 2];
    first[coordinate] = scale;
    let t0 = IdxTensor::from_dense(vec![sites[0].clone(), bond.clone()], first).unwrap();
    let t1 = IdxTensor::from_dense(vec![bond, sites[1].clone()], vec![1.0, 2.0]).unwrap();
    let tree = TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1]).unwrap();
    let projector = Projector::from_pairs([(sites[0].clone(), coordinate)]).unwrap();
    SubDomainTreeTN::new(tree, projector).unwrap()
}

fn node_payloads(patch: &SubDomainTreeTN<usize>) -> Vec<Vec<f64>> {
    let data = patch.data();
    data.node_names()
        .into_iter()
        .map(|node| {
            data.tensor(data.node_index(&node).unwrap())
                .unwrap()
                .to_vec::<f64>()
                .unwrap()
        })
        .collect()
}

#[test]
fn exact_sum_engines_give_the_same_patch() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let terms = [
        patch(&sites, 1, 1.0),
        patch(&sites, 1, 10.0),
        patch(&sites, 1, 100.0),
    ];
    let refs: Vec<_> = terms.iter().collect();

    let binary = SubDomainTreeTN::exact_sum(&refs, SumEngine::Binary).unwrap();
    let nary = SubDomainTreeTN::exact_sum(&refs, SumEngine::Nary).unwrap();
    let fallback =
        SubDomainTreeTN::exact_sum(&refs, SumEngine::NaryWithin { limit_bytes: 0 }).unwrap();

    assert_eq!(nary.projector(), terms[0].projector());
    assert_eq!(node_payloads(&nary), node_payloads(&binary));
    assert_eq!(node_payloads(&fallback), node_payloads(&binary));
    // Column-major over (sites[0], sites[1]); only sites[0] = 1 survives.
    let expected = IdxTensor::from_dense(sites.to_vec(), vec![0.0, 111.0, 0.0, 222.0]).unwrap();
    let error = nary.data().to_dense().unwrap().distance(&expected).unwrap();
    assert!(error < 1e-12, "dense error {error}");
}

#[test]
fn exact_sum_keeps_the_first_terms_budget() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let first = patch(&sites, 1, 1.0).with_budget_squared(0.25).unwrap();
    let second = patch(&sites, 1, 2.0).with_budget_squared(0.5).unwrap();

    let sum = SubDomainTreeTN::exact_sum(&[&first, &second], SumEngine::Nary).unwrap();

    assert_eq!(sum.budget_squared, Some(0.25));
}

#[test]
fn exact_sum_checks_every_term() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let a = patch(&sites, 1, 1.0);
    let b = patch(&sites, 1, 2.0);
    let other_projector = patch(&sites, 0, 3.0);
    let complex = {
        let c = |re: f64| num_complex::Complex64::new(re, 0.5);
        let bond = DynIndex::new_dyn(1);
        let t0 = IdxTensor::from_dense(vec![sites[0].clone(), bond.clone()], vec![c(0.0), c(1.0)])
            .unwrap();
        let t1 = IdxTensor::from_dense(vec![bond, sites[1].clone()], vec![c(1.0), c(2.0)]).unwrap();
        let tree = TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1]).unwrap();
        SubDomainTreeTN::new(tree, b.projector().clone()).unwrap()
    };

    let one_node = {
        let t = IdxTensor::from_dense(sites.to_vec(), vec![0.0, 1.0, 0.0, 2.0]).unwrap();
        let tree = TreeTN::from_tensors(vec![t], vec![0usize]).unwrap();
        SubDomainTreeTN::new(tree, b.projector().clone()).unwrap()
    };

    for engine in [SumEngine::Binary, SumEngine::Nary] {
        assert!(matches!(
            SubDomainTreeTN::exact_sum(&[&a, &b, &one_node], engine),
            Err(PartitionedTreeTNError::TopologyMismatch)
        ));
        assert!(matches!(
            SubDomainTreeTN::exact_sum(&[&a, &b, &other_projector], engine),
            Err(PartitionedTreeTNError::ProjectorMismatch)
        ));
        assert!(matches!(
            SubDomainTreeTN::exact_sum(&[&a, &b, &complex], engine),
            Err(PartitionedTreeTNError::DTypeMismatch { .. })
        ));
        assert!(matches!(
            SubDomainTreeTN::<usize>::exact_sum(&[], engine),
            Err(PartitionedTreeTNError::Empty)
        ));
    }
}

#[test]
fn exact_sum_of_one_term_returns_it() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let a = patch(&sites, 1, 1.0);

    let sum = SubDomainTreeTN::exact_sum(&[&a], SumEngine::Nary).unwrap();

    assert_eq!(node_payloads(&sum), node_payloads(&a));
    assert_eq!(sum.data().link_dims(), a.data().link_dims());
}
