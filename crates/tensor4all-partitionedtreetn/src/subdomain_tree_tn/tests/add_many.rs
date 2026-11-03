//! N-ary exact sum of subdomains with one projector.

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

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

#[test]
fn add_many_sums_every_term_into_one_patch() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let terms = [
        patch(&sites, 1, 1.0),
        patch(&sites, 1, 10.0),
        patch(&sites, 1, 100.0),
    ];

    let sum = SubDomainTreeTN::add_many(&terms.iter().collect::<Vec<_>>()).unwrap();

    assert_eq!(sum.projector(), terms[0].projector());
    assert_eq!(sum.max_bond_dim(), 3);
    // Column-major over (sites[0], sites[1]); only sites[0] = 1 survives.
    let expected = IdxTensor::from_dense(sites.to_vec(), vec![0.0, 111.0, 0.0, 222.0]).unwrap();
    let error = sum.data().to_dense().unwrap().distance(&expected).unwrap();
    assert!(error < 1e-12, "dense error {error}");
}

#[test]
fn add_many_rejects_a_later_term_with_another_projector() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let terms = [
        patch(&sites, 1, 1.0),
        patch(&sites, 1, 2.0),
        patch(&sites, 0, 3.0),
    ];

    let result = SubDomainTreeTN::add_many(&terms.iter().collect::<Vec<_>>());

    assert!(matches!(
        result,
        Err(PartitionedTreeTNError::ProjectorMismatch)
    ));
}

#[test]
fn add_many_rejects_no_terms() {
    let result = SubDomainTreeTN::<usize>::add_many(&[]);

    assert!(matches!(result, Err(PartitionedTreeTNError::Empty)));
}
