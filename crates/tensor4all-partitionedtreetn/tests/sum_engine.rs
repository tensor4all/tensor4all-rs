//! The sum engine of the routines that sum many patches: every engine gives
//! the same result, and `to_treetn_with` reports the method that ran.

use tensor4all_core::{contract, DynIndex, IdxTensor, IndexLike};
use tensor4all_partitionedtreetn::{
    add_with_patching, contract_adaptive, PartitionedContractionOptions, PartitionedTreeTN,
    PatchingOptions, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::{contraction::ContractionOptions, SumEngine, TreeTN};

const ENGINES: [SumEngine; 3] = [
    SumEngine::Binary,
    SumEngine::Nary,
    SumEngine::NaryWithin { limit_bytes: 0 },
];

fn values(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|k| ((seed * 37 + k) as f64 * 0.731).sin())
        .collect()
}

/// A two-node chain `0 - 1` with sites `site0` on node 0 and `site1` on node 1
/// and bond dimension 2.
fn chain(site0: &DynIndex, site1: &DynIndex, seed: usize) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(
        vec![site0.clone(), bond.clone()],
        values(site0.dim() * 2, seed),
    )
    .unwrap();
    let t1 = IdxTensor::from_dense(vec![bond, site1.clone()], values(2 * site1.dim(), seed + 1))
        .unwrap();
    TreeTN::from_tensors(vec![t0, t1], vec![0, 1]).unwrap()
}

/// `tree` split into one patch per value of `index`.
fn split(tree: &TreeTN<IdxTensor, usize>, index: &DynIndex) -> PartitionedTreeTN<usize> {
    PartitionedTreeTN::from_subdomains(
        (0..index.dim())
            .map(|value| {
                let projector = Projector::from_pairs([(index.clone(), value)]).unwrap();
                SubDomainTreeTN::new(tree.clone(), projector).unwrap()
            })
            .collect(),
    )
    .unwrap()
}

/// Two partitions split on the contracted index `k` (dimension 3): every
/// output has three contributions, one per value of `k`, which the
/// contraction sums. Returns them with the dense reference result.
fn contraction_case() -> (
    PartitionedTreeTN<usize>,
    PartitionedTreeTN<usize>,
    IdxTensor,
) {
    let (x, y, k) = (
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(3),
    );
    let left = chain(&x, &k, 0);
    let right = chain(&y, &k, 10);
    let reference = contract(&[&left.to_dense().unwrap(), &right.to_dense().unwrap()]).unwrap();
    (split(&left, &k), split(&right, &k), reference)
}

fn dense(partition: &PartitionedTreeTN<usize>) -> IdxTensor {
    partition.to_treetn().unwrap().to_dense().unwrap()
}

#[test]
fn contract_gives_the_same_result_with_every_engine() {
    let (left, right, reference) = contraction_case();

    // A plain `ContractionOptions` still works and means `Binary`.
    let plain = left
        .contract(&right, &0, ContractionOptions::default())
        .unwrap();
    assert!(dense(&plain).distance(&reference).unwrap() < 1e-10);
    for sum_engine in ENGINES {
        let options = PartitionedContractionOptions {
            sum_engine,
            ..ContractionOptions::default().into()
        };
        let result = left.contract(&right, &0, options.clone()).unwrap();
        let selected = left
            .contract_selected(&right, &0, options, &[Projector::new()])
            .unwrap();

        assert_eq!(result.len(), 1, "{sum_engine:?}");
        let error = dense(&result).distance(&reference).unwrap();
        assert!(error < 1e-10, "{sum_engine:?}: error {error}");
        let error = dense(&selected).distance(&reference).unwrap();
        assert!(error < 1e-10, "{sum_engine:?} (selected): error {error}");
    }
}

#[test]
fn contract_adaptive_gives_the_same_result_with_every_engine() {
    let (left, right, reference) = contraction_case();

    for sum_engine in ENGINES {
        let patching = PatchingOptions {
            sum_engine,
            ..PatchingOptions::default()
        };
        let result =
            contract_adaptive(&left, &right, &0, &ContractionOptions::default(), &patching)
                .unwrap();

        let error = dense(&result).distance(&reference).unwrap();
        assert!(error < 1e-10, "{sum_engine:?}: error {error}");
    }
}

#[test]
fn add_with_patching_sums_equal_keys_with_every_engine() {
    let (x, z) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let projector = Projector::from_pairs([(x.clone(), 1)]).unwrap();
    let terms: Vec<_> = (0..3)
        .map(|seed| SubDomainTreeTN::new(chain(&x, &z, 20 * seed), projector.clone()).unwrap())
        .collect();
    let reference = terms
        .iter()
        .map(|term| term.data().to_dense().unwrap())
        .reduce(|sum, term| {
            sum.axpby(
                tensor4all_core::AnyScalar::new_real(1.0),
                &term,
                tensor4all_core::AnyScalar::new_real(1.0),
            )
            .unwrap()
        })
        .unwrap();

    for sum_engine in ENGINES {
        let patching = PatchingOptions {
            sum_engine,
            ..PatchingOptions::default()
        };
        let result = add_with_patching(terms.clone(), &0, &patching).unwrap();

        let error = dense(&result).distance(&reference).unwrap();
        assert!(error < 1e-10, "{sum_engine:?}: error {error}");
    }
}

#[test]
fn to_treetn_with_reports_the_method_and_the_estimate() {
    let (x, y) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
    let partition = split(&chain(&x, &y, 0), &y);

    let binary = partition.to_treetn_with(SumEngine::Binary).unwrap();
    let nary = partition.to_treetn_with(SumEngine::Nary).unwrap();
    // Summed bond 2 + 2 + 2 = 6: node 0 holds 2 x 6 and node 1 holds 6 x 3
    // elements, 16 bytes each.
    let estimate = (2 * 6 + 6 * 3) * 16;
    let at_limit = partition
        .to_treetn_with(SumEngine::NaryWithin {
            limit_bytes: estimate,
        })
        .unwrap();
    let below_limit = partition
        .to_treetn_with(SumEngine::NaryWithin {
            limit_bytes: estimate - 1,
        })
        .unwrap();

    assert!(!binary.used_nary && nary.used_nary);
    assert!(at_limit.used_nary && !below_limit.used_nary);
    for sum in [&binary, &nary, &at_limit, &below_limit] {
        assert_eq!(sum.estimated_nary_bytes, estimate);
        assert_eq!(sum.network.link_dims(), vec![6]);
    }
    let reference = partition.to_treetn().unwrap().to_dense().unwrap();
    for sum in [binary, nary, at_limit, below_limit] {
        assert!(
            sum.network
                .to_dense()
                .unwrap()
                .distance(&reference)
                .unwrap()
                < 1e-12
        );
    }
}
