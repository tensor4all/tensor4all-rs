//! The group-sum limit of `contract`, `contract_selected` and
//! `add_with_patching` is checked for every group before any group sum is
//! built, so a refused group leaves no sum or truncation done on the groups
//! before it.

use super::*;
use crate::patching::add_with_patching;
use crate::subdomain_tree_tn::EXACT_SUMS_BUILT;

/// Run `operation` and count the exact sums it builds.
fn count_sums<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    EXACT_SUMS_BUILT.with(|count| count.set(0));
    let result = operation();
    (result, EXACT_SUMS_BUILT.with(|count| count.get()))
}

/// The term count and estimate of a `GroupSumLimit` error.
fn refused(result: Result<PartitionedTreeTN>) -> (usize, usize) {
    match result {
        Err(PartitionedTreeTNError::GroupSumLimit {
            terms,
            estimated_bytes,
            ..
        }) => (terms, estimated_bytes),
        other => panic!(
            "expected GroupSumLimit, got {:?}",
            other.map(|partition| partition.len())
        ),
    }
}

/// Keep the patches of `patches` whose `mid` coordinate is not 2, and those
/// whose `other` coordinate is 2.
fn without_mid_2(
    patches: &PartitionedTreeTN,
    mid: &DynIndex,
    other: &DynIndex,
) -> PartitionedTreeTN {
    let kept = patches
        .values()
        .filter(|patch| {
            patch.projector().get(mid) != Some(2) || patch.projector().get(other) == Some(2)
        })
        .cloned()
        .collect();
    PartitionedTreeTN::from_subdomains(kept).unwrap()
}

#[test]
fn contract_checks_every_group_sum_before_building_any() {
    let product = BlockProduct::new(None, 1.0);
    // Without the `mid = 2` patches outside row 2 of `A` and column 2 of `B`,
    // block (2, 2) has three contributions and the other eight blocks two.
    let a = without_mid_2(&product.a, &product.mid, &product.row);
    let b = without_mid_2(&product.b, &product.mid, &product.col);
    let options = |group_sum_limit_bytes| PartitionedContractionOptions {
        group_sum_limit_bytes,
        ..PartitionedContractionOptions::default()
    };
    let wanted = product.keys(&all_blocks());
    let contract = |limit, selected: bool| {
        if selected {
            a.contract_selected(&b, &0, options(limit), &wanted)
        } else {
            a.contract(&b, &0, options(limit))
        }
    };

    let (terms, large) = refused(a.contract_selected(&b, &0, options(0), &[product.key((2, 2))]));
    assert_eq!(terms, 3);
    for selected in [false, true] {
        // A two-term group comes first, so building groups in order before
        // checking the three-term one would build some of them.
        let (terms, small) = refused(contract(0, selected));
        assert_eq!(terms, 2, "selected = {selected}");
        assert!(small < large, "selected = {selected}: {small} >= {large}");

        let (result, built) = count_sums(|| contract(large - 1, selected));
        assert_eq!(refused(result), (3, large), "selected = {selected}");
        assert_eq!(built, 0, "selected = {selected}");

        let (result, built) = count_sums(|| contract(large, selected));
        assert_eq!(result.unwrap().len(), 9, "selected = {selected}");
        assert_eq!(built, 9, "selected = {selected}");
    }
}

#[test]
fn add_with_patching_checks_every_group_sum_before_building_any() {
    let product = BlockProduct::new(None, 1.0);
    let patch = |block: (usize, usize)| {
        let key = Projector::from_pairs([
            (product.row.clone(), block.0),
            (product.mid.clone(), block.1),
        ])
        .unwrap();
        product.a.get(&key).unwrap().clone()
    };
    let options = |group_sum_limit_bytes| PatchingOptions {
        max_bond_dim: None,
        group_sum_limit_bytes,
        ..PatchingOptions::default()
    };
    let three = vec![patch((1, 1)); 3];
    let (terms, large) = refused(add_with_patching(three.clone(), &0, &options(0)));
    assert_eq!(terms, 3);

    // A key with two patches and a key with three: whichever group comes
    // first, none is built before the three-term group is refused.
    let mut inputs = vec![patch((0, 0)); 2];
    inputs.extend(three);
    let (result, built) = count_sums(|| add_with_patching(inputs, &0, &options(large - 1)));
    assert_eq!(refused(result), (3, large));
    assert_eq!(built, 0);
}

#[test]
fn add_with_patching_reports_the_first_oversized_group_in_key_order() {
    let product = BlockProduct::new(None, 1.0);
    let key = |block: (usize, usize)| {
        Projector::from_pairs([
            (product.row.clone(), block.0),
            (product.mid.clone(), block.1),
        ])
        .unwrap()
    };
    let patch = |block| product.a.get(&key(block)).unwrap().clone();
    let options = PatchingOptions {
        max_bond_dim: None,
        group_sum_limit_bytes: 0,
        ..PatchingOptions::default()
    };
    // Both groups are over the limit; the reported one is the first in
    // canonical key order, whatever the input order.
    let expected_terms = if key((0, 0)).canonical_cmp(&key((1, 1))).is_lt() {
        2
    } else {
        3
    };
    let two = vec![patch((0, 0)); 2];
    let three = vec![patch((1, 1)); 3];
    for inputs in [[two.clone(), three.clone()].concat(), [three, two].concat()] {
        let (terms, _) = refused(add_with_patching(inputs, &0, &options));
        assert_eq!(terms, expected_terms);
    }
}
