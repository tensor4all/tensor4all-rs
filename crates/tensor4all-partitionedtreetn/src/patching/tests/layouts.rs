//! The best, worst, and general patch layouts of patched operator–operator
//! contraction (Grosso et al., arXiv:2602.22372v3, section 4.4 and
//! appendix D, Fig. 20) on a branched tree.
//!
//! `A(x, s)` and `B(s, y)` share the contracted index `s`; each of `x`, `s`
//! and `y` has two binary digits. Both operands live on a star with a
//! degree-three hub: the hub carries `s1`, leaf 1 carries `x1` (in `A`) or
//! `y1` (in `B`), leaf 2 carries `x2` or `y2`, and leaf 3 carries `s2`. Every
//! layout has `N_p = 4` patches per operand:
//!
//! | Layout  | `A` fixes  | `B` fixes  | Compatible pairs | Output groups |
//! |---------|------------|------------|------------------|---------------|
//! | best    | `s1, s2`   | `s1, s2`   | `N_p = 4`        | 1 of 4 pairs  |
//! | worst   | `x1, x2`   | `y1, y2`   | `N_p^2 = 16`     | 16 of 1 pair  |
//! | general | `x1, s1`   | `s1, y1`   | `N_p^1.5 = 8`    | 4 of 2 pairs  |
//!
//! The pair counts are the paper's `N_pair`. Values are deterministic and
//! generic, so every pair contributes a nonzero term of full bond rank.

use super::lazy_contraction::{assert_partition, measure, options};
use super::*;
use crate::partitioned_tree_tn::CONTRACTED_PAIRS;
use num_complex::Complex64;
use tensor4all_core::{TensorContractionLike, TensorElement};

const BOND: usize = 3;

struct Sites {
    x: [DynIndex; 2],
    s: [DynIndex; 2],
    y: [DynIndex; 2],
}

/// One operand on the star `0 - {1, 2, 3}`: hub site `hub`, leaf sites
/// `leaves`, and deterministic entries from `value`.
fn operand<T: TensorElement>(
    hub: &DynIndex,
    leaves: [&DynIndex; 3],
    seed: f64,
    value: &impl Fn(f64) -> T,
) -> TreeTN<IdxTensor, usize> {
    let entries = |len: usize, offset: f64| -> Vec<T> {
        (0..len)
            .map(|k| value((seed + offset + 1.618 * k as f64).sin()))
            .collect()
    };
    let bonds: Vec<_> = (0..3).map(|_| DynIndex::new_bond(BOND).unwrap()).collect();
    let mut hub_indices = vec![hub.clone()];
    hub_indices.extend(bonds.iter().cloned());
    let mut tensors =
        vec![IdxTensor::from_dense(hub_indices, entries(2 * BOND.pow(3), 0.0)).unwrap()];
    for (leaf, (bond, site)) in bonds.into_iter().zip(leaves).enumerate() {
        tensors.push(
            IdxTensor::from_dense(
                vec![bond, site.clone()],
                entries(2 * BOND, 0.7 * (leaf + 1) as f64),
            )
            .unwrap(),
        );
    }
    TreeTN::from_tensors(tensors, vec![0usize, 1, 2, 3]).unwrap()
}

/// All `2^fixed.len()` patches of `tree` that fix exactly `fixed`.
fn partition(tree: &TreeTN<IdxTensor, usize>, fixed: &[&DynIndex]) -> PartitionedTreeTN {
    let patches = (0..1usize << fixed.len())
        .map(|bits| {
            let pairs = fixed
                .iter()
                .enumerate()
                .map(|(digit, index)| ((*index).clone(), (bits >> digit) & 1));
            SubDomainTreeTN::new(tree.clone(), Projector::from_pairs(pairs).unwrap()).unwrap()
        })
        .collect();
    PartitionedTreeTN::from_subdomains(patches).unwrap()
}

struct Layout<'a> {
    name: &'static str,
    left: Vec<&'a DynIndex>,
    right: Vec<&'a DynIndex>,
    pairs: usize,
    groups: usize,
    /// Whether an output keeps an unfixed index of the split order; the worst
    /// layout's outputs already fix all of `x` and `y`.
    refines: bool,
}

fn layouts(sites: &Sites) -> [Layout<'_>; 3] {
    let Sites { x, s, y } = sites;
    [
        Layout {
            name: "best",
            left: vec![&s[0], &s[1]],
            right: vec![&s[0], &s[1]],
            pairs: 4,
            groups: 1,
            refines: true,
        },
        Layout {
            name: "worst",
            left: vec![&x[0], &x[1]],
            right: vec![&y[0], &y[1]],
            pairs: 16,
            groups: 16,
            refines: false,
        },
        Layout {
            name: "general",
            left: vec![&x[0], &s[0]],
            right: vec![&s[0], &y[0]],
            pairs: 8,
            groups: 4,
            refines: true,
        },
    ]
}

fn check_layouts<T: TensorElement>(value: impl Fn(f64) -> T) {
    let sites = Sites {
        x: [DynIndex::new_dyn(2), DynIndex::new_dyn(2)],
        s: [DynIndex::new_dyn(2), DynIndex::new_dyn(2)],
        y: [DynIndex::new_dyn(2), DynIndex::new_dyn(2)],
    };
    let a = operand(
        &sites.s[0],
        [&sites.x[0], &sites.x[1], &sites.s[1]],
        0.3,
        &value,
    );
    let b = operand(
        &sites.s[0],
        [&sites.y[0], &sites.y[1], &sites.s[1]],
        1.1,
        &value,
    );
    let expected = IdxTensor::contract(&[&a.to_dense().unwrap(), &b.to_dense().unwrap()]).unwrap();
    let order: Vec<_> = sites.x.iter().chain(&sites.y).cloned().collect();
    let method = ContractionOptions::zipup();

    for layout in layouts(&sites) {
        let left = partition(&a, &layout.left);
        let right = partition(&b, &layout.right);
        let name = layout.name;

        // Non-adaptive patched contraction attempts exactly the compatible pairs.
        CONTRACTED_PAIRS.with(|count| count.set(0));
        let plain = left.contract(&right, &0, method.clone()).unwrap();
        assert_eq!(CONTRACTED_PAIRS.with(|c| c.get()), layout.pairs, "{name}");
        assert_eq!(plain.len(), layout.groups, "{name}");
        assert_partition(&plain, &expected);

        // Without a cap, adaptive contraction groups the same pairs by output
        // projector and contracts each pair once.
        CONTRACTED_PAIRS.with(|count| count.set(0));
        let (uncapped, probes) = measure(|| {
            contract_adaptive(
                &left,
                &right,
                &0,
                &method,
                &options(order.clone(), PatchSplitStrategy::Sequential, None),
            )
            .unwrap()
        });
        assert_eq!(CONTRACTED_PAIRS.with(|c| c.get()), layout.pairs, "{name}");
        let per_group = layout.pairs / layout.groups;
        assert_eq!(
            probes,
            vec![(per_group, per_group, 0); layout.groups],
            "{name}"
        );
        assert_eq!(uncapped.len(), layout.groups, "{name}");
        assert_partition(&uncapped, &expected);

        // With a cap, every returned patch is below it or has no split
        // candidate left, and the values stay exact. Every generic output
        // group reaches rank 2, so a layout with a candidate refines.
        let cap = 2;
        for strategy in [
            PatchSplitStrategy::Sequential,
            PatchSplitStrategy::ExactParameterGain,
        ] {
            let capped = contract_adaptive(
                &left,
                &right,
                &0,
                &method,
                &options(order.clone(), strategy, Some(cap)),
            )
            .unwrap();
            assert_partition(&capped, &expected);
            assert_eq!(
                capped.len() > layout.groups,
                layout.refines,
                "{name} {strategy:?}: {} patches",
                capped.len()
            );
            for (key, patch) in capped.iter() {
                assert!(
                    patch.max_bond_dim() < cap || order.iter().all(|i| key.is_projected_at(i)),
                    "{name} {strategy:?}: {key:?} saturated with a split candidate left"
                );
            }
        }
    }
}

#[test]
fn best_worst_and_general_layouts_on_a_degree_three_star_real() {
    check_layouts(|value| value);
}

#[test]
fn best_worst_and_general_layouts_on_a_degree_three_star_complex() {
    check_layouts(|value| Complex64::new(value, (2.0 * value).sin()));
}
