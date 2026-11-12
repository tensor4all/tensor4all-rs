use super::{level_share, merge_group_charge_bound, prefixes_overlap};

/// Two axes: axis 0 nests `first` inside `second`, axis 1 nests `second`
/// inside `first`. Each axis overlaps, so the leaves overlap even though
/// neither key is a prefix of the other on every axis in the same direction.
#[test]
fn per_axis_prefixes_detect_two_way_nesting() {
    let first = vec![(1usize, 0usize), (2, 0)];
    let second = vec![(2usize, 0usize), (1, 0)];
    assert!(prefixes_overlap(&first, &second));
    assert!(prefixes_overlap(&second, &first));

    // Same-direction nesting on every axis is also an overlap.
    let nested = vec![(1usize, 0usize), (1, 0)];
    let deeper = vec![(2usize, 0usize), (2, 0)];
    assert!(prefixes_overlap(&nested, &deeper));
    assert!(prefixes_overlap(&deeper, &nested));

    // A disagreement on any axis makes the two disjoint.
    let disjoint = vec![(1usize, 0usize), (1, 1)];
    assert!(!prefixes_overlap(&nested, &disjoint));
    assert!(!prefixes_overlap(&disjoint, &nested));

    // One axis nests while the other is disjoint: still disjoint.
    let split = vec![(2usize, 0usize), (1, 1)];
    assert!(!prefixes_overlap(&first, &split));
    assert!(!prefixes_overlap(&split, &first));

    // Different depths on both axes without nesting are disjoint.
    let sideways = vec![(1usize, 1usize), (1, 1)];
    assert!(!prefixes_overlap(&first, &sideways));
}

/// A merge of two multi-term items charges the summed operand term count: two
/// shares per paired term (drop and compression) and one per single-sided
/// term. A three-item group accumulates those bounds in merge order.
#[test]
fn multi_term_merge_charge_bound_follows_the_operand_term_counts() {
    assert_eq!(merge_group_charge_bound(&[3, 2]).expect("bound"), 5);
    assert_eq!(merge_group_charge_bound(&[1, 1]).expect("bound"), 2);
    assert_eq!(merge_group_charge_bound(&[4]).expect("bound"), 0);
    assert_eq!(merge_group_charge_bound(&[]).expect("bound"), 0);
    // (2 + 3) charges for the first merge, then (5 + 1) for the second.
    assert_eq!(merge_group_charge_bound(&[2, 3, 1]).expect("bound"), 11);
}

/// A level that must merge multi-term items can make more charges than it has
/// live items, so the old `remaining / (2 * items)` share could overspend and
/// front-load the allowance. Spending every charge at the per-level share must
/// stay inside what is left before the nonnegative clamp at each level.
#[test]
fn multi_term_merges_do_not_overspend_the_remaining_allowance() {
    let levels = 3usize;
    let mut remaining = 1.0_f64;
    // Charge counts larger than the live item count, as multi-term merges and
    // a stopped region's discarded probe plus its own merge produce them.
    for (level_index, charges) in [13usize, 7, 3].into_iter().enumerate() {
        let levels_left = levels - level_index;
        let share = level_share(remaining, levels_left, charges).expect("share");
        let worst_case_spend = share * charges as f64;
        assert!(
            worst_case_spend <= remaining / levels_left as f64 + 1e-15,
            "level spend {worst_case_spend:e} exceeds its slice of {remaining:e}"
        );
        assert!(
            worst_case_spend <= remaining,
            "level spend {worst_case_spend:e} exceeds the remaining {remaining:e} before the clamp"
        );
        remaining = (remaining - worst_case_spend).max(0.0);
    }

    // No charge means no allocation, and a zero allowance stays zero.
    assert_eq!(level_share(1.0, 2, 0).expect("share"), 0.0);
    assert_eq!(level_share(0.0, 2, 5).expect("share"), 0.0);
}
