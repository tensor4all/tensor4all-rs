use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use super::{
    level_share, merge_group_charge_bound, prefixes_overlap, validate_leaf_geometry,
    CoverageContract, InputKey, ReconstructionTarget, ScheduleGeometry,
};
use crate::projector::index::has_overlapping_pair;
use crate::reconstruction::CoordinateGroup;
use crate::{DynIndex, IdxTensor, PartitionedTreeTN, Projector, SubDomainTreeTN, TreeTN};

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

/// The removed all-pairs scan over the per-axis prefix keys, kept as the
/// quadratic reference the trie replaced.
fn all_pairs_overlap(leaves: &BTreeSet<InputKey>) -> bool {
    for key in leaves {
        for other in leaves {
            if key == other {
                continue;
            }
            if prefixes_overlap(key, other) {
                return true;
            }
        }
    }
    false
}

/// A small linear congruential generator, so the random layouts below are
/// reproducible without a new dependency.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize
    }
}

/// A random dyadic split layout: repeatedly split a random leaf on a random axis
/// that still has a free bit. The result is prefix-free by construction.
fn split_layout(axis_bits: &[usize], leaf_target: usize, rng: &mut Lcg) -> BTreeSet<InputKey> {
    let mut leaves = BTreeSet::from([vec![(0usize, 0usize); axis_bits.len()]]);
    while leaves.len() < leaf_target {
        let splittable: Vec<InputKey> = leaves
            .iter()
            .filter(|key| {
                key.iter()
                    .zip(axis_bits)
                    .any(|((depth, _), bits)| depth < bits)
            })
            .cloned()
            .collect();
        let key = splittable[rng.next() % splittable.len()].clone();
        let axes: Vec<usize> = (0..axis_bits.len())
            .filter(|axis| key[*axis].0 < axis_bits[*axis])
            .collect();
        let axis = axes[rng.next() % axes.len()];
        let (depth, value) = key[axis];
        let mut low = key.clone();
        low[axis] = (depth + 1, value << 1);
        let mut high = key.clone();
        high[axis] = (depth + 1, (value << 1) | 1);
        leaves.remove(&key);
        leaves.insert(low);
        leaves.insert(high);
    }
    leaves
}

/// Patch projectors for `keys`, each carrying the same spectator coordinates.
fn projectors<'a>(
    geometry: &ScheduleGeometry,
    keys: impl IntoIterator<Item = &'a InputKey>,
    spectators: &[DynIndex],
) -> Vec<Projector> {
    keys.into_iter()
        .map(|key| {
            let mut projector = geometry.input_projector(key).expect("input projector");
            for site in spectators {
                projector
                    .insert(site.clone(), 0)
                    .expect("spectator coordinate");
            }
            projector
        })
        .collect()
}

/// The trie check answers exactly what the removed all-pairs scan answered.
///
/// Leaves share one spectator constraint, so two patch projectors agree on every
/// index both fix exactly when their selected prefixes nest on every axis. The
/// only exception is equal keys, which the scan skipped and production rejects
/// before the check; that case pins the `allow_equal` parameter.
#[test]
fn trie_overlap_check_matches_the_pairwise_prefix_scan() {
    let sites: Vec<DynIndex> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let groups = vec![
        CoordinateGroup::new(sites[..2].to_vec()),
        CoordinateGroup::new(sites[2..4].to_vec()),
    ];
    let geometry = ScheduleGeometry::new(&sites[..4], Some(&groups)).expect("geometry");
    let spectators = &sites[4..];
    let axis_bits = [2usize, 2];

    // No pair: the empty preimage and one leaf.
    assert!(!all_pairs_overlap(&BTreeSet::new()));
    assert!(!has_overlapping_pair(&[], false));
    let mut rng = Lcg(0x1325);
    let single = split_layout(&axis_bits, 1, &mut rng);
    let single_refs = projectors(&geometry, single.iter(), spectators);
    let single_refs: Vec<&Projector> = single_refs.iter().collect();
    assert!(!has_overlapping_pair(&single_refs, false));

    for seed in 0..8usize {
        let mut rng = Lcg(seed as u64 * 977 + 1);
        let keys = split_layout(&axis_bits, 12, &mut rng);

        // A prefix-free layout has no overlapping pair.
        assert!(!all_pairs_overlap(&keys));
        let refs = projectors(&geometry, keys.iter(), spectators);
        let refs: Vec<&Projector> = refs.iter().collect();
        assert!(!has_overlapping_pair(&refs, false));

        // A strict ancestor of a leaf overlaps that leaf, so both agree again.
        let leaf = keys
            .iter()
            .find(|key| key.iter().any(|(depth, _)| *depth > 0))
            .expect("a split leaf")
            .clone();
        let axis = leaf
            .iter()
            .position(|(depth, _)| *depth > 0)
            .expect("a split axis");
        let mut ancestor = leaf.clone();
        ancestor[axis] = (leaf[axis].0 - 1, leaf[axis].1 >> 1);
        let mut overlapping = keys.clone();
        overlapping.insert(ancestor);
        assert!(all_pairs_overlap(&overlapping));
        let ancestor_refs = projectors(&geometry, overlapping.iter(), spectators);
        let ancestor_refs: Vec<&Projector> = ancestor_refs.iter().collect();
        assert!(has_overlapping_pair(&ancestor_refs, false));

        // Equal keys: skipped by the scan, rejected earlier in production, and
        // the one input where `allow_equal` decides.
        let duplicate = keys.iter().next().expect("a leaf").clone();
        let duplicate_refs = projectors(
            &geometry,
            keys.iter().chain(std::iter::once(&duplicate)),
            spectators,
        );
        let duplicate_refs: Vec<&Projector> = duplicate_refs.iter().collect();
        assert!(!has_overlapping_pair(&duplicate_refs, true));
        assert!(has_overlapping_pair(&duplicate_refs, false));
    }
}

/// A bond-dimension-one chain over `sites`: every patch is nonzero under any
/// full-depth dyadic projection, so the targets below have real contents.
fn product_chain(sites: &[DynIndex]) -> TreeTN<IdxTensor, usize> {
    let bonds: Vec<DynIndex> = (1..sites.len()).map(|_| DynIndex::new_dyn(1)).collect();
    let tensors = sites
        .iter()
        .enumerate()
        .map(|(node, site)| {
            let mut indices = vec![site.clone()];
            if node > 0 {
                indices.push(bonds[node - 1].clone());
            }
            if node + 1 < sites.len() {
                indices.push(bonds[node].clone());
            }
            let len: usize = indices.iter().map(|index| index.dim).product();
            IdxTensor::from_dense(indices, vec![1.0_f64; len]).expect("tensor")
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..sites.len()).collect()).expect("chain")
}

/// The full-depth dyadic leaf `leaf` of `axis_bits`, axis 0 taking the most
/// significant bits.
fn full_depth_projector(sites: &[DynIndex], axis_bits: &[usize], leaf: usize) -> Projector {
    let total: usize = axis_bits.iter().sum();
    let mut projector = Projector::new();
    let mut start = 0usize;
    let mut shift = total;
    for &bits in axis_bits {
        shift -= bits;
        let value = (leaf >> shift) & ((1usize << bits) - 1);
        for (position, site) in sites[start..start + bits].iter().enumerate() {
            let bit = (value >> (bits - 1 - position)) & 1;
            projector
                .insert(site.clone(), bit)
                .expect("projector coordinate");
        }
        start += bits;
    }
    projector
}

/// The per-axis prefix keys of every full-depth dyadic leaf, in the same bit
/// order as [`full_depth_projector`].
fn full_depth_keys(axis_bits: &[usize]) -> BTreeSet<InputKey> {
    let total: usize = axis_bits.iter().sum();
    (0..1usize << total)
        .map(|leaf| {
            let mut shift = total;
            axis_bits
                .iter()
                .map(|bits| {
                    shift -= bits;
                    (*bits, (leaf >> shift) & ((1usize << bits) - 1))
                })
                .collect::<InputKey>()
        })
        .collect()
}

/// A preimage of one full-depth dyadic leaf per selected coordinate assignment,
/// with its schedule geometry.
fn dyadic_preimage(
    sites: &[DynIndex],
    axis_bits: &[usize],
) -> (ReconstructionTarget<usize>, ScheduleGeometry) {
    let full = SubDomainTreeTN::from_treetn(product_chain(sites)).expect("subdomain");
    let total: usize = axis_bits.iter().sum();
    let leaves = (0..1usize << total)
        .map(|leaf| {
            full.project(&full_depth_projector(sites, axis_bits, leaf))
                .expect("project")
                .expect("nonzero leaf")
        })
        .collect::<Vec<_>>();
    let target = ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(leaves).expect("partition"),
    )
    .expect("target");
    let mut start = 0usize;
    let groups = axis_bits
        .iter()
        .map(|&bits| {
            let group = CoordinateGroup::new(sites[start..start + bits].to_vec());
            start += bits;
            group
        })
        .collect::<Vec<_>>();
    let geometry = ScheduleGeometry::new(sites, Some(&groups)).expect("geometry");
    (target, geometry)
}

/// Timing evidence for #1325: validation time grows near-linearly in the leaf
/// count, and the removed all-pairs scan grows quadratically.
///
/// Ignored because it is a measurement, not a gate: the absolute numbers depend on
/// the host. Run it pinned to one CPU of one L3 domain, in a release-equivalent
/// profile, and record the command and numbers in the PR:
///
/// ```text
/// taskset -c 8 bash -c 'taskset -c 0 cargo test --locked --profile ci \
///   -p tensor4all-partitionedtreetn --lib -- --ignored --nocapture leaf_validation'
/// ```
#[test]
#[ignore = "timing evidence, run explicitly with --ignored --nocapture"]
fn leaf_validation_is_near_linear_at_ten_thousand_leaves() {
    // Both phases measure the same three full-depth dyadic layouts. The removed
    // scan runs first on freshly generated key sets, so its times are not distorted
    // by the large preimages the trie phase builds.
    let sizes = [[6usize, 6], [7, 6], [7, 7]];

    let mut legacy = Vec::with_capacity(sizes.len());
    for axis_bits in sizes {
        let keys = full_depth_keys(&axis_bits);
        let start = Instant::now();
        assert!(!all_pairs_overlap(&keys));
        legacy.push((keys.len(), start.elapsed()));
    }

    println!("leaves   trie_ms   legacy_ms   speedup   trie_ratio");
    let mut previous: Option<(usize, Duration)> = None;
    for (axis_bits, (leaves, legacy)) in sizes.into_iter().zip(legacy) {
        let sites: Vec<DynIndex> = (0..axis_bits.iter().sum())
            .map(|_| DynIndex::new_dyn(2))
            .collect();
        let (target, geometry) = dyadic_preimage(&sites, &axis_bits);
        let mut trie = Duration::MAX;
        let mut keys = BTreeSet::new();
        for _ in 0..3 {
            let start = Instant::now();
            keys = validate_leaf_geometry(
                target.patch_projectors(),
                &geometry,
                CoverageContract::Complete,
            )
            .expect("full-depth dyadic leaves validate");
            trie = trie.min(start.elapsed());
        }
        assert_eq!(keys, full_depth_keys(&axis_bits));
        assert_eq!(keys.len(), leaves);

        let ratio = previous.map(|(_, time)| trie.as_secs_f64() / time.as_secs_f64());
        println!(
            "{leaves:6}  {:>8.3}  {:>9.3}  {:>6.1}x  {:>10}",
            trie.as_secs_f64() * 1e3,
            legacy.as_secs_f64() * 1e3,
            legacy.as_secs_f64() / trie.as_secs_f64(),
            ratio.map_or_else(|| "-".to_string(), |ratio| format!("{ratio:.2}")),
        );
        if let (Some((previous_leaves, _)), Some(ratio)) = (previous, ratio) {
            // 3.5 keeps headroom for a noisy host while still separating the
            // near-linear path (2x per doubling) from the removed scan (4x).
            assert!(
                ratio <= 3.5,
                "validation grew {ratio:.2}x from {previous_leaves} to {leaves} leaves; \
                 linear is 2x and the removed all-pairs scan is 4x"
            );
        }
        previous = Some((leaves, trie));
    }
}
