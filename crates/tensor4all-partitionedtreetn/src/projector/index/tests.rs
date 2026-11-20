use super::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn indices(dims: &[usize]) -> Vec<DynIndex> {
    dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect()
}

/// A projector fixing each index with probability `density`.
fn random_projector(indices: &[DynIndex], density: f64, rng: &mut ChaCha8Rng) -> Projector {
    let mut pairs = Vec::new();
    for index in indices {
        if rng.random_bool(density) {
            pairs.push((index.clone(), rng.random_range(0..index.dim)));
        }
    }
    Projector::from_pairs(pairs).unwrap()
}

/// The leaves of a random split tree along `indices` in a random order: a
/// disjoint, prefix-structured layout like the ones `patch_order` produces.
fn split_layout(indices: &[DynIndex], max_depth: usize, rng: &mut ChaCha8Rng) -> Vec<Projector> {
    fn split(
        projector: Projector,
        free: Vec<DynIndex>,
        depth: usize,
        rng: &mut ChaCha8Rng,
        leaves: &mut Vec<Projector>,
    ) {
        if depth == 0 || free.is_empty() || rng.random_bool(0.25) {
            leaves.push(projector);
            return;
        }
        let position = rng.random_range(0..free.len());
        let mut rest = free.clone();
        let index = rest.remove(position);
        for value in 0..index.dim {
            let mut child = projector.clone();
            child.insert(index.clone(), value).unwrap();
            split(child, rest.clone(), depth - 1, rng, leaves);
        }
    }
    let mut leaves = Vec::new();
    split(
        Projector::new(),
        indices.to_vec(),
        max_depth,
        rng,
        &mut leaves,
    );
    leaves
}

/// Random sets of every shape the callers see: split-tree layouts, random
/// partial projectors, both mixed, with equal keys and the empty projector.
fn random_sets(seed: u64) -> Vec<Vec<Projector>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let pool = indices(&[2, 2, 3, 2, 3, 2, 2, 4]);
    let mut sets = vec![Vec::new(), vec![Projector::new()]];
    for _ in 0..40 {
        let split = split_layout(&pool[..rng.random_range(1..pool.len())], 5, &mut rng);
        let partial: Vec<Projector> = (0..rng.random_range(1..30))
            .map(|_| random_projector(&pool, rng.random_range(0.1..0.9), &mut rng))
            .collect();
        let mut mixed = split.clone();
        mixed.extend(partial.iter().take(3).cloned());
        let mut with_equal = split.clone();
        if let Some(first) = split.first() {
            with_equal.push(first.clone());
        }
        with_equal.push(Projector::new());
        sets.extend([split, partial, mixed, with_equal]);
    }
    sets
}

fn brute_overlapping(set: &[&Projector], query: &Projector) -> Vec<usize> {
    (0..set.len())
        .filter(|&member| set[member].is_compatible_with(query))
        .collect()
}

fn brute_has_pair(set: &[&Projector], allow_equal: bool) -> bool {
    (0..set.len()).any(|left| {
        (left + 1..set.len()).any(|right| {
            !(allow_equal && set[left] == set[right]) && set[left].is_compatible_with(set[right])
        })
    })
}

fn brute_sets_overlap(left: &[&Projector], right: &[&Projector], allow_equal: bool) -> bool {
    left.iter().any(|l| {
        right
            .iter()
            .any(|r| !(allow_equal && l == r) && l.is_compatible_with(r))
    })
}

#[test]
fn compatibility_agrees_with_the_intersection() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let pool = indices(&[2, 3, 2, 4, 2]);
    for _ in 0..2000 {
        let left = random_projector(&pool, 0.5, &mut rng);
        let right = random_projector(&pool, 0.5, &mut rng);
        assert_eq!(
            left.is_compatible_with(&right),
            left.intersection(&right).is_some()
        );
        assert_eq!(
            left.is_compatible_with(&right),
            right.is_compatible_with(&left)
        );
    }
}

#[test]
fn overlapping_members_match_the_pairwise_scan() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    // Queries may fix indices no member fixes, and fix none at all.
    let pool = indices(&[2, 2, 3, 2, 3, 2, 2, 4]);
    for set in random_sets(3) {
        let references: Vec<&Projector> = set.iter().collect();
        let index = ProjectorIndex::new(&references);
        let mut queries: Vec<Projector> = set.clone();
        queries.push(Projector::new());
        queries.extend((0..10).map(|_| random_projector(&pool, 0.4, &mut rng)));
        queries.push(Projector::from_pairs([(DynIndex::new_dyn(2), 1)]).unwrap());
        for query in &queries {
            assert_eq!(
                index.overlapping(query),
                brute_overlapping(&references, query),
                "set of {} members, query {query:?}",
                set.len()
            );
        }
    }
}

#[test]
fn overlapping_pairs_match_the_pairwise_scan() {
    for set in random_sets(4) {
        let references: Vec<&Projector> = set.iter().collect();
        for allow_equal in [false, true] {
            assert_eq!(
                has_overlapping_pair(&references, allow_equal),
                brute_has_pair(&references, allow_equal),
                "set of {} members, allow_equal {allow_equal}",
                set.len()
            );
        }
        assert_eq!(
            Projector::are_disjoint(&set),
            !brute_has_pair(&references, false)
        );
    }
}

#[test]
fn overlapping_sets_match_the_pairwise_scan() {
    let sets = random_sets(5);
    for (left, right) in sets.iter().zip(sets.iter().skip(1)) {
        let left: Vec<&Projector> = left.iter().collect();
        let right: Vec<&Projector> = right.iter().collect();
        for allow_equal in [false, true] {
            let expected = brute_sets_overlap(&left, &right, allow_equal);
            assert_eq!(sets_overlap(&left, &right, allow_equal), expected);
            assert_eq!(sets_overlap(&right, &left, allow_equal), expected);
        }
    }
}

#[test]
fn degenerate_sets() {
    let a = DynIndex::new_dyn(2);
    let zero = Projector::from_pairs([(a.clone(), 0)]).unwrap();
    let one = Projector::from_pairs([(a.clone(), 1)]).unwrap();
    let empty = Projector::new();

    let none: Vec<&Projector> = Vec::new();
    assert!(ProjectorIndex::new(&none).overlapping(&zero).is_empty());
    assert!(!has_overlapping_pair(&none, false));
    assert!(!sets_overlap(&none, &[&zero], false));

    // The empty projector overlaps every member and every query.
    let index = ProjectorIndex::new(&[&empty, &zero, &one]);
    assert_eq!(index.overlapping(&zero), vec![0, 1]);
    assert_eq!(index.overlapping(&empty), vec![0, 1, 2]);
    assert!(has_overlapping_pair(&[&empty, &zero], true));

    // Equal keys overlap unless allowed.
    assert!(has_overlapping_pair(&[&zero, &zero], false));
    assert!(!has_overlapping_pair(&[&zero, &zero, &one], true));
    assert!(sets_overlap(&[&zero], &[&zero], false));
    assert!(!sets_overlap(&[&zero], &[&zero, &one], true));
}
