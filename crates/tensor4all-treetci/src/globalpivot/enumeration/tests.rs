use super::{retain_pivot, EnumerationPoints};
use std::collections::BTreeSet;

/// Small eager reference for arbitrary overlapping blocks and anchor order.
fn reference(anchors: &[Vec<usize>], sides: &[Vec<usize>], dims: &[usize]) -> Vec<Vec<usize>> {
    let mut points = BTreeSet::new();
    for anchor in anchors {
        for side in sides {
            for mut code in 0..side.iter().map(|&site| dims[site]).product() {
                let mut point = anchor.clone();
                for &site in side {
                    point[site] = code % dims[site];
                    code /= dims[site];
                }
                points.insert(point);
            }
        }
    }
    points.into_iter().collect()
}

#[test]
fn sorted_union_matches_eager_reference_for_overlaps_and_mixed_radices() {
    let dims = [2, 1, 3, 2];
    let sides = vec![vec![3, 1, 0], vec![2, 0], vec![1, 3], vec![3, 1, 0]];
    let all_anchors: Vec<_> = (0..2)
        .flat_map(|a| (0..3).flat_map(move |b| (0..2).map(move |c| vec![a, 0, b, c])))
        .collect();
    // Every pair, including equal anchors and reversed order; overlapping
    // sides also share non-anchor points, which must not be repeated.
    for a in &all_anchors {
        for b in &all_anchors {
            let anchors = [a.clone(), b.clone()];
            let actual: Vec<_> = EnumerationPoints::new(&anchors, &sides, &dims).collect();
            assert_eq!(actual, reference(&anchors, &sides, &dims));
        }
    }
}

#[test]
fn empty_and_unit_blocks_are_finite() {
    let anchors = [vec![0, 0]];
    assert_eq!(EnumerationPoints::new(&[], &[vec![0]], &[2]).next(), None);
    assert_eq!(EnumerationPoints::new(&anchors, &[], &[1, 1]).next(), None);
    assert_eq!(
        EnumerationPoints::new(&anchors, &[vec![1, 0], vec![]], &[1, 1]).collect::<Vec<_>>(),
        anchors,
    );
}

#[test]
fn large_star_starts_with_only_sparse_cursor_storage() {
    let arms = 200;
    let n_sites = 5 * arms + 1;
    let dims: Vec<_> = (0..n_sites)
        .map(|site| if site == 0 { 1 } else { 2 })
        .collect();
    let sides: Vec<_> = (0..arms)
        .map(|arm| (1 + 5 * arm..6 + 5 * arm).collect())
        .collect();
    // Distinct anchors differ on the first arm. Its blocks coincide, but
    // the other arms contribute distinct blocks: 30,877 full-site points
    // (about 247 MB of index payload in an eager set).
    let mut anchors = vec![vec![0; n_sites]; 5];
    for (value, anchor) in anchors.iter_mut().enumerate() {
        for bit in 0..3 {
            anchor[1 + bit] = (value >> bit) & 1;
        }
    }
    let mut points = EnumerationPoints::new(&anchors, &sides, &dims);
    assert_eq!(points.heads.len(), 5 * arms);
    assert_eq!(
        points
            .heads
            .iter()
            .map(|head| head.0.coordinates.len())
            .sum::<usize>(),
        5 * arms * 5
    );
    assert!(points.heads.iter().all(|head| anchors
        .iter()
        .any(|anchor| anchor.as_ptr() == head.0.anchor.as_ptr())));
    assert_eq!(points.next(), Some(vec![0; n_sites]));
    let mut second = vec![0; n_sites];
    second[n_sites - 1] = 1;
    assert_eq!(points.next(), Some(second));
    // Consuming two points must not collect the remaining 30,875 points.
    assert_eq!(points.heads.len(), 5 * arms);
}

#[test]
fn accepted_points_stay_bounded_with_stable_ties() {
    for limit in [0, 1, 2, 5, 40] {
        let mut best = Vec::new();
        let mut reference = Vec::new();
        for point in 0..32 {
            let error = (point % 3) as f64;
            let index = vec![point];
            retain_pivot(&mut best, error, &index, limit);
            reference.push((error, index));
            assert!(best.len() <= limit);
        }
        reference.sort_by(|(a, _), (b, _)| b.total_cmp(a));
        reference.truncate(limit);
        assert_eq!(best, reference);
    }
}
