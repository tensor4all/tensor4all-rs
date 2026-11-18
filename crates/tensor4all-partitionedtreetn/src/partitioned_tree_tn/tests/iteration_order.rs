//! Public iteration yields the canonical patch order (#1366): it depends only
//! on the stored patches, not on insertion order or hash-map state.

use super::*;

/// The projector sequence of every public iteration path.
fn orders(partition: &PartitionedTreeTN) -> [Vec<Projector>; 4] {
    [
        partition.projectors().cloned().collect(),
        partition
            .iter()
            .map(|(projector, _)| projector.clone())
            .collect(),
        partition
            .values()
            .map(|patch| patch.projector().clone())
            .collect(),
        partition
            .into_iter()
            .map(|(projector, _)| projector.clone())
            .collect(),
    ]
}

#[test]
fn iteration_order_is_independent_of_insertion_order() {
    let fixture = BlockProduct::new(None, 1.0);
    let patches: Vec<SubDomainTreeTN> = fixture.a.values().cloned().collect();
    assert_eq!(patches.len(), BRANCHES * BRANCHES);
    let canonical: Vec<Projector> = fixture
        .a
        .ordered_patches()
        .into_iter()
        .map(|patch| patch.projector().clone())
        .collect();

    // Forward, reversed, and an interleaved permutation (stride 4 is coprime
    // with 9), each into a fresh map with its own hash state.
    let mut permutations = vec![patches.clone(), patches.iter().rev().cloned().collect()];
    permutations.push(
        (0..patches.len())
            .map(|position| patches[(4 * position) % patches.len()].clone())
            .collect(),
    );
    for permutation in permutations {
        let partition = PartitionedTreeTN::from_subdomains(permutation).unwrap();
        for order in orders(&partition) {
            assert_eq!(order, canonical);
        }
        let owned: Vec<Projector> = partition
            .into_iter()
            .map(|(projector, _)| projector)
            .collect();
        assert_eq!(owned, canonical);
    }
}

#[test]
fn iteration_order_follows_projector_coordinates() {
    let fixture = BlockProduct::new(None, 1.0);
    let blocks: Vec<(usize, usize)> = fixture
        .a
        .projectors()
        .map(|projector| {
            (
                projector.get(&fixture.row).unwrap(),
                projector.get(&fixture.mid).unwrap(),
            )
        })
        .collect();
    let mut expected = blocks.clone();
    expected.sort();
    expected.dedup();
    assert_eq!(expected.len(), BRANCHES * BRANCHES);
    assert_eq!(blocks, expected);
}
