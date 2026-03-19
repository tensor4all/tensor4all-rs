
use super::*;
use tensor4all_core::index::Index;

fn make_index(size: usize) -> DynIndex {
    Index::new_dyn(size)
}

#[test]
fn test_projector_new() {
    let p = Projector::new();
    assert!(p.is_empty());
    assert_eq!(p.len(), 0);
}

#[test]
fn test_projector_from_pairs() {
    let idx0 = make_index(2);
    let idx1 = make_index(3);
    let idx2 = make_index(4);

    let p = Projector::from_pairs([(idx0.clone(), 1), (idx2.clone(), 3)]);
    assert_eq!(p.len(), 2);
    assert!(p.is_projected_at(&idx0));
    assert!(!p.is_projected_at(&idx1));
    assert!(p.is_projected_at(&idx2));
    assert_eq!(p.get(&idx0), Some(1));
    assert_eq!(p.get(&idx1), None);
    assert_eq!(p.get(&idx2), Some(3));
}

#[test]
fn test_projector_intersection_compatible() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);
    let idx2 = make_index(2);

    let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let b = Projector::from_pairs([(idx1.clone(), 0), (idx2.clone(), 1)]);

    let merged = a.intersection(&b).unwrap();
    assert_eq!(merged.len(), 3);
    assert_eq!(merged.get(&idx0), Some(1));
    assert_eq!(merged.get(&idx1), Some(0));
    assert_eq!(merged.get(&idx2), Some(1));
}

#[test]
fn test_projector_intersection_conflict() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);

    let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let b = Projector::from_pairs([(idx1.clone(), 1)]); // Conflict at idx1

    assert!(a.intersection(&b).is_none());
}

#[test]
fn test_projector_common_restriction() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);
    let idx2 = make_index(2);

    let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let b = Projector::from_pairs([(idx1.clone(), 0), (idx2.clone(), 1)]);

    // Only idx1 is in both with the same value
    let common = a.common_restriction(&b);
    assert_eq!(common.len(), 1);
    assert!(!common.is_projected_at(&idx0));
    assert!(common.is_projected_at(&idx1));
    assert_eq!(common.get(&idx1), Some(0));
    assert!(!common.is_projected_at(&idx2));
}

#[test]
fn test_projector_is_compatible_with() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);

    let a = Projector::from_pairs([(idx0.clone(), 1)]);
    let b = Projector::from_pairs([(idx1.clone(), 0)]);
    let c = Projector::from_pairs([(idx0.clone(), 0)]); // Different value at same index

    assert!(a.is_compatible_with(&b)); // No common indices, compatible
    assert!(!a.is_compatible_with(&c)); // Same index, different values
}

#[test]
fn test_projector_is_subset_of() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);
    let idx2 = make_index(2);

    let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0), (idx2.clone(), 1)]);
    let b = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let c = Projector::from_pairs([(idx0.clone(), 1)]);

    assert!(a.is_subset_of(&b)); // a projects more indices
    assert!(a.is_subset_of(&c));
    assert!(b.is_subset_of(&c));
    assert!(!b.is_subset_of(&a));
    assert!(!c.is_subset_of(&a));
}

#[test]
fn test_projector_are_disjoint() {
    let idx0 = make_index(2);

    // Disjoint projectors: different values at same index
    let p1 = Projector::from_pairs([(idx0.clone(), 0)]);
    let p2 = Projector::from_pairs([(idx0.clone(), 1)]);

    assert!(Projector::are_disjoint(&[p1.clone(), p2.clone()]));

    // Non-disjoint: same projection
    let p3 = Projector::from_pairs([(idx0.clone(), 0)]);
    assert!(!Projector::are_disjoint(&[p1, p3]));
}

#[test]
fn test_projector_partial_ord() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);

    let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let b = Projector::from_pairs([(idx0.clone(), 1)]);
    let c = Projector::from_pairs([(idx0.clone(), 0)]); // Incompatible with a and b

    assert!(a < b);
    assert!(b > a);
    assert_eq!(a.partial_cmp(&c), None);
    assert_eq!(b.partial_cmp(&c), None);
}

#[test]
fn test_projector_iteration() {
    let idx0 = make_index(2);
    let idx1 = make_index(3);

    let p = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 2)]);

    let pairs: Vec<_> = p.into_iter().collect();
    assert_eq!(pairs.len(), 2);
}

#[test]
fn test_projector_equality_and_hash() {
    use std::collections::HashSet;

    let idx0 = make_index(2);
    let idx1 = make_index(2);

    let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let b = Projector::from_pairs([(idx1.clone(), 0), (idx0.clone(), 1)]); // Same content
    let c = Projector::from_pairs([(idx0.clone(), 1)]);

    assert_eq!(a, b);
    assert_ne!(a, c);

    let mut set = HashSet::new();
    set.insert(a.clone());
    assert!(set.contains(&b));
    assert!(!set.contains(&c));
}

#[test]
fn test_projector_filter_indices() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);
    let idx2 = make_index(2);

    let p = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0), (idx2.clone(), 1)]);

    let filtered = p.filter_indices(&[idx0.clone(), idx2.clone()]);
    assert_eq!(filtered.len(), 2);
    assert!(filtered.is_projected_at(&idx0));
    assert!(!filtered.is_projected_at(&idx1));
    assert!(filtered.is_projected_at(&idx2));
}
