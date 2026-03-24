use super::*;
use std::collections::HashSet;
use std::thread;

#[test]
fn test_id_generation() {
    let id1 = generate_id();
    let id2 = generate_id();
    let id3 = generate_id();

    // IDs should be unique (random generation, not sequential)
    assert_ne!(id1, id2);
    assert_ne!(id2, id3);
    assert_ne!(id1, id3);

    // IDs should be non-zero (very high probability with u64)
    assert_ne!(id1, 0);
    assert_ne!(id2, 0);
    assert_ne!(id3, 0);
}

#[test]
fn test_thread_local_rng_different_seeds() {
    const NUM_THREADS: usize = 4;
    const IDS_PER_THREAD: usize = 100;

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|_| {
            thread::spawn(|| {
                (0..IDS_PER_THREAD)
                    .map(|_| generate_id())
                    .collect::<Vec<_>>()
            })
        })
        .collect();

    let mut all_ids = HashSet::new();
    for handle in handles {
        let thread_ids = handle.join().unwrap();
        all_ids.extend(thread_ids);
    }

    assert_eq!(
        all_ids.len(),
        NUM_THREADS * IDS_PER_THREAD,
        "All IDs should be unique across threads"
    );
}

#[test]
fn test_index_like_basic() {
    let i: DynIndex = Index::new_dyn(5);

    // Test IndexLike methods
    assert_eq!(i.dim(), 5);

    // Test id() method
    let id = i.id();
    assert_eq!(*id, i.id);
}

#[test]
fn test_index_like_id_methods() {
    let i1: DynIndex = Index::new_dyn(5);
    let i2 = i1.clone();
    let i3: DynIndex = Index::new_dyn(5);

    // same_id should return true for clones
    assert!(i1.same_id(&i2));
    // same_id should return false for different indices
    assert!(!i1.same_id(&i3));

    // has_id should match by ID
    assert!(i1.has_id(i1.id()));
    assert!(i1.has_id(i2.id()));
    assert!(!i1.has_id(i3.id()));
}

#[test]
fn test_index_like_equality() {
    let i1: DynIndex = Index::new_dyn(5);
    let i2 = i1.clone();
    let i3: DynIndex = Index::new_dyn(5);

    // Same index (cloned) should be equal
    assert_eq!(i1, i2);
    // Different index (new ID) should not be equal
    assert_ne!(i1, i3);
}

#[test]
fn test_index_like_in_hashset() {
    let i1: DynIndex = Index::new_dyn(5);
    let i2 = i1.clone();
    let i3: DynIndex = Index::new_dyn(5);

    let mut set = HashSet::new();
    set.insert(i1.clone());

    // Clone of same index should be found
    assert!(set.contains(&i2));
    // Different index should not be found
    assert!(!set.contains(&i3));
}

#[test]
fn test_new_bond() {
    let bond: DynIndex = DynIndex::new_bond(10).unwrap();
    assert_eq!(bond.dim(), 10);

    // Each new_bond creates a unique index
    let bond2: DynIndex = DynIndex::new_bond(10).unwrap();
    assert_ne!(bond, bond2);
}

#[test]
fn test_sim() {
    let tags = TagSet::from_str("Site,x=1").unwrap();
    let i1 = Index::<DynId>::new_dyn_with_tags(5, tags);

    // Create a similar index
    let i2 = i1.sim();

    // Different ID (not equal)
    assert_ne!(i1, i2);
    assert!(!i1.same_id(&i2));

    // Same dimension
    assert_eq!(i1.dim(), i2.dim());

    // Same tags
    assert_eq!(i1.tags, i2.tags);
    assert!(i2.tags.has_tag("Site"));
    assert!(i2.tags.has_tag("x=1"));
}

fn _assert_index_like_bounds<I: IndexLike>() {}

#[test]
fn test_index_satisfies_index_like() {
    // Compile-time check that DynIndex implements IndexLike
    _assert_index_like_bounds::<DynIndex>();
}

#[test]
fn test_conj_state_undirected() {
    let i: DynIndex = Index::new_dyn(5);
    assert_eq!(i.conj_state(), crate::ConjState::Undirected);
}

#[test]
fn test_conj_undirected_noop() {
    let i: DynIndex = Index::new_dyn(5);
    let i_conj = i.conj();
    // For undirected indices, conj() should be a no-op
    assert_eq!(i, i_conj);
    assert_eq!(i.conj_state(), i_conj.conj_state());
}

#[test]
fn test_is_contractable_undirected() {
    let i1: DynIndex = Index::new_dyn(5);
    let i2 = i1.clone();
    let i3: DynIndex = Index::new_dyn(5);

    // Same index (clone) should be contractable
    assert!(i1.is_contractable(&i2));
    // Different index (different ID) should not be contractable
    assert!(!i1.is_contractable(&i3));
}

#[test]
fn test_is_contractable_same_id_dim() {
    let i1: DynIndex = Index::new_dyn(5);
    let i2 = i1.clone();
    let i3: DynIndex = Index::new_dyn(3);

    // Same ID and dim should be contractable for undirected
    assert!(i1.is_contractable(&i2));
    // Different dim should not be contractable even if same ID
    // (but in practice, different IDs are used)
    assert!(!i1.is_contractable(&i3));
}
