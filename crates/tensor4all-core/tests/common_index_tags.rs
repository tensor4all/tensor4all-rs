use tensor4all_core::index::{DefaultIndex as Index, TagSet};

#[test]
fn test_index_with_tags() {
    // Tags are immutable - create with tags from the start
    let idx = Index::new_dyn_with_tag(8, "site,up").unwrap();

    assert_eq!(idx.tags().len(), 2);
    assert!(idx.tags().has_tag("site"));
    assert!(idx.tags().has_tag("up"));
}

#[test]
fn test_index_equality_by_id_only() {
    let idx1 = Index::new_dyn_with_tag(8, "site").unwrap();
    let idx2 = Index::new_dyn_with_tag(8, "bond").unwrap();

    // Different IDs, so not equal
    assert_ne!(idx1, idx2);

    // Same ID, should be equal even with different tags
    let id = idx1.id;
    let idx3 = Index {
        id,
        symm: idx1.symm,
        tags: TagSet::from_str("bond").unwrap(),
    };

    // Now they should be equal because IDs match (tags are ignored)
    assert_eq!(idx1, idx3);
}

#[test]
fn test_index_new_with_tags() {
    let tags = TagSet::from_str("site,up").unwrap();
    let idx = Index::new_dyn_with_tags(8, tags);

    assert_eq!(idx.tags().len(), 2);
    assert!(idx.tags().has_tag("site"));
    assert!(idx.tags().has_tag("up"));
}

#[test]
fn test_index_hash_by_id_only() {
    use std::collections::HashSet;

    let idx1 = Index::new_dyn_with_tag(8, "site").unwrap();
    let idx2 = Index::new_dyn_with_tag(8, "bond").unwrap();

    let mut set = HashSet::new();
    set.insert(idx1.clone());
    set.insert(idx2.clone());

    // Should have 2 entries (different IDs)
    assert_eq!(set.len(), 2);

    // Create another index with same ID as idx1 but different tags
    let idx3 = Index {
        id: idx1.id,
        symm: idx1.symm,
        tags: TagSet::from_str("bond").unwrap(),
    };

    // Should not add because ID matches idx1
    set.insert(idx3);
    assert_eq!(set.len(), 2);
}

#[test]
fn test_index_tags_immutability() {
    // Tags are immutable once created
    let idx = Index::new_dyn_with_tag(8, "site,up").unwrap();

    // Verify tags
    assert_eq!(idx.tags().len(), 2);
    assert!(idx.tags().has_tag("site"));
    assert!(idx.tags().has_tag("up"));

    // To "change" tags, create a new index with the same ID
    let new_tags = TagSet::from_str("bond").unwrap();
    let idx2 = Index {
        id: idx.id,
        symm: idx.symm,
        tags: new_tags,
    };

    assert_eq!(idx2.tags().len(), 1);
    assert!(idx2.tags().has_tag("bond"));
    assert!(!idx2.tags().has_tag("site"));

    // Original index still has original tags
    assert!(idx.tags().has_tag("site"));
}
