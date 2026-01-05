use tensor4all_core_common::index::{DefaultIndex as Index, DynId, TagSet};

#[test]
fn test_from_tags_comma_error() {
    // Comma in tag is an error (comma is reserved as separator in from_str)
    let result = TagSet::from_tags(&["Site,Link"]);
    assert!(result.is_err());

    // Valid usage
    let tags = TagSet::from_tags(&["Site", "Link"]).unwrap();
    assert_eq!(tags.len(), 2);
    assert!(tags.has_tag("Site"));
    assert!(tags.has_tag("Link"));
}

#[test]
fn test_index_dyn() {
    let idx = Index::new_dyn(8);
    assert_eq!(idx.size(), 8);
    assert!(idx.id.0 > 0);
}

#[test]
fn test_index_with_custom_id() {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct StaticId;

    let idx = Index::new_with_size(StaticId, 16);
    assert_eq!(idx.size(), 16);
    assert_eq!(idx.id, StaticId);
}

#[test]
fn test_index_size() {
    // With Arc<TagSet>, Index is now only 32 bytes (16 + 8 + 8)
    assert_eq!(std::mem::size_of::<Index<DynId>>(), 32);
}

#[test]
fn test_index_basic() {
    let idx = Index::<DynId>::new_dyn(8);
    assert_eq!(idx.size(), 8);
    assert!(idx.id.0 > 0);
}

#[test]
fn test_index_with_tag() {
    let idx = Index::<DynId>::new_dyn_with_tag(8, "Site").unwrap();
    assert_eq!(idx.size(), 8);
    assert!(idx.tags.has_tag("Site"));
}

#[test]
fn test_index_shared_tags() {
    // Create shared tags once
    let site_tags = TagSet::from_str("Site").unwrap();

    // Share the same tags across many indices (cheap clone via Arc)
    let i1 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());
    let i2 = Index::<DynId>::new_dyn_with_tags(3, site_tags.clone());
    let i3 = Index::<DynId>::new_dyn_with_tags(4, site_tags.clone());

    // Tags are accessible
    assert!(i1.tags.has_tag("Site"));
    assert!(i2.tags.has_tag("Site"));
    assert!(i3.tags.has_tag("Site"));

    // Sizes are correct
    assert_eq!(i1.size(), 2);
    assert_eq!(i2.size(), 3);
    assert_eq!(i3.size(), 4);
}

#[test]
fn test_index_clone_is_cheap() {
    let site_tags = TagSet::from_str("Site,Link").unwrap();
    let i1 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());

    // Clone the index
    let i2 = i1.clone();

    // IDs are same (cloned, not new)
    assert_eq!(i1.id, i2.id);

    // Tags are accessible on both
    assert!(i1.tags.has_tag("Site"));
    assert!(i2.tags.has_tag("Site"));
}

#[test]
fn test_index_equality() {
    let site_tags = TagSet::from_str("Site").unwrap();
    let i1 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());
    let i2 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());

    // Different IDs, so not equal
    assert_ne!(i1, i2);

    // Clone has same ID
    let i3 = i1.clone();
    assert_eq!(i1, i3);
}
