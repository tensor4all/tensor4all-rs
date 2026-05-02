use super::*;
use crate::T4A_BUFFER_TOO_SMALL;
use std::collections::BTreeSet;
use std::ffi::{CStr, CString};

fn last_error() -> String {
    let mut len = 0usize;
    assert_eq!(
        crate::t4a_last_error_message(std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut buf = vec![0u8; len];
    assert_eq!(
        crate::t4a_last_error_message(buf.as_mut_ptr(), buf.len(), &mut len),
        T4A_SUCCESS
    );
    CStr::from_bytes_until_nul(&buf)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

fn read_tags(index: *const t4a_index) -> BTreeSet<String> {
    let mut len = 0usize;
    assert_eq!(
        t4a_index_tags(index, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut buf = vec![0u8; len];
    assert_eq!(
        t4a_index_tags(index, buf.as_mut_ptr(), buf.len(), &mut len),
        T4A_SUCCESS
    );
    let csv = CStr::from_bytes_until_nul(&buf)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    if csv.is_empty() {
        BTreeSet::new()
    } else {
        csv.split(',').map(str::to_string).collect()
    }
}

fn new_index(dim: usize, tags: Option<&str>, plev: i64) -> *mut t4a_index {
    let tags = tags.map(|s| CString::new(s).unwrap());
    let mut out = std::ptr::null_mut();
    let status = t4a_index_new(
        dim,
        tags.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        plev,
        &mut out,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!out.is_null());
    out
}

fn new_index_with_id(dim: usize, id: u64, tags: Option<&str>, plev: i64) -> *mut t4a_index {
    let tags = tags.map(|s| CString::new(s).unwrap());
    let mut out = std::ptr::null_mut();
    let status = t4a_index_new_with_id(
        dim,
        id,
        tags.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        plev,
        &mut out,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!out.is_null());
    out
}

fn index_equal(a: *const t4a_index, b: *const t4a_index) -> bool {
    let mut equal = -1i32;
    assert_eq!(t4a_index_equal(a, b, &mut equal), T4A_SUCCESS);
    equal != 0
}

fn index_hash(index: *const t4a_index) -> u64 {
    let mut hash = 0u64;
    assert_eq!(t4a_index_hash(index, &mut hash), T4A_SUCCESS);
    hash
}

fn index_id(index: *const t4a_index) -> u64 {
    let mut id = 0u64;
    assert_eq!(t4a_index_id(index, &mut id), T4A_SUCCESS);
    id
}

#[test]
fn test_index_new_rejects_zero_dim() {
    let mut out = std::ptr::null_mut();
    let status = t4a_index_new(0, std::ptr::null(), 0, &mut out);
    assert_eq!(status, T4A_INVALID_ARGUMENT);
    assert!(out.is_null());
    assert!(last_error().contains("dim must be greater than zero"));
}

#[test]
fn test_index_new_rejects_negative_plev() {
    let mut out = std::ptr::null_mut();
    let status = t4a_index_new(4, std::ptr::null(), -1, &mut out);
    assert_eq!(status, T4A_INVALID_ARGUMENT);
    assert!(out.is_null());
    assert!(last_error().contains("plev"));
}

#[test]
fn test_index_tags_roundtrip_and_has_tag() {
    let index = new_index(7, Some("Site,n=1"), 3);

    let mut dim = 0usize;
    let mut plev = -1i64;
    assert_eq!(t4a_index_dim(index, &mut dim), T4A_SUCCESS);
    assert_eq!(t4a_index_plev(index, &mut plev), T4A_SUCCESS);
    assert_eq!(dim, 7);
    assert_eq!(plev, 3);

    let tags = read_tags(index);
    assert_eq!(
        tags,
        ["Site".to_string(), "n=1".to_string()]
            .into_iter()
            .collect()
    );

    let site = CString::new("Site").unwrap();
    let missing = CString::new("Missing").unwrap();
    let mut has_tag = -1i32;
    assert_eq!(
        t4a_index_has_tag(index, site.as_ptr(), &mut has_tag),
        T4A_SUCCESS
    );
    assert_eq!(has_tag, 1);
    assert_eq!(
        t4a_index_has_tag(index, missing.as_ptr(), &mut has_tag),
        T4A_SUCCESS
    );
    assert_eq!(has_tag, 0);

    t4a_index_release(index);
}

#[test]
fn test_index_query_failures_set_last_error() {
    let mut dim = 0usize;
    assert_eq!(t4a_index_dim(std::ptr::null(), &mut dim), T4A_NULL_POINTER);
    assert_eq!(last_error(), "index is null");

    let index = new_index(2, Some("Site"), 0);
    let mut len = 0usize;
    let mut short = [0u8; 2];
    assert_eq!(
        t4a_index_tags(index, short.as_mut_ptr(), short.len(), &mut len),
        T4A_BUFFER_TOO_SMALL
    );
    assert_eq!(len, "Site".len() + 1);
    assert!(last_error().contains("index tags buffer too small"));

    t4a_index_release(index);
}

#[test]
fn test_index_accepts_long_tags_losslessly() {
    let long_tag = "Site3_Site4_Site5";
    let index = new_index(2, Some(long_tag), 0);

    let tags = read_tags(index);
    assert_eq!(tags, [long_tag.to_string()].into_iter().collect());

    let long_tag_c = CString::new(long_tag).unwrap();
    let mut has_tag = -1i32;
    assert_eq!(
        t4a_index_has_tag(index, long_tag_c.as_ptr(), &mut has_tag),
        T4A_SUCCESS
    );
    assert_eq!(has_tag, 1);

    t4a_index_release(index);
}

#[test]
fn test_index_clone_preserves_metadata() {
    let index = new_index(5, Some("Left,Link"), 9);

    let mut cloned = std::ptr::null_mut();
    assert_eq!(t4a_index_clone(index, &mut cloned), T4A_SUCCESS);
    assert!(!cloned.is_null());

    let mut dims = [0usize; 2];
    let mut plevs = [0i64; 2];
    for (slot, ptr) in [index as *const t4a_index, cloned as *const t4a_index]
        .into_iter()
        .enumerate()
    {
        assert_eq!(t4a_index_dim(ptr, &mut dims[slot]), T4A_SUCCESS);
        assert_eq!(t4a_index_plev(ptr, &mut plevs[slot]), T4A_SUCCESS);
    }

    assert_eq!(dims, [5, 5]);
    assert_eq!(plevs, [9, 9]);
    assert_eq!(read_tags(index), read_tags(cloned));
    assert!(index_equal(index, cloned));
    assert_eq!(index_hash(index), index_hash(cloned));

    t4a_index_release(cloned);
    t4a_index_release(index);
}

#[test]
fn test_index_new_with_id_roundtrips_identity() {
    let a = new_index_with_id(5, 42, Some("Left,Link"), 7);
    let b = new_index_with_id(5, 42, Some("Left,Link"), 7);
    let different_plev = new_index_with_id(5, 42, Some("Left,Link"), 8);
    let different_tags = new_index_with_id(5, 42, Some("Left,Other"), 7);
    let different_id = new_index_with_id(5, 43, Some("Left,Link"), 7);

    assert_eq!(index_id(a), 42);
    assert_eq!(index_id(b), 42);
    assert!(index_equal(a, b));
    assert_eq!(index_hash(a), index_hash(b));
    assert!(!index_equal(a, different_plev));
    assert!(!index_equal(a, different_tags));
    assert!(!index_equal(a, different_id));

    t4a_index_release(different_id);
    t4a_index_release(different_tags);
    t4a_index_release(different_plev);
    t4a_index_release(b);
    t4a_index_release(a);
}

#[test]
fn test_index_plev_transforms_preserve_identity_but_use_full_equality() {
    let index = new_index(4, Some("Custom"), 0);

    let mut primed: *mut t4a_index = std::ptr::null_mut();
    assert_eq!(t4a_index_prime(index, &mut primed), T4A_SUCCESS);
    assert!(!primed.is_null());

    let mut explicit: *mut t4a_index = std::ptr::null_mut();
    assert_eq!(t4a_index_set_plev(index, 1, &mut explicit), T4A_SUCCESS);
    assert!(!explicit.is_null());

    let mut unprimed: *mut t4a_index = std::ptr::null_mut();
    assert_eq!(t4a_index_noprime(primed, &mut unprimed), T4A_SUCCESS);
    assert!(!unprimed.is_null());

    let mut plev = -1i64;
    assert_eq!(t4a_index_plev(primed, &mut plev), T4A_SUCCESS);
    assert_eq!(plev, 1);
    assert_eq!(read_tags(index), read_tags(primed));
    assert!(!index_equal(index, primed));
    assert_ne!(index_hash(index), index_hash(primed));
    assert!(index_equal(primed, explicit));
    assert_eq!(index_hash(primed), index_hash(explicit));
    assert!(index_equal(index, unprimed));
    assert_eq!(index_hash(index), index_hash(unprimed));

    t4a_index_release(unprimed);
    t4a_index_release(explicit);
    t4a_index_release(primed);
    t4a_index_release(index);
}

#[test]
fn test_index_set_plev_rejects_negative_plev() {
    let index = new_index(4, None, 0);
    let mut out: *mut t4a_index = std::ptr::null_mut();

    let status = t4a_index_set_plev(index, -1, &mut out);

    assert_eq!(status, T4A_INVALID_ARGUMENT);
    assert!(out.is_null());
    assert!(last_error().contains("plev"));

    t4a_index_release(index);
}
