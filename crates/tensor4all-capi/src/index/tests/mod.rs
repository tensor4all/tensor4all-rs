use super::*;
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
fn test_index_clone_preserves_metadata() {
    let index = new_index(5, Some("Left,Link"), 9);

    let mut cloned = std::ptr::null_mut();
    assert_eq!(t4a_index_clone(index, &mut cloned), T4A_SUCCESS);
    assert!(!cloned.is_null());

    let mut dims = [0usize; 2];
    let mut plevs = [0i64; 2];
    let mut ids = [0u64; 2];
    for (slot, ptr) in [index as *const t4a_index, cloned as *const t4a_index]
        .into_iter()
        .enumerate()
    {
        assert_eq!(t4a_index_dim(ptr, &mut dims[slot]), T4A_SUCCESS);
        assert_eq!(t4a_index_plev(ptr, &mut plevs[slot]), T4A_SUCCESS);
        assert_eq!(t4a_index_id(ptr, &mut ids[slot]), T4A_SUCCESS);
    }

    assert_eq!(dims, [5, 5]);
    assert_eq!(plevs, [9, 9]);
    assert_eq!(ids[0], ids[1]);
    assert_eq!(read_tags(index), read_tags(cloned));

    t4a_index_release(cloned);
    t4a_index_release(index);
}

#[test]
fn test_index_new_with_id_uses_explicit_id() {
    let tags = CString::new("Custom").unwrap();
    let mut out = std::ptr::null_mut();
    let status = t4a_index_new_with_id(4, 0x1234_5678_9abc_def0, tags.as_ptr(), 2, &mut out);
    assert_eq!(status, T4A_SUCCESS);
    assert!(!out.is_null());

    let mut id = 0u64;
    let mut plev = 0i64;
    assert_eq!(t4a_index_id(out, &mut id), T4A_SUCCESS);
    assert_eq!(t4a_index_plev(out, &mut plev), T4A_SUCCESS);
    assert_eq!(id, 0x1234_5678_9abc_def0);
    assert_eq!(plev, 2);
    assert_eq!(read_tags(out), ["Custom".to_string()].into_iter().collect());

    t4a_index_release(out);
}
