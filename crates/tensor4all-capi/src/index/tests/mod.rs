use super::*;
use std::ffi::CString;

fn get_tags_csv(idx: *const t4a_index) -> String {
    let mut len = 0usize;
    assert_eq!(
        t4a_index_get_tags(idx, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut buf = vec![0u8; len];
    assert_eq!(
        t4a_index_get_tags(idx, buf.as_mut_ptr(), buf.len(), &mut len),
        T4A_SUCCESS
    );
    CStr::from_bytes_until_nul(&buf)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

fn get_plev(idx: *const t4a_index) -> i64 {
    let mut plev = i64::MIN;
    assert_eq!(t4a_index_get_plev(idx, &mut plev), T4A_SUCCESS);
    plev
}

#[test]
fn test_index_new() {
    let idx = t4a_index_new(5);
    assert!(!idx.is_null());

    let mut dim: usize = 0;
    let status = t4a_index_dim(idx, &mut dim);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(dim, 5);

    t4a_index_release(idx);
}

#[test]
fn test_index_new_zero_dim() {
    let idx = t4a_index_new(0);
    assert!(idx.is_null());
}

#[test]
fn test_index_with_tags() {
    let tags = CString::new("Site,n=1").unwrap();
    let idx = t4a_index_new_with_tags(3, tags.as_ptr());
    assert!(!idx.is_null());

    // Check has_tag
    let site_tag = CString::new("Site").unwrap();
    assert_eq!(t4a_index_has_tag(idx, site_tag.as_ptr()), 1);

    let n_tag = CString::new("n=1").unwrap();
    assert_eq!(t4a_index_has_tag(idx, n_tag.as_ptr()), 1);

    let missing_tag = CString::new("Missing").unwrap();
    assert_eq!(t4a_index_has_tag(idx, missing_tag.as_ptr()), 0);

    t4a_index_release(idx);
}

#[test]
fn test_index_get_tags() {
    let tags = CString::new("Site,Link").unwrap();
    let idx = t4a_index_new_with_tags(2, tags.as_ptr());
    assert!(!idx.is_null());

    // Query required length
    let mut len: usize = 0;
    let status = t4a_index_get_tags(idx, std::ptr::null_mut(), 0, &mut len);
    assert_eq!(status, T4A_SUCCESS);
    assert!(len > 0);

    // Get tags
    let mut buf = vec![0u8; len];
    let status = t4a_index_get_tags(idx, buf.as_mut_ptr(), len, &mut len);
    assert_eq!(status, T4A_SUCCESS);

    let result = CStr::from_bytes_until_nul(&buf).unwrap().to_str().unwrap();
    assert!(result.contains("Site"));
    assert!(result.contains("Link"));

    t4a_index_release(idx);
}

#[test]
fn test_index_id() {
    let idx = t4a_index_new(4);
    assert!(!idx.is_null());

    let mut id: u64 = 0;
    let status = t4a_index_id(idx, &mut id);
    assert_eq!(status, T4A_SUCCESS);

    // ID should be non-zero (random)
    assert_ne!(id, 0);

    t4a_index_release(idx);
}

#[test]
fn test_index_clone() {
    let idx = t4a_index_new(5);
    assert!(!idx.is_null());

    let cloned = t4a_index_clone(idx);
    assert!(!cloned.is_null());

    // Both should have same dimension
    let mut dim1: usize = 0;
    let mut dim2: usize = 0;
    t4a_index_dim(idx, &mut dim1);
    t4a_index_dim(cloned, &mut dim2);
    assert_eq!(dim1, dim2);

    // Both should have same ID
    let (mut id1, mut id2) = (0u64, 0u64);
    t4a_index_id(idx, &mut id1);
    t4a_index_id(cloned, &mut id2);
    assert_eq!(id1, id2);

    t4a_index_release(idx);
    t4a_index_release(cloned);
}

#[test]
fn test_index_plev_default() {
    let idx = t4a_index_new(5);
    assert!(!idx.is_null());

    assert_eq!(get_plev(idx), 0);

    t4a_index_release(idx);
}

#[test]
fn test_index_set_plev() {
    let idx = t4a_index_new(5);
    assert!(!idx.is_null());

    assert_eq!(t4a_index_set_plev(idx, 4), T4A_SUCCESS);
    assert_eq!(get_plev(idx), 4);

    t4a_index_release(idx);
}

#[test]
fn test_index_prime() {
    let idx = t4a_index_new(5);
    assert!(!idx.is_null());

    assert_eq!(t4a_index_prime(idx), T4A_SUCCESS);
    assert_eq!(t4a_index_prime(idx), T4A_SUCCESS);
    assert_eq!(get_plev(idx), 2);

    t4a_index_release(idx);
}

#[test]
fn test_index_plev_null_checks() {
    let idx = t4a_index_new(5);
    assert!(!idx.is_null());

    let mut plev = 0i64;
    assert_eq!(
        t4a_index_get_plev(std::ptr::null(), &mut plev),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_index_get_plev(idx, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_index_set_plev(std::ptr::null_mut(), 1),
        T4A_NULL_POINTER
    );
    assert_eq!(t4a_index_prime(std::ptr::null_mut()), T4A_NULL_POINTER);

    t4a_index_release(idx);
}

#[test]
fn test_index_clone_preserves_plev() {
    let idx = t4a_index_new(5);
    assert!(!idx.is_null());
    assert_eq!(t4a_index_set_plev(idx, 9), T4A_SUCCESS);

    let cloned = t4a_index_clone(idx);
    assert!(!cloned.is_null());
    assert_eq!(get_plev(idx), 9);
    assert_eq!(get_plev(cloned), 9);

    t4a_index_release(idx);
    t4a_index_release(cloned);
}

#[test]
fn test_index_new_with_id() {
    let id: u64 = 0x12345678_9ABCDEF0;
    let tags = CString::new("Custom").unwrap();

    let idx = t4a_index_new_with_id(7, id, tags.as_ptr());
    assert!(!idx.is_null());

    // Verify ID
    let mut out_id: u64 = 0;
    t4a_index_id(idx, &mut out_id);
    assert_eq!(out_id, id);

    // Verify dimension
    let mut dim: usize = 0;
    t4a_index_dim(idx, &mut dim);
    assert_eq!(dim, 7);

    t4a_index_release(idx);
}

#[test]
fn test_index_constructors_with_null_tags_create_empty_tagsets() {
    let idx = t4a_index_new_with_tags(3, std::ptr::null());
    assert!(!idx.is_null());
    assert_eq!(get_tags_csv(idx), "");
    t4a_index_release(idx);

    let id: u64 = 0xDEADBEEF;
    let idx = t4a_index_new_with_id(4, id, std::ptr::null());
    assert!(!idx.is_null());

    let mut out_id = 0u64;
    assert_eq!(t4a_index_id(idx, &mut out_id), T4A_SUCCESS);
    assert_eq!(out_id, id);
    assert_eq!(get_tags_csv(idx), "");

    t4a_index_release(idx);
}

#[test]
fn test_index_tag_modifiers_and_get_tags_buffer_contract() {
    let idx = t4a_index_new(6);
    assert!(!idx.is_null());

    let site = CString::new("Site").unwrap();
    let n1 = CString::new("n=1").unwrap();
    assert_eq!(t4a_index_add_tag(idx, site.as_ptr()), T4A_SUCCESS);
    assert_eq!(t4a_index_add_tag(idx, n1.as_ptr()), T4A_SUCCESS);
    assert_eq!(t4a_index_has_tag(idx, site.as_ptr()), 1);
    assert_eq!(t4a_index_has_tag(idx, n1.as_ptr()), 1);

    let replacement = CString::new("Left,Link").unwrap();
    assert_eq!(
        t4a_index_set_tags_csv(idx, replacement.as_ptr()),
        T4A_SUCCESS
    );
    assert_eq!(t4a_index_has_tag(idx, site.as_ptr()), 0);

    let left = CString::new("Left").unwrap();
    let link = CString::new("Link").unwrap();
    assert_eq!(t4a_index_has_tag(idx, left.as_ptr()), 1);
    assert_eq!(t4a_index_has_tag(idx, link.as_ptr()), 1);

    let mut required_len = 0usize;
    assert_eq!(
        t4a_index_get_tags(idx, std::ptr::null_mut(), 0, &mut required_len),
        T4A_SUCCESS
    );
    let mut small_buf = vec![0u8; required_len.saturating_sub(1)];
    assert_eq!(
        t4a_index_get_tags(
            idx,
            small_buf.as_mut_ptr(),
            small_buf.len(),
            &mut required_len
        ),
        T4A_BUFFER_TOO_SMALL
    );

    let tags_csv = get_tags_csv(idx);
    assert!(tags_csv.contains("Left"));
    assert!(tags_csv.contains("Link"));

    t4a_index_release(idx);
}

#[test]
fn test_index_validates_null_pointers_utf8_and_tag_limits() {
    let idx = t4a_index_new(5);
    assert!(!idx.is_null());

    let mut dim = 0usize;
    let mut id = 0u64;
    let mut plev = 0i64;
    let mut out_len = 0usize;
    assert_eq!(t4a_index_dim(std::ptr::null(), &mut dim), T4A_NULL_POINTER);
    assert_eq!(t4a_index_dim(idx, std::ptr::null_mut()), T4A_NULL_POINTER);
    assert_eq!(t4a_index_id(std::ptr::null(), &mut id), T4A_NULL_POINTER);
    assert_eq!(t4a_index_id(idx, std::ptr::null_mut()), T4A_NULL_POINTER);
    assert_eq!(
        t4a_index_get_plev(std::ptr::null(), &mut plev),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_index_get_plev(idx, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_index_set_plev(std::ptr::null_mut(), 1),
        T4A_NULL_POINTER
    );
    assert_eq!(t4a_index_prime(std::ptr::null_mut()), T4A_NULL_POINTER);
    assert_eq!(
        t4a_index_get_tags(std::ptr::null(), std::ptr::null_mut(), 0, &mut out_len),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_index_get_tags(idx, std::ptr::null_mut(), 0, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_index_add_tag(std::ptr::null_mut(), std::ptr::null()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_index_set_tags_csv(std::ptr::null_mut(), std::ptr::null()),
        T4A_NULL_POINTER
    );
    assert_eq!(t4a_index_has_tag(std::ptr::null(), std::ptr::null()), -1);

    let invalid_utf8 = c"\xff".as_ptr();
    assert_eq!(t4a_index_add_tag(idx, invalid_utf8), T4A_INVALID_ARGUMENT);
    assert_eq!(
        t4a_index_set_tags_csv(idx, invalid_utf8),
        T4A_INVALID_ARGUMENT
    );
    assert_eq!(t4a_index_has_tag(idx, invalid_utf8), -1);
    assert!(t4a_index_new_with_tags(3, invalid_utf8).is_null());
    assert!(t4a_index_new_with_id(3, 42, invalid_utf8).is_null());

    let too_long = CString::new("1234567890abcdefg").unwrap();
    assert_eq!(t4a_index_add_tag(idx, too_long.as_ptr()), T4A_TAG_TOO_LONG);

    let overflow = CString::new("t1,t2,t3,t4,t5").unwrap();
    assert_eq!(
        t4a_index_set_tags_csv(idx, overflow.as_ptr()),
        T4A_TAG_OVERFLOW
    );

    t4a_index_release(idx);
}
