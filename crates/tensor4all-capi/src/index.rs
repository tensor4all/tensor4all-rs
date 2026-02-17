//! C API functions for Index type
//!
//! Provides functions to create, access, and manipulate tensor indices.

use crate::types::{t4a_index, InternalIndex};
use crate::{
    StatusCode, T4A_BUFFER_TOO_SMALL, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
    T4A_TAG_OVERFLOW, T4A_TAG_TOO_LONG,
};
use std::ffi::{c_char, CStr};
use tensor4all_core::index::Index;
use tensor4all_core::tagset::TagSetError;

// Generate common lifecycle functions
impl_opaque_type_common!(index);

// ============================================================================
// Constructors
// ============================================================================

/// Create a new index with the given dimension
///
/// # Arguments
/// * `dim` - The dimension of the index (must be > 0)
///
/// # Returns
/// A new index pointer, or null if creation fails
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_new(dim: usize) -> *mut t4a_index {
    if dim == 0 {
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let index = Index::new_dyn(dim);
        Box::into_raw(Box::new(t4a_index::new(index)))
    }));

    crate::unwrap_catch_ptr(result)
}

/// Create a new index with the given dimension and tags (comma-separated)
///
/// # Arguments
/// * `dim` - The dimension of the index (must be > 0)
/// * `tags_csv` - Comma-separated tags (e.g., "Site,n=1"), or null for no tags
///
/// # Returns
/// A new index pointer, or null if creation fails (e.g., too many tags)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_new_with_tags(dim: usize, tags_csv: *const c_char) -> *mut t4a_index {
    if dim == 0 {
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if tags_csv.is_null() {
            let index: InternalIndex = Index::new_dyn(dim);
            return Box::into_raw(Box::new(t4a_index::new(index)));
        }

        let c_str = unsafe { CStr::from_ptr(tags_csv) };
        let tags_str = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => return crate::err_null(e),
        };

        // Use new_dyn_with_tag which handles comma-separated tags
        match Index::new_dyn_with_tag(dim, tags_str) {
            Ok(index) => Box::into_raw(Box::new(t4a_index::new(index))),
            Err(e) => crate::err_null(e),
        }
    }));

    crate::unwrap_catch_ptr(result)
}

/// Create a new index with the given dimension, id, and tags
///
/// # Arguments
/// * `dim` - The dimension of the index (must be > 0)
/// * `id` - The 64-bit ID (compatible with ITensors.jl's `IDType = UInt64`)
/// * `tags_csv` - Comma-separated tags (e.g., "Site,n=1"), or null for no tags
///
/// # Returns
/// A new index pointer, or null if creation fails
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_new_with_id(
    dim: usize,
    id: u64,
    tags_csv: *const c_char,
) -> *mut t4a_index {
    if dim == 0 {
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        use tensor4all_core::index::{DynId, TagSet};

        let tags = if tags_csv.is_null() {
            TagSet::new()
        } else {
            let c_str = unsafe { CStr::from_ptr(tags_csv) };
            let tags_str = match c_str.to_str() {
                Ok(s) => s,
                Err(e) => return crate::err_null(e),
            };
            match TagSet::from_str(tags_str) {
                Ok(t) => t,
                Err(e) => return crate::err_null(e),
            }
        };

        let index: InternalIndex = Index::new_with_size_and_tags(DynId(id), dim, tags);
        Box::into_raw(Box::new(t4a_index::new(index)))
    }));

    crate::unwrap_catch_ptr(result)
}

// ============================================================================
// Accessors
// ============================================================================

/// Get the dimension of an index
///
/// # Arguments
/// * `ptr` - Pointer to the index
/// * `out_dim` - Output pointer for the dimension
///
/// # Returns
/// Status code (T4A_SUCCESS or error code)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_dim(ptr: *const t4a_index, out_dim: *mut usize) -> StatusCode {
    if ptr.is_null() || out_dim.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let index = &*ptr;
        *out_dim = index.inner().size();
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the 64-bit ID of an index
///
/// Compatible with ITensors.jl's `IDType = UInt64`.
///
/// # Arguments
/// * `ptr` - Pointer to the index
/// * `out_id` - Output pointer for the ID
///
/// # Returns
/// Status code (T4A_SUCCESS or error code)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_id(ptr: *const t4a_index, out_id: *mut u64) -> StatusCode {
    if ptr.is_null() || out_id.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let index = &*ptr;
        *out_id = index.inner().id.0;
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the tags of an index as a comma-separated UTF-8 string
///
/// If `buf` is null, only writes the required buffer length to `out_len`.
/// Otherwise, writes the tags to `buf` (with null terminator) if it fits.
///
/// # Arguments
/// * `ptr` - Pointer to the index
/// * `buf` - Output buffer for the tags (can be null to query length)
/// * `buf_len` - Length of the buffer
/// * `out_len` - Output pointer for the required length (including null terminator)
///
/// # Returns
/// Status code (T4A_SUCCESS, T4A_BUFFER_TOO_SMALL, or error code)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_get_tags(
    ptr: *const t4a_index,
    buf: *mut u8,
    buf_len: usize,
    out_len: *mut usize,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let index = &*ptr;
        let tags = index.inner().tags();

        // Build comma-separated string using iterator
        let tag_strings: Vec<String> = tags.iter().map(|s| s.to_string()).collect();
        let tags_csv = tag_strings.join(",");
        let required_len = tags_csv.len() + 1; // +1 for null terminator

        *out_len = required_len;

        if buf.is_null() {
            return T4A_SUCCESS;
        }

        if buf_len < required_len {
            return T4A_BUFFER_TOO_SMALL;
        }

        // Copy to buffer
        std::ptr::copy_nonoverlapping(tags_csv.as_ptr(), buf, tags_csv.len());
        *buf.add(tags_csv.len()) = 0; // null terminator

        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Modifiers
// ============================================================================

/// Add a single tag to an index
///
/// # Arguments
/// * `ptr` - Pointer to the index
/// * `tag` - The tag to add (null-terminated C string)
///
/// # Returns
/// Status code (T4A_SUCCESS, T4A_TAG_OVERFLOW, T4A_TAG_TOO_LONG, or error code)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_add_tag(ptr: *mut t4a_index, tag: *const c_char) -> StatusCode {
    use tensor4all_core::TagSetLike;

    if ptr.is_null() || tag.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let index = &mut *ptr;
        let c_str = CStr::from_ptr(tag);

        let tag_str = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => return crate::err_status(e, T4A_INVALID_ARGUMENT),
        };

        // TagSet is Arc-wrapped, use TagSetLike::add_tag which does copy-on-write
        match index.inner_mut().tags.add_tag(tag_str) {
            Ok(()) => T4A_SUCCESS,
            Err(TagSetError::TooManyTags { .. }) => T4A_TAG_OVERFLOW,
            Err(TagSetError::TagTooLong { .. }) => T4A_TAG_TOO_LONG,
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Set all tags from a comma-separated string (replaces existing tags)
///
/// # Arguments
/// * `ptr` - Pointer to the index
/// * `tags_csv` - Comma-separated tags (e.g., "Site,n=1")
///
/// # Returns
/// Status code (T4A_SUCCESS, T4A_TAG_OVERFLOW, T4A_TAG_TOO_LONG, or error code)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_set_tags_csv(
    ptr: *mut t4a_index,
    tags_csv: *const c_char,
) -> StatusCode {
    use tensor4all_core::index::TagSet;

    if ptr.is_null() || tags_csv.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let index = &mut *ptr;
        let c_str = CStr::from_ptr(tags_csv);

        let tags_str = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => return crate::err_status(e, T4A_INVALID_ARGUMENT),
        };

        // Create new TagSet from the comma-separated string and replace existing
        match TagSet::from_str(tags_str) {
            Ok(new_tags) => {
                index.inner_mut().tags = new_tags;
                T4A_SUCCESS
            }
            Err(TagSetError::TooManyTags { .. }) => T4A_TAG_OVERFLOW,
            Err(TagSetError::TagTooLong { .. }) => T4A_TAG_TOO_LONG,
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Check if an index has a specific tag
///
/// # Arguments
/// * `ptr` - Pointer to the index
/// * `tag` - The tag to check for (null-terminated C string)
///
/// # Returns
/// 1 if the tag exists, 0 if not, negative on error
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_has_tag(ptr: *const t4a_index, tag: *const c_char) -> i32 {
    if ptr.is_null() || tag.is_null() {
        return -1;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let index = &*ptr;
        let c_str = CStr::from_ptr(tag);

        let tag_str = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => return crate::err_status(e, -1),
        };

        if index.inner().tags().has_tag(tag_str) {
            1
        } else {
            0
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

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
}
