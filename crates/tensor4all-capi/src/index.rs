//! C API for immutable `Index` handles.

use std::ffi::{c_char, CStr};

use crate::types::{t4a_index, InternalIndex};
use crate::{
    capi_error, clone_opaque, is_assigned_opaque, release_opaque, run_catching, StatusCode,
    T4A_BUFFER_TOO_SMALL, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
};
use tensor4all_core::index::Index;

/// Release an index handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_release(obj: *mut t4a_index) {
    release_opaque(obj);
}

/// Clone an index handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_clone(src: *const t4a_index, out: *mut *mut t4a_index) -> StatusCode {
    clone_opaque(src, out)
}

/// Check whether an index handle is assigned.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_is_assigned(obj: *const t4a_index) -> i32 {
    is_assigned_opaque(obj)
}

fn read_optional_tags_csv(tags_csv: *const c_char) -> Result<Option<String>, (StatusCode, String)> {
    if tags_csv.is_null() {
        return Ok(None);
    }

    let c_str = unsafe { CStr::from_ptr(tags_csv) };
    let tags = c_str
        .to_str()
        .map_err(|e| capi_error(T4A_INVALID_ARGUMENT, e))?;
    Ok(Some(tags.to_string()))
}

fn build_index(
    dim: usize,
    tags_csv: *const c_char,
    plev: i64,
) -> Result<InternalIndex, (StatusCode, String)> {
    if dim == 0 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "dim must be greater than zero",
        ));
    }
    if plev < 0 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "plev must be greater than or equal to zero",
        ));
    }

    let tags = read_optional_tags_csv(tags_csv)?;
    let index = match tags {
        Some(tags) => {
            Index::new_dyn_with_tag(dim, &tags).map_err(|e| capi_error(T4A_INVALID_ARGUMENT, e))?
        }
        None => Index::new_dyn(dim),
    };

    Ok(index.set_plev(plev))
}

/// Create a new index with explicit tags and prime level.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_new(
    dim: usize,
    tags_csv: *const c_char,
    plev: i64,
    out: *mut *mut t4a_index,
) -> StatusCode {
    run_catching(out, || {
        Ok(t4a_index::new(build_index(dim, tags_csv, plev)?))
    })
}

/// Create a new index with an explicit 64-bit ID.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_new_with_id(
    dim: usize,
    id: u64,
    tags_csv: *const c_char,
    plev: i64,
    out: *mut *mut t4a_index,
) -> StatusCode {
    if dim == 0 {
        return crate::err_status("dim must be greater than zero", T4A_INVALID_ARGUMENT);
    }
    if plev < 0 {
        return crate::err_status(
            "plev must be greater than or equal to zero",
            T4A_INVALID_ARGUMENT,
        );
    }

    use tensor4all_core::index::{DynId, TagSet};

    run_catching(out, || {
        let tags = match read_optional_tags_csv(tags_csv)? {
            Some(tags) => {
                TagSet::from_str(&tags).map_err(|e| capi_error(T4A_INVALID_ARGUMENT, e))?
            }
            None => TagSet::new(),
        };

        let index = Index::new_with_size_and_tags(DynId(id), dim, tags).set_plev(plev);
        Ok(t4a_index::new(index))
    })
}

/// Get the dimension of an index.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_dim(ptr: *const t4a_index, out_dim: *mut usize) -> StatusCode {
    if ptr.is_null() || out_dim.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        *out_dim = (*ptr).inner().size();
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the 64-bit ID of an index.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_id(ptr: *const t4a_index, out_id: *mut u64) -> StatusCode {
    if ptr.is_null() || out_id.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        *out_id = (*ptr).inner().id.0;
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the prime level of an index.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_plev(ptr: *const t4a_index, out_plev: *mut i64) -> StatusCode {
    if ptr.is_null() || out_plev.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        *out_plev = (*ptr).inner().plev;
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Copy tags as a comma-separated UTF-8 string.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_tags(
    ptr: *const t4a_index,
    buf: *mut u8,
    buf_len: usize,
    out_len: *mut usize,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let tags = (*ptr)
            .inner()
            .tags()
            .iter()
            .map(|tag| tag.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let required_len = tags.len() + 1;
        *out_len = required_len;

        if buf.is_null() {
            return T4A_SUCCESS;
        }
        if buf_len < required_len {
            return T4A_BUFFER_TOO_SMALL;
        }

        std::ptr::copy_nonoverlapping(tags.as_ptr(), buf, tags.len());
        *buf.add(tags.len()) = 0;
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Query whether an index has the provided tag.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_has_tag(
    ptr: *const t4a_index,
    tag: *const c_char,
    out_has_tag: *mut i32,
) -> StatusCode {
    if ptr.is_null() || tag.is_null() || out_has_tag.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let c_str = CStr::from_ptr(tag);
        let tag = c_str
            .to_str()
            .map_err(|e| crate::err_status(e, T4A_INVALID_ARGUMENT))?;
        *out_has_tag = if (*ptr).inner().tags().has_tag(tag) {
            1
        } else {
            0
        };
        Ok::<StatusCode, StatusCode>(T4A_SUCCESS)
    }));

    match result {
        Ok(Ok(code)) => code,
        Ok(Err(code)) => code,
        Err(panic) => crate::unwrap_catch(Err(panic)),
    }
}

#[cfg(test)]
mod tests;
