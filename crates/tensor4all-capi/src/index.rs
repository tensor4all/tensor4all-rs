//! C API for immutable `Index` handles.

use std::collections::hash_map::DefaultHasher;
use std::ffi::{c_char, CStr};
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::types::{t4a_index, InternalIndex};
use crate::{
    capi_error, clone_opaque, is_assigned_opaque, panic_message, release_opaque, run_catching,
    set_last_error, CapiResult, StatusCode, T4A_BUFFER_TOO_SMALL, T4A_INTERNAL_ERROR,
    T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
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

fn require_index<'a>(ptr: *const t4a_index, what: &str) -> CapiResult<&'a InternalIndex> {
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, format!("{what} is null")));
    }
    Ok(unsafe { &*ptr }.inner())
}

fn run_value<T: Copy, F>(out: *mut T, f: F) -> StatusCode
where
    F: FnOnce() -> CapiResult<T>,
{
    if out.is_null() {
        return T4A_NULL_POINTER;
    }

    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => {
            unsafe { *out = value };
            T4A_SUCCESS
        }
        Ok(Err((code, msg))) => {
            set_last_error(&msg);
            code
        }
        Err(panic) => {
            let msg = panic_message(&*panic);
            set_last_error(&msg);
            T4A_INTERNAL_ERROR
        }
    }
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

/// Compare two full index handles for equality.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_equal(
    lhs: *const t4a_index,
    rhs: *const t4a_index,
    out_equal: *mut i32,
) -> StatusCode {
    run_value(out_equal, || {
        let lhs = require_index(lhs, "lhs")?;
        let rhs = require_index(rhs, "rhs")?;
        Ok((lhs == rhs) as i32)
    })
}

/// Hash the full index value for process-local hash tables.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_hash(ptr: *const t4a_index, out_hash: *mut u64) -> StatusCode {
    run_value(out_hash, || {
        let index = require_index(ptr, "index")?;
        let mut hasher = DefaultHasher::new();
        index.hash(&mut hasher);
        Ok(hasher.finish())
    })
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

/// Return a new index handle with prime level incremented by one.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_prime(ptr: *const t4a_index, out: *mut *mut t4a_index) -> StatusCode {
    run_catching(out, || {
        let index = require_index(ptr, "index")?;
        Ok(t4a_index::new(index.prime()))
    })
}

/// Return a new index handle with prime level reset to zero.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_noprime(ptr: *const t4a_index, out: *mut *mut t4a_index) -> StatusCode {
    run_catching(out, || {
        let index = require_index(ptr, "index")?;
        Ok(t4a_index::new(index.noprime()))
    })
}

/// Return a new index handle with an explicit prime level.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_set_plev(
    ptr: *const t4a_index,
    plev: i64,
    out: *mut *mut t4a_index,
) -> StatusCode {
    run_catching(out, || {
        if plev < 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "plev must be greater than or equal to zero",
            ));
        }
        let index = require_index(ptr, "index")?;
        Ok(t4a_index::new(index.set_plev(plev)))
    })
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
