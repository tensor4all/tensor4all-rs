//! Utility functions for string handling.

use std::borrow::Borrow;
use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::ptr;

use crate::error::Result;
use crate::sys::{c_char, size_t};

/// Convert a zero-terminated string (`const char *`) into a `String`.
///
/// # Safety
///
/// The memory pointed to by `string` must be valid for constructing a `CStr`
/// containing valid UTF-8.
pub unsafe fn string_from_cstr(string: *const c_char) -> String {
    unsafe { String::from_utf8_unchecked(CStr::from_ptr(string).to_bytes().to_vec()) }
}

/// Convert a `String` or a `&str` into a zero-terminated string (`const char *`).
pub fn to_cstring<S: Borrow<str>>(string: S) -> Result<CString> {
    let string = string.borrow();
    #[allow(clippy::map_err_ignore)]
    CString::new(string).map_err(|_| format!("null byte in string: {string:?}").into())
}

/// Convert a fixed-length (possibly zero-terminated) char buffer to a string.
///
/// # Panics
///
/// Panics if the bytes are not valid UTF-8.
pub fn string_from_fixed_bytes(bytes: &[c_char], len: usize) -> String {
    let len = bytes.iter().position(|&c| c == 0).unwrap_or(len);
    let bytes = &bytes[..len];
    let s = unsafe { std::str::from_utf8(&*(bytes as *const [c_char] as *const [u8])).unwrap() };
    s.to_owned()
}

/// Write a string into a fixed-length char buffer (possibly truncating it).
pub fn string_to_fixed_bytes(s: &str, buf: &mut [c_char]) {
    let mut s = s;
    while s.as_bytes().len() > buf.len() {
        s = &s[..(s.len() - 1)];
    }
    let bytes = s.as_bytes();
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr().cast(), bytes.len());
    }
    for c in &mut buf[bytes.len()..] {
        *c = 0;
    }
}

/// Get a string from an HDF5 function that returns the length of the string.
///
/// # Safety
///
/// `func` must expect a pointer to a buffer and its size.
/// If the pointer is null, `func` must return the length of the message.
/// Otherwise, `func` must try to write a string into the buffer that is valid for constructing
/// a `CStr` and contain valid UTF-8.
#[doc(hidden)]
pub unsafe fn get_h5_str<T, F>(func: F) -> Result<String>
where
    F: Fn(*mut c_char, size_t) -> T,
    T: TryInto<isize>,
{
    let len = 1_isize + (func(ptr::null_mut(), 0)).try_into().unwrap_or(-1);
    if len <= 0 {
        return Err("negative string length in get_h5_str()".into());
    }
    if len == 1 {
        Ok(String::new())
    } else {
        let mut buf = vec![0; len as usize];
        func(buf.as_mut_ptr(), len as _);
        // SAFETY: buf contains a valid UTF-8 C string
        Ok(string_from_cstr(buf.as_ptr()))
    }
}
