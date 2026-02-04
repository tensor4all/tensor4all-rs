#![allow(non_camel_case_types, non_upper_case_globals)]
//! Functions for handling errors (H5E)
//!
//! Types and constants for HDF5 error handling.

use std::mem;

use super::types::*;

pub use H5E_direction_t::*;
pub use H5E_type_t::*;

/// Default error stack
pub const H5E_DEFAULT: hid_t = 0;

/// Error types
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5E_type_t {
    H5E_MAJOR = 0,
    H5E_MINOR = 1,
}

pub type H5E_major_t = hid_t;
pub type H5E_minor_t = hid_t;

/// Error info structure (version 1, deprecated)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5E_error1_t {
    pub maj_num: H5E_major_t,
    pub min_num: H5E_minor_t,
    pub func_name: *const c_char,
    pub file_name: *const c_char,
    pub line: c_uint,
    pub desc: *const c_char,
}

impl Default for H5E_error1_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Error info structure (version 2)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5E_error2_t {
    pub cls_id: hid_t,
    pub maj_num: hid_t,
    pub min_num: hid_t,
    pub line: c_uint,
    pub func_name: *const c_char,
    pub file_name: *const c_char,
    pub desc: *const c_char,
}

impl Default for H5E_error2_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Error traversal direction
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5E_direction_t {
    H5E_WALK_UPWARD = 0,
    H5E_WALK_DOWNWARD = 1,
}

/// Walk callback (version 1, deprecated)
pub type H5E_walk1_t = Option<
    unsafe extern "C" fn(n: c_int, err_desc: *mut H5E_error1_t, client_data: *mut c_void) -> herr_t,
>;

/// Auto callback (version 1, deprecated)
pub type H5E_auto1_t = Option<unsafe extern "C" fn(client_data: *mut c_void) -> herr_t>;

/// Walk callback (version 2)
pub type H5E_walk2_t = Option<
    unsafe extern "C" fn(
        n: c_uint,
        err_desc: *const H5E_error2_t,
        client_data: *mut c_void,
    ) -> herr_t,
>;

/// Auto callback (version 2)
pub type H5E_auto2_t =
    Option<unsafe extern "C" fn(estack: hid_t, client_data: *mut c_void) -> herr_t>;

// Type aliases for current API
pub type H5E_auto_t = H5E_auto2_t;
pub type H5E_error_t = H5E_error2_t;
pub type H5E_walk_t = H5E_walk2_t;
