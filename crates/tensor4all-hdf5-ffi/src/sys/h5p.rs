#![allow(non_camel_case_types, non_upper_case_globals)]
//! Creating and manipulating property lists (H5P)
//!
//! Types and constants for HDF5 property list operations.

use super::types::*;

// Property list constants
pub const H5P_CRT_ORDER_TRACKED: c_uint = 0x0001;
pub const H5P_CRT_ORDER_INDEXED: c_uint = 0x0002;
pub const H5P_DEFAULT: hid_t = 0;

/// Class create callback
pub type H5P_cls_create_func_t =
    Option<unsafe extern "C" fn(prop_id: hid_t, create_data: *mut c_void) -> herr_t>;

/// Class copy callback
pub type H5P_cls_copy_func_t = Option<
    unsafe extern "C" fn(new_prop_id: hid_t, old_prop_id: hid_t, copy_data: *mut c_void) -> herr_t,
>;

/// Class close callback
pub type H5P_cls_close_func_t =
    Option<unsafe extern "C" fn(prop_id: hid_t, close_data: *mut c_void) -> herr_t>;

/// Property callback type 1 (create, copy, close)
pub type H5P_prp_cb1_t =
    Option<unsafe extern "C" fn(name: *const c_char, size: size_t, value: *mut c_void) -> herr_t>;

/// Property callback type 2 (set, get, delete)
pub type H5P_prp_cb2_t = Option<
    extern "C" fn(prop_id: hid_t, name: *const c_char, size: size_t, value: *mut c_void) -> herr_t,
>;

pub type H5P_prp_create_func_t = H5P_prp_cb1_t;
pub type H5P_prp_set_func_t = H5P_prp_cb2_t;
pub type H5P_prp_get_func_t = H5P_prp_cb2_t;
pub type H5P_prp_delete_func_t = H5P_prp_cb2_t;
pub type H5P_prp_copy_func_t = H5P_prp_cb1_t;
pub type H5P_prp_close_func_t = H5P_prp_cb1_t;

/// Property compare callback
pub type H5P_prp_compare_func_t = Option<
    unsafe extern "C" fn(value1: *const c_void, value2: *const c_void, size: size_t) -> c_int,
>;

/// Property iterate callback
pub type H5P_iterate_t =
    Option<unsafe extern "C" fn(id: hid_t, name: *const c_char, iter_data: *mut c_void) -> herr_t>;
