#![allow(non_camel_case_types, non_upper_case_globals)]
//! Creating and manipulating links (H5L)
//!
//! Types and constants for HDF5 link operations.

use std::mem;

use super::h5o::H5O_token_t;
use super::h5t::H5T_cset_t;
use super::types::*;

pub use H5L_type_t::*;

// Constants
pub const H5L_MAX_LINK_NAME_LEN: u32 = !0;
pub const H5L_SAME_LOC: hid_t = 0;
pub const H5L_LINK_CLASS_T_VERS: c_uint = 0;

/// Link types
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5L_type_t {
    H5L_TYPE_ERROR = -1,
    H5L_TYPE_HARD = 0,
    H5L_TYPE_SOFT = 1,
    H5L_TYPE_EXTERNAL = 64,
    H5L_TYPE_MAX = 255,
}

pub const H5L_TYPE_BUILTIN_MAX: H5L_type_t = H5L_type_t::H5L_TYPE_SOFT;
pub const H5L_TYPE_UD_MIN: H5L_type_t = H5L_type_t::H5L_TYPE_EXTERNAL;

/// Link info structure (version 1)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5L_info1_t {
    pub type_: H5L_type_t,
    pub corder_valid: hbool_t,
    pub corder: i64,
    pub cset: H5T_cset_t,
    pub u: H5L_info1_t__u,
}

impl Default for H5L_info1_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5L_info1_t__u {
    value: [u64; 1usize],
}

impl Default for H5L_info1_t__u {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

impl H5L_info1_t__u {
    pub unsafe fn address(&mut self) -> *mut haddr_t {
        &self.value as *const [u64; 1] as *mut haddr_t
    }
    pub unsafe fn val_size(&mut self) -> *mut size_t {
        &self.value as *const [u64; 1] as *mut size_t
    }
}

/// Link info structure (version 2)
#[repr(C)]
#[derive(Copy, Clone)]
pub struct H5L_info2_t {
    pub type_: H5L_type_t,
    pub corder_valid: hbool_t,
    pub corder: i64,
    pub cset: H5T_cset_t,
    pub u: H5L_info2_t__u,
}

impl Default for H5L_info2_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union H5L_info2_t__u {
    pub token: H5O_token_t,
    pub val_size: size_t,
}

impl Default for H5L_info2_t__u {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

impl H5L_info2_t__u {
    pub unsafe fn token(&mut self) -> *mut H5O_token_t {
        &mut self.token as *mut H5O_token_t
    }
    pub unsafe fn val_size_ptr(&mut self) -> *mut size_t {
        &mut self.val_size as *mut size_t
    }
}

/// Link create callback
pub type H5L_create_func_t = Option<
    extern "C" fn(
        link_name: *const c_char,
        loc_group: hid_t,
        lnkdata: *const c_void,
        lnkdata_size: size_t,
        lcpl_id: hid_t,
    ) -> herr_t,
>;

/// Link move callback
pub type H5L_move_func_t = Option<
    extern "C" fn(
        new_name: *const c_char,
        new_loc: hid_t,
        lnkdata: *const c_void,
        lnkdata_size: size_t,
    ) -> herr_t,
>;

/// Link copy callback
pub type H5L_copy_func_t = Option<
    extern "C" fn(
        new_name: *const c_char,
        new_loc: hid_t,
        lnkdata: *const c_void,
        lnkdata_size: size_t,
    ) -> herr_t,
>;

/// Link traverse callback
pub type H5L_traverse_func_t = Option<
    extern "C" fn(
        link_name: *const c_char,
        cur_group: hid_t,
        lnkdata: *const c_void,
        lnkdata_size: size_t,
        lapl_id: hid_t,
    ) -> hid_t,
>;

/// Link delete callback
pub type H5L_delete_func_t = Option<
    extern "C" fn(
        link_name: *const c_char,
        file: hid_t,
        lnkdata: *const c_void,
        lnkdata_size: size_t,
    ) -> herr_t,
>;

/// Link query callback
pub type H5L_query_func_t = Option<
    extern "C" fn(
        link_name: *const c_char,
        lnkdata: *const c_void,
        lnkdata_size: size_t,
        buf: *mut c_void,
        buf_size: size_t,
    ) -> ssize_t,
>;

/// User-defined link class
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5L_class_t {
    pub version: c_int,
    pub id: H5L_type_t,
    pub comment: *const c_char,
    pub create_func: H5L_create_func_t,
    pub move_func: H5L_move_func_t,
    pub copy_func: H5L_copy_func_t,
    pub trav_func: H5L_traverse_func_t,
    pub del_func: H5L_delete_func_t,
    pub query_func: H5L_query_func_t,
}

impl Default for H5L_class_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Link iteration callback (version 1)
pub type H5L_iterate1_t = Option<
    unsafe extern "C" fn(
        group: hid_t,
        name: *const c_char,
        info: *const H5L_info1_t,
        op_data: *mut c_void,
    ) -> herr_t,
>;

/// Link iteration callback (version 2)
pub type H5L_iterate2_t = Option<
    unsafe extern "C" fn(
        group: hid_t,
        name: *const c_char,
        info: *const H5L_info2_t,
        op_data: *mut c_void,
    ) -> herr_t,
>;

/// External link traverse callback
pub type H5L_elink_traverse_t = Option<
    extern "C" fn(
        parent_file_name: *const c_char,
        parent_group_name: *const c_char,
        child_file_name: *const c_char,
        child_object_name: *const c_char,
        acc_flags: *mut c_uint,
        fapl_id: hid_t,
        op_data: *mut c_void,
    ) -> herr_t,
>;

// Type aliases for current API
pub type H5L_info_t = H5L_info2_t;
pub type H5L_iterate_t = H5L_iterate2_t;
