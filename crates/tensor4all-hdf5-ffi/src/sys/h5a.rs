#![allow(non_camel_case_types, non_upper_case_globals)]
//! Creating and manipulating HDF5 attributes (H5A)
//!
//! Types and constants for HDF5 attribute operations.

use std::mem;

use super::h5o::H5O_msg_crt_idx_t;
use super::h5t::H5T_cset_t;
use super::types::*;

/// Attribute info structure
#[repr(C)]
#[derive(Copy, Clone)]
pub struct H5A_info_t {
    pub corder_valid: hbool_t,
    pub corder: H5O_msg_crt_idx_t,
    pub cset: H5T_cset_t,
    pub data_size: hsize_t,
}

impl Default for H5A_info_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Attribute operator type (deprecated, version 1)
pub type H5A_operator1_t = Option<
    unsafe extern "C" fn(
        location_id: hid_t,
        attr_name: *const c_char,
        operator_data: *mut c_void,
    ) -> herr_t,
>;

/// Attribute operator type (version 2)
pub type H5A_operator2_t = Option<
    unsafe extern "C" fn(
        location_id: hid_t,
        attr_name: *const c_char,
        ainfo: *const H5A_info_t,
        op_data: *mut c_void,
    ) -> herr_t,
>;

// Type aliases for backward compatibility
pub type H5A_operator_t = H5A_operator2_t;
