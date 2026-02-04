#![allow(non_camel_case_types, non_upper_case_globals)]
//! Manipulating object identifiers and object names (H5I)
//!
//! Types for HDF5 identifier management.

use super::types::*;

pub use H5I_type_t::*;

/// Invalid HDF5 ID constant
pub const H5I_INVALID_HID: hid_t = -1;

/// Types of objects in an HDF5 file
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5I_type_t {
    H5I_UNINIT = -2,
    H5I_BADID = -1,
    H5I_FILE = 1,
    H5I_GROUP = 2,
    H5I_DATATYPE = 3,
    H5I_DATASPACE = 4,
    H5I_DATASET = 5,
    H5I_MAP = 6,
    H5I_ATTR = 7,
    H5I_VFL = 8,
    H5I_VOL = 9,
    H5I_GENPROP_CLS = 10,
    H5I_GENPROP_LST = 11,
    H5I_ERROR_CLASS = 12,
    H5I_ERROR_MSG = 13,
    H5I_ERROR_STACK = 14,
    H5I_SPACE_SEL_ITER = 15,
    H5I_EVENTSET = 16,
    H5I_NTYPES = 17,
}

/// Free function type for identifiers
pub type H5I_free_t = Option<unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> herr_t>;

/// Search function type for identifiers
pub type H5I_search_func_t =
    Option<unsafe extern "C" fn(obj: *mut c_void, id: hid_t, key: *mut c_void) -> c_int>;

/// Iterate function type for identifiers
pub type H5I_iterate_func_t = Option<unsafe extern "C" fn(id: hid_t, udata: *mut c_void) -> herr_t>;

/// Future realize callback type
pub type H5I_future_realize_func_t = Option<
    unsafe extern "C" fn(future_object: *mut c_void, actual_object_id: *mut hid_t) -> herr_t,
>;

/// Future discard callback type
pub type H5I_future_discard_func_t =
    Option<unsafe extern "C" fn(future_object: *mut c_void) -> herr_t>;
