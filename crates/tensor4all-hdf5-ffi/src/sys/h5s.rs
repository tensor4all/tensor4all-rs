#![allow(non_camel_case_types, non_upper_case_globals)]
//! Creating and manipulating dataspaces (H5S)
//!
//! Types and constants for HDF5 dataspace operations.

use super::types::*;

pub use H5S_class_t::*;
pub use H5S_sel_type::*;
pub use H5S_seloper_t::*;

// Constants
pub const H5S_ALL: hid_t = 0;
pub const H5S_UNLIMITED: hsize_t = !0;
pub const H5S_MAX_RANK: c_uint = 32;

// Selection iterator flags
pub const H5S_SEL_ITER_GET_SEQ_LIST_SORTED: c_uint = 0x0001;
pub const H5S_SEL_ITER_SHARE_WITH_DATASPACE: c_uint = 0x0002;

/// Dataspace class
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5S_class_t {
    H5S_NO_CLASS = -1,
    H5S_SCALAR = 0,
    H5S_SIMPLE = 1,
    H5S_NULL = 2,
}

/// Selection operation
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5S_seloper_t {
    H5S_SELECT_NOOP = -1,
    H5S_SELECT_SET = 0,
    H5S_SELECT_OR = 1,
    H5S_SELECT_AND = 2,
    H5S_SELECT_XOR = 3,
    H5S_SELECT_NOTB = 4,
    H5S_SELECT_NOTA = 5,
    H5S_SELECT_APPEND = 6,
    H5S_SELECT_PREPEND = 7,
    H5S_SELECT_INVALID = 8,
}

/// Selection type
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5S_sel_type {
    H5S_SEL_ERROR = -1,
    H5S_SEL_NONE = 0,
    H5S_SEL_POINTS = 1,
    H5S_SEL_HYPERSLABS = 2,
    H5S_SEL_ALL = 3,
    H5S_SEL_N = 4,
}
