#![allow(non_camel_case_types, non_upper_case_globals)]
//! Creating and manipulating groups of objects inside an HDF5 file (H5G)
//!
//! Types and constants for HDF5 group operations.

use std::mem;

use super::h5l::H5L_type_t;
use super::h5o::H5O_stat_t;
use super::types::*;

pub use H5G_obj_t::*;
pub use H5G_storage_type_t::*;

// Same location constant
pub const H5G_SAME_LOC: hid_t = 0;

// Link type aliases
pub const H5G_LINK_ERROR: H5L_type_t = H5L_type_t::H5L_TYPE_ERROR;
pub const H5G_LINK_HARD: H5L_type_t = H5L_type_t::H5L_TYPE_HARD;
pub const H5G_LINK_SOFT: H5L_type_t = H5L_type_t::H5L_TYPE_SOFT;

pub type H5G_link_t = H5L_type_t;

// Number of types
pub const H5G_NTYPES: c_uint = 256;
pub const H5G_NLIBTYPES: c_uint = 8;
pub const H5G_NUSERTYPES: c_uint = H5G_NTYPES - H5G_NLIBTYPES;

/// Compute user type value
pub const fn H5G_USERTYPE(x: c_uint) -> c_uint {
    8 + x
}

/// Storage type for groups
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5G_storage_type_t {
    H5G_STORAGE_TYPE_UNKNOWN = -1,
    H5G_STORAGE_TYPE_SYMBOL_TABLE = 0,
    H5G_STORAGE_TYPE_COMPACT = 1,
    H5G_STORAGE_TYPE_DENSE = 2,
}

/// Group info structure
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5G_info_t {
    pub storage_type: H5G_storage_type_t,
    pub nlinks: hsize_t,
    pub max_corder: i64,
    pub mounted: hbool_t,
}

impl Default for H5G_info_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Object types (deprecated)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum H5G_obj_t {
    H5G_UNKNOWN = -1,
    H5G_GROUP = 0,
    H5G_DATASET = 1,
    H5G_TYPE = 2,
    H5G_LINK = 3,
    H5G_UDLINK = 4,
    H5G_RESERVED_5 = 5,
    H5G_RESERVED_6 = 6,
    H5G_RESERVED_7 = 7,
}

/// Group stat structure (deprecated)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5G_stat_t {
    pub fileno: [c_ulong; 2],
    pub objno: [c_ulong; 2],
    pub nlink: c_uint,
    pub type_: H5G_obj_t,
    pub mtime: time_t,
    pub linklen: size_t,
    pub ohdr: H5O_stat_t,
}

impl Default for H5G_stat_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}
