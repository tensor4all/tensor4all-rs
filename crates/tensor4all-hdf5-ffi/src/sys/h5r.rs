#![allow(non_camel_case_types, non_upper_case_globals)]
//! Creating and manipulating references (H5R)
//!
//! Types and constants for HDF5 reference operations.

use std::mem;

use super::types::*;

pub use H5R_type_t::*;

/// Maximum reference size
pub const H5R_REF_BUF_SIZE: usize = 64;

/// Reference types
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5R_type_t {
    H5R_BADTYPE = -1,
    H5R_OBJECT1 = 0,
    H5R_DATASET_REGION1 = 1,
    H5R_OBJECT2 = 2,
    H5R_DATASET_REGION2 = 3,
    H5R_ATTR = 4,
    H5R_MAXTYPE = 5,
}

/// Reference structure (new API)
#[repr(C)]
#[derive(Copy, Clone)]
pub struct H5R_ref_t {
    pub u: H5R_ref_t__u,
}

impl Default for H5R_ref_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union H5R_ref_t__u {
    pub __data: [u8; H5R_REF_BUF_SIZE],
    pub align: i64,
}

impl Default for H5R_ref_t__u {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Object reference (deprecated)
pub type hobj_ref_t = haddr_t;

/// Dataset region reference size (deprecated)
pub const H5R_DSET_REG_REF_BUF_SIZE: usize = 12;

/// Dataset region reference (deprecated)
#[repr(C)]
#[derive(Copy, Clone)]
pub struct hdset_reg_ref_t {
    pub __data: [u8; H5R_DSET_REG_REF_BUF_SIZE],
}

impl Default for hdset_reg_ref_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}
