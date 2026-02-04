#![allow(non_camel_case_types, non_upper_case_globals)]
//! General purpose library functions (H5)
//!
//! Types and constants for the HDF5 general library interface.

use std::mem;

use super::types::*;

pub use H5_index_t::*;
pub use H5_iter_order_t::*;

// Iteration return values
pub const H5_ITER_ERROR: c_int = -1;
pub const H5_ITER_CONT: c_int = 0;
pub const H5_ITER_STOP: c_int = 1;

// Address constants
pub const HADDR_UNDEF: haddr_t = !0;
pub const HADDR_MAX: haddr_t = HADDR_UNDEF - 1;

/// Iteration order type
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5_iter_order_t {
    H5_ITER_UNKNOWN = -1,
    H5_ITER_INC = 0,
    H5_ITER_DEC = 1,
    H5_ITER_NATIVE = 2,
    H5_ITER_N = 3,
}

/// Index type
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5_index_t {
    H5_INDEX_UNKNOWN = -1,
    H5_INDEX_NAME = 0,
    H5_INDEX_CRT_ORDER = 1,
    H5_INDEX_N = 2,
}

/// Index heap info
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5_ih_info_t {
    pub index_size: hsize_t,
    pub heap_size: hsize_t,
}

impl Default for H5_ih_info_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}
