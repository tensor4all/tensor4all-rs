#![allow(non_camel_case_types, non_upper_case_globals)]
//! Creating and manipulating HDF5 files (H5F)
//!
//! Types and constants for HDF5 file operations.

use std::mem;

use super::h5::H5_ih_info_t;
use super::types::*;

pub use H5F_close_degree_t::*;
pub use H5F_libver_t::*;
pub use H5F_mem_t::*;
pub use H5F_scope_t::*;

// File access flags
pub const H5F_ACC_RDONLY: c_uint = 0x0000;
pub const H5F_ACC_RDWR: c_uint = 0x0001;
pub const H5F_ACC_TRUNC: c_uint = 0x0002;
pub const H5F_ACC_EXCL: c_uint = 0x0004;
pub const H5F_ACC_CREAT: c_uint = 0x0010;
pub const H5F_ACC_DEFAULT: c_uint = 0xffff;
pub const H5F_ACC_SWMR_WRITE: c_uint = 0x0020;
pub const H5F_ACC_SWMR_READ: c_uint = 0x0040;

// Object type flags
pub const H5F_OBJ_FILE: c_uint = 0x0001;
pub const H5F_OBJ_DATASET: c_uint = 0x0002;
pub const H5F_OBJ_GROUP: c_uint = 0x0004;
pub const H5F_OBJ_DATATYPE: c_uint = 0x0008;
pub const H5F_OBJ_ATTR: c_uint = 0x0010;
pub const H5F_OBJ_ALL: c_uint =
    H5F_OBJ_FILE | H5F_OBJ_DATASET | H5F_OBJ_GROUP | H5F_OBJ_DATATYPE | H5F_OBJ_ATTR;
pub const H5F_OBJ_LOCAL: c_uint = 0x0020;

pub const H5F_FAMILY_DEFAULT: hsize_t = 0;
pub const H5F_UNLIMITED: hsize_t = !0;

/// Scope of file operations
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5F_scope_t {
    H5F_SCOPE_LOCAL = 0,
    H5F_SCOPE_GLOBAL = 1,
}

/// How aggressively to close file
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5F_close_degree_t {
    H5F_CLOSE_DEFAULT = 0,
    H5F_CLOSE_WEAK = 1,
    H5F_CLOSE_SEMI = 2,
    H5F_CLOSE_STRONG = 3,
}

impl Default for H5F_close_degree_t {
    fn default() -> Self {
        Self::H5F_CLOSE_DEFAULT
    }
}

/// Memory type for file driver
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5F_mem_t {
    H5FD_MEM_NOLIST = -1,
    H5FD_MEM_DEFAULT = 0,
    H5FD_MEM_SUPER = 1,
    H5FD_MEM_BTREE = 2,
    H5FD_MEM_DRAW = 3,
    H5FD_MEM_GHEAP = 4,
    H5FD_MEM_LHEAP = 5,
    H5FD_MEM_OHDR = 6,
    H5FD_MEM_NTYPES = 7,
}

/// Library version bounds
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5F_libver_t {
    H5F_LIBVER_ERROR = -1,
    H5F_LIBVER_EARLIEST = 0,
    H5F_LIBVER_V18 = 1,
    H5F_LIBVER_V110 = 2,
    H5F_LIBVER_V112 = 3,
    H5F_LIBVER_V114 = 4,
    H5F_LIBVER_NBOUNDS = 5,
}

pub const H5F_LIBVER_LATEST: H5F_libver_t = H5F_libver_t::H5F_LIBVER_V114;

impl Default for H5F_libver_t {
    fn default() -> Self {
        H5F_LIBVER_LATEST
    }
}

/// File info structure (version 1, deprecated)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5F_info1_t {
    pub super_ext_size: hsize_t,
    pub sohm: H5F_info1_t__sohm,
}

impl Default for H5F_info1_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5F_info1_t__sohm {
    pub hdr_size: hsize_t,
    pub msgs_info: H5_ih_info_t,
}

impl Default for H5F_info1_t__sohm {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// File info structure (version 2)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5F_info2_t {
    pub super_: H5F_info2_t__super,
    pub free: H5F_info2_t__free,
    pub sohm: H5F_info2_t__sohm,
}

impl Default for H5F_info2_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5F_info2_t__super {
    pub version: c_uint,
    pub super_size: hsize_t,
    pub super_ext_size: hsize_t,
}

impl Default for H5F_info2_t__super {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5F_info2_t__free {
    pub version: c_uint,
    pub meta_size: hsize_t,
    pub tot_space: hsize_t,
}

impl Default for H5F_info2_t__free {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5F_info2_t__sohm {
    pub version: c_uint,
    pub hdr_size: hsize_t,
    pub msgs_info: H5_ih_info_t,
}

impl Default for H5F_info2_t__sohm {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Retry info structure
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5F_retry_info_t {
    pub nbins: c_uint,
    pub retries: [*mut u32; 21usize],
}

impl Default for H5F_retry_info_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Section info structure
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5F_sect_info_t {
    pub addr: haddr_t,
    pub size: hsize_t,
}

impl Default for H5F_sect_info_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// File space type
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5F_file_space_type_t {
    H5F_FILE_SPACE_DEFAULT = 0,
    H5F_FILE_SPACE_ALL_PERSIST = 1,
    H5F_FILE_SPACE_ALL = 2,
    H5F_FILE_SPACE_AGGR_VFD = 3,
    H5F_FILE_SPACE_VFD = 4,
    H5F_FILE_SPACE_NTYPES = 5,
}

pub use H5F_file_space_type_t::*;

/// File space strategy
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug, Default)]
pub enum H5F_fspace_strategy_t {
    #[default]
    H5F_FSPACE_STRATEGY_FSM_AGGR = 0,
    H5F_FSPACE_STRATEGY_PAGE = 1,
    H5F_FSPACE_STRATEGY_AGGR = 2,
    H5F_FSPACE_STRATEGY_NONE = 3,
    H5F_FSPACE_STRATEGY_NTYPES = 4,
}

pub use H5F_fspace_strategy_t::*;

/// Flush callback type
pub type H5F_flush_cb_t =
    Option<unsafe extern "C" fn(object_id: hid_t, udata: *mut c_void) -> herr_t>;

// Type aliases for backward compatibility
pub type H5F_info_t = H5F_info2_t;
pub type H5F_info_t__free = H5F_info2_t__free;
pub type H5F_info_t__sohm = H5F_info2_t__sohm;
pub type H5F_info_t__super = H5F_info2_t__super;
