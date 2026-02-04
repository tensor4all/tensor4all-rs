#![allow(non_camel_case_types, non_upper_case_globals)]
//! Creating and manipulating HDF5 objects (H5O)
//!
//! Types and constants for HDF5 object operations.

use std::mem;

use super::h5::H5_ih_info_t;
use super::types::*;

pub use H5O_mcdt_search_ret_t::*;
pub use H5O_type_t::*;

// Constants
pub const H5O_COPY_SHALLOW_HIERARCHY_FLAG: c_uint = 0x0001;
pub const H5O_COPY_EXPAND_SOFT_LINK_FLAG: c_uint = 0x0002;
pub const H5O_COPY_EXPAND_EXT_LINK_FLAG: c_uint = 0x0004;
pub const H5O_COPY_EXPAND_REFERENCE_FLAG: c_uint = 0x0008;
pub const H5O_COPY_WITHOUT_ATTR_FLAG: c_uint = 0x0010;
pub const H5O_COPY_PRESERVE_NULL_FLAG: c_uint = 0x0020;
pub const H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG: c_uint = 0x0040;
pub const H5O_COPY_ALL: c_uint = 0x007F;

pub const H5O_SHMESG_NONE_FLAG: c_uint = 0x0000;
pub const H5O_SHMESG_SDSPACE_FLAG: c_uint = 0x0001;
pub const H5O_SHMESG_DTYPE_FLAG: c_uint = 0x0002;
pub const H5O_SHMESG_FILL_FLAG: c_uint = 0x0004;
pub const H5O_SHMESG_PLINE_FLAG: c_uint = 0x0008;
pub const H5O_SHMESG_ATTR_FLAG: c_uint = 0x0010;
pub const H5O_SHMESG_ALL_FLAG: c_uint = H5O_SHMESG_SDSPACE_FLAG
    | H5O_SHMESG_DTYPE_FLAG
    | H5O_SHMESG_FILL_FLAG
    | H5O_SHMESG_PLINE_FLAG
    | H5O_SHMESG_ATTR_FLAG;

pub const H5O_HDR_CHUNK0_SIZE: c_uint = 0x03;
pub const H5O_HDR_ATTR_CRT_ORDER_TRACKED: c_uint = 0x04;
pub const H5O_HDR_ATTR_CRT_ORDER_INDEXED: c_uint = 0x08;
pub const H5O_HDR_ATTR_STORE_PHASE_CHANGE: c_uint = 0x10;
pub const H5O_HDR_STORE_TIMES: c_uint = 0x20;
pub const H5O_HDR_ALL_FLAGS: c_uint = H5O_HDR_CHUNK0_SIZE
    | H5O_HDR_ATTR_CRT_ORDER_TRACKED
    | H5O_HDR_ATTR_CRT_ORDER_INDEXED
    | H5O_HDR_ATTR_STORE_PHASE_CHANGE
    | H5O_HDR_STORE_TIMES;

pub const H5O_SHMESG_MAX_NINDEXES: c_uint = 8;
pub const H5O_SHMESG_MAX_LIST_SIZE: c_uint = 5000;

/// Object token type
pub const H5O_MAX_TOKEN_SIZE: usize = 16;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct H5O_token_t {
    pub __data: [u8; H5O_MAX_TOKEN_SIZE],
}

impl Default for H5O_token_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Object types
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5O_type_t {
    H5O_TYPE_UNKNOWN = -1,
    H5O_TYPE_GROUP = 0,
    H5O_TYPE_DATASET = 1,
    H5O_TYPE_NAMED_DATATYPE = 2,
    H5O_TYPE_MAP = 3,
    H5O_TYPE_NTYPES = 4,
}

/// Header info
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5O_hdr_info_t {
    pub version: c_uint,
    pub nmesgs: c_uint,
    pub nchunks: c_uint,
    pub flags: c_uint,
    pub space: H5O_hdr_info_t__space,
    pub mesg: H5O_hdr_info_t__mesg,
}

impl Default for H5O_hdr_info_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5O_hdr_info_t__space {
    pub total: hsize_t,
    pub meta: hsize_t,
    pub mesg: hsize_t,
    pub free: hsize_t,
}

impl Default for H5O_hdr_info_t__space {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5O_hdr_info_t__mesg {
    pub present: u64,
    pub shared: u64,
}

impl Default for H5O_hdr_info_t__mesg {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Object info (version 1)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5O_info1_t {
    pub fileno: c_ulong,
    pub addr: haddr_t,
    pub type_: H5O_type_t,
    pub rc: c_uint,
    pub atime: time_t,
    pub mtime: time_t,
    pub ctime: time_t,
    pub btime: time_t,
    pub num_attrs: hsize_t,
    pub hdr: H5O_hdr_info_t,
    pub meta_size: H5O_info1_t__meta_size,
}

impl Default for H5O_info1_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5O_info1_t__meta_size {
    pub obj: H5_ih_info_t,
    pub attr: H5_ih_info_t,
}

impl Default for H5O_info1_t__meta_size {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Object info (version 2)
#[repr(C)]
#[derive(Copy, Clone)]
pub struct H5O_info2_t {
    pub fileno: c_ulong,
    pub token: H5O_token_t,
    pub type_: H5O_type_t,
    pub rc: c_uint,
    pub atime: time_t,
    pub mtime: time_t,
    pub ctime: time_t,
    pub btime: time_t,
    pub num_attrs: hsize_t,
}

impl Default for H5O_info2_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Native info
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5O_native_info_t {
    pub hdr: H5O_hdr_info_t,
    pub meta_size: H5O_native_info_t__meta_size,
}

impl Default for H5O_native_info_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5O_native_info_t__meta_size {
    pub obj: H5_ih_info_t,
    pub attr: H5_ih_info_t,
}

impl Default for H5O_native_info_t__meta_size {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Object stat (deprecated)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5O_stat_t {
    pub size: hsize_t,
    pub free: hsize_t,
    pub nmesgs: c_uint,
    pub nchunks: c_uint,
}

impl Default for H5O_stat_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Message creation index
pub type H5O_msg_crt_idx_t = u32;

/// Merged committed datatype search callback return
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5O_mcdt_search_ret_t {
    H5O_MCDT_SEARCH_ERROR = -1,
    H5O_MCDT_SEARCH_CONT = 0,
    H5O_MCDT_SEARCH_STOP = 1,
}

/// Iterate callback (version 1)
pub type H5O_iterate1_t = Option<
    unsafe extern "C" fn(
        obj: hid_t,
        name: *const c_char,
        info: *const H5O_info1_t,
        op_data: *mut c_void,
    ) -> herr_t,
>;

/// Iterate callback (version 2)
pub type H5O_iterate2_t = Option<
    unsafe extern "C" fn(
        obj: hid_t,
        name: *const c_char,
        info: *const H5O_info2_t,
        op_data: *mut c_void,
    ) -> herr_t,
>;

/// Merged datatype search callback
pub type H5O_mcdt_search_cb_t =
    Option<unsafe extern "C" fn(op_data: *mut c_void) -> H5O_mcdt_search_ret_t>;

// Type aliases for current API
pub type H5O_info_t = H5O_info2_t;
pub type H5O_iterate_t = H5O_iterate2_t;
