#![allow(non_camel_case_types, non_upper_case_globals)]
//! Creating and manipulating datatypes (H5T)
//!
//! Types and constants for HDF5 datatype operations.

use std::mem;

use super::types::*;

pub use H5T_bkg_t::*;
pub use H5T_class_t::*;
pub use H5T_cmd_t::*;
pub use H5T_conv_except_t::*;
pub use H5T_conv_ret_t::*;
pub use H5T_cset_t::*;
pub use H5T_direction_t::*;
pub use H5T_norm_t::*;
pub use H5T_order_t::*;
pub use H5T_pad_t::*;
pub use H5T_pers_t::*;
pub use H5T_sign_t::*;
pub use H5T_str_t::*;

// Constants
pub const H5T_VARIABLE: size_t = !0;
pub const H5T_OPAQUE_TAG_MAX: c_uint = 256;

/// Datatype class
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_class_t {
    H5T_NO_CLASS = -1,
    H5T_INTEGER = 0,
    H5T_FLOAT = 1,
    H5T_TIME = 2,
    H5T_STRING = 3,
    H5T_BITFIELD = 4,
    H5T_OPAQUE = 5,
    H5T_COMPOUND = 6,
    H5T_REFERENCE = 7,
    H5T_ENUM = 8,
    H5T_VLEN = 9,
    H5T_ARRAY = 10,
    H5T_NCLASSES = 11,
}

/// Byte order
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_order_t {
    H5T_ORDER_ERROR = -1,
    H5T_ORDER_LE = 0,
    H5T_ORDER_BE = 1,
    H5T_ORDER_VAX = 2,
    H5T_ORDER_MIXED = 3,
    H5T_ORDER_NONE = 4,
}

/// Sign type for integers
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_sign_t {
    H5T_SGN_ERROR = -1,
    H5T_SGN_NONE = 0,
    H5T_SGN_2 = 1,
    H5T_NSGN = 2,
}

/// Floating-point normalization
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_norm_t {
    H5T_NORM_ERROR = -1,
    H5T_NORM_IMPLIED = 0,
    H5T_NORM_MSBSET = 1,
    H5T_NORM_NONE = 2,
}

/// Character set
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_cset_t {
    H5T_CSET_ERROR = -1,
    H5T_CSET_ASCII = 0,
    H5T_CSET_UTF8 = 1,
    H5T_CSET_RESERVED_2 = 2,
    H5T_CSET_RESERVED_3 = 3,
    H5T_CSET_RESERVED_4 = 4,
    H5T_CSET_RESERVED_5 = 5,
    H5T_CSET_RESERVED_6 = 6,
    H5T_CSET_RESERVED_7 = 7,
    H5T_CSET_RESERVED_8 = 8,
    H5T_CSET_RESERVED_9 = 9,
    H5T_CSET_RESERVED_10 = 10,
    H5T_CSET_RESERVED_11 = 11,
    H5T_CSET_RESERVED_12 = 12,
    H5T_CSET_RESERVED_13 = 13,
    H5T_CSET_RESERVED_14 = 14,
    H5T_CSET_RESERVED_15 = 15,
}

impl Default for H5T_cset_t {
    fn default() -> Self {
        Self::H5T_CSET_ASCII
    }
}

pub const H5T_NCSET: H5T_cset_t = H5T_cset_t::H5T_CSET_RESERVED_2;

/// String padding
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_str_t {
    H5T_STR_ERROR = -1,
    H5T_STR_NULLTERM = 0,
    H5T_STR_NULLPAD = 1,
    H5T_STR_SPACEPAD = 2,
    H5T_STR_RESERVED_3 = 3,
    H5T_STR_RESERVED_4 = 4,
    H5T_STR_RESERVED_5 = 5,
    H5T_STR_RESERVED_6 = 6,
    H5T_STR_RESERVED_7 = 7,
    H5T_STR_RESERVED_8 = 8,
    H5T_STR_RESERVED_9 = 9,
    H5T_STR_RESERVED_10 = 10,
    H5T_STR_RESERVED_11 = 11,
    H5T_STR_RESERVED_12 = 12,
    H5T_STR_RESERVED_13 = 13,
    H5T_STR_RESERVED_14 = 14,
    H5T_STR_RESERVED_15 = 15,
}

pub const H5T_NSTR: H5T_str_t = H5T_str_t::H5T_STR_RESERVED_3;

/// Bit padding
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_pad_t {
    H5T_PAD_ERROR = -1,
    H5T_PAD_ZERO = 0,
    H5T_PAD_ONE = 1,
    H5T_PAD_BACKGROUND = 2,
    H5T_NPAD = 3,
}

/// Conversion command
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_cmd_t {
    H5T_CONV_INIT = 0,
    H5T_CONV_CONV = 1,
    H5T_CONV_FREE = 2,
}

/// Background buffer requirement
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_bkg_t {
    H5T_BKG_NO = 0,
    H5T_BKG_TEMP = 1,
    H5T_BKG_YES = 2,
}

/// Conversion data
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct H5T_cdata_t {
    pub command: H5T_cmd_t,
    pub need_bkg: H5T_bkg_t,
    pub recalc: hbool_t,
    pub _priv: *mut c_void,
}

impl Default for H5T_cdata_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Conversion persistence
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_pers_t {
    H5T_PERS_DONTCARE = -1,
    H5T_PERS_HARD = 0,
    H5T_PERS_SOFT = 1,
}

/// Conversion direction
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_direction_t {
    H5T_DIR_DEFAULT = 0,
    H5T_DIR_ASCEND = 1,
    H5T_DIR_DESCEND = 2,
}

/// Conversion exception type
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_conv_except_t {
    H5T_CONV_EXCEPT_RANGE_HI = 0,
    H5T_CONV_EXCEPT_RANGE_LOW = 1,
    H5T_CONV_EXCEPT_PRECISION = 2,
    H5T_CONV_EXCEPT_TRUNCATE = 3,
    H5T_CONV_EXCEPT_PINF = 4,
    H5T_CONV_EXCEPT_NINF = 5,
    H5T_CONV_EXCEPT_NAN = 6,
}

/// Conversion return value
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5T_conv_ret_t {
    H5T_CONV_ABORT = -1,
    H5T_CONV_UNHANDLED = 0,
    H5T_CONV_HANDLED = 1,
}

/// Variable-length data
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct hvl_t {
    pub len: size_t,
    pub p: *mut c_void,
}

impl Default for hvl_t {
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

/// Conversion function type
pub type H5T_conv_t = Option<
    extern "C" fn(
        src_id: hid_t,
        dst_id: hid_t,
        cdata: *mut H5T_cdata_t,
        nelmts: size_t,
        buf_stride: size_t,
        bkg_stride: size_t,
        buf: *mut c_void,
        bkg: *mut c_void,
        dset_xfer_plist: hid_t,
    ) -> herr_t,
>;

/// Conversion exception callback type
pub type H5T_conv_except_func_t = Option<
    extern "C" fn(
        except_type: H5T_conv_except_t,
        src_id: hid_t,
        dst_id: hid_t,
        src_buf: *mut c_void,
        dst_buf: *mut c_void,
        user_data: *mut c_void,
    ) -> H5T_conv_ret_t,
>;
