//! HDF5 basic type definitions
//!
//! This module provides the fundamental type aliases used throughout the HDF5 API.

#![allow(non_camel_case_types)]

// Re-export raw types for convenience
pub use std::os::raw::{
    c_char, c_double, c_float, c_int, c_long, c_longlong, c_uchar, c_uint, c_ulong, c_ulonglong,
    c_void,
};

// Basic HDF5 types
pub type herr_t = c_int;
pub type htri_t = c_int;
pub type hsize_t = u64;
pub type hssize_t = i64;
pub type haddr_t = u64;
pub type hbool_t = u8;
pub type hid_t = i64;

// Size types from libc
pub type size_t = usize;
pub type ssize_t = isize;
pub type time_t = i64;
pub type off_t = i64;
