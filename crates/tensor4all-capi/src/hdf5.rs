//! C API for HDF5 serialization (ITensors.jl compatible format).
//!
//! Provides save/load functions for tensors and tensor trains in HDF5 format.

use std::ffi::CStr;
use std::panic::catch_unwind;

use crate::types::{t4a_tensor, t4a_tensortrain};
use crate::{StatusCode, T4A_INTERNAL_ERROR, T4A_NULL_POINTER, T4A_SUCCESS};

/// Helper to convert a C string pointer to a Rust &str.
///
/// Returns None if the pointer is null or the string is not valid UTF-8.
fn cstr_to_str<'a>(ptr: *const libc::c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().ok()
}

/// Save a tensor as an ITensors.jl-compatible `ITensor` in an HDF5 file.
///
/// # Arguments
/// - `filepath`: Path to the HDF5 file (will be created/overwritten)
/// - `name`: Name of the HDF5 group to write the tensor to
/// - `tensor`: Tensor to save
///
/// # Returns
/// `T4A_SUCCESS` on success, or an error code on failure.
///
/// # Safety
/// All pointers must be valid. String pointers must be null-terminated UTF-8.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_hdf5_save_itensor(
    filepath: *const libc::c_char,
    name: *const libc::c_char,
    tensor: *const t4a_tensor,
) -> StatusCode {
    let filepath = match cstr_to_str(filepath) {
        Some(s) => s,
        None => return T4A_NULL_POINTER,
    };
    let name = match cstr_to_str(name) {
        Some(s) => s,
        None => return T4A_NULL_POINTER,
    };
    if tensor.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tensor_ref = unsafe { &*tensor };
        match tensor4all_hdf5::save_itensor(filepath, name, tensor_ref.inner()) {
            Ok(()) => T4A_SUCCESS,
            Err(_) => T4A_INTERNAL_ERROR,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Load a tensor from an ITensors.jl-compatible `ITensor` in an HDF5 file.
///
/// # Arguments
/// - `filepath`: Path to the HDF5 file
/// - `name`: Name of the HDF5 group containing the tensor
/// - `out`: Output pointer to write the loaded tensor
///
/// # Returns
/// `T4A_SUCCESS` on success, or an error code on failure.
///
/// # Safety
/// All pointers must be valid. String pointers must be null-terminated UTF-8.
/// The returned tensor must be freed with `t4a_tensor_release()`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_hdf5_load_itensor(
    filepath: *const libc::c_char,
    name: *const libc::c_char,
    out: *mut *mut t4a_tensor,
) -> StatusCode {
    let filepath = match cstr_to_str(filepath) {
        Some(s) => s,
        None => return T4A_NULL_POINTER,
    };
    let name = match cstr_to_str(name) {
        Some(s) => s,
        None => return T4A_NULL_POINTER,
    };
    if out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(
        || match tensor4all_hdf5::load_itensor(filepath, name) {
            Ok(tensor) => {
                let boxed = Box::new(t4a_tensor::new(tensor));
                unsafe { *out = Box::into_raw(boxed) };
                T4A_SUCCESS
            }
            Err(_) => T4A_INTERNAL_ERROR,
        },
    ));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Save a tensor train as an ITensorMPS.jl-compatible `MPS` in an HDF5 file.
///
/// # Arguments
/// - `filepath`: Path to the HDF5 file (will be created/overwritten)
/// - `name`: Name of the HDF5 group to write the MPS to
/// - `tt`: Tensor train to save
///
/// # Returns
/// `T4A_SUCCESS` on success, or an error code on failure.
///
/// # Safety
/// All pointers must be valid. String pointers must be null-terminated UTF-8.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_hdf5_save_mps(
    filepath: *const libc::c_char,
    name: *const libc::c_char,
    tt: *const t4a_tensortrain,
) -> StatusCode {
    let filepath = match cstr_to_str(filepath) {
        Some(s) => s,
        None => return T4A_NULL_POINTER,
    };
    let name = match cstr_to_str(name) {
        Some(s) => s,
        None => return T4A_NULL_POINTER,
    };
    if tt.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt_ref = unsafe { &*tt };
        match tensor4all_hdf5::save_mps(filepath, name, tt_ref.inner()) {
            Ok(()) => T4A_SUCCESS,
            Err(_) => T4A_INTERNAL_ERROR,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Load a tensor train from an ITensorMPS.jl-compatible `MPS` in an HDF5 file.
///
/// # Arguments
/// - `filepath`: Path to the HDF5 file
/// - `name`: Name of the HDF5 group containing the MPS
/// - `out`: Output pointer to write the loaded tensor train
///
/// # Returns
/// `T4A_SUCCESS` on success, or an error code on failure.
///
/// # Safety
/// All pointers must be valid. String pointers must be null-terminated UTF-8.
/// The returned tensor train must be freed with `t4a_tensortrain_release()`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_hdf5_load_mps(
    filepath: *const libc::c_char,
    name: *const libc::c_char,
    out: *mut *mut t4a_tensortrain,
) -> StatusCode {
    let filepath = match cstr_to_str(filepath) {
        Some(s) => s,
        None => return T4A_NULL_POINTER,
    };
    let name = match cstr_to_str(name) {
        Some(s) => s,
        None => return T4A_NULL_POINTER,
    };
    if out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(
        || match tensor4all_hdf5::load_mps(filepath, name) {
            Ok(tt) => {
                let boxed = Box::new(t4a_tensortrain::new(tt));
                unsafe { *out = Box::into_raw(boxed) };
                T4A_SUCCESS
            }
            Err(_) => T4A_INTERNAL_ERROR,
        },
    ));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}
