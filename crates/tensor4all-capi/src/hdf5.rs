//! C API for HDF5 serialization (ITensors.jl compatible format).
//!
//! Provides save/load functions for tensors and tensor trains in HDF5 format.

use std::ffi::CStr;
use std::panic::catch_unwind;

use crate::types::{t4a_tensor, t4a_tensortrain};
use crate::{StatusCode, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};

/// Convert a C string pointer to a Rust `&str`, distinguishing null from invalid UTF-8.
fn cstr_to_str_checked<'a>(ptr: *const libc::c_char) -> Result<&'a str, StatusCode> {
    if ptr.is_null() {
        return Err(T4A_NULL_POINTER);
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| T4A_INVALID_ARGUMENT)
}

/// Extract two C strings and run a closure inside `catch_unwind`.
fn with_two_cstrs_and_unwind<F>(
    s1: *const libc::c_char,
    s2: *const libc::c_char,
    f: F,
) -> StatusCode
where
    F: FnOnce(&str, &str) -> StatusCode + std::panic::UnwindSafe,
{
    let s1 = match cstr_to_str_checked(s1) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let s2 = match cstr_to_str_checked(s2) {
        Ok(s) => s,
        Err(code) => return code,
    };
    catch_unwind(|| f(s1, s2)).unwrap_or(T4A_INTERNAL_ERROR)
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
    if tensor.is_null() {
        return T4A_NULL_POINTER;
    }
    with_two_cstrs_and_unwind(filepath, name, |fp, nm| {
        let tensor_ref = unsafe { &*tensor };
        match tensor4all_hdf5::save_itensor(fp, nm, tensor_ref.inner()) {
            Ok(()) => T4A_SUCCESS,
            Err(_) => T4A_INTERNAL_ERROR,
        }
    })
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
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    with_two_cstrs_and_unwind(
        filepath,
        name,
        |fp, nm| match tensor4all_hdf5::load_itensor(fp, nm) {
            Ok(tensor) => {
                let boxed = Box::new(t4a_tensor::new(tensor));
                unsafe { *out = Box::into_raw(boxed) };
                T4A_SUCCESS
            }
            Err(_) => T4A_INTERNAL_ERROR,
        },
    )
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
    if tt.is_null() {
        return T4A_NULL_POINTER;
    }
    with_two_cstrs_and_unwind(filepath, name, |fp, nm| {
        let tt_ref = unsafe { &*tt };
        match tensor4all_hdf5::save_mps(fp, nm, tt_ref.inner()) {
            Ok(()) => T4A_SUCCESS,
            Err(_) => T4A_INTERNAL_ERROR,
        }
    })
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
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    with_two_cstrs_and_unwind(filepath, name, |fp, nm| {
        match tensor4all_hdf5::load_mps(fp, nm) {
            Ok(tt) => {
                let boxed = Box::new(t4a_tensortrain::new(tt));
                unsafe { *out = Box::into_raw(boxed) };
                T4A_SUCCESS
            }
            Err(_) => T4A_INTERNAL_ERROR,
        }
    })
}
