//! C API for HDF5 serialization (ITensors.jl compatible format).
//!
//! Provides save/load functions for tensors and tensor trains in HDF5 format.
//! The HDF5 crate internally works with `TensorTrain`, but the C API exposes
//! `t4a_treetn` (DefaultTreeTN). Conversions happen at the boundary.

use std::ffi::CStr;
use std::panic::catch_unwind;

use tensor4all_itensorlike::TensorTrain;
use tensor4all_treetn::DefaultTreeTN;

use crate::types::{t4a_tensor, t4a_treetn};
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
    crate::unwrap_catch(catch_unwind(|| f(s1, s2)))
}

/// Initialize the HDF5 library by loading it from the specified path.
///
/// This must be called before using any HDF5 functions when the library is built
/// with runtime-loading mode. In link mode, this is a no-op.
///
/// # Arguments
/// - `library_path`: Path to the HDF5 shared library (e.g., libhdf5.so or libhdf5.dylib).
///   If null, attempts to use a default system path.
///
/// # Returns
/// `T4A_SUCCESS` on success, or an error code on failure.
///
/// # Safety
/// The library path must be a valid null-terminated UTF-8 string if non-null.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_hdf5_init(library_path: *const libc::c_char) -> StatusCode {
    let path = if library_path.is_null() {
        None
    } else {
        match cstr_to_str_checked(library_path) {
            Ok(s) => Some(s),
            Err(code) => return code,
        }
    };

    crate::unwrap_catch(catch_unwind(|| {
        match tensor4all_hdf5::hdf5_init(path) {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }))
}

/// Check if the HDF5 library is initialized.
///
/// # Returns
/// `true` if HDF5 is initialized and ready to use, `false` otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_hdf5_is_initialized() -> bool {
    tensor4all_hdf5::hdf5_is_initialized()
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
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
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
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        },
    )
}

/// Save a tree tensor network (MPS) as an ITensorMPS.jl-compatible `MPS` in an HDF5 file.
///
/// Internally converts the `DefaultTreeTN<usize>` to a `TensorTrain` for HDF5 serialization.
///
/// # Arguments
/// - `filepath`: Path to the HDF5 file (will be created/overwritten)
/// - `name`: Name of the HDF5 group to write the MPS to
/// - `ttn`: Tree tensor network (MPS) to save
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
    ttn: *const t4a_treetn,
) -> StatusCode {
    if ttn.is_null() {
        return T4A_NULL_POINTER;
    }
    with_two_cstrs_and_unwind(filepath, name, |fp, nm| {
        let ttn_ref = unsafe { &*ttn };
        let treetn: &DefaultTreeTN<usize> = ttn_ref.inner();
        // Convert DefaultTreeTN → TensorTrain for HDF5 serialization
        let tensors: Vec<_> = treetn
            .node_names()
            .iter()
            .filter_map(|name| {
                let node_idx = treetn.node_index(name)?;
                treetn.tensor(node_idx).cloned()
            })
            .collect();
        match TensorTrain::new(tensors) {
            Ok(tt) => match tensor4all_hdf5::save_mps(fp, nm, &tt) {
                Ok(()) => T4A_SUCCESS,
                Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
            },
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    })
}

/// Load a tree tensor network (MPS) from an ITensorMPS.jl-compatible `MPS` in an HDF5 file.
///
/// Internally loads as a `TensorTrain` and converts to `DefaultTreeTN<usize>`.
///
/// # Arguments
/// - `filepath`: Path to the HDF5 file
/// - `name`: Name of the HDF5 group containing the MPS
/// - `out`: Output pointer to write the loaded tree tensor network
///
/// # Returns
/// `T4A_SUCCESS` on success, or an error code on failure.
///
/// # Safety
/// All pointers must be valid. String pointers must be null-terminated UTF-8.
/// The returned tree tensor network must be freed with `t4a_treetn_release()`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_hdf5_load_mps(
    filepath: *const libc::c_char,
    name: *const libc::c_char,
    out: *mut *mut t4a_treetn,
) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    with_two_cstrs_and_unwind(filepath, name, |fp, nm| {
        match tensor4all_hdf5::load_mps(fp, nm) {
            Ok(tt) => {
                // Convert TensorTrain → DefaultTreeTN<usize>
                let tensors: Vec<_> = tt.tensors().into_iter().cloned().collect();
                let node_names: Vec<usize> = (0..tensors.len()).collect();
                match DefaultTreeTN::from_tensors(tensors, node_names) {
                    Ok(treetn) => {
                        let boxed = Box::new(t4a_treetn::new(treetn));
                        unsafe { *out = Box::into_raw(boxed) };
                        T4A_SUCCESS
                    }
                    Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
                }
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    })
}
