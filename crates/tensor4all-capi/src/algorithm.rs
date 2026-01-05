//! C API for algorithm selection types
//!
//! Provides functions to work with algorithm enums from C/FFI.

use crate::{
    t4a_compression_algorithm, t4a_contraction_algorithm, t4a_factorize_algorithm,
    StatusCode, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
};
use std::ffi::CStr;
use std::os::raw::c_char;

// ============================================================================
// Factorize Algorithm
// ============================================================================

/// Get factorize algorithm from integer value.
///
/// # Arguments
/// * `value` - Integer value (0=SVD, 1=LU, 2=CI)
/// * `out_alg` - Output pointer for the algorithm
///
/// # Returns
/// - `T4A_SUCCESS` on success
/// - `T4A_NULL_POINTER` if out_alg is null
/// - `T4A_INVALID_ARGUMENT` if value is not a valid algorithm
#[unsafe(no_mangle)]
pub extern "C" fn t4a_factorize_algorithm_from_i32(
    value: i32,
    out_alg: *mut t4a_factorize_algorithm,
) -> StatusCode {
    if out_alg.is_null() {
        return T4A_NULL_POINTER;
    }

    let alg = match value {
        0 => t4a_factorize_algorithm::SVD,
        1 => t4a_factorize_algorithm::LU,
        2 => t4a_factorize_algorithm::CI,
        _ => return T4A_INVALID_ARGUMENT,
    };

    unsafe {
        *out_alg = alg;
    }
    T4A_SUCCESS
}

/// Get the name of a factorize algorithm.
///
/// # Arguments
/// * `alg` - The algorithm
///
/// # Returns
/// A static string pointer (never null, valid for the lifetime of the program)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_factorize_algorithm_name(alg: t4a_factorize_algorithm) -> *const c_char {
    let name = match alg {
        t4a_factorize_algorithm::SVD => "svd\0",
        t4a_factorize_algorithm::LU => "lu\0",
        t4a_factorize_algorithm::CI => "ci\0",
    };
    name.as_ptr() as *const c_char
}

/// Get the default factorize algorithm.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_factorize_algorithm_default() -> t4a_factorize_algorithm {
    t4a_factorize_algorithm::default()
}

// ============================================================================
// Contraction Algorithm
// ============================================================================

/// Get contraction algorithm from integer value.
///
/// # Arguments
/// * `value` - Integer value (0=Naive, 1=ZipUp, 2=Fit)
/// * `out_alg` - Output pointer for the algorithm
///
/// # Returns
/// - `T4A_SUCCESS` on success
/// - `T4A_NULL_POINTER` if out_alg is null
/// - `T4A_INVALID_ARGUMENT` if value is not a valid algorithm
#[unsafe(no_mangle)]
pub extern "C" fn t4a_contraction_algorithm_from_i32(
    value: i32,
    out_alg: *mut t4a_contraction_algorithm,
) -> StatusCode {
    if out_alg.is_null() {
        return T4A_NULL_POINTER;
    }

    let alg = match value {
        0 => t4a_contraction_algorithm::Naive,
        1 => t4a_contraction_algorithm::ZipUp,
        2 => t4a_contraction_algorithm::Fit,
        _ => return T4A_INVALID_ARGUMENT,
    };

    unsafe {
        *out_alg = alg;
    }
    T4A_SUCCESS
}

/// Get the name of a contraction algorithm.
///
/// # Arguments
/// * `alg` - The algorithm
///
/// # Returns
/// A static string pointer (never null, valid for the lifetime of the program)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_contraction_algorithm_name(alg: t4a_contraction_algorithm) -> *const c_char {
    let name = match alg {
        t4a_contraction_algorithm::Naive => "naive\0",
        t4a_contraction_algorithm::ZipUp => "zipup\0",
        t4a_contraction_algorithm::Fit => "fit\0",
    };
    name.as_ptr() as *const c_char
}

/// Get the default contraction algorithm.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_contraction_algorithm_default() -> t4a_contraction_algorithm {
    t4a_contraction_algorithm::default()
}

// ============================================================================
// Compression Algorithm
// ============================================================================

/// Get compression algorithm from integer value.
///
/// # Arguments
/// * `value` - Integer value (0=SVD, 1=LU, 2=CI, 3=Variational)
/// * `out_alg` - Output pointer for the algorithm
///
/// # Returns
/// - `T4A_SUCCESS` on success
/// - `T4A_NULL_POINTER` if out_alg is null
/// - `T4A_INVALID_ARGUMENT` if value is not a valid algorithm
#[unsafe(no_mangle)]
pub extern "C" fn t4a_compression_algorithm_from_i32(
    value: i32,
    out_alg: *mut t4a_compression_algorithm,
) -> StatusCode {
    if out_alg.is_null() {
        return T4A_NULL_POINTER;
    }

    let alg = match value {
        0 => t4a_compression_algorithm::SVD,
        1 => t4a_compression_algorithm::LU,
        2 => t4a_compression_algorithm::CI,
        3 => t4a_compression_algorithm::Variational,
        _ => return T4A_INVALID_ARGUMENT,
    };

    unsafe {
        *out_alg = alg;
    }
    T4A_SUCCESS
}

/// Get the name of a compression algorithm.
///
/// # Arguments
/// * `alg` - The algorithm
///
/// # Returns
/// A static string pointer (never null, valid for the lifetime of the program)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_compression_algorithm_name(alg: t4a_compression_algorithm) -> *const c_char {
    let name = match alg {
        t4a_compression_algorithm::SVD => "svd\0",
        t4a_compression_algorithm::LU => "lu\0",
        t4a_compression_algorithm::CI => "ci\0",
        t4a_compression_algorithm::Variational => "variational\0",
    };
    name.as_ptr() as *const c_char
}

/// Get the default compression algorithm.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_compression_algorithm_default() -> t4a_compression_algorithm {
    t4a_compression_algorithm::default()
}

// ============================================================================
// Algorithm from string
// ============================================================================

/// Get factorize algorithm from string name.
///
/// # Arguments
/// * `name` - Algorithm name ("svd", "lu", "ci")
/// * `out_alg` - Output pointer for the algorithm
///
/// # Returns
/// - `T4A_SUCCESS` on success
/// - `T4A_NULL_POINTER` if name or out_alg is null
/// - `T4A_INVALID_ARGUMENT` if name is not a valid algorithm name
#[unsafe(no_mangle)]
pub extern "C" fn t4a_factorize_algorithm_from_name(
    name: *const c_char,
    out_alg: *mut t4a_factorize_algorithm,
) -> StatusCode {
    if name.is_null() || out_alg.is_null() {
        return T4A_NULL_POINTER;
    }

    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(_) => return T4A_INVALID_ARGUMENT,
        }
    };

    let alg = match name_str.to_lowercase().as_str() {
        "svd" => t4a_factorize_algorithm::SVD,
        "lu" => t4a_factorize_algorithm::LU,
        "ci" | "cross" | "crossinterpolation" => t4a_factorize_algorithm::CI,
        _ => return T4A_INVALID_ARGUMENT,
    };

    unsafe {
        *out_alg = alg;
    }
    T4A_SUCCESS
}

/// Get contraction algorithm from string name.
///
/// # Arguments
/// * `name` - Algorithm name ("naive", "zipup", "fit")
/// * `out_alg` - Output pointer for the algorithm
///
/// # Returns
/// - `T4A_SUCCESS` on success
/// - `T4A_NULL_POINTER` if name or out_alg is null
/// - `T4A_INVALID_ARGUMENT` if name is not a valid algorithm name
#[unsafe(no_mangle)]
pub extern "C" fn t4a_contraction_algorithm_from_name(
    name: *const c_char,
    out_alg: *mut t4a_contraction_algorithm,
) -> StatusCode {
    if name.is_null() || out_alg.is_null() {
        return T4A_NULL_POINTER;
    }

    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(_) => return T4A_INVALID_ARGUMENT,
        }
    };

    let alg = match name_str.to_lowercase().as_str() {
        "naive" => t4a_contraction_algorithm::Naive,
        "zipup" | "zip_up" | "zip-up" => t4a_contraction_algorithm::ZipUp,
        "fit" | "variational" => t4a_contraction_algorithm::Fit,
        _ => return T4A_INVALID_ARGUMENT,
    };

    unsafe {
        *out_alg = alg;
    }
    T4A_SUCCESS
}

/// Get compression algorithm from string name.
///
/// # Arguments
/// * `name` - Algorithm name ("svd", "lu", "ci", "variational")
/// * `out_alg` - Output pointer for the algorithm
///
/// # Returns
/// - `T4A_SUCCESS` on success
/// - `T4A_NULL_POINTER` if name or out_alg is null
/// - `T4A_INVALID_ARGUMENT` if name is not a valid algorithm name
#[unsafe(no_mangle)]
pub extern "C" fn t4a_compression_algorithm_from_name(
    name: *const c_char,
    out_alg: *mut t4a_compression_algorithm,
) -> StatusCode {
    if name.is_null() || out_alg.is_null() {
        return T4A_NULL_POINTER;
    }

    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(_) => return T4A_INVALID_ARGUMENT,
        }
    };

    let alg = match name_str.to_lowercase().as_str() {
        "svd" => t4a_compression_algorithm::SVD,
        "lu" => t4a_compression_algorithm::LU,
        "ci" | "cross" | "crossinterpolation" => t4a_compression_algorithm::CI,
        "variational" | "fit" => t4a_compression_algorithm::Variational,
        _ => return T4A_INVALID_ARGUMENT,
    };

    unsafe {
        *out_alg = alg;
    }
    T4A_SUCCESS
}
