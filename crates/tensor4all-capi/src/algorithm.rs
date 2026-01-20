//! C API for algorithm selection types
//!
//! Provides functions to work with algorithm enums from C/FFI.

use crate::{
    t4a_compression_algorithm, t4a_contraction_algorithm, t4a_factorize_algorithm, StatusCode,
    T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
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
        3 => t4a_factorize_algorithm::QR,
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
        t4a_factorize_algorithm::QR => "qr\0",
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
        "qr" => t4a_factorize_algorithm::QR,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::ptr;

    #[test]
    fn test_factorize_algorithm_from_i32() {
        let mut out = t4a_factorize_algorithm::CI;
        assert_eq!(
            t4a_factorize_algorithm_from_i32(0, &mut out as *mut _),
            T4A_SUCCESS
        );
        assert_eq!(out, t4a_factorize_algorithm::SVD);

        assert_eq!(
            t4a_factorize_algorithm_from_i32(3, &mut out as *mut _),
            T4A_SUCCESS
        );
        assert_eq!(out, t4a_factorize_algorithm::QR);

        assert_eq!(
            t4a_factorize_algorithm_from_i32(123, &mut out as *mut _),
            T4A_INVALID_ARGUMENT
        );
        assert_eq!(
            t4a_factorize_algorithm_from_i32(0, ptr::null_mut()),
            T4A_NULL_POINTER
        );
    }

    #[test]
    fn test_contraction_algorithm_from_i32() {
        let mut out = t4a_contraction_algorithm::Fit;
        assert_eq!(
            t4a_contraction_algorithm_from_i32(0, &mut out as *mut _),
            T4A_SUCCESS
        );
        assert_eq!(out, t4a_contraction_algorithm::Naive);

        assert_eq!(
            t4a_contraction_algorithm_from_i32(2, &mut out as *mut _),
            T4A_SUCCESS
        );
        assert_eq!(out, t4a_contraction_algorithm::Fit);

        assert_eq!(
            t4a_contraction_algorithm_from_i32(-1, &mut out as *mut _),
            T4A_INVALID_ARGUMENT
        );
        assert_eq!(
            t4a_contraction_algorithm_from_i32(0, ptr::null_mut()),
            T4A_NULL_POINTER
        );
    }

    #[test]
    fn test_compression_algorithm_from_i32() {
        let mut out = t4a_compression_algorithm::Variational;
        assert_eq!(
            t4a_compression_algorithm_from_i32(0, &mut out as *mut _),
            T4A_SUCCESS
        );
        assert_eq!(out, t4a_compression_algorithm::SVD);

        assert_eq!(
            t4a_compression_algorithm_from_i32(3, &mut out as *mut _),
            T4A_SUCCESS
        );
        assert_eq!(out, t4a_compression_algorithm::Variational);

        assert_eq!(
            t4a_compression_algorithm_from_i32(99, &mut out as *mut _),
            T4A_INVALID_ARGUMENT
        );
        assert_eq!(
            t4a_compression_algorithm_from_i32(0, ptr::null_mut()),
            T4A_NULL_POINTER
        );
    }

    #[test]
    fn test_algorithm_name_functions() {
        unsafe {
            let svd = CStr::from_ptr(t4a_factorize_algorithm_name(t4a_factorize_algorithm::SVD))
                .to_str()
                .unwrap();
            assert_eq!(svd, "svd");

            let zipup = CStr::from_ptr(t4a_contraction_algorithm_name(
                t4a_contraction_algorithm::ZipUp,
            ))
            .to_str()
            .unwrap();
            assert_eq!(zipup, "zipup");

            let var = CStr::from_ptr(t4a_compression_algorithm_name(
                t4a_compression_algorithm::Variational,
            ))
            .to_str()
            .unwrap();
            assert_eq!(var, "variational");
        }
    }

    #[test]
    fn test_algorithm_from_name() {
        let mut out_f = t4a_factorize_algorithm::CI;
        let svd = CString::new("SVD").unwrap();
        assert_eq!(
            t4a_factorize_algorithm_from_name(svd.as_ptr(), &mut out_f as *mut _),
            T4A_SUCCESS
        );
        assert_eq!(out_f, t4a_factorize_algorithm::SVD);

        let mut out_c = t4a_contraction_algorithm::Naive;
        let zip_up = CString::new("zip-up").unwrap();
        assert_eq!(
            t4a_contraction_algorithm_from_name(zip_up.as_ptr(), &mut out_c as *mut _),
            T4A_SUCCESS
        );
        assert_eq!(out_c, t4a_contraction_algorithm::ZipUp);

        let mut out_comp = t4a_compression_algorithm::SVD;
        let fit = CString::new("fit").unwrap();
        assert_eq!(
            t4a_compression_algorithm_from_name(fit.as_ptr(), &mut out_comp as *mut _),
            T4A_SUCCESS
        );
        assert_eq!(out_comp, t4a_compression_algorithm::Variational);

        let bad = CString::new("nope").unwrap();
        assert_eq!(
            t4a_factorize_algorithm_from_name(bad.as_ptr(), &mut out_f as *mut _),
            T4A_INVALID_ARGUMENT
        );
        assert_eq!(
            t4a_factorize_algorithm_from_name(ptr::null(), &mut out_f as *mut _),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_factorize_algorithm_from_name(bad.as_ptr(), ptr::null_mut()),
            T4A_NULL_POINTER
        );
    }
}
