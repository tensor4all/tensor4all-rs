
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
