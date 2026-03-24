use super::*;
use num_complex::Complex64;

#[test]
fn test_storage_kind_from_storage_diag() {
    // Test DiagF64
    let diag_f64 = Storage::from_diag_f64_col_major(vec![1.0, 2.0], 2).unwrap();
    assert_eq!(
        t4a_storage_kind::from_storage(&diag_f64),
        t4a_storage_kind::DiagF64
    );

    // Test DiagC64
    let diag_c64 = Storage::from_diag_c64_col_major(
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        2,
    )
    .unwrap();
    assert_eq!(
        t4a_storage_kind::from_storage(&diag_c64),
        t4a_storage_kind::DiagC64
    );
}

#[test]
fn test_storage_kind_from_storage_dense() {
    let dense_f64 = Storage::from_dense_f64_col_major(vec![1.0, 2.0], &[2]).unwrap();
    assert_eq!(
        t4a_storage_kind::from_storage(&dense_f64),
        t4a_storage_kind::DenseF64
    );

    let dense_c64 = Storage::from_dense_c64_col_major(
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
        &[2],
    )
    .unwrap();
    assert_eq!(
        t4a_storage_kind::from_storage(&dense_c64),
        t4a_storage_kind::DenseC64
    );
}

#[test]
fn test_canonical_form_roundtrip() {
    for form in [
        tensor4all_itensorlike::CanonicalForm::Unitary,
        tensor4all_itensorlike::CanonicalForm::LU,
        tensor4all_itensorlike::CanonicalForm::CI,
    ] {
        let ffi_form = t4a_canonical_form::from(form);
        let roundtrip = tensor4all_itensorlike::CanonicalForm::from(ffi_form);
        assert_eq!(roundtrip, form);
    }
}

#[test]
fn test_factorize_algorithm_roundtrip() {
    for alg in [
        tensor4all_core::FactorizeAlg::SVD,
        tensor4all_core::FactorizeAlg::LU,
        tensor4all_core::FactorizeAlg::CI,
        tensor4all_core::FactorizeAlg::QR,
    ] {
        let ffi_alg = t4a_factorize_algorithm::from(alg);
        let roundtrip = tensor4all_core::FactorizeAlg::from(ffi_alg);
        assert_eq!(roundtrip, alg);
    }
}

#[test]
fn test_contract_method_roundtrip_for_both_rust_apis() {
    for method in [
        tensor4all_itensorlike::ContractMethod::Zipup,
        tensor4all_itensorlike::ContractMethod::Fit,
        tensor4all_itensorlike::ContractMethod::Naive,
    ] {
        let ffi_method = t4a_contract_method::from(method);
        let roundtrip = tensor4all_itensorlike::ContractMethod::from(ffi_method);
        assert_eq!(roundtrip, method);
    }

    for method in [
        tensor4all_treetn::treetn::contraction::ContractionMethod::Zipup,
        tensor4all_treetn::treetn::contraction::ContractionMethod::Fit,
        tensor4all_treetn::treetn::contraction::ContractionMethod::Naive,
    ] {
        let ffi_method = t4a_contract_method::from(method);
        let roundtrip = tensor4all_treetn::treetn::contraction::ContractionMethod::from(ffi_method);
        assert_eq!(roundtrip, method);
    }
}

#[test]
fn test_unfolding_scheme_roundtrip() {
    for scheme in [
        quanticsgrids::UnfoldingScheme::Fused,
        quanticsgrids::UnfoldingScheme::Interleaved,
    ] {
        let ffi_scheme = t4a_unfolding_scheme::from(scheme);
        let roundtrip = quanticsgrids::UnfoldingScheme::from(ffi_scheme);
        assert_eq!(roundtrip, scheme);
    }
}

#[test]
fn test_boundary_condition_roundtrip() {
    for boundary in [
        tensor4all_quanticstransform::BoundaryCondition::Periodic,
        tensor4all_quanticstransform::BoundaryCondition::Open,
    ] {
        let ffi_boundary = t4a_boundary_condition::from(boundary);
        let roundtrip = tensor4all_quanticstransform::BoundaryCondition::from(ffi_boundary);
        assert_eq!(roundtrip, boundary);
    }
}
