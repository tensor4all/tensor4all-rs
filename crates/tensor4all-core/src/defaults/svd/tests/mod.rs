use super::*;
use crate::index::DefaultIndex as Index;
use tenferro::Tensor as NativeTensor;

#[test]
fn compute_retained_rank_handles_edge_cases() {
    assert_eq!(compute_retained_rank(&[], 1.0e-12), 1);
    assert_eq!(compute_retained_rank(&[0.0, 0.0], 1.0e-6), 1);
    assert_eq!(compute_retained_rank(&[5.0, 1.0e-9], 1.0e-6), 1);
    assert_eq!(compute_retained_rank(&[5.0, 1.0], 1.0e-12), 2);
}

#[test]
fn singular_values_from_native_accepts_real_and_complex_dense() {
    let dense = NativeTensor::from_slice(&[3.0_f64, 1.5], &[2]).unwrap();
    assert_eq!(singular_values_from_native(&dense).unwrap(), vec![3.0, 1.5]);

    let complex =
        NativeTensor::from_slice(&[Complex64::new(1.0, 2.0), Complex64::new(0.5, -4.0)], &[2])
            .unwrap();
    assert_eq!(
        singular_values_from_native(&complex).unwrap(),
        vec![1.0, 0.5]
    );
}

#[test]
fn set_default_svd_rtol_rejects_invalid_values() {
    let original = default_svd_rtol();
    assert!(set_default_svd_rtol(f64::NAN).is_err());
    assert!(set_default_svd_rtol(-1.0).is_err());
    set_default_svd_rtol(original).unwrap();
}

#[test]
fn svd_options_accessors_roundtrip() {
    let by_rtol = SvdOptions::with_rtol(1.0e-5);
    assert_eq!(by_rtol.rtol(), Some(1.0e-5));
    assert_eq!(by_rtol.max_rank(), None);

    let by_rank = SvdOptions::with_max_rank(7);
    assert_eq!(by_rank.rtol(), None);
    assert_eq!(by_rank.max_rank(), Some(7));
}

#[test]
fn singular_values_from_native_rejects_unsupported_scalar_types() {
    let tensor = NativeTensor::from_slice(&[1.0_f32, 2.0], &[2]).unwrap();
    let err = singular_values_from_native(&tensor).unwrap_err();
    assert!(err
        .to_string()
        .contains("unsupported singular-value scalar type"));
}

#[test]
fn svd_with_max_rank_truncates_native_outputs() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor =
        TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![3.0, 0.0, 0.0, 1.0]).unwrap();

    let (u, s, v) = svd_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &SvdOptions::with_max_rank(1),
    )
    .unwrap();

    assert_eq!(u.indices.last().unwrap().dim, 1);
    assert_eq!(s.indices[0].dim, 1);
    assert_eq!(s.indices[1].dim, 1);
    assert_eq!(v.indices.last().unwrap().dim, 1);
}

#[test]
fn svd_with_invalid_rtol_is_rejected_before_linalg() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![0.0; 4]).unwrap();

    let nan = svd_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &SvdOptions::with_rtol(f64::NAN),
    );
    assert!(matches!(nan, Err(SvdError::InvalidRtol(v)) if v.is_nan()));

    let negative = svd_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &SvdOptions::with_rtol(-1.0),
    );
    assert!(matches!(negative, Err(SvdError::InvalidRtol(v)) if v == -1.0));
}
