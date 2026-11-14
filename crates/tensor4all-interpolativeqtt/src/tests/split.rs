use super::*;

#[test]
fn split_rejects_unrepresentable_midpoints_on_any_axis() {
    let one_ulp = f64::from_bits(1.0_f64.to_bits() + 1);
    let smallest = f64::from_bits(1);
    for (lower, upper) in [
        (vec![1.0], vec![one_ulp]),
        (vec![0.0], vec![smallest]),
        (vec![0.0, 1.0], vec![1.0, one_ulp]),
        (vec![1.0], vec![1.0]),
        (vec![-f64::MAX], vec![f64::MAX]),
    ] {
        let error = NInterval::new(&lower, &upper).unwrap().split().unwrap_err();
        assert!(
            matches!(error, InterpolativeQttError::InvalidArgument { ref message }
            if message.contains("midpoint is not representable")),
            "{error}"
        );
    }
}

#[test]
fn split_preserves_representable_children_and_order() {
    let upper = f64::from_bits(1.0_f64.to_bits() + 2);
    let mid = f64::from_bits(1.0_f64.to_bits() + 1);
    let parts = NInterval::new(&[1.0, -2.0], &[upper, 2.0])
        .unwrap()
        .split()
        .unwrap();
    assert_eq!(
        parts,
        vec![
            NInterval::new(&[1.0, -2.0], &[mid, 0.0]).unwrap(),
            NInterval::new(&[mid, -2.0], &[upper, 0.0]).unwrap(),
            NInterval::new(&[1.0, 0.0], &[mid, 2.0]).unwrap(),
            NInterval::new(&[mid, 0.0], &[upper, 2.0]).unwrap(),
        ]
    );
}

#[test]
fn existing_refined_constructors_reject_unrepresentable_midpoints() {
    let upper = f64::from_bits(1.0_f64.to_bits() + 1);
    let options = InterpolativeQttOptions::default();
    let f = |_: &[f64]| -> f64 { panic!("invalid split reached sampling") };
    for result in [
        interpolate_multi_scale_nd(f, &[1.0], &[upper], 3, 2, &[], &options),
        interpolate_adaptive_nd(f, &[1.0], &[upper], 3, 2, 1e-8, &[], &options),
    ] {
        assert!(
            matches!(result, Err(InterpolativeQttError::InvalidArgument { ref message })
            if message.contains("midpoint is not representable"))
        );
    }
}
