use super::*;

#[test]
fn external_kind_is_named_external() {
    assert_eq!(
        tensor_dtype_name(DType::External(core::any::TypeId::of::<u8>())),
        "external"
    );
}

#[test]
fn into_dense_values_from_native_col_major_moves_the_buffer() {
    // Consuming extraction must preserve every value in column-major order for
    // all four supported scalar kinds, not just the real ones.
    let f32_values: Vec<f32> = vec![1.0, -2.0, 3.5, 4.25, -5.5, 6.0];
    let tensor =
        f32::dense_native_tensor_from_col_major_owned(f32_values.clone(), &[2, 3]).unwrap();
    assert_eq!(
        f32::into_dense_values_from_native_col_major(tensor).unwrap(),
        f32_values
    );

    let f64_values: Vec<f64> = vec![1.0, -2.0, 3.5, 4.25, -5.5, 6.0];
    let tensor =
        f64::dense_native_tensor_from_col_major_owned(f64_values.clone(), &[2, 3]).unwrap();
    assert_eq!(
        f64::into_dense_values_from_native_col_major(tensor).unwrap(),
        f64_values
    );

    let c32_values: Vec<Complex32> = (1..=6)
        .map(|index| Complex32::new(index as f32, -(index as f32) / 2.0))
        .collect();
    let tensor =
        Complex32::dense_native_tensor_from_col_major_owned(c32_values.clone(), &[2, 3]).unwrap();
    assert_eq!(
        Complex32::into_dense_values_from_native_col_major(tensor).unwrap(),
        c32_values
    );

    let c64_values: Vec<Complex64> = (1..=6)
        .map(|index| Complex64::new(index as f64, -(index as f64) / 2.0))
        .collect();
    let tensor =
        Complex64::dense_native_tensor_from_col_major_owned(c64_values.clone(), &[2, 3]).unwrap();
    assert_eq!(
        Complex64::into_dense_values_from_native_col_major(tensor).unwrap(),
        c64_values
    );
}

#[test]
fn into_dense_values_from_native_col_major_rejects_a_foreign_dtype() {
    let tensor = f32::dense_native_tensor_from_col_major(&[1.0_f32, 2.0], &[2]).unwrap();
    let error = f64::into_dense_values_from_native_col_major(tensor).unwrap_err();
    assert!(
        error.to_string().contains("tensor dtype mismatch"),
        "unexpected error: {error}"
    );
}
