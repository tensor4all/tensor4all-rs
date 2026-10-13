use super::*;
use tenferro_cpu::CpuBackend;

#[test]
fn round_trip_preserves_all_host_dtypes_in_target_context() {
    macro_rules! assert_round_trip {
        ($ty:ty, $variant:ident, $dtype:expr, $values:expr) => {{
            let values: Vec<$ty> = $values;
            let native = Tensor::from_vec_col_major(vec![values.len()], values.clone()).unwrap();
            let logical = LogicalTensor::from_native(&native).unwrap();
            let target = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap());
            let rebuilt = target.reconstruct(&logical).unwrap();

            assert_eq!(logical.shape(), &[values.len()]);
            assert_eq!(logical.dtype(), $dtype);
            assert_eq!(logical.data().len(), values.len());
            assert!(matches!(logical.data(), LogicalTensorData::$variant(_)));
            assert_eq!(rebuilt.as_slice::<$ty>().unwrap(), values);
        }};
    }

    assert_round_trip!(f32, F32, DType::F32, vec![1.0, 2.0]);
    assert_round_trip!(f64, F64, DType::F64, vec![1.0, 2.0]);
    assert_round_trip!(i32, I32, DType::I32, vec![1, 2]);
    assert_round_trip!(i64, I64, DType::I64, vec![1, 2]);
    assert_round_trip!(bool, Bool, DType::Bool, vec![true, false]);
    assert_round_trip!(
        Complex32,
        C32,
        DType::C32,
        vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)]
    );
    assert_round_trip!(
        Complex64,
        C64,
        DType::C64,
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]
    );

    let empty = LogicalTensor::new(vec![0], LogicalTensorData::F64(Vec::new())).unwrap();
    assert!(empty.data().is_empty());
}

#[test]
fn validation_rejects_overflow_and_length_mismatch() {
    assert!(matches!(
        LogicalTensor::new(vec![usize::MAX, 2], LogicalTensorData::F64(Vec::new())),
        Err(LogicalTensorError::ShapeOverflow { .. })
    ));
    assert!(matches!(
        LogicalTensor::new(vec![2], LogicalTensorData::F64(vec![1.0])),
        Err(LogicalTensorError::ElementCountMismatch { .. })
    ));
}
