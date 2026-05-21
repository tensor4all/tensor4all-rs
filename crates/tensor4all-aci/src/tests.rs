use crate::{AciError, AciOptions, ElementwiseBatch};

#[test]
fn default_options_are_conservative() {
    let options = AciOptions::<f64>::default();
    assert_eq!(options.max_iters, 20);
    assert_eq!(options.min_iters, 2);
    assert_eq!(options.max_bond_dim, usize::MAX);
    assert!((options.tolerance - 1e-12).abs() < 1e-15);
    assert!(!options.scale_tolerance);
    assert!(options.initial_guess.is_none());
}

#[test]
fn elementwise_batch_uses_column_major_input_point_layout() {
    let values = vec![10.0, 20.0, 11.0, 21.0, 12.0, 22.0];
    let batch = ElementwiseBatch::new(&values, 2, 3).unwrap();
    assert_eq!(batch.n_inputs(), 2);
    assert_eq!(batch.n_points(), 3);
    assert_eq!(batch.get(0, 0).unwrap(), 10.0);
    assert_eq!(batch.get(1, 0).unwrap(), 20.0);
    assert_eq!(batch.get(0, 2).unwrap(), 12.0);
    assert_eq!(batch.get(1, 2).unwrap(), 22.0);
    assert_eq!(batch.as_col_major_slice(), values.as_slice());
}

#[test]
fn elementwise_batch_rejects_bad_length() {
    let err = ElementwiseBatch::<f64>::new(&[1.0, 2.0, 3.0], 2, 2).unwrap_err();
    assert!(err.to_string().contains("length"));
}

#[test]
fn elementwise_batch_rejects_zero_inputs() {
    let err = ElementwiseBatch::<f64>::new(&[], 0, 1).unwrap_err();
    assert!(matches!(err, AciError::EmptyInputs));
    assert!(err.to_string().contains("at least one input"));
}

#[test]
fn elementwise_batch_rejects_zero_points() {
    let err = ElementwiseBatch::<f64>::new(&[], 1, 0).unwrap_err();
    assert!(matches!(err, AciError::EmptyInputs));
    assert!(err.to_string().contains("interpolation point"));
}

#[test]
fn elementwise_batch_rejects_shape_overflow() {
    let err = ElementwiseBatch::<f64>::new(&[], usize::MAX, 2).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("overflows usize"));
}

#[test]
fn elementwise_batch_rejects_out_of_range_input_index() {
    let values = [10.0, 20.0, 11.0, 21.0];
    let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();

    let err = batch.get(2, 0).unwrap_err();
    assert!(matches!(
        err,
        AciError::BatchIndexOutOfBounds {
            axis: "input",
            index: 2,
            len: 2,
        }
    ));
    let message = err.to_string();
    assert!(message.contains("input index"));
    assert!(message.contains("out of bounds"));
    assert!(message.contains("len 2"));
}

#[test]
fn elementwise_batch_rejects_out_of_range_point_index() {
    let values = [10.0, 20.0, 11.0, 21.0];
    let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();

    let err = batch.get(0, 2).unwrap_err();
    assert!(matches!(
        err,
        AciError::BatchIndexOutOfBounds {
            axis: "point",
            index: 2,
            len: 2,
        }
    ));
    let message = err.to_string();
    assert!(message.contains("point index"));
    assert!(message.contains("out of bounds"));
    assert!(message.contains("len 2"));
}
