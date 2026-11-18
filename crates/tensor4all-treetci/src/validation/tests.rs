use super::*;

#[test]
fn samples_are_deduplicated_updated_and_replayed_in_observation_order() {
    let mut history = SampleHistory::new(&[2, 3], 4096).unwrap();
    history
        .record(
            GlobalIndexBatch::new(&[1, 2, 0, 1, 1, 2], 2, 3).unwrap(),
            &[7.0, 4.0, 9.0],
        )
        .unwrap();
    assert_eq!(history.len(), 2);
    assert!(!history.exhausted());
    let mut points = vec![99];
    let mut values = vec![99.0];
    assert_eq!(history.copy_batch(0, 1, &mut points, &mut values), 1);
    assert_eq!(points, [1, 2]);
    assert_eq!(values, [9.0]);
    assert_eq!(
        history.copy_batch(1, usize::MAX, &mut points, &mut values),
        1
    );
    assert_eq!(points, [0, 1]);
    assert_eq!(values, [4.0]);
    assert_eq!(history.copy_batch(3, 1, &mut points, &mut values), 0);
    assert!(points.is_empty() && values.is_empty());
}

#[test]
fn a_full_budget_keeps_known_samples_and_marks_missing_new_samples() {
    let mut sizing = SampleHistory::new(&[2], 4096).unwrap();
    sizing
        .record(GlobalIndexBatch::new(&[0], 1, 1).unwrap(), &[2.0])
        .unwrap();
    let mut history = SampleHistory::new(&[2], sizing.retained_bytes()).unwrap();
    history
        .record(
            GlobalIndexBatch::new(&[0, 1, 0], 1, 3).unwrap(),
            &[2.0, 3.0, 5.0],
        )
        .unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history.retained_bytes(), sizing.retained_bytes());
    assert!(history.exhausted());
    let mut points = Vec::new();
    let mut values = Vec::new();
    history.copy_batch(0, 1, &mut points, &mut values);
    assert_eq!(points, [0]);
    assert_eq!(values, [5.0]);
}

#[test]
fn invalid_shapes_values_and_dimensions_are_rejected() {
    assert!(SampleHistory::<f64>::new(&[0], 4096).is_err());
    let mut history = SampleHistory::<f64>::new(&[2, 2], 4096).unwrap();
    assert!(history
        .record(GlobalIndexBatch::new(&[0], 1, 1).unwrap(), &[1.0])
        .is_err());
    assert!(history
        .record(GlobalIndexBatch::new(&[0, 0], 2, 1).unwrap(), &[])
        .is_err());
    assert!(history
        .record(GlobalIndexBatch::new(&[2, 0], 2, 1).unwrap(), &[1.0])
        .is_err());
    assert_eq!(history.len(), 0);
}

#[test]
fn wide_compact_keys_preserve_exact_points() {
    let dims = vec![2; 513];
    let mut history = SampleHistory::new(&dims, 1 << 16).unwrap();
    let mut point = vec![0; 513];
    point[512] = 1;
    history
        .record(GlobalIndexBatch::new(&point, 513, 1).unwrap(), &[3.0])
        .unwrap();
    let mut points = Vec::new();
    let mut values = Vec::new();
    assert_eq!(history.copy_batch(0, 1, &mut points, &mut values), 1);
    assert_eq!(points, point);
    assert_eq!(values, [3.0]);
}

#[test]
fn final_materialization_does_not_extend_a_completed_validation_set() {
    let mut history = SampleHistory::new(&[2], 4096).unwrap();
    history
        .record(GlobalIndexBatch::new(&[0], 1, 1).unwrap(), &[2.0])
        .unwrap();
    history.finish_recording();
    history
        .record(GlobalIndexBatch::new(&[0, 1], 1, 2).unwrap(), &[9.0, 5.0])
        .unwrap();
    assert_eq!(history.len(), 1);
    let mut points = Vec::new();
    let mut values = Vec::new();
    history.copy_batch(0, 2, &mut points, &mut values);
    assert_eq!(points, [0]);
    assert_eq!(values, [2.0]);
}
