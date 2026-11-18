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
    assert!(!history.limit_reached());
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
fn a_full_budget_evicts_recent_samples_without_growing_storage() {
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
    assert!(history.limit_reached());
    assert_eq!(history.evictions(), 2);
    let mut points = Vec::new();
    let mut values = Vec::new();
    history.copy_batch(0, 1, &mut points, &mut values);
    assert_eq!(points, [0]);
    assert_eq!(values, [5.0]);
}

fn with_capacity(capacity: usize) -> SampleHistory<f64> {
    let mut sizing = SampleHistory::new(&[8], 4096).unwrap();
    sizing
        .record(GlobalIndexBatch::new(&[0], 1, 1).unwrap(), &[0.0])
        .unwrap();
    SampleHistory::new(&[8], capacity * sizing.retained_bytes()).unwrap()
}

#[test]
fn ordinary_traffic_cannot_evict_the_strongest_residual_witness() {
    let mut history = with_capacity(2);
    history
        .record(GlobalIndexBatch::new(&[0, 1], 1, 2).unwrap(), &[0.0, 10.0])
        .unwrap();
    let bytes = history.retained_bytes();
    history.begin_validation();
    history
        .observe_residuals(
            GlobalIndexBatch::new(&[0, 1], 1, 2).unwrap(),
            &[0.0, 10.0],
            &[0.0, 3.0],
        )
        .unwrap();
    history
        .record(
            GlobalIndexBatch::new(&[2, 3, 4, 5], 1, 4).unwrap(),
            &[20.0, 30.0, 40.0, 50.0],
        )
        .unwrap();
    assert!(history.contains(&[1]).unwrap());
    assert!(history.contains(&[5]).unwrap());
    assert!(!history.contains(&[0]).unwrap());
    assert_eq!(history.len(), 2);
    assert_eq!(history.retained_bytes(), bytes);
    assert_eq!(history.evictions(), 4);
    let mut points = Vec::new();
    let mut values = Vec::new();
    history.copy_batch(0, 2, &mut points, &mut values);
    assert_eq!(points, [5, 1]);
    assert_eq!(values, [50.0, 10.0]);
}

#[test]
fn a_stronger_measured_point_can_replace_a_witness_in_a_one_sample_pool() {
    let mut history = with_capacity(1);
    history
        .record(GlobalIndexBatch::new(&[0], 1, 1).unwrap(), &[2.0])
        .unwrap();
    history.begin_validation();
    history
        .observe_residuals(GlobalIndexBatch::new(&[0], 1, 1).unwrap(), &[2.0], &[4.0])
        .unwrap();
    history
        .record(GlobalIndexBatch::new(&[1], 1, 1).unwrap(), &[3.0])
        .unwrap();
    history
        .observe_residuals(GlobalIndexBatch::new(&[1], 1, 1).unwrap(), &[3.0], &[2.0])
        .unwrap();
    assert!(history.contains(&[0]).unwrap());
    assert!(!history.contains(&[1]).unwrap());
    assert_eq!(history.evictions(), 0);

    history.begin_validation();
    history
        .observe_residuals(GlobalIndexBatch::new(&[0], 1, 1).unwrap(), &[2.0], &[0.0])
        .unwrap();
    history
        .observe_residuals(GlobalIndexBatch::new(&[1], 1, 1).unwrap(), &[3.0], &[2.0])
        .unwrap();
    assert!(!history.contains(&[0]).unwrap());
    assert!(history.contains(&[1]).unwrap());
    assert_eq!(history.evictions(), 1);
    assert_eq!(history.len(), 1);
}

#[test]
fn a_zero_budget_limits_coverage_without_storing_samples() {
    let mut history = with_capacity(0);
    history
        .record(GlobalIndexBatch::new(&[0], 1, 1).unwrap(), &[2.0])
        .unwrap();
    history.begin_validation();
    history
        .observe_residuals(GlobalIndexBatch::new(&[0], 1, 1).unwrap(), &[2.0], &[1.0])
        .unwrap();
    assert_eq!(history.len(), 0);
    assert_eq!(history.retained_bytes(), 0);
    assert_eq!(history.evictions(), 0);
    assert!(history.limit_reached());
    assert!(!history.contains(&[0]).unwrap());
}

#[test]
fn invalid_residual_batches_do_not_change_the_pool() {
    let mut history = with_capacity(2);
    let batch = GlobalIndexBatch::new(&[0], 1, 1).unwrap();
    assert!(history.observe_residuals(batch, &[], &[1.0]).is_err());
    assert!(history.observe_residuals(batch, &[1.0], &[]).is_err());
    for error in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(history.observe_residuals(batch, &[1.0], &[error]).is_err());
    }
    assert!(history.contains(&[8]).is_err());
    assert_eq!(history.len(), 0);
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
    history
        .observe_residuals(GlobalIndexBatch::new(&[1], 1, 1).unwrap(), &[9.0], &[10.0])
        .unwrap();
    assert_eq!(history.len(), 1);
    let mut points = Vec::new();
    let mut values = Vec::new();
    history.copy_batch(0, 2, &mut points, &mut values);
    assert_eq!(points, [0]);
    assert_eq!(values, [2.0]);
}
