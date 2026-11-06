use super::*;
use tensor4all_tensorbackend::StorageKind;

#[test]
fn plans_diag_diag_partial_as_diag_output() {
    let operands = vec![
        OperandLayout::new(vec![3, 3], vec![0, 0]).unwrap(),
        OperandLayout::new(vec![3, 3], vec![0, 0]).unwrap(),
    ];
    let spec = StructuredContractionSpec {
        input_labels: vec![vec![0, 1], vec![1, 2]],
        output_labels: vec![0, 2],
        retained_labels: Default::default(),
    };

    let plan = StructuredContractionPlan::new(&operands, &spec).unwrap();
    assert_eq!(plan.output_axis_classes, vec![0, 0]);
    assert_eq!(plan.output_payload_roots.len(), 1);
}

#[test]
fn plans_general_structured_output_classes() {
    let operands = vec![
        OperandLayout::new(vec![2, 2, 3], vec![0, 0, 1]).unwrap(),
        OperandLayout::new(vec![3, 5, 5], vec![0, 1, 1]).unwrap(),
    ];
    let spec = StructuredContractionSpec {
        input_labels: vec![vec![0, 1, 2], vec![2, 3, 4]],
        output_labels: vec![0, 1, 3, 4],
        retained_labels: Default::default(),
    };

    let plan = StructuredContractionPlan::new(&operands, &spec).unwrap();
    assert_eq!(plan.output_axis_classes, vec![0, 0, 1, 1]);
    assert_eq!(plan.output_payload_dims, vec![2, 5]);
}

#[test]
fn detects_payload_repeated_labels() {
    let operands = vec![
        OperandLayout::new(vec![3, 3], vec![0, 1]).unwrap(),
        OperandLayout::new(vec![3, 3], vec![0, 0]).unwrap(),
    ];
    let spec = StructuredContractionSpec {
        input_labels: vec![vec![0, 1], vec![0, 1]],
        output_labels: vec![],
        retained_labels: Default::default(),
    };

    let plan = StructuredContractionPlan::new(&operands, &spec).unwrap();
    assert!(plan.operand_plans[0].has_repeated_roots());
}

#[test]
fn normalizes_repeated_payload_roots_by_extracting_diagonal() {
    let payload =
        tenferro::Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let (normalized, roots) = normalize_payload_for_roots(&payload, &[0, 0]).unwrap();

    assert_eq!(normalized.shape(), &[2]);
    assert_eq!(normalized.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    assert_eq!(roots, vec![0]);
}

#[test]
fn storage_payload_native_roundtrip_preserves_compact_payload() {
    let storage = Storage::new_structured(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1, 2],
        vec![0, 1, 0],
    )
    .unwrap();

    let native = storage_payload_native(&storage).unwrap();
    assert_eq!(native.shape(), &[2, 3]);
    assert_eq!(
        native.as_slice::<f64>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let rebuilt = storage_from_payload_native(native, &[2, 3], vec![0, 1, 0]).unwrap();
    assert_eq!(rebuilt.storage_kind(), StorageKind::Structured);
    assert_eq!(rebuilt.payload_dims(), &[2, 3]);
    assert_eq!(rebuilt.axis_classes(), &[0, 1, 0]);
    assert_eq!(
        rebuilt.payload_f64_col_major_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}
