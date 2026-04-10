use tensor4all_core::{DynIndex, LinearizationOrder, TensorDynLen, TensorLike};

#[test]
fn unfuse_index_column_major_preserves_column_major_layout() {
    let fused = DynIndex::new_dyn(4);
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![fused.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let unfused = tensor
        .unfuse_index(
            &fused,
            &[left.clone(), right.clone()],
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();

    let expected = TensorDynLen::from_dense(vec![left, right], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(unfused.isapprox(&expected, 1e-12, 0.0));
}

#[test]
fn unfuse_index_row_major_reorders_fused_axis_meaning() {
    let fused = DynIndex::new_dyn(4);
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![fused.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let unfused = tensor
        .unfuse_index(
            &fused,
            &[left.clone(), right.clone()],
            LinearizationOrder::RowMajor,
        )
        .unwrap();

    let expected = TensorDynLen::from_dense(vec![left, right], vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    assert!(unfused.isapprox(&expected, 1e-12, 0.0));
}

#[test]
fn unfuse_index_rejects_dimension_mismatch() {
    let fused = DynIndex::new_dyn(4);
    let tensor = TensorDynLen::from_dense(vec![fused.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let err = tensor
        .unfuse_index(
            &fused,
            &[DynIndex::new_dyn(2), DynIndex::new_dyn(3)],
            LinearizationOrder::ColumnMajor,
        )
        .unwrap_err();

    assert!(err
        .to_string()
        .contains("product of new index dimensions must match the replaced index dimension"));
}
