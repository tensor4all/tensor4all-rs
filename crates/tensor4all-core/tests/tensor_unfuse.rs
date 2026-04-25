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

#[test]
fn fuse_indices_column_major_roundtrips_unfuse_index() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(6).unwrap();
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone(), k.clone()], data).unwrap();

    let fused_tensor = tensor
        .fuse_indices(
            &[i.clone(), j.clone()],
            fused.clone(),
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();
    assert_eq!(fused_tensor.dims(), vec![6, 2]);

    let roundtrip = fused_tensor
        .unfuse_index(&fused, &[i, j], LinearizationOrder::ColumnMajor)
        .unwrap();
    assert!(roundtrip.isapprox(&tensor, 1e-12, 0.0));
}

#[test]
fn fuse_indices_row_major_roundtrips_unfuse_index() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(6).unwrap();
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone(), k.clone()], data).unwrap();

    let fused_tensor = tensor
        .fuse_indices(
            &[i.clone(), j.clone()],
            fused.clone(),
            LinearizationOrder::RowMajor,
        )
        .unwrap();
    assert_eq!(fused_tensor.dims(), vec![6, 2]);

    let roundtrip = fused_tensor
        .unfuse_index(&fused, &[i, j], LinearizationOrder::RowMajor)
        .unwrap();
    assert!(roundtrip.isapprox(&tensor, 1e-12, 0.0));
}

#[test]
fn fuse_indices_supports_non_adjacent_axes() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(4).unwrap();
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone(), k.clone()], data).unwrap();

    let fused_tensor = tensor
        .fuse_indices(
            &[i.clone(), k.clone()],
            fused.clone(),
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();
    assert_eq!(fused_tensor.indices(), &[fused.clone(), j.clone()]);

    let roundtrip = fused_tensor
        .unfuse_index(&fused, &[i, k], LinearizationOrder::ColumnMajor)
        .unwrap()
        .permuteinds(tensor.indices())
        .unwrap();
    assert!(roundtrip.isapprox(&tensor, 1e-12, 0.0));
}

#[test]
fn fuse_indices_rejects_empty_old_indices() {
    let i = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(2).unwrap();
    let tensor = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0]).unwrap();

    let err = tensor
        .fuse_indices(&[], fused, LinearizationOrder::ColumnMajor)
        .unwrap_err();

    assert!(err
        .to_string()
        .contains("fuse_indices requires at least one index to fuse"));
}

#[test]
fn fuse_indices_rejects_duplicate_old_indices() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let fused = DynIndex::new_link(4).unwrap();
    let tensor =
        TensorDynLen::from_dense(vec![i.clone(), j], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let err = tensor
        .fuse_indices(&[i.clone(), i], fused, LinearizationOrder::ColumnMajor)
        .unwrap_err();

    assert!(err.to_string().contains("duplicate index in old_indices"));
}

#[test]
fn fuse_indices_rejects_missing_index() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let missing = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(6).unwrap();
    let tensor =
        TensorDynLen::from_dense(vec![i.clone(), j], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let err = tensor
        .fuse_indices(&[i, missing], fused, LinearizationOrder::ColumnMajor)
        .unwrap_err();

    assert!(err.to_string().contains("not found in tensor"));
}

#[test]
fn fuse_indices_rejects_dimension_mismatch() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let fused = DynIndex::new_link(5).unwrap();
    let tensor = TensorDynLen::from_dense(
        vec![i.clone(), j.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let err = tensor
        .fuse_indices(&[i, j], fused, LinearizationOrder::ColumnMajor)
        .unwrap_err();

    assert!(err
        .to_string()
        .contains("product of old index dimensions must match the replacement index dimension"));
}

#[test]
fn fuse_indices_rejects_duplicate_result_index() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let tensor =
        TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let err = tensor
        .fuse_indices(&[i], j, LinearizationOrder::ColumnMajor)
        .unwrap_err();

    assert!(err
        .to_string()
        .contains("fuse_indices result would contain duplicate index"));
}
