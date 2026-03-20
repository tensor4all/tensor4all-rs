
use super::*;

#[test]
fn test_direct_sum_simple() {
    // Create two tensors with one common index
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(4);

    // A[i, j] - 2x3 tensor
    let a = TensorDynLen::from_dense(
        vec![i.clone(), j.clone()],
        vec![
            1.0, 2.0, 3.0, // i=0
            4.0, 5.0, 6.0, // i=1
        ],
    )
    .unwrap();

    // B[i, k] - 2x4 tensor
    let b = TensorDynLen::from_dense(
        vec![i.clone(), k.clone()],
        vec![
            10.0, 20.0, 30.0, 40.0, // i=0
            50.0, 60.0, 70.0, 80.0, // i=1
        ],
    )
    .unwrap();

    // Direct sum along (j, k)
    let (result, new_indices) = direct_sum(&a, &b, &[(j.clone(), k.clone())]).unwrap();

    // Result should be 2x7 (common i, merged j+k)
    assert_eq!(result.dims().len(), 2);
    let result_dims = result.dims();
    assert_eq!(result_dims[0], 2); // i dimension
    assert_eq!(result_dims[1], 7); // j+k dimension (3+4)
    assert_eq!(new_indices.len(), 1);
    assert_eq!(new_indices[0].dim(), 7);

    // Check column-major logical values.
    let data = result.to_vec_f64().unwrap();
    assert_eq!(
        data,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,]
    );
}

#[test]
fn test_direct_sum_multiple_pairs() {
    // Create tensors with two indices to be paired
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(3);
    let l = DynIndex::new_dyn(3);

    // A[i, j] - 2x2 tensor
    let a = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    // B[k, l] - 3x3 tensor (no common indices)
    let b = TensorDynLen::from_dense(
        vec![k.clone(), l.clone()],
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    )
    .unwrap();

    // Direct sum along (i, k) and (j, l)
    let (result, new_indices) =
        direct_sum(&a, &b, &[(i.clone(), k.clone()), (j.clone(), l.clone())]).unwrap();

    // Result should be 5x5 (2+3, 2+3)
    assert_eq!(result.dims().len(), 2);
    let result_dims = result.dims();
    assert_eq!(result_dims[0], 5);
    assert_eq!(result_dims[1], 5);
    assert_eq!(new_indices.len(), 2);
}
