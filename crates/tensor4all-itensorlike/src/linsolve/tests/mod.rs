use super::*;
use tensor4all_core::{DynIndex, IdxTensor};

fn make_tensor(indices: Vec<DynIndex>, data: Vec<f64>) -> IdxTensor {
    IdxTensor::from_dense(indices, data).unwrap()
}

#[test]
fn infer_index_mappings_returns_typed_error() {
    let op_in = DynIndex::new_dyn(2);
    let op_out = DynIndex::new_dyn(2);
    let init_site = DynIndex::new_dyn(2);

    let operator = TensorTrain::new(vec![make_tensor(
        vec![op_in, op_out],
        vec![1.0, 0.0, 0.0, 1.0],
    )])
    .unwrap();
    let init = TensorTrain::new(vec![make_tensor(vec![init_site], vec![1.0, 2.0])]).unwrap();

    let result: crate::error::Result<_> = infer_index_mappings(&operator, &init);
    let err = result.unwrap_err();

    assert!(matches!(err, TensorTrainError::OperationError { .. }));
    assert!(err.to_string().contains("Cannot auto-infer index mappings"));
}

#[test]
fn infer_index_mappings_handles_same_id_primed_output() {
    let init_site = DynIndex::new_dyn(2);
    let output_site = init_site.prime();

    let operator = TensorTrain::new(vec![make_tensor(
        vec![output_site.clone(), init_site.clone()],
        vec![1.0, 0.0, 0.0, 1.0],
    )])
    .unwrap();
    let init =
        TensorTrain::new(vec![make_tensor(vec![init_site.clone()], vec![1.0, 2.0])]).unwrap();

    let (input_mapping, output_mapping) = infer_index_mappings(&operator, &init).unwrap();
    let input_mapping = input_mapping.unwrap();
    let output_mapping = output_mapping.unwrap();

    assert_eq!(input_mapping[&0].true_index, init_site);
    assert_eq!(input_mapping[&0].internal_index, init_site);
    assert_eq!(output_mapping[&0].true_index, init_site);
    assert_eq!(output_mapping[&0].internal_index, output_site);
}
