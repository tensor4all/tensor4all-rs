use anyhow::Result;
use num_complex::Complex64;
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, TensorTrain};
use tensor4all_treetn::{
    tensor_train_to_treetn, tensor_train_to_treetn_with_names,
    tensor_train_to_treetn_with_names_and_site_indices,
};

fn two_site_tensor_train_f64() -> TensorTrain<f64> {
    TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2),
        tensor3_from_data(vec![1.0, 0.5, -1.0, 2.0], 2, 2, 1),
    ])
    .expect("valid two-site tensor train")
}

fn two_site_tensor_train_c64() -> TensorTrain<Complex64> {
    TensorTrain::new(vec![
        tensor3_from_data(
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(-1.0, 0.5),
            ],
            1,
            2,
            2,
        ),
        tensor3_from_data(
            vec![
                Complex64::new(1.0, -0.5),
                Complex64::new(0.5, 1.5),
                Complex64::new(-1.0, 0.0),
                Complex64::new(2.0, 0.25),
            ],
            2,
            2,
            1,
        ),
    ])
    .expect("valid complex two-site tensor train")
}

#[test]
fn tensor_train_to_treetn_preserves_dense_values() -> Result<()> {
    let tt = two_site_tensor_train_f64();

    let (treetn, site_indices) = tensor_train_to_treetn(&tt)?;
    let dense = treetn.contract_to_tensor()?;
    let (values, _shape) = tt.fulltensor();
    let expected = TensorDynLen::from_dense(site_indices.clone(), values)?;

    assert_eq!(treetn.node_names(), vec![0, 1]);
    assert_eq!(site_indices.len(), tt.len());
    assert!((&dense - &expected).maxabs() < 1.0e-12);
    Ok(())
}

#[test]
fn tensor_train_to_treetn_with_names_uses_requested_node_names() -> Result<()> {
    let tt = two_site_tensor_train_f64();
    let node_names = vec!["site0".to_string(), "site1".to_string()];

    let (treetn, site_indices) = tensor_train_to_treetn_with_names(&tt, node_names.clone())?;

    assert_eq!(treetn.node_names(), node_names);
    assert_eq!(site_indices.len(), 2);
    Ok(())
}

#[test]
fn tensor_train_to_treetn_supports_complex_scalars() -> Result<()> {
    let tt = two_site_tensor_train_c64();

    let (treetn, site_indices) = tensor_train_to_treetn(&tt)?;
    let dense = treetn.contract_to_tensor()?;
    let (values, _shape) = tt.fulltensor();
    let expected = TensorDynLen::from_dense(site_indices, values)?;

    assert!((&dense - &expected).maxabs() < 1.0e-12);
    Ok(())
}

#[test]
fn tensor_train_to_treetn_with_site_indices_preserves_supplied_ids() -> Result<()> {
    let tt = two_site_tensor_train_f64();
    let node_names = vec!["left".to_string(), "right".to_string()];
    let site_indices = vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];

    let treetn = tensor_train_to_treetn_with_names_and_site_indices(
        &tt,
        node_names.clone(),
        site_indices.clone(),
    )?;
    let dense = treetn.contract_to_tensor()?;

    assert_eq!(treetn.node_names(), node_names);
    let dense_indices = dense.external_indices();
    assert_eq!(dense_indices.len(), site_indices.len());
    assert_eq!(dense_indices[0].id(), site_indices[0].id());
    assert_eq!(dense_indices[1].id(), site_indices[1].id());
    Ok(())
}
