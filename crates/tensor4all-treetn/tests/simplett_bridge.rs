use anyhow::Result;
use num_complex::Complex64;
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, TensorTrain};
use tensor4all_treetn::{
    insert_onehot_site_in_treetn_chain, tensor_train_to_treetn, tensor_train_to_treetn_with_names,
    tensor_train_to_treetn_with_names_and_site_indices, treetn_to_tensor_train, TreeTN,
};

fn two_site_tensor_train_f64() -> TensorTrain<f64> {
    TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2).unwrap(),
        tensor3_from_data(vec![1.0, 0.5, -1.0, 2.0], 2, 2, 1).unwrap(),
    ])
    .expect("valid two-site tensor train")
}

fn single_site_tensor_train_f64() -> TensorTrain<f64> {
    TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, -2.0, 3.5], 1, 3, 1).unwrap()
    ])
    .expect("valid single-site tensor train")
}

fn three_site_tensor_train_f64() -> TensorTrain<f64> {
    TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2).unwrap(),
        tensor3_from_data(
            vec![
                1.0, 0.5, -1.0, 2.0, //
                0.25, -0.75, 1.5, 0.0,
            ],
            2,
            2,
            2,
        )
        .unwrap(),
        tensor3_from_data(vec![0.5, 1.5, -2.0, 1.0], 2, 2, 1).unwrap(),
    ])
    .expect("valid three-site tensor train")
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
        )
        .unwrap(),
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
        )
        .unwrap(),
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
    assert!(dense.distance(&expected).unwrap() < 1.0e-12);
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

    assert!(dense.distance(&expected).unwrap() < 1.0e-12);
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

#[test]
fn tensor_train_to_treetn_single_site_preserves_dense_values() -> Result<()> {
    let tt = single_site_tensor_train_f64();

    let (treetn, site_indices) = tensor_train_to_treetn(&tt)?;
    let dense = treetn.contract_to_tensor()?;
    let (values, _shape) = tt.fulltensor();
    let expected = TensorDynLen::from_dense(site_indices.clone(), values)?;

    assert_eq!(treetn.node_names(), vec![0]);
    assert_eq!(site_indices.len(), 1);
    assert!(dense.distance(&expected).unwrap() < 1.0e-12);
    Ok(())
}

#[test]
fn tensor_train_to_treetn_three_site_preserves_dense_values() -> Result<()> {
    let tt = three_site_tensor_train_f64();

    let (treetn, site_indices) = tensor_train_to_treetn(&tt)?;
    let dense = treetn.contract_to_tensor()?;
    let (values, _shape) = tt.fulltensor();
    let expected = TensorDynLen::from_dense(site_indices, values)?;

    assert_eq!(treetn.node_names(), vec![0, 1, 2]);
    assert!(dense.distance(&expected).unwrap() < 1.0e-12);
    Ok(())
}

#[test]
fn treetn_to_tensor_train_roundtrips_three_site_values() -> Result<()> {
    let tt = three_site_tensor_train_f64();
    let (treetn, _site_indices) = tensor_train_to_treetn(&tt)?;

    let roundtrip = treetn_to_tensor_train::<f64>(treetn)?;

    assert_eq!(roundtrip.site_dims(), tt.site_dims());
    assert_eq!(roundtrip.link_dims(), tt.link_dims());

    let (actual, actual_shape) = roundtrip.fulltensor();
    let (expected, expected_shape) = tt.fulltensor();
    assert_eq!(actual_shape, expected_shape);
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn treetn_to_tensor_train_roundtrips_complex_values() -> Result<()> {
    let tt = two_site_tensor_train_c64();
    let (treetn, _site_indices) = tensor_train_to_treetn(&tt)?;

    let roundtrip = treetn_to_tensor_train::<Complex64>(treetn)?;

    assert_eq!(roundtrip.site_dims(), tt.site_dims());
    assert_eq!(roundtrip.link_dims(), tt.link_dims());

    let (actual, actual_shape) = roundtrip.fulltensor();
    let (expected, expected_shape) = tt.fulltensor();
    assert_eq!(actual_shape, expected_shape);
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn treetn_to_tensor_train_handles_local_axis_permutations() -> Result<()> {
    let tt = two_site_tensor_train_f64();
    let site0 = DynIndex::new_dyn(2);
    let site1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_bond(2)?;
    let t0 = TensorDynLen::from_dense(vec![bond.clone(), site0], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    let t1 = TensorDynLen::from_dense(vec![site1, bond], vec![1.0_f64, -1.0, 0.5, 2.0])?;
    let treetn = TreeTN::from_tensors(vec![t0, t1], vec![0, 1])?;

    let roundtrip = treetn_to_tensor_train::<f64>(treetn)?;

    assert_eq!(roundtrip.site_dims(), tt.site_dims());
    assert_eq!(roundtrip.link_dims(), tt.link_dims());
    assert_eq!(roundtrip.fulltensor(), tt.fulltensor());
    Ok(())
}

#[test]
fn treetn_to_tensor_train_rejects_ad_tracked_site_tensor() -> Result<()> {
    let tt = single_site_tensor_train_f64();
    let (mut treetn, _site_indices) = tensor_train_to_treetn(&tt)?;
    let node = treetn.node_index(&0).unwrap();
    let tracked_tensor = treetn.tensor(node).unwrap().clone().enable_grad()?;
    treetn.replace_tensor(node, tracked_tensor)?;

    let err = treetn_to_tensor_train::<f64>(treetn).unwrap_err();

    assert!(err.to_string().contains("tracked autodiff"));
    Ok(())
}

#[test]
fn insert_onehot_site_in_treetn_chain_prepends_fixed_site() -> Result<()> {
    let tt = two_site_tensor_train_f64();
    let (treetn, old_sites) = tensor_train_to_treetn(&tt)?;
    let inserted_site = DynIndex::new_dyn(2);

    let result = insert_onehot_site_in_treetn_chain::<f64>(treetn, 0, inserted_site.clone(), 0)?;
    let dense = result.contract_to_tensor()?;
    let (old_values, _) = tt.fulltensor();
    let expected = TensorDynLen::from_dense(
        vec![inserted_site, old_sites[0].clone(), old_sites[1].clone()],
        vec![
            old_values[0],
            0.0,
            old_values[1],
            0.0,
            old_values[2],
            0.0,
            old_values[3],
            0.0,
        ],
    )?;

    assert_eq!(result.node_names(), vec![0, 1, 2]);
    assert!(dense.distance(&expected).unwrap() < 1.0e-12);
    Ok(())
}

#[test]
fn insert_onehot_site_in_treetn_chain_preserves_edge_bond_flow() -> Result<()> {
    let tt = two_site_tensor_train_f64();
    let (treetn, old_sites) = tensor_train_to_treetn(&tt)?;
    let inserted_site = DynIndex::new_dyn(2);

    let result = insert_onehot_site_in_treetn_chain::<f64>(treetn, 1, inserted_site.clone(), 1)?;
    let dense = result.contract_to_tensor()?;
    let (old_values, _) = tt.fulltensor();
    let expected = TensorDynLen::from_dense(
        vec![old_sites[0].clone(), inserted_site, old_sites[1].clone()],
        vec![
            0.0,
            0.0,
            old_values[0],
            old_values[1],
            0.0,
            0.0,
            old_values[2],
            old_values[3],
        ],
    )?;

    assert_eq!(result.node_names(), vec![0, 1, 2]);
    assert_eq!(
        treetn_to_tensor_train::<f64>(result.clone())?.site_dims(),
        vec![2, 2, 2]
    );
    assert!(dense.distance(&expected).unwrap() < 1.0e-12);
    Ok(())
}

#[test]
fn insert_onehot_site_in_treetn_chain_appends_fixed_site() -> Result<()> {
    let tt = two_site_tensor_train_f64();
    let (treetn, old_sites) = tensor_train_to_treetn(&tt)?;
    let inserted_site = DynIndex::new_dyn(2);

    let result = insert_onehot_site_in_treetn_chain::<f64>(treetn, 2, inserted_site.clone(), 1)?;
    let dense = result.contract_to_tensor()?;
    let (old_values, _) = tt.fulltensor();
    let expected = TensorDynLen::from_dense(
        vec![old_sites[0].clone(), old_sites[1].clone(), inserted_site],
        vec![
            0.0,
            0.0,
            0.0,
            0.0,
            old_values[0],
            old_values[1],
            old_values[2],
            old_values[3],
        ],
    )?;

    assert_eq!(result.node_names(), vec![0, 1, 2]);
    assert!(dense.distance(&expected).unwrap() < 1.0e-12);
    Ok(())
}

#[test]
fn insert_onehot_site_in_treetn_chain_rejects_invalid_position() -> Result<()> {
    let tt = two_site_tensor_train_f64();
    let (treetn, _) = tensor_train_to_treetn(&tt)?;

    let err =
        insert_onehot_site_in_treetn_chain::<f64>(treetn, 3, DynIndex::new_dyn(2), 0).unwrap_err();

    assert!(err.to_string().contains("position 3 is out of range 0..=2"));
    Ok(())
}

#[test]
fn insert_onehot_site_in_treetn_chain_rejects_invalid_fixed_value() -> Result<()> {
    let tt = two_site_tensor_train_f64();
    let (treetn, _) = tensor_train_to_treetn(&tt)?;

    let err =
        insert_onehot_site_in_treetn_chain::<f64>(treetn, 0, DynIndex::new_dyn(2), 2).unwrap_err();

    assert!(err
        .to_string()
        .contains("fixed value 2 exceeds site dimension 2"));
    Ok(())
}

#[test]
fn tensor_train_to_treetn_with_names_rejects_length_mismatch() {
    let tt = two_site_tensor_train_f64();

    let err = tensor_train_to_treetn_with_names(&tt, vec!["site0".to_string()]).unwrap_err();

    assert!(err
        .to_string()
        .contains("node_names length 1 must match tensor-train length 2"));
}

#[test]
fn tensor_train_to_treetn_with_site_indices_rejects_length_mismatch() {
    let tt = two_site_tensor_train_f64();

    let err = tensor_train_to_treetn_with_names_and_site_indices(
        &tt,
        vec!["left".to_string(), "right".to_string()],
        vec![DynIndex::new_dyn(2)],
    )
    .unwrap_err();

    assert!(err
        .to_string()
        .contains("site_indices length 1 must match tensor-train length 2"));
}

#[test]
fn tensor_train_to_treetn_with_site_indices_rejects_dimension_mismatch() {
    let tt = two_site_tensor_train_f64();

    let err = tensor_train_to_treetn_with_names_and_site_indices(
        &tt,
        vec!["left".to_string(), "right".to_string()],
        vec![DynIndex::new_dyn(3), DynIndex::new_dyn(2)],
    )
    .unwrap_err();

    assert!(err
        .to_string()
        .contains("site index 0 has dim 3 but tensor-train site 0 has dim 2"));
}

#[test]
fn tensor_train_to_treetn_empty_tensor_train_requires_empty_metadata() {
    let empty_tt = TensorTrain::<f64>::new(vec![]).expect("empty tensor train should construct");

    let (treetn, site_indices) =
        tensor_train_to_treetn_with_names(&empty_tt, Vec::<String>::new()).unwrap();
    assert!(treetn.node_names().is_empty());
    assert!(site_indices.is_empty());

    let err = tensor_train_to_treetn_with_names_and_site_indices(
        &empty_tt,
        Vec::<String>::new(),
        vec![DynIndex::new_dyn(2)],
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("empty tensor train requires zero site indices"));
}
