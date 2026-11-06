//! Tests for `restructure_to` on a Y-shape branching topology.

use super::*;

/// Create a Y-shape TreeTN with site indices on every node:
///     A (site_a)
///     |
///     B (site_b)
///    / \
///   C   D
/// (site_c) (site_d)
#[allow(clippy::type_complexity)]
fn y_shape_treetn() -> anyhow::Result<(
    TreeTN<IdxTensor, String>,
    DynIndex,
    DynIndex,
    DynIndex,
    DynIndex,
)> {
    let mut tn = TreeTN::<IdxTensor, String>::new();
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);
    let bond_ab = DynIndex::new_dyn(3);
    let bond_bc = DynIndex::new_dyn(3);
    let bond_bd = DynIndex::new_dyn(3);
    let tensor_a = IdxTensor::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6])?;
    tn.add_tensor("A".to_string(), tensor_a).unwrap();
    let tensor_b = IdxTensor::from_dense(
        vec![
            bond_ab.clone(),
            bond_bc.clone(),
            bond_bd.clone(),
            site_b.clone(),
        ],
        vec![1.0; 54],
    )?;
    tn.add_tensor("B".to_string(), tensor_b).unwrap();
    let tensor_c = IdxTensor::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6])?;
    tn.add_tensor("C".to_string(), tensor_c).unwrap();
    let tensor_d = IdxTensor::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6])?;
    tn.add_tensor("D".to_string(), tensor_d).unwrap();
    let n_a = tn.node_index(&"A".to_string()).unwrap();
    let n_b = tn.node_index(&"B".to_string()).unwrap();
    let n_c = tn.node_index(&"C".to_string()).unwrap();
    let n_d = tn.node_index(&"D".to_string()).unwrap();
    tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
    tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();
    tn.connect(n_b, &bond_bd, n_d, &bond_bd).unwrap();
    Ok((tn, site_a, site_b, site_c, site_d))
}

#[test]
fn test_restructure_to_y_shape_fuse_subtree() -> anyhow::Result<()> {
    let (tn, site_a, site_b, site_c, site_d) = y_shape_treetn()?;

    let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    target
        .add_node("A".to_string(), HashSet::from([site_a]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("BCD".to_string(), HashSet::from([site_b, site_c, site_d]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"A".to_string(), &"BCD".to_string())
        .map_err(anyhow::Error::msg)?;

    let result = tn.restructure_to(&target, &RestructureOptions::default())?;
    assert_eq!(result.node_count(), 2);
    assert_eq!(result.edge_count(), 1);
    assert!(
        result
            .site_index_network()
            .share_equivalent_site_index_network(&target),
        "Y-shape fused to chain must match target"
    );
    let dense_expected = tn.contract_to_tensor()?;
    let dense_actual = result.contract_to_tensor()?;
    assert!(dense_actual.distance(&dense_expected).unwrap() < 1e-12);
    Ok(())
}

#[test]
fn test_restructure_to_y_shape_swap_only() -> anyhow::Result<()> {
    // Build a Y-shape with all bonds visible
    let mut tn = TreeTN::<IdxTensor, String>::new();
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);
    let bond_ab = DynIndex::new_dyn(3);
    let bond_bc = DynIndex::new_dyn(3);
    let bond_bd = DynIndex::new_dyn(3);

    let tensor_a = IdxTensor::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6])?;
    tn.add_tensor("A".to_string(), tensor_a)?;
    let tensor_b = IdxTensor::from_dense(
        vec![
            bond_ab.clone(),
            bond_bc.clone(),
            bond_bd.clone(),
            site_b.clone(),
        ],
        vec![1.0; 54],
    )?;
    tn.add_tensor("B".to_string(), tensor_b)?;
    let tensor_c = IdxTensor::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6])?;
    tn.add_tensor("C".to_string(), tensor_c)?;
    let tensor_d = IdxTensor::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6])?;
    tn.add_tensor("D".to_string(), tensor_d)?;

    let n_a = tn.node_index(&"A".to_string()).unwrap();
    let n_b = tn.node_index(&"B".to_string()).unwrap();
    let n_c = tn.node_index(&"C".to_string()).unwrap();
    let n_d = tn.node_index(&"D".to_string()).unwrap();
    tn.connect(n_a, &bond_ab, n_b, &bond_ab)?;
    tn.connect(n_b, &bond_bc, n_c, &bond_bc)?;
    tn.connect(n_b, &bond_bd, n_d, &bond_bd)?;

    // SwapOnly: reassign site a→X, b→Y, c→Z, d→W, same Y topology
    let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    target
        .add_node("X".to_string(), HashSet::from([site_a]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("Y".to_string(), HashSet::from([site_b]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("Z".to_string(), HashSet::from([site_c]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("W".to_string(), HashSet::from([site_d]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"X".to_string(), &"Y".to_string())
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"Y".to_string(), &"Z".to_string())
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"Y".to_string(), &"W".to_string())
        .map_err(anyhow::Error::msg)?;

    let result = tn.restructure_to(&target, &RestructureOptions::default())?;
    assert_eq!(result.node_count(), 4);
    assert_eq!(result.edge_count(), 3);
    assert!(
        result
            .site_index_network()
            .share_equivalent_site_index_network(&target),
        "Y-shape swapped must match target"
    );
    let dense_expected = tn.contract_to_tensor()?;
    let dense_actual = result.contract_to_tensor()?;
    assert!(dense_actual.distance(&dense_expected).unwrap() < 1e-12);
    Ok(())
}

#[test]
fn test_restructure_to_y_shape_internal_center() -> anyhow::Result<()> {
    // Y-shape with internal center (B has no site indices after connection):
    //     A(site_a)
    //       |
    //     B(no site)
    //      / \
    // C(site_c) D(site_d)
    let mut tn = TreeTN::<IdxTensor, String>::new();
    let site_a = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);
    let bond_ab = DynIndex::new_dyn(3);
    let bond_bc = DynIndex::new_dyn(3);
    let bond_bd = DynIndex::new_dyn(3);

    let tensor_a = IdxTensor::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6])?;
    tn.add_tensor("A".to_string(), tensor_a)?;
    let tensor_b = IdxTensor::from_dense(
        vec![bond_ab.clone(), bond_bc.clone(), bond_bd.clone()],
        vec![1.0; 27],
    )?;
    tn.add_tensor("B".to_string(), tensor_b)?;
    let tensor_c = IdxTensor::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6])?;
    tn.add_tensor("C".to_string(), tensor_c)?;
    let tensor_d = IdxTensor::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6])?;
    tn.add_tensor("D".to_string(), tensor_d)?;

    let n_a = tn.node_index(&"A".to_string()).unwrap();
    let n_b = tn.node_index(&"B".to_string()).unwrap();
    let n_c = tn.node_index(&"C".to_string()).unwrap();
    let n_d = tn.node_index(&"D".to_string()).unwrap();
    tn.connect(n_a, &bond_ab, n_b, &bond_ab)?;
    tn.connect(n_b, &bond_bc, n_c, &bond_bc)?;
    tn.connect(n_b, &bond_bd, n_d, &bond_bd)?;

    // Restructure: fuse C+D into one node (B is internal, pulled in)
    let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    target
        .add_node("A".to_string(), HashSet::from([site_a]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("CD".to_string(), HashSet::from([site_c, site_d]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"A".to_string(), &"CD".to_string())
        .map_err(anyhow::Error::msg)?;

    let result = tn.restructure_to(&target, &RestructureOptions::default())?;
    assert_eq!(result.node_count(), 2);
    let dense_expected = tn.contract_to_tensor()?;
    let dense_actual = result.contract_to_tensor()?;
    assert!(dense_actual.distance(&dense_expected).unwrap() < 1e-12);
    Ok(())
}
