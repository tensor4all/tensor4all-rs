//! Tests for `restructure_to` on chain topologies: fuse, split, swap, and
//! their mixed cases.

use super::*;

type FourSiteChainCase = (
    TreeTN<IdxTensor, String>,
    DynIndex,
    DynIndex,
    DynIndex,
    DynIndex,
);

fn two_node_chain() -> anyhow::Result<(TreeTN<IdxTensor, String>, DynIndex, DynIndex)> {
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);
    let t0 = IdxTensor::from_dense(vec![left.clone(), bond.clone()], vec![1.0, 0.0])?;
    let t1 = IdxTensor::from_dense(vec![bond, right.clone()], vec![1.0, 0.0])?;
    let treetn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![t0, t1],
        vec!["A".to_string(), "B".to_string()],
    )?;
    Ok((treetn, left, right))
}

fn two_node_groups_of_two() -> anyhow::Result<FourSiteChainCase> {
    let x0 = DynIndex::new_dyn(2);
    let x1 = DynIndex::new_dyn(2);
    let y0 = DynIndex::new_dyn(2);
    let y1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let left_tensor = IdxTensor::from_dense(
        vec![x0.clone(), x1.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    )?;
    let right_tensor = IdxTensor::from_dense(
        vec![bond, y0.clone(), y1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    )?;
    let treetn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![left_tensor, right_tensor],
        vec!["Left".to_string(), "Right".to_string()],
    )?;
    Ok((treetn, x0, x1, y0, y1))
}

fn three_node_chain_for_swap() -> anyhow::Result<FourSiteChainCase> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let s3 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(
        vec![s0.clone(), s1.clone(), b01.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
    )?;
    let t1 = IdxTensor::from_dense(
        vec![b01.clone(), s2.clone(), b12.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )?;
    let t2 = IdxTensor::from_dense(vec![b12, s3.clone()], vec![1.0, 2.0, 3.0, 4.0])?;
    let treetn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![t0, t1, t2],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )?;
    Ok((treetn, s0, s1, s2, s3))
}

fn four_node_interleaved_chain() -> anyhow::Result<FourSiteChainCase> {
    let x0 = DynIndex::new_dyn(2);
    let x1 = DynIndex::new_dyn(2);
    let y0 = DynIndex::new_dyn(2);
    let y1 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);
    let b23 = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(vec![x0.clone(), b01.clone()], vec![1.0, 0.0, 0.0, 1.0])?;
    let t1 = IdxTensor::from_dense(
        vec![b01.clone(), x1.clone(), b12.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )?;
    let t2 = IdxTensor::from_dense(
        vec![b12.clone(), y0.clone(), b23.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )?;
    let t3 = IdxTensor::from_dense(vec![b23, y1.clone()], vec![1.0, 0.0, 0.0, 1.0])?;
    let treetn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![t0, t1, t2, t3],
        vec![
            "0".to_string(),
            "1".to_string(),
            "2".to_string(),
            "3".to_string(),
        ],
    )?;
    Ok((treetn, x0, x1, y0, y1))
}

#[test]
fn test_restructure_to_fuse_only_matches_target_structure() -> anyhow::Result<()> {
    let (treetn, left, right) = two_node_chain()?;

    let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    target
        .add_node("AB".to_string(), HashSet::from([left, right]))
        .map_err(anyhow::Error::msg)?;

    let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
    let dense_expected = treetn.contract_to_tensor()?;
    let dense_actual = result.contract_to_tensor()?;

    assert_eq!(result.node_count(), 1);
    assert_eq!(result.site_index_network().node_count(), 1);
    assert!(dense_actual.distance(&dense_expected).unwrap() < 1e-12);

    Ok(())
}

#[test]
fn test_restructure_to_absorbs_site_free_dangling_leaf() -> anyhow::Result<()> {
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![left.clone(), b01.clone()], vec![1.0, 2.0, 3.0, 4.0])?;
    let t1 = IdxTensor::from_dense(
        vec![b01.clone(), right.clone(), b12.clone()],
        (1..=8).map(|value| value as f64 / 3.0).collect(),
    )?;
    let t2 = IdxTensor::from_dense(vec![b12], vec![0.5, -1.25])?;
    let treetn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![t0, t1, t2],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )?;

    let before = treetn.to_dense()?;
    let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    target.add_node("A".to_string(), HashSet::from([left]))?;
    target.add_node("B".to_string(), HashSet::from([right]))?;
    target.add_edge(&"A".to_string(), &"B".to_string())?;

    let result = treetn.restructure_to(&target, &RestructureOptions::default())?;

    assert_eq!(result.node_count(), 2);
    assert_eq!(result.edge_count(), 1);
    assert!(result
        .site_index_network()
        .share_equivalent_site_index_network(&target));
    assert!(result.to_dense()?.sub(&before)?.maxabs()? < 1.0e-12);

    Ok(())
}

#[test]
fn test_restructure_to_absorbs_site_free_internal_node() -> anyhow::Result<()> {
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![left.clone(), b01.clone()], vec![1.0, 2.0, 3.0, 4.0])?;
    let t1 = IdxTensor::from_dense(vec![b01.clone(), b12.clone()], vec![0.5, -1.0, 1.25, 0.75])?;
    let t2 = IdxTensor::from_dense(vec![b12, right.clone()], vec![2.0, -0.5, 1.5, 3.0])?;
    let treetn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![t0, t1, t2],
        vec!["A".to_string(), "M".to_string(), "B".to_string()],
    )?;

    let before = treetn.to_dense()?;
    let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    target.add_node("A".to_string(), HashSet::from([left]))?;
    target.add_node("B".to_string(), HashSet::from([right]))?;
    target.add_edge(&"A".to_string(), &"B".to_string())?;

    let result = treetn.restructure_to(&target, &RestructureOptions::default())?;

    assert_eq!(result.node_count(), 2);
    assert_eq!(result.edge_count(), 1);
    assert!(result
        .site_index_network()
        .share_equivalent_site_index_network(&target));
    assert!(result.to_dense()?.sub(&before)?.maxabs()? < 1.0e-12);

    Ok(())
}

#[test]
fn test_restructure_to_split_only_matches_target_structure() -> anyhow::Result<()> {
    let (treetn, left, right) = two_node_chain()?;

    let mut fused_target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    fused_target
        .add_node(
            "AB".to_string(),
            HashSet::from([left.clone(), right.clone()]),
        )
        .map_err(anyhow::Error::msg)?;
    let fused = treetn.restructure_to(&fused_target, &RestructureOptions::default())?;

    let mut split_target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    split_target
        .add_node("Left".to_string(), HashSet::from([left]))
        .map_err(anyhow::Error::msg)?;
    split_target
        .add_node("Right".to_string(), HashSet::from([right]))
        .map_err(anyhow::Error::msg)?;
    split_target
        .add_edge(&"Left".to_string(), &"Right".to_string())
        .map_err(anyhow::Error::msg)?;

    let result = fused.restructure_to(&split_target, &RestructureOptions::default())?;
    let dense_expected = fused.contract_to_tensor()?;
    let dense_actual = result.contract_to_tensor()?;

    assert_eq!(result.node_count(), 2);
    assert!(dense_actual.distance(&dense_expected).unwrap() < 1e-12);

    Ok(())
}

#[test]
fn test_restructure_to_swap_only_matches_target_structure() -> anyhow::Result<()> {
    let (treetn, s0, s1, s2, s3) = three_node_chain_for_swap()?;

    let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    target
        .add_node("X".to_string(), HashSet::from([s0.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("Y".to_string(), HashSet::from([s1.clone(), s2.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("Z".to_string(), HashSet::from([s3.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"X".to_string(), &"Y".to_string())
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"Y".to_string(), &"Z".to_string())
        .map_err(anyhow::Error::msg)?;

    let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
    let dense_expected = treetn.contract_to_tensor()?;
    let dense_actual = result.contract_to_tensor()?;

    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&s0)
            .map(|name| name.as_str()),
        Some("X")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&s1)
            .map(|name| name.as_str()),
        Some("Y")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&s2)
            .map(|name| name.as_str()),
        Some("Y")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&s3)
            .map(|name| name.as_str()),
        Some("Z")
    );
    assert!(dense_actual.distance(&dense_expected).unwrap() < 1e-10);

    Ok(())
}

#[test]
fn test_restructure_to_split_then_fuse_mixed_case() -> anyhow::Result<()> {
    let (treetn, x0, x1, y0, y1) = two_node_groups_of_two()?;

    let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    target
        .add_node("X".to_string(), HashSet::from([x0.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("Y".to_string(), HashSet::from([x1.clone(), y0.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("Z".to_string(), HashSet::from([y1.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"X".to_string(), &"Y".to_string())
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"Y".to_string(), &"Z".to_string())
        .map_err(anyhow::Error::msg)?;

    let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
    let dense_expected = treetn.contract_to_tensor()?;
    let dense_actual = result.contract_to_tensor()?;

    assert_eq!(result.node_count(), 3);
    assert_eq!(result.edge_count(), 2);
    assert!(result
        .site_index_network()
        .share_equivalent_site_index_network(&target));
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&x0)
            .map(|name| name.as_str()),
        Some("X")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&x1)
            .map(|name| name.as_str()),
        Some("Y")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&y0)
            .map(|name| name.as_str()),
        Some("Y")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&y1)
            .map(|name| name.as_str()),
        Some("Z")
    );
    assert!(dense_actual.distance(&dense_expected).unwrap() < 1e-10);

    Ok(())
}

#[test]
fn test_restructure_to_swap_then_fuse_mixed_case() -> anyhow::Result<()> {
    let (treetn, x0, x1, y0, y1) = four_node_interleaved_chain()?;

    let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    target
        .add_node("X".to_string(), HashSet::from([x0.clone(), y0.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("Y".to_string(), HashSet::from([x1.clone(), y1.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"X".to_string(), &"Y".to_string())
        .map_err(anyhow::Error::msg)?;

    let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
    let dense_expected = treetn.contract_to_tensor()?;
    let dense_actual = result.contract_to_tensor()?;

    assert_eq!(result.node_count(), 2);
    assert_eq!(result.edge_count(), 1);
    assert!(result
        .site_index_network()
        .share_equivalent_site_index_network(&target));
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&x0)
            .map(|name| name.as_str()),
        Some("X")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&y0)
            .map(|name| name.as_str()),
        Some("X")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&x1)
            .map(|name| name.as_str()),
        Some("Y")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&y1)
            .map(|name| name.as_str()),
        Some("Y")
    );
    assert!(dense_actual.distance(&dense_expected).unwrap() < 1e-10);

    Ok(())
}

#[test]
fn test_restructure_to_two_node_swap_only_cross_pairing() -> anyhow::Result<()> {
    let (treetn, x0, x1, y0, y1) = two_node_groups_of_two()?;

    let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    target
        .add_node("X".to_string(), HashSet::from([x0.clone(), y0.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("Y".to_string(), HashSet::from([x1.clone(), y1.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"X".to_string(), &"Y".to_string())
        .map_err(anyhow::Error::msg)?;

    let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
    let dense_expected = treetn.contract_to_tensor()?;
    let dense_actual = result.contract_to_tensor()?;

    assert_eq!(result.node_count(), 2);
    assert_eq!(result.edge_count(), 1);
    assert!(result
        .site_index_network()
        .share_equivalent_site_index_network(&target));
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&x0)
            .map(|name| name.as_str()),
        Some("X")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&y0)
            .map(|name| name.as_str()),
        Some("X")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&x1)
            .map(|name| name.as_str()),
        Some("Y")
    );
    assert_eq!(
        result
            .site_index_network()
            .find_node_by_index(&y1)
            .map(|name| name.as_str()),
        Some("Y")
    );
    assert!(dense_actual.distance(&dense_expected).unwrap() < 1e-10);

    Ok(())
}

#[test]
fn test_restructure_to_site_permutation_preserves_target_path() -> anyhow::Result<()> {
    let (treetn, x0, x1, y0, y1) = four_node_interleaved_chain()?;

    let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    target
        .add_node("0".to_string(), HashSet::from([x1.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("1".to_string(), HashSet::from([x0.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("2".to_string(), HashSet::from([y1.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_node("3".to_string(), HashSet::from([y0.clone()]))
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"0".to_string(), &"1".to_string())
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"1".to_string(), &"2".to_string())
        .map_err(anyhow::Error::msg)?;
    target
        .add_edge(&"2".to_string(), &"3".to_string())
        .map_err(anyhow::Error::msg)?;

    let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
    let dense_expected = treetn.contract_to_tensor()?;
    let dense_actual = result.contract_to_tensor()?;

    assert!(result
        .site_index_network()
        .share_equivalent_site_index_network(&target));
    assert!(dense_actual.distance(&dense_expected).unwrap() < 1e-10);

    Ok(())
}
