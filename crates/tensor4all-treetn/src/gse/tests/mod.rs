use super::*;

use std::collections::HashMap;

fn basis_vector(dim: usize, selected: usize) -> Vec<f64> {
    let mut values = vec![0.0; dim];
    values[selected] = 1.0;
    values
}

fn product_chain_state(left_dim: usize, right_dim: usize) -> TreeTN<IdxTensor, &'static str> {
    let left = DynIndex::new_dyn(left_dim);
    let right = DynIndex::new_dyn(right_dim);
    let bond = DynIndex::new_dyn(1);
    let left_tensor =
        IdxTensor::from_dense(vec![left, bond.clone()], basis_vector(left_dim, 0)).unwrap();
    let right_tensor =
        IdxTensor::from_dense(vec![bond.clone(), right], basis_vector(right_dim, 0)).unwrap();
    let mut state = TreeTN::<IdxTensor, &'static str>::new();
    let left_node = state.add_tensor("site0", left_tensor).unwrap();
    let right_node = state.add_tensor("site1", right_tensor).unwrap();
    state.connect(left_node, &bond, right_node, &bond).unwrap();
    state
}

fn product_chain3_state(tail_bond_dim: usize) -> TreeTN<IdxTensor, &'static str> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(1);
    let b12 = DynIndex::new_dyn(tail_bond_dim);
    let t0 = IdxTensor::from_dense(vec![s0, b01.clone()], vec![1.0, 0.0]).unwrap();
    let mut t1_data = vec![0.0; 2 * tail_bond_dim];
    t1_data[0] = 1.0;
    let t1 = IdxTensor::from_dense(vec![b01.clone(), s1, b12.clone()], t1_data).unwrap();
    let mut t2_data = vec![0.0; tail_bond_dim * 2];
    t2_data[0] = 1.0;
    let t2 = IdxTensor::from_dense(vec![b12.clone(), s2], t2_data).unwrap();

    let mut state = TreeTN::<IdxTensor, &'static str>::new();
    let n0 = state.add_tensor("site0", t0).unwrap();
    let n1 = state.add_tensor("site1", t1).unwrap();
    let n2 = state.add_tensor("site2", t2).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();
    state.connect(n1, &b12, n2, &b12).unwrap();
    state
}

#[test]
fn gse_option_builders_set_reference_knobs() {
    let policy = SvdTruncationPolicy::new(1.0e-8);
    let options = GseOptions::<usize>::default()
        .with_reference_max_bond_dim(7)
        .with_reference_svd_policy(policy)
        .with_normalize_references(false)
        .with_expand_before_first_sweep(false);

    assert_eq!(options.reference_max_bond_dim, Some(7));
    assert_eq!(options.reference_svd_policy, Some(policy));
    assert!(!options.normalize_references);
    assert!(!options.expand_before_first_sweep);
}

#[test]
fn square_site_mapping_errors_convert_to_gse_errors() {
    let err = GseError::from(SquareSiteMappingError::UnsupportedStateSiteCount {
        node: "site0".to_string(),
        count: 2,
    });
    assert!(matches!(
        err,
        GseError::UnsupportedStateSiteCount { node, count }
            if node == "site0" && count == 2
    ));

    let err = GseError::from(SquareSiteMappingError::UnsupportedMultipleSiteMappings {
        node: "site1".to_string(),
        role: "input",
        count: 2,
    });
    assert!(matches!(
        err,
        GseError::UnsupportedMultipleSiteMappings { node, role, count }
            if node == "site1" && role == "input" && count == 2
    ));

    let err = GseError::from(SquareSiteMappingError::MissingMapping {
        node: "site2".to_string(),
        role: "output",
    });
    assert!(matches!(
        err,
        GseError::MissingMapping { node, role }
            if node == "site2" && role == "output"
    ));

    let err = GseError::from(SquareSiteMappingError::InvalidMapping {
        node: "site3".to_string(),
        reason: "bad dimension".to_string(),
    });
    assert!(matches!(
        err,
        GseError::InvalidMapping { node, reason }
            if node == "site3" && reason == "bad dimension"
    ));
}

#[test]
fn identity_on_index_pairs_handles_empty_products_and_mismatches() {
    let scalar = identity_on_index_pairs(&[], &[]).unwrap();
    assert!(scalar.indices().is_empty());
    assert_eq!(scalar.to_vec::<f64>().unwrap(), vec![1.0]);

    let left0 = DynIndex::new_dyn(2);
    let left1 = DynIndex::new_dyn(3);
    let right0 = DynIndex::new_dyn(2);
    let right1 = DynIndex::new_dyn(3);
    let identity = identity_on_index_pairs(
        &[left0.clone(), left1.clone()],
        &[right0.clone(), right1.clone()],
    )
    .unwrap();
    assert_eq!(identity.indices(), &[left0.clone(), right0, left1, right1]);
    assert_eq!(identity.sum().unwrap().real(), 6.0);

    let err = identity_on_index_pairs(&[left0], &[]).unwrap_err();
    assert!(matches!(
        err,
        GseError::Algorithm { context, .. }
            if context == "GSE failed to build identity on q-space"
    ));
}

#[test]
fn adjoint_by_index_groups_swaps_groups_and_rejects_mismatched_lengths() {
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![left.clone(), right.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let adjoint = adjoint_by_index_groups(
        &tensor,
        std::slice::from_ref(&left),
        std::slice::from_ref(&right),
    )
    .unwrap();
    assert_eq!(adjoint.indices(), &[right.clone(), left.clone()]);
    assert_eq!(adjoint.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    let err = adjoint_by_index_groups(&tensor, &[left], &[]).unwrap_err();
    assert!(matches!(
        err,
        GseError::Algorithm { context, .. }
            if context == "GSE failed to adjoint grouped tensor"
    ));
}

#[test]
fn validate_reference_rejects_site_dimension_mismatch() {
    let target = product_chain_state(2, 2);
    let reference = product_chain_state(3, 2);

    let err = validate_reference(&target, &reference).unwrap_err();
    assert!(matches!(
        err,
        GseError::InvalidMapping { reason, .. }
            if reason == "reference site dimension does not match target"
    ));
}

#[test]
fn single_site_index_reports_missing_node_and_multi_site_nodes() {
    let target = product_chain_state(2, 2);

    let err = single_site_index(&target, &"missing").unwrap_err();
    assert!(matches!(
        err,
        GseError::UnsupportedStateSiteCount { node, count }
            if node == "\"missing\"" && count == 0
    ));

    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense(vec![s0, s1], vec![0.0; 4]).unwrap();
    let multi =
        TreeTN::<IdxTensor, &'static str>::from_tensors(vec![tensor], vec!["site0"]).unwrap();
    let err = single_site_index(&multi, &"site0").unwrap_err();
    assert!(matches!(
        err,
        GseError::UnsupportedStateSiteCount { node, count }
            if node == "\"site0\"" && count == 2
    ));
}

#[test]
fn map_q_indices_rejects_parent_bond_and_child_bond_dimension_mismatch() {
    let target = product_chain3_state(2);
    let reference = product_chain3_state(3);
    let site1 = single_site_index(&target, &"site1").unwrap();
    let parent_edge = target.edge_between(&"site0", &"site1").unwrap();
    let parent_bond = target.bond_index(parent_edge).unwrap().clone();
    let child_edge = target.edge_between(&"site1", &"site2").unwrap();
    let child_bond = target.bond_index(child_edge).unwrap().clone();

    let err = map_q_indices(
        &target,
        &reference,
        &"site1",
        &"site0",
        &[site1.clone(), parent_bond],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        GseError::Algorithm { context, .. }
            if context == "GSE parent bond cannot appear in q index map"
    ));

    let err = map_q_indices(
        &target,
        &reference,
        &"site1",
        &"site0",
        &[site1, child_bond],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        GseError::InvalidMapping { reason, .. }
            if reason == "reference child-side bond dimension does not match target"
    ));
}

#[test]
fn move_center_to_region_full_rank_handles_errors_and_path_moves() {
    let mut uncached = product_chain_state(2, 2);
    let err = move_center_to_region_full_rank(&mut uncached, &[]).unwrap_err();
    assert!(err.to_string().contains("empty region"));

    let err = move_center_to_region_full_rank(&mut uncached, &["site0"]).unwrap_err();
    assert!(err.to_string().contains("not canonicalized"));

    let mut state = product_chain_state(2, 2)
        .canonicalize(["site0"], CanonicalizationOptions::forced())
        .unwrap();
    move_center_to_region_full_rank(&mut state, &["site0"]).unwrap();
    assert!(state.canonical_region().contains(&"site0"));

    move_center_to_region_full_rank(&mut state, &["site1"]).unwrap();
    assert!(state.canonical_region().contains(&"site1"));
}

#[test]
fn gse_tdvp_options_default_matches_component_defaults() {
    let options = GseTdvpOptions::<usize>::default();
    let gse = GseOptions::<usize>::default();
    let tdvp = TdvpOptions::default();

    assert_eq!(options.gse.krylov_dim, gse.krylov_dim);
    assert_eq!(options.gse.density_weight_cutoff, gse.density_weight_cutoff);
    assert_eq!(options.gse.hermitian_tol, gse.hermitian_tol);
    assert_eq!(options.gse.normalize_references, gse.normalize_references);
    assert_eq!(options.tdvp.nsweeps, tdvp.nsweeps);
    assert_eq!(options.tdvp.max_bond_dim, tdvp.max_bond_dim);
}

#[test]
fn validate_options_rejects_zero_reference_max_bond_dim() {
    let options = GseOptions::<usize>::default().with_reference_max_bond_dim(0);
    let err = validate_options(&options).unwrap_err();

    assert!(matches!(
        err,
        GseError::InvalidOption { option, reason }
            if option == "reference_max_bond_dim" && reason.contains("greater than zero")
    ));
}

/// A GSE operator that is never contracted: `build_references` fails while
/// validating the requested contraction center, before the MPO is touched.
fn empty_operator() -> LinearOperator<IdxTensor, &'static str> {
    LinearOperator::new(TreeTN::new(), HashMap::new(), HashMap::new())
}

#[test]
fn build_references_reports_operator_application_failure() {
    let init = product_chain_state(2, 2);
    let options = GseOptions::<&'static str> {
        krylov_dim: 1,
        reference_apply: ApplyOptions {
            center: Some("missing"),
            ..ApplyOptions::zipup()
        },
        ..GseOptions::<&'static str>::default()
    };

    let err = build_references(&empty_operator(), &init, &"site0", &options).unwrap_err();

    assert!(matches!(
        err,
        GseError::Algorithm { context, .. }
            if context == "GSE failed to apply operator while building references"
    ));
}

#[test]
fn move_center_to_region_full_rank_sweeps_along_the_path() {
    let mut state = product_chain3_state(2)
        .canonicalize(["site0"], CanonicalizationOptions::forced())
        .unwrap();
    let dense_before = state.to_dense().unwrap();

    move_center_to_region_full_rank(&mut state, &["site2"]).unwrap();

    assert!(state.canonical_region().contains(&"site2"));
    assert!(state.is_canonicalized());
    state.verify_internal_consistency().unwrap();
    let dense_after = state.to_dense().unwrap();
    assert!(dense_before.distance(&dense_after).unwrap() < 1.0e-12);
}
