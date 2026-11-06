use tensor4all_core::{DynId, DynIndex, IdxTensor, TagSet};

use super::*;

fn one_node_state(node: usize, indices: Vec<DynIndex>) -> TreeTN<IdxTensor, usize> {
    let len = indices.iter().map(|idx| idx.dim()).product();
    let tensor = IdxTensor::from_dense(indices, vec![1.0_f64; len]).unwrap();
    TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![node]).unwrap()
}

#[test]
fn index_sets_match_distinguishes_same_id_prime_pair() {
    let updater = SquareLinsolveUpdater::<IdxTensor, usize>::new(
        TreeTN::new(),
        TreeTN::new(),
        LinsolveOptions::default(),
    );
    let i = DynIndex::new_dyn(2);
    let i_prime = i.prime();

    assert!(updater.index_sets_match(std::slice::from_ref(&i), std::slice::from_ref(&i)));
    assert!(!updater.index_sets_match(&[i], &[i_prime]));
}

#[test]
fn verify_counts_only_full_index_overlap() {
    let id = DynId(700);
    let state_site = DynIndex::new_with_tags(id, 2, TagSet::from_str("state").unwrap());
    let op_site = DynIndex::new_with_tags(id, 2, TagSet::from_str("operator").unwrap());
    let state = one_node_state(0, vec![state_site.clone()]);
    let rhs = one_node_state(0, vec![state_site]);
    let operator = one_node_state(0, vec![op_site]);

    let updater =
        SquareLinsolveUpdater::<IdxTensor, usize>::new(operator, rhs, LinsolveOptions::default());
    let report = updater.verify(&state).unwrap();

    assert_eq!(report.node_details[0].common_index_count, 0);
    assert_eq!(report.warnings.len(), 1);
}

#[test]
fn validate_mpo_external_indices_keeps_same_id_primed_external_leg() {
    let contracted = DynIndex::new_dyn(2);
    let external = contracted.prime();
    let state = one_node_state(0, vec![external.clone(), contracted.clone()]);
    let rhs = one_node_state(0, vec![external.clone(), contracted.clone()]);

    let op_in = DynIndex::new_dyn(2);
    let op_out = DynIndex::new_dyn(2);
    let operator = one_node_state(0, vec![op_out.clone(), op_in.clone()]);

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        0,
        IndexMapping {
            true_index: contracted.clone(),
            internal_index: op_in,
        },
    );
    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        0,
        IndexMapping {
            true_index: contracted,
            internal_index: op_out,
        },
    );

    let mut updater = SquareLinsolveUpdater::<IdxTensor, usize>::with_index_mappings(
        operator,
        input_mapping,
        output_mapping,
        rhs,
        LinsolveOptions::default(),
    );

    updater.validate_mpo_external_indices(&state).unwrap();
}

#[test]
fn local_gmres_options_match_krylovkit_restart_convention() {
    let options = LinsolveOptions::default()
        .with_gmres_restart_dim(30)
        .with_gmres_max_restarts(10)
        .with_gmres_tol(1.0e-8);

    let gmres_options = local_gmres_options(&options).unwrap();

    assert_eq!(gmres_options.max_iter, 30);
    assert_eq!(gmres_options.max_restarts, 10);
    assert_eq!(gmres_options.rtol, 1.0e-8);
}

#[test]
fn local_gmres_options_does_not_convert_maxiter_to_total_step_limit() {
    let options = LinsolveOptions::default()
        .with_gmres_restart_dim(30)
        .with_gmres_max_restarts(100);

    let gmres_options = local_gmres_options(&options).unwrap();

    assert_eq!(gmres_options.max_iter, 30);
    assert_eq!(gmres_options.max_restarts, 100);
}

#[test]
fn local_gmres_options_reject_zero_iteration_parameters() {
    assert!(local_gmres_options(&LinsolveOptions::default().with_gmres_restart_dim(0)).is_err());
    assert!(local_gmres_options(&LinsolveOptions::default().with_gmres_max_restarts(0)).is_err());
}

#[test]
fn one_site_update_rejects_malformed_transport_regions() {
    let mut updater = SquareLinsolveUpdater::<IdxTensor, usize>::new(
        TreeTN::new(),
        TreeTN::new(),
        LinsolveOptions::default().with_nsite(1),
    );
    for (nodes, new_center) in [
        (vec![], 0),
        (vec![0, 1, 2], 2),
        (vec![0, 1], 0),
        (vec![0], 1),
    ] {
        let error = updater
            .update(
                TreeTN::new(),
                &LocalUpdateStep { nodes, new_center },
                &TreeTN::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("oriented transport edge"));
    }
}

#[test]
fn one_site_update_requires_source_at_orthogonality_center() {
    let bond = DynIndex::new_dyn(2);
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(vec![s0, bond.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let t1 = IdxTensor::from_dense(vec![bond, s1], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();
    let state = TreeTN::<IdxTensor, usize>::from_tensors(vec![t0, t1], vec![0, 1])
        .unwrap()
        .canonicalize([1], crate::CanonicalizationOptions::default())
        .unwrap();
    let mut updater = SquareLinsolveUpdater::<IdxTensor, usize>::new(
        TreeTN::new(),
        TreeTN::new(),
        LinsolveOptions::default().with_nsite(1),
    );
    let error = updater
        .update(
            state.clone(),
            &LocalUpdateStep {
                nodes: vec![0, 1],
                new_center: 1,
            },
            &state,
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("orthogonality center"),
        "{error}"
    );
}
