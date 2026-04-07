use super::*;
use crate::random::{random_treetn, LinkSpace};
use crate::SiteIndexNetwork;
use std::collections::{HashMap, HashSet};
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::TensorDynLen;

type DynIndex = Index<DynId, TagSet>;

fn make_index(dim: usize) -> DynIndex {
    Index::new_dyn(dim)
}

fn build_identity_operator(sites: &[(String, DynIndex)]) -> LinearOperator<TensorDynLen, String> {
    let mut mpo = TreeTN::new();
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for (name, true_index) in sites {
        let internal_input = make_index(true_index.dim());
        let internal_output = make_index(true_index.dim());
        let tensor = TensorDynLen::from_dense(
            vec![internal_output.clone(), internal_input.clone()],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();

        mpo.add_tensor(name.clone(), tensor).unwrap();
        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: internal_input,
            },
        );
        output_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: internal_output,
            },
        );
    }

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn build_chain_state() -> (TreeTN<TensorDynLen, String>, Vec<(String, DynIndex)>) {
    let s0 = make_index(2);
    let s1 = make_index(2);
    let s2 = make_index(2);
    let s3 = make_index(2);
    let b01 = make_index(2);
    let b12 = make_index(2);
    let b23 = make_index(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 =
        TensorDynLen::from_dense(vec![b01.clone(), s1.clone(), b12.clone()], vec![1.0; 8]).unwrap();
    let t2 =
        TensorDynLen::from_dense(vec![b12.clone(), s2.clone(), b23.clone()], vec![1.0; 8]).unwrap();
    let t3 = TensorDynLen::from_dense(vec![b23.clone(), s3.clone()], vec![1.0; 4]).unwrap();

    let state = TreeTN::from_tensors(
        vec![t0, t1, t2, t3],
        vec![
            "site0".to_string(),
            "site1".to_string(),
            "site2".to_string(),
            "site3".to_string(),
        ],
    )
    .unwrap();

    (
        state,
        vec![
            ("site0".to_string(), s0),
            ("site1".to_string(), s1),
            ("site2".to_string(), s2),
            ("site3".to_string(), s3),
        ],
    )
}

fn build_tree_state() -> (TreeTN<TensorDynLen, String>, Vec<(String, DynIndex)>) {
    let s_a = make_index(2);
    let s_b = make_index(2);
    let s_c = make_index(2);
    let s_d = make_index(2);
    let s_e = make_index(2);
    let b_ab = make_index(2);
    let b_bc = make_index(2);
    let b_bd = make_index(2);
    let b_be = make_index(2);

    let t_a = TensorDynLen::from_dense(vec![s_a.clone(), b_ab.clone()], vec![1.0; 4]).unwrap();
    let t_b = TensorDynLen::from_dense(
        vec![b_ab.clone(), s_b.clone(), b_bc.clone(), b_bd.clone()],
        vec![1.0; 16],
    )
    .unwrap();
    let t_c = TensorDynLen::from_dense(vec![b_bc.clone(), s_c.clone()], vec![1.0; 4]).unwrap();
    let t_d = TensorDynLen::from_dense(vec![b_bd.clone(), s_d.clone(), b_be.clone()], vec![1.0; 8])
        .unwrap();
    let t_e = TensorDynLen::from_dense(vec![b_be.clone(), s_e.clone()], vec![1.0; 4]).unwrap();

    let state = TreeTN::from_tensors(
        vec![t_a, t_b, t_c, t_d, t_e],
        vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
            "E".to_string(),
        ],
    )
    .unwrap();

    (
        state,
        vec![
            ("A".to_string(), s_a),
            ("B".to_string(), s_b),
            ("C".to_string(), s_c),
            ("D".to_string(), s_d),
            ("E".to_string(), s_e),
        ],
    )
}

fn assert_identity_application(
    state: &TreeTN<TensorDynLen, String>,
    operator: &LinearOperator<TensorDynLen, String>,
) {
    use crate::operator::apply_linear_operator;

    let result = apply_linear_operator(operator, state, ApplyOptions::default()).unwrap();
    let result_dense = result.to_dense().unwrap();
    let state_dense = state.to_dense().unwrap();
    assert!((&result_dense - &state_dense).maxabs() < 1e-10);
}

#[test]
fn test_apply_options_builder() {
    let opts = ApplyOptions::zipup().with_max_rank(50).with_rtol(1e-10);
    assert_eq!(opts.method, ContractionMethod::Zipup);
    assert_eq!(opts.max_rank, Some(50));
    assert_eq!(opts.rtol, Some(1e-10));
}

#[test]
fn test_linear_operator_tensor_index() {
    // Create a simple LinearOperator
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    net.add_node(
        "N0".to_string(),
        [s0_in.clone(), s0_out.clone()]
            .into_iter()
            .collect::<HashSet<_>>(),
    )
    .unwrap();

    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let mpo = random_treetn::<f64, _, _>(&mut rng, &net, link_space);

    let true_s0 = make_index(2);
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    input_mapping.insert(
        "N0".to_string(),
        IndexMapping {
            true_index: true_s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    output_mapping.insert(
        "N0".to_string(),
        IndexMapping {
            true_index: true_s0.clone(),
            internal_index: s0_out.clone(),
        },
    );

    let lin_op = LinearOperator::new(mpo, input_mapping, output_mapping);

    // Test external_indices
    let ext_indices = lin_op.external_indices();
    assert_eq!(ext_indices.len(), 2);

    // Test replaceind
    let new_idx = make_index(2);
    let replaced = lin_op.replaceind(&true_s0, &new_idx).unwrap();
    assert!(replaced.get_input_mapping(&"N0".to_string()).is_some());
}

#[test]
fn test_arc_linear_operator_cow() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s0 = make_index(2);
    net.add_node(
        "N0".to_string(),
        [s0.clone()].into_iter().collect::<HashSet<_>>(),
    )
    .unwrap();

    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let mpo = random_treetn::<f64, _, _>(&mut rng, &net, link_space);

    let arc_op = ArcLinearOperator::new(mpo, HashMap::new(), HashMap::new());

    // Clone should share the Arc
    let arc_op2 = arc_op.clone();
    assert!(Arc::ptr_eq(&arc_op.mpo, &arc_op2.mpo));

    // Mutating one should not affect the other (CoW)
    let mut arc_op3 = arc_op.clone();
    let _mpo_mut = arc_op3.mpo_mut();
    // After make_mut, the Arcs should be different if there were other refs
    // (In this case, arc_op still holds a reference)
    assert!(!Arc::ptr_eq(&arc_op.mpo, &arc_op3.mpo));
}

#[test]
fn test_apply_linear_operator_full_coverage() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    // Create a 2-site state
    let mut state = TreeTN::<TensorDynLen, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let b01 = make_index(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 = TensorDynLen::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 4]).unwrap();

    let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();

    // Create identity operator covering both sites
    let mut mpo = TreeTN::<TensorDynLen, String>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0]; // Identity matrix
    let t0_mpo = TensorDynLen::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        TensorDynLen::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data)
            .unwrap();

    let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
    let n1_mpo = mpo.add_tensor("site1".to_string(), t1_mpo).unwrap();
    mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    input_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        "site1".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_in.clone(),
        },
    );
    output_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        "site1".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_out.clone(),
        },
    );

    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

    // Test apply with default options
    let result = apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    assert_eq!(result.node_count(), 2);

    // Test apply with different methods
    let result_fit = apply_linear_operator(&operator, &state, ApplyOptions::fit()).unwrap();
    assert_eq!(result_fit.node_count(), 2);

    let result_naive = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    assert_eq!(result_naive.node_count(), 2);
}

#[test]
fn test_apply_linear_operator_partial() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    // Create a 3-site state
    let mut state = TreeTN::<TensorDynLen, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let s2 = make_index(2);
    let b01 = make_index(2);
    let b12 = make_index(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 =
        TensorDynLen::from_dense(vec![b01.clone(), s1.clone(), b12.clone()], vec![1.0; 8]).unwrap();
    let t2 = TensorDynLen::from_dense(vec![b12.clone(), s2.clone()], vec![1.0; 4]).unwrap();

    let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
    let n2 = state.add_tensor("site2".to_string(), t2).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();
    state.connect(n1, &b12, n2, &b12).unwrap();

    // Create operator covering only site0 and site1 (partial)
    let mut mpo = TreeTN::<TensorDynLen, String>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0];
    let t0_mpo = TensorDynLen::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        TensorDynLen::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data)
            .unwrap();

    let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
    let n1_mpo = mpo.add_tensor("site1".to_string(), t1_mpo).unwrap();
    mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    input_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        "site1".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_in.clone(),
        },
    );
    output_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        "site1".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_out.clone(),
        },
    );

    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

    // Test apply with partial operator (should extend with identity on site2)
    let result = apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    assert_eq!(result.node_count(), 3);
}

#[test]
fn test_apply_linear_operator_non_contiguous_chain() {
    let (state, sites) = build_chain_state();
    let operator = build_identity_operator(&[sites[0].clone(), sites[2].clone()]);

    assert_identity_application(&state, &operator);
}

#[test]
fn test_apply_linear_operator_non_contiguous_tree() {
    let (state, sites) = build_tree_state();
    let operator = build_identity_operator(&[sites[0].clone(), sites[4].clone()]);

    assert_identity_application(&state, &operator);
}

#[test]
fn test_apply_linear_operator_error_cases() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    // Create a 2-site state
    let mut state = TreeTN::<TensorDynLen, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let b01 = make_index(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 = TensorDynLen::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 4]).unwrap();

    let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
    state.connect(n0, &b01, n1, &b01).unwrap();

    // Create operator with extra node not in state (should error)
    let mut mpo = TreeTN::<TensorDynLen, String>::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    let s2_in = make_index(2); // site2 doesn't exist in state
    let s2_out = make_index(2);
    let b_mpo = make_index(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0];
    let t0_mpo = TensorDynLen::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t2_mpo =
        TensorDynLen::from_dense(vec![b_mpo.clone(), s2_out.clone(), s2_in.clone()], id_data)
            .unwrap();

    let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
    let n2_mpo = mpo.add_tensor("site2".to_string(), t2_mpo).unwrap();
    mpo.connect(n0_mpo, &b_mpo, n2_mpo, &b_mpo).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    input_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        "site2".to_string(),
        IndexMapping {
            true_index: s1.clone(), // Using s1 as true index for site2
            internal_index: s2_in.clone(),
        },
    );
    output_mapping.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        "site2".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s2_out.clone(),
        },
    );

    let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

    // Should error because operator has node "site2" not in state
    let result = apply_linear_operator(&operator, &state, ApplyOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_linear_operator_replaceinds() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let s0_in = make_index(2);
    let s0_out = make_index(2);
    net.add_node(
        "N0".to_string(),
        [s0_in.clone(), s0_out.clone()]
            .into_iter()
            .collect::<HashSet<_>>(),
    )
    .unwrap();

    let link_space = LinkSpace::uniform(2);
    let mut rng = rand::rng();
    let mpo = random_treetn::<f64, _, _>(&mut rng, &net, link_space);

    let true_s0 = make_index(2);
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    input_mapping.insert(
        "N0".to_string(),
        IndexMapping {
            true_index: true_s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    output_mapping.insert(
        "N0".to_string(),
        IndexMapping {
            true_index: true_s0.clone(),
            internal_index: s0_out.clone(),
        },
    );

    let lin_op = LinearOperator::new(mpo, input_mapping, output_mapping);

    // Test replaceinds
    let new_idx1 = make_index(2);
    let new_idx2 = make_index(2);
    let replaced = lin_op
        .replaceinds(
            std::slice::from_ref(&true_s0),
            std::slice::from_ref(&new_idx1),
        )
        .unwrap();
    assert!(replaced.get_input_mapping(&"N0".to_string()).is_some());

    // Test replaceinds with multiple indices
    let new_idx3 = make_index(2);
    let replaced2 = lin_op
        .replaceinds(&[true_s0.clone(), true_s0.clone()], &[new_idx2, new_idx3])
        .unwrap();
    assert!(replaced2.get_input_mapping(&"N0".to_string()).is_some());

    // Test replaceinds error case (length mismatch)
    let result = lin_op.replaceinds(
        std::slice::from_ref(&true_s0),
        &[new_idx1.clone(), new_idx1],
    );
    assert!(result.is_err());
}
